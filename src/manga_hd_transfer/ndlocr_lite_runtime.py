from __future__ import annotations

"""Isolated NDLOCR-Lite runtime adapter.

NDLOCR-Lite is an ONNXRuntime-based full-page OCR engine.  Folirina keeps it in
its own Python 3.10-3.13 environment so the GUI and the Torch/MPS runtime never
inherit its dependency graph.  The optional CoreML execution path is provider
level only and always falls back to the official CPU path when an operator is
not supported.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import xml.etree.ElementTree as ET

from .storage_paths import model_home
from .ocr_base import OCRBackend
from .tls_support import apply_runtime_tls_environment

ProgressFn = Callable[[str], None]
NDL_VERSION = "1.2.3"
NDL_ARCHIVE_URL = f"https://github.com/ndl-lab/ndlocr-lite/archive/refs/tags/v{NDL_VERSION}.zip"


def _find_compatible_python() -> Path:
    """Find a native 64-bit Python 3.10-3.13 without importing deep runtime."""
    candidates: list[str] = []
    for name in ("MHD_NDL_PYTHON", "MHD_OCR_PYTHON", "MHD_TORCH_PYTHON"):
        value = str(os.environ.get(name, "") or "").strip()
        if value:
            candidates.append(value)
    candidates.append(sys.executable)
    for name in ("python3.13", "python3.12", "python3.11", "python3.10", "python3"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    seen: set[str] = set()
    for raw in candidates:
        if raw in seen:
            continue
        seen.add(raw)
        candidate = Path(raw).expanduser()
        if not candidate.is_file():
            continue
        try:
            proc = subprocess.run([str(candidate), "-c", "import platform,sys;print(sys.version_info.minor);print(platform.architecture()[0]);print(platform.machine().lower())"], capture_output=True, text=True, timeout=15)
            lines = proc.stdout.strip().splitlines()
            if proc.returncode != 0 or len(lines) < 3:
                continue
            minor = int(lines[0])
            if not (10 <= minor <= 13) or lines[1] != "64bit":
                continue
            if platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} and lines[2] not in {"arm64", "aarch64"}:
                continue
            return candidate.resolve()
        except Exception:
            continue
    raise RuntimeError("NDLOCR-Lite 需要可用的 64 位 Python 3.10～3.13；可设置 MHD_NDL_PYTHON 指向对应解释器。")


@dataclass(frozen=True, slots=True)
class NDLOCRLiteRuntimeStatus:
    ready: bool
    python: str | None
    source_path: str | None
    detail: str
    version: str = NDL_VERSION
    coreml_available: bool = False


def runtime_root() -> Path:
    return model_home().parent / "runtime" / "ndlocr-lite"


def venv_dir() -> Path:
    return runtime_root() / ".venv"


def venv_python() -> Path:
    return venv_dir() / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def source_probe_script() -> str:
    return "import pathlib,ocr; print(pathlib.Path(ocr.__file__).resolve().parent)"


def _emit(cb: ProgressFn | None, msg: str) -> None:
    if cb is not None:
        cb(str(msg))


def _env() -> dict[str, str]:
    env = os.environ.copy()
    proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
    if proxy:
        env.update({"HTTP_PROXY": proxy, "HTTPS_PROXY": proxy, "http_proxy": proxy, "https_proxy": proxy})
    return apply_runtime_tls_environment(env, runtime_root())


def _run(cmd: list[str], *, progress: ProgressFn | None = None, timeout: float = 3600.0, env: dict[str, str] | None = None) -> tuple[int, str]:
    _emit(progress, "执行：" + " ".join(str(x) for x in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace", env=env or _env())
    tail: list[str] = []
    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if line:
            tail.append(line)
            if len(tail) > 120:
                tail = tail[-120:]
            low = line.lower()
            if any(token in low for token in ("error", "failed", "warning", "installing", "successfully", "already satisfied")):
                _emit(progress, line)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
        raise TimeoutError(f"NDLOCR-Lite 命令超时：{cmd[0]}")
    return int(code), "\n".join(tail[-80:])


def _source_path(py: Path) -> Path | None:
    try:
        proc = subprocess.run([str(py), "-c", source_probe_script()], capture_output=True, text=True, encoding="utf-8", errors="replace", env=_env(), timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    value = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    p = Path(value).expanduser() if value else None
    return p.resolve() if p and p.is_dir() else None


def runtime_status() -> NDLOCRLiteRuntimeStatus:
    py = venv_python()
    if not py.is_file():
        return NDLOCRLiteRuntimeStatus(False, None, None, "NDLOCR-Lite 独立 venv 未安装。")
    source = _source_path(py)
    if source is None:
        return NDLOCRLiteRuntimeStatus(False, str(py), None, "NDLOCR-Lite 已有 venv，但无法导入 ocr 模块。")
    coreml = False
    try:
        probe = subprocess.run([str(py), "-c", "import onnxruntime as o; print('CoreMLExecutionProvider' in o.get_available_providers())"], capture_output=True, text=True, env=_env(), timeout=30)
        coreml = probe.returncode == 0 and probe.stdout.strip().lower() == "true"
    except Exception:
        pass
    return NDLOCRLiteRuntimeStatus(True, str(py), str(source), f"NDLOCR-Lite v{NDL_VERSION} 已就绪：{py} · source={source}", coreml_available=coreml)


def ensure_runtime(progress: ProgressFn | None = None) -> NDLOCRLiteRuntimeStatus:
    status = runtime_status()
    if status.ready:
        _emit(progress, status.detail)
        return status
    runtime_root().mkdir(parents=True, exist_ok=True)
    py_base = _find_compatible_python()
    if not venv_python().is_file():
        _emit(progress, f"创建 NDLOCR-Lite 独立 Python venv：{venv_dir()}")
        code, tail = _run([str(py_base), "-m", "venv", str(venv_dir())], progress=progress, timeout=300)
        if code != 0:
            raise RuntimeError("NDLOCR-Lite venv 创建失败：\n" + tail[-5000:])
    py = venv_python()
    code, tail = _run([str(py), "-m", "pip", "install", "--upgrade", "pip"], progress=progress, timeout=900)
    if code != 0:
        raise RuntimeError("NDLOCR-Lite pip 自升级失败：\n" + tail[-5000:])
    _emit(progress, f"安装 NDLOCR-Lite v{NDL_VERSION}（隔离 CPU/ONNXRuntime，后续可尝试 CoreML EP）。")
    code, tail = _run([str(py), "-m", "pip", "install", "--upgrade", NDL_ARCHIVE_URL], progress=progress, timeout=1800)
    if code != 0:
        raise RuntimeError("NDLOCR-Lite 官方 venv 安装失败：\n" + tail[-6000:])
    status = runtime_status()
    if not status.ready:
        raise RuntimeError("NDLOCR-Lite 安装后仍无法导入：\n" + status.detail)
    _emit(progress, status.detail)
    return status


def _write_ndl_wrapper(path: Path) -> None:
    path.write_text(
        '''import os, sys, runpy\n\nMODULE_ROOT = os.environ.get("MHD_NDL_SOURCE", "")\nif MODULE_ROOT:\n    sys.path.insert(0, MODULE_ROOT)\n\nif os.environ.get("MHD_NDL_COREML", "0") == "1":\n    try:\n        import onnxruntime as ort\n        _orig = ort.InferenceSession\n        _wrapped = getattr(ort, "_folirina_coreml_wrapped", False)\n        if not _wrapped:\n            def _session(*args, providers=None, **kwargs):\n                available = set(ort.get_available_providers())\n                if "CoreMLExecutionProvider" in available:\n                    try:\n                        return _orig(*args, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"], **kwargs)\n                    except Exception:\n                        pass\n                return _orig(*args, providers=providers, **kwargs)\n            ort.InferenceSession = _session\n            ort._folirina_coreml_wrapped = True\n    except Exception:\n        pass\n\nrunpy.run_module("ocr", run_name="__main__")\n''', encoding="utf-8")


def _cli_args(py: Path) -> str:
    try:
        proc = subprocess.run([str(py), "-c", "import ocr,inspect; import pathlib; print(pathlib.Path(ocr.__file__).resolve())"], capture_output=True, text=True, env=_env(), timeout=30)
        if proc.returncode != 0:
            return ""
        return proc.stdout.strip()
    except Exception:
        return ""


def _build_cli(py: Path, source: Path, input_path: Path, output_dir: Path, lang: str) -> list[str]:
    # NDLOCR-Lite's documented GUI/runtime entrypoint accepts --sourceimg and
    # --output.  We inspect --help once per subprocess to keep minor CLI drift
    # across patch releases from breaking the adapter.
    try:
        help_proc = subprocess.run([str(py), "-m", "ocr", "--help"], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(source), env=_env(), timeout=30)
        help_text = (help_proc.stdout or "") + "\n" + (help_proc.stderr or "")
    except Exception:
        help_text = ""
    cmd = [str(py), "-m", "ocr"]
    if "--sourceimg" in help_text:
        cmd += ["--sourceimg", str(input_path)]
    elif "--source" in help_text:
        cmd += ["--source", str(input_path)]
    else:
        cmd += ["--sourceimg", str(input_path)]
    cmd += ["--output", str(output_dir)]
    if "--langcode" in help_text:
        cmd += ["--langcode", str(lang or "ja")]
    if "--json" in help_text:
        cmd += ["--json"]
    if "--xml" in help_text:
        cmd += ["--xml"]
    return cmd


def _point_list(value: Any) -> list[list[float]] | None:
    if isinstance(value, dict):
        for key in ("points", "polygon", "poly", "vertices"):
            if key in value:
                return _point_list(value[key])
    if isinstance(value, (list, tuple)) and len(value) >= 3 and all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value):
        return [[float(p[0]), float(p[1])] for p in value]
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value):
        x0,y0,x1,y1 = map(float, value); return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    return None


def _bbox_from_any(row: dict[str, Any]) -> list[list[float]] | None:
    for key in ("bbox", "box", "rect"):
        poly = _point_list(row.get(key))
        if poly:
            return poly
    for keyset in (("x","y","width","height"),("x0","y0","x1","y1")):
        if all(k in row for k in keyset):
            vals = [float(row[k]) for k in keyset]
            if keyset == ("x","y","width","height"):
                x,y,w,h=vals; return [[x,y],[x+w,y],[x+w,y+h],[x,y+h]]
            x0,y0,x1,y1=vals; return [[x0,y0],[x1,y0],[x1,y1],[x0,y1]]
    return None


def _json_rows(obj: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, list):
        for v in obj: rows.extend(_json_rows(v))
    elif isinstance(obj, dict):
        text = obj.get("text") or obj.get("string") or obj.get("transcription")
        poly = _point_list(obj.get("polygon") or obj.get("poly") or obj.get("points")) or _bbox_from_any(obj)
        if text is not None and poly:
            rows.append({"text": str(text), "polygon": poly, "confidence": obj.get("confidence", obj.get("score", 1.0)), "order": obj.get("reading_order", obj.get("order", len(rows))), "raw": obj})
        for key, value in obj.items():
            if key not in {"text","string","transcription","polygon","poly","points","bbox","box","rect"}:
                rows.extend(_json_rows(value))
    return rows


def _parse_xml(path: Path) -> list[dict[str, Any]]:
    tree = ET.parse(path)
    rows: list[dict[str, Any]] = []
    for node in tree.iter():
        text = None
        for child in list(node):
            tag = child.tag.split("}")[-1].upper()
            if tag in {"STRING", "TEXT", "TRANSCRIPTION"}:
                text = (child.text or "").strip()
                break
        if text is None:
            text = str(node.attrib.get("text") or node.attrib.get("STRING") or node.attrib.get("TRANSCRIPTION") or "").strip()
        if not text:
            continue
        polygon: list[list[float]] | None = None
        for child in node.iter():
            tag = child.tag.split("}")[-1].upper()
            if tag in {"POLYGON", "POINTS", "SHAPE"}:
                raw = child.attrib.get("POINTS") or (child.text or "")
                nums = [float(v) for v in re.findall(r"[-+]?\d+(?:\.\d+)?", raw)]
                if len(nums) >= 6 and len(nums) % 2 == 0:
                    polygon = [[nums[i], nums[i+1]] for i in range(0, len(nums), 2)]
                    break
        if polygon is None:
            box = _bbox_from_any(node.attrib)
            polygon = box
        if polygon:
            rows.append({"text": text, "polygon": polygon, "confidence": node.attrib.get("confidence", node.attrib.get("score", 1.0)), "order": node.attrib.get("order", len(rows)), "raw": node.attrib})
    return rows


def _collect_output_files(out: Path) -> list[Path]:
    return sorted([p for p in out.rglob("*") if p.is_file() and p.suffix.lower() in {".json", ".xml", ".tei"}]) if out.exists() else []


def _parse_outputs(out: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in _collect_output_files(out):
        try:
            if path.suffix.lower() == ".json":
                rows.extend(_json_rows(json.loads(path.read_text(encoding="utf-8"))))
            else:
                rows.extend(_parse_xml(path))
        except Exception:
            continue
    # Some NDL exports repeat the same line in JSON + XML. Keep the first exact
    # match and make order deterministic.
    dedup: dict[tuple[str, tuple[tuple[float,float],...]], dict[str,Any]] = {}
    for row in rows:
        key = (str(row.get("text", "")).strip(), tuple((round(float(x),2),round(float(y),2)) for x,y in row.get("polygon") or []))
        dedup.setdefault(key, row)
    rows = list(dedup.values())
    rows.sort(key=lambda r: (float(r.get("order", 0) or 0), (r.get("polygon") or [[0,0]])[0][1], (r.get("polygon") or [[0,0]])[0][0]))
    return rows


def recognize_file(image_path: str | Path, *, lang: str = "ja", runtime_path: str | Path | None = None, coreml: bool = False, coreml_compute_units: str = "ane", timeout: int = 300, assumed_confidence: float = 0.72) -> dict[str, Any]:
    py = Path(runtime_path).expanduser() / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python") if runtime_path else venv_python()
    source = _source_path(py)
    if not py.is_file() or source is None:
        raise RuntimeError("NDLOCR-Lite 独立运行时尚未就绪。请在模型中心安装 NDLOCR-Lite 依赖。")
    src = Path(image_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)
    with tempfile.TemporaryDirectory(prefix="folirina-ndl-") as td:
        root = Path(td); out = root / "output"; out.mkdir()
        wrapper = root / "entry.py"; _write_ndl_wrapper(wrapper)
        env = _env(); env["MHD_NDL_SOURCE"] = str(source); env["MHD_NDL_COREML"] = "1" if coreml else "0"; env["MHD_NDL_COREML_UNITS"] = str(coreml_compute_units or "ane")
        cmd = _build_cli(py, source, src, out, lang)
        # Run through the wrapper so provider patching happens before NDL imports\n        # DEIM/PARSeq modules and constructs ONNXRuntime sessions.\n        cmd = [str(py), str(wrapper), *cmd[3:]]
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(source), env=env, timeout=max(30,int(timeout)))
        rows = _parse_outputs(out) if proc.returncode == 0 else []
        if proc.returncode != 0:
            raise RuntimeError("NDLOCR-Lite 识别失败：\n" + ((proc.stdout or "") + "\n" + (proc.stderr or ""))[-6000:])
        return {"items": [{"text": r["text"], "polygon": r["polygon"], "confidence": float(r.get("confidence", assumed_confidence) or assumed_confidence), "reading_order": int(float(r.get("order",i) or i))} for i,r in enumerate(rows)], "device": "coreml-experimental" if coreml else "cpu", "confidence_source": "engine_score_or_assumed", "isolated_runtime": True}


class NDLOCRLiteBackend(OCRBackend):
    supports_crop_recognition = False
    supports_rectified_input = True
    retry_crops = False

    def __init__(self, lang: str, config: Any | None = None) -> None:
        self._config = config
        self.lang = str(lang or "ja")
        status = runtime_status()
        if not status.ready:
            raise RuntimeError(status.detail)
        self.runtime_python = status.python
        self._status = status

    def recognize(self, image, *, image_path=None):
        if image_path is None:
            with tempfile.TemporaryDirectory(prefix="folirina-ndl-image-") as td:
                p = Path(td) / "page.png"
                import cv2
                if not cv2.imwrite(str(p), image):
                    raise RuntimeError("无法为 NDLOCR-Lite 创建临时输入图像")
                return self._recognize_path(p, image.shape[:2])
        return self._recognize_path(Path(image_path), image.shape[:2])

    def _recognize_path(self, path: Path, shape: tuple[int,int]):
        from .models import TextBlock
        payload = recognize_file(
            path, lang=self.lang,
            runtime_path=str(Path(self.runtime_python).resolve().parent.parent.parent),
            coreml=bool(getattr(self._config, "ndlocr_lite_coreml_auto", True) and self._status.coreml_available and platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"}),
            coreml_compute_units=str(getattr(self._config, "ndlocr_lite_coreml_compute_units", "ane") or "ane"),
            timeout=int(getattr(self._config, "ndlocr_lite_timeout_seconds", 300)),
            assumed_confidence=float(getattr(self._config, "ndlocr_lite_assumed_confidence", 0.72)),
        )
        blocks=[]
        for i,row in enumerate(payload.get("items") or []):
            poly=[(float(x),float(y)) for x,y in row.get("polygon") or []]
            if len(poly)<3 or not str(row.get("text") or "").strip():
                continue
            blocks.append(TextBlock(id=f"ndl-{i:04d}",polygon=poly,text=str(row["text"]).strip(),confidence=float(row.get("confidence",getattr(self._config,"ndlocr_lite_assumed_confidence",0.72))),kind="unknown",reading_order=int(row.get("reading_order",i)),meta={"backend":"ndlocr_lite","device":payload.get("device"),"confidence_source":payload.get("confidence_source"),"full_page":True,"coreml_experimental":payload.get("device")=="coreml-experimental"}))
        return blocks

    def recognize_region(self, *args, **kwargs):
        raise RuntimeError("NDLOCR-Lite 是整页 OCR，不支持局部 crop 识别；请使用 Hayai OCR / 48px AR OCR 进行分区识别。")


__all__=["NDLOCRLiteBackend","NDLOCRLiteRuntimeStatus","runtime_status","ensure_runtime","recognize_file","NDL_VERSION"]
