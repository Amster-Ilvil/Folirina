from __future__ import annotations

"""Isolated runtime for optional local vision/OCR processors.

The Qt application may run on Python 3.14 where a partially-compatible PyTorch
wheel can be importable but fail later in torch.distributed/RPC initialisation.
All optional vision models therefore run in the already isolated Python
3.10-3.13 Torch environment used by deep registration.  A persistent JSONL
worker keeps models cached across pages and prevents Torch/Ultralytics from ever
being imported into the GUI process.
"""

from dataclasses import dataclass
import atexit
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
from typing import Any, Callable

import numpy as np

from .deep_registration_runtime import (
    ensure_runtime as ensure_deep_runtime,
    require_runtime_python,
    runtime_root as deep_runtime_root,
    _install_env as deep_install_env,
    _pip_install as deep_pip_install,
    _run as deep_run,
)

ProgressFn = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class VisionRuntimeStatus:
    key: str
    ready: bool
    python: str | None
    detail: str


def worker_script() -> Path:
    return Path(__file__).with_name("vision_model_worker.py")


def _emit(cb: ProgressFn | None, msg: str) -> None:
    if cb is not None:
        cb(str(msg))


def _feature_marker(py: Path, key: str) -> tuple[bool, str]:
    imports = {
        "mangalens": "import torch,torchvision,ultralytics",
        "ysg_obb": "import torch,torchvision,ultralytics",
        "rtdetr_v2": "import torch,torchvision,transformers,safetensors",
        "sam2": "import torch,torchvision,sam2",
        "koharu_layout": "import torch,torchvision,rfdetr,safetensors,PIL",
        "manga_ocr": "import torch,torchvision,transformers,safetensors,PIL",
        "baberu_ocr": "import onnxruntime,numpy,PIL",
        "ocr48px": "import torch,einops,numpy,PIL",
    }
    statement = imports.get(str(key))
    if not statement:
        return False, f"未知视觉运行时：{key}"
    script = statement + "; print('ready')"
    try:
        proc = subprocess.run(
            [str(py), "-c", script], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
            env=deep_install_env(),
        )
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        return False, (proc.stderr or proc.stdout or "import failed").strip()[-4000:]
    return True, "ready"


def runtime_status(key: str) -> VisionRuntimeStatus:
    try:
        py = require_runtime_python()
    except Exception as exc:
        return VisionRuntimeStatus(str(key), False, None, "共享独立 Torch 运行环境未就绪：" + str(exc))
    ok, detail = _feature_marker(py, str(key))
    label = {
        "mangalens":"MangaLens", "ysg_obb":"YSG YOLO OBB", "rtdetr_v2":"RT-DETR-v2", "sam2":"SAM 2.1",
        "koharu_layout":"Koharu Layout RF-DETR Seg 2XL", "manga_ocr":"Manga OCR",
        "baberu_ocr":"Baberu OCR", "ocr48px":"48px AR OCR",
    }.get(str(key), str(key))
    if ok:
        return VisionRuntimeStatus(str(key), True, str(py), f"{label} 独立运行环境已就绪：{py}")
    return VisionRuntimeStatus(str(key), False, str(py), f"{label} 独立运行环境缺依赖：{detail}")


def _install_mangalens(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "在独立 Torch venv 中安装 MangaLens / Ultralytics；不会修改 GUI Python 3.14。")
    deep_pip_install(py, [
        "filelock>=3.16.1", "matplotlib>=3.3", "pyyaml>=5.3.1", "requests>=2.23",
        "psutil>=5.8", "polars>=0.20", "ultralytics-thop>=2.1.6", "pillow>=9.4", "scipy>=1.10",
    ], progress)
    # The shared runtime already owns opencv-python-headless + torch.  Prevent
    # Ultralytics dependency resolution from adding opencv-python/Qt into it.
    deep_pip_install(py, ["ultralytics>=8.3"], progress, extra=["--no-deps"])


def _install_ysg_obb(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "在独立 Torch venv 中安装 YSG YOLO OBB / Ultralytics；不会修改 GUI Python 3.14。")
    _install_mangalens(py, progress)


def _install_rtdetr(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "在独立 Torch venv 中安装 RT-DETR-v2 运行依赖。")
    deep_pip_install(py, ["transformers>=4.48,<6", "safetensors>=0.4", "pillow>=9.4"], progress)


def _install_sam2(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "在独立 Torch venv 中安装 SAM 2.1 运行依赖；macOS 禁用 CUDA 扩展。")
    deep_pip_install(py, ["torch>=2.5.1", "torchvision>=0.20.1", "hydra-core>=1.3.2", "iopath>=0.1.10", "tqdm>=4.66.1", "pillow>=9.4"], progress)
    urls = [
        "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip",
        "https://codeload.github.com/facebookresearch/sam2/zip/refs/heads/main",
    ]
    failures: list[str] = []
    old_flag = os.environ.get("SAM2_BUILD_CUDA")
    os.environ["SAM2_BUILD_CUDA"] = "0"
    try:
        for url in urls:
            code, tail = deep_run(
                [str(py), "-m", "pip", "install", "--disable-pip-version-check", "--upgrade", "--no-deps", url],
                progress, timeout=3600,
            )
            if code == 0:
                return
            failures.append(tail[-2500:])
    finally:
        if old_flag is None:
            os.environ.pop("SAM2_BUILD_CUDA", None)
        else:
            os.environ["SAM2_BUILD_CUDA"] = old_flag
    raise RuntimeError("SAM 2.1 官方源码包安装失败。\n" + "\n".join(failures[-2:]))


def _install_koharu_layout(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "安装 Koharu Layout RF-DETR Seg 2XL 运行依赖（固定兼容 RF-DETR 1.7.x）。")
    deep_pip_install(py, ["rfdetr>=1.7,<1.8", "safetensors>=0.4", "pillow>=9.4"], progress)


def _install_manga_ocr(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "安装 Manga OCR 独立 Transformers 运行依赖。")
    deep_pip_install(py, ["transformers>=4.48,<5", "safetensors>=0.4", "pillow>=9.4", "fugashi>=1.3", "unidic-lite>=1.0"], progress)


def _install_baberu_ocr(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "安装 Baberu OCR ONNX 运行依赖；不安装 PyTorch/Transformers 额外模型代码。")
    deep_pip_install(py, ["onnxruntime>=1.18", "pillow>=9.4"], progress)


def _install_ocr48px(py: Path, progress: ProgressFn | None) -> None:
    _emit(progress, "安装 48px AR OCR 独立运行依赖；复用共享 Torch/MPS 环境。")
    deep_pip_install(py, ["einops>=0.8,<1", "pillow>=9.4"], progress)


def ensure_runtime(key: str, progress: ProgressFn | None = None) -> VisionRuntimeStatus:
    key = str(key)
    if key not in {"mangalens", "ysg_obb", "rtdetr_v2", "sam2", "koharu_layout", "manga_ocr", "baberu_ocr", "ocr48px"}:
        raise ValueError(key)
    # Reuse the isolated Torch venv.  This means one Torch/MPS stack is shared
    # by LightGlue, LoFTR and optional vision models instead of duplicating GBs.
    ensure_deep_runtime(progress)
    py = require_runtime_python()
    status = runtime_status(key)
    if status.ready:
        _emit(progress, status.detail)
        return status
    if key == "mangalens":
        _install_mangalens(py, progress)
    elif key == "ysg_obb":
        _install_ysg_obb(py, progress)
    elif key == "rtdetr_v2":
        _install_rtdetr(py, progress)
    elif key == "sam2":
        _install_sam2(py, progress)
    elif key == "koharu_layout":
        _install_koharu_layout(py, progress)
    elif key == "manga_ocr":
        _install_manga_ocr(py, progress)
    elif key == "baberu_ocr":
        _install_baberu_ocr(py, progress)
    else:
        _install_ocr48px(py, progress)
    ok, detail = _feature_marker(py, key)
    if not ok:
        raise RuntimeError(f"{key} 独立运行环境安装后仍无法导入。\n{detail}")
    status = runtime_status(key)
    _emit(progress, status.detail)
    return status


class _PersistentWorker:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._proc: subprocess.Popen[str] | None = None
        self._responses: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stderr_tail: list[str] = []
        self._counter = 0

    def _start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        py = require_runtime_python()
        self._responses = queue.Queue(); self._stderr_tail = []
        self._proc = subprocess.Popen(
            [str(py), str(worker_script())],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            env=deep_install_env(),
        )
        assert self._proc.stdout is not None and self._proc.stderr is not None
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._read_stderr, args=(self._proc,), daemon=True).start()

    def _read_stdout(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                self._responses.put(row)

    def _read_stderr(self, proc: subprocess.Popen[str]) -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            txt = line.rstrip()
            if txt:
                self._stderr_tail.append(txt)
                if len(self._stderr_tail) > 100:
                    del self._stderr_tail[:50]

    def request(self, payload: dict[str, Any], timeout: float = 600.0) -> dict[str, Any]:
        with self._lock:
            self._start()
            assert self._proc is not None and self._proc.stdin is not None
            self._counter += 1; rid = self._counter
            row = dict(payload); row["request_id"] = rid
            try:
                self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n"); self._proc.stdin.flush()
            except Exception:
                self.stop(); self._start()
                assert self._proc is not None and self._proc.stdin is not None
                self._proc.stdin.write(json.dumps(row, ensure_ascii=False) + "\n"); self._proc.stdin.flush()
            while True:
                try:
                    result = self._responses.get(timeout=timeout)
                except queue.Empty as exc:
                    detail = "\n".join(self._stderr_tail[-30:])
                    self.stop()
                    raise RuntimeError("视觉模型 Worker 超时。\n" + detail[-5000:]) from exc
                if int(result.get("request_id", -1)) != rid:
                    continue
                if not result.get("ok"):
                    raise RuntimeError(str(result.get("error") or "视觉模型 Worker 执行失败"))
                return result

    def stop(self) -> None:
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is None:
                return
            try:
                if proc.stdin:
                    proc.stdin.write(json.dumps({"op":"shutdown","request_id":-1}) + "\n"); proc.stdin.flush()
            except Exception:
                pass
            try:
                proc.terminate(); proc.wait(timeout=3)
            except Exception:
                try: proc.kill()
                except Exception: pass


_WORKER = _PersistentWorker()
atexit.register(_WORKER.stop)


def _run_image_request(op: str, image: np.ndarray, payload: dict[str, Any]) -> dict[str, Any]:
    root = deep_runtime_root(); root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="mhd-vision-", dir=str(root)) as td:
        p = Path(td) / "image.npy"
        np.save(p, np.asarray(image), allow_pickle=False)
        return _WORKER.request({"op": op, "image": str(p), **payload})


def run_mangalens(image: np.ndarray, *, model_path: str, confidence: float, imgsz: int, device: str) -> dict[str, Any]:
    status = runtime_status("mangalens")
    if not status.ready:
        raise RuntimeError("MangaLens 独立运行环境尚未就绪。请在模型中心点击 MangaLens → 安装依赖。\n" + status.detail)
    return _run_image_request("mangalens", image, {
        "model_path": str(model_path), "confidence": float(confidence), "imgsz": int(imgsz), "device": str(device),
    })


def run_rtdetr(image: np.ndarray, *, model_ref: str, local_only: bool, confidence: float, imgsz: int, device: str) -> dict[str, Any]:
    status = runtime_status("rtdetr_v2")
    if not status.ready:
        raise RuntimeError("RT-DETR-v2 独立运行环境尚未就绪。请在模型中心点击 RT-DETR-v2 → 安装依赖。\n" + status.detail)
    return _run_image_request("rtdetr_v2", image, {
        "model_ref": str(model_ref), "local_only": bool(local_only), "confidence": float(confidence), "imgsz": int(imgsz), "device": str(device),
    })


def run_sam2(image: np.ndarray, *, checkpoint: str | None, model_id: str, config_file: str, allow_download: bool, prompts: list[dict[str, Any]], device: str) -> dict[str, Any]:
    status = runtime_status("sam2")
    if not status.ready:
        raise RuntimeError("SAM 2.1 独立运行环境尚未就绪。请在模型中心点击 SAM 2.1 → 安装依赖。\n" + status.detail)
    return _run_image_request("sam2", image, {
        "checkpoint": checkpoint or "", "model_id": str(model_id), "config_file": str(config_file),
        "allow_download": bool(allow_download), "prompts": prompts, "device": str(device),
    })


def run_ysg_obb(
    image: np.ndarray, *, model_path: str | Path, confidence: float = .25,
    iou: float = .50, imgsz: int = 1600, device: str = "auto",
) -> dict[str, Any]:
    return _run_image_request("ysg_obb", image, {
        "model_path": str(model_path), "confidence": float(confidence),
        "iou": float(iou), "imgsz": int(imgsz), "device": str(device),
    })


def run_koharu_layout(
    image: np.ndarray, *, model_dir: str, device: str,
    text_threshold: float = .25, sfx_threshold: float = .20,
    bubble_threshold: float = .50, panel_threshold: float = .50,
    shape: int = 1152, postprocess_max_side: int = 1152,
    postprocess_downscale_trigger_side: int = 2048,
) -> dict[str, Any]:
    status = runtime_status("koharu_layout")
    if not status.ready:
        raise RuntimeError("Koharu Layout 独立运行环境尚未就绪。请在模型中心安装依赖。\n" + status.detail)
    return _run_image_request("koharu_layout", image, {
        "model_dir": str(model_dir), "device": str(device), "shape": int(shape),
        "postprocess_max_side": int(postprocess_max_side),
        "postprocess_downscale_trigger_side": int(postprocess_downscale_trigger_side),
        "text_threshold": float(text_threshold), "sfx_threshold": float(sfx_threshold),
        "bubble_threshold": float(bubble_threshold), "panel_threshold": float(panel_threshold),
    })


def run_manga_ocr(image: np.ndarray, *, model_dir: str, device: str, max_new_tokens: int = 128) -> dict[str, Any]:
    status = runtime_status("manga_ocr")
    if not status.ready:
        raise RuntimeError("Manga OCR 独立运行环境尚未就绪。请在模型中心安装依赖。\n" + status.detail)
    return _run_image_request("manga_ocr", image, {
        "model_dir": str(model_dir), "device": str(device), "max_new_tokens": int(max_new_tokens),
    })


def run_baberu_ocr(image: np.ndarray, *, model_dir: str, max_new_tokens: int = 128) -> dict[str, Any]:
    status = runtime_status("baberu_ocr")
    if not status.ready:
        raise RuntimeError("Baberu OCR ONNX 运行环境尚未就绪。请在模型中心安装依赖。\n" + status.detail)
    return _run_image_request("baberu_ocr", image, {
        "model_dir": str(model_dir), "max_new_tokens": int(max_new_tokens),
    })


def run_ocr48px(
    image: np.ndarray, *, model_dir: str, device: str = "auto",
    beams_k: int = 5, max_seq_length: int = 255,
) -> dict[str, Any]:
    status = runtime_status("ocr48px")
    if not status.ready:
        raise RuntimeError("48px AR OCR 独立运行环境尚未就绪。请在模型中心安装依赖。\n" + status.detail)
    return _run_image_request("ocr48px", image, {
        "model_dir": str(model_dir), "device": str(device),
        "beams_k": int(beams_k), "max_seq_length": int(max_seq_length),
    })
