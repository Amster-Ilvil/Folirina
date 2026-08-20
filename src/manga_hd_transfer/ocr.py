from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import OCRConfig
from .model_downloads import discovered_paddle_model_dirs
from .geometry import bbox_polygon, polygon_centroid
from .models import TextBlock
from .ocr_base import OCRBackend

logger = logging.getLogger(__name__)


class NullOCRBackend(OCRBackend):
    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        return []


class SidecarOCRBackend(OCRBackend):
    supports_crop_recognition = False
    supports_rectified_input = False

    """Reads deterministic OCR data from `<image>.ocr.json` or a configured sidecar.

    Accepted payloads:
      {"blocks": [{"polygon": [[x,y],...], "text": "...", "confidence": .99}]}
      [{...}, {...}]
    """

    def __init__(self, suffix: str = ".ocr.json") -> None:
        self.suffix = suffix

    def sidecar_path(self, image_path: str | Path) -> Path:
        p = Path(image_path)
        # page.png -> page.ocr.json
        if self.suffix.startswith("."):
            return p.with_suffix(self.suffix)
        return p.parent / f"{p.stem}{self.suffix}"

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if image_path is None:
            raise ValueError("SidecarOCRBackend requires image_path")
        sidecar = self.sidecar_path(image_path)
        if not sidecar.exists():
            raise FileNotFoundError(f"OCR sidecar not found: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        rows = payload.get("blocks", []) if isinstance(payload, dict) else payload
        blocks: list[TextBlock] = []
        for i, row in enumerate(rows):
            poly = row.get("polygon")
            if not poly and row.get("bbox"):
                poly = bbox_polygon(row["bbox"])
            if not poly:
                continue
            blocks.append(
                TextBlock(
                    id=str(row.get("id") or f"ocr-{i:04d}"),
                    polygon=[(float(x), float(y)) for x, y in poly],
                    text=str(row.get("text", "")),
                    confidence=float(row.get("confidence", 1.0)),
                    kind=str(row.get("kind", "unknown")),
                    reading_order=int(row.get("reading_order", i)),
                    bubble_id=row.get("bubble_id"),
                    meta=dict(row.get("meta", {})),
                )
            )
            mask_path = row.get("mask_path") or blocks[-1].meta.get("mask_path")
            if mask_path:
                mp = Path(mask_path)
                if not mp.is_absolute():
                    mp = sidecar.parent / mp
                blocks[-1].meta["mask_path"] = str(mp)
        return blocks


def _find_ocr_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if "rec_texts" in obj and ("rec_polys" in obj or "dt_polys" in obj or "rec_boxes" in obj):
            return obj
        for v in obj.values():
            found = _find_ocr_dict(v)
            if found is not None:
                return found
    return None


def _result_to_dict(res: Any) -> dict[str, Any]:
    candidates = []
    for attr in ("json", "res", "result"):
        if hasattr(res, attr):
            value = getattr(res, attr)
            try:
                value = value() if callable(value) else value
            except Exception:
                pass
            candidates.append(value)
    if hasattr(res, "to_dict"):
        try:
            candidates.append(res.to_dict())
        except Exception:
            pass
    candidates.append(res)
    for candidate in candidates:
        found = _find_ocr_dict(candidate)
        if found is not None:
            return found
    raise ValueError(f"Unsupported PaddleOCR result structure: {type(res)!r}")


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR backend hosted in an isolated compatible Python venv.

    Current macOS Paddle wheels are arm64 CPython 3.9-3.13 only.  The desktop
    GUI may run on a newer Python (or under Rosetta), so importing PaddleOCR in
    process is deliberately avoided.  A persistent JSONL worker keeps model
    startup cost out of per-page recognition.
    """

    def __init__(self, lang: str, config: OCRConfig | None = None, device: str | None = None, profile_override: str | None = None) -> None:
        self.config = config or OCRConfig()
        self.lang = str(lang)
        self.profile_override = profile_override
        self.selection: dict[str, Any] = {}
        self.retry_crops = True
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._start_worker(device=device)

    def _start_worker(self, device: str | None = None) -> None:
        from .paddle_runtime import require_runtime_python as require_ocr_python, worker_script as ocr_worker_script, runtime_root as ocr_runtime_root
        from .paddle_doc_runtime import ensure_runtime as ensure_doc_runtime, require_runtime_python as require_doc_python, worker_script as doc_worker_script, runtime_root as doc_runtime_root
        from .paddle_sources import paddle_model_source_attempts, paddle_source_environment, PADDLE_MODEL_SOURCE_LABELS
        from .paddle_profiles import resolve_paddle_model_selection
        from .tls_support import apply_runtime_tls_environment

        selection = resolve_paddle_model_selection(self.config, self.profile_override)
        self.selection = dict(selection)
        pipeline_name = str(selection.get("pipeline") or "ocr")
        local_det, local_rec = discovered_paddle_model_dirs()
        det_dir = (self.config.paddle_text_detection_model_dir or (str(local_det) if local_det else None)) if pipeline_name == "ocr" else None
        rec_dir = (self.config.paddle_text_recognition_model_dir or (str(local_rec) if local_rec else None)) if pipeline_name == "ocr" else None
        if pipeline_name in {"vl", "structure"}:
            # VL / Structure have their own dependency surface and venv.  A repair
            # must never mutate the classic PP-OCRv6 runtime.
            ensure_doc_runtime()
            py = require_doc_python()
            worker = doc_worker_script()
            tls_root = doc_runtime_root()
        else:
            py = require_ocr_python()
            worker = ocr_worker_script()
            tls_root = ocr_runtime_root()
        self.retry_crops = pipeline_name == "ocr"
        cmd = [
            str(py), "-u", str(worker), "--lang", self.lang,
            "--pipeline", pipeline_name,
            "--ocr-version", str(selection["ocr_version"] or self.config.ocr_version),
            "--model-profile", str(selection["profile"]),
        ]
        if selection.get("pipeline_version"):
            cmd += ["--pipeline-version", str(selection["pipeline_version"])]
        if selection.get("det_name"):
            cmd += ["--det-name", str(selection["det_name"])]
        if selection.get("rec_name"):
            cmd += ["--rec-name", str(selection["rec_name"])]
        if det_dir:
            cmd += ["--det-dir", str(det_dir)]
        if rec_dir:
            cmd += ["--rec-dir", str(rec_dir)]
        if device and str(device).lower() in {"cpu", "gpu"}:
            cmd += ["--device", str(device).lower()]

        # With complete offline model dirs, never touch a model hub. Otherwise
        # retry sources only during worker initialization, before any page is
        # emitted; this avoids duplicate OCR output on fallback.
        sources = ("local",) if det_dir and rec_dir else paddle_model_source_attempts(getattr(self.config, "paddle_model_source", "auto"))
        failures: list[str] = []
        self.model_source = ""
        for source in sources:
            base = os.environ.copy()
            env = base if source == "local" else paddle_source_environment(source, base)
            proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
            if proxy:
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    env[key] = proxy
            env = apply_runtime_tls_environment(env, tls_root)
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                )
                self._start_stdout_pump()
                ready = self._read_json(expected={"ready", "probe", "init"})
                if ready.get("ok"):
                    self.model_source = source
                    return
                detail = str(ready.get("error", "未知错误"))
            except Exception as exc:
                detail = str(exc)
            label = "本地离线模型" if source == "local" else PADDLE_MODEL_SOURCE_LABELS.get(source, source)
            failures.append(f"{label}: {detail[-3500:]}")
            self.close()

        raise RuntimeError(
            "PaddleOCR 独立运行环境初始化失败；已在 OCR 输出前尝试全部允许的模型源。\n"
            + "\n\n".join(failures[-4:])[-10000:]
        )

    def _start_stdout_pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("PaddleOCR worker stdout 不可用")
        q: queue.Queue[str | None] = queue.Queue()
        self._stdout_queue = q

        def pump() -> None:
            try:
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)

        thread = threading.Thread(target=pump, name="mhd-paddle-worker-output", daemon=True)
        self._stdout_thread = thread
        thread.start()

    def _read_json(self, *, expected: set[str], max_lines: int = 400) -> dict[str, Any]:
        proc = self._proc
        q = self._stdout_queue
        if proc is None or q is None:
            raise RuntimeError("PaddleOCR worker 未启动")
        diagnostics: list[str] = []
        startup = bool(expected.intersection({"ready", "probe", "init"}))
        env_name = "MHD_PADDLE_STARTUP_TIMEOUT" if startup else "MHD_PADDLE_REQUEST_TIMEOUT"
        default_timeout = 900.0 if startup else 300.0
        try:
            timeout = max(30.0, float(os.environ.get(env_name, default_timeout)))
        except Exception:
            timeout = default_timeout
        deadline = time.monotonic() + timeout
        lines_seen = 0
        while lines_seen < max_lines:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"PaddleOCR worker {'初始化/模型下载' if startup else '识别'}等待超过 {int(timeout)} 秒。"
                    + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                )
            try:
                line = q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"PaddleOCR worker 提前退出 (code={code})"
                        + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                    )
                continue
            if line is None:
                code = proc.poll()
                raise RuntimeError(
                    "PaddleOCR worker 输出结束" + (f" (code={code})" if code is not None else "")
                    + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                )
            lines_seen += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                diagnostics.append(raw)
                continue
            if isinstance(payload, dict) and str(payload.get("type", "")) in expected:
                return payload
            diagnostics.append(raw)
        raise RuntimeError("PaddleOCR worker 输出过多非协议日志，无法取得结果。\n" + "\n".join(diagnostics[-30:]))

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            self.close()
            self._start_worker()
            proc = self._proc
        assert proc is not None and proc.stdin is not None
        fd, tmp_name = tempfile.mkstemp(prefix="mhd-paddle-", suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(tmp_name, image):
                raise RuntimeError("无法写入 PaddleOCR 临时图像")
            proc.stdin.write(json.dumps({"cmd": "predict", "path": tmp_name}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            payload = self._read_json(expected={"result"})
            if not payload.get("ok"):
                pipeline_name = str(self.selection.get("pipeline") or "ocr")
                profile_name = str(self.selection.get("profile") or self.config.paddle_model_profile or "")
                stage = str(payload.get("stage") or "predict")
                raise RuntimeError(
                    f"PaddleOCR 识别失败 [{pipeline_name}/{profile_name}/{stage}]："
                    + str(payload.get("error", "未知错误"))
                )
            blocks: list[TextBlock] = []
            for i, row in enumerate(payload.get("blocks") or []):
                poly = row.get("polygon") or []
                if len(poly) < 3:
                    continue
                blocks.append(
                    TextBlock(
                        id=str(row.get("id") or f"ocr-{i:04d}"),
                        polygon=[(float(x), float(y)) for x, y in poly],
                        text=str(row.get("text", "")),
                        confidence=float(row.get("confidence", 1.0)),
                        reading_order=i,
                        kind=str(row.get("kind") or row.get("label") or "unknown"),
                        meta={
                            "backend": "paddle",
                            "paddle_pipeline": str(self.selection.get("pipeline") or "ocr"),
                            "paddle_profile": str(self.selection.get("profile") or ""),
                            "paddle_pipeline_version": str(self.selection.get("pipeline_version") or ""),
                            "ocr_version": str(self.selection.get("ocr_version") or self.config.ocr_version),
                            "isolated_runtime": True, "model_source": getattr(self, "model_source", ""),
                            **dict(row.get("meta") or {}),
                        },
                    )
                )
            return sort_reading_order(blocks)
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._stdout_queue = None
        self._stdout_thread = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"cmd":"close"}\n'); proc.stdin.flush()
                proc.wait(timeout=3)
        except Exception:
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                try: proc.kill()
                except Exception: pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None: stream.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RetryingOCRBackend(OCRBackend):
    """Re-runs low-confidence crops at higher scale and keeps evidence for arbitration."""

    def __init__(self, primary: OCRBackend, threshold: float = 0.80, scale: float = 2.0) -> None:
        self.primary = primary
        self.threshold = threshold
        self.scale = max(1.0, scale)

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        blocks = self.primary.recognize(image, image_path=image_path)
        h, w = image.shape[:2]
        for block in blocks:
            if block.confidence >= self.threshold:
                continue
            x0, y0, x1, y1 = block.bbox
            pad = max(3, round(min(x1 - x0, y1 - y0) * 0.12))
            ix0, iy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ix1, iy1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            crop = image[iy0:iy1, ix0:ix1]
            up = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            sharpen = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.7, 0)
            variants = [up, cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)]
            if getattr(self.primary, "config", None) is not None and getattr(self.primary.config, "photo_ocr_preprocess", True):
                clip = float(getattr(self.primary.config, "photo_ocr_clahe_clip", 2.2))
                clahe = cv2.createCLAHE(clipLimit=max(1.0, clip), tileGridSize=(8, 8)).apply(gray)
                local = cv2.adaptiveThreshold(
                    clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 11,
                )
                variants.extend([
                    cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(local, cv2.COLOR_GRAY2BGR),
                ])
            candidates = [{"text": block.text, "confidence": block.confidence, "variant": "original"}]
            for vi, variant in enumerate(variants):
                try:
                    reread = self.primary.recognize(variant, image_path=None)
                except Exception:
                    continue
                if not reread:
                    continue
                # Prefer the most confident, text-bearing region in the crop.
                cand = max(reread, key=lambda b: (b.confidence, len(b.text.strip())))
                candidates.append({"text": cand.text, "confidence": cand.confidence, "variant": f"retry_{vi}"})
            best = max(candidates, key=lambda x: (x["confidence"], len(x["text"].strip())))
            block.meta["ocr_candidates"] = candidates
            if best["confidence"] > block.confidence + 0.015 and best["text"].strip():
                block.meta["ocr_original"] = {"text": block.text, "confidence": block.confidence}
                block.text = str(best["text"])
                block.confidence = float(best["confidence"])
                block.meta["ocr_selected"] = best["variant"]
        return blocks


class InjectedOCRBackend(OCRBackend):
    """Test/embedding adapter that returns caller-provided blocks by image path or one static list."""

    def __init__(self, blocks: list[TextBlock] | dict[str, list[TextBlock]]) -> None:
        self.blocks = blocks

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if isinstance(self.blocks, dict):
            key = str(image_path) if image_path is not None else ""
            values = self.blocks.get(key, [])
        else:
            values = self.blocks
        return [
            TextBlock(
                id=b.id,
                polygon=list(b.polygon),
                text=b.text,
                confidence=b.confidence,
                kind=b.kind,
                reading_order=b.reading_order,
                bubble_id=b.bubble_id,
                meta=dict(b.meta),
            )
            for b in values
        ]


def sort_reading_order(blocks: list[TextBlock], vertical: bool = False) -> list[TextBlock]:
    if vertical:
        ordered = sorted(blocks, key=lambda b: (-b.centroid[0], b.centroid[1]))
    else:
        # Row-aware sort: quantize Y by median block height to avoid tiny OCR jitter.
        heights = [max(1.0, b.bbox[3] - b.bbox[1]) for b in blocks]
        band = float(np.median(heights)) * 0.7 if heights else 16.0
        ordered = sorted(blocks, key=lambda b: (round(b.centroid[1] / max(1.0, band)), b.centroid[0]))
    for i, block in enumerate(ordered):
        block.reading_order = i
    return ordered


def filter_blocks(blocks: list[TextBlock], min_confidence: float) -> tuple[list[TextBlock], list[TextBlock]]:
    accepted, review = [], []
    for b in blocks:
        (accepted if b.confidence >= min_confidence else review).append(b)
    return accepted, review


def build_backend(config: OCRConfig, lang: str, backend: str | None = None, *, role: str | None = None) -> OCRBackend:
    name = (backend or config.backend).lower().strip()
    if name in {"manga_ocr", "manga-ocr", "baberu_ocr", "baberu-ocr", "ocr48px", "48px"}:
        from .koharu_ocr import KoharuCropOCRBackend
        normalized = {"manga-ocr":"manga_ocr", "baberu-ocr":"baberu_ocr", "48px":"ocr48px"}.get(name,name)
        return KoharuCropOCRBackend(normalized, lang, config)
    if name == "paddle":
        return PaddleOCRBackend(lang, config)
    from .paddle_profiles import backend_profile_key
    paddle_profile = backend_profile_key(name)
    if paddle_profile:
        return PaddleOCRBackend(lang, config, profile_override=paddle_profile)
    if name in {"external", "external_ocr", "external_json", "external_md"}:
        from .external_ocr import ExternalOCRBackend
        selected_role = str(role or "source").strip().lower()
        if selected_role == "target":
            path = getattr(config, "external_target_ocr_path", None)
            start_page = getattr(config, "external_target_start_page", 1)
        else:
            path = getattr(config, "external_source_ocr_path", None)
            start_page = getattr(config, "external_source_start_page", 1)
        if not path:
            raise RuntimeError(f"{selected_role.upper()} 已选择外部 OCR，但尚未指定 JSON/MD 文件。")
        return ExternalOCRBackend(path, start_page=start_page, ignored_labels=getattr(config, "external_ocr_ignore_labels", ()))
    # Apple OCR uses the same user-facing route as Novel Formatter:
    # Swift VisionKit Live Text first, macOS ExtractText Shortcut fallback.
    if name in {"apple", "apple_live_text", "apple_auto", "live_text"}:
        from .apple_live_text import AppleAutoLiveTextBackend
        return AppleAutoLiveTextBackend(lang, config)
    if name in {"apple_visionkit", "visionkit"}:
        from .apple_live_text import AppleVisionKitLiveTextBackend
        return AppleVisionKitLiveTextBackend(lang, config)
    if name in {"apple_shortcut", "shortcut", "shortcuts"}:
        from .apple_live_text import AppleShortcutLiveTextBackend
        return AppleShortcutLiveTextBackend(lang, config)
    if name == "sidecar":
        return SidecarOCRBackend(config.sidecar_suffix)
    if name == "none":
        return NullOCRBackend()
    raise ValueError(f"Unknown OCR backend: {name}")
