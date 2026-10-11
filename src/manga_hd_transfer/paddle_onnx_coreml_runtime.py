from __future__ import annotations

"""Optional PaddleOCR -> ONNX Runtime -> CoreML bridge.

This module is intentionally capability-only in the desktop process. The heavy
Paddle/ONNX stack remains inside the existing isolated Paddle venv. Nothing in
this file changes the default ``paddle`` backend or silently converts models.
"""

from dataclasses import dataclass
import os
import platform
import subprocess
from pathlib import Path


ORT_PACKAGE_SPEC = "onnxruntime>=1.26,<1.31"


@dataclass(frozen=True, slots=True)
class PaddleONNXCoreMLStatus:
    ready: bool
    python: str | None
    onnxruntime_version: str = ""
    coreml_available: bool = False
    paddleocr_version: str = ""
    engine_api: bool = False
    detail: str = ""


def provider_options(compute_units: str = "ane") -> dict[str, str]:
    key = str(compute_units or "ane").lower().strip()
    mapping = {
        "ane": "CPUAndNeuralEngine",
        "cpu_and_neural_engine": "CPUAndNeuralEngine",
        "gpu": "CPUAndGPU",
        "cpu_and_gpu": "CPUAndGPU",
        "all": "ALL",
        "cpu": "CPUOnly",
    }
    return {
        "MLComputeUnits": mapping.get(key, "CPUAndNeuralEngine"),
        "ModelFormat": "MLProgram",
    }


def runtime_status() -> PaddleONNXCoreMLStatus:
    try:
        from .paddle_runtime import runtime_status as paddle_runtime_status
        base = paddle_runtime_status()
    except Exception as exc:
        return PaddleONNXCoreMLStatus(False, None, detail=f"Paddle runtime probe failed: {exc}")
    if not base.ready or not base.python:
        return PaddleONNXCoreMLStatus(False, base.python, detail="PaddleOCR isolated runtime is not ready.")
    py = Path(base.python)
    if platform.system() == "Darwin" and platform.machine().lower() not in {"arm64", "aarch64"}:
        return PaddleONNXCoreMLStatus(False, str(py), detail="CoreML bridge requires native Apple Silicon Python.")
    script = r'''
import json
from importlib.metadata import version, PackageNotFoundError
try:
    import onnxruntime as ort
    providers = list(ort.get_available_providers())
    ort_ver = version("onnxruntime")
except Exception as exc:
    print(json.dumps({"ok":False,"error":f"onnxruntime: {type(exc).__name__}: {exc}"})); raise SystemExit(0)
try:
    import paddleocr
    pver = version("paddleocr")
    cls = getattr(paddleocr, "PaddleOCR", None)
    import inspect
    engine_api = bool(cls is not None and "engine" in inspect.signature(cls).parameters)
except Exception as exc:
    pver = ""
    engine_api = False
    print(json.dumps({"ok":True,"ort_ver":ort_ver,"providers":providers,"paddleocr_ver":pver,"engine_api":engine_api,"detail":f"PaddleOCR probe: {type(exc).__name__}: {exc}"})); raise SystemExit(0)
print(json.dumps({"ok":True,"ort_ver":ort_ver,"providers":providers,"paddleocr_ver":pver,"engine_api":engine_api}))
'''
    try:
        proc = subprocess.run([str(py), "-c", script], capture_output=True, text=True, timeout=30)
        raw = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        import json
        payload = json.loads(raw) if raw else {}
    except Exception as exc:
        return PaddleONNXCoreMLStatus(False, str(py), detail=f"CoreML bridge probe failed: {type(exc).__name__}: {exc}")
    if proc.returncode != 0 or not payload.get("ok"):
        return PaddleONNXCoreMLStatus(False, str(py), detail=str(payload.get("error") or "ONNX Runtime unavailable"))
    providers = {str(x) for x in payload.get("providers") or []}
    coreml = "CoreMLExecutionProvider" in providers
    engine_api = bool(payload.get("engine_api"))
    ready = bool(coreml and engine_api and platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"})
    detail = "ONNXRuntime CoreML + PaddleOCR engine=onnxruntime ready" if ready else ""
    if not coreml:
        detail = "ONNXRuntime CoreMLExecutionProvider unavailable in Paddle venv."
    elif not engine_api:
        detail = "Installed PaddleOCR does not expose the engine=onnxruntime API."
    return PaddleONNXCoreMLStatus(
        ready=ready, python=str(py), onnxruntime_version=str(payload.get("ort_ver") or ""),
        coreml_available=coreml, paddleocr_version=str(payload.get("paddleocr_ver") or ""),
        engine_api=engine_api, detail=detail,
    )


def ensure_onnxruntime(progress=None) -> PaddleONNXCoreMLStatus:
    """Install ONNX Runtime into the existing Paddle venv only when explicitly requested."""
    status = runtime_status()
    if status.coreml_available and status.engine_api:
        return status
    if not status.python:
        raise RuntimeError(status.detail or "PaddleOCR Python runtime unavailable")
    import subprocess as sp
    cmd = [str(status.python), "-m", "pip", "install", "--upgrade", "--prefer-binary", ORT_PACKAGE_SPEC]
    if progress:
        progress("为 Paddle ONNXRuntime/CoreML 路径安装隔离依赖…")
    proc = sp.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        tail = (proc.stdout or "")[-1500:] + "\n" + (proc.stderr or "")[-1500:]
        raise RuntimeError("ONNX Runtime 安装失败：" + tail)
    return runtime_status()


__all__ = ["PaddleONNXCoreMLStatus", "provider_options", "runtime_status", "ensure_onnxruntime", "ORT_PACKAGE_SPEC"]
