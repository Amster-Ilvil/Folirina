from __future__ import annotations

"""Platform/runtime policy for efficient local inference.

The module intentionally does not import torch at module import time.  This lets
macOS set ``PYTORCH_ENABLE_MPS_FALLBACK`` before the first torch import and keeps
core/OpenCV-only installs lightweight.
"""

import gc
import os
import platform
import threading
import sys
import importlib.util
from dataclasses import dataclass
from typing import Any

import cv2

from .m6_runtime_plan import build_m6_runtime_plan

from .accelerator_policy import (
    apply_torch_worker_policy,
    detect_apple_hardware,
    recommended_cpu_threads,
    recommended_mps_memory_fraction,
    resolve_precision,
    runtime_env_updates,
)


_ACCELERATOR_LOCK = threading.RLock()
_CONFIGURED = False
_LAST_POLICY: dict[str, Any] = {}
_LAST_THREADS = 1
_LAST_MPS_MEMORY_FRACTION = 0.82
_TORCH_POLICY_APPLIED = False


@dataclass(slots=True)
class DeviceInfo:
    requested: str
    selected: str
    available: bool
    name: str
    torch_installed: bool
    mps_built: bool = False
    mps_available: bool = False
    allocated_gb: float | None = None
    driver_allocated_gb: float | None = None
    recommended_max_gb: float | None = None
    chip: str = ""
    architecture: str = ""
    memory_gb: float | None = None
    hardware_profile: str = "generic"
    precision: str = "fp32"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "selected": self.selected,
            "available": self.available,
            "name": self.name,
            "torch_installed": self.torch_installed,
            "mps_built": self.mps_built,
            "mps_available": self.mps_available,
            "allocated_gb": self.allocated_gb,
            "driver_allocated_gb": self.driver_allocated_gb,
            "recommended_max_gb": self.recommended_max_gb,
            "chip": self.chip,
            "architecture": self.architecture,
            "memory_gb": self.memory_gb,
            "hardware_profile": self.hardware_profile,
            "precision": self.precision,
            "note": self.note,
        }


def _apply_torch_policy(torch) -> None:
    """Apply the shared live hardware policy after Torch is imported."""
    global _TORCH_POLICY_APPLIED
    if _TORCH_POLICY_APPLIED:
        return
    try:
        apply_torch_worker_policy(torch)
    finally:
        _TORCH_POLICY_APPLIED = True


def _import_torch():
    # CPU-only runs must remain usable when an optional/broken Torch build is
    # installed. OpenCV/CPU transfer routes do not require Torch.
    if os.environ.get("MANGA_HD_DISABLE_TORCH", "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    try:
        import torch
        _apply_torch_policy(torch)
        return torch
    except Exception:
        return None


def configure_runtime(config=None) -> dict[str, Any]:
    """Apply conservative CPU/MPS runtime limits.

    OpenCV/PyTorch default to consuming every CPU core which is often slower for
    this mixed I/O + CV + GPU workload and makes the Qt UI stutter.  The policy is
    deliberately idempotent and can be called from CLI, GUI and tests.
    """
    global _CONFIGURED, _LAST_POLICY, _LAST_THREADS, _LAST_MPS_MEMORY_FRACTION, _TORCH_POLICY_APPLIED

    ratio = float(getattr(config, "cpu_thread_ratio", 0.50) if config is not None else 0.50)
    min_threads = int(getattr(config, "min_cpu_threads", 1) if config is not None else 1)
    max_threads = int(getattr(config, "max_cpu_threads", 8) if config is not None else 8)
    cores = max(1, os.cpu_count() or 4)
    hw = detect_apple_hardware()
    m6_auto = bool(getattr(config, "m6_auto", True) if config is not None else True)
    thread_auto = bool(getattr(config, "cpu_thread_auto", True) if config is not None else True)
    auto_hw = bool(m6_auto and (hw.is_apple_silicon or hw.accelerator_backend != "cpu"))
    if thread_auto and auto_hw:
        threads = recommended_cpu_threads(hw, default=max(1, int(round(cores * ratio))))
        threads = max(min_threads, min(max_threads, int(threads)))
    else:
        threads = max(min_threads, min(max_threads, int(round(cores * ratio))))

    # Must be set before importing torch for unsupported MPS op fallback to work.
    if platform.system() == "Darwin" and bool(getattr(config, "mps_fallback", True) if config is not None else True):
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))

    try:
        cv2.setNumThreads(threads)
    except Exception:
        pass
    actual_opencv_threads = int(cv2.getNumThreads())

    _LAST_THREADS = int(threads)
    memory_auto = bool(getattr(config, "mps_memory_auto", True) if config is not None else True)
    configured_fraction = float(getattr(config, "mps_memory_fraction", 0.82) if config is not None else 0.82)
    _LAST_MPS_MEMORY_FRACTION = (
        recommended_mps_memory_fraction(hw) if memory_auto and hw.mps_available and m6_auto else configured_fraction
    )
    # Do not import Torch just to configure a run that may stay entirely on
    # OpenCV/CPU. If Torch was already loaded by a requested model, refresh its
    # limits; otherwise _import_torch() applies them lazily on first real use.
    loaded_torch = sys.modules.get("torch")
    if loaded_torch is not None:
        _TORCH_POLICY_APPLIED = False
        _apply_torch_policy(loaded_torch)

    # Propagate the selected policy to isolated Torch/vision workers.
    try:
        requested_precision = str(getattr(config, "mps_precision", "auto") if config is not None else "auto")
        effective_precision = requested_precision
        if requested_precision == "auto" and hw.is_apple_silicon and not m6_auto:
            effective_precision = "fp32"
        for key, value in runtime_env_updates(
            precision=effective_precision,
            memory_fraction=_LAST_MPS_MEMORY_FRACTION,
        ).items():
            os.environ[key] = str(value)
        os.environ["MHD_CPU_THREADS"] = str(threads)
    except Exception:
        pass

    _CONFIGURED = True
    _LAST_POLICY = {
        "cpu_cores": cores,
        "cpu_threads": threads,
        "opencv_effective_threads": actual_opencv_threads,
        "opencv_thread_limit_honored": actual_opencv_threads <= threads,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
        "mps_memory_fraction": _LAST_MPS_MEMORY_FRACTION,
        "mps_memory_auto": memory_auto,
        "mps_precision": effective_precision,
        "m6_auto": m6_auto,
        "hardware_profile": hw.profile,
        "chip": hw.chip,
        "memory_gb": hw.memory_gb,
        "platform": platform.platform(),
    }
    return dict(_LAST_POLICY)


def _mps_available(torch) -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


def select_device(preferred: str = "auto") -> str:
    preferred = (preferred or "auto").lower().strip()
    if preferred not in {"auto", "mps", "cuda", "cpu"}:
        preferred = "auto"
    torch = _import_torch()
    if torch is None:
        return "cpu"
    if preferred == "cpu":
        return "cpu"
    if preferred == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if preferred == "mps":
        return "mps" if _mps_available(torch) else "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def torch_device(preferred: str = "auto"):
    torch = _import_torch()
    if torch is None:
        raise RuntimeError("PyTorch is not installed")
    return torch.device(select_device(preferred))


def device_info(preferred: str = "auto", *, probe_torch: bool = True) -> DeviceInfo:
    preferred = (preferred or "auto").lower().strip()
    hw = detect_apple_hardware()
    if preferred == "cpu":
        installed = importlib.util.find_spec("torch") is not None
        return DeviceInfo(
            preferred, "cpu", True, "CPU", installed, chip=hw.chip, architecture=hw.architecture,
            memory_gb=hw.memory_gb, hardware_profile=hw.profile, precision="fp32", note="CPU 模式。"
        )
    if not probe_torch and "torch" not in sys.modules:
        installed = importlib.util.find_spec("torch") is not None
        return DeviceInfo(
            preferred, "cpu", True, "CPU", installed, chip=hw.chip, architecture=hw.architecture,
            memory_gb=hw.memory_gb, hardware_profile=hw.profile, precision="fp32",
            note="Torch 未加载；本次当前路径保持 OpenCV/CPU，未为设备探测额外加载 Torch。",
        )
    torch = _import_torch()
    if torch is None:
        return DeviceInfo(
            preferred, "cpu", True, "CPU", False, chip=hw.chip, architecture=hw.architecture,
            memory_gb=hw.memory_gb, hardware_profile=hw.profile, precision="fp32",
            note="PyTorch 未安装；仅使用 OpenCV/CPU。"
        )

    selected = select_device(preferred)
    mps_built = False
    mps_available = False
    try:
        mps_built = bool(torch.backends.mps.is_built())
        mps_available = bool(torch.backends.mps.is_available())
    except Exception:
        pass

    if selected == "mps":
        name = "Apple MPS"
        try:
            if hasattr(torch.backends.mps, "get_name"):
                name = str(torch.backends.mps.get_name()) or name
        except Exception:
            pass
        def gb(fn):
            try:
                return round(float(fn()) / 1024**3, 3)
            except Exception:
                return None
        precision = resolve_precision("mps", os.environ.get("MHD_MPS_PRECISION", "auto"), hw)
        return DeviceInfo(
            preferred, selected, True, name, True,
            mps_built=mps_built, mps_available=mps_available,
            allocated_gb=gb(getattr(torch.mps, "current_allocated_memory", lambda: 0)),
            driver_allocated_gb=gb(getattr(torch.mps, "driver_allocated_memory", lambda: 0)),
            recommended_max_gb=gb(getattr(torch.mps, "recommended_max_memory", lambda: 0)),
            chip=hw.chip, architecture=hw.architecture, memory_gb=hw.memory_gb,
            hardware_profile=hw.profile, precision=precision,
            note="MPS 深度模型可使用硬件感知精度；失败时同设备回退 FP32，再回退 CPU。OpenCV/SIFT 继续走 CPU。",
        )
    if selected == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
        return DeviceInfo(
            preferred, selected, True, name, True, mps_built=mps_built, mps_available=mps_available,
            chip=hw.chip, architecture=hw.architecture, memory_gb=hw.memory_gb,
            hardware_profile=hw.profile, precision="fp32",
        )
    if preferred == "mps":
        note = "MPS 不可用，已回退 CPU。"
    elif preferred == "cuda":
        note = "CUDA 不可用，已回退 CPU。"
    else:
        note = "未检测到可用 CUDA/MPS，使用 CPU。"
    return DeviceInfo(
        preferred, "cpu", True, "CPU", True, mps_built=mps_built, mps_available=mps_available,
        chip=hw.chip, architecture=hw.architecture, memory_gb=hw.memory_gb,
        hardware_profile=hw.profile, precision="fp32", note=note
    )


def accelerator_lock():
    """Return the global inference lock.

    MPS model inference is intentionally serialized. CPU preprocessing may run in
    other threads, but concurrent Metal model execution causes memory spikes and
    often lowers throughput on unified-memory Macs.
    """
    return _ACCELERATOR_LOCK


def synchronize(preferred: str = "auto") -> None:
    # Synchronization is meaningful only after a Torch-backed operation actually
    # loaded Torch. Do not import a multi-hundred-MB optional backend just to ask
    # an unused accelerator to synchronize.
    if "torch" not in sys.modules:
        return
    torch = _import_torch()
    if torch is None:
        return
    device = select_device(preferred)
    try:
        if device == "mps" and _mps_available(torch):
            torch.mps.synchronize()
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def empty_accelerator_cache(preferred: str = "auto") -> None:
    gc.collect()
    # Same lazy rule as synchronize(): if Torch was never used there is no Torch
    # cache to release, and importing it here would create the very memory/cost we
    # are trying to avoid every N pages.
    if "torch" not in sys.modules:
        return
    torch = _import_torch()
    if torch is None:
        return
    device = select_device(preferred)
    try:
        if device == "mps" and _mps_available(torch):
            torch.mps.empty_cache()
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def runtime_summary(preferred: str = "auto", *, probe_torch: bool = False) -> dict[str, Any]:
    if not _CONFIGURED:
        configure_runtime(None)
    hw = detect_apple_hardware()
    from .accelerator_policy import hardware_fingerprint, memory_budget_gb, current_available_memory_gb
    hardware = hw.to_dict()
    hardware.update({
        "hardware_fingerprint": hardware_fingerprint(hw),
        "available_memory_gb": current_available_memory_gb(),
        "memory_budget_gb": round(memory_budget_gb(hw), 2),
    })
    summary = {
        "policy": dict(_LAST_POLICY),
        "hardware": hardware,
        "device": device_info(preferred, probe_torch=probe_torch).to_dict(),
        "m6_runtime_plan": build_m6_runtime_plan(None, hw=hw).to_dict(),
    }
    try:
        from .m6_runtime_plan import m6_ocr_backend_order
        summary["m6_japanese_ocr"] = {
            "order": list(m6_ocr_backend_order(config=None, japanese_vertical=True, hw=hw)),
            "policy": "whole-page NDL → crop Hayai → crop 48px",
        }
    except Exception as exc:
        summary["m6_japanese_ocr"] = {"order": [], "error": f"{type(exc).__name__}: {exc}"}
    try:
        from .mlx_runtime import status as mlx_status
        summary["mlx"] = mlx_status()
    except Exception as exc:
        summary["mlx"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import onnxruntime as ort
        from .onnx_coreml import probe_coreml
        summary["coreml"] = {"version": getattr(ort, "__version__", ""), **probe_coreml(ort)}
    except Exception as exc:
        summary["coreml"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return summary
