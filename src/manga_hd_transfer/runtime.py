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
from dataclasses import dataclass
from typing import Any

import cv2


_ACCELERATOR_LOCK = threading.RLock()
_CONFIGURED = False
_LAST_POLICY: dict[str, Any] = {}


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
            "note": self.note,
        }


def _import_torch():
    try:
        import torch
        return torch
    except Exception:
        return None


def configure_runtime(config=None) -> dict[str, Any]:
    """Apply conservative CPU/MPS runtime limits.

    OpenCV/PyTorch default to consuming every CPU core which is often slower for
    this mixed I/O + CV + GPU workload and makes the Qt UI stutter.  The policy is
    deliberately idempotent and can be called from CLI, GUI and tests.
    """
    global _CONFIGURED, _LAST_POLICY

    ratio = float(getattr(config, "cpu_thread_ratio", 0.50) if config is not None else 0.50)
    min_threads = int(getattr(config, "min_cpu_threads", 1) if config is not None else 1)
    max_threads = int(getattr(config, "max_cpu_threads", 8) if config is not None else 8)
    cores = max(1, os.cpu_count() or 4)
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

    torch = _import_torch()
    if torch is not None:
        try:
            torch.set_num_threads(threads)
        except Exception:
            pass
        try:
            # Inter-op parallelism has high overhead for our page-at-a-time jobs.
            torch.set_num_interop_threads(max(1, min(2, threads)))
        except RuntimeError:
            # PyTorch only allows this before parallel work starts.
            pass
        except Exception:
            pass

        if _mps_available(torch):
            fraction = float(getattr(config, "mps_memory_fraction", 0.82) if config is not None else 0.82)
            if 0.10 <= fraction <= 1.0 and hasattr(torch.mps, "set_per_process_memory_fraction"):
                try:
                    torch.mps.set_per_process_memory_fraction(fraction)
                except Exception:
                    pass

    _CONFIGURED = True
    _LAST_POLICY = {
        "cpu_cores": cores,
        "cpu_threads": threads,
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0"),
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


def device_info(preferred: str = "auto") -> DeviceInfo:
    torch = _import_torch()
    if torch is None:
        return DeviceInfo(preferred, "cpu", True, "CPU", False, note="PyTorch 未安装；仅使用 OpenCV/CPU。")

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
        return DeviceInfo(
            preferred, selected, True, name, True,
            mps_built=mps_built, mps_available=mps_available,
            allocated_gb=gb(getattr(torch.mps, "current_allocated_memory", lambda: 0)),
            driver_allocated_gb=gb(getattr(torch.mps, "driver_allocated_memory", lambda: 0)),
            recommended_max_gb=gb(getattr(torch.mps, "recommended_max_memory", lambda: 0)),
            note="LightGlue / LoFTR / MangaLens / 可选 Torch 超分可走 MPS；OpenCV/SIFT 继续走 CPU。",
        )
    if selected == "cuda":
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA"
        return DeviceInfo(preferred, selected, True, name, True, mps_built=mps_built, mps_available=mps_available)
    note = "MPS 不可用。" if preferred == "mps" else "CPU 模式。"
    return DeviceInfo(preferred, "cpu", True, "CPU", True, mps_built=mps_built, mps_available=mps_available, note=note)


def accelerator_lock():
    """Return the global inference lock.

    MPS model inference is intentionally serialized. CPU preprocessing may run in
    other threads, but concurrent Metal model execution causes memory spikes and
    often lowers throughput on unified-memory Macs.
    """
    return _ACCELERATOR_LOCK


def synchronize(preferred: str = "auto") -> None:
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


def runtime_summary(preferred: str = "auto") -> dict[str, Any]:
    if not _CONFIGURED:
        configure_runtime(None)
    return {"policy": dict(_LAST_POLICY), "device": device_info(preferred).to_dict()}
