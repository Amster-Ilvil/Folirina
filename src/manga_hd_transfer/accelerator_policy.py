from __future__ import annotations

"""Cross-platform, capability-driven accelerator/resource policy.

Inspired by Novel Formatter's split between stable hardware facts and live
resource pressure: hardware identity is cached briefly, available memory is
re-read for recommendations, and accelerator capability is reported without
forcing a heavyweight model import.  Existing Folirina call sites keep using
``AppleHardwareInfo`` for compatibility, but the object now describes generic
Apple/CPU/CUDA/ONNX environments as well.
"""

from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from typing import Any


@dataclass(frozen=True, slots=True)
class AppleHardwareInfo:
    platform: str
    architecture: str
    chip: str
    generation: str
    memory_gb: float | None
    cpu_cores: int | None
    macos: str
    is_apple_silicon: bool
    is_m6: bool
    # Generic runtime capability fields.  Defaults preserve the old positional
    # constructor used by existing tests and integrations.
    available_memory_gb: float | None = None
    accelerator_backend: str = "cpu"
    accelerator_name: str = ""
    accelerator_memory_gb: float | None = None
    accelerator_free_memory_gb: float | None = None
    mps_available: bool = False
    cuda_available: bool = False
    mlx_available: bool = False
    onnx_providers: tuple[str, ...] = ()
    unified_memory: bool = False

    @property
    def profile(self) -> str:
        mem = float(self.memory_gb or 0.0)
        if self.is_apple_silicon:
            tier = "conservative" if mem <= 17.0 else ("balanced" if mem <= 25.0 else "performance")
            family = self.generation.lower() or "apple"
            return f"{family}_{tier}"
        if self.accelerator_backend == "cuda":
            vram = float(self.accelerator_memory_gb or 0.0)
            return "cuda_high" if vram >= 16 else ("cuda_medium" if vram >= 8 else "cuda_low")
        if self.accelerator_backend in {"directml", "migraphx", "openvino"}:
            return f"{self.accelerator_backend}_accelerated"
        return "generic"

    @property
    def accelerator_available(self) -> bool:
        return self.accelerator_backend != "cpu" or self.mlx_available

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform": self.platform,
            "architecture": self.architecture,
            "chip": self.chip,
            "generation": self.generation,
            "memory_gb": self.memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "cpu_cores": self.cpu_cores,
            "macos": self.macos,
            "is_apple_silicon": self.is_apple_silicon,
            "is_m6": self.is_m6,
            "profile": self.profile,
            "accelerator_backend": self.accelerator_backend,
            "accelerator_name": self.accelerator_name,
            "accelerator_memory_gb": self.accelerator_memory_gb,
            "accelerator_free_memory_gb": self.accelerator_free_memory_gb,
            "mps_available": self.mps_available,
            "cuda_available": self.cuda_available,
            "mlx_available": self.mlx_available,
            "onnx_providers": list(self.onnx_providers),
            "unified_memory": self.unified_memory or self.is_apple_silicon,
        }


_STABLE: AppleHardwareInfo | None = None
_STABLE_TS = 0.0


def _run(command: list[str], timeout: float = 4.0) -> str:
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception:
        return ""
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


def _sysctl(name: str) -> str:
    if str(sys.platform).lower() != "darwin":
        return ""
    return _run(["/usr/sbin/sysctl", "-n", name], timeout=4.0)


def _windows_gpu_info() -> tuple[str, float]:
    if os.name != "nt":
        return "", 0.0
    tool = shutil.which("powershell") or shutil.which("pwsh")
    if not tool:
        return "", 0.0
    script = "Get-CimInstance Win32_VideoController | Select-Object -First 1 Name,AdapterRAM | ConvertTo-Csv -NoTypeInformation"
    raw = _run([tool, "-NoProfile", "-Command", script], timeout=6.0)
    lines = raw.splitlines()
    if len(lines) < 2:
        return "", 0.0
    try:
        import csv
        row = next(csv.reader([lines[1]]))
        name = str(row[0]).strip() if row else ""
        ram_text = row[1].strip() if len(row) > 1 else ""
        ram = float(ram_text) / (1024 ** 3) if ram_text.isdigit() else 0.0
        return name, ram
    except Exception:
        return "", 0.0


def _memory_bytes() -> tuple[float, float]:
    override = os.environ.get("MHD_TEST_TOTAL_MEMORY_GB", "").strip() or os.environ.get("NOVEL_FORMATTER_TEST_TOTAL_MEMORY_GB", "").strip()
    if override:
        try:
            total = float(override) * (1024 ** 3)
            available_text = os.environ.get("MHD_TEST_AVAILABLE_MEMORY_GB", "").strip() or os.environ.get("NOVEL_FORMATTER_TEST_AVAILABLE_MEMORY_GB", "").strip()
            available = float(available_text) * (1024 ** 3) if available_text else total * 0.60
            return total, min(total, max(0.0, available))
        except ValueError:
            pass
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return float(vm.total), float(vm.available)
    except Exception:
        pass
    if os.name == "nt":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sAvailExtendedVirtual", ctypes.c_ulonglong)]
            state = MEMORYSTATUSEX(); state.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):
                return float(state.ullTotalPhys), float(state.ullAvailPhys)
        except Exception:
            pass
    if sys.platform == "darwin":
        raw = _sysctl("hw.memsize")
        try:
            total = float(int(raw))
        except ValueError:
            total = 0.0
        return total, 0.0
    try:
        total = float(int(_run([shutil.which("getconf") or "getconf", "_PHYS_PAGES"]))) * float(int(_run([shutil.which("getconf") or "getconf", "PAGE_SIZE"])))
        return total, total * 0.55
    except Exception:
        return 0.0, 0.0


def current_available_memory_gb() -> float:
    _, available = _memory_bytes()
    return round(available / (1024 ** 3), 1) if available else 0.0


def hardware_fingerprint(hw: AppleHardwareInfo | None = None) -> str:
    """Return a stable local fingerprint for the concrete hardware/runtime facts.

    Live available memory is intentionally excluded so pressure changes do not
    invalidate calibration or page caches.
    """
    hw = hw or detect_system_hardware()
    payload = {
        "platform": hw.platform,
        "architecture": hw.architecture,
        "chip": hw.chip,
        "generation": hw.generation,
        "memory_gb": hw.memory_gb,
        "cpu_cores": hw.cpu_cores,
        "accelerator_backend": hw.accelerator_backend,
        "accelerator_name": hw.accelerator_name,
        "accelerator_memory_gb": hw.accelerator_memory_gb,
        "onnx_providers": list(hw.onnx_providers),
        "unified_memory": bool(hw.unified_memory or hw.is_apple_silicon),
    }
    raw = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def _chip_generation(chip: str) -> str:
    text = str(chip or "").upper().replace("APPLE", "").strip()
    for generation in ("M10", "M9", "M8", "M7", "M6", "M5", "M4", "M3", "M2", "M1"):
        if generation in text:
            return generation
    return ""


def _apple_identity() -> tuple[bool, str, str, bool]:
    machine = platform.machine().strip().lower()
    apple = platform.system() == "Darwin" and machine in {"arm64", "aarch64"}
    if not apple:
        return False, "", "", False
    chip = os.environ.get("MHD_TEST_APPLE_CHIP", "").strip() or _sysctl("machdep.cpu.brand_string")
    if not chip:
        raw = _run(["/usr/sbin/system_profiler", "SPHardwareDataType", "-json"], timeout=20.0)
        try:
            rows = json.loads(raw).get("SPHardwareDataType", [])
            chip = next((str(row.get("chip_type") or row.get("cpu_type") or "").strip() for row in rows if isinstance(row, dict)), "")
        except Exception:
            chip = ""
    if not chip:
        chip = platform.processor() or "Apple Silicon"
    generation = _chip_generation(chip)
    return True, chip, generation, generation == "M6"


def _probe_onnx_providers() -> tuple[str, ...]:
    try:
        import onnxruntime as ort  # type: ignore
        return tuple(str(x) for x in ort.get_available_providers())
    except Exception:
        return ()


def _probe_mps() -> bool:
    if str(sys.platform).lower() != "darwin":
        return False
    try:
        torch = sys.modules.get("torch")
        if torch is not None:
            return bool(getattr(getattr(torch, "backends", None), "mps", None) and torch.backends.mps.is_available())
        # Capability discovery must not load the heavyweight runtime into Qt.
        return _run([sys.executable, "-c", "import torch; print(int(torch.backends.mps.is_available()))"], timeout=8.0).strip() == "1"
    except Exception:
        return False


def _probe_mlx() -> bool:
    if str(sys.platform).lower() != "darwin":
        return False
    try:
        import mlx.core as mx  # type: ignore
        return bool(mx)
    except Exception:
        return False


def _nvidia_info() -> tuple[str, float, float]:
    tool = shutil.which("nvidia-smi")
    if not tool:
        return "", 0.0, 0.0
    raw = _run([tool, "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"], timeout=5.0)
    if not raw:
        return "", 0.0, 0.0
    parts = [x.strip() for x in raw.splitlines()[0].split(",")]
    try:
        return parts[0], float(parts[1]) / 1024.0, float(parts[2]) / 1024.0
    except Exception:
        return (parts[0] if parts else "NVIDIA GPU"), 0.0, 0.0


def detect_system_hardware(*, refresh: bool = False) -> AppleHardwareInfo:
    """Detect hardware facts once briefly; recommendations use live memory separately."""
    global _STABLE, _STABLE_TS
    now = time.monotonic()
    if _STABLE is not None and not refresh and now - _STABLE_TS < 30.0:
        live = current_available_memory_gb()
        if live:
            return replace(_STABLE, available_memory_gb=live)
        return _STABLE

    total, available = _memory_bytes()
    total_gb = round(total / (1024 ** 3), 1) if total else None
    available_gb = round(available / (1024 ** 3), 1) if available else None
    apple, chip, generation, is_m6 = _apple_identity()
    machine = platform.machine().lower()
    cores = max(1, int(os.cpu_count() or 1))
    onnx = _probe_onnx_providers()
    nvidia_name, nvidia_vram, nvidia_free = _nvidia_info()
    windows_gpu_name, windows_gpu_vram = _windows_gpu_info()
    mps = apple and _probe_mps()
    mlx = apple and _probe_mlx()
    cuda = bool(nvidia_name) or "CUDAExecutionProvider" in onnx
    directml = os.name == "nt" and "DmlExecutionProvider" in onnx
    migraphx = "MIGraphXExecutionProvider" in onnx
    openvino = "OpenVINOExecutionProvider" in onnx or "IntelExecutionProvider" in onnx

    backend = "cpu"
    acc_name = ""
    acc_mem = acc_free = None
    if mps:
        backend, acc_name = "mps", chip or "Apple MPS"
    elif cuda:
        backend, acc_name = "cuda", nvidia_name or "CUDA"
        acc_mem, acc_free = (nvidia_vram or None), (nvidia_free or None)
    elif directml:
        backend, acc_name = "directml", windows_gpu_name or "DirectML GPU"
        acc_mem, acc_free = (windows_gpu_vram or None), None
    elif migraphx:
        backend, acc_name = "migraphx", "AMD GPU"
    elif openvino:
        backend, acc_name = "openvino", windows_gpu_name or "Intel/OpenVINO"
        acc_mem, acc_free = (windows_gpu_vram or None), (windows_gpu_vram or None)
    elif apple:
        backend, acc_name = "apple_gpu", chip or "Apple GPU"

    _STABLE = AppleHardwareInfo(
        platform=platform.system(),
        architecture=machine,
        chip=chip,
        generation=generation,
        memory_gb=total_gb,
        cpu_cores=cores,
        macos=platform.mac_ver()[0] or "",
        is_apple_silicon=apple,
        is_m6=is_m6,
        available_memory_gb=available_gb,
        accelerator_backend=backend,
        accelerator_name=acc_name,
        accelerator_memory_gb=acc_mem,
        accelerator_free_memory_gb=acc_free,
        mps_available=mps,
        cuda_available=cuda,
        mlx_available=mlx,
        onnx_providers=onnx,
        unified_memory=apple,
    )
    _STABLE_TS = now
    return _STABLE


def detect_apple_hardware() -> AppleHardwareInfo:
    """Backward-compatible alias used throughout Folirina."""
    return detect_system_hardware()


def memory_budget_gb(hw: AppleHardwareInfo | None = None) -> float:
    hw = hw or detect_system_hardware()
    available = float(hw.available_memory_gb or 0.0)
    if available <= 0:
        # A caller-supplied hardware object may intentionally omit live pressure
        # (e.g. deterministic tests or an offline diagnostic). Use a stable
        # fraction instead of borrowing an unrelated host process's memory.
        available = float(hw.memory_gb or 0.0) * 0.60 or current_available_memory_gb()
    total = float(hw.memory_gb or 0.0)
    if total <= 0:
        return 2.0
    if hw.unified_memory or hw.is_apple_silicon:
        return max(1.5, min(total * 0.50, available * 0.78))
    system_budget = max(1.5, min(total * 0.60, available * 0.82))
    free_vram = float(hw.accelerator_free_memory_gb or 0.0)
    if free_vram and hw.accelerator_backend in {"cuda", "directml", "migraphx", "openvino"}:
        return max(1.5, min(system_budget, free_vram * 0.72))
    return system_budget


def recommended_mps_memory_fraction(hw: AppleHardwareInfo | None = None) -> float:
    hw = hw or detect_system_hardware()
    if hw.accelerator_backend != "mps":
        # Compatibility for callers/tests that construct AppleHardwareInfo
        # directly without probing Torch; real detected MPS uses live headroom.
        if hw.is_m6 and hw.memory_gb:
            mem = float(hw.memory_gb)
            return 0.75 if mem <= 17.0 else (0.82 if mem <= 25.0 else 0.88)
        return 0.82
    # Live-headroom policy: avoid fixed 16/24/32 GB assumptions.  PyTorch's
    # recommended max is obtained in the Torch process itself when available.
    available = float(hw.available_memory_gb or 0.0)
    if available <= 0:
        available = float(hw.memory_gb or 0.0) * 0.60 or current_available_memory_gb()
    total = float(hw.memory_gb or 0.0)
    if total <= 0:
        return 0.82
    reserve = 0.58 if available < 3 else (0.70 if available < 5 else (0.82 if available < 8 else 0.90))
    live_fraction = min(0.95, max(0.55, (available / total) * reserve * 1.55))
    return round(live_fraction, 3)


def _available_for(hw: AppleHardwareInfo) -> float:
    value = float(hw.available_memory_gb or 0.0)
    return value if value > 0 else current_available_memory_gb()


def recommended_cpu_threads(hw: AppleHardwareInfo | None = None, *, default: int = 8) -> int:
    hw = hw or detect_system_hardware()
    cores = int(hw.cpu_cores or default or 1)
    reserve = 4 if (hw.unified_memory or hw.is_apple_silicon) else 2
    # Only apply the extra low-memory reserve when the hardware snapshot itself
    # carries live headroom.  Explicit/synthetic snapshots (used by policy
    # callers and tests) must not inherit pressure from the machine executing
    # this function; that made identical M6 profiles return different thread
    # counts depending on the host running the check.
    available = float(hw.available_memory_gb or 0.0)
    if 0.0 < available < 3.0:
        reserve += 1
    return max(2, min(12, cores - reserve))


def recommended_prepare_workers(hw: AppleHardwareInfo | None = None) -> int:
    hw = hw or detect_system_hardware()
    cores = int(hw.cpu_cores or 4)
    available = _available_for(hw)
    unified = bool(hw.unified_memory or hw.is_apple_silicon)
    cap = 8 if unified else 6
    workers = min(cap, max(2, cores - (4 if unified else 2)))
    if 0 < available < 3:
        workers = min(workers, 2)
    elif 0 < available < 5:
        workers = min(workers, 4)
    return workers


def recommended_encode_workers(hw: AppleHardwareInfo | None = None) -> int:
    hw = hw or detect_system_hardware()
    available = _available_for(hw)
    unified = bool(hw.unified_memory or hw.is_apple_silicon)
    workers = min(4 if unified else 3, max(1, int(hw.cpu_cores or 4) // 4))
    if 0 < available < 3:
        workers = 1
    return workers


def recommended_batch(
    hw: AppleHardwareInfo | None = None,
    *,
    memory_cost_per_item: float = 0.75,
    cap: int = 12,
    cpu_default: int = 4,
) -> int:
    hw = hw or detect_system_hardware()
    budget = memory_budget_gb(hw)
    if hw.accelerator_backend == "cpu" and not hw.is_apple_silicon:
        value = max(1, min(cap, max(cpu_default, recommended_cpu_threads(hw) // 2)))
    else:
        value = int(budget / max(0.25, memory_cost_per_item))
        if _available_for(hw) and _available_for(hw) < 2.5:
            value = min(value, 2)
    value = max(1, min(cap, value))
    return max(1, value)

def resolve_precision(device: str, requested: str = "auto", hw: AppleHardwareInfo | None = None) -> str:
    key = str(requested or "auto").lower().strip()
    if key not in {"auto", "fp32", "fp16", "bf16"}:
        key = "auto"
    if key != "auto":
        return key
    hw = hw or detect_system_hardware()
    if str(device).lower().strip() == "mps" and hw.is_apple_silicon:
        return "fp16"
    return "fp32"


def env_precision(default: str = "auto") -> str:
    return str(os.environ.get("MHD_MPS_PRECISION", default) or default).strip().lower()


def autocast_context(torch, device: str, requested: str = "auto", hw: AppleHardwareInfo | None = None):
    resolved = resolve_precision(device, requested, hw)
    if str(device).lower().strip() != "mps" or resolved == "fp32":
        return nullcontext()
    dtype = torch.float16 if resolved == "fp16" else torch.bfloat16
    try:
        return torch.autocast(device_type="mps", dtype=dtype)
    except Exception:
        return nullcontext()


def apply_torch_worker_policy(torch) -> None:
    """Apply thread and live MPS safety policy inside the actual Torch process."""
    try:
        raw_threads = str(os.environ.get("MHD_CPU_THREADS", "") or "").strip()
        if raw_threads:
            threads = max(1, int(raw_threads))
        else:
            cuda_ready = bool(getattr(getattr(torch, "cuda", None), "is_available", lambda: False)())
            mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
            mps_ready = bool(mps_backend is not None and getattr(mps_backend, "is_available", lambda: False)())
            threads = min(4, max(1, int(os.cpu_count() or 1))) if not cuda_ready and not mps_ready else 0
        if threads > 0:
            torch.set_num_threads(int(threads))
            try:
                torch.set_num_interop_threads(max(1, min(2, int(threads))))
            except RuntimeError:
                pass
    except Exception:
        pass
    try:
        mps = getattr(torch, "mps", None)
        backends = getattr(torch, "backends", None)
        mps_backend = getattr(backends, "mps", None) if backends is not None else None
        fraction = float(os.environ.get("MHD_MPS_MEMORY_FRACTION", "0") or 0)
        # Test doubles and lightweight Torch shims may expose mps without a
        # full backends.mps object; preserve the inherited env setting there.
        if mps is None:
            return
        if mps_backend is None:
            if 0.10 <= fraction <= 1.0 and hasattr(mps, "set_per_process_memory_fraction"):
                mps.set_per_process_memory_fraction(float(fraction))
            return
        if not mps_backend.is_available():
            return
        try:
            recommended = int(mps.recommended_max_memory()) if hasattr(mps, "recommended_max_memory") else 0
        except Exception:
            recommended = 0
        available_gb = current_available_memory_gb()
        if recommended > 0 and available_gb > 0:
            reserve_fraction = 0.90
            if available_gb < 3.0:
                reserve_fraction = 0.58
            elif available_gb < 5.0:
                reserve_fraction = 0.70
            elif available_gb < 8.0:
                reserve_fraction = 0.82
            available_bytes = available_gb * (1024 ** 3)
            target = min(recommended * reserve_fraction, available_bytes * 0.82)
            live_fraction = max(0.55, min(1.0, target / recommended))
            if 0.10 <= live_fraction <= 1.0:
                fraction = live_fraction
        if 0.10 <= fraction <= 1.0 and hasattr(mps, "set_per_process_memory_fraction"):
            mps.set_per_process_memory_fraction(float(fraction))
    except Exception:
        # Runtime safety is an optimization hint. Never prevent model startup.
        pass


def runtime_env_updates(*, precision: str = "auto", memory_fraction: float | None = None) -> dict[str, str]:
    hw = detect_system_hardware()
    effective = str(precision or "auto")
    return {
        "MHD_HARDWARE_PROFILE": hw.profile,
        "MHD_HARDWARE_JSON": json.dumps(hw.to_dict(), ensure_ascii=False, separators=(",", ":")),
        "MHD_ACCELERATOR_BACKEND": hw.accelerator_backend,
        "MHD_HARDWARE_FINGERPRINT": hardware_fingerprint(hw),
        "MHD_MPS_PRECISION": effective,
        "MHD_MPS_MEMORY_FRACTION": str(float(memory_fraction) if memory_fraction is not None else recommended_mps_memory_fraction(hw)),
        "MHD_CPU_THREADS": str(recommended_cpu_threads(hw)),
    }


def accelerator_recommended_working_set_gb(hw: AppleHardwareInfo | None = None) -> float:
    hw = hw or detect_system_hardware()
    if hw.accelerator_backend != "mps":
        return 0.0
    try:
        # Native Metal failures cannot be caught by Python. Diagnostics must
        # not import Torch or initialize Metal in the GUI process.
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(int(torch.mps.recommended_max_memory()) if torch.backends.mps.is_available() else 0)"],
            capture_output=True, text=True, timeout=8, check=False,
        )
        if probe.returncode != 0:
            return 0.0
        value = max(0, int(probe.stdout.strip().splitlines()[-1]))
        return round(value / (1024 ** 3), 2)
    except Exception:
        return 0.0


def diagnostic_snapshot() -> dict[str, Any]:
    hw = detect_system_hardware(refresh=True)
    return {
        **hw.to_dict(),
        "available_memory_gb": current_available_memory_gb(),
        "memory_budget_gb": round(memory_budget_gb(hw), 2),
        "accelerator_recommended_working_set_gb": accelerator_recommended_working_set_gb(hw),
        "recommended": {
            "cpu_threads": recommended_cpu_threads(hw),
            "prepare_workers": recommended_prepare_workers(hw),
            "encode_workers": recommended_encode_workers(hw),
            "generic_batch": recommended_batch(hw),
        },
    }


__all__ = [
    "AppleHardwareInfo", "apply_torch_worker_policy", "autocast_context", "current_available_memory_gb",
    "detect_apple_hardware", "detect_system_hardware", "env_precision", "hardware_fingerprint", "memory_budget_gb",
    "recommended_batch", "recommended_cpu_threads", "recommended_encode_workers", "recommended_mps_memory_fraction",
    "recommended_prepare_workers", "resolve_precision", "runtime_env_updates",
    "accelerator_recommended_working_set_gb", "diagnostic_snapshot",
]
