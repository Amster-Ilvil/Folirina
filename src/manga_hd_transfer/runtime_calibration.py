from __future__ import annotations

"""Persistent, hardware-scoped runtime calibration.

Only non-semantic throughput/resource observations are stored. OCR text, images,
model outputs and page content are never persisted by this module.
"""

import json
import os
import platform
from pathlib import Path
import threading
import time
from typing import Any

_LOCK = threading.RLock()
_MAX_ENGINES = 16


def _root() -> Path:
    override = os.environ.get("MHD_RUNTIME_CALIBRATION_HOME", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if platform.system() == "Darwin":
        return home / "Library" / "Caches" / "Folirina" / "runtime-calibration"
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        return (Path(base) if base else home / "AppData" / "Local") / "Folirina" / "runtime-calibration"
    xdg = os.environ.get("XDG_CACHE_HOME", "").strip()
    return (Path(xdg).expanduser() if xdg else home / ".cache") / "folirina" / "runtime-calibration"


def _payload(hw: Any | None = None) -> dict[str, Any]:
    if hw is None:
        from .accelerator_policy import detect_system_hardware
        hw = detect_system_hardware()
    return {
        "platform": hw.platform,
        "architecture": hw.architecture,
        "generation": hw.generation,
        "chip": hw.chip,
        "cpu_cores": hw.cpu_cores,
        "memory_gb": hw.memory_gb,
        "accelerator_backend": hw.accelerator_backend,
        "accelerator_name": hw.accelerator_name,
        "accelerator_memory_gb": hw.accelerator_memory_gb,
        "onnx_providers": list(hw.onnx_providers),
    }


def hardware_fingerprint(hw: Any | None = None) -> str:
    from .accelerator_policy import hardware_fingerprint as _hardware_fingerprint
    return _hardware_fingerprint(hw)


def _path(hw: Any | None = None) -> Path:
    return _root() / f"{hardware_fingerprint(hw)}.json"


def load(hw: Any | None = None) -> dict[str, Any]:
    try:
        with _LOCK:
            data = json.loads(_path(hw).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def record(engine: str, *, batch_size: int, elapsed_seconds: float, items: int,
           available_memory_gb: float = 0.0, success: bool = True,
           resource_error: bool = False) -> None:
    name = str(engine or "runtime").strip().lower() or "runtime"
    batch = max(1, int(batch_size))
    elapsed = max(0.001, float(elapsed_seconds))
    items = max(1, int(items))
    throughput = items / elapsed
    try:
        available = max(0.0, float(available_memory_gb))
    except Exception:
        available = 0.0
    with _LOCK:
        from .accelerator_policy import detect_system_hardware
        hw = detect_system_hardware()
        data = load(hw)
        data.setdefault("schema", "folirina.runtime_calibration.v1")
        data.setdefault("hardware", _payload(hw))
        engines = data.setdefault("engines", {})
        if not isinstance(engines, dict):
            engines = {}
            data["engines"] = engines
        if len(engines) >= _MAX_ENGINES and name not in engines:
            oldest = sorted(engines.items(), key=lambda kv: float(kv[1].get("last_seen", 0.0)))[:1]
            if oldest:
                engines.pop(oldest[0][0], None)
        state = engines.setdefault(name, {"samples": 0, "best_throughput": 0.0})
        state["samples"] = min(10000, int(state.get("samples", 0)) + 1)
        state["last_seen"] = time.time()
        state["last_batch"] = batch
        state["last_available_memory_gb"] = round(available, 3)
        state["last_success"] = bool(success)
        if success:
            current_best = float(state.get("best_throughput", 0.0) or 0.0)
            if not state.get("best_batch") or throughput > current_best * 1.03:
                state["best_batch"] = batch
                state["best_throughput"] = throughput
            state["safe_batch_ceiling"] = max(batch, int(state.get("safe_batch_ceiling", 0) or 0))
        if resource_error:
            backoff = max(1, batch // 2)
            old = int(state.get("safe_batch_ceiling", 0) or 0)
            state["safe_batch_ceiling"] = backoff if old == 0 else min(old, backoff)
            state["resource_backoffs"] = min(1000, int(state.get("resource_backoffs", 0)) + 1)
        path = _path(hw)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)


def safe_batch_ceiling(engine: str) -> int | None:
    try:
        state = load().get("engines", {}).get(str(engine or "runtime").strip().lower(), {})
        value = int(state.get("safe_batch_ceiling", 0) or 0)
        return value if value > 0 else None
    except Exception:
        return None


def best_batch(engine: str) -> int | None:
    try:
        state = load().get("engines", {}).get(str(engine or "runtime").strip().lower(), {})
        value = int(state.get("best_batch", 0) or 0)
        return value if value > 0 else None
    except Exception:
        return None


def clear() -> None:
    try:
        _path().unlink(missing_ok=True)
    except Exception:
        pass


__all__ = ["hardware_fingerprint", "load", "record", "safe_batch_ceiling", "best_batch", "clear"]
