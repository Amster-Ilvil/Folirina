from __future__ import annotations

"""Optional MLX capability probe and small safe runtime helpers.

MLX is deliberately optional: Folirina's production vision path remains
PyTorch/MPS and Core ML/ONNX.  This module only exposes capability detection and
an isolated micro-benchmark hook so MLX can be evaluated without replacing a
working model backend.
"""

from typing import Any


def status() -> dict[str, Any]:
    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:
        return {
            "available": False,
            "version": "",
            "device": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    version = getattr(mx, "__version__", "")
    try:
        device = str(mx.default_device())
    except Exception:
        device = ""
    return {
        "available": True,
        "version": str(version),
        "device": device,
        "error": "",
    }


def benchmark_matmul(size: int = 512, repeat: int = 3) -> dict[str, Any]:
    """Run a tiny MLX matmul only when MLX is installed.

    This is intentionally not used by the production image pipeline.  It is a
    capability/smoke test for Apple-native MLX, with no persistent model state.
    """
    import time

    try:
        import mlx.core as mx  # type: ignore
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    n = max(64, min(4096, int(size)))
    reps = max(1, min(20, int(repeat)))
    a = mx.random.normal((n, n), dtype=mx.float16)
    b = mx.random.normal((n, n), dtype=mx.float16)
    mx.eval(a, b)
    times: list[float] = []
    for _ in range(reps):
        start = time.perf_counter()
        c = a @ b
        mx.eval(c)
        times.append(time.perf_counter() - start)
    return {
        "available": True,
        "size": n,
        "repeat": reps,
        "seconds": times,
        "mean_seconds": sum(times) / len(times),
        "device": str(mx.default_device()),
        "version": str(getattr(mx, "__version__", "")),
    }


__all__ = ["status", "benchmark_matmul"]
