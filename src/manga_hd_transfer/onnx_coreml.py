from __future__ import annotations

"""Safe ONNX Runtime CoreML/ANE provider selection for Apple Silicon.

CoreML is optional.  The caller always supplies CPUExecutionProvider as the
last fallback.  Provider options follow the current ONNX Runtime CoreML API;
when a newer option is unsupported by an older wheel, the helper retries with a
minimal option set instead of failing the OCR pipeline.
"""

from contextlib import contextmanager
import platform
from typing import Any, Iterator


def available_providers(ort) -> list[str]:
    try:
        return [str(x) for x in ort.get_available_providers()]
    except Exception:
        return []


def coreml_compute_units(requested: str = "auto") -> str:
    key = str(requested or "auto").strip().lower()
    if key in {"ane", "neural_engine", "neural-engine"}:
        return "CPUAndNeuralEngine"
    if key in {"gpu", "cpuandgpu"}:
        return "CPUAndGPU"
    if key in {"cpu", "cpuonly"}:
        return "CPUOnly"
    return "ALL"


def coreml_provider_options(*, compute_units: str = "auto", cache_dir: str | None = None,
                            static_shapes: bool = False, profile: bool = False) -> dict[str, str]:
    options = {
        "ModelFormat": "MLProgram",
        "MLComputeUnits": coreml_compute_units(compute_units),
        "RequireStaticInputShapes": "1" if static_shapes else "0",
        "EnableOnSubgraphs": "0",
        "SpecializationStrategy": "FastPrediction",
        "ProfileComputePlan": "1" if profile else "0",
    }
    if cache_dir:
        options["ModelCacheDirectory"] = str(cache_dir)
    return options


@contextmanager
def patch_inference_session_for_coreml(ort, *, compute_units: str = "auto",
                                             cache_dir: str | None = None,
                                             static_shapes: bool = False,
                                             profile: bool = False) -> Iterator[dict[str, Any]]:
    """Temporarily force imported third-party ONNX code to register CoreML first.

    This is intentionally scoped to model construction so application-global
    ONNX Runtime state is restored even when model initialization fails.
    """
    providers_available = available_providers(ort)
    enabled = platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"} and "CoreMLExecutionProvider" in providers_available
    meta: dict[str, Any] = {
        "enabled": bool(enabled),
        "providers_available": providers_available,
        "compute_units": coreml_compute_units(compute_units),
        "provider": "CoreMLExecutionProvider" if enabled else "CPUExecutionProvider",
    }
    if not enabled:
        yield meta
        return

    original = ort.InferenceSession
    options = coreml_provider_options(compute_units=compute_units, cache_dir=cache_dir, static_shapes=static_shapes, profile=profile)
    def _session(path, sess_options=None, providers=None, provider_options=None, **kwargs):
        wanted = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        try:
            return original(path, sess_options=sess_options, providers=wanted, provider_options=[options, {}], **kwargs)
        except TypeError:
            # Older Python bindings may not accept provider_options.
            return original(path, sess_options=sess_options, providers=wanted, **kwargs)
        except Exception:
            # Keep compatibility if a specific CoreML option is unsupported by
            # the installed native runtime.
            minimal = {"MLComputeUnits": coreml_compute_units(compute_units)}
            return original(path, sess_options=sess_options, providers=wanted, provider_options=[minimal, {}], **kwargs)

    ort.InferenceSession = _session
    try:
        yield meta
    finally:
        ort.InferenceSession = original


def probe_coreml(ort) -> dict[str, Any]:
    providers = available_providers(ort)
    return {
        "available": "CoreMLExecutionProvider" in providers,
        "providers": providers,
        "apple_silicon": platform.system() == "Darwin" and platform.machine().lower() in {"arm64", "aarch64"},
    }


__all__ = ["available_providers", "coreml_compute_units", "coreml_provider_options", "patch_inference_session_for_coreml", "probe_coreml"]
