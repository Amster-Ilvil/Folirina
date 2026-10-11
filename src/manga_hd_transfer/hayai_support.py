from __future__ import annotations

"""Small, dependency-free helpers for Folirina's Hayai OCR integration."""

import json
from pathlib import Path
from typing import Any

HAYAI_PACKAGE_VERSION = "2.3.0"
HAYAI_DEFAULT_MODEL_ID = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
HAYAI_LEGACY_V2_MODEL_ID = "JustANormalTinkerer/hayai-ocr-v2"
HAYAI_LITERT_MODEL_ID = "JustANormalTinkerer/hayai-ocr-v2-tflite"
HAYAI_PATCH_BUDGETS = (256, 384, 512)
HAYAI_DEFAULT_PATCH_BUDGET = 384
HAYAI_READY_MARKER = "folirina.hayai.ready.json"


def normalize_hayai_backend(value: str | None) -> str:
    backend = str(value or "auto").strip().lower()
    return backend if backend in {"auto", "torch", "litert"} else "auto"


def normalize_hayai_quantize(value: str | None) -> str:
    quant = str(value or "none").strip().lower()
    return quant if quant in {"none", "int8", "int4"} else "none"


def normalize_hayai_patch_budget(value: int | str | None, *, backend: str | None = None) -> int:
    b = normalize_hayai_backend(backend)
    if b == "litert":
        # Upstream LiteRT exports have a fixed 256-patch graph.
        return 256
    try:
        parsed = int(value) if value is not None else HAYAI_DEFAULT_PATCH_BUDGET
    except (TypeError, ValueError):
        parsed = HAYAI_DEFAULT_PATCH_BUDGET
    if parsed in HAYAI_PATCH_BUDGETS:
        return parsed
    # Keep old configs safe and deterministic instead of passing arbitrary values.
    return min(HAYAI_PATCH_BUDGETS, key=lambda x: abs(x - parsed))


def hayai_model_family(*, use_v2: bool, backend: str | None) -> str:
    b = normalize_hayai_backend(backend)
    if b == "litert":
        return "v2-litert"
    return "v2" if bool(use_v2) else "v2.5-nova"


def hayai_marker_path(cache_dir: str | Path) -> Path:
    return Path(cache_dir).expanduser() / HAYAI_READY_MARKER


def hayai_marker_payload(*, cache_dir: str | Path, runtime_python: str | None, quantize: str | None, use_v2: bool, backend: str | None, max_num_patches: int | str | None) -> dict[str, Any]:
    b = normalize_hayai_backend(backend)
    return {
        "engine": "hayai-ocr",
        "package_version": HAYAI_PACKAGE_VERSION,
        "model_family": hayai_model_family(use_v2=use_v2, backend=b),
        "model_id": HAYAI_LITERT_MODEL_ID if b == "litert" else (HAYAI_LEGACY_V2_MODEL_ID if use_v2 else HAYAI_DEFAULT_MODEL_ID),
        "runtime_python": runtime_python,
        "cache_dir": str(Path(cache_dir).expanduser().resolve()),
        "quantize": normalize_hayai_quantize(quantize),
        "use_v2": bool(use_v2),
        "backend": b,
        "max_num_patches": normalize_hayai_patch_budget(max_num_patches, backend=b),
    }


def hayai_ready_marker_matches(cache_dir: str | Path, *, quantize: str | None, use_v2: bool, backend: str | None, max_num_patches: int | str | None) -> tuple[bool, str]:
    marker = hayai_marker_path(cache_dir)
    if not marker.is_file():
        return False, f"missing {HAYAI_READY_MARKER}"
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"invalid Hayai ready marker: {exc}"
    expected = hayai_marker_payload(
        cache_dir=cache_dir, runtime_python=payload.get("runtime_python"),
        quantize=quantize, use_v2=use_v2, backend=backend, max_num_patches=max_num_patches,
    )
    for key in ("engine", "package_version", "model_family", "model_id", "quantize", "use_v2", "backend", "max_num_patches"):
        if payload.get(key) != expected.get(key):
            return False, f"Hayai ready marker mismatch: {key}={payload.get(key)!r}, expected {expected.get(key)!r}"
    return True, "ready"
