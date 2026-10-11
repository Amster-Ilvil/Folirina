from __future__ import annotations

"""Cache-aware OCR primitives used by page orchestration.

The service contains no transfer-mode logic.  It owns backend construction,
full-page OCR caching and source rectification only.
"""

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cache import PageStageCache, image_stage_signature
from .geometry import transform_points, transform_to_homography
from .ocr import OCRBackend, NullOCRBackend, RetryingOCRBackend, build_backend

logger = logging.getLogger(__name__)


def _set_ocr_stats(
    stats: dict[str, Any],
    role: str,
    *,
    route: str,
    route_entered: bool,
    backend: OCRBackend,
    model_invoked: bool,
    evidence_available: bool,
) -> None:
    """Record truthful OCR execution semantics alongside the legacy route string.

    ``route_entered`` means the mode reached an OCR decision point.  It is not
    equivalent to invoking a recognition model: a missing/Null backend is a
    valid route outcome and must never be reported as OCR usage.  Cache hits
    consume OCR evidence without invoking the model in the current run.
    """
    stats[f"ocr_{role}"] = str(route)
    stats[f"ocr_{role}_route_entered"] = bool(route_entered)
    stats[f"ocr_{role}_backend"] = type(backend).__name__
    stats[f"ocr_{role}_backend_available"] = not isinstance(backend, NullOCRBackend)
    stats[f"ocr_{role}_model_invoked"] = bool(model_invoked)
    stats[f"ocr_{role}_evidence_available"] = bool(evidence_available)


def build_ocr_backend_soft(
    ocr_config: Any,
    lang: str,
    backend_name: str,
    *,
    role: str,
    soft_failures: list[str] | None = None,
) -> OCRBackend:
    """Build one OCR backend while preserving the project's optional-backend policy."""
    try:
        backend = build_backend(ocr_config, lang, backend_name, role=role)
    except (RuntimeError, ImportError, ModuleNotFoundError, ValueError) as exc:
        if not bool(getattr(ocr_config, "soft_fail_missing_backend", True)):
            raise
        message = f"OCR backend {backend_name!r} unavailable: {exc}"
        if soft_failures is not None:
            soft_failures.append(message)
        fallback_name = str(getattr(ocr_config, "soft_fail_fallback_backend", "tesseract") or "").strip().lower()
        if fallback_name and fallback_name not in {"none", backend_name.lower().strip()}:
            try:
                backend = build_backend(ocr_config, lang, fallback_name, role=role)
                fallback_msg = f"{message}; fallback -> {fallback_name!r}"
                logger.warning("%s", fallback_msg)
                if soft_failures is not None:
                    soft_failures.append(fallback_msg)
                return backend
            except (RuntimeError, ImportError, ModuleNotFoundError, ValueError) as fallback_exc:
                fallback_message = f"OCR fallback {fallback_name!r} unavailable: {fallback_exc}"
                logger.warning("%s; continuing without OCR evidence", fallback_message)
                if soft_failures is not None:
                    soft_failures.append(fallback_message)
        else:
            logger.warning("%s; continuing without OCR evidence", message)
        return NullOCRBackend()

    if (
        role == "source"
        and bool(getattr(ocr_config, "retry_low_confidence", False))
        and bool(getattr(backend, "retry_crops", False))
        and not isinstance(backend, NullOCRBackend)
    ):
        return RetryingOCRBackend(
            backend,
            float(getattr(ocr_config, "retry_confidence", 0.0)),
            float(getattr(ocr_config, "retry_scale", 1.0)),
        )
    return backend


def recognize_cached(
    role: str,
    backend: OCRBackend,
    image: np.ndarray,
    image_path: str | Path,
    *,
    ocr_config: Any,
    cache: PageStageCache,
    cache_enabled: bool,
    stats: dict[str, Any],
) -> list:
    # NullOCR is an explicit no-evidence route.  Do not load/save empty cache
    # entries under it: that would let a later receipt claim a cache hit even
    # though no recognizer has ever evaluated the image.
    if isinstance(backend, NullOCRBackend):
        _set_ocr_stats(
            stats, role, route="null_backend", route_entered=True, backend=backend,
            model_invoked=False, evidence_available=False,
        )
        return []

    sig = image_stage_signature(
        image_path,
        ocr_config,
        {
            "role": role,
            "backend": type(backend).__name__,
            "lang": ocr_config.source_lang if role == "source" else ocr_config.target_lang,
        },
    )
    if cache_enabled:
        hit = cache.load_blocks(role, sig)
        if hit is not None:
            _set_ocr_stats(
                stats, role, route="hit", route_entered=True, backend=backend,
                model_invoked=False, evidence_available=True,
            )
            return hit
    blocks = backend.recognize(image, image_path=image_path)
    if cache_enabled:
        cache.save_blocks(role, sig, blocks)
    _set_ocr_stats(
        stats, role, route="miss", route_entered=True, backend=backend,
        model_invoked=True, evidence_available=True,
    )
    return blocks


def should_rectify_source(ocr_config: Any, backend: OCRBackend, registration: Any) -> bool:
    return bool(
        getattr(ocr_config, "rectify_source_with_registration", False)
        and bool(getattr(backend, "supports_rectified_input", True))
        and float(registration.confidence) >= float(getattr(ocr_config, "rectify_min_registration_confidence", 0.0))
    )


def recognize_source_rectified_cached(
    backend: OCRBackend,
    source: np.ndarray,
    source_path: str | Path,
    target_shape: tuple[int, int],
    registration: Any,
    *,
    ocr_config: Any,
    cache: PageStageCache,
    cache_enabled: bool,
    stats: dict[str, Any],
) -> list:
    if isinstance(backend, NullOCRBackend):
        _set_ocr_stats(
            stats, "source", route="null_backend", route_entered=True, backend=backend,
            model_invoked=False, evidence_available=False,
        )
        return []

    H = transform_to_homography(registration.matrix)
    th, tw = target_shape
    rect_scale = 1.0
    if bool(getattr(ocr_config, "rectify_preserve_source_resolution", False)):
        sh, sw = source.shape[:2]
        density_scale = min(sw / max(1, tw), sh / max(1, th))
        rect_scale = float(np.clip(density_scale, 1.0, float(getattr(ocr_config, "rectify_max_scale", 1.0))))
        long_side = max(tw * rect_scale, th * rect_scale)
        max_long_side = float(getattr(ocr_config, "rectify_max_long_side", long_side))
        if long_side > max_long_side:
            rect_scale *= max_long_side / long_side
            rect_scale = max(1.0, rect_scale)

    S = np.array(
        [[rect_scale, 0.0, 0.0], [0.0, rect_scale, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    H_ocr = S @ H
    sig = image_stage_signature(
        source_path,
        ocr_config,
        {
            "role": "source",
            "backend": type(backend).__name__,
            "lang": ocr_config.source_lang,
            "rectified": True,
            "target_shape": list(target_shape),
            "rectified_scale": round(rect_scale, 4),
            "registration": np.round(H, 6).tolist(),
        },
    )
    if cache_enabled:
        hit = cache.load_blocks("source", sig)
        if hit is not None:
            _set_ocr_stats(
                stats, "source", route="hit_rectified", route_entered=True, backend=backend,
                model_invoked=False, evidence_available=True,
            )
            return hit

    rectified = cv2.warpPerspective(
        source,
        H_ocr,
        (max(1, round(tw * rect_scale)), max(1, round(th * rect_scale))),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    blocks = backend.recognize(rectified, image_path=None)
    try:
        inv = np.linalg.inv(H_ocr)
    except np.linalg.LinAlgError:
        inv = np.eye(3, dtype=np.float64)
    for block in blocks:
        block.polygon = transform_points(block.polygon, inv)
        block.meta["ocr_rectified_source"] = True
        block.meta["registration_confidence"] = float(registration.confidence)
        block.meta["ocr_rectified_scale"] = float(rect_scale)
    if cache_enabled:
        cache.save_blocks("source", sig, blocks)
    _set_ocr_stats(
        stats, "source", route="miss_rectified", route_entered=True, backend=backend,
        model_invoked=True, evidence_available=True,
    )
    return blocks


__all__ = [
    "build_ocr_backend_soft",
    "recognize_cached",
    "recognize_source_rectified_cached",
    "should_rectify_source",
]
