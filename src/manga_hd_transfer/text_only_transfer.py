from __future__ import annotations

"""Compatibility facade for text-only transfer helpers.

The existing implementation lives in :mod:`text_only_transfer_core` unchanged.
This facade re-exports its complete API (including historically imported private
helpers) and supplies the white-container faint-line cleanup helper that is
already required by the repository regression suite.
"""

from typing import Any

import cv2
import numpy as np

from . import text_only_transfer_core as _core

# Preserve the historical module surface exactly, including single-underscore
# helpers that other internal modules/tests may import directly.
for _name in dir(_core):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_core, _name)


def cleanup_white_container_line_artifacts(
    image: np.ndarray,
    target: np.ndarray,
    paper_mask: np.ndarray,
    source_text_mask: np.ndarray,
    *,
    min_difference: int = 10,
    min_span_px: int = 12,
    min_aspect: float = 3.0,
    source_protect_px: int = 4,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Restore unsupported faint line/rule artifacts to TARGET paper pixels.

    This is deliberately narrower than generic inpainting.  A candidate must:
    - be inside a trusted white/neutral paper mask;
    - differ meaningfully from the original TARGET pixel;
    - have no nearby SOURCE Chinese text support; and
    - form a thin/elongated connected component.

    Accepted pixels are copied *from TARGET* byte-for-byte, so the helper cannot
    invent background colour or erase legitimate SOURCE-supported punctuation.
    """
    if image.shape != target.shape:
        raise ValueError("line-artifact cleanup image/target shapes must match")
    if paper_mask.shape != image.shape[:2] or source_text_mask.shape != paper_mask.shape:
        raise ValueError("line-artifact cleanup masks must match image canvas")

    out = image.copy()
    removed = np.zeros(paper_mask.shape, dtype=np.uint8)
    paper = paper_mask > 0
    if not np.any(paper):
        return out, removed, {
            "white_line_artifacts_removed": 0,
            "white_line_artifact_components": 0,
            "target_background_authority": True,
        }

    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target.astype(np.uint8)
    if image.ndim == 3:
        hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        neutral_target = hsv[..., 1] <= 48
    else:
        neutral_target = np.ones_like(paper, dtype=bool)

    protect_radius = max(0, int(source_protect_px))
    if protect_radius > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (protect_radius * 2 + 1, protect_radius * 2 + 1),
        )
        source_near = cv2.dilate((source_text_mask > 0).astype(np.uint8), kernel) > 0
    else:
        source_near = source_text_mask > 0

    # Only look for newly-darkened residuals on bright TARGET paper. Existing
    # TARGET borders/rules have little/no image-vs-target delta and are excluded.
    delta = target_gray.astype(np.int16) - image_gray.astype(np.int16)
    candidate = (
        paper
        & neutral_target
        & (target_gray >= 210)
        & (delta >= max(1, int(min_difference)))
        & (~source_near)
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate.astype(np.uint8), 8)
    kept_components = 0
    for label in range(1, count):
        x, y, width, height, area = [int(v) for v in stats[label]]
        if area <= 0:
            continue
        long_side = max(width, height)
        short_side = max(1, min(width, height))
        aspect = float(long_side / short_side)
        # Admit one-pixel/AA rules and longer thin remnants; reject compact glyphs.
        line_like = (
            (long_side >= max(3, int(min_span_px)) and aspect >= float(min_aspect))
            or (long_side >= max(8, int(min_span_px) // 2) and short_side <= 2 and area >= 6)
        )
        if not line_like:
            continue
        component = labels == label
        if np.any(component & source_near):
            continue
        removed[component] = 255
        kept_components += 1

    if cv2.countNonZero(removed) > 0:
        sel = removed > 0
        out[sel] = target[sel]

    return out, removed, {
        "white_line_artifacts_removed": int(cv2.countNonZero(removed)),
        "white_line_artifact_components": int(kept_components),
        "source_text_protected": True,
        "target_background_authority": True,
    }


# Include every historical core name plus the compatibility helper for callers
# that inspect __all__, without exposing facade bookkeeping names.
__all__ = sorted({
    name for name in dir(_core) if not name.startswith("__")
} | {"cleanup_white_container_line_artifacts"})
