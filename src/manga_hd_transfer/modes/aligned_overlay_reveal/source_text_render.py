from __future__ import annotations

"""Mode-private SOURCE glyph rendering for aligned-overlay reveal.

This helper intentionally mirrors the shared historical raster math without
importing ``text_only_transfer``.  The aligned renderer owns this capsule so
Koharu-driven reveal can remain architecture-isolated from other modes.
"""

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)


def source_text_render(source: np.ndarray, source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build SOURCE-derived glyph opacity plus grayscale ink tone."""
    sg = _gray(source).astype(np.float32)
    use = source_mask > 0
    alpha = np.zeros(source_mask.shape, np.float32)
    if not np.any(use):
        return alpha, sg

    dil = cv2.dilate(source_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    ring = dil & ~use
    if np.count_nonzero(ring) >= 16:
        paper = float(np.percentile(sg[ring], 70.0))
    else:
        paper = float(np.percentile(sg, 82.0))
    core_vals = sg[use]
    core = float(np.percentile(core_vals, 15.0)) if core_vals.size else 0.0
    denom = max(32.0, paper - core)
    raw = np.clip((paper - sg) / denom, 0.0, 1.0)

    dist = cv2.distanceTransform(use.astype(np.uint8), cv2.DIST_L2, 3)
    strong = use & ((dist >= 1.05) | (raw >= 0.58) | (sg <= (paper - max(18.0, denom * 0.34))))
    near = cv2.dilate(
        strong.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1
    ) > 0
    fringe = use & (~strong) & near & (raw >= 0.06)

    alpha[fringe] = np.clip(raw[fringe] * 0.65, 0.0, 0.28)
    alpha[strong] = np.clip(np.maximum(raw[strong], 0.72), 0.0, 1.0)
    alpha[use & (~near) & (~strong)] = 0.0
    return np.clip(alpha, 0.0, 1.0), sg


__all__ = ["source_text_render"]
