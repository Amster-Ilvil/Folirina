from __future__ import annotations

"""Qt-free brush reveal primitives for the region composite workbench.

The Japanese/current raster is treated as the top layer and the registered old
Chinese SOURCE as the underlay.  Two independent manual masks are maintained:

* ``transparent``: soft/feathered top-layer transparency;
* ``hole``: hard cut-out of the top layer.

The hole mask always wins where both masks overlap.  Painting one reveal type
clears the other type under the same stroke, so the result is deterministic.
Restore strokes clear both masks.  All writes can optionally be clipped to the
current region selection.
"""

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


BRUSH_MODES = {"transparent", "hole", "restore"}


@dataclass(frozen=True)
class BrushStrokeDiagnostics:
    mode: str
    bbox: tuple[int, int, int, int]
    stroke_pixels: int
    changed_pixels: int
    limited: bool


def _normalise_points(points: Iterable[tuple[int, int] | list[int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for value in points or []:
        if value is None or len(value) < 2:
            continue
        out.append((int(value[0]), int(value[1])))
    return out


def stroke_bbox(
    shape: tuple[int, int],
    points: Iterable[tuple[int, int] | list[int]],
    diameter: int,
    *,
    pad: int = 2,
) -> tuple[int, int, int, int] | None:
    h, w = map(int, shape)
    pts = _normalise_points(points)
    if h <= 0 or w <= 0 or not pts:
        return None
    radius = max(1, int(round(max(1, int(diameter)) / 2.0))) + max(0, int(pad))
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    x0 = max(0, min(xs) - radius); y0 = max(0, min(ys) - radius)
    x1 = min(w, max(xs) + radius + 1); y1 = min(h, max(ys) + radius + 1)
    if x1 <= x0 or y1 <= y0:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def _stroke_alpha(
    shape: tuple[int, int],
    points: list[tuple[int, int]],
    diameter: int,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    radius = max(1, int(round(max(1, int(diameter)) / 2.0)))
    local = [(int(x - x0), int(y - y0)) for x, y in points]
    if len(local) == 1:
        cv2.circle(out, local[0], radius, 255, -1, lineType=cv2.LINE_AA)
    else:
        width = max(1, int(diameter))
        for a, b in zip(local[:-1], local[1:]):
            cv2.line(out, a, b, 255, width, lineType=cv2.LINE_AA)
        cv2.circle(out, local[0], radius, 255, -1, lineType=cv2.LINE_AA)
        cv2.circle(out, local[-1], radius, 255, -1, lineType=cv2.LINE_AA)
    return out


def paint_reveal_stroke_inplace(
    transparent_mask: np.ndarray,
    hole_mask: np.ndarray,
    points: Iterable[tuple[int, int] | list[int]],
    diameter: int,
    mode: str,
    *,
    limit_mask: np.ndarray | None = None,
) -> BrushStrokeDiagnostics:
    """Apply one brush segment to two reveal masks without allocating a page copy."""
    if transparent_mask.shape != hole_mask.shape or transparent_mask.ndim != 2:
        raise ValueError("brush reveal masks must be same-size 2D arrays")
    key = str(mode or "").strip().lower()
    if key not in BRUSH_MODES:
        raise ValueError(f"unknown brush reveal mode: {mode}")
    pts = _normalise_points(points)
    bbox = stroke_bbox(transparent_mask.shape, pts, diameter)
    if bbox is None:
        return BrushStrokeDiagnostics(key, (0, 0, 0, 0), 0, 0, limit_mask is not None)
    x0, y0, x1, y1 = bbox
    stroke = _stroke_alpha(transparent_mask.shape, pts, diameter, bbox)
    if limit_mask is not None:
        if limit_mask.shape != transparent_mask.shape:
            raise ValueError("brush limit mask shape mismatch")
        gate = np.asarray(limit_mask[y0:y1, x0:x1], np.uint8)
        stroke = cv2.bitwise_and(stroke, gate)
    active = stroke > 0
    stroke_pixels = int(np.count_nonzero(active))
    if stroke_pixels <= 0:
        return BrushStrokeDiagnostics(key, bbox, 0, 0, limit_mask is not None)

    tr = transparent_mask[y0:y1, x0:x1]
    ho = hole_mask[y0:y1, x0:x1]
    before_tr = tr[active].copy(); before_ho = ho[active].copy()
    if key == "transparent":
        tr[active] = np.maximum(tr[active], stroke[active])
        ho[active] = 0
    elif key == "hole":
        ho[active] = 255
        tr[active] = 0
    else:  # restore
        tr[active] = 0
        ho[active] = 0
    changed = int(np.count_nonzero((tr[active] != before_tr) | (ho[active] != before_ho)))
    return BrushStrokeDiagnostics(key, bbox, stroke_pixels, changed, limit_mask is not None)


def _inward_feather(mask: np.ndarray, feather_px: int) -> np.ndarray:
    binary = (np.asarray(mask, np.uint8) > 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, np.uint8)
    px = max(0, min(64, int(feather_px)))
    if px <= 0:
        return binary * 255
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    alpha = np.clip(distance / max(1.0, float(px)), 0.0, 1.0)
    alpha[binary == 0] = 0.0
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def reveal_alpha(
    transparent_mask: np.ndarray,
    hole_mask: np.ndarray,
    *,
    transparent_feather_px: int = 3,
    transparent_opacity: float = 1.0,
) -> np.ndarray:
    if transparent_mask.shape != hole_mask.shape:
        raise ValueError("brush reveal masks must have the same shape")
    tr = _inward_feather(transparent_mask, transparent_feather_px).astype(np.float32)
    opacity = float(np.clip(float(transparent_opacity), 0.0, 1.0))
    tr *= opacity
    hole = np.where(np.asarray(hole_mask) > 0, 255.0, 0.0).astype(np.float32)
    return np.maximum(tr, hole).astype(np.uint8)


def compose_reveal_patch(
    base_bgr: np.ndarray,
    underlay_bgr: np.ndarray,
    transparent_mask: np.ndarray,
    hole_mask: np.ndarray,
    *,
    transparent_feather_px: int = 3,
    transparent_opacity: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return live preview BGR and exact replayable BGRA underlay patch."""
    if base_bgr.shape != underlay_bgr.shape or base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("brush reveal base/underlay shape mismatch")
    if transparent_mask.shape != base_bgr.shape[:2] or hole_mask.shape != base_bgr.shape[:2]:
        raise ValueError("brush reveal mask shape mismatch")
    alpha = reveal_alpha(
        transparent_mask, hole_mask,
        transparent_feather_px=transparent_feather_px,
        transparent_opacity=transparent_opacity,
    )
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    preview = np.clip(
        base_bgr.astype(np.float32) * (1.0 - a) + underlay_bgr.astype(np.float32) * a,
        0, 255,
    ).astype(np.uint8)
    patch = np.zeros((base_bgr.shape[0], base_bgr.shape[1], 4), np.uint8)
    patch[:, :, :3] = underlay_bgr
    patch[:, :, 3] = alpha
    return preview, patch


def mask_bbox(*masks: np.ndarray) -> list[int]:
    valid = [np.asarray(mask) > 0 for mask in masks if isinstance(mask, np.ndarray) and mask.ndim == 2]
    if not valid:
        return []
    union = valid[0].copy()
    for item in valid[1:]:
        if item.shape != union.shape:
            raise ValueError("brush reveal mask shape mismatch")
        union |= item
    ys, xs = np.nonzero(union)
    if xs.size == 0:
        return []
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def mask_counts(transparent_mask: np.ndarray, hole_mask: np.ndarray) -> dict[str, int]:
    tr = np.asarray(transparent_mask) > 0
    ho = np.asarray(hole_mask) > 0
    return {
        "transparent_pixels": int(np.count_nonzero(tr & ~ho)),
        "hole_pixels": int(np.count_nonzero(ho)),
        "union_pixels": int(np.count_nonzero(tr | ho)),
    }
