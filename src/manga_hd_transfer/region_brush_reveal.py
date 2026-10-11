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
    target_bgr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Clean TARGET ink in the stroke, then reveal SOURCE ink over its colour.

    The saved patch includes both cleaned background pixels and SOURCE ink so
    replay has exactly the same erase-before-reveal order as the live preview.
    """
    if base_bgr.shape != underlay_bgr.shape or base_bgr.ndim != 3 or base_bgr.shape[2] != 3:
        raise ValueError("brush reveal base/underlay shape mismatch")
    if transparent_mask.shape != base_bgr.shape[:2] or hole_mask.shape != base_bgr.shape[:2]:
        raise ValueError("brush reveal mask shape mismatch")
    alpha = reveal_alpha(
        transparent_mask, hole_mask,
        transparent_feather_px=transparent_feather_px,
        transparent_opacity=transparent_opacity,
    )
    # Transparent strokes reveal neutral dark ink, not SOURCE paper or colour.
    # Hole strokes deliberately retain their exact full-underlay semantics.
    channels = underlay_bgr.astype(np.float32)
    darkest = channels.min(axis=2)
    brightest = channels.max(axis=2)
    ink = np.clip((225.0 - brightest) / 45.0, 0.0, 1.0)
    ink *= (brightest - darkest <= 45.0)
    transparent_alpha = alpha.astype(np.float32) * ink
    support = (transparent_mask > 0) & (hole_mask == 0)
    reference = base_bgr if target_bgr is None else target_bgr
    if reference.shape != base_bgr.shape:
        raise ValueError("brush TARGET reference shape mismatch")
    ref_max = reference.max(axis=2)
    ref_min = reference.min(axis=2)
    local_paper = cv2.morphologyEx(ref_max, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    neutral_faint_ink = ((ref_max < 245)
                        & ((ref_max.astype(np.int16) - ref_min.astype(np.int16)) <= 25)
                        & (local_paper.astype(np.int16) - ref_max.astype(np.int16) >= 8))
    clear = (((ref_max < 210) | neutral_faint_ink) & support).astype(np.uint8) * 255
    # A manual transparent brush explicitly authorises TARGET ink cleanup.
    # The old SOURCE-ink prerequisite made a Japanese glyph return unchanged
    # when the paired Chinese area happened to be blank or offset.
    if np.any(clear):
        clear = cv2.dilate(clear, np.ones((5, 5), np.uint8))
        clear[~support] = 0
        # Telea on the entire 256-320px preview tile is needlessly expensive
        # for a 20px Japanese stroke. Work on the tight dirty ROI plus enough
        # context for the 3px inpainting neighbourhood. This also makes each
        # brush stroke independent of the page's total pixel count.
        xx,yy,ww,hh=cv2.boundingRect(cv2.findNonZero(clear))
        pad=12
        xa=max(0,xx-pad); ya=max(0,yy-pad)
        xb=min(reference.shape[1],xx+ww+pad); yb=min(reference.shape[0],yy+hh+pad)
        cleaned=reference.copy()
        tile=cv2.inpaint(reference[ya:yb,xa:xb],clear[ya:yb,xa:xb],3,cv2.INPAINT_TELEA)
        local=clear[ya:yb,xa:xb]>0
        cleaned_roi=cleaned[ya:yb,xa:xb]
        cleaned_roi[local]=tile[local]
    else:
        clear[:] = 0
        cleaned = reference
    # The clean background owns erased pixels fully, but SOURCE still obeys
    # the requested opacity and feather even where it overlaps erased ink.
    ink_a = (ink * np.where(clear > 0, alpha.astype(np.float32) / 255.0, 1.0))[:, :, None]
    replacement = np.clip(cleaned.astype(np.float32) * (1.0 - ink_a) + channels * ink_a, 0, 255).astype(np.uint8)
    # Publication covers cleanup as well as ink, while preserving stroke
    # feather/opacity. Never use SOURCE white as the fill.
    # Erasure must complete before source opacity/feather is applied. Mixing
    # the original Japanese back into a cleaned pixel leaves ghost strokes.
    publish_alpha = np.where(clear > 0, 255, np.where(transparent_alpha > 0, alpha, 0))
    # Do not publish the *entire* stroke from the original TARGET. In a review
    # session base may already contain accepted Chinese/art edits: publishing
    # untouched TARGET pixels over them causes the infamous "Japanese comes
    # back after a brush stroke" regression. Only cleaned or added ink owns
    # pixels in the transparent stroke; a hole is an explicit full reveal.
    if target_bgr is not None:
        # Exception: a previously bleached WHITE patch on coloured TARGET is
        # an obvious non-canonical artifact that should be restored. Existing
        # dark Chinese glyphs remain protected from TARGET overwrites.
        # A white pixel over a *black TARGET glyph* is usually a previously
        # accepted JP cleanup, NOT damage. Restoring TARGET there would visibly
        # resurrect Japanese on the next stroke. Only restore white damage on
        # clearly coloured (non-ink) TARGET backgrounds.
        ref_min=reference.min(axis=2); ref_max=reference.max(axis=2)
        coloured_target=(ref_max>=210)&(ref_min>=25)&((ref_max.astype(np.int16)-ref_min.astype(np.int16))>=30)
        white_damage=support & (base_bgr.min(axis=2)>=250) & coloured_target
        publish_alpha = np.where(white_damage, alpha, publish_alpha)
    replacement = np.where((hole_mask > 0)[:, :, None], underlay_bgr, replacement)
    alpha = np.where(hole_mask > 0, alpha, publish_alpha).astype(np.uint8)
    a = alpha.astype(np.float32)[:, :, None] / 255.0
    preview = np.clip(
        base_bgr.astype(np.float32) * (1.0 - a) + replacement.astype(np.float32) * a,
        0, 255,
    ).astype(np.uint8)
    patch = np.zeros((base_bgr.shape[0], base_bgr.shape[1], 4), np.uint8)
    patch[:, :, :3] = replacement
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
