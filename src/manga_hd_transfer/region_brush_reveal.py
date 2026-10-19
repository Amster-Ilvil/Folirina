from __future__ import annotations

"""Qt-free brush reveal primitives for the region composite workbench.

The Japanese/current raster is treated as the top layer and the registered old
Chinese SOURCE as the underlay.  Two independent manual masks are maintained:

* ``transparent``: soft/feathered top-layer transparency;
* ``hole``: hard cut-out of the top layer.

The hole mask always wins where both masks overlap.  Painting one reveal type
clears the other type under the same stroke, so the result is deterministic.
Restore strokes clear both reveal masks and, when a restore mask is supplied,
publish original TARGET pixels over the committed result. All writes can
optionally be clipped to the current region selection.
"""

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


BRUSH_MODES = {"transparent", "hole", "restore"}


def label_verified_white_bubbles(verified_white_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Label disconnected, already-verified white balloon interiors once per editor."""
    mask = np.asarray(verified_white_mask, np.uint8)
    if mask.ndim != 2:
        raise ValueError("verified white-bubble mask must be 2D")
    _count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), connectivity=8
    )
    return labels, stats


def touched_verified_white_bubble_labels(
    transparent_mask: np.ndarray,
    hole_mask: np.ndarray,
    labels: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> list[int]:
    """Return white-bubble components reached by a newly painted brush segment."""
    if transparent_mask.shape != hole_mask.shape or labels.shape != transparent_mask.shape:
        raise ValueError("brush and white-bubble label masks must have matching shapes")
    x0, y0, x1, y1 = [int(v) for v in bbox]
    h, w = transparent_mask.shape
    x0, x1 = max(0, min(w, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h, y0)), max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return []
    local_labels = labels[y0:y1, x0:x1]
    painted = ((transparent_mask[y0:y1, x0:x1] > 0)
               | (hole_mask[y0:y1, x0:x1] > 0))
    touched = np.unique(local_labels[painted & (local_labels > 0)])
    return [int(value) for value in touched]


def expand_to_verified_white_bubbles_inplace(
    transparent_mask: np.ndarray,
    hole_mask: np.ndarray,
    labels: np.ndarray,
    stats: np.ndarray,
    component_labels: list[int],
) -> list[tuple[int, int, int, int]]:
    """Make each touched white balloon a complete transparent SOURCE reveal.

    A white speech balloon is one replacement unit: a short brush stroke must
    not leave Japanese elsewhere inside that same verified balloon. Coloured
    regions are absent from ``labels`` and therefore cannot be expanded here.
    """
    if transparent_mask.shape != hole_mask.shape or labels.shape != transparent_mask.shape:
        raise ValueError("brush and white-bubble label masks must have matching shapes")
    boxes: list[tuple[int, int, int, int]] = []
    h, w = transparent_mask.shape
    for label in sorted({int(value) for value in component_labels if int(value) > 0}):
        if label >= len(stats):
            continue
        x, y, bw, bh = [int(v) for v in stats[label, :4]]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w, x + bw), min(h, y + bh)
        if x1 <= x0 or y1 <= y0:
            continue
        local = labels[y0:y1, x0:x1] == label
        transparent = transparent_mask[y0:y1, x0:x1]
        hole = hole_mask[y0:y1, x0:x1]
        transparent[local] = 255
        hole[local] = 0
        boxes.append((x0, y0, x1, y1))
    return boxes


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
    restore_mask: np.ndarray | None = None,
) -> BrushStrokeDiagnostics:
    """Apply one brush segment to two reveal masks without allocating a page copy."""
    if transparent_mask.shape != hole_mask.shape or transparent_mask.ndim != 2:
        raise ValueError("brush reveal masks must be same-size 2D arrays")
    if restore_mask is not None and restore_mask.shape != transparent_mask.shape:
        raise ValueError("brush restore mask shape mismatch")
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
    restore = restore_mask[y0:y1, x0:x1] if restore_mask is not None else None
    before_restore = restore[active].copy() if restore is not None else None
    if key == "transparent":
        tr[active] = np.maximum(tr[active], stroke[active])
        ho[active] = 0
    elif key == "hole":
        ho[active] = 255
        tr[active] = 0
    else:  # restore
        tr[active] = 0
        ho[active] = 0
    changed_mask = (tr[active] != before_tr) | (ho[active] != before_ho)
    if restore is not None:
        restore[active] = 255 if key == "restore" else 0
        changed_mask |= restore[active] != before_restore
    changed = int(np.count_nonzero(changed_mask))
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
    verified_white_mask: np.ndarray | None = None,
    restore_mask: np.ndarray | None = None,
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
    if verified_white_mask is not None and verified_white_mask.shape != support.shape:
        raise ValueError("brush white-bubble authority shape mismatch")
    white_support = support & (verified_white_mask > 0) if verified_white_mask is not None else np.zeros_like(support)
    reference = base_bgr if target_bgr is None else target_bgr
    if reference.shape != base_bgr.shape:
        raise ValueError("brush TARGET reference shape mismatch")
    ref_max = reference.max(axis=2)
    ref_min = reference.min(axis=2)
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    local_background = cv2.morphologyEx(ref_gray, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    ink_contrast = local_background.astype(np.int16) - ref_gray.astype(np.int16)
    local_paper = cv2.morphologyEx(ref_max, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    neutral_faint_ink = ((ref_max < 245)
                        & ((ref_max.astype(np.int16) - ref_min.astype(np.int16)) <= 25)
                        & (local_paper.astype(np.int16) - ref_max.astype(np.int16) >= 8))
    # Absolute darkness classified an entire dark blue/purple background as
    # Japanese ink. Local luminance contrast also catches red/coloured glyphs
    # while preserving a flat coloured background under the stroke.
    clear = (((ink_contrast >= 12) | neutral_faint_ink) & support).astype(np.uint8) * 255
    # A manual transparent brush explicitly authorises TARGET ink cleanup.
    # The old SOURCE-ink prerequisite made a Japanese glyph return unchanged
    # when the paired Chinese area happened to be blank or offset.
    if np.any(clear):
        clear = cv2.dilate(clear, np.ones((5, 5), np.uint8))
        clear[~support] = 0
        clear[white_support] = 0
    if np.any(clear):
        # Telea on the entire 256-320px preview tile is needlessly expensive
        # for a 20px Japanese stroke. Work on the tight dirty ROI plus enough
        # context for the 3px inpainting neighbourhood. This also makes each
        # brush stroke independent of the page's total pixel count.
        xx,yy,ww,hh=cv2.boundingRect(cv2.findNonZero(clear))
        pad=12
        xa=max(0,xx-pad); ya=max(0,yy-pad)
        xb=min(reference.shape[1],xx+ww+pad); yb=min(reference.shape[0],yy+hh+pad)
        cleaned=reference.copy()
        roi = reference[ya:yb,xa:xb]
        local_clear = clear[ya:yb,xa:xb]
        samples = roi[local_clear == 0]
        paper = np.median(samples, axis=0) if len(samples) >= 64 else None
        flat = False
        if paper is not None:
            deviations = np.max(np.abs(samples.astype(np.float32)-paper), axis=1)
            flat = bool(np.percentile(deviations, 95) <= 10)
        # Reconstruct proven flat paper exactly. Telea on a large connected
        # glyph mask can pull black strokes into triangular smear artifacts.
        if flat:
            tile = roi.copy()
            tile[local_clear > 0] = np.rint(paper).astype(np.uint8)
        else:
            tile=cv2.inpaint(roi,local_clear,3,cv2.INPAINT_TELEA)
        local=clear[ya:yb,xa:xb]>0
        cleaned_roi=cleaned[ya:yb,xa:xb]
        cleaned_roi[local]=tile[local]
    else:
        clear[:] = 0
        cleaned = reference.copy()
    # White balloon paper is deterministic: erase the painted TARGET area
    # completely before showing SOURCE ink. A faint glyph is never sampled back
    # into the fill, and tile boundaries share the same page-level authority.
    if np.any(white_support):
        cleaned[white_support] = 255
        clear[white_support] = 255
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
    # Restore owns original TARGET pixels even when base already contains a
    # committed Chinese result. Clearing transient reveal masks cannot do this.
    if restore_mask is not None:
        if restore_mask.shape != alpha.shape:
            raise ValueError("brush restore mask shape mismatch")
        if target_bgr is None and np.any(restore_mask):
            raise ValueError("brush restore requires original TARGET")
        restored = restore_mask > 0
        replacement[restored] = reference[restored]
        alpha[restored] = 255
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


def mask_counts(transparent_mask: np.ndarray, hole_mask: np.ndarray, restore_mask: np.ndarray | None = None) -> dict[str, int]:
    tr = np.asarray(transparent_mask) > 0
    ho = np.asarray(hole_mask) > 0
    counts = {
        "transparent_pixels": int(np.count_nonzero(tr & ~ho)),
        "hole_pixels": int(np.count_nonzero(ho)),
        "union_pixels": int(np.count_nonzero(tr | ho)),
    }
    if restore_mask is not None:
        restored = np.asarray(restore_mask) > 0
        if restored.shape != tr.shape:
            raise ValueError("brush restore mask shape mismatch")
        counts["restore_pixels"] = int(np.count_nonzero(restored))
        counts["union_pixels"] = int(np.count_nonzero(tr | ho | restored))
    return counts
