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
from typing import Callable, Iterable

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


def _compact_text_components(candidate: np.ndarray, *, core: np.ndarray | None = None) -> np.ndarray:
    """Conservative raster text evidence, before clipping to the brush.

    Thresholding alone calls book outlines, hatching and solid shadows 'ink'.
    Reject long lines, broad fills and components cut by the context boundary.
    Fixed image-pixel thresholds keep preview tiles and saved patches consistent.
    This is not OCR: ambiguous artwork is deliberately left for explicit tools.
    """
    binary = (np.asarray(candidate) > 0).astype(np.uint8)
    structure = binary if core is None else (np.asarray(core) > 0).astype(np.uint8)
    if structure.shape != binary.shape:
        raise ValueError("text component core shape mismatch")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(count, np.uint8)
    height, width = binary.shape
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        longest, shortest = max(w, h), max(1, min(w, h))
        if area < 2 or longest > 160:
            continue
        if longest > 32 and longest / shortest > 6:
            continue
        if area >= 900 and area / max(1, w * h) >= .70:
            # Dense CJK glyphs can legitimately exceed 70% ink coverage.
            # Keep components with substantial enclosed counterspaces (口/啊,
            # etc.); unlike glyphs, solid shadows have no internal white space.
            # Use strong strokes to read counters: scan blur/AA can close the
            # counters at the permissive 225-level extraction threshold.
            component = ((labels[y:y+h, x:x+w] == label)
                         & (structure[y:y+h, x:x+w] > 0)).astype(np.uint8)
            spaces = cv2.copyMakeBorder(1-component, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=1)
            cv2.floodFill(spaces, None, (0, 0), 0)
            counter_area = int(np.count_nonzero(spaces[1:-1, 1:-1]))
            if counter_area < max(6, int(round(w * h * .02))):
                continue
        if longest >= 12 and (x == 0 or y == 0 or x + w == width or y + h == height):
            continue
        keep[label] = 1
    return keep[labels]


def _source_text_ink_layer(source_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract compact black/grey SOURCE lettering, never coloured foreground.

    The SOURCE may be colour artwork, so a global ``pixel < white`` threshold
    mistakes yellow paper and other saturated fills for ink.  Estimate the
    local paper, identify neutral dark stroke seeds, then retain only compact
    stroke-like components. A one-pixel colour-mixed antialias fringe may be
    recovered around black seeds; coloured lettering has no such authority.
    Recover black-ink coverage relative to the SOURCE paper, never its colour.
    Returns coverage [0..1] and a BGR foreground colour image.
    """
    source = np.asarray(source_bgr, np.uint8)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("SOURCE text layer must be a BGR image")
    height, width = source.shape[:2]
    min_dim = min(height, width)
    kernel = min(31, min_dim if min_dim % 2 else min_dim - 1)
    if kernel >= 3:
        paper_bgr = cv2.medianBlur(source, kernel)
    else:
        paper_bgr = source.copy()
    # Separate neutral scan paper from glyphs before component filtering.
    # An absolute <225 test joins grey paper, lettering and balloon edges into
    # one huge component, which the artwork guard then correctly rejects.
    # Use a local bright-paper envelope instead: morphological closing fills
    # dark strokes (including dense glyph interiors that fool the median) but
    # follows slow scan shading. Like region-local mask refinement in CTD /
    # BallonsTranslator, brightness is evidence relative to the paper, not a
    # global claim that every non-white SOURCE pixel is text. Keep the existing
    # compact-component guard; recover coverage against paper without copying it.
    channel_max = source.max(axis=2)
    channel_min = source.min(axis=2)
    channel_spread = channel_max - channel_min
    neutral = channel_spread <= 24
    if kernel >= 3:
        paper_max = cv2.morphologyEx(
            channel_max, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8)
        )
    else:
        paper_max = channel_max
    paper_contrast = paper_max.astype(np.int16) - channel_max.astype(np.int16)
    # A textured scan can still form bridges at an 8-level local contrast.
    # Follow the upstream region-thresholding principle with a dark Otsu seed,
    # then recover only its immediate antialias fringe. The permissive paper
    # threshold is never allowed to flood the whole grey balloon again.
    otsu_level, _ = cv2.threshold(channel_max, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    neutral_core = neutral & (channel_max <= min(180.0, otsu_level)) & (paper_contrast >= 24)
    # Coloured ink mixed with a differently coloured paper can cross grey in
    # its AA fringe. Such a fringe is not a black glyph seed: the real coloured
    # interior next to it is substantially darker in luminance. Require seeds
    # to be near the local dark interior; AA is recovered only after this test.
    source_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    coloured_luma = np.where(channel_spread > 45, source_gray, 255).astype(np.uint8)
    local_coloured_dark = cv2.erode(coloured_luma, np.ones((5, 5), np.uint8))
    neutral_core &= source_gray.astype(np.int16) <= local_coloured_dark.astype(np.int16) + 16
    seed_neighbourhood = cv2.dilate(neutral_core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    neutral_candidate = (channel_max < 225) & (paper_contrast >= 8) & seed_neighbourhood
    candidate = neutral_candidate
    core = neutral_core
    accepted = _compact_text_components(candidate, core=core)
    # Grey scan halos can join otherwise separate strokes / bottom dots (然)
    # into a dense, counterless candidate. Independently verify the strong
    # components first, then admit only their one-pixel AA fringe. Do not admit
    # the rest of a rejected component or relax the artwork size/fill guards.
    strong_neutral = _compact_text_components(neutral_core)
    strong_fringe = cv2.dilate(strong_neutral, np.ones((3, 3), np.uint8)) > 0
    accepted = accepted | (neutral_candidate & strong_fringe).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (candidate & accepted).astype(np.uint8), connectivity=8
    )
    coverage = np.zeros((height, width), np.float32)
    foreground = np.zeros((height, width, 3), np.float32)
    src_f = source.astype(np.float32)
    paper_f = paper_bgr.astype(np.float32)
    for label in range(1, count):
        x, y, w, h, _area = map(int, stats[label])
        local_labels = labels[y:y+h, x:x+w]
        component = local_labels == label
        core_pixels = component & core[y:y+h, x:x+w]
        if not np.any(core_pixels):
            continue
        pad = max(4, min(12, int(round(max(w, h) * .15))))
        xa, ya = max(0, x-pad), max(0, y-pad)
        xb, yb = min(width, x+w+pad), min(height, y+h+pad)
        context_candidate = candidate[ya:yb, xa:xb]
        paper_samples = src_f[ya:yb, xa:xb][~context_candidate]
        if len(paper_samples) >= 8:
            paper = np.median(paper_samples, axis=0)
        else:
            paper = np.median(paper_f[y:y+h, x:x+w][component], axis=0)
        # Manga scans usually store nominally black ink as dark gray because
        # of print, JPEG and resampling. Treat achromatic lettering as black
        # ink and recover its coverage, otherwise the whole Chinese layer is
        # pasted in its scan-gray tone and looks faded on saturated TARGET.
        # Transparent reveal is deliberately black-ink-only. Hole reveal below
        # remains the explicit tool for copying SOURCE colour/background/art.
        ink_colour = np.zeros(3, np.float32)
        direction = paper - ink_colour
        denom = float(np.dot(direction, direction))
        if denom < 64.0:
            continue
        ys_local, xs_local = np.nonzero(component)
        ys, xs = ys_local + y, xs_local + x
        observed = paper - src_f[ys, xs]
        alpha = np.clip((observed @ direction) / denom, 0.0, 1.0)
        coverage[ys, xs] = alpha
        foreground[ys, xs] = ink_colour
    return coverage, foreground


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
    inpaint_callback: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict]] | None = None,
    target_text_mask: np.ndarray | None = None,
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
    # Transparent strokes reveal only the SOURCE lettering layer, not SOURCE
    # paper/art or coloured lettering. Recover only black/grey foreground and
    # its antialias coverage relative to the local SOURCE paper.
    # Hole strokes deliberately retain their exact full-underlay semantics.
    ink, source_foreground = _source_text_ink_layer(underlay_bgr)
    transparent_alpha = alpha.astype(np.float32) * ink
    support = (transparent_mask > 0) & (hole_mask == 0)
    if verified_white_mask is not None and verified_white_mask.shape != support.shape:
        raise ValueError("brush white-bubble authority shape mismatch")
    white_support = support & (verified_white_mask > 0) if verified_white_mask is not None else np.zeros_like(support)
    reference = base_bgr if target_bgr is None else target_bgr
    if reference.shape != base_bgr.shape:
        raise ValueError("brush TARGET reference shape mismatch")
    from .outlined_text_mask import outlined_target_text_mask
    if target_text_mask is not None and target_text_mask.shape != support.shape:
        raise ValueError('brush TARGET text evidence mask shape mismatch')
    outlined = outlined_target_text_mask(reference, scope=support.astype(np.uint8), text_seed=target_text_mask)
    if target_text_mask is not None:
        # Restore the complete detector footprint, including kana/furigana and
        # white rims joined to white paper. Never promote its bounding box.
        semantic=cv2.dilate((target_text_mask>0).astype(np.uint8),np.ones((7,7),np.uint8))
        semantic[~support]=0
        outlined|=semantic*255
    outlined[~support] = 0
    from .coloured_paper_repair import repair_coloured_paper_leaks
    working, paper_repair, _paper_diag = repair_coloured_paper_leaks(
        base_bgr, reference, support.astype(np.uint8), protected_ink=(ink > 0).astype(np.uint8),
    )
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
    target_candidates = (ink_contrast >= 12) | neutral_faint_ink
    target_core = (ink_contrast >= 30) | (neutral_faint_ink & (ref_gray <= 180))
    target_text = _compact_text_components(target_candidates, core=target_core) > 0
    # Original TARGET locates JP ink, but is not the current background. Never
    # re-inpaint already removed JP or mix SOURCE AA against the old lettering.
    remaining_target = np.max(np.abs(working.astype(np.int16)-reference.astype(np.int16)), axis=2) <= 24
    base_gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    base_max, base_min = working.max(axis=2), working.min(axis=2)
    base_background = cv2.morphologyEx(base_gray, cv2.MORPH_CLOSE, np.ones((31,31),np.uint8))
    base_paper = cv2.morphologyEx(base_max, cv2.MORPH_CLOSE, np.ones((31,31),np.uint8))
    # Coloured paper often has mild scan/compression texture. Require stronger
    # contrast there; faint neutral gray glyphs retain their separate paper test.
    current_ink = ((base_background.astype(np.int16)-base_gray.astype(np.int16) >= 24)
                   | ((base_max.astype(np.int16)-base_min.astype(np.int16) <= 25)
                      & (base_paper.astype(np.int16)-base_max.astype(np.int16) >= 8)))
    clear = (target_text & support & remaining_target & current_ink).astype(np.uint8) * 255
    # A manual transparent brush explicitly authorises TARGET ink cleanup.
    # The old SOURCE-ink prerequisite made a Japanese glyph return unchanged
    # when the paired Chinese area happened to be blank or offset.
    white_clear = np.zeros_like(support)
    if np.any(clear):
        clear = cv2.dilate(clear, np.ones((5, 5), np.uint8))
        clear[~support] = 0
        clear[~remaining_target] = 0
        # The AA halo must not erase a rejected structural line next to text.
        clear[target_candidates & ~target_text] = 0
        white_clear = (clear > 0) & white_support
        clear[white_support] = 0
    # Original white outlines are part of the Japanese glyph, not SOURCE paper
    # leaks or immutable illustration highlights. Remove their complete proven
    # footprint before revealing Chinese, including cores already erased white.
    clear[outlined > 0] = 255
    if np.any(outlined) and inpaint_callback is not None:
        # Model conditioning must use pristine TARGET, never partially erased
        # white silhouettes or Chinese pixels from the current result.
        repaired, _diagnostics = inpaint_callback(reference.copy(), clear.copy())
        if not isinstance(repaired, np.ndarray) or repaired.shape != reference.shape or repaired.dtype != np.uint8:
            raise ValueError("background repair must return matching uint8 BGR")
        cleaned = working.copy()
        cleaned[clear > 0] = repaired[clear > 0]
    elif np.any(clear):
        # Telea on the entire 256-320px preview tile is needlessly expensive
        # for a 20px Japanese stroke. Work on the tight dirty ROI plus enough
        # context for the 3px inpainting neighbourhood. This also makes each
        # brush stroke independent of the page's total pixel count.
        xx,yy,ww,hh=cv2.boundingRect(cv2.findNonZero(clear))
        pad=12
        xa=max(0,xx-pad); ya=max(0,yy-pad)
        xb=min(reference.shape[1],xx+ww+pad); yb=min(reference.shape[0],yy+hh+pad)
        cleaned=working.copy()
        roi = working[ya:yb,xa:xb]
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
        cleaned = working.copy()
    # White paper owns only confirmed glyph pixels and their clipped AA halo,
    # never the entire stroke/semantic polygon (which may include artwork).
    if np.any(white_clear):
        cleaned[white_clear] = 255
        clear[white_clear] = 255
    # The clean background owns erased pixels fully, but SOURCE still obeys
    # the requested opacity and feather even where it overlaps erased ink.
    # For pixels already made opaque by JP cleanup, fold brush feather into
    # the source ink coverage. Else the PNG patch itself is feathered at
    # publish time, so keep its inner glyph coverage unfeathered to avoid the
    # former opacity-squared/faint-edge bug.
    brush_alpha = alpha.astype(np.float32) / 255.0
    ink_a = (ink * np.where(clear > 0, brush_alpha, 1.0))[:, :, None]
    replacement = np.clip(
        cleaned.astype(np.float32) * (1.0 - ink_a)
        + source_foreground * ink_a,
        0, 255,
    ).astype(np.uint8)
    # Publication covers cleanup as well as ink, while preserving stroke
    # feather/opacity. Never use SOURCE white as the fill.
    # Erasure must complete before source opacity/feather is applied. Mixing
    # the original Japanese back into a cleaned pixel leaves ghost strokes.
    publish_alpha = np.where(clear > 0, 255, np.where(transparent_alpha > 0, alpha, 0))
    publish_alpha = np.where(paper_repair > 0,255,publish_alpha)
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
        white_damage=support & (base_bgr.min(axis=2)>=250) & coloured_target & (outlined == 0)
        if np.any(white_damage):
            repaired = np.clip(
                reference.astype(np.float32) * (1.0 - ink[:, :, None])
                + source_foreground * ink[:, :, None],
                0, 255,
            ).astype(np.uint8)
            replacement[white_damage] = repaired[white_damage]
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
