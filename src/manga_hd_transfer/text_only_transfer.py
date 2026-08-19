from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)


def _adaptive_ink_threshold(gray: np.ndarray, use: np.ndarray, *, floor: int = 125, ceiling: int = 218) -> int:
    vals = gray[use]
    if vals.size == 0:
        return 185
    # Estimate the local paper/artwork level and look for substantially darker
    # strokes.  This works on white speech bubbles as well as coloured artwork.
    bg = float(np.percentile(vals, 72.0))
    spread = float(np.percentile(vals, 85.0) - np.percentile(vals, 25.0))
    margin = max(24.0, min(58.0, 24.0 + spread * 0.18))
    return int(np.clip(bg - margin, floor, ceiling))


def _component_text_selector(
    ink: np.ndarray,
    unique: np.ndarray,
    region: np.ndarray,
    *,
    min_area: int = 2,
    min_unique_pixels: int = 2,
    min_unique_ratio: float = 0.055,
    max_component_fraction: float = 0.10,
) -> np.ndarray:
    """Select changed compact ink components while rejecting common artwork.

    A component is retained when enough of its pixels are edition-exclusive.
    Common panel lines, hair outlines and bubble borders normally have a nearby
    counterpart in both editions and therefore fail this gate.  The gate is
    deliberately based on *structure*, never SOURCE colour/background pixels.
    """
    binary = ((ink > 0) & (region > 0)).astype(np.uint8)
    uniq = (unique > 0) & (region > 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    region_area = max(1, int(np.count_nonzero(region)))
    h, w = binary.shape
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area < int(min_area):
            continue
        comp = labels == i
        up = int(np.count_nonzero(comp & uniq))
        ratio = float(up / max(1, area))
        area_fraction = float(area / region_area)
        span_x = float(bw / max(1, w))
        span_y = float(bh / max(1, h))
        fill = float(area / max(1, bw * bh))
        # Large/long common structures are almost always panel art or a balloon
        # outline.  Keep a truly changed large glyph only when its uniqueness is
        # overwhelming and it is not a thin line.
        if area_fraction > float(max_component_fraction):
            if ratio < 0.42 or fill < 0.16:
                continue
        if (span_x > 0.82 or span_y > 0.82) and fill < 0.16 and ratio < 0.55:
            continue
        needed = max(int(min_unique_pixels), int(round(area * float(min_unique_ratio))))
        if up < needed:
            continue
        out[comp] = 1
    return out * 255



def _compact_text_clusters(mask: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Keep locally dense groups of changed ink, not scattered artwork edges."""
    src = (mask > 0).astype(np.uint8)
    use = region_mask > 0
    if int(np.count_nonzero(src)) == 0:
        return mask
    ys, xs = np.where(use)
    if xs.size == 0:
        return np.zeros_like(mask)
    rw = max(1, int(xs.max() - xs.min() + 1)); rh = max(1, int(ys.max() - ys.min() + 1))
    gap = max(4, min(14, int(round(min(rw, rh) * 0.035))))
    dil = cv2.dilate(src, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dil, 8)
    out = np.zeros_like(src)
    for i in range(1, count):
        x, y, bw, bh, _ = [int(v) for v in stats[i]]
        group = labels == i
        ink = int(np.count_nonzero(src & group))
        if ink < 6:
            continue
        density = float(ink / max(1, bw * bh))
        span = max(float(bw / rw), float(bh / rh))
        # Text columns/rows are locally dense. Sparse groups spanning most of a
        # panel are typically hair, clothing seams or frame lines.
        if density < 0.018 and (bw * bh) > 420:
            continue
        if span > 0.78 and density < 0.055:
            continue
        out[src.astype(bool) & group] = 1
    out[~use] = 0
    return out * 255




def prune_source_text_mask(mask: np.ndarray, region_mask: np.ndarray, *,
                           anchor_min_area: int = 12, tiny_max_area: int = 24,
                           max_dim: int = 12, support_radius: int = 20) -> np.ndarray:
    """Remove isolated SOURCE specks while preserving nearby punctuation.

    Earlier versions could admit tiny non-text fragments from scan noise or
    bubble-edge differences. This pass keeps real text anchors plus punctuation
    spatially consistent with the dominant text column/row, while dropping
    isolated components that merely happen to sit near the broader support halo.
    """
    binary = ((mask > 0) & (region_mask > 0)).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return np.zeros_like(mask)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    comps: list[tuple[int,int,int,int,int,int]] = []
    out = np.zeros_like(binary)
    anchors = np.zeros_like(binary)
    region_area = max(1, int(np.count_nonzero(region_mask)))
    dyn_anchor = max(int(anchor_min_area), min(28, max(10, int(round(region_area * 0.00035)))))
    for i in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[i]]
        if area <= 0:
            continue
        comps.append((i, x, y, bw, bh, area))
        if area >= dyn_anchor and max(bw, bh) <= max(44, int(round(max(mask.shape) * 0.08))):
            anchors[labels == i] = 1
    if int(np.count_nonzero(anchors)) == 0 and comps:
        max_area = max(a for _i, _x, _y, _bw, _bh, a in comps)
        for i, _x, _y, _bw, _bh, area in comps:
            if area >= max(2, int(round(max_area * 0.6))):
                anchors[labels == i] = 1
    if int(np.count_nonzero(anchors)) == 0:
        return mask.copy()
    ay, ax = np.where(anchors > 0)
    ax0, ax1 = int(ax.min()), int(ax.max())
    ay0, ay1 = int(ay.min()), int(ay.max())
    aw = max(1, ax1 - ax0 + 1); ah = max(1, ay1 - ay0 + 1)
    vertical = ah >= int(round(aw * 1.20))
    horizontal = aw >= int(round(ah * 1.20))
    rad = max(6, int(support_radius))
    support = cv2.dilate(anchors, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rad * 2 + 1, rad * 2 + 1))) > 0
    for i, x, y, bw, bh, area in comps:
        comp = labels == i
        if area >= dyn_anchor:
            out[comp] = 1
            continue
        if area > int(tiny_max_area) or max(bw, bh) > int(max_dim):
            continue
        if not np.any(comp & support):
            continue
        cx = x + bw / 2.0; cy = y + bh / 2.0
        near_text = False
        if vertical:
            near_text = (ax0 - 18 <= cx <= ax1 + 18) and (y <= ay1 + 28 and (y + bh) >= ay0 - 28)
        elif horizontal:
            near_text = (ay0 - 18 <= cy <= ay1 + 18) and (x <= ax1 + 28 and (x + bw) >= ax0 - 28)
        else:
            near_text = (x <= ax1 + 22 and (x + bw) >= ax0 - 22 and y <= ay1 + 22 and (y + bh) >= ay0 - 22)
        if near_text:
            out[comp] = 1
    if cv2.countNonZero(out) == 0:
        return mask.copy()
    out[region_mask == 0] = 0
    return out * 255

def _relax_source_text_mask(mask: np.ndarray, source: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Recover thin SOURCE glyph fragments and slightly widen the final mask.

    Whole-page alignment mode should not lose top ellipsis dots or antialias
    fringes merely because the strict changed-ink gate clipped them away.  This
    helper only adds dark SOURCE pixels that stay close to the existing text lane
    and then performs a tiny directional dilation inside the trusted region.
    """
    base = ((mask > 0) & (region_mask > 0)).astype(np.uint8) * 255
    if cv2.countNonZero(base) <= 0:
        return base
    use = region_mask > 0
    sg = _gray(source)
    sth = _adaptive_ink_threshold(sg, use)
    all_ink = ((sg < sth) & use).astype(np.uint8) * 255

    ys, xs = np.where(base > 0)
    if xs.size == 0:
        return base
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = max(1, x1 - x0 + 1); bh = max(1, y1 - y0 + 1)
    vertical = bh >= int(round(bw * 1.15))
    horizontal = bw >= int(round(bh * 1.15))

    support_r = max(5, min(14, int(round(min(max(bw, bh), 120) * 0.08))))
    support = cv2.dilate((base > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_r * 2 + 1, support_r * 2 + 1)), iterations=1)
    corridor = (support > 0) & use

    # Admit very small/delicate components near the current text lane, especially
    # for vertical ellipsis / punctuation hovering above the main text body.
    extra = np.zeros_like(base)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(((all_ink > 0) & corridor).astype(np.uint8), 8)
    lane_pad_x = max(8, min(20, int(round(bw * 0.28))))
    lane_pad_y = max(10, min(26, int(round(bh * 0.22))))
    lane_x0, lane_x1 = x0 - lane_pad_x, x1 + lane_pad_x
    lane_y0, lane_y1 = y0 - lane_pad_y, y1 + lane_pad_y
    for lab in range(1, n):
        cx, cy, cw, ch, area = [int(v) for v in stats[lab]]
        if area <= 0:
            continue
        comp = labels == lab
        if area > max(42, int(round((bw * bh) * 0.06))):
            continue
        touch = int(np.count_nonzero(comp & (support > 0)))
        if touch <= 0:
            continue
        comp_cx = cx + cw * 0.5
        comp_cy = cy + ch * 0.5
        near_lane = (lane_x0 <= comp_cx <= lane_x1) and (lane_y0 <= comp_cy <= lane_y1)
        if vertical:
            near_lane = near_lane and (x0 - 18 <= comp_cx <= x1 + 18)
        elif horizontal:
            near_lane = near_lane and (y0 - 18 <= comp_cy <= y1 + 18)
        if not near_lane:
            continue
        # Preserve very thin pieces; they are usually punctuation / antialias.
        if area <= 16 or min(cw, ch) <= 2 or max(cw, ch) <= 14:
            extra[comp] = 255
            continue
        if float(area / max(1, cw * ch)) >= 0.20:
            extra[comp] = 255

    out = cv2.bitwise_or(base, extra)
    if vertical:
        k_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    elif horizontal:
        k_dir = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    else:
        k_dir = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = cv2.dilate(out, k_dir, iterations=1)
    out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    out[region_mask == 0] = 0
    # Keep the relaxed mask inside a modest corridor to avoid text bleeding into artwork.
    guard = cv2.dilate((base > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    out[guard == 0] = 0
    return out


def _keep_target_components_near_source(target_mask: np.ndarray, source_mask: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Only erase TARGET ink spatially associated with SOURCE translated text."""
    if cv2.countNonZero(target_mask) == 0 or cv2.countNonZero(source_mask) == 0:
        return target_mask
    ys, xs = np.where(region_mask > 0)
    rw = max(1, int(xs.max() - xs.min() + 1)) if xs.size else target_mask.shape[1]
    rh = max(1, int(ys.max() - ys.min() + 1)) if ys.size else target_mask.shape[0]
    radius = max(10, min(44, int(round(max(rw, rh) * 0.10))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    near = cv2.dilate((source_mask > 0).astype(np.uint8), k) > 0
    # Clip to the spatial association window instead of keeping an entire
    # connected component.  On coloured art a Japanese glyph can touch a hair or
    # effect line; retaining the whole connected component would erase that art.
    out = ((target_mask > 0) & near & (region_mask > 0)).astype(np.uint8) * 255
    return out

def changed_text_masks(
    source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    *,
    tolerance_px: int = 2,
    min_unique_ratio: float = 0.055,
    max_component_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return SOURCE Chinese and TARGET Japanese ink masks only.

    This routine is intentionally background-blind.  It never returns a paper,
    skin, clothing or coloured-fill mask.  SOURCE/TARGET structure that exists in
    both editions is treated as artwork and excluded from transfer.
    """
    if source.shape[:2] != target.shape[:2] or region_mask.shape != source.shape[:2]:
        raise ValueError("text-only transfer inputs must share the same canvas")
    use = region_mask > 0
    empty = np.zeros(region_mask.shape, np.uint8)
    if not np.any(use):
        return empty.copy(), empty.copy(), {"reason": "empty_region"}

    sg = _gray(source)
    tg = _gray(target)
    sth = _adaptive_ink_threshold(sg, use)
    tth = _adaptive_ink_threshold(tg, use)
    s_ink = ((sg < sth) & use).astype(np.uint8) * 255
    t_ink = ((tg < tth) & use).astype(np.uint8) * 255

    tol = max(1, int(tolerance_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    s_near = cv2.dilate(s_ink, k)
    t_near = cv2.dilate(t_ink, k)
    s_unique = cv2.bitwise_and(s_ink, cv2.bitwise_not(t_near))
    t_unique = cv2.bitwise_and(t_ink, cv2.bitwise_not(s_near))

    s_sel = _component_text_selector(
        s_ink, s_unique, region_mask,
        min_unique_ratio=min_unique_ratio,
        max_component_fraction=max_component_fraction,
    )
    t_sel = _component_text_selector(
        t_ink, t_unique, region_mask,
        min_unique_ratio=min_unique_ratio,
        max_component_fraction=max_component_fraction,
    )

    # If the strict component gate finds only edges of a changed glyph, admit
    # connected ink touching those seeds.  This fills the glyph interior without
    # allowing an arbitrary rectangular SOURCE patch to leak through.
    def grow_from_seed(all_ink: np.ndarray, selected: np.ndarray) -> np.ndarray:
        if cv2.countNonZero(selected) == 0:
            return selected
        count, labels, _stats, _ = cv2.connectedComponentsWithStats((all_ink > 0).astype(np.uint8), 8)
        keep = np.zeros_like(all_ink)
        touched = set(int(x) for x in np.unique(labels[selected > 0]) if int(x) > 0)
        for lab in touched:
            keep[labels == lab] = 255
        keep[~use] = 0
        return keep

    s_sel = grow_from_seed(s_ink, s_sel)
    t_sel = grow_from_seed(t_ink, t_sel)
    s_sel = _compact_text_clusters(s_sel, region_mask)
    s_sel = prune_source_text_mask(s_sel, region_mask)
    s_sel = _relax_source_text_mask(s_sel, source, region_mask)
    t_sel = _compact_text_clusters(t_sel, region_mask)
    t_sel = _keep_target_components_near_source(t_sel, s_sel, region_mask)
    return s_sel, t_sel, {
        "source_threshold": int(sth),
        "target_threshold": int(tth),
        "source_text_pixels": int(cv2.countNonZero(s_sel)),
        "target_text_pixels": int(cv2.countNonZero(t_sel)),
        "region_pixels": int(np.count_nonzero(use)),
        "contract": "text_only_no_source_background",
    }


def source_text_render(source: np.ndarray, source_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build SOURCE-derived glyph opacity plus grayscale ink tone.

    The final published text should visually follow the *Chinese source page*,
    not an artificial solid-black silhouette.  We therefore keep the broader
    mask for coverage, but the actual rendering colour/tone comes from SOURCE
    grayscale and the opacity comes from SOURCE darkness relative to its local
    paper.  This preserves anti-aliased edges without the gray/black halo caused
    by forcing a minimum opacity over the entire relaxed mask.
    """
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
    near = cv2.dilate(strong.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1) > 0
    fringe = use & (~strong) & near & (raw >= 0.06)

    alpha[fringe] = np.clip(raw[fringe] * 0.65, 0.0, 0.28)
    alpha[strong] = np.clip(np.maximum(raw[strong], 0.72), 0.0, 1.0)
    alpha[use & (~near) & (~strong)] = 0.0
    return np.clip(alpha, 0.0, 1.0), sg


def source_text_alpha(source: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """Backward-compatible opacity accessor."""
    alpha, _ = source_text_render(source, source_mask)
    return alpha



def _small_interior_target_components(ink: np.ndarray, region_mask: np.ndarray, *, max_area: int = 48, max_dim: int = 10) -> np.ndarray:
    """Keep tiny TARGET punctuation/noise inside a trusted white container.

    The normal compact-cluster gate intentionally drops isolated 2-5px dots.
    Those are exactly the remnants that show up after replacement when the old
    Japanese text contains ellipsis / dakuten-sized pieces.  Bubble outlines are
    protected by excluding components that touch the region boundary ring.
    """
    binary = ((ink > 0) & (region_mask > 0)).astype(np.uint8)
    if int(np.count_nonzero(binary)) == 0:
        return np.zeros_like(region_mask)
    use = (region_mask > 0).astype(np.uint8)
    eroded = cv2.erode(use, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    boundary_ring = (use > 0) & (eroded == 0)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > int(max_area) or max(bw, bh) > int(max_dim):
            continue
        comp = labels == lab
        if np.any(comp & boundary_ring):
            continue
        out[comp] = 1
    return out * 255


def target_text_mask_in_container(target: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    """Select compact TARGET lettering inside a trusted container interior."""
    use = region_mask > 0
    if not np.any(use):
        return np.zeros(region_mask.shape, np.uint8)
    tg = _gray(target)
    tth = _adaptive_ink_threshold(tg, use)
    t_all = ((tg < tth) & use).astype(np.uint8) * 255
    out = _component_text_selector(
        t_all, t_all, region_mask,
        min_unique_pixels=1, min_unique_ratio=0.0, max_component_fraction=0.07,
    )
    return _compact_text_clusters(out, region_mask)





def clear_broad_neutral_paper_components(
    rendered: np.ndarray,
    target: np.ndarray,
    clear_mask: np.ndarray,
    *,
    min_paper_ratio: float = 0.62,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Handle broad OCR clear regions whose TARGET background is proven paper.

    OCR/reletter clear masks can be *whole text-box rectangles* rather than only
    Japanese glyph strokes. Passing those rectangles to Telea/AI inpainting lets
    nearby panel lines and halftone leak into an otherwise white box, producing
    the grey triangular shadows seen on p-005.  For a connected clear component
    that is already mostly bright/neutral TARGET paper, the background does not
    need to be invented at all:

      * mark the whole broad component as handled so no interpolating inpaint sees it;
      * derive compact TARGET text ink inside that proven paper component;
      * restore only those ink/fringe pixels to the component's own TARGET paper colour;
      * preserve all other original TARGET pixels, including scan texture and borders.

    This helper is intentionally generic but callers gate it to OCR modes only.
    ``handled`` is the broad area removed from later inpaint; ``changed`` is the
    much smaller set of pixels that were actually rewritten to paper colour.
    """
    if rendered.shape != target.shape or clear_mask.shape != target.shape[:2]:
        raise ValueError("paper-clear inputs must share one TARGET canvas")
    out = rendered.copy()
    handled = np.zeros(clear_mask.shape, np.uint8)
    changed = np.zeros(clear_mask.shape, np.uint8)
    work = (np.asarray(clear_mask) > 0).astype(np.uint8)
    if int(np.count_nonzero(work)) == 0:
        return out, handled, changed, {
            "broad_paper_components": 0,
            "broad_paper_handled_pixels": 0,
            "broad_paper_changed_pixels": 0,
            "broad_paper_rejected_components": 0,
            "broad_paper_min_ratio": float(min_paper_ratio),
        }

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    accepted_rows: list[dict[str, Any]] = []
    rejected = 0
    for lab in range(1, count):
        comp = labels == lab
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area < 12:
            rejected += 1
            continue
        paper_sel = comp & (gray >= 205) & (hsv[..., 1] <= 55)
        paper_pixels = int(np.count_nonzero(paper_sel))
        paper_ratio = float(paper_pixels / max(1, area))
        if paper_pixels < 20 or paper_ratio < float(min_paper_ratio):
            rejected += 1
            continue
        paper_gray = gray[paper_sel]
        if paper_gray.size == 0 or float(np.median(paper_gray)) < 220.0:
            rejected += 1
            continue

        comp_u8 = comp.astype(np.uint8) * 255
        # Reuse the existing compact-ink selector: unlike a rectangular clear
        # mask, it rejects long panel/bubble outlines and keeps the actual JP
        # lettering. A one-pixel expansion absorbs antialias fringes.
        glyph = target_text_mask_in_container(target, comp_u8)
        if cv2.countNonZero(glyph) > 0:
            glyph = cv2.dilate(
                glyph,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                iterations=1,
            )
            glyph = cv2.bitwise_and(glyph, comp_u8)

        paper = np.median(target[paper_sel], axis=0).astype(np.uint8)
        if cv2.countNonZero(glyph) > 0:
            out[glyph > 0] = paper
            changed[glyph > 0] = 255
        # Crucial: even unchanged white pixels in the broad rectangle are
        # removed from downstream inpainting. They are already valid TARGET.
        handled[comp] = 255
        x, y, bw, bh = [int(v) for v in stats[lab, :4]]
        accepted_rows.append({
            "bbox": [x, y, x + bw, y + bh],
            "area": area,
            "paper_ratio": round(paper_ratio, 4),
            "paper_bgr": [int(x) for x in paper.tolist()],
            "glyph_clear_pixels": int(cv2.countNonZero(glyph)),
        })

    return out, handled, changed, {
        "broad_paper_components": len(accepted_rows),
        "broad_paper_handled_pixels": int(cv2.countNonZero(handled)),
        "broad_paper_changed_pixels": int(cv2.countNonZero(changed)),
        "broad_paper_rejected_components": int(rejected),
        "broad_paper_min_ratio": float(min_paper_ratio),
        "broad_paper_regions": accepted_rows,
    }


def clear_to_target_paper(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Fill a cleared white-container region with target paper colour, not Telea.

    This restores the v1.0.6 behaviour for ordinary white speech bubbles while
    still keeping TARGET as the only background source.
    """
    out = rendered.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    sel = mask > 0
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    bright = sel & (gray >= 205)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    out[sel] = paper
    return out



def clear_text_components_to_local_paper(
    rendered: np.ndarray,
    target: np.ndarray,
    clear_mask: np.ndarray,
    region_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Clear only components surrounded by neutral bright TARGET paper.

    Used for large/localized white candidates that may include人物/artwork.  A
    component is cleared only when its own local ring proves that it actually
    sits on white/neutral paper.  This prevents a broad candidate from bleaching
    skin, purple effects, halftone or hair while avoiding Telea speck artifacts.
    """
    out=rendered.copy(); accepted=np.zeros_like(clear_mask)
    binary=((clear_mask>0)&(region_mask>0)).astype(np.uint8)
    if int(np.count_nonzero(binary))==0:
        return out,accepted,{"local_paper_components":0,"local_paper_clear_pixels":0,"local_paper_rejected_components":0}
    gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY); hsv=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    kept=rejected=0
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
    for lab in range(1,count):
        comp=labels==lab; area=int(stats[lab,cv2.CC_STAT_AREA])
        if area<=0: continue
        ring=cv2.dilate(comp.astype(np.uint8),k,iterations=1)>0
        ring &= ~comp; ring &= (region_mask>0)
        good=ring & (gray>=205) & (hsv[...,1]<=55)
        if int(np.count_nonzero(good)) < max(10,min(80,int(round(area*0.22)))):
            rejected+=1; continue
        paper=np.median(target[good],axis=0).astype(np.uint8)
        out[comp]=paper; accepted[comp]=255; kept+=1
    return out,accepted,{"local_paper_components":int(kept),"local_paper_clear_pixels":int(cv2.countNonZero(accepted)),"local_paper_rejected_components":int(rejected)}

def cleanup_target_residual_specks(
    image: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    source_text_mask: np.ndarray,
    clear_mask: np.ndarray,
    *,
    white_container: bool = False,
    inpaint_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove isolated TARGET-only ink residue without touching SOURCE punctuation.

    Candidates are connected components from the *original TARGET* dark ink, not
    arbitrary dark pixels in the rendered result. A component is removable only
    when it is small, has no nearby SOURCE Chinese support, and still leaves dark
    pixels after the primary clear. Long/large balloon borders therefore survive
    even when they run along the edge of the region, while old Japanese punctuation
    dots and antialias fragments are cleaned.
    """
    if image.shape != target.shape or region_mask.shape != image.shape[:2]:
        raise ValueError("residual-speck cleanup inputs must share the same canvas")
    if source_text_mask.shape != region_mask.shape or clear_mask.shape != region_mask.shape:
        raise ValueError("residual-speck cleanup masks must share the same canvas")
    use = region_mask > 0
    if int(np.count_nonzero(use)) == 0:
        return image.copy(), np.zeros_like(region_mask), {"residual_specks_removed": 0}
    # White-container masks are intentionally inset to protect outlines. Old
    # Japanese punctuation can sit just outside that strict write envelope, so
    # search a small halo too. Removal still requires a tiny TARGET component on
    # neutral bright paper; coloured artwork/hair therefore remains protected.
    if white_container:
        search_use = cv2.dilate(use.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1) > 0
    else:
        search_use = use.copy()

    # Any legitimate SOURCE glyph or punctuation gets a generous protection halo.
    protect_r = 3
    k_src = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (protect_r * 2 + 1, protect_r * 2 + 1))
    source_near = cv2.dilate((source_text_mask > 0).astype(np.uint8), k_src) > 0

    tg = _gray(target)
    og = _gray(image)
    tthr = _adaptive_ink_threshold(tg, use)
    target_dark = (tg < min(205, max(135, tthr + 18))) & search_use & (~source_near)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) if target.ndim == 3 else None
    output_dark = og < min(200, max(130, tthr + 14))

    near_r = 6 if white_container else 3
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (near_r * 2 + 1, near_r * 2 + 1))
    near_clear = cv2.dilate((clear_mask > 0).astype(np.uint8), k_near) > 0

    count, labels, stats, _ = cv2.connectedComponentsWithStats(target_dark.astype(np.uint8), 8)
    clean = np.zeros_like(region_mask)
    max_area = 160 if white_container else 18
    max_dim = 30 if white_container else 8
    kept = 0
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area <= 0 or area > max_area or max(bw, bh) > max_dim:
            # Balloon/panel borders are large or long and are protected here.
            continue
        comp = labels == lab
        residual = comp & output_dark
        if not np.any(residual):
            continue
        if not white_container and not np.any(comp & near_clear):
            continue
        if white_container:
            # Require neutral bright paper around the component, especially when
            # it lies in the halo outside the strict transfer mask.
            ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
            ring &= ~comp
            vals = tg[ring]
            if vals.size == 0 or float(np.median(vals)) < 212.0:
                continue
            if target_hsv is not None:
                sats = target_hsv[..., 1][ring]
                if sats.size and float(np.median(sats)) > 46.0:
                    continue
            if not np.any(comp & near_clear) and not np.any(comp & use) and (area > 30 or max(bw, bh) > 11):
                continue
        clean[residual] = 255
        kept += 1

    if cv2.countNonZero(clean) == 0:
        return image.copy(), clean, {
            "residual_specks_removed": 0,
            "residual_speck_components": 0,
            "source_punctuation_protected": True,
        }
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    clean[source_near] = 0
    clean[~search_use] = 0
    # White paper is deterministic: restoring TARGET-local paper is both cleaner
    # and faster than Telea, and avoids introducing fresh dark specks. Coloured
    # regions still need structural inpaint.
    if white_container:
        out = clear_to_target_paper(image, target, clean)
    else:
        out = cv2.inpaint(image, clean, max(1.0, float(inpaint_radius)), cv2.INPAINT_TELEA)
    return out, clean, {
        "residual_specks_removed": int(cv2.countNonZero(clean)),
        "residual_speck_components": int(kept),
        "source_punctuation_protected": True,
    }


def cleanup_rendered_white_container_artifacts(
    image: np.ndarray,
    region_mask: np.ndarray,
    source_text_mask: np.ndarray,
    *,
    inpaint_radius: float = 1.8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove tiny post-composite blobs on white paper not supported by SOURCE text.

    These are neither legitimate Chinese glyphs nor coloured background. They are
    usually Telea/inpaint specks or antialias fragments left inside white speech
    bubbles. Protection is based on a generous dilation of the SOURCE text mask,
    so genuine punctuation and ellipsis survive.
    """
    if image.shape[:2] != region_mask.shape or source_text_mask.shape != region_mask.shape:
        raise ValueError("rendered-artifact cleanup inputs must share shape")
    use = region_mask > 0
    if int(np.count_nonzero(use)) == 0:
        return image.copy(), np.zeros_like(region_mask), {"rendered_artifact_specks_removed": 0}
    gray = _gray(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV) if image.ndim == 3 else None
    support = cv2.dilate((source_text_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
    # Search only on neutral paper-like zones. Absolute brightness is checked per
    # component via the surrounding ring, so admit even light-gray artifacts.
    paper = use.copy()
    if hsv is not None:
        paper &= (hsv[..., 1] <= 58)
    dark = (gray <= 238) & paper & (~support)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    clean = np.zeros_like(region_mask)
    kept = 0
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area <= 0 or area > 96 or max(bw, bh) > 18:
            continue
        comp = labels == lab
        ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
        ring &= ~comp
        ring &= use
        if not np.any(ring):
            continue
        vals = gray[ring]
        ring_med = float(np.median(vals)) if vals.size else 0.0
        comp_med = float(np.median(gray[comp]))
        if vals.size == 0 or ring_med < 210.0:
            continue
        if comp_med > ring_med - 12.0:
            continue
        if hsv is not None:
            sats = hsv[..., 1][ring]
            if sats.size and float(np.median(sats)) > 46.0:
                continue
        clean[comp] = 255
        kept += 1
    if cv2.countNonZero(clean) == 0:
        return image.copy(), clean, {
            "rendered_artifact_specks_removed": 0,
            "rendered_artifact_components": 0,
        }
    # This helper is white-container only. Restore a robust local paper colour
    # instead of Telea so the final defensive pass cannot manufacture fresh
    # gray/black pixels on otherwise flat paper.
    ring = cv2.dilate((clean > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    ring &= (clean == 0) & use
    pixels = image[ring]
    if len(pixels):
        paper = np.median(pixels, axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    out = image.copy()
    out[clean > 0] = paper
    return out, clean, {
        "rendered_artifact_specks_removed": int(cv2.countNonZero(clean)),
        "rendered_artifact_components": int(kept),
    }



def white_container_paper_mask(
    target: np.ndarray,
    region_mask: np.ndarray,
    source_text_mask: np.ndarray | None = None,
    *,
    min_gray: int = 205,
    max_saturation: int = 58,
) -> np.ndarray:
    """Recover neutral TARGET paper inside a white-container candidate."""
    if target.shape[:2] != region_mask.shape:
        raise ValueError("white paper mask inputs must share shape")
    use = region_mask > 0
    if not np.any(use):
        return np.zeros_like(region_mask)
    gray = _gray(target)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) if target.ndim == 3 else None
    neutral = use & (gray >= int(min_gray))
    if hsv is not None:
        neutral &= hsv[..., 1] <= int(max_saturation)
    cand = neutral.astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats((cand > 0).astype(np.uint8), 8)
    if count <= 1:
        return cand
    anchor = None
    if source_text_mask is not None and source_text_mask.shape == region_mask.shape and cv2.countNonZero(source_text_mask) > 0:
        anchor = cv2.dilate((source_text_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
    rows = []
    for lab in range(1, count):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        comp = labels == lab
        support = int(np.count_nonzero(comp & anchor)) if anchor is not None else 0
        rows.append((support, area, lab))
    if not rows:
        return np.zeros_like(region_mask)
    if anchor is not None and max(x[0] for x in rows) > 0:
        keep = [lab for support, area, lab in rows if support > 0 and area >= 16]
    else:
        keep = [max(rows, key=lambda x: x[1])[2]]
    out = np.zeros_like(region_mask)
    for lab in keep:
        out[labels == lab] = 255

    # Rectangular narration boxes are common and TARGET Japanese can physically
    # touch one side of the box. In a threshold-derived white mask that creates
    # an *open notch*, so ordinary contour hole filling cannot recover the missing
    # interior and a fragment of JP survives forever. When the selected white
    # component already occupies almost its entire bounding rectangle, complete
    # that rectangle before contour filling. Oval/starburst balloons have much
    # lower bbox fill ratios and therefore do not enter this branch.
    ys2, xs2 = np.where(out > 0)
    if xs2.size:
        bx0, bx1 = int(xs2.min()), int(xs2.max()) + 1
        by0, by1 = int(ys2.min()), int(ys2.max()) + 1
        bbox_area = max(1, (bx1 - bx0) * (by1 - by0))
        bbox_fill_ratio = float(cv2.countNonZero(out[by0:by1, bx0:bx1]) / bbox_area)
        if bbox_fill_ratio >= 0.86:
            rect = np.zeros_like(out)
            rect[by0:by1, bx0:bx1] = 255
            rect[~use] = 0
            out = rect

    # Fill enclosed dark text holes inside the selected neutral component so the
    # paper mask describes the container interior, not only already-white pixels.
    contours, _ = cv2.findContours(out, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(out)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, -1)
        out = filled
    out[~use] = 0
    return out


def target_container_border_mask(target: np.ndarray, region_mask: np.ndarray, *, band_px: int = 4) -> np.ndarray:
    """Detect long dark TARGET rules/balloon outlines near a container boundary."""
    if target.shape[:2] != region_mask.shape:
        raise ValueError("border-mask inputs must share shape")
    use = (region_mask > 0).astype(np.uint8)
    if cv2.countNonZero(use) == 0:
        return np.zeros_like(region_mask)
    r = max(1, int(band_px))
    er = cv2.erode(use, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)), iterations=1)
    band = (use > 0) & (er == 0)
    gray = _gray(target)
    vals = gray[use > 0]
    local_bg = float(np.percentile(vals, 72.0)) if vals.size else 245.0
    thr = int(np.clip(local_bg - 48.0, 80, 190))
    dark = ((gray <= thr) & (use > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    ys, xs = np.where(use > 0)
    rw = max(1, int(xs.max() - xs.min() + 1)) if xs.size else region_mask.shape[1]
    rh = max(1, int(ys.max() - ys.min() + 1)) if ys.size else region_mask.shape[0]
    out = np.zeros_like(region_mask)
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area <= 0:
            continue
        comp = labels == lab
        if not np.any(comp & band):
            continue
        span_x = bw / max(1.0, float(rw)); span_y = bh / max(1.0, float(rh))
        fill = area / max(1.0, float(bw * bh))
        long_rule = ((span_x >= 0.42 and bh <= max(7, int(round(rh * 0.10)))) or
                     (span_y >= 0.42 and bw <= max(7, int(round(rw * 0.10)))))
        outline_like = span_x >= 0.62 and span_y >= 0.45 and fill <= 0.28
        if long_rule or outline_like:
            out[comp] = 255
    return out


def target_white_container_text_mask(target: np.ndarray, paper_mask: np.ndarray) -> np.ndarray:
    """Return compact TARGET lettering on neutral paper, excluding border rules."""
    if target.shape[:2] != paper_mask.shape:
        raise ValueError("target white-text mask inputs must share shape")
    use = paper_mask > 0
    if not np.any(use):
        return np.zeros_like(paper_mask)
    gray = _gray(target)
    thr = _adaptive_ink_threshold(gray, use, floor=100, ceiling=210)
    dark = ((gray < min(210, thr + 18)) & use).astype(np.uint8)
    border = target_container_border_mask(target, paper_mask, band_px=4) > 0
    dark[border] = 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    ys, xs = np.where(use)
    rw = max(1, int(xs.max() - xs.min() + 1)); rh = max(1, int(ys.max() - ys.min() + 1))
    out = np.zeros_like(paper_mask)
    region_area = max(1, int(np.count_nonzero(use)))
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > max(220, int(round(region_area * 0.10))):
            continue
        span_x = bw / max(1.0, float(rw)); span_y = bh / max(1.0, float(rh))
        fill = area / max(1.0, float(bw * bh))
        if (span_x > 0.72 or span_y > 0.80) and fill < 0.20:
            continue
        out[labels == lab] = 255
    return out


def remove_container_boundary_line_components(mask: np.ndarray, region_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Remove mask components that are clearly container boundary rules."""
    if mask.shape != region_mask.shape:
        raise ValueError("boundary-line inputs must share shape")
    src = ((mask > 0) & (region_mask > 0)).astype(np.uint8)
    if cv2.countNonZero(src) == 0:
        return np.zeros_like(mask), 0
    use = (region_mask > 0).astype(np.uint8)
    er = cv2.erode(use, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    band = (use > 0) & (er == 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src, 8)
    ys, xs = np.where(use > 0)
    rw = max(1, int(xs.max() - xs.min() + 1)); rh = max(1, int(ys.max() - ys.min() + 1))
    out = src.copy(); removed = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        comp = labels == lab
        if not np.any(comp & band):
            continue
        sx = bw / float(rw); sy = bh / float(rh); fill = area / max(1.0, float(bw * bh))
        line_like = ((sx >= 0.42 and bh <= max(8, int(rh * 0.12))) or
                     (sy >= 0.42 and bw <= max(8, int(rw * 0.12))))
        outline_like = sx >= 0.55 and sy >= 0.40 and fill <= 0.30
        if line_like or outline_like:
            removed += int(np.count_nonzero(comp)); out[comp] = 0
    return out.astype(np.uint8) * 255, int(removed)


def white_container_write_envelope(target: np.ndarray, region_mask: np.ndarray, paper_mask: np.ndarray, *, inset_px: int = 2, border_guard_px: int = 2) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a TARGET-paper write envelope protected from HD border/rule pixels."""
    if target.shape[:2] != region_mask.shape or paper_mask.shape != region_mask.shape:
        raise ValueError("white envelope inputs must share shape")
    env = ((region_mask > 0) & (paper_mask > 0)).astype(np.uint8) * 255
    before = int(cv2.countNonZero(env))
    inset = max(0, int(inset_px))
    if inset > 0 and before > 0:
        er = cv2.erode(env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1)), iterations=1)
        if cv2.countNonZero(er) > 0:
            env = er
    border = target_container_border_mask(target, region_mask, band_px=max(3, inset + 2))
    border_pixels = int(cv2.countNonZero(border))
    guard = max(0, int(border_guard_px))
    if border_pixels > 0:
        protected = border
        if guard > 0:
            protected = cv2.dilate(border, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1)), iterations=1)
        env[protected > 0] = 0
    return env, {
        "paper_pixels": int(cv2.countNonZero(paper_mask)), "border_pixels": border_pixels,
        "envelope_pixels": int(cv2.countNonZero(env)), "removed_pixels": int(max(0, before - cv2.countNonZero(env))),
        "inset_px": inset, "border_guard_px": guard,
    }



def clear_uniform_white_container_interior(
    image: np.ndarray,
    target: np.ndarray,
    safe_mask: np.ndarray,
    *,
    min_paper_ratio: float = 0.68,
    max_robust_spread: float = 14.0,
    min_gray: int = 205,
    max_saturation: int = 58,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Blank a confirmed near-solid white container *before* Chinese is drawn.

    This intentionally follows the robust strategy used by mature manga
    translation editors: once a closed speech/narration container is proven to
    have a near-uniform paper background, do not chase individual JP glyph
    fragments. Restore the whole protected interior to the TARGET paper colour,
    then let the caller composite SOURCE Chinese ink afterwards.

    The operation is deliberately limited to ``safe_mask``; callers must exclude
    the HD target border/rules first. Coloured/halftone/artwork containers fail the
    paper-ratio/uniformity gate and keep using the normal text-mask + inpaint path.
    """
    if image.shape != target.shape or image.shape[:2] != safe_mask.shape:
        raise ValueError("uniform white-container inputs must share shape")
    use = safe_mask > 0
    area = int(np.count_nonzero(use))
    empty = np.zeros_like(safe_mask)
    if area < 16:
        return image.copy(), empty, {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "empty_safe_mask",
            "white_full_clear_pixels": 0,
        }

    gray = _gray(target)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) if target.ndim == 3 else None
    support = use & (gray >= int(min_gray))
    if hsv is not None:
        support &= hsv[..., 1] <= int(max_saturation)
    support_count = int(np.count_nonzero(support))
    paper_ratio = float(support_count / max(1, area))
    if support_count < 12:
        return image.copy(), empty, {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "insufficient_paper_support",
            "white_full_clear_paper_ratio": paper_ratio,
            "white_full_clear_pixels": 0,
        }

    pixels = target[support].astype(np.float32)
    paper = np.median(pixels, axis=0)
    dev = np.abs(pixels - paper)
    robust_spread = float(np.max(np.percentile(dev, 90.0, axis=0))) if len(pixels) else 999.0
    qualifies = paper_ratio >= float(min_paper_ratio) and robust_spread <= float(max_robust_spread)
    if not qualifies:
        return image.copy(), empty, {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "not_uniform_white_container",
            "white_full_clear_paper_ratio": paper_ratio,
            "white_full_clear_robust_spread": robust_spread,
            "white_full_clear_min_paper_ratio": float(min_paper_ratio),
            "white_full_clear_max_robust_spread": float(max_robust_spread),
            "white_full_clear_pixels": 0,
        }

    out = image.copy()
    fill = np.clip(np.round(paper), 0, 255).astype(np.uint8)
    out[use] = fill
    clear = use.astype(np.uint8) * 255
    return out, clear, {
        "white_full_clear_applied": True,
        "white_full_clear_reason": "uniform_target_paper",
        "white_full_clear_pixels": int(cv2.countNonZero(clear)),
        "white_full_clear_paper_ratio": paper_ratio,
        "white_full_clear_robust_spread": robust_spread,
        "white_full_clear_paper_bgr": fill.tolist(),
        "white_full_clear_min_paper_ratio": float(min_paper_ratio),
        "white_full_clear_max_robust_spread": float(max_robust_spread),
        "background_policy": "target_median_paper_only",
    }

def cleanup_target_residual_text_in_white_container(image: np.ndarray, target: np.ndarray, paper_mask: np.ndarray, source_text_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove TARGET Japanese components that remain on confirmed white paper."""
    if image.shape != target.shape or image.shape[:2] != paper_mask.shape or source_text_mask.shape != paper_mask.shape:
        raise ValueError("white residual cleanup inputs must share shape")
    target_text = target_white_container_text_mask(target, paper_mask)
    if cv2.countNonZero(target_text) == 0:
        return image.copy(), np.zeros_like(paper_mask), {"white_residual_text_removed": 0, "white_residual_components": 0}
    protect = cv2.dilate((source_text_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    cand = (target_text > 0) & (~protect) & (paper_mask > 0) & (_gray(image) <= 215)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    clean = np.zeros_like(paper_mask); kept = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > 900 or bw > 70 or bh > 90:
            continue
        clean[labels == lab] = 255; kept += 1
    if cv2.countNonZero(clean) == 0:
        return image.copy(), clean, {"white_residual_text_removed": 0, "white_residual_components": 0}
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    clean[protect] = 0; clean[paper_mask == 0] = 0
    out = clear_to_target_paper(image, target, clean)
    return out, clean, {"white_residual_text_removed": int(cv2.countNonZero(clean)), "white_residual_components": int(kept)}


def cleanup_white_container_line_artifacts(image: np.ndarray, target: np.ndarray, paper_mask: np.ndarray, source_text_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove unsupported faint short/line artifacts on confirmed neutral paper."""
    if image.shape != target.shape or image.shape[:2] != paper_mask.shape or source_text_mask.shape != paper_mask.shape:
        raise ValueError("white line cleanup inputs must share shape")
    use = paper_mask > 0
    if not np.any(use):
        return image.copy(), np.zeros_like(paper_mask), {"white_line_artifacts_removed": 0, "white_line_artifact_components": 0}
    gray = _gray(image)
    protect = cv2.dilate((source_text_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    cand = use & (~protect) & (gray >= 120) & (gray <= 238)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    clean = np.zeros_like(paper_mask); kept = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > 260 or max(bw, bh) > 90:
            continue
        comp = labels == lab
        ring = cv2.dilate(comp.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1) > 0
        ring &= ~comp; ring &= use
        if not np.any(ring):
            continue
        ring_med = float(np.median(gray[ring])); comp_med = float(np.median(gray[comp]))
        if ring_med < 225.0 or comp_med > ring_med - 8.0:
            continue
        fill = area / max(1.0, float(bw * bh))
        elongated = max(bw, bh) >= max(10, 3 * max(1, min(bw, bh))) or fill <= 0.34
        if not elongated and area > 28:
            continue
        clean[comp] = 255; kept += 1
    if cv2.countNonZero(clean) == 0:
        return image.copy(), clean, {"white_line_artifacts_removed": 0, "white_line_artifact_components": 0}
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    clean[protect] = 0; clean[~use] = 0
    out = clear_to_target_paper(image, target, clean)
    return out, clean, {"white_line_artifacts_removed": int(cv2.countNonZero(clean)), "white_line_artifact_components": int(kept)}

def cleanup_tight_border_residuals(
    image: np.ndarray,
    target: np.ndarray,
    paper_mask: np.ndarray,
    source_text_mask: np.ndarray,
    *,
    max_thickness: int = 3,
    min_span_ratio: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Remove thin dark residual rules *inside* paper near its safe boundary.

    This is intentionally not a generic line remover. A candidate must be on
    TARGET paper, outside the Chinese protection halo, elongated, and touch the
    inner paper boundary band. The actual balloon/narration outline normally
    lies outside ``paper_mask`` and therefore cannot be erased here.
    """
    if image.shape != target.shape or image.shape[:2] != paper_mask.shape or source_text_mask.shape != paper_mask.shape:
        raise ValueError("tight-border cleanup inputs must share shape")
    use = paper_mask > 0
    if not np.any(use):
        empty = np.zeros_like(paper_mask)
        return image.copy(), empty, {"tight_border_residuals_removed": 0, "tight_border_residual_components": 0}
    protect = cv2.dilate(
        (source_text_mask > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1,
    ) > 0
    eroded = cv2.erode((use.astype(np.uint8) * 255), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    edge_band = use & (~eroded)
    gray = _gray(image)
    cand = use & (~protect) & (gray < 160)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    ys, xs = np.where(use)
    rw = max(1, int(xs.max() - xs.min() + 1)); rh = max(1, int(ys.max() - ys.min() + 1))
    clean = np.zeros_like(paper_mask)
    kept = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 4 or area > max(420, int(round(np.count_nonzero(use) * 0.08))):
            continue
        comp = labels == lab
        if not np.any(comp & edge_band):
            continue
        is_h = bh <= int(max_thickness) and bw >= max(12, int(round(rw * float(min_span_ratio))))
        is_v = bw <= int(max_thickness) and bh >= max(12, int(round(rh * float(min_span_ratio))))
        if not (is_h or is_v):
            continue
        clean[comp] = 255
        kept += 1
    if kept <= 0:
        return image.copy(), clean, {"tight_border_residuals_removed": 0, "tight_border_residual_components": 0}
    clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    clean[protect] = 0
    clean[~use] = 0
    out = clear_to_target_paper(image, target, clean)
    return out, clean, {
        "tight_border_residuals_removed": int(cv2.countNonZero(clean)),
        "tight_border_residual_components": int(kept),
    }


def transfer_text_only(
    target: np.ndarray,
    source: np.ndarray,
    region_mask: np.ndarray,
    *,
    tolerance_px: int = 2,
    clear_dilate_px: int = 1,
    inpaint_radius: float = 2.5,
    white_container: bool = False,
    localized_white_text: bool = False,
    white_full_clear_enabled: bool = True,
    white_full_clear_min_paper_ratio: float = 0.68,
    white_full_clear_max_robust_spread: float = 14.0,
    white_write_inset_px: int = 1,
    white_write_border_guard_px: int = 1,
    white_clear_inset_px: int = 0,
    white_clear_border_guard_px: int = 0,
    target_clear_region_mask: np.ndarray | None = None,
    forced_target_clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Replace lettering while keeping TARGET background/artwork authoritative.

    The only permitted changes are (1) inpainting TARGET-only text strokes and
    (2) drawing SOURCE-only Chinese ink as neutral black. SOURCE RGB background
    pixels are never composited, even for a white speech bubble.
    """
    s_mask, t_mask, diag = changed_text_masks(
        source, target, region_mask, tolerance_px=tolerance_px
    )
    diag["changed_source_text_pixels"] = int(diag.get("source_text_pixels", 0))
    diag["changed_target_text_pixels"] = int(diag.get("target_text_pixels", 0))
    paper_mask = None
    white_env = None
    white_clear_env = None
    white_env_diag: dict[str, Any] = {}
    white_clear_env_diag: dict[str, Any] = {}
    clear_region_mask = region_mask
    if target_clear_region_mask is not None:
        candidate_clear = np.asarray(target_clear_region_mask, dtype=np.uint8)
        if candidate_clear.shape == region_mask.shape and cv2.countNonZero(candidate_clear) > 0:
            clear_region_mask = (candidate_clear > 0).astype(np.uint8) * 255
    # Once a candidate has independently proved to be a white text container,
    # use the complete compact dark ink inside its protected interior.  This
    # removes every Japanese stroke and restores every Chinese stroke while still
    # never copying SOURCE paper RGB.  The *changed* masks above remain the
    # evidence used by the caller to reject panel/artwork false positives.
    if bool(white_container):
        use = region_mask > 0
        clear_use = clear_region_mask > 0
        sg = _gray(source); tg = _gray(target)
        sth = _adaptive_ink_threshold(sg, use); tth = _adaptive_ink_threshold(tg, clear_use)
        s_all = ((sg < sth) & use).astype(np.uint8) * 255
        t_all = ((tg < tth) & clear_use).astype(np.uint8) * 255
        s_mask = _component_text_selector(
            s_all, s_all, region_mask,
            min_unique_pixels=1, min_unique_ratio=0.0, max_component_fraction=0.07,
        )
        t_mask = _component_text_selector(
            t_all, t_all, clear_region_mask,
            min_unique_pixels=1, min_unique_ratio=0.0, max_component_fraction=0.07,
        )
        s_mask = _compact_text_clusters(s_mask, region_mask)
        t_mask = _compact_text_clusters(t_mask, region_mask)
        # Do not lose tiny Japanese punctuation just because it is too small to
        # form a dense text cluster. This is TARGET-only cleanup; SOURCE noise is
        # deliberately not admitted by the same rule.
        t_mask = cv2.bitwise_or(t_mask, _small_interior_target_components(t_all, clear_region_mask))
        s_mask = prune_source_text_mask(s_mask, region_mask)
        s_mask = _relax_source_text_mask(s_mask, source, region_mask)
        # A white narration/speech-box outline is geometry, never translated
        # lettering. This helper existed before v1.3.10 but was not wired into
        # the published white-container path, allowing a top/bottom rule to be
        # copied as black "Chinese ink".
        s_mask, source_boundary_removed = remove_container_boundary_line_components(s_mask, region_mask)
        diag["source_boundary_line_pixels_removed"] = int(source_boundary_removed)
        # v1.3: geometry may overshoot into coloured/dark artwork. Derive the
        # actual neutral paper from TARGET, then make it the only writable
        # white-container envelope. This is the missing implementation promised
        # by REPAIR_PLAN_20260814.
        paper_mask = white_container_paper_mask(target, clear_region_mask, s_mask)
        white_env, white_env_diag = white_container_write_envelope(
            target, region_mask, paper_mask,
            inset_px=max(0, int(white_write_inset_px)),
            border_guard_px=max(0, int(white_write_border_guard_px)),
        )
        # TARGET clearing is allowed closer to the true HD border than SOURCE
        # writing.  The actual long border/rule pixels remain protected, but the
        # generic inset ring is not restored wholesale; this prevents an edge JP
        # glyph from surviving merely because it touched the box margin.
        white_clear_env, white_clear_env_diag = white_container_write_envelope(
            target, clear_region_mask, paper_mask,
            inset_px=max(0, int(white_clear_inset_px)),
            border_guard_px=max(0, int(white_clear_border_guard_px)),
        )
        if cv2.countNonZero(white_env) > 0:
            halo = cv2.dilate(white_env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
            s_mask[~halo] = 0
        if white_clear_env is not None and cv2.countNonZero(white_clear_env) > 0:
            clear_halo = cv2.dilate(white_clear_env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
            t_mask[~clear_halo] = 0
        diag["white_container_complete_ink"] = True
        diag["white_container_write_envelope"] = white_env_diag
        diag["white_container_clear_envelope"] = white_clear_env_diag
        diag["source_text_pixels"] = int(cv2.countNonZero(s_mask))
        diag["target_text_pixels"] = int(cv2.countNonZero(t_mask))
    elif bool(localized_white_text) and cv2.countNonZero(s_mask) > 0:
        # Large contour candidates may contain a real nested white balloon plus
        # unrelated art. Complete the glyphs only around the proven changed
        # SOURCE text cluster, never across the whole contour.
        use = region_mask > 0
        sg = _gray(source); tg = _gray(target)
        sth = _adaptive_ink_threshold(sg, use); tth = _adaptive_ink_threshold(tg, use)
        s_all = ((sg < sth) & use).astype(np.uint8) * 255
        t_all = ((tg < tth) & use).astype(np.uint8) * 255
        # Tight SOURCE growth fills shared/antialiased portions of each Chinese
        # glyph while excluding distant artwork.
        tight_r = 8
        tight = cv2.dilate((s_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tight_r * 2 + 1, tight_r * 2 + 1))) > 0
        s_complete = (s_all > 0) & tight
        s_mask = s_complete.astype(np.uint8) * 255
        # TARGET Japanese layout can shift more within the same bubble; use a
        # broader association window but still clip every pixel to it.
        ys, xs = np.where(s_mask > 0)
        if xs.size:
            span = max(int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1))
        else:
            span = 40
        broad_r = max(18, min(44, int(round(span * 0.28))))
        broad = cv2.dilate((s_mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (broad_r * 2 + 1, broad_r * 2 + 1))) > 0
        t_complete = (t_all > 0) & broad
        t_mask = t_complete.astype(np.uint8) * 255
        tiny = _small_interior_target_components(t_all, region_mask)
        t_mask = cv2.bitwise_or(t_mask, cv2.bitwise_and(tiny, broad.astype(np.uint8) * 255))
        s_mask = prune_source_text_mask(s_mask, region_mask)
        s_mask = _relax_source_text_mask(s_mask, source, region_mask)
        diag["localized_white_complete_ink"] = True
        diag["source_text_pixels"] = int(cv2.countNonZero(s_mask))
        diag["target_text_pixels"] = int(cv2.countNonZero(t_mask))
    out = target.copy()
    clear = t_mask.copy()
    forced_clear_pixels = 0
    if forced_target_clear_mask is not None:
        forced = np.asarray(forced_target_clear_mask, dtype=np.uint8)
        if forced.shape == region_mask.shape:
            forced = ((forced > 0) & (region_mask > 0)).astype(np.uint8) * 255
            if cv2.countNonZero(forced) > 0:
                clear = cv2.bitwise_or(clear, forced)
                forced_clear_pixels = int(cv2.countNonZero(forced))
    diag["forced_target_clear_pixels"] = int(forced_clear_pixels)
    diag["forced_target_clear_before_source_paint"] = bool(forced_clear_pixels > 0)
    # General text-only callers may provide a narrow TARGET-authority mask while
    # keeping a wider region for SOURCE Chinese discovery.  Restrict automatic
    # TARGET difference clearing to that narrow mask; otherwise small registration
    # differences in burst rays, borders, hair or screentone can be erased.
    if target_clear_region_mask is not None and not bool(white_container):
        clear_gate = np.asarray(target_clear_region_mask, dtype=np.uint8)
        if clear_gate.shape == region_mask.shape and cv2.countNonZero(clear_gate) > 0:
            t_mask = cv2.bitwise_and(t_mask, (clear_gate > 0).astype(np.uint8) * 255)
            clear = cv2.bitwise_and(clear, (clear_gate > 0).astype(np.uint8) * 255)
            if forced_target_clear_mask is not None:
                forced = np.asarray(forced_target_clear_mask, dtype=np.uint8)
                if forced.shape == region_mask.shape:
                    forced = ((forced > 0) & (clear_gate > 0)).astype(np.uint8) * 255
                    clear = cv2.bitwise_or(clear, forced)
            diag["target_clear_region_restricted"] = True
            diag["target_clear_region_pixels"] = int(cv2.countNonZero(clear_gate))
    full_clear_diag = {
        "white_full_clear_applied": False,
        "white_full_clear_reason": "not_requested",
        "white_full_clear_pixels": 0,
    }
    # Confirmed uniform white speech/narration containers are blanked as a whole
    # protected interior, then SOURCE Chinese is drawn below. This eliminates JP
    # leftovers by construction instead of relying on perfect per-glyph detection.
    if (
        bool(white_container) and bool(white_full_clear_enabled)
        and white_clear_env is not None and cv2.countNonZero(white_clear_env) > 0
    ):
        full_out, full_mask, full_clear_diag = clear_uniform_white_container_interior(
            out, target, white_clear_env,
            min_paper_ratio=float(white_full_clear_min_paper_ratio),
            max_robust_spread=float(white_full_clear_max_robust_spread),
        )
        if bool(full_clear_diag.get("white_full_clear_applied", False)):
            out = full_out
            clear = full_mask

    if not bool(full_clear_diag.get("white_full_clear_applied", False)):
        d = max(0, int(clear_dilate_px))
        if d > 0 and cv2.countNonZero(clear):
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
            clear = cv2.dilate(clear, k)
            clear[region_mask == 0] = 0
        if bool(white_container) and white_clear_env is not None and cv2.countNonZero(white_clear_env) > 0:
            clear_halo = cv2.dilate(white_clear_env, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            clear = cv2.bitwise_and(clear, clear_halo)
    local_paper_diag={"local_paper_components":0,"local_paper_clear_pixels":0,"local_paper_rejected_components":0}
    if cv2.countNonZero(clear) and not bool(full_clear_diag.get("white_full_clear_applied", False)):
        if bool(white_container):
            out = clear_to_target_paper(out, target, clear)
        elif bool(localized_white_text):
            out, accepted_clear, local_paper_diag = clear_text_components_to_local_paper(out,target,clear,region_mask)
            clear=accepted_clear
        else:
            out = cv2.inpaint(out, clear, float(inpaint_radius), cv2.INPAINT_TELEA)
    clean_bg = out.copy()

    alpha, source_gray = source_text_render(source, s_mask)
    if np.any(alpha > 0):
        # Render with SOURCE-derived grayscale ink, not synthetic solid black.
        # This preserves the Chinese page's anti-aliased letter appearance while
        # still never copying SOURCE background pixels.
        a3 = alpha[..., None]
        ink = np.repeat(source_gray[..., None], 3, axis=2).astype(np.float32)
        out = np.clip(out.astype(np.float32) * (1.0 - a3) + ink * a3, 0, 255).astype(np.uint8)

    # Remove only small TARGET-origin leftovers that survived the first clear.
    # SOURCE punctuation is protected by ``s_mask`` and therefore never treated
    # as a speck. This is deliberately post-composite so we inspect the actual
    # published pixels rather than only the proposed mask.
    residual_diag = {"residual_specks_removed": 0, "residual_speck_components": 0, "source_punctuation_protected": True}
    artifact_diag = {"rendered_artifact_specks_removed": 0, "rendered_artifact_components": 0}
    line_diag = {"white_line_artifacts_removed": 0, "white_line_artifact_components": 0}
    tight_border_diag = {"tight_border_residuals_removed": 0, "tight_border_residual_components": 0}
    white_residual_diag = {"white_residual_text_removed": 0, "white_residual_components": 0}
    if bool(white_container):
        # Confirmed ordinary white bubble: normalize only TARGET-derived paper,
        # never the broad detector rectangle. This fixes overshoot into dark art.
        protect = cv2.dilate((s_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
        paper_write = white_env if white_env is not None and cv2.countNonZero(white_env) > 0 else region_mask
        swap = (paper_write > 0) & (~protect)
        out[swap] = clean_bg[swap]
        if paper_mask is None:
            paper_mask = white_container_paper_mask(target, region_mask, s_mask)
        out, residual_text_mask, white_residual_diag = cleanup_target_residual_text_in_white_container(
            out, target, paper_mask, s_mask
        )
        out, line_mask, line_diag = cleanup_white_container_line_artifacts(
            out, target, paper_mask, s_mask
        )
        out, tight_border_mask, tight_border_diag = cleanup_tight_border_residuals(
            out, target, paper_mask, s_mask
        )
        # Final TARGET-origin punctuation/dakuten/AA fragment sweep. Unlike the
        # coloured path, white paper is restored from TARGET paper colour rather
        # than Telea, so cleanup cannot manufacture new black specks.
        out, residual_mask, residual_diag = cleanup_target_residual_specks(
            out, target, paper_mask, s_mask, clear,
            white_container=True,
            inpaint_radius=max(1.5, float(inpaint_radius)),
        )
        # Last defensive pass catches tiny post-clear artifacts unsupported by
        # SOURCE Chinese. This helper existed before v1.3.6 but was not wired into
        # the published white-container chain.
        out, artifact_mask, artifact_diag = cleanup_rendered_white_container_artifacts(
            out, paper_mask, s_mask, inpaint_radius=max(1.2, float(inpaint_radius)),
        )
    else:
        # Large/localized or coloured candidates are never normalized as an
        # entire white container. Only already-associated text components can
        # change, so人物/皮肤/网点 remain byte-stable outside those components.
        out, residual_mask, residual_diag = cleanup_target_residual_specks(
            out, target, region_mask, s_mask, clear,
            white_container=False,
            inpaint_radius=max(1.5, float(inpaint_radius)),
        )

    changed = np.any(out != target, axis=2)
    write = np.zeros(region_mask.shape, np.uint8)
    write[changed] = 255
    diag.update({
        "cleared_target_pixels": int(cv2.countNonZero(clear)),
        "source_ink_pixels": int(cv2.countNonZero(s_mask)),
        "write_pixels": int(cv2.countNonZero(write)),
        "background_policy": "target_only",
        **residual_diag,
        **artifact_diag,
        **white_residual_diag,
        **line_diag,
        **tight_border_diag,
        **local_paper_diag,
        **full_clear_diag,
    })
    return out, write, s_mask, diag
