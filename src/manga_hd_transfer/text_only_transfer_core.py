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


def source_text_alpha(source: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    """Build antialiased opacity from the SOURCE glyph topology only."""
    sg = _gray(source).astype(np.float32)
    use = source_mask > 0
    alpha = np.zeros(source_mask.shape, np.float32)
    if not np.any(use):
        return alpha
    # A robust local paper estimate; antialias shades between glyph core and the
    # old paper become fractional opacity, but the paper itself has alpha 0.
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
    alpha[use] = np.maximum(0.35, raw[use])
    return np.clip(alpha, 0.0, 1.0)



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
    max_area = 110 if white_container else 18
    max_dim = 24 if white_container else 8
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
            if vals.size == 0 or float(np.median(vals)) < 215.0:
                continue
            if target_hsv is not None:
                sats = target_hsv[..., 1][ring]
                if sats.size and float(np.median(sats)) > 42.0:
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
    out = cv2.inpaint(image, clean, max(1.0, float(inpaint_radius)), cv2.INPAINT_TELEA)
    return out, clean, {
        "rendered_artifact_specks_removed": int(cv2.countNonZero(clean)),
        "rendered_artifact_components": int(kept),
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
    # Once a candidate has independently proved to be a white text container,
    # use the complete compact dark ink inside its protected interior.  This
    # removes every Japanese stroke and restores every Chinese stroke while still
    # never copying SOURCE paper RGB.  The *changed* masks above remain the
    # evidence used by the caller to reject panel/artwork false positives.
    if bool(white_container):
        use = region_mask > 0
        sg = _gray(source); tg = _gray(target)
        sth = _adaptive_ink_threshold(sg, use); tth = _adaptive_ink_threshold(tg, use)
        s_all = ((sg < sth) & use).astype(np.uint8) * 255
        t_all = ((tg < tth) & use).astype(np.uint8) * 255
        s_mask = _component_text_selector(
            s_all, s_all, region_mask,
            min_unique_pixels=1, min_unique_ratio=0.0, max_component_fraction=0.07,
        )
        t_mask = _component_text_selector(
            t_all, t_all, region_mask,
            min_unique_pixels=1, min_unique_ratio=0.0, max_component_fraction=0.07,
        )
        s_mask = _compact_text_clusters(s_mask, region_mask)
        t_mask = _compact_text_clusters(t_mask, region_mask)
        # Do not lose tiny Japanese punctuation just because it is too small to
        # form a dense text cluster. This is TARGET-only cleanup; SOURCE noise is
        # deliberately not admitted by the same rule.
        t_mask = cv2.bitwise_or(t_mask, _small_interior_target_components(t_all, region_mask))
        s_mask = prune_source_text_mask(s_mask, region_mask)
        diag["white_container_complete_ink"] = True
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
        diag["localized_white_complete_ink"] = True
        diag["source_text_pixels"] = int(cv2.countNonZero(s_mask))
        diag["target_text_pixels"] = int(cv2.countNonZero(t_mask))
    out = target.copy()
    clear = t_mask.copy()
    d = max(0, int(clear_dilate_px))
    if d > 0 and cv2.countNonZero(clear):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
        clear = cv2.dilate(clear, k)
        clear[region_mask == 0] = 0
    local_paper_diag={"local_paper_components":0,"local_paper_clear_pixels":0,"local_paper_rejected_components":0}
    if cv2.countNonZero(clear):
        if bool(white_container):
            out = clear_to_target_paper(out, target, clear)
        elif bool(localized_white_text):
            out, accepted_clear, local_paper_diag = clear_text_components_to_local_paper(out,target,clear,region_mask)
            clear=accepted_clear
        else:
            out = cv2.inpaint(out, clear, float(inpaint_radius), cv2.INPAINT_TELEA)
    clean_bg = out.copy()

    alpha = source_text_alpha(source, s_mask)
    if np.any(alpha > 0):
        # Neutral black carries the old glyph *shape* but never the old page fill.
        a3 = alpha[..., None]
        out = np.clip(out.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)

    # Remove only small TARGET-origin leftovers that survived the first clear.
    # SOURCE punctuation is protected by ``s_mask`` and therefore never treated
    # as a speck. This is deliberately post-composite so we inspect the actual
    # published pixels rather than only the proposed mask.
    residual_diag = {"residual_specks_removed": 0, "residual_speck_components": 0, "source_punctuation_protected": True}
    artifact_diag = {"rendered_artifact_specks_removed": 0, "rendered_artifact_components": 0}
    if bool(white_container):
        # Confirmed ordinary white bubble: no Telea. Keep clean TARGET paper
        # everywhere except protected Chinese glyphs.
        inner = cv2.erode((region_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        protect = cv2.dilate((s_mask > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
        swap = (inner > 0) & (~protect)
        out[swap] = clean_bg[swap]
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
        **local_paper_diag,
    })
    return out, write, s_mask, diag
