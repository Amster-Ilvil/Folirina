from __future__ import annotations

"""Pure raster/mask primitives used by Mask Replace renderers.

This module is intentionally renderer-independent.  It owns deterministic
array transforms for target-boundary cleanup, source glyph-footprint rescue,
write-envelope expansion and alpha reconstruction.  It must not import the
page pipeline, GUI, review service or the high-level mask renderer.

The functions are progressively extracted from :mod:`mask_transfer` without
algorithm changes; keeping them isolated makes pixel-level regressions easier
to test and prevents geometry/raster helpers from acquiring orchestration state.
"""

import math

import cv2
import numpy as np

from ...config import MaskReplaceConfig
from .geometry_ops import _bbox_from_mask

def _target_white_ratio(image: np.ndarray, mask: np.ndarray, threshold: int) -> float:
    if cv2.countNonZero(mask) == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sel = mask > 0
    return float(np.mean(gray[sel] >= int(threshold))) if np.any(sel) else 0.0

def _alpha_from_mask(mask: np.ndarray, feather_px: int) -> np.ndarray:
    alpha = (mask > 0).astype(np.float32)
    if feather_px > 0:
        k = feather_px * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), max(0.5, feather_px * 0.55))
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha

def _expand_target_clear_mask_with_text_components(target: np.ndarray, target_mask: np.ndarray, cfg=None) -> np.ndarray:
    """Add compact target text components just outside an imperfect balloon mask.

    Bright-region balloon detection can stop at Japanese glyphs that nearly bridge
    the interior, especially in tiny balloons. The missing glyph tops then survive
    replacement and visually mix with Chinese. Add only compact dark components
    touching a small dilation of the target interior; reject long outline/panel
    components so the balloon border remains intact.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return target_mask.copy()
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pad = max(5, int(round(0.10 * max(bw, bh))))
    near = cv2.dilate(
        target_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    ) > 0
    dark = ((gray < 190) & near).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    out = target_mask.copy()
    mask_area = max(1, cv2.countNonZero(target_mask))
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 2 or area > 0.14 * mask_area:
            continue
        # v0.8.21: vertical Japanese columns can legitimately span most of a
        # balloon's height, and horizontal captions can span most of its width.
        # The old independent 42%/48% limits rejected exactly those glyph groups.
        # Keep long components only when the orthogonal dimension remains
        # text-column-like; true balloon/panel outlines are long in both axes or
        # become very large connected components and are still rejected above.
        long_vertical = ch <= 0.88 * bh and cw <= 0.28 * bw
        long_horizontal = cw <= 0.88 * bw and ch <= 0.30 * bh
        compact = cw <= 0.48 * bw and ch <= 0.54 * bh
        # Tiny target text boxes often have a final Japanese column pressed
        # against the edge. It is thin enough to be mistaken for the box border;
        # accept it only inside the confirmed text-box bbox and only when it is
        # clearly text-column-like, never as a general edge expansion.
        thin_edge_text = (
            bool(getattr(cfg, "paired_diff_clear_thin_edge_text", True))
            and cw <= float(getattr(cfg, "paired_diff_clear_thin_edge_text_max_width_ratio", 0.10)) * bw
            and ch <= float(getattr(cfg, "paired_diff_clear_thin_edge_text_max_height_ratio", 0.82)) * bh
        )
        if not (compact or long_vertical or long_horizontal or thin_edge_text):
            continue
        comp = (labels == i).astype(np.uint8) * 255
        # Include antialiased fringe around the old Japanese glyph so review-first
        # replacement does not leave a faint gray ghost after the black core is
        # cleared. One pixel is enough and still far smaller than balloon outlines.
        comp = cv2.dilate(comp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = np.maximum(out, comp)
    return out

def _extract_photo_text_cluster(
    aligned_source: np.ndarray,
    target_mask: np.ndarray,
    *,
    pad_ratio: float,
    dark_threshold: int,
    cluster_gap_ratio: float,
) -> tuple[np.ndarray, tuple[int, int, int, int], float, float] | None:
    """Extract the dominant compact source-ink cluster around a target bubble.

    The input page is already in target coordinates.  We intentionally do not use
    OCR or character boxes: compact dark connected components are grouped by local
    proximity, while long balloon/panel/art lines are discarded.  Returning the
    original raster footprint (plus a soft alpha later) lets mask replacement keep
    punctuation, glyph shapes, column layout and relative spacing verbatim.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None
    h, w = aligned_source.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(8, int(round(max(bw, bh) * float(np.clip(pad_ratio, 0.08, 0.55)))))
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(w, x1 + pad), min(h, y1 + pad)
    roi = aligned_source[ya:yb, xa:xb]
    if roi.size == 0:
        return None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gate_full = cv2.dilate(
        target_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    )
    gate = gate_full[ya:yb, xa:xb] > 0
    vals = gray[gate]
    if vals.size < 80:
        return None
    paper = float(np.percentile(vals, 88.0))
    thr = float(np.clip(min(float(dark_threshold), paper - 34.0), 105.0, 190.0))
    ink = ((gray <= thr) & gate).astype(np.uint8) * 255

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    clean = np.zeros_like(ink)
    gate_area = max(1, int(np.count_nonzero(gate)))
    rw, rh = ink.shape[1], ink.shape[0]
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 2 or area > 0.12 * gate_area:
            continue
        # Long components are almost always balloon outlines, panel borders,
        # hair/furniture edges or hatch lines rather than a CJK glyph component.
        if cw > 0.45 * rw or ch > 0.45 * rh:
            continue
        clean[labels == i] = 255
    if cv2.countNonZero(clean) < 12:
        return None

    # Join adjacent glyph pieces/characters into text blocks.  This is the key
    # outlier guard missing from the old recenter path: isolated speckles or art
    # fragments no longer enlarge the text bbox and force destructive shrinking.
    gap = int(np.clip(round(min(bw, bh) * float(cluster_gap_ratio)), 3, 14))
    grouped = cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_RECT, (gap * 2 + 1, gap * 2 + 1)),
    )
    gn, glabels, _, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    tm_roi = target_mask[ya:yb, xa:xb] > 0
    tcx = (x0 + x1) * 0.5 - xa
    tcy = (y0 + y1) * 0.5 - ya
    best: tuple[float, np.ndarray] | None = None
    for i in range(1, gn):
        region = glabels == i
        original = (clean > 0) & region
        area = int(np.count_nonzero(original))
        if area < 12:
            continue
        ys, xs = np.where(original)
        bx0, by0, bx1, by1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
        bbox_area = max(1, (bx1 - bx0) * (by1 - by0))
        fill = area / bbox_area
        overlap = float(np.count_nonzero(original & tm_roi) / max(1, area))
        cx, cy = float(xs.mean()), float(ys.mean())
        dist = math.hypot((cx - tcx) / bw, (cy - tcy) / bh)
        # A real dialogue block is compact, mostly intersects its target bubble,
        # and sits near that bubble.  Area stays dominant so punctuation columns
        # and multi-column CJK layouts remain together.
        score = (
            area
            * (0.62 + 0.38 * min(1.0, fill / 0.10))
            * (0.70 + 0.30 * overlap)
            / (1.0 + 0.22 * dist)
        )
        if best is None or score > best[0]:
            best = (score, original)
    if best is None:
        return None

    selected = best[1].astype(np.uint8) * 255
    ys, xs = np.where(selected > 0)
    if len(xs) == 0:
        return None
    ibox_local = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    ibox = (
        ibox_local[0] + xa,
        ibox_local[1] + ya,
        ibox_local[2] + xa,
        ibox_local[3] + ya,
    )

    # Re-introduce the antialiased fringe around the accepted components, but do
    # not resurrect long components that were deliberately classified as art.
    fringe = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    denom = max(35.0, paper - 82.0)
    alpha = np.clip((paper - gray.astype(np.float32)) / denom, 0.0, 1.0)
    alpha = np.power(alpha, 0.92)
    alpha[~fringe] = 0.0
    alpha[selected > 0] = np.maximum(alpha[selected > 0], 0.72)

    alpha_global = np.zeros(target_mask.shape, np.float32)
    alpha_global[ya:yb, xa:xb] = alpha
    raw_outside_ratio = float(
        np.count_nonzero((selected > 0) & (~tm_roi)) / max(1, cv2.countNonZero(selected))
    )
    ink_ratio = float(cv2.countNonZero(selected) / max(1, cv2.countNonZero(target_mask)))
    return alpha_global, ibox, raw_outside_ratio, ink_ratio

def _fit_alpha_into_target_mask(
    alpha: np.ndarray,
    target_mask: np.ndarray,
    text_bbox: tuple[int, int, int, int],
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, float, float, float, float]:
    """Move/shrink one raster glyph block minimally until all strokes fit.

    Search starts at scale 1.0 and only reduces scale if translation alone cannot
    reach the configured containment target.  X/Y are never scaled independently,
    so glyph proportions and the source typesetting remain unchanged.
    """
    h, w = target_mask.shape
    x0, y0, x1, y1 = text_bbox
    if x1 <= x0 or y1 <= y0:
        return None, 0.0, 0.0, 1.0, 0.0
    crop = alpha[y0:y1, x0:x1]
    if crop.size == 0 or float(crop.max()) <= 0:
        return None, 0.0, 0.0, 1.0, 0.0

    tbox = _bbox_from_mask(target_mask)
    if tbox is None:
        return None, 0.0, 0.0, 1.0, 0.0
    tbw, tbh = max(1, tbox[2] - tbox[0]), max(1, tbox[3] - tbox[1])
    inset = max(0, int(getattr(cfg, "photo_pair_glyph_rescue_safe_inset_px", 1)))
    safe = target_mask.copy()
    if inset > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
        eroded = cv2.erode(safe, k)
        if cv2.countNonZero(eroded) > max(20, int(cv2.countNonZero(safe) * 0.70)):
            safe = eroded
    safe_bool = safe > 0

    min_scale = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_min_scale", 0.86), 0.65, 1.0))
    step = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_scale_step", 0.02), 0.01, 0.10))
    scales = [1.0]
    s = 1.0 - step
    while s >= min_scale - 1e-9:
        scales.append(round(s, 4))
        s -= step
    required = float(np.clip(getattr(cfg, "photo_pair_glyph_rescue_min_coverage", 0.995), 0.94, 1.0))
    max_shift = max(3, int(round(max(tbw, tbh) * float(np.clip(
        getattr(cfg, "photo_pair_glyph_rescue_max_shift_ratio", 0.14), 0.03, 0.30
    )))))

    cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
    best_any: tuple[float, float, int, int, float, np.ndarray, int, int] | None = None
    for scale in scales:
        nw = max(1, int(round(crop.shape[1] * scale)))
        nh = max(1, int(round(crop.shape[0] * scale)))
        interp = cv2.INTER_AREA if scale < 0.999 else cv2.INTER_LINEAR
        scaled = cv2.resize(crop, (nw, nh), interpolation=interp)
        binary = scaled >= 0.14
        yy0, xx0 = np.where(binary)
        if len(xx0) < 8:
            continue
        base_x = int(round(cx - nw * 0.5))
        base_y = int(round(cy - nh * 0.5))

        local_best: tuple[float, int, int] | None = None
        # Manhattan shells make the first full-coverage hit the least invasive.
        for radius in range(0, max_shift + 1):
            shell: list[tuple[int, int]] = []
            if radius == 0:
                shell = [(0, 0)]
            else:
                for dx in range(-radius, radius + 1):
                    dy = radius - abs(dx)
                    shell.append((dx, dy))
                    if dy:
                        shell.append((dx, -dy))
            for dx, dy in shell:
                gx = xx0 + base_x + dx
                gy = yy0 + base_y + dy
                valid = (gx >= 0) & (gx < w) & (gy >= 0) & (gy < h)
                if not np.any(valid):
                    coverage = 0.0
                else:
                    vv = valid.copy()
                    inside = np.zeros_like(vv)
                    inside[vv] = safe_bool[gy[vv], gx[vv]]
                    coverage = float(np.count_nonzero(inside) / len(xx0))
                if local_best is None or coverage > local_best[0] + 1e-9:
                    local_best = (coverage, dx, dy)
                if coverage >= required:
                    out = np.zeros_like(alpha, np.float32)
                    px, py = base_x + dx, base_y + dy
                    sx0, sy0 = max(0, -px), max(0, -py)
                    dx0, dy0 = max(0, px), max(0, py)
                    cw = min(nw - sx0, w - dx0); ch = min(nh - sy0, h - dy0)
                    if cw <= 0 or ch <= 0:
                        continue
                    out[dy0:dy0 + ch, dx0:dx0 + cw] = scaled[sy0:sy0 + ch, sx0:sx0 + cw]
                    return out, float(dx), float(dy), float(scale), coverage
        if local_best is not None:
            coverage, dx, dy = local_best
            key = (coverage, scale, -(abs(dx) + abs(dy)))
            if best_any is None or key > (best_any[0], best_any[1], -best_any[2]):
                best_any = (coverage, scale, abs(dx) + abs(dy), dx, dy, scaled, base_x, base_y)

    # Do not publish a partially clipped "rescue". The ordinary crisp path below
    # is safer if the complete raster block cannot be contained with small motion.
    return None, 0.0, 0.0, 1.0, (best_any[0] if best_any is not None else 0.0)

def _reconstruct_photo_glyph_footprint_layer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float, dict[str, float]]:
    """Rescue complete source glyphs that straddle an imperfect target mask.

    This is a pure mask/pixel path: no OCR, no transcript, no font rendering.  It
    only activates when the dominant translated source-ink cluster demonstrably
    has pixels outside the target bubble mask.  The whole raster text block is
    then translated (and, only if necessary, uniformly shrunk) to achieve almost
    full containment.  Relative layout, punctuation and glyph shapes are preserved.
    """
    if not bool(getattr(cfg, "photo_pair_glyph_rescue_enabled", True)):
        return None, 0.0, {}
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None, 0.0, {}
    h, w = target.shape[:2]
    area_ratio = cv2.countNonZero(target_mask) / max(1, h * w)
    if area_ratio > float(getattr(cfg, "photo_pair_glyph_rescue_max_area_ratio", 0.028)):
        return None, 0.0, {}
    if _target_white_ratio(target, target_mask, cfg.ink_target_white_threshold) < 0.62:
        return None, 0.0, {}

    extracted = _extract_photo_text_cluster(
        aligned_source,
        target_mask,
        pad_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_pad_ratio", 0.30)),
        dark_threshold=int(getattr(cfg, "photo_pair_glyph_rescue_dark_threshold", 175)),
        cluster_gap_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_cluster_gap_ratio", 0.07)),
    )
    if extracted is None:
        return None, 0.0, {}
    alpha, ibox, outside_ratio, ink_ratio = extracted
    trigger = float(getattr(cfg, "photo_pair_glyph_rescue_min_outside_ink_ratio", 0.003))
    if outside_ratio < trigger:
        return None, ink_ratio, {"outside_ratio": outside_ratio}

    fitted, dx, dy, scale, coverage = _fit_alpha_into_target_mask(alpha, target_mask, ibox, cfg)
    if fitted is None:
        return None, ink_ratio, {"outside_ratio": outside_ratio, "coverage": coverage}

    out = target.copy()
    clear = clear_mask if clear_mask is not None else target_mask
    cb = _bbox_from_mask(clear)
    if cb is None:
        return None, ink_ratio, {}
    cx0, cy0, cx1, cy1 = cb
    clear_local = clear[cy0:cy1, cx0:cx1] > 0
    tgt_crop = target[cy0:cy1, cx0:cx1]
    tgray = cv2.cvtColor(tgt_crop, cv2.COLOR_BGR2GRAY)
    bright = clear_local & (tgray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt_crop[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([255.0, 255.0, 255.0], np.float32)
    clear_roi = out[cy0:cy1, cx0:cx1]
    clear_roi[clear_local] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    out[cy0:cy1, cx0:cx1] = clear_roi

    a = np.clip(fitted[..., None], 0.0, 1.0)
    out = np.clip(out.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    return out, ink_ratio, {
        "outside_ratio": outside_ratio,
        "coverage": coverage,
        "dx": dx,
        "dy": dy,
        "scale": scale,
    }

def _reconstruct_photo_recentered_ink_layer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Recover a complete small translated glyph block and re-center it.

    Cross-edition pages often keep the same speech balloon while the translated
    Chinese columns occupy slightly different positions from the Japanese text.
    The v0.8.16 implementation shares the clustered source-ink extractor with the
    boundary rescue path, so isolated JPEG/art speckles cannot enlarge the text
    bbox and force the entire block to shrink. No OCR or glyph generation occurs.
    """
    box = _bbox_from_mask(target_mask)
    if box is None:
        return None, 0.0
    h, w = target.shape[:2]
    x0, y0, x1, y1 = box
    area_ratio = cv2.countNonZero(target_mask) / max(1, h * w)
    if area_ratio > float(getattr(cfg, "photo_pair_recenter_max_area_ratio", 0.006)):
        return None, 0.0
    bw, bh = x1 - x0, y1 - y0
    if min(bw, bh) < 24:
        return None, 0.0
    if _target_white_ratio(target, target_mask, cfg.ink_target_white_threshold) < 0.62:
        return None, 0.0

    extracted = _extract_photo_text_cluster(
        aligned_source,
        target_mask,
        pad_ratio=float(getattr(cfg, "photo_pair_recenter_pad_ratio", 0.30)),
        dark_threshold=int(getattr(cfg, "photo_pair_recenter_dark_threshold", 175)),
        cluster_gap_ratio=float(getattr(cfg, "photo_pair_glyph_rescue_cluster_gap_ratio", 0.07)),
    )
    if extracted is None:
        return None, 0.0
    alpha_global, ibox, _, _ = extracted
    ix0, iy0, ix1, iy1 = ibox
    if ix1 - ix0 < 5 or iy1 - iy0 < 5:
        return None, 0.0
    crop = alpha_global[iy0:iy1, ix0:ix1]
    if crop.size == 0 or float(crop.max()) <= 0:
        return None, 0.0

    # Derive a safe destination interior from the target balloon and fit with one
    # scale factor only—never squeeze X/Y independently.
    safe = cv2.erode(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    sbox = _bbox_from_mask(safe) or box
    sx0, sy0, sx1, sy1 = sbox
    fit_ratio = float(np.clip(getattr(cfg, "photo_pair_recenter_fit_ratio", 0.88), 0.60, 0.96))
    avail_w = max(1.0, (sx1 - sx0) * fit_ratio)
    avail_h = max(1.0, (sy1 - sy0) * fit_ratio)
    scale = min(avail_w / max(1, crop.shape[1]), avail_h / max(1, crop.shape[0]))
    if not np.isfinite(scale) or scale <= 0:
        return None, 0.0
    nw = max(1, int(round(crop.shape[1] * scale)))
    nh = max(1, int(round(crop.shape[0] * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
    alpha_small = np.clip(cv2.resize(crop, (nw, nh), interpolation=interp), 0.0, 1.0)

    cx, cy = (sx0 + sx1) * 0.5, (sy0 + sy1) * 0.5
    dx = int(round(cx - nw * 0.5)); dy = int(round(cy - nh * 0.5))
    dx = max(0, min(w - nw, dx)); dy = max(0, min(h - nh, dy))

    out = target.copy()
    tgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clear = clear_mask if clear_mask is not None else target_mask
    sel = clear > 0
    bright = (target_mask > 0) & (tgray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(target[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([255.0, 255.0, 255.0], np.float32)
    out[sel] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    roi_out = out[dy:dy + nh, dx:dx + nw].astype(np.float32)
    a = alpha_small[..., None]
    roi_out = roi_out * (1.0 - a)
    out[dy:dy + nh, dx:dx + nw] = np.clip(roi_out, 0, 255).astype(np.uint8)
    ink_ratio = float(np.count_nonzero(alpha_small > 0.10) / max(1, cv2.countNonZero(target_mask)))
    if ink_ratio < cfg.ink_min_ratio or ink_ratio > 0.36:
        return None, ink_ratio
    return out, ink_ratio

def _complex_text_ink_map(image: np.ndarray) -> np.ndarray:
    """Language-agnostic compact ink map for open/coloured text transfer."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    eq = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    ink = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10,
    )
    ink = cv2.bitwise_or(ink, (eq <= 150).astype(np.uint8) * 255)
    return ink

def _select_changed_text_components(
    ink: np.ndarray,
    unique_seed: np.ndarray,
    gate: np.ndarray,
    gap_px: int,
) -> np.ndarray:
    """Keep whole compact ink groups that contain language-change evidence.

    ``unique_seed`` contains strokes present only in this edition after a small
    tolerance.  We use it only to *select* a group, then restore all compact ink
    components belonging to that group so overlapping Chinese/Japanese strokes do
    not create broken glyphs.  Long panel/balloon/hair lines are discarded before
    grouping.
    """
    masked = ((ink > 0) & (gate > 0)).astype(np.uint8)
    if not np.any(masked):
        return np.zeros_like(ink, np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(masked, 8)
    ys, xs = np.where(gate > 0)
    if len(xs) == 0:
        return np.zeros_like(ink, np.uint8)
    gw = max(1, int(xs.max() - xs.min() + 1)); gh = max(1, int(ys.max() - ys.min() + 1))
    gate_area = max(1, int(np.count_nonzero(gate)))
    clean = np.zeros_like(ink, np.uint8)
    for i in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(40, int(0.12 * gate_area)):
            continue
        if cw > 0.64 * gw or ch > 0.64 * gh:
            # Allow a tall narrow text column / wide short caption, but not long
            # line-art structures spanning the candidate in both directions.
            if not ((ch <= 0.92 * gh and cw <= 0.22 * gw) or (cw <= 0.92 * gw and ch <= 0.24 * gh)):
                continue
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if aspect > 12.0:
            continue
        clean[labels == i] = 255
    if cv2.countNonZero(clean) == 0:
        return clean

    gap = max(2, int(gap_px))
    grouped = cv2.dilate(
        clean,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gap * 2 + 1, gap * 2 + 1)),
    )
    gn, glabels, _, _ = cv2.connectedComponentsWithStats((grouped > 0).astype(np.uint8), 8)
    out = np.zeros_like(clean)
    seed = (unique_seed > 0) & (gate > 0)
    for i in range(1, gn):
        region = glabels == i
        original = (clean > 0) & region
        pixels = int(np.count_nonzero(original))
        if pixels < 8:
            continue
        evidence = int(np.count_nonzero(seed & region))
        if evidence < max(2, int(round(0.015 * pixels))):
            continue
        out[original] = 255
    return out

def _soft_ink_alpha(image: np.ndarray, ink_mask: np.ndarray, gate: np.ndarray) -> np.ndarray:
    """Recover antialiased glyph opacity without copying the source background."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gvals = gray[gate > 0]
    ivals = gray[ink_mask > 0]
    if gvals.size == 0 or ivals.size == 0:
        return (ink_mask > 0).astype(np.float32)
    paper = float(np.percentile(gvals, 82.0))
    dark = float(np.percentile(ivals, 18.0))
    denom = max(20.0, paper - dark)
    tone = np.clip((paper - gray) / denom, 0.0, 1.0)
    support = cv2.GaussianBlur((ink_mask > 0).astype(np.float32), (0, 0), 0.65)
    return np.clip(np.maximum(tone * (ink_mask > 0), support * 0.72), 0.0, 1.0)

def _compact_container_ink(
    image: np.ndarray,
    gate: np.ndarray,
    threshold: int,
    cfg: MaskReplaceConfig,
    *,
    gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return compact glyph-like dark components inside a coloured container."""
    core = gate.copy()
    erode_px = max(0, int(getattr(cfg, "paired_diff_saturated_core_erode_px", 5)))
    if erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1))
        e = cv2.erode(core, k)
        if cv2.countNonZero(e) >= max(100, int(cv2.countNonZero(core) * 0.58)):
            core = e
    box = _bbox_from_mask(core)
    if box is None:
        return np.zeros_like(core), core
    x0, y0, x1, y1 = box
    gw, gh = max(1, x1 - x0), max(1, y1 - y0)
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    raw = ((gray <= int(threshold)) & (core > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    out = np.zeros_like(raw)
    gate_area = max(1, cv2.countNonZero(core))
    for i in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(600, int(0.025 * gate_area)):
            continue
        # Reject star outlines, panel rules and speed lines while keeping CJK
        # radicals/strokes.  Long components are accepted only when their other
        # axis remains narrow enough to be character-like.
        if (cw > 0.32 * gw and ch < 0.045 * gh) or (ch > 0.32 * gh and cw < 0.045 * gw):
            continue
        if cw > 0.45 * gw or ch > 0.45 * gh:
            if not ((ch < 0.75 * gh and cw < 0.12 * gw) or (cw < 0.75 * gw and ch < 0.12 * gh)):
                continue
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if aspect > 18.0:
            continue
        out[labels == i] = 255
    return out, core

def _target_edge_distance(gray: np.ndarray, edge_threshold: int = 175) -> np.ndarray:
    edges = cv2.Canny(gray, max(0, edge_threshold // 2), max(1, edge_threshold))
    return cv2.distanceTransform((edges == 0).astype(np.uint8), cv2.DIST_L2, 3)

def _expand_safe_write_mask(
    base_mask: np.ndarray,
    safe_envelope: np.ndarray,
    source_image: np.ndarray,
    target_image: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    max_px: int | None = None,
) -> tuple[np.ndarray, dict]:
    """Grow a bubble write mask a few pixels inside a safe target envelope.

    The target-driven path may intentionally inset the write mask to protect the
    outline. When SOURCE/TARGET container interiors differ slightly, that inset can
    clip Chinese edge strokes or leave a narrow ring where Japanese remnants stay
    visible. Grow only a few pixels, only inside ``safe_envelope``, and only away
    from strong target edges.
    """
    m = (base_mask > 0).astype(np.uint8)
    env = (safe_envelope > 0).astype(np.uint8)
    if not bool(getattr(cfg, 'mask_write_gap_fill_enabled', True)):
        return m * 255, {'enabled': False, 'iterations': 0, 'added_pixels': 0}
    if m.shape != env.shape or source_image.shape[:2] != m.shape or target_image.shape[:2] != m.shape:
        return (m * 255), {'enabled': True, 'shape_mismatch': True, 'iterations': 0, 'added_pixels': 0}
    max_px = max(0, int(max_px if max_px is not None else getattr(cfg, 'mask_write_gap_fill_max_px', 3)))
    if max_px <= 0:
        return m * 255, {'enabled': True, 'iterations': 0, 'added_pixels': 0}
    source_gray = cv2.cvtColor(source_image, cv2.COLOR_BGR2GRAY) if source_image.ndim == 3 else source_image.astype(np.uint8)
    target_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if target_image.ndim == 3 else target_image.astype(np.uint8)
    edge_dist = _target_edge_distance(target_gray)
    if source_image.ndim == 3:
        raw_support = (np.max(source_image, axis=2) > 4).astype(np.uint8)
        # Include black glyph cores that are surrounded by valid white/gray patch
        # pixels, but do not let zero-filled canvas outside a placed patch become
        # eligible merely because target Japanese ink is dark there.
        source_support = cv2.dilate(raw_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))) > 0
    else:
        source_support = np.ones_like(m, dtype=bool)
    src_white_thr = int(getattr(cfg, 'mask_write_gap_fill_source_white_threshold', 238))
    src_dark_thr = int(getattr(cfg, 'mask_write_gap_fill_source_dark_threshold', 205))
    tgt_dark_thr = int(getattr(cfg, 'mask_write_gap_fill_target_dark_threshold', 185))
    edge_floor = float(getattr(cfg, 'mask_write_gap_fill_target_edge_distance_px', 1.35))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = m.copy()
    before = int(np.count_nonzero(out))
    iterations = 0
    for _ in range(max_px):
        dil = cv2.dilate(out, kernel)
        ring = (dil > 0) & (out == 0) & (env > 0)
        if not np.any(ring):
            break
        edge_safe = (edge_dist >= edge_floor) | ((target_gray <= tgt_dark_thr) & (env > 0))
        allow = ring & source_support & edge_safe & (
            (source_gray >= src_white_thr) | (source_gray <= src_dark_thr) | (target_gray <= tgt_dark_thr)
        )
        added = int(np.count_nonzero(allow))
        if added == 0:
            break
        out[allow] = 1
        iterations += 1
    return (out * 255), {
        'enabled': True,
        'iterations': int(iterations),
        'added_pixels': int(max(0, int(np.count_nonzero(out)) - before)),
        'edge_floor_px': float(edge_floor),
    }

def _rigid_target_write_envelope(target_mask: np.ndarray, cfg: MaskReplaceConfig) -> np.ndarray:
    """Return the only writable part of a rigid target container.

    The container outline belongs to TARGET geometry.  Earlier versions eroded
    the patch once, but the later gap-fill stage used the full target mask as its
    safe envelope and could grow back over the protected outline.  Keep one
    canonical envelope so clear, patch, and gap-fill all obey the same border
    contract.
    """
    env = (target_mask > 0).astype(np.uint8) * 255
    if not bool(getattr(cfg, "rigid_container_full_patch_preserve_target_border", True)):
        return env
    inset = max(
        0,
        int(getattr(cfg, "rigid_container_target_inset_px", 1)),
        int(getattr(cfg, "rigid_container_full_patch_target_inset_px", 1)),
    )
    if inset <= 0:
        return env
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
    er = cv2.erode(env, k)
    return er if cv2.countNonZero(er) > 0 else env

def _compact_target_glyph_fringe(
    target: np.ndarray,
    target_mask: np.ndarray,
    *,
    target_gray: np.ndarray | None = None,
) -> np.ndarray:
    """Recover only character-sized dark components straddling a partial mask.

    Structural changed-text masks sometimes cut through the last Japanese column.
    Expanding the whole white region is unsafe (tails can connect to panel paper),
    so this helper adds only compact dark components that physically touch the
    trusted mask. Long balloon outlines, gutters and panel rules are rejected.
    """
    box = _bbox_from_mask(target_mask)
    out = np.zeros_like(target_mask)
    if box is None:
        return out
    x0, y0, x1, y1 = box; bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    gray = target_gray if target_gray is not None else cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    pad = max(4, int(round(0.12 * max(bw, bh))))
    near = cv2.dilate(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * pad + 1, 2 * pad + 1))) > 0
    dark = ((gray < 190) & near).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    touch = cv2.dilate(target_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    mask_area = max(1, cv2.countNonZero(target_mask))
    max_w = max(8, int(round(0.20 * bw))); max_h = max(8, int(round(0.20 * bh)))
    max_area = max(80, int(round(0.035 * mask_area)))
    for i in range(1, n):
        _, _, cw, ch, area = map(int, stats[i])
        if area < 4 or area > max_area or cw > max_w or ch > max_h:
            continue
        comp = labels == i
        if not np.any(comp & touch):
            continue
        # Only a straddling/near-edge component is useful here. Components fully
        # inside target_mask are already erased by the whole-container clear.
        outside = comp & (target_mask == 0)
        if not np.any(outside):
            continue
        cm = comp.astype(np.uint8) * 255
        cm = cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        out = np.maximum(out, cm)
    return out

__all__ = [
    "_target_white_ratio",
    "_alpha_from_mask",
    "_expand_target_clear_mask_with_text_components",
    "_extract_photo_text_cluster",
    "_fit_alpha_into_target_mask",
    "_reconstruct_photo_glyph_footprint_layer",
    "_reconstruct_photo_recentered_ink_layer",
    "_complex_text_ink_map",
    "_select_changed_text_components",
    "_soft_ink_alpha",
    "_compact_container_ink",
    "_target_edge_distance",
    "_expand_safe_write_mask",
    "_rigid_target_write_envelope",
    "_compact_target_glyph_fringe",
]
