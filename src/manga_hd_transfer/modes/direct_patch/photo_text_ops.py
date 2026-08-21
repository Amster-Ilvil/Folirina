from __future__ import annotations

"""Photographed-source text normalization/reconstruction helpers.

All output pixels are derived from already aligned SOURCE evidence and TARGET
paper; this module does not own region matching, mode routing or final compositing.
"""

import cv2
import numpy as np

from ...config import MaskReplaceConfig
from .geometry_ops import _bbox_from_mask
from .raster_primitives import _target_white_ratio

def _reconstruct_photo_crisp_layer(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    dest_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    allow_nonwhite_target: bool = False,
) -> tuple[np.ndarray | None, float]:
    """Recover photographed Chinese lettering as clean antialiased target-space ink.

    This is deliberately not OCR and does not invent glyphs. It estimates the
    source paper illumination, converts only locally-dark source detail into a
    soft ink alpha, removes tiny camera/JPEG speckles, guards a few pixels near
    the balloon boundary, clears the Japanese text on the clean target paper,
    then composites neutral black ink. Compared with direct photo-pixel pasting,
    it avoids blur, glare and duplicated balloon outlines; compared with hard
    binarization, it keeps antialiasing and does not thicken adjacent CJK strokes.
    """
    box = _bbox_from_mask(dest_mask)
    if box is None:
        return None, 0.0
    x0, y0, x1, y1 = box
    dmask = dest_mask[y0:y1, x0:x1] > 0
    pmask = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(dmask) < 80 or np.count_nonzero(pmask) < 35:
        return None, 0.0

    # Never derive "text" from the photographed balloon border. A small erosion
    # keeps genuine lettering while dropping the doubled-outline artifact seen in
    # phone-shot editions. Keep the full destination mask for clearing Japanese.
    guard = max(0, int(cfg.photo_pair_crisp_border_guard_px))
    if guard > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1))
        core = cv2.erode((pmask.astype(np.uint8) * 255), k) > 0
        # Very tiny balloons may collapse under erosion; retain the original mask.
        if np.count_nonzero(core) >= max(30, int(np.count_nonzero(pmask) * 0.55)):
            pmask = core

    tgt = target[y0:y1, x0:x1]
    target_mask_u8 = dmask.astype(np.uint8) * 255
    if (not allow_nonwhite_target
            and _target_white_ratio(tgt, target_mask_u8, cfg.ink_target_white_threshold) < max(0.70, cfg.ink_target_white_ratio)):
        return None, 0.0

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    work = gray.copy()
    paper_seed = int(np.percentile(gray[pmask], 90)) if np.any(pmask) else 245
    work[~pmask] = np.uint8(np.clip(paper_seed, 180, 255))

    side = max(9, min(x1 - x0, y1 - y0))
    ksize = int(np.clip(round(side * 0.20), 15, 55)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    bg = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (0, 0), max(1.0, ksize / 11.0))
    detail = np.maximum(bg.astype(np.float32) - gray.astype(np.float32), 0.0)

    # A gentle local-contrast branch recovers strokes affected by curved-page
    # illumination without turning broad gray shadows into ink.
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(6, 6))
    eq = clahe.apply(work)
    eq_bg = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel)
    eq_bg = cv2.GaussianBlur(eq_bg, (0, 0), max(1.0, ksize / 11.0))
    eq_detail = np.maximum(eq_bg.astype(np.float32) - eq.astype(np.float32), 0.0)
    detail = np.maximum(detail, eq_detail * 0.92)
    detail[~pmask] = 0.0

    floor = float(max(0.0, cfg.photo_pair_crisp_detail_floor))
    positive = detail[pmask & (detail > floor)]
    if positive.size < 16:
        return None, 0.0
    # Soft thresholds are the key difference from the older binary recovery.
    # Low contrast becomes antialiasing, strong strokes become black, and nearby
    # strokes do not get merged by morphological closing.
    lo = max(floor, float(np.percentile(positive, 18)) * 0.70)
    hi = max(lo + 8.0, float(np.percentile(positive, 88)) * 0.92)
    alpha = np.clip((detail - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    gamma = float(np.clip(cfg.photo_pair_crisp_alpha_gamma, 0.45, 1.40))
    alpha = np.power(alpha, gamma)

    amount = float(np.clip(cfg.photo_pair_crisp_unsharp_amount, 0.0, 1.5))
    if amount > 0:
        blur = cv2.GaussianBlur(alpha, (0, 0), 0.70)
        alpha = np.clip(alpha + amount * (alpha - blur), 0.0, 1.0)
    alpha[~pmask] = 0.0

    # Component cleanup is driven by a permissive seed, while the final edges
    # remain soft. This removes isolated photo grain without redrawing glyphs.
    seed = (alpha >= 0.16).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(seed, 8)
    keep = np.zeros_like(seed, np.uint8)
    min_area = max(1, int(cfg.photo_pair_crisp_min_component_area))
    max_area = max(12, int(np.count_nonzero(pmask) * cfg.photo_pair_crisp_max_ink_ratio))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            keep[labels == i] = 1
    alpha[keep == 0] = 0.0
    ink_ratio = float(np.count_nonzero(alpha >= 0.16) / max(1, np.count_nonzero(pmask)))
    if ink_ratio < cfg.ink_min_ratio or ink_ratio > cfg.photo_pair_crisp_max_ink_ratio:
        return None, ink_ratio

    clear_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    bright = dmask & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([252.0, 252.0, 252.0], np.float32)

    out = target.copy()
    roi = out[y0:y1, x0:x1].astype(np.float32)
    roi[dmask] = paper_bgr
    a = alpha[..., None]
    black = np.zeros_like(roi, np.float32)
    roi = roi * (1.0 - a) + black * a
    out[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)
    return out, ink_ratio


def _normalize_photo_text_pixels(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    dest_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> np.ndarray | None:
    """Flatten photographed bubble illumination while preserving real glyph pixels.

    Phone photos can be perfectly registered but still contain a smooth gray/blue
    glare field. Copying those pixels makes the clean target bubble look dirty,
    while hard thresholding destroys antialiasing and can merge small CJK strokes.
    This deterministic transform estimates the local paper illumination by a
    grayscale morphological closing, transfers only the *dark detail relative to
    that paper field*, and paints it over the clean target paper colour.

    No glyph is generated or inferred here: every dark detail comes from the
    registered source photograph. If the target is not a mostly-white text area,
    return None so the caller can use the normal pixel/ink/reletter policies.
    """
    box = _bbox_from_mask(dest_mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    dmask = dest_mask[y0:y1, x0:x1] > 0
    pmask = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(dmask) < 80 or np.count_nonzero(pmask) < 40:
        return None

    tgt = target[y0:y1, x0:x1]
    target_mask_u8 = dmask.astype(np.uint8) * 255
    if _target_white_ratio(tgt, target_mask_u8, cfg.ink_target_white_threshold) < max(0.72, cfg.ink_target_white_ratio):
        return None

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Outside the valid source region must not influence the estimated light field.
    work = gray.copy()
    if np.any(pmask):
        paper_seed = int(np.percentile(gray[pmask], 88))
    else:
        paper_seed = 245
    work[~pmask] = np.uint8(np.clip(paper_seed, 180, 255))

    side = max(9, min(x1 - x0, y1 - y0))
    # Kernel should be wider than typical glyph strokes/characters but remain
    # local enough to follow page glare and curved-book illumination gradients.
    k = int(np.clip(round(side * 0.22), 17, 61)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    background = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), max(1.2, k / 10.0))

    detail = np.maximum(background.astype(np.float32) - gray.astype(np.float32), 0.0)
    # Specular glare can turn one side of a glyph from black into mid-gray while
    # leaving the local paper even brighter. A CLAHE branch recovers that *local*
    # contrast without changing character topology; take the stronger evidence
    # from the raw and equalized views.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6, 6))
    eq = clahe.apply(work)
    eq_bg = cv2.morphologyEx(eq, cv2.MORPH_CLOSE, kernel)
    eq_bg = cv2.GaussianBlur(eq_bg, (0, 0), max(1.2, k / 10.0))
    eq_detail = np.maximum(eq_bg.astype(np.float32) - eq.astype(np.float32), 0.0)
    detail = np.maximum(detail, eq_detail * 1.12)
    # Ignore tiny photographic texture/compression residue so a clean Japanese
    # bubble does not inherit gray smudges from the phone photo.
    floor = float(max(0.0, cfg.photo_pair_normalize_detail_floor))
    if floor > 0:
        detail = np.maximum(detail - floor, 0.0)
    detail[~pmask] = 0.0
    positive = detail[pmask & (detail > 2.0)]
    if positive.size < 20:
        return None

    # Normalize contrast from the photographed page to a clean manga-paper range.
    # A percentile scale keeps antialiasing while preventing a few black cores from
    # making every other stroke too pale.
    p92 = float(np.percentile(positive, 92))
    scale = float(np.clip((225.0 / max(45.0, p92)) * cfg.photo_pair_normalize_contrast_gain, 1.0, 2.7))

    clear_gray = cv2.cvtColor(tgt, cv2.COLOR_BGR2GRAY)
    bright = dmask & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper_bgr = np.median(tgt[bright], axis=0).astype(np.float32)
    else:
        paper_bgr = np.array([250.0, 250.0, 250.0], np.float32)
    paper_luma = float(np.mean(paper_bgr))

    corrected = np.clip(paper_luma - detail * scale, 0, 255).astype(np.uint8)
    amount = float(np.clip(cfg.photo_pair_normalize_unsharp_amount, 0.0, 1.0))
    if amount > 0:
        blur = cv2.GaussianBlur(corrected, (0, 0), 0.75)
        corrected = cv2.addWeighted(corrected, 1.0 + amount, blur, -amount, 0)

    out = target.copy()
    # Clear the entire clean target interior first so Japanese glyphs disappear,
    # then place normalized source detail only where valid source pixels exist.
    roi = out[y0:y1, x0:x1]
    roi[dmask] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
    # Keep neutral antialiasing; source photo chroma/glare is intentionally dropped.
    norm_bgr = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
    roi[pmask] = norm_bgr[pmask]
    out[y0:y1, x0:x1] = roi
    return out


def _reconstruct_ink_layer(
    warped_img: np.ndarray,
    target: np.ndarray,
    paste_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    clear_mask: np.ndarray | None = None,
) -> tuple[np.ndarray | None, float]:
    """Recover crisp black glyphs from a soft photographed source bubble.

    This path never invents characters. It only binarizes the already-existing
    source ink after geometric alignment, cleans tiny camera/JPEG speckles,
    clears the target bubble interior to its local bright paper colour, and
    paints the recovered ink deterministically. It is intentionally limited to
    mostly-white speech/narration interiors.
    """
    box = _bbox_from_mask(paste_mask)
    if not box:
        return None, 0.0
    x0, y0, x1, y1 = box
    pm = paste_mask[y0:y1, x0:x1] > 0
    if np.count_nonzero(pm) < 40:
        return None, 0.0
    tgt = target[y0:y1, x0:x1]
    if _target_white_ratio(tgt, (pm.astype(np.uint8) * 255), cfg.ink_target_white_threshold) < cfg.ink_target_white_ratio:
        return None, 0.0

    src = warped_img[y0:y1, x0:x1]
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    # Photographed pages often have smooth illumination gradients. CLAHE makes
    # dark strokes locally separable without hallucinating lost details.
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    block = max(15, int(cfg.ink_adaptive_block_size) | 1)
    adaptive = cv2.adaptiveThreshold(
        eq, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        block, int(cfg.ink_adaptive_c),
    )
    # A conservative global threshold catches thick dark strokes that adaptive
    # thresholding can split under glare. Keep only pixels inside the target.
    vals = eq[pm]
    if vals.size:
        # Preserve antialiased/photographed glyph strokes, not only the darkest
        # core pixels. A local paper-relative threshold plus masked Otsu is much
        # more stable for small Chinese characters photographed slightly out of
        # focus (where the old threshold could collapse a glyph into black bars).
        q = float(np.percentile(vals, 32))
        global_ink = (eq <= min(185.0, q + 10.0)).astype(np.uint8) * 255
        paper_level = float(np.percentile(vals, 86))
        relative_thr = float(np.clip(paper_level - 26.0, 105.0, 210.0))
        relative_ink = (eq <= relative_thr).astype(np.uint8) * 255
        try:
            otsu_thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            otsu_thr = float(np.clip(otsu_thr + 10.0, 100.0, 205.0))
            otsu_ink = (eq <= otsu_thr).astype(np.uint8) * 255
        except cv2.error:
            otsu_ink = np.zeros_like(eq, np.uint8)
        ink = cv2.bitwise_or(adaptive, global_ink)
        ink = cv2.bitwise_or(ink, relative_ink)
        ink = cv2.bitwise_or(ink, otsu_ink)
    else:
        ink = adaptive
    ink[~pm] = 0

    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    cleaned = np.zeros_like(ink)
    max_area = max(8, int(np.count_nonzero(pm) * cfg.ink_max_component_area_ratio))
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if cfg.ink_min_component_area <= area <= max_area:
            cleaned[labels == i] = 255
    # One tiny close reconnects antialiased strokes after photographic blur.
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    cleaned[~pm] = 0
    ratio = cv2.countNonZero(cleaned) / max(1, int(np.count_nonzero(pm)))
    if ratio < cfg.ink_min_ratio or ratio > cfg.ink_max_ratio:
        return None, float(ratio)

    out = target.copy()
    # Estimate the clean target paper colour from the brightest pixels in the
    # destination interior, then remove Japanese glyphs deterministically.
    # For photographed pairs ``clear_mask`` may be larger than the valid source
    # mask: clear the whole clean target balloon, but only derive Chinese ink from
    # source pixels that actually belong to the matched source balloon.
    full_clear = clear_mask if clear_mask is not None else paste_mask
    cb = _bbox_from_mask(full_clear)
    if cb is None:
        return None, float(ratio)
    cx0, cy0, cx1, cy1 = cb
    clear_local = full_clear[cy0:cy1, cx0:cx1] > 0
    target_clear_crop = target[cy0:cy1, cx0:cx1]
    clear_gray = cv2.cvtColor(target_clear_crop, cv2.COLOR_BGR2GRAY)
    bright = clear_local & (clear_gray >= cfg.ink_target_white_threshold)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target_clear_crop[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    clear_roi = out[cy0:cy1, cx0:cx1]
    clear_roi[clear_local] = paper
    out[cy0:cy1, cx0:cx1] = clear_roi
    roi = out[y0:y1, x0:x1]
    roi[cleaned > 0] = (0, 0, 0)
    out[y0:y1, x0:x1] = roi
    return out, float(ratio)


__all__ = ['_reconstruct_photo_crisp_layer', '_normalize_photo_text_pixels', '_reconstruct_ink_layer']
