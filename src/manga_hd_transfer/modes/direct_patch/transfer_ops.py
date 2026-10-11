from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .geometry_ops import (
    BubblePatchMatch,
    _identity_like, _bbox_from_mask, _mask_iou, _target_coverage, _edge_touch_sides,
    _centroid, _ordered_offsets, _solidify_container_mask, _bubble_mask, _warp_mask,
    _shape_score, match_bubbles, _bbox_fit_matrix, _bbox_uniform_fit_matrix,
    _local_translation_ecc, _ocr_guided_region_gate,
)
from .selection_policy import (
    _publication_safety_enabled,
    _rigid_container_stats,
    _rigid_container_pair_eligible,
    _support_mask_textlike,
    _rigid_source_text_support,
)
from .raster_primitives import (
    _target_white_ratio, _alpha_from_mask, _expand_target_clear_mask_with_text_components,
    _extract_photo_text_cluster, _fit_alpha_into_target_mask, _reconstruct_photo_glyph_footprint_layer,
    _reconstruct_photo_recentered_ink_layer, _complex_text_ink_map, _select_changed_text_components,
    _soft_ink_alpha, _compact_container_ink, _target_edge_distance, _expand_safe_write_mask,
    _rigid_target_write_envelope, _compact_target_glyph_fringe,
)

from .quality_ops import _masked_sharpness, _pixel_enhance_text_raster, _normalize_bubble_background
from .warp_ops import _soft_mask_iou, _subpixel_translation_refine, _warp_source_patch, _shift, _photo_pair_salvage_warp
from .photo_text_ops import _reconstruct_photo_crisp_layer, _normalize_photo_text_pixels, _reconstruct_ink_layer
from .content_audit import _evaluate_content_completeness, _repair_content_region, finalize_transfer_records
from .transfer_models import MaskTransferRecord, MaskTransferResult
from .raster_policy import paired_proxy_geometry_risk
from ...external_command import run_external_command
from .text_transfer import target_text_mask_in_container, cleanup_target_residual_specks, clear_uniform_white_container_interior, white_container_paper_mask, white_container_write_envelope, target_container_border_mask, remove_container_boundary_line_components
from .source_clarity import enhance_white_source_patch as _enhance_white_source_patch
from ...config import MaskReplaceConfig
from ...geometry import polygon_bbox, transform_points, transform_to_homography
from ...io_utils import read_image, write_image
from ...models import BubbleInstance, RegistrationResult, TextUnit, UnitMatch















































def _dominant_saturated_container_mask(
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> np.ndarray | None:
    """Recover the flat-colour interior of a burst balloon near ``region_mask``.

    This route is deliberately colour-geometry only.  It is used for saturated
    yellow/red burst balloons where the ordinary white-container detector cannot
    provide an interior mask.  Text holes are closed by filling the selected
    colour component's *external* contour; panel/SFX colours with a different hue
    therefore stay outside the writable area.
    """
    if not bool(getattr(cfg, "paired_diff_saturated_container_enabled", True)):
        return None
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None
    h, w = region_mask.shape
    x0, y0, x1, y1 = box
    pad = max(12, int(round(max(x1 - x0, y1 - y0) * 0.10)))
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(w, x1 + pad), min(h, y1 + pad)
    local = np.zeros((h, w), np.uint8)
    local[ya:yb, xa:xb] = 255

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    val = hsv[..., 2]
    hue = hsv[..., 0]
    min_sat = int(getattr(cfg, "paired_diff_saturated_min_saturation", 72))
    min_val = int(getattr(cfg, "paired_diff_saturated_min_value", 160))
    near = cv2.dilate(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) > 0
    seed = (local > 0) & near & (sat >= min_sat) & (val >= min_val)
    hs = hue[seed]
    min_pixels = int(getattr(cfg, "paired_diff_saturated_min_pixels", 180))
    if hs.size < min_pixels:
        return None

    # Quantise hue into 30 bins (6 OpenCV hue units ~= 12 degrees) to survive
    # JPEG/halftone variation without accidentally merging a purple panel with a
    # yellow burst.  Hue distance is circular on [0, 180).
    hist = np.bincount((hs.astype(np.int32) // 6), minlength=30)
    best_bin = int(np.argmax(hist))
    dominant_fraction = float(hist[best_bin] / max(1, hs.size))
    if dominant_fraction < float(getattr(cfg, "paired_diff_saturated_min_dominant_fraction", 0.34)):
        return None
    center = best_bin * 6 + 3
    d1 = (hue.astype(np.int16) - center) % 180
    d2 = (center - hue.astype(np.int16)) % 180
    hue_dist = np.minimum(d1, d2)
    tol = int(getattr(cfg, "paired_diff_saturated_hue_tolerance", 10))
    colour = (
        (local > 0)
        & (hue_dist <= tol)
        & (sat >= max(48, min_sat - 12))
        & (val >= max(130, min_val - 10))
    ).astype(np.uint8) * 255
    close_px = max(1, int(getattr(cfg, "paired_diff_saturated_component_close_px", 4)))
    colour = cv2.morphologyEx(
        colour, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px * 2 + 1, close_px * 2 + 1)),
    )
    n, labels, stats, _ = cv2.connectedComponentsWithStats((colour > 0).astype(np.uint8), 8)
    rr = region_mask > 0
    best: tuple[float, np.ndarray] | None = None
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_pixels:
            continue
        comp = labels == i
        overlap = int(np.count_nonzero(comp & rr))
        region_area = max(1, int(np.count_nonzero(rr)))
        overlap_ratio = float(overlap / region_area)
        if overlap < max(12, int(0.015 * area)):
            continue
        if overlap_ratio < float(getattr(cfg, "paired_diff_saturated_min_region_overlap_ratio", 0.15)):
            continue
        score = float(overlap + 0.15 * area)
        if best is None or score > best[0]:
            best = (score, comp)
    if best is None:
        return None
    raw = best[1].astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    filled = np.zeros_like(raw)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    if cv2.countNonZero(filled) < min_pixels:
        return None
    return filled








def _transfer_saturated_text_container(
    aligned_source: np.ndarray,
    target: np.ndarray,
    colour_gate: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Clean a flat-colour burst and rebuild only the source Chinese glyph ink."""
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    thr = int(getattr(cfg, "paired_diff_saturated_text_dark_threshold", 182))
    src_sel, core = _compact_container_ink(aligned_source, colour_gate, thr, cfg)
    tgt_sel, _ = _compact_container_ink(target, colour_gate, thr, cfg)
    src_count = int(cv2.countNonZero(src_sel))
    tgt_count = int(cv2.countNonZero(tgt_sel))
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        return None, empty, empty, {"reason": "insufficient_saturated_source_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)):
        return None, empty, empty, {"reason": "insufficient_saturated_target_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clear = tgt_sel.copy()
    # v1.3: the compact target mask identifies glyph cores, but JPEG/print
    # antialiasing on saturated fills can be much brighter than the core threshold.
    # Admit only low-saturation darker pixels immediately around verified cores.
    aa_added = 0
    aa_r = max(0, int(getattr(cfg, "paired_diff_saturated_antialias_expand_px", 2)))
    if aa_r > 0 and cv2.countNonZero(clear) > 0:
        halo = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (aa_r * 2 + 1, aa_r * 2 + 1))) > 0
        bg_probe = (core > 0) & (clear == 0) & (hsv[..., 1] >= max(40, int(getattr(cfg, "paired_diff_saturated_min_saturation", 72)) - 18))
        bg_gray = float(np.median(gray[bg_probe])) if int(np.count_nonzero(bg_probe)) >= 32 else float(np.median(gray[core > 0]))
        contrast = max(4, int(getattr(cfg, "paired_diff_saturated_antialias_contrast", 8)))
        sat_max = int(getattr(cfg, "paired_diff_saturated_antialias_max_saturation", 96))
        fringe = halo & (core > 0) & (clear == 0) & (gray <= bg_gray - contrast) & (hsv[..., 1] <= sat_max)
        aa_added = int(np.count_nonzero(fringe))
        clear[fringe] = 255
    d = max(1, int(getattr(cfg, "paired_diff_saturated_clear_dilate_px", 2)))
    clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
    clear = cv2.bitwise_and(clear, core)

    bg_sel = (
        (colour_gate > 0)
        & (hsv[..., 1] >= max(40, int(getattr(cfg, "paired_diff_saturated_min_saturation", 72)) - 18))
        & (hsv[..., 2] >= int(getattr(cfg, "paired_diff_saturated_min_value", 160)))
        & (gray > thr + 18)
    )
    if np.count_nonzero(bg_sel) >= 80:
        paper_bgr = np.median(target[bg_sel], axis=0).astype(np.float32)
    else:
        paper_bgr = np.median(target[colour_gate > 0], axis=0).astype(np.float32)

    out = target.copy()
    # Flat bursts can be restored exactly with the median fill. Gradients/halftone
    # need local structure, so prefer Telea there instead of painting a flat patch.
    bg_pixels = target[bg_sel] if np.any(bg_sel) else target[colour_gate > 0]
    bg_std = float(np.mean(np.std(bg_pixels.astype(np.float32), axis=0))) if bg_pixels.size else 999.0
    flat_limit = float(getattr(cfg, "paired_diff_saturated_flat_std_threshold", 10.0))
    if bg_std <= flat_limit:
        out[clear > 0] = np.clip(paper_bgr, 0, 255).astype(np.uint8)
        clear_backend = "flat-median"
    else:
        out = cv2.inpaint(target, clear, float(getattr(cfg, "paired_diff_saturated_inpaint_radius", 3.0)), cv2.INPAINT_TELEA)
        clear_backend = "opencv-telea"
    alpha = _soft_ink_alpha(aligned_source, src_sel, core)
    # Saturated-container source pages in this route are monochrome scans.  Use
    # the source raster *opacity/topology* but neutral black ink; never composite
    # the source white background/halftone into the colour master.
    a3 = alpha[..., None]
    out = np.clip(out.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
    write = np.maximum(clear, (alpha * 255).astype(np.uint8))
    changed = np.any(out != target, axis=2)
    write[changed] = 255
    return out, write, src_sel, {
        "reason": "ok_saturated_container",
        "source_ink_pixels": src_count,
        "target_ink_pixels": tgt_count,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "antialias_added_pixels": int(aa_added),
        "clear_backend": clear_backend,
        "background_std": bg_std,
        "boundary_touch": False,
        "clear_mask": clear,
        "target_ink_mask": tgt_sel,
        "container_mask": colour_gate,
    }


def _source_guided_monochrome_recolor(
    aligned_source: np.ndarray,
    target: np.ndarray,
    container_mask: np.ndarray,
    target_clear_mask: np.ndarray,
    source_ink_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Rebuild a large coloured semantic-text hole from registered SOURCE luminance.

    This is intentionally narrow: it only activates for a low-saturation SOURCE
    against a saturated TARGET container, and it only writes the admitted TARGET
    invalidation footprint plus a small SOURCE-glyph halo.  TARGET pixels outside
    that footprint remain byte-identical.  The SOURCE raster is never used as a
    colour donor; its luminance/halftone structure is mapped onto chroma learned
    from intact TARGET background around the text.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "koharu_semantic_source_recolor_enabled", True)):
        return None, empty, {"reason": "source_recolor_disabled"}
    if any(x is None or x.shape != shape for x in (container_mask, target_clear_mask, source_ink_mask)):
        return None, empty, {"reason": "source_recolor_shape_mismatch"}
    clear_pixels = int(cv2.countNonZero(target_clear_mask))
    if clear_pixels < int(getattr(cfg, "koharu_semantic_source_recolor_min_clear_pixels", 2500)):
        return None, empty, {"reason": "source_recolor_clear_too_small", "clear_pixels": clear_pixels}

    container = container_mask > 0
    if int(np.count_nonzero(container)) < max(2000, clear_pixels * 2):
        return None, empty, {"reason": "source_recolor_container_too_small"}
    src_hsv = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2HSV)
    tgt_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    src_sat = src_hsv[..., 1][container]
    intact = container & ~(cv2.dilate(target_clear_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0)
    tgt_sat = tgt_hsv[..., 1][intact]
    if src_sat.size < 256 or tgt_sat.size < 256:
        return None, empty, {"reason": "source_recolor_insufficient_samples"}
    src_sat_p90 = float(np.percentile(src_sat, 90))
    tgt_sat_p50 = float(np.percentile(tgt_sat, 50))
    # HSV saturation becomes unstable on dark antialiased/halftone pixels: a
    # visually monochrome scan can report S~100 from only a few code values of
    # channel noise.  Accept the source as effectively monochrome when the
    # brighter structural pixels still have a small BGR channel spread.
    src_rgb = aligned_source[container]
    src_v = src_rgb.max(axis=1)
    bright = src_rgb[src_v >= 140]
    src_chroma_spread_p90 = 255.0
    if bright.shape[0] >= 256:
        spread = bright.max(axis=1).astype(np.float32) - bright.min(axis=1).astype(np.float32)
        src_chroma_spread_p90 = float(np.percentile(spread, 90))
    max_sat = float(getattr(cfg, "koharu_semantic_source_recolor_max_source_saturation_p90", 28.0))
    max_spread = float(getattr(cfg, "koharu_semantic_source_recolor_max_source_chroma_spread_p90", 30.0))
    if src_sat_p90 > max_sat and src_chroma_spread_p90 > max_spread:
        return None, empty, {
            "reason": "source_recolor_source_not_monochrome",
            "source_saturation_p90": src_sat_p90,
            "source_chroma_spread_p90": src_chroma_spread_p90,
        }
    if tgt_sat_p50 < float(getattr(cfg, "koharu_semantic_source_recolor_min_target_saturation_p50", 58.0)):
        return None, empty, {"reason": "source_recolor_target_not_saturated", "target_saturation_p50": tgt_sat_p50}

    src_lab = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2LAB)
    tgt_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # ``_complex_text_ink_map`` is deliberately sensitive and therefore also
    # sees halftone/radial texture on monochrome SOURCE pages.  Before a large
    # recolour, keep only dense connected glyph bodies near the admitted TARGET
    # text footprint.  This prevents starburst texture from becoming a giant
    # SOURCE overlay while retaining thick translated SFX strokes.
    raw_ink = (source_ink_mask > 0).astype(np.uint8)
    near_target = cv2.dilate(
        (target_clear_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31)),
    ) > 0
    ncc, labels, stats, _ = cv2.connectedComponentsWithStats(raw_ink, 8)
    dense_ink = np.zeros(shape, np.uint8)
    min_component_area = max(48, int(round(clear_pixels * 0.003)))
    kept_components = 0
    for comp_id in range(1, ncc):
        x, y, w, h, area = [int(v) for v in stats[comp_id]]
        if area < min_component_area or w <= 0 or h <= 0:
            continue
        fill_ratio = float(area / max(1, w * h))
        if fill_ratio < 0.30:
            continue
        comp = labels == comp_id
        overlap = float(np.count_nonzero(comp & near_target) / max(1, area))
        if overlap < 0.55:
            continue
        dense_ink[comp] = 255
        kept_components += 1
    if kept_components <= 0 or cv2.countNonZero(dense_ink) < min_component_area:
        return None, empty, {
            "reason": "source_recolor_no_dense_source_glyphs",
            "source_saturation_p90": src_sat_p90,
            "target_saturation_p50": tgt_sat_p50,
        }

    halo_px = max(2, int(getattr(cfg, "koharu_semantic_source_recolor_text_halo_px", 11)))
    halo = cv2.dilate(
        dense_ink,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_px * 2 + 1, halo_px * 2 + 1)),
    )
    source_bg = container & ~(halo > 0)
    target_bg = intact
    src_l = src_lab[..., 0][source_bg]
    tgt_l = tgt_lab[..., 0][target_bg]
    if src_l.size < 512 or tgt_l.size < 512:
        return None, empty, {"reason": "source_recolor_insufficient_luma_samples"}

    hs = np.bincount(src_l.astype(np.int32), minlength=256).astype(np.float64)
    ht = np.bincount(tgt_l.astype(np.int32), minlength=256).astype(np.float64)
    if hs.sum() <= 0 or ht.sum() <= 0:
        return None, empty, {"reason": "source_recolor_empty_histogram"}
    cs = np.cumsum(hs / hs.sum())
    ct = np.cumsum(ht / ht.sum())
    l_lut = np.array([min(255, int(np.searchsorted(ct, cs[i], side="left"))) for i in range(256)], dtype=np.uint8)
    mapped_l = l_lut[src_lab[..., 0]]

    # Learn TARGET chroma as a robust function of lightness.  A 7-level window
    # survives halftone/JPEG noise while preserving bright/dark radial bands.
    med_a = np.full(256, np.nan, dtype=np.float32)
    med_b = np.full(256, np.nan, dtype=np.float32)
    tgt_L = tgt_lab[..., 0]
    for level in range(256):
        lo, hi = max(0, level - 3), min(255, level + 3)
        sel = target_bg & (tgt_L >= lo) & (tgt_L <= hi)
        if int(np.count_nonzero(sel)) >= 48:
            med_a[level] = float(np.median(tgt_lab[..., 1][sel]))
            med_b[level] = float(np.median(tgt_lab[..., 2][sel]))
    good = np.isfinite(med_a) & np.isfinite(med_b)
    if int(np.count_nonzero(good)) < 8:
        return None, empty, {"reason": "source_recolor_insufficient_chroma_bins"}
    idx = np.arange(256, dtype=np.float32)
    med_a = np.interp(idx, idx[good], med_a[good]).astype(np.uint8)
    med_b = np.interp(idx, idx[good], med_b[good]).astype(np.uint8)
    recolor_lab = src_lab.copy()
    recolor_lab[..., 0] = mapped_l
    recolor_lab[..., 1] = med_a[mapped_l]
    recolor_lab[..., 2] = med_b[mapped_l]
    recolored = cv2.cvtColor(recolor_lab, cv2.COLOR_LAB2BGR)

    # LAB lightness matching can still yield near-neutral highlights when the
    # SOURCE scan contains white speed-line peaks.  A saturated TARGET burst has
    # no neutral paper inside the container, so clamp only low-saturation
    # recoloured background pixels to the intact TARGET hue/saturation envelope.
    target_sat_floor = int(np.percentile(tgt_sat, 20))
    hue_samples = tgt_hsv[..., 0][intact & (tgt_hsv[..., 1] >= max(48, target_sat_floor // 2))]
    dominant_hue = 0
    if hue_samples.size >= 64:
        hue_hist = np.bincount((hue_samples.astype(np.int32) // 6), minlength=30)
        dominant_hue = int(np.argmax(hue_hist) * 6 + 3)
        recolor_hsv = cv2.cvtColor(recolored, cv2.COLOR_BGR2HSV)
        low_sat = container & (recolor_hsv[..., 1] < target_sat_floor)
        recolor_hsv[..., 0][low_sat] = dominant_hue
        recolor_hsv[..., 1][low_sat] = target_sat_floor
        recolored = cv2.cvtColor(recolor_hsv, cv2.COLOR_HSV2BGR)

    # Rebuild only the admitted TARGET hole plus the SOURCE text support.  The
    # latter lets a translated glyph extend a little beyond the Japanese mask.
    replace = cv2.bitwise_or(
        cv2.dilate(target_clear_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))),
        halo,
    )
    replace = cv2.bitwise_and(replace, container_mask)
    out = target.copy()
    alpha = cv2.GaussianBlur((replace.astype(np.float32) / 255.0), (0, 0), 1.2)[..., None]
    out = np.clip(target.astype(np.float32) * (1.0 - alpha) + recolored.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

    # SOURCE glyph cores are neutral black/white artwork and should retain their
    # exact raster style.  The bounded halo carries the original white outline.
    text_alpha = cv2.GaussianBlur((halo.astype(np.float32) / 255.0), (0, 0), 0.6)[..., None]
    out = np.clip(out.astype(np.float32) * (1.0 - text_alpha) + aligned_source.astype(np.float32) * text_alpha, 0, 255).astype(np.uint8)
    write = np.zeros(shape, np.uint8)
    write[np.any(out != target, axis=2)] = 255
    return out, write, {
        "reason": "ok_source_guided_monochrome_recolor",
        "clear_backend": "source-guided-monochrome-recolor",
        "clear_pixels": clear_pixels,
        "write_pixels": int(cv2.countNonZero(write)),
        "source_saturation_p90": round(src_sat_p90, 3),
        "source_chroma_spread_p90": round(src_chroma_spread_p90, 3),
        "target_saturation_p50": round(tgt_sat_p50, 3),
        "source_text_halo_pixels": int(cv2.countNonZero(halo)),
        "source_dense_glyph_pixels": int(cv2.countNonZero(dense_ink)),
        "source_dense_glyph_components": int(kept_components),
        "target_saturation_floor": int(target_sat_floor),
        "target_dominant_hue": int(dominant_hue),
        "_source_dense_glyph_mask": dense_ink,
    }




def _raw_group_area_similarity(a: float, b: float) -> float:
    """Return a symmetric 0..1 scale-compatibility score for raw boxes."""
    a = max(1.0, float(a)); b = max(1.0, float(b))
    return float(min(a, b) / max(a, b))


def _plain_paper_text_components(
    image: np.ndarray,
    gate: np.ndarray,
    *,
    gray_max: int,
    min_value: int,
    max_saturation: int,
    min_area: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float]], np.ndarray]:
    """Return text-like dark components locally surrounded by bright paper.

    The support test deliberately uses a *ring* around each component rather
    than a rectangular white-region detector.  This allows slanted note/letter
    text near the paper edge while excluding the hand, page border and large
    artwork components that can share the same low saturation.
    """
    shape = gate.shape
    out = np.zeros(shape, np.uint8)
    if image.shape[:2] != shape or cv2.countNonZero(gate) <= 0:
        return out, np.zeros(shape, np.int32), np.zeros((1, 5), np.int32), [], out.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    paper = (
        (hsv[..., 2] >= int(min_value))
        & (hsv[..., 1] <= int(max_saturation))
        & (gate > 0)
    )
    # Select the dominant neutral-paper surface first.  Without this ownership
    # step, short black burst rays immediately outside a slanted sheet can have
    # "paper on one side" and be mistaken for glyph fragments.
    paper_u8 = paper.astype(np.uint8)
    paper_closed = cv2.morphologyEx(
        paper_u8,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    pn, plabels, pstats, _ = cv2.connectedComponentsWithStats(paper_closed, 8)
    paper_surface = np.zeros(shape, np.uint8)
    if pn > 1:
        best_id = max(range(1, pn), key=lambda i: int(pstats[i, cv2.CC_STAT_AREA]))
        paper_surface[plabels == best_id] = 255
        # Expand just far enough to cover the dark centre of normal CJK strokes.
        # Do *not* fill the external contour: on p-203 the white letter and an
        # overlapping white caption form a shape that can enclose a red wedge;
        # filling that contour would incorrectly grant write authority to the
        # red burst artwork.
        paper_surface = cv2.dilate(
            paper_surface, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        paper_surface = cv2.bitwise_and(paper_surface, gate)
    # Label dark components *before* clipping them to paper support.  Clipping
    # first can chop a long red-burst ray at the paper edge into a tiny fragment
    # that looks exactly like a glyph.  The full component geometry lets us
    # reject such outside artwork by paper-support fraction and aspect.
    candidate = ((gray <= int(gray_max)) & (gate > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    ys, xs = np.where(gate > 0)
    if xs.size == 0:
        return out, labels, stats, [], paper_surface
    bw = int(xs.max() - xs.min() + 1); bh = int(ys.max() - ys.min() + 1)
    gate_area = max(1, int(cv2.countNonZero(gate)))
    max_area = max(3000, int(round(gate_area * 0.03)))
    max_w = max(100, int(round(bw * 0.30)))
    max_h = max(140, int(round(bh * 0.30)))
    rows: list[dict[str, float]] = []
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    for comp_id in range(1, n):
        x, y, cw, ch, area = [int(v) for v in stats[comp_id]]
        if area < int(min_area) or area > max_area or cw < 1 or ch < 2:
            continue
        if cw > max_w or ch > max_h:
            continue
        fill = float(area / max(1, cw * ch))
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if fill < 0.015 or aspect > 22.0:
            continue
        component = (labels == comp_id).astype(np.uint8) * 255
        support_fraction = float(
            np.count_nonzero((component > 0) & (paper_surface > 0)) / max(1, area)
        )
        if support_fraction < 0.72:
            continue
        dilated = cv2.dilate(component, ring_kernel)
        ring = (dilated > 0) & (component == 0) & (gate > 0)
        ring_count = int(np.count_nonzero(ring))
        paper_ratio = float(np.mean(paper[ring])) if ring_count else 0.0
        # Tiny glyph fragments near the slanted paper boundary have paper on
        # only one side.  Larger components need stronger all-around paper
        # evidence so a hand, shirt or panel line is never admitted as text.
        paper_floor = 0.10 if area <= 180 and max(cw, ch) <= 45 else 0.28
        if paper_ratio < paper_floor:
            continue
        out[labels == comp_id] = 255
        rows.append({
            "id": float(comp_id), "area": float(area), "paper_ratio": paper_ratio,
            "paper_support_fraction": support_fraction,
            "x": float(x), "y": float(y), "w": float(cw), "h": float(ch),
        })
    return out, labels, stats, rows, paper_surface


def _select_changed_plain_paper_components(
    full_mask: np.ndarray,
    other_mask: np.ndarray,
    labels: np.ndarray,
    *,
    tolerance_px: int,
    min_unique_fraction: float,
) -> np.ndarray:
    """Keep whole glyph components that have at least a small edition-unique seed."""
    out = np.zeros(full_mask.shape, np.uint8)
    if cv2.countNonZero(full_mask) <= 0:
        return out
    tol = max(1, int(tolerance_px))
    near_other = cv2.dilate(
        (other_mask > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1)),
    )
    unique = cv2.bitwise_and(full_mask, cv2.bitwise_not(near_other))
    ids = np.unique(labels[full_mask > 0])
    for comp_id in ids:
        comp_id = int(comp_id)
        if comp_id <= 0:
            continue
        component = labels == comp_id
        area = int(np.count_nonzero(component))
        unique_px = int(np.count_nonzero((unique > 0) & component))
        if unique_px >= 2 and unique_px / max(1, area) >= float(min_unique_fraction):
            out[component] = 255
    return out


def _strict_raw_group_plain_paper_text_transfer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_text_gate: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Transfer a large paired text group on a bright neutral paper object.

    SOURCE/TARGET paper, borders and the object/hand holding the paper remain
    untouched.  Only changed dark glyph components supported by bright neutral
    paper are erased/written.  This covers slanted letter/note pages such as
    p-203 without turning the entire sheet into a rectangular SOURCE patch.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "koharu_semantic_raw_group_plain_paper_enabled", True)):
        return None, empty, empty, {"reason": "raw_group_plain_disabled"}
    if region_mask.shape != shape or cv2.countNonZero(region_mask) <= 0:
        return None, empty, empty, {"reason": "raw_group_plain_empty_region"}

    configured_pad = max(0, int(getattr(cfg, "koharu_semantic_raw_group_plain_boundary_pad_px", 12)))
    region_box = _bbox_from_mask(region_mask)
    if region_box is not None:
        rx0, ry0, rx1, ry1 = [int(v) for v in region_box]
        long_side = max(1, rx1 - rx0, ry1 - ry0)
    else:
        long_side = 1
    # Koharu raw boxes can stop at the first/last detected line on a slanted
    # letter.  Grow the *analysis* neighborhood (not write authority) enough to
    # recover edge lines.  Component admission still requires membership in the
    # dominant bright-neutral paper surface, so nearby red artwork/hand/captions
    # do not become text authority.
    adaptive_pad = min(112, int(round(long_side * 0.18)))
    pad = max(configured_pad, adaptive_pad)
    analysis_gate = region_mask.copy()
    if pad > 0:
        analysis_gate = cv2.dilate(
            analysis_gate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
        )
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    min_v = int(getattr(cfg, "koharu_semantic_raw_group_plain_min_value", 198))
    max_sat = int(getattr(cfg, "koharu_semantic_raw_group_plain_max_saturation", 84))
    core = region_mask > 0
    bright_ratio = float(np.mean((target_hsv[..., 2][core] >= min_v) & (target_hsv[..., 1][core] <= max_sat)))
    min_bright = float(getattr(cfg, "koharu_semantic_raw_group_plain_min_bright_ratio", 0.62))
    if bright_ratio < min_bright:
        return None, empty, empty, {
            "reason": "raw_group_plain_not_bright_paper",
            "bright_paper_ratio": round(bright_ratio, 6),
        }

    min_area = int(getattr(cfg, "koharu_semantic_raw_group_plain_component_min_area", 5))
    target_all, target_labels, _, target_rows, target_paper_surface = _plain_paper_text_components(
        target, analysis_gate,
        gray_max=int(getattr(cfg, "koharu_semantic_raw_group_plain_target_gray_max", 194)),
        min_value=min_v, max_saturation=max_sat, min_area=min_area,
    )
    source_gate = analysis_gate
    if isinstance(source_text_gate, np.ndarray) and source_text_gate.shape == shape and cv2.countNonZero(source_text_gate) > 0:
        source_gate = cv2.bitwise_and(analysis_gate, cv2.dilate(
            source_text_gate,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)) if pad > 0 else np.ones((1, 1), np.uint8),
        ))
    source_all, source_labels, _, source_rows, _source_paper_surface = _plain_paper_text_components(
        aligned_source, source_gate,
        gray_max=int(getattr(cfg, "koharu_semantic_raw_group_plain_source_gray_max", 188)),
        min_value=min_v, max_saturation=max_sat, min_area=min_area,
    )
    tol = int(getattr(cfg, "koharu_semantic_raw_group_plain_pair_tolerance_px", 2))
    unique_fraction = float(getattr(cfg, "koharu_semantic_raw_group_plain_component_unique_fraction", 0.06))
    target_core = _select_changed_plain_paper_components(
        target_all, source_all, target_labels,
        tolerance_px=tol, min_unique_fraction=unique_fraction,
    )
    source_core = _select_changed_plain_paper_components(
        source_all, target_all, source_labels,
        tolerance_px=tol, min_unique_fraction=unique_fraction,
    )
    # SOURCE monochrome editions can have black burst rays immediately outside
    # the slanted paper.  Even if a tiny ray fragment passes the component
    # heuristic, it must never gain write authority on TARGET coloured artwork.
    # Clip SOURCE glyph authority back to the TARGET-proven paper interior.
    paper_write_surface = cv2.erode(
        target_paper_surface, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    source_core = cv2.bitwise_and(source_core, paper_write_surface)
    tgt_px = int(cv2.countNonZero(target_core)); src_px = int(cv2.countNonZero(source_core))
    if src_px < 80 or tgt_px < 80:
        return None, empty, empty, {
            "reason": "raw_group_plain_insufficient_changed_glyphs",
            "source_stroke_pixels": src_px,
            "target_stroke_pixels": tgt_px,
            "source_components": len(source_rows),
            "target_components": len(target_rows),
            "bright_paper_ratio": round(bright_ratio, 6),
        }

    clear_px = max(2, int(getattr(cfg, "koharu_semantic_raw_group_plain_target_clear_px", 3)))
    clear = cv2.dilate(
        target_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clear_px * 2 + 1, clear_px * 2 + 1)),
    )
    # Keep the erase footprint on/near the proven paper field.  Dark glyph cores
    # themselves are included explicitly so their centres are not lost from the
    # bright-paper support test.
    clear = cv2.bitwise_and(clear, np.maximum(target_paper_surface, target_core))
    out = cv2.inpaint(target, clear, 3.0, cv2.INPAINT_TELEA)

    src_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    near = cv2.dilate(source_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = np.clip((245.0 - src_gray.astype(np.float32)) / 120.0, 0.0, 1.0)
    alpha *= ((near > 0) & (source_gate > 0)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.35)
    a3 = alpha[..., None]
    out = np.clip(
        out.astype(np.float32) * (1.0 - a3) + aligned_source.astype(np.float32) * a3,
        0, 255,
    ).astype(np.uint8)
    changed = np.any(out != target, axis=2)
    write = np.zeros(shape, np.uint8); write[changed] = 255
    return out, write, source_core, {
        "reason": "ok_raw_group_plain_paper_text_transfer",
        "clear_backend": "paired-raw-group-plain-paper-text",
        "clear_mask": clear,
        "target_ink_mask": target_core,
        "source_ink_pixels": src_px,
        "target_ink_pixels": tgt_px,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "write_pixels": int(cv2.countNonZero(write)),
        "source_components": int(len(source_rows)),
        "target_components": int(len(target_rows)),
        "bright_paper_ratio": round(bright_ratio, 6),
        "plain_paper": True,
        "boundary_touch": False,
    }


def _paired_text_raw_plain_paper_transfer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    raw_gate: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    paired_proof: dict[str, object],
    raw_gate_diag: dict[str, object],
    target_confidence: float,
    source_text_gate: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Safely rescue a high-confidence paired open-text raw box on plain paper.

    Koharu occasionally under-segments a multi-column text block to one narrow
    polygon while its raw box still spans the complete text group (Vol.04 p0048
    class).  Broad Telea remains forbidden.  Instead, only a strongly paired,
    high-confidence bilateral text group on a proven bright/neutral paper field
    may reuse the component-level plain-paper renderer.  Writes are clipped back
    to the paired raw-box union so surrounding artwork is byte-identical.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "koharu_semantic_paired_text_raw_plain_paper_enabled", True)):
        return None, empty, empty, {"reason": "paired_text_raw_plain_disabled"}
    if str(paired_proof.get("proof_mode") or "") != "paired_text" or not bool(paired_proof.get("admit", False)):
        return None, empty, empty, {"reason": "paired_text_raw_plain_requires_strong_pair"}
    if raw_gate.shape != shape or cv2.countNonZero(raw_gate) <= 0 or not bool(raw_gate_diag.get("expanded", False)):
        return None, empty, empty, {"reason": "paired_text_raw_plain_missing_raw_gate"}

    min_target_conf = float(getattr(cfg, "koharu_semantic_paired_text_raw_plain_min_target_confidence", 0.85))
    min_source_conf = float(getattr(cfg, "koharu_semantic_paired_text_raw_plain_min_source_confidence", 0.85))
    min_overlap = float(getattr(cfg, "koharu_semantic_paired_text_raw_plain_min_raw_overlap", 0.80))
    min_area = float(getattr(cfg, "koharu_semantic_paired_text_raw_plain_min_area_similarity", 0.60))
    max_page_ratio = float(getattr(cfg, "koharu_semantic_paired_text_raw_plain_max_page_area_ratio", 0.03))
    source_conf = float(paired_proof.get("source_confidence", 0.0) or 0.0)
    raw_overlap = float(raw_gate_diag.get("raw_box_overlap", 0.0) or 0.0)
    area_similarity = float(raw_gate_diag.get("raw_box_area_similarity", 0.0) or 0.0)
    raw_page_ratio = float(cv2.countNonZero(raw_gate) / max(1, shape[0] * shape[1]))
    if float(target_confidence) < min_target_conf or source_conf < min_source_conf:
        return None, empty, empty, {
            "reason": "paired_text_raw_plain_low_confidence",
            "target_confidence": round(float(target_confidence), 6),
            "source_confidence": round(source_conf, 6),
        }
    if raw_overlap < min_overlap or area_similarity < min_area or raw_page_ratio > max_page_ratio:
        return None, empty, empty, {
            "reason": "paired_text_raw_plain_geometry_rejected",
            "raw_box_overlap": round(raw_overlap, 6),
            "raw_box_area_similarity": round(area_similarity, 6),
            "raw_page_area_ratio": round(raw_page_ratio, 6),
        }

    out, write, source_core, diag = _strict_raw_group_plain_paper_text_transfer(
        aligned_source, target, raw_gate, cfg, source_text_gate=source_text_gate,
    )
    diag = dict(diag)
    if out is None or cv2.countNonZero(write) <= 0:
        diag.setdefault("reason", "paired_text_raw_plain_component_rejected")
        return None, empty, empty, diag

    # The plain-paper analyser intentionally uses a padded neighbourhood to
    # recover edge glyphs. For ordinary paired-text rescue the write authority
    # remains the paired raw-box union: never leak into adjacent artwork.
    outside = raw_gate == 0
    clipped = out.copy()
    clipped[outside] = target[outside]
    write = write.copy(); write[outside] = 0
    source_core = source_core.copy(); source_core[outside] = 0
    for key in ("clear_mask", "target_ink_mask"):
        mask = diag.get(key)
        if isinstance(mask, np.ndarray) and mask.shape == shape:
            mask = mask.copy(); mask[outside] = 0; diag[key] = mask
    diag.update({
        "reason": "ok_paired_text_raw_plain_paper_transfer",
        "clear_backend": "paired-raw-group-plain-paper-text",
        "paired_text_raw_plain_paper_rescue": True,
        "raw_page_area_ratio": round(raw_page_ratio, 6),
        "raw_box_overlap": round(raw_overlap, 6),
        "raw_box_area_similarity": round(area_similarity, 6),
        "target_confidence": round(float(target_confidence), 6),
        "source_confidence": round(source_conf, 6),
        "write_pixels": int(cv2.countNonZero(write)),
    })
    return clipped, write, source_core, diag


def _strict_raw_group_large_text_transfer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_text_gate: np.ndarray | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Transfer very large paired display text on a saturated TARGET background.

    This path is reachable only after the strict two-sided raw-group proof.  It
    exists for detector under-segmentation where the polygon covers only a tiny
    piece of a much larger title/SFX group (p-140 class).  TARGET dark glyphs are
    locally inpainted inside the paired raw-box union; SOURCE paper is never
    copied.  Registered SOURCE glyphs are composited by luminance-derived alpha.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None, empty, empty, {"reason": "raw_group_large_empty_region"}
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if bw < 48 or bh < 96:
        return None, empty, empty, {"reason": "raw_group_large_region_too_small"}

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat_vals = hsv[..., 1][region_mask > 0]
    if sat_vals.size < 128:
        return None, empty, empty, {"reason": "raw_group_large_insufficient_samples"}
    sat_p50 = float(np.percentile(sat_vals, 50))
    min_sat = float(getattr(cfg, "koharu_semantic_raw_group_large_min_target_saturation_p50", 80.0))
    if sat_p50 < min_sat:
        plain = _strict_raw_group_plain_paper_text_transfer(
            aligned_source, target, region_mask, cfg, source_text_gate=source_text_gate,
        )
        if plain[0] is not None:
            return plain
        diag = dict(plain[3])
        diag.setdefault("reason", "raw_group_large_target_not_saturated")
        diag["target_saturation_p50"] = round(sat_p50, 3)
        return None, empty, empty, diag

    source_gate = region_mask
    if isinstance(source_text_gate, np.ndarray) and source_text_gate.shape == shape and cv2.countNonZero(source_text_gate) > 0:
        source_gate = cv2.bitwise_and(source_text_gate, region_mask)
    source_box = _bbox_from_mask(source_gate)
    if source_box is None:
        return None, empty, empty, {"reason": "raw_group_large_missing_source_gate"}
    sx0, sy0, sx1, sy1 = [int(v) for v in source_box]
    sbw, sbh = max(1, sx1 - sx0), max(1, sy1 - sy0)

    src_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    src_gray_max = int(getattr(cfg, "koharu_semantic_raw_group_large_source_gray_max", 180))
    src_candidate = ((src_gray <= src_gray_max) & (source_gate > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src_candidate, 8)
    source_core = np.zeros(shape, np.uint8)
    source_min_area = int(getattr(cfg, "koharu_semantic_raw_group_large_source_component_min_area", 25))
    source_max_area = max(2000, int(round(max(1, cv2.countNonZero(source_gate)) * 0.20)))
    source_max_w = max(96, int(round(sbw * 1.05)))
    source_max_h = max(120, int(round(sbh * 0.30)))
    source_components = 0
    for comp_id in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[comp_id]]
        if area < source_min_area or area > source_max_area or cw < 2 or ch < 3:
            continue
        if cw > source_max_w or ch > source_max_h:
            continue
        fill = float(area / max(1, cw * ch))
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if fill < 0.06 or aspect > 18.0:
            continue
        source_core[labels == comp_id] = 255
        source_components += 1

    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    tgt_gray_max = int(getattr(cfg, "koharu_semantic_raw_group_large_target_gray_max", 175))
    tgt_candidate = ((tgt_gray <= tgt_gray_max) & (region_mask > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tgt_candidate, 8)
    target_core = np.zeros(shape, np.uint8)
    target_min_area = int(getattr(cfg, "koharu_semantic_raw_group_large_target_component_min_area", 35))
    target_max_area = max(4000, int(round(max(1, cv2.countNonZero(region_mask)) * 0.20)))
    target_max_w = max(120, int(round(bw * 0.65)))
    target_max_h = max(140, int(round(bh * 0.35)))
    target_components = 0
    for comp_id in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[comp_id]]
        if area < target_min_area or area > target_max_area or cw < 2 or ch < 2:
            continue
        if cw > target_max_w or ch > target_max_h:
            continue
        fill = float(area / max(1, cw * ch))
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if fill < 0.035 or aspect > 18.0:
            continue
        target_core[labels == comp_id] = 255
        target_components += 1

    src_px = int(cv2.countNonZero(source_core))
    tgt_px = int(cv2.countNonZero(target_core))
    region_px = max(1, int(cv2.countNonZero(region_mask)))
    if src_px < 180 or tgt_px < 180:
        return None, empty, empty, {
            "reason": "raw_group_large_insufficient_text_strokes",
            "source_stroke_pixels": src_px,
            "target_stroke_pixels": tgt_px,
            "source_components": source_components,
            "target_components": target_components,
        }
    if src_px / region_px > 0.30 or tgt_px / region_px > 0.35:
        return None, empty, empty, {
            "reason": "raw_group_large_strokes_too_dense",
            "source_stroke_pixels": src_px,
            "target_stroke_pixels": tgt_px,
        }

    clear_px = max(2, int(getattr(cfg, "koharu_semantic_raw_group_large_target_clear_px", 4)))
    clear = cv2.dilate(
        target_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clear_px * 2 + 1, clear_px * 2 + 1)),
    )
    clear = cv2.bitwise_and(clear, region_mask)
    radius = float(getattr(cfg, "koharu_semantic_raw_group_large_inpaint_radius", 5.0))
    out = cv2.inpaint(target, clear, radius, cv2.INPAINT_TELEA)

    # Recover antialiased SOURCE glyph edges without copying the white SOURCE
    # paper. Only pixels near proven source components can contribute alpha.
    near = cv2.dilate(source_core, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = np.clip((245.0 - src_gray.astype(np.float32)) / 120.0, 0.0, 1.0)
    alpha *= ((near > 0) & (source_gate > 0)).astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.4)
    a3 = alpha[..., None]
    out = np.clip(
        out.astype(np.float32) * (1.0 - a3) + aligned_source.astype(np.float32) * a3,
        0, 255,
    ).astype(np.uint8)

    changed = np.any(out != target, axis=2)
    write = np.zeros(shape, np.uint8)
    write[changed] = 255
    return out, write, source_core, {
        "reason": "ok_raw_group_large_text_transfer",
        "clear_backend": "paired-raw-group-large-text-inpaint",
        "clear_mask": clear,
        "target_ink_mask": target_core,
        "source_ink_pixels": src_px,
        "target_ink_pixels": tgt_px,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "write_pixels": int(cv2.countNonZero(write)),
        "source_components": int(source_components),
        "target_components": int(target_components),
        "source_gray_max": int(src_gray_max),
        "target_gray_max": int(tgt_gray_max),
        "target_saturation_p50": round(sat_p50, 3),
        "target_clear_px": int(clear_px),
        "raw_group_large_typography": True,
        "boundary_touch": False,
    }

def _raw_box_blackhat_text_transfer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_text_gate: np.ndarray | None = None,
    large_raw_group: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Transfer paired open text inside a detector raw-box without copying SOURCE paper.

    This route is intentionally narrow.  It is used only after SOURCE/TARGET
    Koharu text boxes have already been paired and the polygon is proven badly
    under-segmented.  A blurred black-hat transform suppresses manga screen tone
    while retaining the much larger CJK strokes, so both the translated SOURCE
    glyphs and the original TARGET glyphs can be recovered across the full raw
    box.  TARGET text is removed locally; SOURCE background is never copied.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None, empty, empty, {"reason": "raw_blackhat_empty_region"}
    x0, y0, x1, y1 = [int(v) for v in box]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    if bw < 24 or bh < 48:
        return None, empty, empty, {"reason": "raw_blackhat_region_too_small"}

    # A strict bilateral raw-group has already passed registration, scale, overlap
    # and edition-change proof.  Let its specialised renderer decide whether the
    # TARGET is saturated artwork or a bright neutral paper object.  Applying the
    # ordinary "must be coloured" gate first made the plain-paper branch
    # unreachable on real letter/note pages (p-203 class).
    if large_raw_group:
        return _strict_raw_group_large_text_transfer(
            aligned_source, target, region_mask, cfg, source_text_gate=source_text_gate
        )

    # Ordinary raw-box rescue remains restricted to coloured artwork.  White
    # speech balloons and plain paper have safer container-specific paths.
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat_vals = hsv[..., 1][region_mask > 0]
    if sat_vals.size < 128:
        return None, empty, empty, {"reason": "raw_blackhat_insufficient_samples"}
    sat_p50 = float(np.percentile(sat_vals, 50))
    if sat_p50 < float(getattr(cfg, "koharu_semantic_raw_blackhat_min_target_saturation_p50", 48.0)):
        return None, empty, empty, {
            "reason": "raw_blackhat_target_not_coloured",
            "target_saturation_p50": round(sat_p50, 3),
        }

    # p-133 class: the detector polygon is incomplete but paired raw boxes are
    # reliable.  Do not run a generic black-hat over the union rectangle: manga
    # screen tone is also high-frequency and becomes false SOURCE ink.  Extract
    # SOURCE/TARGET strokes with evidence appropriate to each rendition.
    source_gate = region_mask
    if isinstance(source_text_gate, np.ndarray) and source_text_gate.shape == shape and cv2.countNonZero(source_text_gate) > 0:
        source_gate = cv2.bitwise_and(source_text_gate, region_mask)

    src_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    src_dark_max = int(getattr(cfg, "koharu_semantic_raw_source_core_gray_max", 105))
    src_candidate = ((src_gray <= src_dark_max) & (source_gate > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(src_candidate, 8)
    source_core = np.zeros(shape, np.uint8)
    source_gate_area = max(1, int(cv2.countNonZero(source_gate)))
    src_min_area = max(12, min(45, int(round(source_gate_area * 0.0015))))
    src_max_area = max(320, int(round(source_gate_area * 0.075)))
    src_components = 0
    for comp_id in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[comp_id]]
        if area < src_min_area or area > src_max_area or cw < 3 or ch < 5:
            continue
        if cw > 64 or ch > 72:
            continue
        fill = float(area / max(1, cw * ch))
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if fill < 0.12 or aspect > 9.0:
            continue
        source_core[labels == comp_id] = 255
        src_components += 1

    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    tgt_sat = hsv[..., 1]
    tgt_sat_max = int(getattr(cfg, "koharu_semantic_raw_target_core_saturation_max", 80))
    tgt_gray_max = int(getattr(cfg, "koharu_semantic_raw_target_core_gray_max", 135))
    tgt_candidate = ((tgt_sat <= tgt_sat_max) & (tgt_gray <= tgt_gray_max) & (region_mask > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(tgt_candidate, 8)
    target_core = np.zeros(shape, np.uint8)
    region_area = max(1, int(cv2.countNonZero(region_mask)))
    tgt_min_area = max(10, min(18, int(round(region_area * 0.0004))))
    tgt_max_area = max(260, int(round(region_area * 0.060)))
    tgt_components = 0
    for comp_id in range(1, n):
        _x, _y, cw, ch, area = [int(v) for v in stats[comp_id]]
        if area < tgt_min_area or area > tgt_max_area or cw < 2 or ch < 4:
            continue
        if cw > 58 or ch > 78:
            continue
        fill = float(area / max(1, cw * ch))
        aspect = max(cw / max(1.0, ch), ch / max(1.0, cw))
        if fill < 0.07 or aspect > 12.0:
            continue
        target_core[labels == comp_id] = 255
        tgt_components += 1

    src_px = int(cv2.countNonZero(source_core))
    tgt_px = int(cv2.countNonZero(target_core))
    region_px = max(1, int(cv2.countNonZero(region_mask)))
    min_src = int(getattr(cfg, "koharu_semantic_raw_blackhat_min_source_pixels", 180))
    min_tgt = int(getattr(cfg, "koharu_semantic_raw_blackhat_min_target_pixels", 120))
    if src_px < min_src or tgt_px < min_tgt:
        return None, empty, empty, {
            "reason": "raw_blackhat_insufficient_text_strokes",
            "source_stroke_pixels": src_px,
            "target_stroke_pixels": tgt_px,
        }
    if src_px / region_px > 0.38 or tgt_px / region_px > 0.38:
        return None, empty, empty, {
            "reason": "raw_blackhat_strokes_too_dense",
            "source_stroke_pixels": src_px,
            "target_stroke_pixels": tgt_px,
        }

    # Dilate the detected Japanese black core only enough to remove the printed
    # white outline.  TELEA performed substantially better than NS on the
    # coloured halftone p-133 background and keeps the repair local.
    clear_radius = max(2, int(getattr(cfg, "koharu_semantic_raw_target_outline_clear_px", 5)))
    clear = cv2.dilate(
        target_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clear_radius * 2 + 1, clear_radius * 2 + 1)),
    )
    clear = cv2.bitwise_and(clear, region_mask)
    inpaint_radius = float(getattr(cfg, "koharu_semantic_raw_blackhat_inpaint_radius", 3.0))
    out = cv2.inpaint(target, clear, inpaint_radius, cv2.INPAINT_TELEA)

    # Restore the SOURCE publication style using only proven text support.  A
    # narrow bright ring carries the white outline; SOURCE screen tone/paper is
    # never admitted because pixels farther than a few source-stroke pixels away
    # are excluded.
    src_hsv = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2HSV)
    outline_k = max(3, int(getattr(cfg, "koharu_semantic_raw_blackhat_outline_kernel", 7)))
    if outline_k % 2 == 0:
        outline_k += 1
    outline_zone = cv2.dilate(
        source_core,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outline_k, outline_k)),
    )
    bright_floor = int(getattr(cfg, "koharu_semantic_raw_blackhat_outline_brightness", 178))
    outline = (
        (outline_zone > 0)
        & (source_core == 0)
        & (src_gray >= bright_floor)
        & (src_hsv[..., 1] < 120)
        & (source_gate > 0)
    ).astype(np.uint8) * 255
    outline_alpha = cv2.GaussianBlur(outline.astype(np.float32) / 255.0, (0, 0), 0.30)[..., None]
    out = np.clip(
        out.astype(np.float32) * (1.0 - outline_alpha) + 255.0 * outline_alpha,
        0, 255,
    ).astype(np.uint8)
    core_alpha = cv2.GaussianBlur(source_core.astype(np.float32) / 255.0, (0, 0), 0.30)[..., None]
    out = np.clip(
        out.astype(np.float32) * (1.0 - core_alpha)
        + aligned_source.astype(np.float32) * core_alpha,
        0, 255,
    ).astype(np.uint8)

    changed = np.any(out != target, axis=2)
    write = np.zeros(shape, np.uint8)
    write[changed] = 255
    return out, write, source_core, {
        "reason": "ok_raw_box_blackhat_text_transfer",
        "clear_backend": "paired-raw-box-blackhat-inpaint",
        "clear_mask": clear,
        "target_ink_mask": target_core,
        "source_ink_pixels": src_px,
        "target_ink_pixels": tgt_px,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "write_pixels": int(cv2.countNonZero(write)),
        "outline_pixels": int(cv2.countNonZero(outline)),
        "target_saturation_p50": round(sat_p50, 3),
        "source_core_components": int(src_components),
        "target_core_components": int(tgt_components),
        "source_core_gray_max": int(src_dark_max),
        "target_core_saturation_max": int(tgt_sat_max),
        "target_core_gray_max": int(tgt_gray_max),
        "target_outline_clear_px": int(clear_radius),
        "boundary_touch": False,
        "raw_box_rescue": True,
    }


def _paired_source_label_raster_transfer(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    paired_proof: dict[str, object] | None = None,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Transfer a compact SOURCE-only paper/label behind proven paired open text.

    Some cross-rendition manga pages print the translated SOURCE lettering on a
    small white backing label while the TARGET edition prints Japanese directly
    over artwork. Ink-only transfer is structurally incapable of reproducing that
    SOURCE representation: it either leaves Japanese ghosts or produces black
    Chinese glyphs on the wrong background.  This route is deliberately much
    narrower than normal semantic recovery:

    * the SOURCE/TARGET text pair must already have passed the bilateral
      ``lowconf_paired_raw_text`` proof;
    * a compact dense bright/neutral rectangle must be present only in SOURCE;
    * TARGET clear authority comes from repeated text-component clusters inside
      the proven raw group, while long decorative chains and isolated stars are
      preserved;
    * only the target clear footprint and the detected SOURCE label rectangle are
      writable. Everything else stays byte-identical.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    proof = dict(paired_proof or {})
    if not bool(getattr(cfg, "koharu_semantic_source_label_raster_enabled", True)):
        return None, empty, empty, {"reason": "source_label_raster_disabled"}
    if str(proof.get("proof_mode") or "") != "lowconf_paired_raw_text" or not bool(proof.get("admit", False)):
        return None, empty, empty, {"reason": "source_label_requires_strict_paired_text_proof"}
    if float(proof.get("raw_box_overlap", 0.0) or 0.0) < float(getattr(cfg, "koharu_semantic_source_label_min_raw_overlap", 0.90)):
        return None, empty, empty, {"reason": "source_label_raw_overlap_too_low"}
    if float(proof.get("raw_box_area_similarity", 0.0) or 0.0) < float(getattr(cfg, "koharu_semantic_source_label_min_area_similarity", 0.60)):
        return None, empty, empty, {"reason": "source_label_raw_scale_mismatch"}
    if region_mask.shape != shape or cv2.countNonZero(region_mask) <= 0:
        return None, empty, empty, {"reason": "source_label_empty_raw_gate"}

    raw_box = _bbox_from_mask(region_mask)
    if raw_box is None:
        return None, empty, empty, {"reason": "source_label_missing_raw_bbox"}
    rx0, ry0, rx1, ry1 = [int(v) for v in raw_box]
    rw, rh = max(1, rx1-rx0), max(1, ry1-ry0)
    raw_area = max(1, rw*rh)

    # Analyse a small halo around the raw group because Koharu raw boxes can end
    # at the final glyph while a SOURCE-only label extends a few pixels farther.
    padx = max(8, min(32, int(round(rw * 0.14))))
    pady = max(10, min(42, int(round(rh * 0.16))))
    ax0=max(0,rx0-padx); ay0=max(0,ry0-pady); ax1=min(shape[1],rx1+padx); ay1=min(shape[0],ry1+pady)
    src_roi = aligned_source[ay0:ay1, ax0:ax1]
    tgt_roi = target[ay0:ay1, ax0:ax1]
    if src_roi.size == 0 or tgt_roi.size == 0:
        return None, empty, empty, {"reason": "source_label_empty_analysis_roi"}

    src_hsv = cv2.cvtColor(src_roi, cv2.COLOR_BGR2HSV)
    src_gray = cv2.cvtColor(src_roi, cv2.COLOR_BGR2GRAY)
    tgt_hsv = cv2.cvtColor(tgt_roi, cv2.COLOR_BGR2HSV)
    bright_v = int(getattr(cfg, "koharu_semantic_source_label_bright_value", 220))
    max_sat = int(getattr(cfg, "koharu_semantic_source_label_max_saturation", 48))
    bright = ((src_hsv[...,2] >= bright_v) & (src_hsv[...,1] <= max_sat)).astype(np.float32)
    k = max(5, int(getattr(cfg, "koharu_semantic_source_label_dense_window_px", 13)))
    if k % 2 == 0:
        k += 1
    dense_fraction = float(getattr(cfg, "koharu_semantic_source_label_dense_fraction", 0.58))
    local_fraction = cv2.boxFilter(bright, -1, (k,k), normalize=True)
    dense = (local_fraction >= dense_fraction).astype(np.uint8)
    dense = cv2.morphologyEx(dense, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dense, 8)

    min_area = int(getattr(cfg, "koharu_semantic_source_label_component_min_area", 450))
    min_w = int(getattr(cfg, "koharu_semantic_source_label_component_min_width", 24))
    min_h = int(getattr(cfg, "koharu_semantic_source_label_component_min_height", 28))
    bbox_pad = max(0, int(getattr(cfg, "koharu_semantic_source_label_bbox_pad_px", 3)))
    min_bright = float(getattr(cfg, "koharu_semantic_source_label_min_bright_ratio", 0.34))
    min_gap = float(getattr(cfg, "koharu_semantic_source_label_min_bright_gap", 0.16))
    min_dark = float(getattr(cfg, "koharu_semantic_source_label_min_dark_ratio", 0.06))
    max_area_fraction = float(getattr(cfg, "koharu_semantic_source_label_max_area_fraction", 0.65))
    best: tuple[float, tuple[int,int,int,int], dict[str,float]] | None = None
    raw_local = np.zeros(dense.shape, np.uint8)
    raw_local[max(0,ry0-ay0):min(dense.shape[0],ry1-ay0), max(0,rx0-ax0):min(dense.shape[1],rx1-ax0)] = 1
    for comp_id in range(1,n):
        x,y,w,h,area=[int(v) for v in stats[comp_id]]
        if area < min_area or w < min_w or h < min_h:
            continue
        aspect=max(w/max(1.0,h), h/max(1.0,w))
        if aspect > 4.0:
            continue
        comp=(labels==comp_id)
        raw_overlap=float(np.count_nonzero(comp & (raw_local>0))/max(1,area))
        if raw_overlap < 0.60:
            continue
        lx0=max(0,x-bbox_pad); ly0=max(0,y-bbox_pad); lx1=min(dense.shape[1],x+w+bbox_pad); ly1=min(dense.shape[0],y+h+bbox_pad)
        area_fraction=float(((lx1-lx0)*(ly1-ly0))/raw_area)
        if area_fraction > max_area_fraction or area_fraction < 0.035:
            continue
        sh=src_hsv[ly0:ly1,lx0:lx1]; th=tgt_hsv[ly0:ly1,lx0:lx1]; sg=src_gray[ly0:ly1,lx0:lx1]
        if sh.size == 0:
            continue
        sbright=float(np.mean((sh[...,2]>=bright_v)&(sh[...,1]<=max_sat)))
        tbright=float(np.mean((th[...,2]>=bright_v)&(th[...,1]<=max_sat)))
        dark_ratio=float(np.mean(sg < 165))
        if sbright < min_bright or (sbright-tbright) < min_gap or dark_ratio < min_dark:
            continue
        score=float(area * (sbright-tbright) * (0.5 + min(0.5,dark_ratio)))
        diag={"source_bright_ratio":sbright,"target_bright_ratio":tbright,"dark_ratio":dark_ratio,
              "dense_component_area":float(area),"raw_overlap":raw_overlap,"area_fraction":area_fraction}
        if best is None or score > best[0]:
            best=(score,(ax0+lx0,ay0+ly0,ax0+lx1,ay0+ly1),diag)
    if best is None:
        return None, empty, empty, {"reason":"source_label_not_proven"}
    _, label_box, label_diag = best
    lx0,ly0,lx1,ly1=label_box

    # TARGET open lettering is identified by repeated compact bright-outline
    # components.  Long narrow chains (decorative beads/rules) and isolated
    # stars are intentionally excluded.  Vertical and horizontal text clusters
    # are both supported; a qualifying cluster needs multiple components and a
    # meaningful span along its writing axis.
    tgt_raw = target[ry0:ry1,rx0:rx1]
    raw_gray = cv2.cvtColor(tgt_raw, cv2.COLOR_BGR2GRAY)
    outline_value = int(getattr(cfg, "koharu_semantic_source_label_target_outline_value", 180))
    outlines=(raw_gray>=outline_value).astype(np.uint8)
    outlines=cv2.morphologyEx(outlines,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)))
    cn, clabels, cstats, _ = cv2.connectedComponentsWithStats(outlines,8)
    comp_min=max(20,int(getattr(cfg,"koharu_semantic_source_label_target_component_min_area",90)))
    comps=[]
    for cid in range(1,cn):
        x,y,w,h,area=[int(v) for v in cstats[cid]]
        if area<comp_min or w<6 or h<6:
            continue
        # Long raw-box edge chains/rules are common decoration, not lettering.
        if (h>=0.60*rh and w<=0.30*rw) or (w>=0.60*rw and h<=0.30*rh):
            continue
        if x<=1 or x+w>=rw-1:
            continue
        if w>0.48*rw or h>0.58*rh:
            continue
        comps.append({"id":cid,"x":x,"y":y,"w":w,"h":h,"area":area,
                      "cx":x+w/2.0,"cy":y+h/2.0})
    if len(comps)<2:
        return None, empty, empty, {"reason":"source_label_no_target_text_cluster","label_box":list(label_box)}

    center_ratio=float(getattr(cfg,"koharu_semantic_source_label_target_cluster_center_ratio",0.15))
    span_ratio=float(getattr(cfg,"koharu_semantic_source_label_target_cluster_span_ratio",0.24))
    def _cluster(rows, axis):
        key="cx" if axis=="vertical" else "cy"; limit=max(10.0,(rw if axis=="vertical" else rh)*center_ratio)
        clusters=[]
        for row in sorted(rows,key=lambda r:r[key]):
            attached=False
            for cluster in clusters:
                center=float(np.mean([r[key] for r in cluster]))
                if abs(float(row[key])-center)<=limit:
                    cluster.append(row); attached=True; break
            if not attached:
                clusters.append([row])
        accepted=[]
        for cluster in clusters:
            if len(cluster)<2: continue
            if axis=="vertical":
                lo=min(r["y"] for r in cluster); hi=max(r["y"]+r["h"] for r in cluster); span=hi-lo; need=rh*span_ratio
            else:
                lo=min(r["x"] for r in cluster); hi=max(r["x"]+r["w"] for r in cluster); span=hi-lo; need=rw*span_ratio
            if span>=need and sum(r["area"] for r in cluster)>=comp_min*2:
                accepted.extend(cluster)
        return accepted
    selected_by_id={r["id"]:r for r in (_cluster(comps,"vertical")+_cluster(comps,"horizontal"))}
    selected=list(selected_by_id.values())
    if not selected:
        return None, empty, empty, {"reason":"source_label_target_clusters_unproven","label_box":list(label_box)}

    target_core=np.zeros(shape,np.uint8)
    local_core=target_core[ry0:ry1,rx0:rx1]
    for row in selected:
        local_core[clabels==int(row["id"])]=255
    dilate_px=max(1,int(getattr(cfg,"koharu_semantic_source_label_target_clear_dilate_px",3)))
    clear=cv2.dilate(target_core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(dilate_px*2+1,dilate_px*2+1)))
    clear=cv2.bitwise_and(clear,region_mask)
    if cv2.countNonZero(clear)<comp_min:
        return None, empty, empty, {"reason":"source_label_target_clear_too_small","label_box":list(label_box)}

    out=cv2.inpaint(target,clear,float(getattr(cfg,"koharu_semantic_source_label_target_inpaint_radius",3.0)),cv2.INPAINT_TELEA)
    # Copy only the proven compact label. This intentionally carries SOURCE paper
    # and antialiased Chinese glyphs because the paper is part of the translated
    # edition's text presentation, while artwork outside this rectangle remains
    # TARGET-native.
    out[ly0:ly1,lx0:lx1]=aligned_source[ly0:ly1,lx0:lx1]
    write=np.zeros(shape,np.uint8); write[np.any(out!=target,axis=2)]=255
    label_mask=np.zeros(shape,np.uint8); label_mask[ly0:ly1,lx0:lx1]=255
    src_gray_full=cv2.cvtColor(aligned_source,cv2.COLOR_BGR2GRAY)
    source_ink=((src_gray_full<190)&(label_mask>0)).astype(np.uint8)*255
    return out,write,source_ink,{
        "reason":"ok_paired_source_label_raster",
        "clear_backend":"paired-source-label-raster",
        "clear_mask":clear,
        "target_ink_mask":target_core,
        "source_label_box":[int(v) for v in label_box],
        "source_label_pixels":int(cv2.countNonZero(label_mask)),
        "target_clear_pixels":int(cv2.countNonZero(clear)),
        "target_text_component_count":int(len(selected)),
        "write_pixels":int(cv2.countNonZero(write)),
        "boundary_touch":False,
        "raw_box_rescue":True,
        "label_proof":label_diag,
    }

def _transfer_open_complex_text_region(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    target_clear_hint: np.ndarray | None = None,
    prefer_source_recolor: bool = False,
    source_text_gate: np.ndarray | None = None,
    raw_group_mode: bool = False,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Erase only Japanese glyph strokes and composite the registered Chinese ink.

    No rectangular crop, OCR re-typesetting or whole coloured-background fill is
    used.  The source/target ink maps select changed glyph groups, Japanese groups
    are inpainted with a 1-2px fringe, then the source raster glyphs are composited
    at their registered location.  This is the safe path for burst balloons, open
    captions and text on artwork.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    box = _bbox_from_mask(region_mask)
    if box is None:
        return None, empty, empty, {"reason": "empty_region"}
    x0, y0, x1, y1 = box
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(6, int(round(max(bw, bh) * float(getattr(cfg, "paired_diff_complex_region_pad_ratio", 0.16)))))
    gate = cv2.dilate(
        region_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1)),
    )
    # Keep the dilation bounded around the actual candidate; very large free-text
    # islands otherwise risk touching unrelated speech bubbles across a gutter.
    rect = np.zeros(shape, np.uint8)
    xa, ya = max(0, x0 - pad), max(0, y0 - pad)
    xb, yb = min(shape[1], x1 + pad), min(shape[0], y1 + pad)
    rect[ya:yb, xa:xb] = 255
    gate = cv2.bitwise_and(gate, rect)

    src_ink = _complex_text_ink_map(aligned_source)
    tgt_ink = _complex_text_ink_map(target)
    tol = max(1, int(getattr(cfg, "paired_diff_ink_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    src_near = cv2.dilate(src_ink, k)
    tgt_near = cv2.dilate(tgt_ink, k)
    src_unique = cv2.bitwise_and(src_ink, cv2.bitwise_not(tgt_near))
    tgt_unique = cv2.bitwise_and(tgt_ink, cv2.bitwise_not(src_near))
    gap = int(getattr(cfg, "paired_diff_complex_group_gap_px", 5))
    src_sel = _select_changed_text_components(src_ink, src_unique, gate, gap)
    tgt_sel = _select_changed_text_components(tgt_ink, tgt_unique, gate, gap)
    src_count = int(cv2.countNonZero(src_sel)); tgt_count = int(cv2.countNonZero(tgt_sel))
    # Cross-rendition scans can move/blur a glyph enough that the strict
    # edition-exclusive seed disappears, even though compact source/target
    # lettering is visibly present inside the trusted local region. Recover
    # those groups from the local ink map as a bounded fallback; never expand
    # beyond ``gate`` or copy a rectangular coloured background.
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        src_sel = _select_changed_text_components(src_ink, src_ink, gate, gap)
        src_count = int(cv2.countNonZero(src_sel))
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)):
        tgt_sel = _select_changed_text_components(tgt_ink, tgt_ink, gate, gap)
        tgt_count = int(cv2.countNonZero(tgt_sel))
    if src_count < int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16)):
        return None, empty, empty, {"reason": "insufficient_source_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}
    # A trusted structural region may contain Chinese source lettering where
    # the HD target has no separable Japanese component (for example a glyph
    # lost in a bright/halftone edge). In that case there is nothing to erase;
    # the bounded source ink can still be written safely.
    target_missing_but_source_present = (
        tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12))
        and src_count >= int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16))
    )
    if tgt_count < int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12)) and not target_missing_but_source_present:
        return None, empty, empty, {"reason": "insufficient_target_ink", "source_ink_pixels": src_count, "target_ink_pixels": tgt_count}

    clear = tgt_sel.copy()
    target_content_mask = tgt_sel.copy()
    semantic_hint_pixels = 0
    semantic_hint_admitted = False
    if target_clear_hint is not None and target_clear_hint.shape == shape:
        # Koharu's child text mask is stronger evidence than a grayscale ink
        # threshold for outlined/coloured glyphs.  A yellow glyph on a magenta
        # burst can be visually bright, so the old dark-ink map saw only its
        # black outline and left the coloured fill behind.  Admit the semantic
        # text footprint only when it is a bounded subset of the trusted parent
        # region and actually touches the selected target glyph group.
        hint = cv2.bitwise_and((target_clear_hint > 0).astype(np.uint8) * 255, gate)
        hint_area = int(cv2.countNonZero(hint))
        gate_area = max(1, int(cv2.countNonZero(gate)))
        if hint_area > 0 and hint_area <= int(round(gate_area * 0.42)):
            near = cv2.dilate(
                (tgt_sel > 0).astype(np.uint8) * 255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
            )
            touch = int(np.count_nonzero((hint > 0) & (near > 0)))
            # Require real target-ink contact so a coarse detector polygon can
            # never become an unconditional rectangular clear instruction.
            if touch >= max(8, int(round(0.002 * hint_area))):
                clear = np.maximum(clear, hint)
                # ``hint`` is a semantic erase authority and may fill a coarse
                # polygon around coloured glyphs; it is not itself literal ink
                # evidence.  For residual QA, keep only target ink components
                # that lie inside/near that semantic footprint.  This prevents
                # intact dark radial texture from being scored as leftover
                # Japanese after a successful recolour.
                audit_near = cv2.dilate(hint, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
                target_content_mask = cv2.bitwise_and(tgt_sel, audit_near)
                semantic_hint_pixels = hint_area
                semantic_hint_admitted = True
    d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
    clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
    clear = cv2.bitwise_and(clear, gate)

    recolor_diag: dict = {}
    recolor_out = None
    recolor_write = np.zeros(shape, np.uint8)
    raw_blackhat_source_mask = None
    if prefer_source_recolor:
        recolor_out, recolor_write, raw_blackhat_source_mask, recolor_diag = _raw_box_blackhat_text_transfer(
            aligned_source, target, region_mask, cfg, source_text_gate=source_text_gate,
            large_raw_group=raw_group_mode,
        )
    if recolor_out is None and (semantic_hint_admitted or prefer_source_recolor):
        recolor_out, recolor_write, recolor_diag = _source_guided_monochrome_recolor(
            aligned_source, target, region_mask, clear, src_sel, cfg
        )
    if recolor_out is not None:
        out = recolor_out
        write = recolor_write
        dense_source_mask = recolor_diag.pop("_source_dense_glyph_mask", None)
        specialized_clear = recolor_diag.pop("clear_mask", None)
        specialized_target_ink = recolor_diag.pop("target_ink_mask", None)
        if isinstance(raw_blackhat_source_mask, np.ndarray) and raw_blackhat_source_mask.shape == shape:
            src_sel = raw_blackhat_source_mask.astype(np.uint8)
            src_count = int(cv2.countNonZero(src_sel))
            if isinstance(specialized_target_ink, np.ndarray) and specialized_target_ink.shape == shape:
                target_content_mask = specialized_target_ink.astype(np.uint8)
                tgt_count = int(cv2.countNonZero(target_content_mask))
            if isinstance(specialized_clear, np.ndarray) and specialized_clear.shape == shape:
                clear = specialized_clear.astype(np.uint8)
        elif isinstance(dense_source_mask, np.ndarray) and dense_source_mask.shape == shape:
            # Downstream completeness/repair must audit the filtered translated
            # glyphs, not the intentionally sensitive raw ink map.  Otherwise
            # halftone starburst texture is mistaken for missing Chinese and the
            # repair pass paints monochrome SOURCE texture back over the colour
            # master.
            src_sel = dense_source_mask.astype(np.uint8)
            src_count = int(cv2.countNonZero(src_sel))
        clear_backend = str(recolor_diag.get("clear_backend") or "source-guided-monochrome-recolor")
    else:
        out = cv2.inpaint(target, clear, 2.5, cv2.INPAINT_TELEA)
        alpha = _soft_ink_alpha(aligned_source, src_sel, gate)
        a3 = alpha[..., None]
        out = np.clip(
            aligned_source.astype(np.float32) * a3 + out.astype(np.float32) * (1.0 - a3),
            0, 255,
        ).astype(np.uint8)
        write = np.maximum(clear, (alpha * 255).astype(np.uint8))
        clear_backend = "opencv-telea"
    # Float alpha can change a pixel by one code value even when uint8(alpha*255)
    # rounds to zero.  Audit/export masks must describe the *exact* write footprint
    # so pixels outside it are guaranteed byte-identical to the target.
    changed = np.any(out != target, axis=2)
    write[changed] = 255

    # Candidate-boundary contact is not a hard drop: keep a reversible preview so
    # the page cannot silently retain Japanese, but make the review requirement
    # explicit. The expanded gate normally prevents this in complete source pages.
    edge_band = np.zeros(shape, np.uint8)
    edge_band[ya:yb, xa:xb] = 255
    inner = cv2.erode(edge_band, np.ones((3, 3), np.uint8))
    boundary_touch = bool(np.count_nonzero((src_sel > 0) & (edge_band > inner)))
    return out, write, src_sel, {
        "reason": "ok",
        "source_ink_pixels": src_count,
        "target_ink_pixels": tgt_count,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "boundary_touch": boundary_touch,
        "clear_mask": clear,
        "target_ink_mask": target_content_mask,
        "semantic_target_clear_hint_pixels": int(semantic_hint_pixels),
        "semantic_target_clear_hint_admitted": bool(semantic_hint_admitted),
        "clear_backend": clear_backend,
        "source_recolor": recolor_diag,
    }



def _semantic_local_change_ratio(
    aligned_source: np.ndarray,
    target: np.ndarray,
    gate: np.ndarray,
    cfg: MaskReplaceConfig,
) -> float:
    """Return a tolerant local ink-shape difference score in ``gate``.

    The score is used only for standalone Koharu text/SFX recovery.  Bubble
    regions already have stronger container semantics; free text printed on
    artwork needs an extra proof that SOURCE and TARGET actually differ before
    any automatic pixel write is attempted.
    """
    if gate is None or cv2.countNonZero(gate) <= 0:
        return 0.0
    src = (_complex_text_ink_map(aligned_source) > 0) & (gate > 0)
    tgt = (_complex_text_ink_map(target) > 0) & (gate > 0)
    if not np.any(src) or not np.any(tgt):
        return 0.0
    tol = max(1, int(getattr(cfg, "paired_diff_ink_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    src8 = src.astype(np.uint8) * 255
    tgt8 = tgt.astype(np.uint8) * 255
    src_near = cv2.dilate(src8, k) > 0
    tgt_near = cv2.dilate(tgt8, k) > 0
    unique = (src & ~tgt_near) | (tgt & ~src_near)
    denom = max(1, int(np.count_nonzero(src | tgt)))
    return float(np.count_nonzero(unique) / denom)



def _semantic_paired_open_text_change_proof(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_gate: np.ndarray,
    source_evidence: Any | None,
    registration: RegistrationResult | None,
    cfg: MaskReplaceConfig,
    *,
    target_row: Any | None = None,
) -> dict[str, object]:
    """Return a conservative two-sided proof for translated open text.

    The normal lane still requires a >=0.70 SOURCE Koharu text seed.  A lower
    confidence SOURCE seed is allowed only as a *paired-rendition rescue*: the
    TARGET seed must be strong, page registration must be reliable, the warped
    SOURCE polygon and raw boxes must agree in both position and scale, and a
    1px-tolerant raster comparison must prove that the glyph shapes changed.

    This prevents a monochrome/scan SOURCE confidence dip from dropping real
    translated open text (Vol.04 p0036 class) without globally lowering the
    semantic detector threshold for scene props or same-language SFX.
    """
    out: dict[str, object] = {
        "admit": False, "source_item_index": None, "projected_overlap": 0.0,
        "sensitive_change_ratio": 0.0, "source_confidence": 0.0,
        "proof_mode": "paired_text", "reason": "no_source_evidence",
    }
    if source_evidence is None or registration is None or not bool(getattr(source_evidence, "available", False)):
        return out
    if target_gate is None or target_gate.shape != target.shape[:2] or cv2.countNonZero(target_gate) <= 0:
        out["reason"] = "empty_target_gate"; return out

    def _over_min(a: np.ndarray, b: np.ndarray) -> float:
        aa = int(cv2.countNonZero(a)); bb = int(cv2.countNonZero(b))
        if aa <= 0 or bb <= 0 or a.shape != b.shape:
            return 0.0
        return float(np.count_nonzero((a > 0) & (b > 0)) / max(1, min(aa, bb)))

    normal_floor = 0.70
    low_floor = float(getattr(cfg, "koharu_semantic_paired_text_lowconf_source_min_confidence", 0.45))
    best = None
    for row in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(row, "label", "")) != "text":
            continue
        conf = float(getattr(row, "confidence", 0.0) or 0.0)
        if conf < min(normal_floor, low_floor):
            continue
        sm = getattr(row, "mask", None)
        if not isinstance(sm, np.ndarray) or cv2.countNonZero(sm) <= 0:
            continue
        pm = _warp_mask((sm > 0).astype(np.uint8) * 255, registration.matrix, target.shape[:2])
        ov = _over_min(target_gate, pm)
        # Position agreement is primary. Confidence only breaks near-ties; an
        # unrelated 0.95 text item must never beat the true 0.53 paired item.
        score = float(ov + min(0.02, conf * 0.001))
        if best is None or score > best[0]:
            best = (score, ov, row, pm)
    if best is None:
        out["reason"] = "no_source_text_above_rescue_floor"; return out
    _, overlap, row, _ = best
    source_conf = float(getattr(row, "confidence", 0.0) or 0.0)
    out["projected_overlap"] = float(overlap)
    out["source_confidence"] = source_conf
    out["source_item_index"] = (getattr(row, "meta", {}) or {}).get("item_index")

    lowconf_lane = source_conf < normal_floor
    if lowconf_lane:
        out["proof_mode"] = "lowconf_paired_raw_text"
        target_conf = float(getattr(target_row, "confidence", 0.0) or 0.0) if target_row is not None else 0.0
        out["target_confidence"] = target_conf
        if target_conf < float(getattr(cfg, "koharu_semantic_paired_text_lowconf_target_min_confidence", 0.85)):
            out["reason"] = "lowconf_source_without_strong_target"; return out
        if float(getattr(registration, "confidence", 0.0) or 0.0) < float(
            getattr(cfg, "koharu_semantic_paired_text_lowconf_min_registration", 0.80)
        ):
            out["reason"] = "lowconf_source_weak_registration"; return out
        if overlap < float(getattr(cfg, "koharu_semantic_paired_text_lowconf_min_overlap", 0.55)):
            out["reason"] = "lowconf_source_weak_projected_overlap"; return out

        traw = list((getattr(target_row, "meta", {}) or {}).get("raw_box") or []) if target_row is not None else []
        sraw = list((getattr(row, "meta", {}) or {}).get("raw_box") or [])
        if len(traw) != 4 or len(sraw) != 4:
            out["reason"] = "lowconf_source_missing_raw_box"; return out
        H = transform_to_homography(registration.matrix)
        pts = np.asarray([
            [float(sraw[0]), float(sraw[1])], [float(sraw[2]), float(sraw[1])],
            [float(sraw[2]), float(sraw[3])], [float(sraw[0]), float(sraw[3])],
        ], np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            out["reason"] = "lowconf_source_invalid_raw_projection"; return out
        sproj = [float(mapped[:, 0].min()), float(mapped[:, 1].min()), float(mapped[:, 0].max()), float(mapped[:, 1].max())]
        ix0=max(float(traw[0]),sproj[0]); iy0=max(float(traw[1]),sproj[1])
        ix1=min(float(traw[2]),sproj[2]); iy1=min(float(traw[3]),sproj[3])
        inter=max(0.0,ix1-ix0)*max(0.0,iy1-iy0)
        tarea=max(1.0,(float(traw[2])-float(traw[0]))*(float(traw[3])-float(traw[1])))
        sarea=max(1.0,(sproj[2]-sproj[0])*(sproj[3]-sproj[1]))
        raw_overlap=float(inter/max(1.0,min(tarea,sarea)))
        area_similarity=_raw_group_area_similarity(tarea,sarea)
        out["raw_box_overlap"] = round(raw_overlap, 6)
        out["raw_box_area_similarity"] = round(area_similarity, 6)
        out["source_projected_raw_box"] = [round(v, 3) for v in sproj]
        if raw_overlap < float(getattr(cfg, "koharu_semantic_paired_text_lowconf_min_raw_overlap", 0.85)):
            out["reason"] = "lowconf_source_weak_raw_overlap"; return out
        if area_similarity < float(getattr(cfg, "koharu_semantic_paired_text_lowconf_min_raw_area_similarity", 0.55)):
            out["reason"] = "lowconf_source_raw_scale_mismatch"; return out
        min_sensitive = float(getattr(cfg, "koharu_semantic_paired_text_lowconf_min_change_ratio", 0.15))
    else:
        min_overlap = float(getattr(cfg, "koharu_semantic_paired_text_min_overlap", 0.45))
        if overlap < min_overlap:
            out["reason"] = "weak_projected_overlap"; return out
        min_sensitive = float(getattr(cfg, "koharu_semantic_paired_text_min_change_ratio", 0.30))

    src = (_complex_text_ink_map(aligned_source) > 0) & (target_gate > 0)
    tgt = (_complex_text_ink_map(target) > 0) & (target_gate > 0)
    if not np.any(src) or not np.any(tgt):
        out["reason"] = "missing_local_ink"; return out
    tol = max(1, int(getattr(cfg, "koharu_semantic_paired_text_tolerance_px", 1)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    src_near = cv2.dilate(src.astype(np.uint8) * 255, k) > 0
    tgt_near = cv2.dilate(tgt.astype(np.uint8) * 255, k) > 0
    unique = (src & ~tgt_near) | (tgt & ~src_near)
    sensitive = float(np.count_nonzero(unique) / max(1, int(np.count_nonzero(src | tgt))))
    out["sensitive_change_ratio"] = sensitive
    out["admit"] = bool(sensitive >= min_sensitive)
    out["reason"] = (
        "lowconf_paired_raw_changed_text" if out["admit"] and lowconf_lane else
        "paired_changed_text" if out["admit"] else
        "lowconf_paired_text_shape_too_similar" if lowconf_lane else
        "paired_text_shape_too_similar"
    )
    return out



def _semantic_paired_raw_group_change_proof(
    aligned_source: np.ndarray,
    target: np.ndarray,
    target_row: Any,
    source_evidence: Any | None,
    registration: RegistrationResult | None,
    cfg: MaskReplaceConfig,
) -> dict[str, object]:
    """Prove a translated low-confidence text group from paired Koharu raw boxes.

    This is the p-140 safety valve.  It does *not* lower the normal semantic text
    threshold.  A low-confidence TARGET text seed can be admitted only when the
    registered SOURCE also exposes an under-segmented text raw box, the two raw
    boxes overlap strongly, registration itself is reliable, and the whole group
    shows a large tolerant glyph-shape change.  Same-language SFX therefore stay
    protected even if Koharu happens to call them text.
    """
    out: dict[str, object] = {
        "admit": False,
        "proof_mode": "raw_group",
        "source_item_index": None,
        "raw_box_overlap": 0.0,
        "raw_group_change_ratio": 0.0,
        "reason": "no_source_evidence",
    }
    if source_evidence is None or registration is None or not bool(getattr(source_evidence, "available", False)):
        return out
    reg_conf = float(getattr(registration, "confidence", 0.0) or 0.0)
    if reg_conf < float(getattr(cfg, "koharu_semantic_paired_raw_group_min_registration", 0.70)):
        out["reason"] = "weak_registration"
        return out
    tconf = float(getattr(target_row, "confidence", 0.0) or 0.0)
    if tconf < float(getattr(cfg, "koharu_semantic_paired_raw_group_min_target_confidence", 0.35)):
        out["reason"] = "weak_target_confidence"
        return out
    tmask = getattr(target_row, "mask", None)
    traw = list((getattr(target_row, "meta", {}) or {}).get("raw_box") or [])
    if not isinstance(tmask, np.ndarray) or tmask.ndim != 2 or len(traw) != 4:
        out["reason"] = "missing_target_raw_group"
        return out
    tmask_area = max(1.0, float(cv2.countNonZero(tmask)))
    traw_area = max(1.0, (float(traw[2]) - float(traw[0])) * (float(traw[3]) - float(traw[1])))
    underseg_min = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_underseg_ratio", 2.0))
    if traw_area / tmask_area < underseg_min:
        out["reason"] = "target_not_undersegmented"
        return out

    H = transform_to_homography(registration.matrix)
    min_sconf = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_source_confidence", 0.60))
    best = None
    for srow in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(srow, "label", "")) != "text" or float(getattr(srow, "confidence", 0.0) or 0.0) < min_sconf:
            continue
        smask = getattr(srow, "mask", None)
        sraw = list((getattr(srow, "meta", {}) or {}).get("raw_box") or [])
        if not isinstance(smask, np.ndarray) or smask.ndim != 2 or len(sraw) != 4:
            continue
        smask_area = max(1.0, float(cv2.countNonZero(smask)))
        sraw_area = max(1.0, (float(sraw[2]) - float(sraw[0])) * (float(sraw[3]) - float(sraw[1])))
        if sraw_area / smask_area < underseg_min:
            continue
        pts = np.asarray([
            [float(sraw[0]), float(sraw[1])], [float(sraw[2]), float(sraw[1])],
            [float(sraw[2]), float(sraw[3])], [float(sraw[0]), float(sraw[3])],
        ], np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            continue
        sproj = [float(mapped[:, 0].min()), float(mapped[:, 1].min()), float(mapped[:, 0].max()), float(mapped[:, 1].max())]
        ix0 = max(float(traw[0]), sproj[0]); iy0 = max(float(traw[1]), sproj[1])
        ix1 = min(float(traw[2]), sproj[2]); iy1 = min(float(traw[3]), sproj[3])
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        sarea = max(1.0, (sproj[2] - sproj[0]) * (sproj[3] - sproj[1]))
        raw_overlap = float(inter / max(1.0, min(traw_area, sarea)))
        area_similarity = _raw_group_area_similarity(traw_area, sarea)
        # A tiny box fully inside the TARGET group has raw_overlap=1.0 but is
        # not a plausible rendition mate.  Rank by overlap *and* scale so the
        # corresponding full letter/title block wins over an unrelated caption.
        pair_score = float(raw_overlap * np.sqrt(area_similarity))
        if best is None or pair_score > best[0]:
            best = (pair_score, raw_overlap, area_similarity, srow, sproj)
    if best is None:
        out["reason"] = "no_paired_source_raw_group"
        return out
    _, raw_overlap, area_similarity, srow, sproj = best
    out["raw_box_overlap"] = round(float(raw_overlap), 6)
    out["raw_box_area_similarity"] = round(float(area_similarity), 6)
    out["source_item_index"] = (getattr(srow, "meta", {}) or {}).get("item_index")
    min_area_similarity = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_area_similarity", 0.17))
    if area_similarity < min_area_similarity:
        out["reason"] = "raw_group_scale_mismatch"
        return out
    if raw_overlap < float(getattr(cfg, "koharu_semantic_paired_raw_group_min_overlap", 0.75)):
        out["reason"] = "weak_raw_group_overlap"
        return out

    x0 = max(0, int(np.floor(min(float(traw[0]), sproj[0])))); y0 = max(0, int(np.floor(min(float(traw[1]), sproj[1]))))
    x1 = min(target.shape[1], int(np.ceil(max(float(traw[2]), sproj[2])))); y1 = min(target.shape[0], int(np.ceil(max(float(traw[3]), sproj[3]))))
    if x1 <= x0 or y1 <= y0:
        out["reason"] = "empty_raw_group"
        return out
    raw_gate = np.zeros(target.shape[:2], np.uint8)
    raw_gate[y0:y1, x0:x1] = 255
    change = float(_semantic_local_change_ratio(aligned_source, target, raw_gate, cfg))
    out["raw_group_change_ratio"] = round(change, 6)
    if change < float(getattr(cfg, "koharu_semantic_paired_raw_group_min_change_ratio", 0.50)):
        out["reason"] = "raw_group_shape_too_similar"
        return out
    out["admit"] = True
    out["reason"] = "paired_raw_group_changed_text"
    return out

def _semantic_paired_raw_box_gate(
    target_row: Any,
    base_gate: np.ndarray,
    source_evidence: Any | None,
    registration: RegistrationResult | None,
    paired_proof: dict[str, object],
    shape: tuple[int, int],
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, dict[str, object]]:
    diag: dict[str, object] = {"expanded": False, "reason": "not_eligible"}
    change_proved = bool(paired_proof.get("admit", False) or paired_proof.get("ordinary_change_admitted", False))
    if not change_proved or source_evidence is None or registration is None:
        return None, diag
    traw = list((getattr(target_row, "meta", {}) or {}).get("raw_box") or [])
    if len(traw) != 4:
        diag["reason"] = "missing_target_raw_box"
        return None, diag
    base_box = _bbox_from_mask(base_gate)
    if base_box is None:
        diag["reason"] = "empty_base_gate"
        return None, diag
    bx0, by0, bx1, by1 = [float(v) for v in base_box]
    base_area = max(1.0, (bx1-bx0)*(by1-by0))
    target_raw_area = max(1.0, (float(traw[2])-float(traw[0]))*(float(traw[3])-float(traw[1])))
    if target_raw_area / base_area < 1.8:
        diag["reason"] = "target_mask_not_undersegmented"
        return None, diag

    H = transform_to_homography(registration.matrix)
    best = None
    proof_mode = str(paired_proof.get("proof_mode") or "")
    raw_source_floor = (
        float(getattr(cfg, "koharu_semantic_paired_raw_group_min_source_confidence", 0.60))
        if proof_mode == "raw_group" else
        float(getattr(cfg, "koharu_semantic_paired_text_lowconf_source_min_confidence", 0.45))
        if proof_mode == "lowconf_paired_raw_text" else
        0.70
    )
    proof_source_index = paired_proof.get("source_item_index")
    source_rows = list(getattr(source_evidence, "items", []) or [])
    if proof_source_index is not None:
        exact = [
            srow for srow in source_rows
            if (getattr(srow, "meta", {}) or {}).get("item_index") == proof_source_index
        ]
        # The raw-group proof already selected a SOURCE mate.  Reuse it instead
        # of running a second overlap-only search that can drift to a contained
        # small caption on the same paper object.
        if exact:
            source_rows = exact
    for srow in source_rows:
        if str(getattr(srow, "label", "")) != "text" or float(getattr(srow, "confidence", 0.0) or 0.0) < raw_source_floor:
            continue
        sraw = list((getattr(srow, "meta", {}) or {}).get("raw_box") or [])
        if len(sraw) != 4:
            continue
        pts = np.asarray([[float(sraw[0]),float(sraw[1])],[float(sraw[2]),float(sraw[1])],
                          [float(sraw[2]),float(sraw[3])],[float(sraw[0]),float(sraw[3])]],
                         np.float32).reshape(-1,1,2)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1,2)
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            continue
        sproj=[float(mapped[:,0].min()),float(mapped[:,1].min()),float(mapped[:,0].max()),float(mapped[:,1].max())]
        ix0=max(float(traw[0]),sproj[0]); iy0=max(float(traw[1]),sproj[1]); ix1=min(float(traw[2]),sproj[2]); iy1=min(float(traw[3]),sproj[3])
        inter=max(0.0,ix1-ix0)*max(0.0,iy1-iy0)
        sarea=max(1.0,(sproj[2]-sproj[0])*(sproj[3]-sproj[1]))
        raw_overlap=float(inter/max(1.0,min(target_raw_area,sarea)))
        area_similarity = _raw_group_area_similarity(target_raw_area, sarea)
        pair_score = float(raw_overlap * np.sqrt(area_similarity))
        if best is None or pair_score > best[0]:
            best=(pair_score,raw_overlap,area_similarity,srow,sproj)
    if best is None or best[1] < 0.45:
        diag["reason"] = "weak_raw_box_overlap"
        diag["raw_box_overlap"] = 0.0 if best is None else float(best[1])
        return None, diag
    _,raw_overlap,area_similarity,srow,sproj=best
    raw_group_mode = str(paired_proof.get("proof_mode") or "") == "raw_group"
    if raw_group_mode:
        min_area_similarity = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_area_similarity", 0.17))
        if area_similarity < min_area_similarity:
            diag["reason"] = "raw_group_scale_mismatch"
            diag["raw_box_overlap"] = round(float(raw_overlap), 4)
            diag["raw_box_area_similarity"] = round(float(area_similarity), 4)
            return None, diag
    x0=max(0,int(np.floor(min(float(traw[0]),sproj[0])))); y0=max(0,int(np.floor(min(float(traw[1]),sproj[1]))))
    x1=min(shape[1],int(np.ceil(max(float(traw[2]),sproj[2])))); y1=min(shape[0],int(np.ceil(max(float(traw[3]),sproj[3]))))
    if x1<=x0 or y1<=y0:
        diag["reason"] = "empty_raw_union"
        return None, diag
    union_area=float((x1-x0)*(y1-y0))
    if raw_group_mode:
        page_area = max(1.0, float(shape[0] * shape[1]))
        max_page_ratio = float(getattr(cfg, "koharu_semantic_paired_raw_group_max_page_area_ratio", 0.18))
        if union_area / page_area > max_page_ratio:
            diag["reason"] = "raw_group_union_too_large_for_page"
            diag["page_area_ratio"] = round(float(union_area/page_area), 6)
            return None, diag
        min_raw_overlap = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_overlap", 0.75))
        if raw_overlap < min_raw_overlap:
            diag["reason"] = "raw_group_overlap_below_strict_floor"
            return None, diag
    elif union_area/base_area > 12.0:
        diag["reason"] = "raw_union_too_large"
        return None, diag
    out=np.zeros(shape,np.uint8); out[y0:y1,x0:x1]=255
    diag.update({"expanded":True,"reason":"paired_raw_box_undersegmentation_rescue",
                 "target_raw_box":[round(float(v),3) for v in traw],
                 "source_item_index":(getattr(srow,"meta",{}) or {}).get("item_index"),
                 "source_projected_raw_box":[round(float(v),3) for v in sproj],
                 "raw_box_overlap":round(float(raw_overlap),4),
                 "raw_box_area_similarity":round(float(area_similarity),4),
                 "bbox_area_ratio":round(union_area/base_area,4),
                 "change_proof":"paired_sensitive" if bool(paired_proof.get("admit",False)) else "ordinary_conservative"})
    return out, diag


def _strict_raw_group_semantic_result_safe(
    paired_proof: dict[str, object],
    raw_gate_diag: dict[str, object],
    transfer_diag: dict[str, object],
    rec: MaskTransferRecord,
    cfg: MaskReplaceConfig,
) -> bool:
    """Return True only for a fully proven bilateral raw-group result.

    Standalone open text normally remains reviewable.  This exception is for a
    strict SOURCE/TARGET raw-group that has already passed registration/scale/
    overlap/change proof and then completed through a bounded specialised
    renderer with no residual TARGET text.
    """
    if not bool(paired_proof.get("admit", False)) or str(paired_proof.get("proof_mode") or "") != "raw_group":
        return False
    if not bool(raw_gate_diag.get("expanded", False)):
        return False
    min_overlap = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_overlap", 0.75))
    min_area = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_area_similarity", 0.17))
    if float(paired_proof.get("raw_box_overlap", 0.0) or 0.0) < min_overlap:
        return False
    if float(paired_proof.get("raw_box_area_similarity", 0.0) or 0.0) < min_area:
        return False
    backend = str(transfer_diag.get("clear_backend") or "")
    if backend not in {"paired-raw-group-large-text-inpaint", "paired-raw-group-plain-paper-text"}:
        return False
    if bool(transfer_diag.get("boundary_touch", False)) or not bool(getattr(rec, "content_complete", False)):
        return False
    max_res = float(getattr(cfg, "content_completeness_max_target_residual", 0.10))
    return float(getattr(rec, "target_residual_ratio", 1.0) or 0.0) <= max_res


def _semantic_boundary_touch_result_safe(
    rec: MaskTransferRecord,
    cfg: MaskReplaceConfig,
    *,
    standalone: bool,
    boundary_touch: bool,
) -> bool:
    """Allow a boundary-touching container only with very strong independent proof.

    Touching the candidate boundary is a useful truncation warning, but it is not
    itself evidence of missing text.  Normal speech balloons can legitimately
    reach a detector edge.  Downgrade that geometry-only warning only when the
    transfer is non-standalone, the independent completeness audit passed, almost
    all SOURCE ink survives, and essentially no TARGET text remains (p-207 class).
    """
    if standalone or not boundary_touch or not bool(getattr(rec, "content_complete", False)):
        return False
    min_cov = float(getattr(cfg, "koharu_semantic_boundary_safe_min_source_coverage", 0.98))
    max_res = float(getattr(cfg, "koharu_semantic_boundary_safe_max_target_residual", 0.01))
    return (
        float(getattr(rec, "source_ink_coverage", 0.0) or 0.0) >= min_cov
        and float(getattr(rec, "target_residual_ratio", 1.0) or 0.0) <= max_res
    )


def _semantic_page_furniture_gate(gate: np.ndarray, shape: tuple[int, int], cfg) -> bool:
    """Return True for tiny running-header/page-number semantic text.

    Koharu legitimately detects this print as text, but it is publication
    furniture rather than translation content.  It must never enter the
    standalone semantic completion path, otherwise tiny page numbers can become
    synthetic Direct/Mask rejections after SOURCE ink extraction.
    """
    if gate is None or gate.shape != shape or cv2.countNonZero(gate) <= 0:
        return False
    if not bool(getattr(cfg, "paired_diff_protect_page_furniture", True)):
        return False
    box = _bbox_from_mask(gate)
    if box is None:
        return False
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = shape
    bw=max(1,x1-x0); bh=max(1,y1-y0)
    narrow = bw <= w * float(getattr(cfg, "paired_diff_page_furniture_max_width_ratio", 0.18))
    short = bh <= h * float(getattr(cfg, "paired_diff_page_furniture_max_height_ratio", 0.12))
    top_ratio = float(getattr(cfg, "paired_diff_page_furniture_top_ratio", 0.10))
    # Semantic evidence is slightly noisier at the printed page-number baseline
    # than paired-diff boxes.  0.94 safely covers the footer band while the
    # narrow+short constraints prevent dialogue balloons near the bottom edge
    # from being classified as furniture.
    bottom_ratio = min(float(getattr(cfg, "paired_diff_page_furniture_bottom_ratio", 0.965)), 0.94)
    edge = bool(y1 <= h * top_ratio or y0 >= h * bottom_ratio)
    return bool(edge and narrow and short)

def transfer_koharu_semantic_regions(
    aligned_source: np.ndarray,
    target: np.ndarray,
    evidence: Any,
    cfg: MaskReplaceConfig | None = None,
    *,
    exclude_mask: np.ndarray | None = None,
    include_sfx: bool = False,
    bubble_min_confidence: float | None = None,
    text_min_confidence: float | None = None,
    source_evidence: Any | None = None,
    registration: RegistrationResult | None = None,
    strict_raw_group_only: bool = False,
) -> MaskTransferResult:
    """Recover Koharu regions missed by container pairing without OCR.

    Koharu Layout can correctly identify a coloured/open text container even
    when Paired Diff refuses to classify it as a rigid white balloon.  Treat
    those semantic regions as first-class candidates and reuse the registered
    SOURCE raster ink.  Only changed glyph components are cleared/written; the
    TARGET colour/artwork background is never replaced by a rectangular SOURCE
    crop.

    This pass is intentionally *completion only*: regions already touched by a
    successful Direct/Mask renderer are skipped.  Standalone text (no parent
    bubble) additionally requires a high local SOURCE/TARGET ink-change score.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.shape[:2] != shape:
        raise ValueError("aligned_source must be in target coordinates")
    if evidence is None or not bool(getattr(evidence, "available", False)):
        empty_rgba = np.zeros((shape[0], shape[1], 4), np.uint8)
        return MaskTransferResult(target.copy(), empty_rgba, np.zeros(shape, np.uint8), [], [], np.zeros(shape, np.uint8))
    if exclude_mask is None or exclude_mask.shape != shape:
        exclude_mask = np.zeros(shape, np.uint8)

    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite = np.zeros(shape, np.uint8)
    clear_all = np.zeros(shape, np.uint8)
    raw_rescue_allow = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    patch_matches: list[BubblePatchMatch] = []

    max_overlap = float(getattr(cfg, "koharu_semantic_max_existing_overlap", 0.28))
    # A selected primary detector owns candidate admission.  Do not silently
    # re-threshold a Koharu bubble/text above the threshold that already admitted
    # it at the layout stage; doing so creates the confusing state "detected in
    # overlay, never replaced".  Renderer-specific floors remain the fallback
    # only when no primary threshold was supplied.
    bubble_min = float(
        bubble_min_confidence if bubble_min_confidence is not None
        else getattr(cfg, "koharu_semantic_bubble_min_confidence", 0.70)
    )
    text_min = float(
        text_min_confidence if text_min_confidence is not None
        else getattr(cfg, "koharu_semantic_text_min_confidence", 0.75)
    )
    change_min = float(getattr(cfg, "koharu_semantic_standalone_min_change_ratio", 0.18))

    bubbles = [x for x in list(getattr(evidence, "items", []) or []) if str(getattr(x, "label", "")) == "bubble"]
    texts = [x for x in list(getattr(evidence, "items", []) or []) if str(getattr(x, "label", "")) == "text"]
    if include_sfx:
        texts.extend([x for x in list(getattr(evidence, "items", []) or []) if str(getattr(x, "label", "")) == "sfx"])

    def overlap_ratio(mask: np.ndarray, other: np.ndarray) -> float:
        area = max(1, cv2.countNonZero(mask))
        return float(np.count_nonzero((mask > 0) & (other > 0)) / area)

    # First recover missed semantic containers. A text child or a saturated
    # target fill is sufficient evidence that this is a text-bearing region.
    candidates: list[tuple[str, Any, np.ndarray, bool, np.ndarray | None]] = []
    standalone_pair_proofs: dict[int, dict[str, object]] = {}
    raw_group_claimed_sources: set[object] = set()
    for row in ([] if strict_raw_group_only else sorted(bubbles, key=lambda x: float(getattr(x, "confidence", 0.0)), reverse=True)):
        if float(getattr(row, "confidence", 0.0)) < bubble_min:
            continue
        gate = np.asarray(getattr(row, "mask", None), dtype=np.uint8) if getattr(row, "mask", None) is not None else None
        if gate is None or gate.shape != shape or cv2.countNonZero(gate) <= 0:
            continue
        if overlap_ratio(gate, exclude_mask) > max_overlap:
            continue
        child_text_masks = [
            np.asarray(t.mask, dtype=np.uint8)
            for t in texts
            if getattr(t, "mask", None) is not None
            and np.asarray(t.mask).shape == shape
            and overlap_ratio(np.asarray(t.mask, dtype=np.uint8), gate) >= 0.35
        ]
        has_text_child = bool(child_text_masks)
        target_text_hint = None
        if child_text_masks:
            target_text_hint = np.zeros(shape, np.uint8)
            for child_mask in child_text_masks:
                target_text_hint = np.maximum(target_text_hint, child_mask)
        coloured = _dominant_saturated_container_mask(target, gate, cfg) is not None
        if not has_text_child and not coloured:
            continue
        candidates.append(("bubble", row, gate, False, target_text_hint))

    # Then recover high-confidence standalone text not already owned by a bubble.
    # These are always marked reviewable and require changed-ink proof.
    for row in sorted(texts, key=lambda x: float(getattr(x, "confidence", 0.0)), reverse=True):
        gate = np.asarray(getattr(row, "mask", None), dtype=np.uint8) if getattr(row, "mask", None) is not None else None
        if gate is None or gate.shape != shape or cv2.countNonZero(gate) <= 0:
            continue
        if overlap_ratio(gate, exclude_mask) > max_overlap:
            continue
        if any(overlap_ratio(gate, np.asarray(b.mask, dtype=np.uint8)) >= 0.55 for b in bubbles if getattr(b, "mask", None) is not None):
            continue
        if _semantic_page_furniture_gate(gate, shape, cfg):
            continue
        local_change = _semantic_local_change_ratio(aligned_source, target, gate, cfg)
        if strict_raw_group_only:
            paired_proof = {"admit": False, "reason": "strict_raw_group_required"}
        else:
            paired_proof = _semantic_paired_open_text_change_proof(
                aligned_source, target, gate, source_evidence, registration, cfg, target_row=row
            )
        if not bool(paired_proof.get("admit", False)):
            raw_group_proof = _semantic_paired_raw_group_change_proof(
                aligned_source, target, row, source_evidence, registration, cfg
            )
            if bool(raw_group_proof.get("admit", False)):
                source_key = raw_group_proof.get("source_item_index")
                if source_key in raw_group_claimed_sources:
                    continue
                raw_group_claimed_sources.add(source_key)
                paired_proof = raw_group_proof
        if strict_raw_group_only and str(paired_proof.get("proof_mode") or "") != "raw_group":
            continue
        # Low-confidence TARGET-only open text is frequently stylised SFX that
        # Koharu happened to label as text (p-135 class). Do not turn such a
        # weak detector hit into a hard Direct rejection unless SOURCE provides
        # a strong paired-text proof. High-confidence target text remains fully
        # audited by semantic QA when it cannot be transferred.
        semantic_floor = max(text_min, float(getattr(cfg, "koharu_semantic_text_min_confidence", 0.75)))
        if float(getattr(row, "confidence", 0.0)) < semantic_floor and not bool(paired_proof.get("admit", False)):
            continue
        if local_change < change_min and not bool(paired_proof.get("admit", False)):
            continue
        paired_proof = dict(paired_proof)
        paired_proof["ordinary_change_admitted"] = bool(local_change >= change_min)
        paired_proof["ordinary_change_ratio"] = round(float(local_change), 6)
        standalone_pair_proofs[id(row)] = paired_proof
        candidates.append((str(getattr(row, "label", "text") or "text"), row, gate, True, None))

    for index, (kind, row, gate, standalone, target_text_hint) in enumerate(candidates):
        active_exclude = np.maximum(exclude_mask, composite)
        if overlap_ratio(gate, active_exclude) > max_overlap:
            continue
        paired_proof = standalone_pair_proofs.get(id(row), {}) if standalone else {}
        raw_gate, raw_gate_diag = _semantic_paired_raw_box_gate(
            row, gate, source_evidence, registration, paired_proof, shape, cfg
        ) if standalone else (None, {"expanded": False, "reason": "not_standalone"})
        transfer_gate = raw_gate if raw_gate is not None else gate
        source_text_gate = None
        if raw_gate is not None:
            sproj = list(raw_gate_diag.get("source_projected_raw_box") or [])
            if len(sproj) == 4:
                sx0=max(0,int(np.floor(float(sproj[0])))); sy0=max(0,int(np.floor(float(sproj[1]))))
                sx1=min(shape[1],int(np.ceil(float(sproj[2])))); sy1=min(shape[0],int(np.ceil(float(sproj[3]))))
                if sx1 > sx0 and sy1 > sy0:
                    source_text_gate=np.zeros(shape,np.uint8)
                    source_text_gate[sy0:sy1,sx0:sx1]=255
        if raw_gate is not None and str(paired_proof.get("proof_mode") or "") == "lowconf_paired_raw_text":
            text_img, write_mask, source_ink_mask, diag = _paired_source_label_raster_transfer(
                aligned_source, rendered, transfer_gate, cfg, paired_proof=paired_proof
            )
        else:
            text_img, write_mask, source_ink_mask, diag = None, np.zeros(shape, np.uint8), np.zeros(shape, np.uint8), {"reason": "source_label_not_applicable"}
        if text_img is None:
            text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
                aligned_source, rendered, transfer_gate, cfg, target_clear_hint=target_text_hint,
                prefer_source_recolor=bool(raw_gate is not None), source_text_gate=source_text_gate,
                raw_group_mode=(str(paired_proof.get("proof_mode") or "") == "raw_group"),
            )
        # Never permit a broad detector-box rescue to degrade into large-area
        # Telea on artwork. If colour reconstruction cannot prove itself safe,
        # fall back to the original narrow semantic mask.
        if raw_gate is not None and str(diag.get("clear_backend") or "") not in {
            "source-guided-monochrome-recolor", "paired-raw-box-blackhat-inpaint",
            "paired-raw-group-large-text-inpaint", "paired-raw-group-plain-paper-text",
            "paired-source-label-raster",
        }:
            # A larger under-segmented paired text group is allowed to expand
            # only when a specialised component renderer independently proves a
            # bright neutral paper field. This fixes the p0048 class without
            # permitting broad Telea over artwork.
            plain_rescue = _paired_text_raw_plain_paper_transfer(
                aligned_source, rendered, raw_gate, cfg,
                paired_proof=paired_proof, raw_gate_diag=raw_gate_diag,
                target_confidence=float(getattr(row, "confidence", 0.0) or 0.0),
                source_text_gate=source_text_gate,
            )
            if plain_rescue[0] is not None:
                text_img, write_mask, source_ink_mask, diag = plain_rescue
                raw_gate_diag = dict(raw_gate_diag)
                raw_gate_diag["plain_paper_rescue"] = True
                raw_gate_diag["reason"] = "paired_text_raw_plain_paper_rescue"
            else:
                plain_reject_diag = dict(plain_rescue[3])
                # A very small, strongly paired high-confidence text raw group may
                # legitimately need bounded Telea: Koharu sometimes returns only one
                # glyph in its polygon while raw_box contains the full short utterance
                # (Vol.04 p0023 class). Never allow this exception for low-confidence
                # or large artwork regions.
                proof_mode = str(paired_proof.get("proof_mode") or "")
                raw_area_ratio = float(cv2.countNonZero(raw_gate) / max(1, shape[0] * shape[1]))
                tiny_paired_telea = bool(
                    proof_mode == "paired_text"
                    and str(diag.get("clear_backend") or "") == "opencv-telea"
                    and float(raw_gate_diag.get("raw_box_overlap", 0.0) or 0.0) >= float(getattr(cfg, "koharu_semantic_paired_text_raw_telea_min_raw_overlap", 0.80))
                    and float(raw_gate_diag.get("raw_box_area_similarity", 0.0) or 0.0) >= float(getattr(cfg, "koharu_semantic_paired_text_raw_telea_min_area_similarity", 0.60))
                    and raw_area_ratio <= float(getattr(cfg, "koharu_semantic_paired_text_raw_telea_max_page_area_ratio", 0.006))
                )
                if tiny_paired_telea:
                    raw_gate_diag = dict(raw_gate_diag)
                    raw_gate_diag["tiny_paired_telea"] = True
                    raw_gate_diag["raw_page_area_ratio"] = round(raw_area_ratio, 6)
                else:
                    text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
                        aligned_source, rendered, gate, cfg, target_clear_hint=target_text_hint
                    )
                    raw_gate_diag = dict(raw_gate_diag)
                    raw_gate_diag["expanded"] = False
                    raw_gate_diag["reason"] = "raw_box_recolor_not_available"
        rid = str((getattr(row, "meta", {}) or {}).get("item_index", index))
        source_id = f"koharu-semantic-src-{kind}-{rid}"
        target_id = f"koharu-semantic-dst-{kind}-{rid}"
        conf = float(getattr(row, "confidence", 0.0))
        rec = MaskTransferRecord(source_id, target_id, conf, False, "not_applied")
        rec.geometry_mode = "free_text" if standalone else "complex_text"
        active_gate = transfer_gate if (raw_gate is not None and bool(raw_gate_diag.get("expanded", False))) else gate
        box = _bbox_from_mask(active_gate) or (0, 0, 0, 0)
        rec.source_bbox = tuple(int(v) for v in box)
        rec.target_bbox = tuple(int(v) for v in box)
        rec.sr_backend = "koharu-semantic-registered-ink"
        rec.meta = {
            "layout_label": str(getattr(row, "label", kind)),
            "layout_confidence": conf,
            "semantic_completion": True,
            "standalone": bool(standalone),
            "local_change_ratio": round(_semantic_local_change_ratio(aligned_source, target, gate, cfg), 4),
            "paired_source_text_proof": dict(standalone_pair_proofs.get(id(row), {})) if standalone else {},
            "paired_raw_box_rescue": dict(raw_gate_diag) if standalone else {},
            "semantic_target_clear_hint_pixels": int(diag.get("semantic_target_clear_hint_pixels", 0) or 0),
            "semantic_target_clear_hint_admitted": bool(diag.get("semantic_target_clear_hint_admitted", False)),
            "semantic_clear_backend": str(diag.get("clear_backend") or ""),
            "semantic_source_recolor": dict(diag.get("source_recolor") or {}),
        }
        patch_matches.append(BubblePatchMatch(
            source_id, target_id, conf, 0.0, 1.0, 0.0, 1.0,
            ["koharu-semantic-first", "registered-raster-ink", "background-preserving"],
        ))
        # Expanded raw-box rescues are deliberately broad, but they are never
        # allowed to overwrite a region already claimed by an earlier accepted
        # transfer on this page.  This is critical for a slanted letter whose
        # detector raw box geometrically overlaps a separate caption bubble.
        if text_img is not None and raw_gate is not None and cv2.countNonZero(active_exclude) > 0:
            protected = active_exclude > 0
            text_img[protected] = rendered[protected]
            write_mask[protected] = 0
            if isinstance(source_ink_mask, np.ndarray) and source_ink_mask.shape == shape:
                source_ink_mask[protected] = 0
            for diag_key in ("clear_mask", "target_ink_mask"):
                diag_mask = diag.get(diag_key)
                if isinstance(diag_mask, np.ndarray) and diag_mask.shape == shape:
                    diag_mask = diag_mask.copy()
                    diag_mask[protected] = 0
                    diag[diag_key] = diag_mask
            diag["protected_prior_write_pixels"] = int(np.count_nonzero(protected))

        if text_img is None or cv2.countNonZero(write_mask) <= 0:
            rec.reason = str(diag.get("reason") or "koharu_semantic_component_transfer_failed")
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue

        rendered = text_img
        composite = np.maximum(composite, write_mask)
        if standalone and bool(raw_gate_diag.get("expanded", False)):
            raw_rescue_allow = np.maximum(raw_rescue_allow, write_mask)
        diag_clear = diag.get("clear_mask")
        if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
            clear_all = np.maximum(clear_all, diag_clear)
        use = write_mask > 0
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        layer[use, :3] = rgb[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        rec.applied = True
        rec.mask_iou = 1.0
        rec.target_coverage = 1.0
        rec.spill_ratio = 0.0
        rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(gate)))
        rec.clarity_mode = "koharu-semantic-open-text-ink-transfer"
        _evaluate_content_completeness(
            rec, source_ink_mask, diag.get("target_ink_mask"), rendered, cfg,
            target_invalidation_mask=write_mask,
        )
        if not bool(getattr(rec, "content_complete", False)):
            repaired, repaired_mask, repair_diag = _repair_content_region(
                rec, rendered, aligned_source, target, write_mask, gate,
                source_ink_mask, diag.get("target_ink_mask"), cfg,
            )
            rec.meta["content_auto_repair"] = repair_diag
            if bool(repair_diag.get("improved", False)):
                rendered = repaired
                write_mask = np.maximum(write_mask, repaired_mask)
                composite = np.maximum(composite, write_mask)
                clear_all = np.maximum(clear_all, repaired_mask)
                use = write_mask > 0
                rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
                layer[use, :3] = rgb[use]
                layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        boundary_touch = bool(diag.get("boundary_touch", False))
        strict_raw_safe = bool(standalone) and _strict_raw_group_semantic_result_safe(
            paired_proof, raw_gate_diag, diag, rec, cfg
        )
        boundary_touch_safe = _semantic_boundary_touch_result_safe(
            rec, cfg, standalone=bool(standalone), boundary_touch=boundary_touch
        )
        rec.meta["strict_raw_group_verified_safe"] = bool(strict_raw_safe)
        rec.meta["boundary_touch_verified_safe"] = bool(boundary_touch_safe)
        if (standalone and not strict_raw_safe) or (boundary_touch and not boundary_touch_safe) or not bool(getattr(rec, "content_complete", False)):
            rec.reason = "applied_koharu_semantic_review_candidate"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = (
                "standalone_open_text_semantic_recovery" if standalone else
                "source_text_cluster_touches_candidate_boundary" if boundary_touch else
                "semantic_content_incomplete"
            )
            rec.restorable = True
            rec.editable = True
        else:
            rec.reason = "applied_koharu_semantic_registered_components"
        records.append(rec)

    result = MaskTransferResult(rendered, layer, composite, patch_matches, records, clear_all)
    # Detector-undersegmentation rescue writes beyond the narrow Koharu polygon,
    # but only after a two-sided text proof and successful source-guided colour
    # reconstruction. Preserve that exact write footprint through the downstream
    # Koharu PROTECT guard; surrounding artwork remains protected byte-for-byte.
    if cv2.countNonZero(raw_rescue_allow) > 0:
        result.authority_allow_mask = raw_rescue_allow
    return result




def transfer_ocr_guided_text_units(
    aligned_source: np.ndarray,
    target: np.ndarray,
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    matches: list[UnitMatch],
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
    *,
    exclude_mask: np.ndarray | None = None,
) -> MaskTransferResult:
    """Recover OCR-confirmed text regions without OCR re-lettering.

    This is a completeness fallback for regions that closed-balloon/paired-diff
    geometry missed (structured boxes, burst balloons and open text).  OCR only
    provides the correspondence and polygon gate.  Final pixels come exclusively
    from the registered source raster; target Japanese glyph components are
    inpainted individually.  Low-confidence matches remain reversible review
    candidates instead of being silently discarded.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.shape[:2] != shape:
        raise ValueError("aligned_source must be in target coordinates")
    if exclude_mask is None or exclude_mask.shape != shape:
        exclude_mask = np.zeros(shape, np.uint8)

    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite = np.zeros(shape, np.uint8)
    clear_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    patch_matches: list[BubblePatchMatch] = []
    src = {u.id: u for u in source_units}
    dst = {u.id: u for u in target_units}
    candidate_floor = float(getattr(cfg, "ocr_guided_candidate_min_match_confidence", 0.42))
    auto_floor = float(getattr(cfg, "ocr_guided_auto_apply_min_match_confidence", 0.64))
    ocr_floor = float(getattr(cfg, "ocr_guided_min_ocr_confidence", 0.45))
    max_overlap = float(getattr(cfg, "ocr_guided_max_existing_overlap", 0.18))

    for match in matches:
        if match.relation != "one_to_one" or float(match.confidence) < candidate_floor:
            continue
        su = src.get(match.source_unit_id); tu = dst.get(match.target_unit_id)
        if su is None or tu is None or not str(su.text or "").strip():
            continue
        if min(float(su.confidence), float(tu.confidence)) < ocr_floor:
            continue
        gate = _ocr_guided_region_gate(su, tu, registration, shape, cfg)
        if gate is None or cv2.countNonZero(gate) == 0:
            continue
        overlap = int(np.count_nonzero((gate > 0) & (exclude_mask > 0))) / max(1, cv2.countNonZero(gate))
        if overlap > max_overlap:
            continue

        source_id = str(su.bubble_id or su.id)
        target_id = str(tu.bubble_id or tu.id)
        confidence = float(min(match.confidence, su.confidence, tu.confidence))
        rec = MaskTransferRecord(source_id, target_id, confidence, False, "not_applied")
        rec.geometry_mode = "ocr_guided_components"
        rec.source_bbox = tuple(int(round(v)) for v in su.bbox)
        rec.target_bbox = tuple(int(round(v)) for v in tu.bbox)
        patch_matches.append(BubblePatchMatch(
            source_id, target_id, confidence, float(match.cost), 0.0, 0.0, 1.0,
            ["ocr-geometry-only", "registered-raster-ink"],
        ))

        text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
            aligned_source, rendered, gate, cfg
        )
        rec.sr_backend = "ocr-guided-components"
        rec.sr_scale = 1.0
        rec.mask_iou = 1.0
        rec.spill_ratio = 0.0
        rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(gate)))
        if text_img is None:
            # A geometrically credible match with insufficient raster evidence is
            # intentionally visible to QA instead of disappearing from the run.
            rec.reason = str(diag.get("reason") or "ocr_guided_component_transfer_failed")
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue

        rendered = text_img
        composite = np.maximum(composite, write_mask)
        diag_clear = diag.get("clear_mask")
        if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
            clear_all = np.maximum(clear_all, diag_clear)
        use = write_mask > 0
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        layer[use, :3] = rgb[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        rec.applied = True
        rec.target_coverage = 1.0
        rec.clarity_mode = "ocr-guided-ink-transfer"
        _evaluate_content_completeness(
            rec, source_ink_mask, diag.get("target_ink_mask"), rendered, cfg
        )
        low = confidence < auto_floor
        boundary_touch = bool(diag.get("boundary_touch", False))
        if low or boundary_touch:
            rec.reason = "applied_ocr_guided_review_candidate"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = (
                "source_text_cluster_touches_candidate_boundary" if boundary_touch
                else "low_confidence_ocr_geometry_match"
            )
            rec.restorable = True
            rec.editable = True
        else:
            rec.reason = "applied_ocr_guided_registered_components"
        records.append(rec)
        exclude_mask = np.maximum(exclude_mask, write_mask)

    return MaskTransferResult(rendered, layer, composite, patch_matches, records, clear_all)





def _target_mask_is_white_container(target: np.ndarray, region_mask: np.ndarray, cfg: MaskReplaceConfig) -> tuple[bool, dict[str, float | bool]]:
    region_u8 = (region_mask > 0).astype(np.uint8) * 255
    pixels = int(cv2.countNonZero(region_u8))
    if pixels <= 0:
        return False, {"is_white": False, "reason": "empty_region", "paper_ratio": 0.0, "robust_spread": 255.0}
    paper = white_container_paper_mask(target, region_u8, None)
    paper_ratio = float(cv2.countNonZero(paper) / max(1, pixels))
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target.astype(np.uint8)
    vals = gray[region_u8 > 0]
    robust_spread = float(np.percentile(vals, 90.0) - np.percentile(vals, 10.0)) if vals.size > 0 else 255.0
    min_ratio = float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68))
    max_spread = float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0))
    ok = bool(paper_ratio >= min_ratio and robust_spread <= max_spread)
    return ok, {
        "is_white": ok,
        "reason": "paper_ratio_and_spread" if ok else "not_uniform_white",
        "paper_ratio": paper_ratio,
        "robust_spread": robust_spread,
        "min_paper_ratio": min_ratio,
        "max_robust_spread": max_spread,
    }


def _maybe_apply_white_source_clarity(
    source_image: np.ndarray,
    target_image: np.ndarray,
    target_region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply source clarity with explicit SOURCE/TARGET coordinate authorities.

    Paired aligned routes use one target-space mask for both images. Rigid routes
    keep the original SOURCE raster and therefore have a source-space mask plus a
    separate target-space white-container mask. Never index an original SOURCE
    image with TARGET geometry when the editions have different dimensions.
    """
    enabled = bool(getattr(cfg, "direct_white_clarity_enhance_enabled", True))
    white_ok, white_diag = _target_mask_is_white_container(target_image, target_region_mask, cfg)
    diag: dict[str, object] = {"enabled": enabled, "target_region": white_diag, "applied": False, "reason": "disabled" if not enabled else "target_not_white"}
    if not enabled or not white_ok:
        return source_image.copy(), diag
    sm = target_region_mask if source_region_mask is None else np.asarray(source_region_mask)
    if sm.shape != source_image.shape[:2]:
        diag["reason"] = "source_mask_shape_mismatch"
        diag["source_shape"] = list(source_image.shape[:2])
        diag["source_mask_shape"] = list(sm.shape)
        return source_image.copy(), diag
    source_text = target_text_mask_in_container(source_image, sm)
    enhanced, clarity_diag = _enhance_white_source_patch(
        source_image,
        sm,
        source_text,
        enabled=enabled,
        alpha_gamma=float(getattr(cfg, "direct_white_clarity_alpha_gamma", 1.0)),
        black_boost=int(getattr(cfg, "direct_white_clarity_black_boost", 0)),
        pure_white_floor=int(getattr(cfg, "direct_white_clarity_pure_white_floor", 248)),
        min_text_pixels=int(getattr(cfg, "direct_white_clarity_min_text_pixels", 18)),
    )
    diag.update({"applied": bool(clarity_diag.get("applied", False)), "reason": str(clarity_diag.get("reason", "ok")), "source": clarity_diag})
    return enhanced, diag


def _rigid_source_raster(
    source: np.ndarray,
    target_reference: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    target_shape: tuple[int, int],
    base_scale: float,
    cfg: MaskReplaceConfig,
    *,
    source_gray: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, dict[str, object]] | None:
    """Place the complete source lettering raster with uniform scale only.

    Returns ``(soft_alpha, ink_mask, scale, dx, dy, ink_coverage, mask_containment, clarity_diag)``
    in target coordinates.  ``soft_alpha`` is derived from the original source
    grayscale raster as one field; characters are never split/reassembled.
    """
    sbox = _bbox_from_mask(source_mask); tbox = _bbox_from_mask(target_mask)
    if sbox is None or tbox is None:
        return None
    sx0, sy0, sx1, sy1 = sbox
    src_mask = source_mask.copy()
    sinset = max(0, int(getattr(cfg, "rigid_container_source_inset_px", 1)))
    if sinset:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sinset * 2 + 1, sinset * 2 + 1))
        er = cv2.erode(src_mask, k)
        if cv2.countNonZero(er) > 0:
            src_mask = er
    target_inner = target_mask.copy()
    tinset = max(0, int(getattr(cfg, "rigid_container_target_inset_px", 1)))
    if tinset:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tinset * 2 + 1, tinset * 2 + 1))
        er = cv2.erode(target_inner, k)
        if cv2.countNonZero(er) > 0:
            target_inner = er

    clarity_source, clarity_diag = _maybe_apply_white_source_clarity(source, target_reference, target_mask, cfg, source_region_mask=source_mask)
    gray_source = clarity_source if clarity_diag.get("applied", False) else source
    gray = source_gray if (source_gray is not None and not bool(clarity_diag.get("applied", False))) else cv2.cvtColor(gray_source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if gray.dtype != np.float32:
        gray = gray.astype(np.float32)
    crop_gray = gray[sy0:sy1, sx0:sx1]
    crop_mask = src_mask[sy0:sy1, sx0:sx1]
    vals = crop_gray[crop_mask > 0]
    if vals.size < 8:
        return None
    paper = float(np.percentile(vals, float(getattr(cfg, "rigid_container_paper_percentile", 90.0))))
    if paper < 180.0:
        return None
    # Normalise scan paper to white and derive one continuous opacity field.  On
    # white this is exactly equivalent to the normalised source grayscale raster,
    # including antialiasing/halftone edge values; no glyph component logic occurs.
    alpha = np.clip((paper - crop_gray) / max(1.0, paper), 0.0, 1.0).astype(np.float32)
    floor = float(np.clip(getattr(cfg, "rigid_container_alpha_floor", 0.055), 0.0, 0.30))
    alpha[alpha < floor] = 0.0
    alpha *= (crop_mask.astype(np.float32) / 255.0)
    if np.count_nonzero(alpha >= max(0.08, floor)) < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)):
        return None

    scx, scy = _centroid(src_mask); tcx, tcy = _centroid(target_inner)
    rel_cx, rel_cy = scx - sx0, scy - sy0
    min_factor = float(np.clip(getattr(cfg, "rigid_container_min_scale_factor", 0.94), 0.80, 1.0))
    step = float(np.clip(getattr(cfg, "rigid_container_scale_step", 0.01), 0.005, 0.05))
    factors = []
    f = 1.0
    while f >= min_factor - 1e-8:
        factors.append(f); f -= step
    max_shift = max(0, int(getattr(cfg, "rigid_container_max_shift_px", 7)))
    min_ink_cov = float(np.clip(getattr(cfg, "rigid_container_min_ink_coverage", 0.985), 0.90, 1.0))
    min_mask_contain = float(np.clip(getattr(cfg, "rigid_container_min_mask_containment", 0.955), 0.90, 1.0))
    offset_patience = max(6, int(getattr(cfg, "rigid_container_offset_patience", 24)))
    offsets = _ordered_offsets(max_shift)
    th, tw = target_shape
    best = None
    for factor in factors:
        scale = float(base_scale * factor)
        nw = max(1, int(round((sx1 - sx0) * scale))); nh = max(1, int(round((sy1 - sy0) * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
        ra = cv2.resize(alpha, (nw, nh), interpolation=interp)
        rm = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
        bx = int(round(tcx - rel_cx * scale)); by = int(round(tcy - rel_cy * scale))
        tested = 0
        for dx, dy in offsets:
            tested += 1
            px, py = bx + dx, by + dy
            xa, ya = max(0, px), max(0, py); xb, yb = min(tw, px + nw), min(th, py + nh)
            if xb <= xa or yb <= ya:
                continue
            aa = ra[ya - py:yb - py, xa - px:xb - px]
            mm = rm[ya - py:yb - py, xa - px:xb - px] > 0
            ink = aa >= max(0.08, floor)
            ink_n = int(np.count_nonzero(ink)); mask_n = int(np.count_nonzero(mm))
            if ink_n < int(getattr(cfg, "content_completeness_min_ink_pixels", 18)) or mask_n <= 0:
                continue
            tg = target_inner[ya:yb, xa:xb] > 0
            outer = target_mask[ya:yb, xa:xb] > 0
            ink_cov = float(np.count_nonzero(ink & tg) / ink_n)
            mask_cov = float(np.count_nonzero(mm & outer) / mask_n)
            # Prefer complete ink first, then source-container containment,
            # then the smallest possible nudge/shrink from paired geometry.
            score = 4.0 * ink_cov + mask_cov - 0.002 * (abs(dx) + abs(dy)) - 0.08 * (1.0 - factor)
            if best is None or score > best[0]:
                best = (score, ink_cov, mask_cov, scale, px, py, ra)
            if ink_cov >= min_ink_cov and mask_cov >= min_mask_contain and tested >= min(4, offset_patience):
                break
            if tested >= offset_patience and best is not None and best[1] >= min_ink_cov and best[2] >= min_mask_contain:
                break
        if best is not None and best[1] >= min_ink_cov and best[2] >= min_mask_contain:
            break
    if best is None or best[1] < min_ink_cov or best[2] < min_mask_contain:
        return None
    _, ink_cov, mask_cov, scale, px, py, ra = best
    full_alpha = np.zeros((th, tw), np.float32)
    nh, nw = ra.shape
    xa, ya = max(0, px), max(0, py); xb, yb = min(tw, px + nw), min(th, py + nh)
    if xb <= xa or yb <= ya:
        return None
    local = ra[ya - py:yb - py, xa - px:xb - px].copy()
    local *= (target_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    full_alpha[ya:yb, xa:xb] = np.maximum(full_alpha[ya:yb, xa:xb], local)
    ink_mask = (full_alpha >= max(0.08, floor)).astype(np.uint8) * 255
    # dx/dy are reported relative to the pure centroid placement at the selected scale.
    base_x = tcx - rel_cx * scale; base_y = tcy - rel_cy * scale
    return full_alpha, ink_mask, scale, float(px - base_x), float(py - base_y), float(ink_cov), float(mask_cov), clarity_diag




def _rigid_container_full_patch(
    source: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    target_shape: tuple[int, int],
    scale: float,
    dx: float,
    dy: float,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Render the whole source container interior into target coordinates.

    Unlike alpha-only lettering compositing, this copies the source white paper
    + Chinese text as one locked patch.  For same-layout editions this is the
    most faithful route: if the container shapes match, Japanese cannot leak and
    Chinese glyphs cannot be partially clipped by target-side text geometry.
    """
    sbox = _bbox_from_mask(source_mask)
    if sbox is None:
        return None
    sx0, sy0, sx1, sy1 = sbox
    crop_img = source[sy0:sy1, sx0:sx1]
    crop_mask = source_mask[sy0:sy1, sx0:sx1]
    if crop_img.size == 0 or cv2.countNonZero(crop_mask) <= 0:
        return None

    scx, scy = _centroid(source_mask)
    tcx, tcy = _centroid(target_mask)
    rel_cx, rel_cy = scx - sx0, scy - sy0
    px = int(round((tcx - rel_cx * scale) + dx))
    py = int(round((tcy - rel_cy * scale) + dy))
    nw = max(1, int(round((sx1 - sx0) * scale)))
    nh = max(1, int(round((sy1 - sy0) * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    resized_img = cv2.resize(crop_img, (nw, nh), interpolation=interp)
    resized_mask = cv2.resize(crop_mask, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # One canonical border-safe envelope is shared by patch placement and the
    # later gap-fill stage.  This prevents source box outlines from darkening a
    # preserved TARGET outline after the initial inset.
    write_mask = _rigid_target_write_envelope(target_mask, cfg)

    th, tw = target_shape
    full_rgb = np.zeros((th, tw, 3), np.uint8)
    full_alpha = np.zeros((th, tw), np.float32)
    xa, ya = max(0, px), max(0, py)
    xb, yb = min(tw, px + nw), min(th, py + nh)
    if xb <= xa or yb <= ya:
        return None
    local_img = resized_img[ya - py:yb - py, xa - px:xb - px]
    local_mask = resized_mask[ya - py:yb - py, xa - px:xb - px].astype(np.float32) / 255.0
    local_mask *= (write_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    blur_px = max(0, int(getattr(cfg, 'rigid_container_full_patch_mask_blur_px', 1)))
    if blur_px > 0:
        k = blur_px * 2 + 1
        local_mask = cv2.GaussianBlur(local_mask, (k, k), 0)
        local_mask = np.clip(local_mask, 0.0, 1.0)
        local_mask *= (write_mask[ya:yb, xa:xb].astype(np.float32) / 255.0)
    full_rgb[ya:yb, xa:xb] = local_img
    full_alpha[ya:yb, xa:xb] = local_mask
    return full_rgb, (full_alpha * 255.0).astype(np.uint8)




def transfer_rigid_container_rasters(
    source: np.ndarray,
    target_reference: np.ndarray,
    base_image: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    """Transfer safe white speech/text containers without distorting glyphs.

    This is the preferred path for same-layout B/W Chinese -> colour Japanese
    pairs.  It deliberately works from the *original* source page rather than an
    affine/dense-flow warped source.  Pair geometry supplies one uniform local
    scale and translation; only source lettering opacity is moved. TARGET remains
    the sole background/colour authority.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target_reference.shape[:2]
    empty_layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    empty_mask = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "rigid_container_transfer_enabled", True)):
        return MaskTransferResult(base_image.copy(), empty_layer, empty_mask.copy(), [], [], empty_mask.copy())
    if base_image.shape[:2] != shape:
        raise ValueError("base_image and target_reference must share target coordinates")

    # v0.8.28 performance: these multi-megapixel colour-space conversions are
    # page invariants. Compute them once instead of once per bubble/container.
    source_gray_u8 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    source_gray_f32 = source_gray_u8.astype(np.float32)
    source_hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    target_gray = cv2.cvtColor(target_reference, cv2.COLOR_BGR2GRAY)
    target_hsv = cv2.cvtColor(target_reference, cv2.COLOR_BGR2HSV)

    target_by_id = {b.id: b for b in target_bubbles}
    target_by_source = {
        str(b.meta.get("paired_source_id")): b
        for b in target_bubbles
        if str(b.meta.get("paired_source_id") or "")
    }
    pairs: list[tuple[BubbleInstance, BubbleInstance]] = []
    used_target_ids: set[str] = set()
    for index, sb in enumerate(source_bubbles):
        tid = str(sb.meta.get("paired_target_id") or "")
        tb = target_by_id.get(tid) if tid else None
        # Target-driven OCR-free completion stores the relation on TARGET only.
        # Older code silently dropped these pairs whenever at least one ordinary
        # paired_target_id existed in the same batch.
        if tb is None:
            tb = target_by_source.get(str(sb.id))
        if tb is None and len(source_bubbles) == len(target_bubbles) and index < len(target_bubbles):
            candidate = target_bubbles[index]
            if candidate.id not in used_target_ids:
                tb = candidate
        if tb is not None and tb.id not in used_target_ids:
            pairs.append((sb, tb))
            used_target_ids.add(tb.id)

    rendered = base_image.copy(); layer = empty_layer.copy(); composite = empty_mask.copy(); clear_all = empty_mask.copy()
    records: list[MaskTransferRecord] = []; matches: list[BubblePatchMatch] = []
    for sb, tb in pairs:
        sm_raw = _bubble_mask(sb, source.shape[:2]); tm_raw = _bubble_mask(tb, shape)
        # v0.8.26: detector geometry and true writable container interior are
        # different layers. Solidify both masks before any clear/clip operation.
        # This prevents target Japanese glyphs from surviving in mask notches and
        # prevents source Chinese strokes from being clipped by source-side text.
        sm = _solidify_container_mask(sm_raw, cfg); tm = _solidify_container_mask(tm_raw, cfg)
        sbox = _bbox_from_mask(sm); tbox = _bbox_from_mask(tm)
        conf = float(min(sb.confidence, tb.confidence))
        # Target-driven recovery is used when the source page is much larger
        # than the HD target (common for the Chinese scan).  Its page scale is
        # intentionally below the ordinary rigid-container range; geometry is
        # still locked by the target mask and the source raster is resized with
        # one scalar only.  Keep the stricter range for normal candidates.
        eligibility_cfg = cfg
        placement_cfg = cfg
        if bool(sb.meta.get("target_driven_recovery")) or str(sb.meta.get("backend", "")) == "unseeded_white":
            if hasattr(cfg, "model_copy"):
                eligibility_cfg = cfg.model_copy(deep=True)
            else:
                eligibility_cfg = copy.copy(cfg)
            object.__setattr__(eligibility_cfg, "rigid_container_min_uniform_scale", min(
                0.20, float(getattr(cfg, "rigid_container_min_uniform_scale", 0.35))
            ))
            # OCR-free completion candidates already passed a target-space white-container
            # detector plus a registered ink-change gate. Their SOURCE crop can still
            # look less white than the TARGET because Chinese glyphs are denser or the
            # scan is darker/noisier. Relax the source/shape thresholds so missed open
            # balloons and starbursts can upgrade from warped text transfer to the rigid
            # original-source patch path.
            object.__setattr__(eligibility_cfg, "rigid_container_min_source_white_ratio", min(
                0.48, float(getattr(cfg, "rigid_container_min_source_white_ratio", 0.78))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_target_white_ratio", min(
                0.55, float(getattr(cfg, "rigid_container_min_target_white_ratio", 0.75))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_fill_ratio", min(
                0.18, float(getattr(cfg, "rigid_container_min_fill_ratio", 0.55))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_fill_ratio", min(
                0.14, float(getattr(cfg, "rigid_container_spiky_min_fill_ratio", 0.30))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_white_ratio", min(
                0.55, float(getattr(cfg, "rigid_container_spiky_min_white_ratio", 0.78))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_ink_coverage", min(
                0.88, float(getattr(cfg, "rigid_container_min_ink_coverage", 0.985))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_mask_containment", min(
                0.88, float(getattr(cfg, "rigid_container_min_mask_containment", 0.955))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_min_scale_factor", min(
                0.88, float(getattr(cfg, "rigid_container_min_scale_factor", 0.94))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_max_shift_px", max(
                12, int(getattr(cfg, "rigid_container_max_shift_px", 7))
            ))
            object.__setattr__(eligibility_cfg, "rigid_container_offset_patience", max(
                48, int(getattr(cfg, "rigid_container_offset_patience", 24))
            ))
            placement_cfg = eligibility_cfg
            if bool(sb.meta.get("target_driven_colored")):
                # A coloured burst is intentionally not white on the target;
                # its geometry is still trusted, while source pixels provide
                # the Chinese raster and the whole target shape is cleared.
                object.__setattr__(eligibility_cfg, "rigid_container_min_target_white_ratio", 0.0)
                object.__setattr__(eligibility_cfg, "rigid_container_max_target_saturation_median", 255.0)
                object.__setattr__(eligibility_cfg, "rigid_container_min_target_dark_ratio", 0.002)
                object.__setattr__(eligibility_cfg, "rigid_container_min_fill_ratio", 0.30)
                object.__setattr__(eligibility_cfg, "rigid_container_spiky_min_fill_ratio", 0.20)
                object.__setattr__(eligibility_cfg, "rigid_container_max_dark_ratio", 1.0)
        eligible, diag = _rigid_container_pair_eligible(
            source, target_reference, sm, tm, eligibility_cfg,
            source_gray=source_gray_u8, source_hsv=source_hsv,
            target_gray=target_gray, target_hsv=target_hsv,
        )
        if not eligible:
            continue  # caller sends unhandled regions down the legacy/saturated paths
        text_support_ok, text_support_diag = _rigid_source_text_support(
            source, target_reference, sm, tm, eligibility_cfg,
        )
        diag["source_text_support"] = text_support_diag
        if not text_support_ok:
            continue
        base_scale = float(diag.get("uniform_scale", 1.0))
        placed = _rigid_source_raster(source, target_reference, sm, tm, shape, base_scale, placement_cfg, source_gray=source_gray_f32)
        if placed is None:
            continue
        alpha, source_ink_mask, scale, dx, dy, ink_cov, mask_cov, white_clarity_diag = placed
        source_ink_mask, source_boundary_removed = remove_container_boundary_line_components(source_ink_mask, tm)
        if source_boundary_removed > 0:
            alpha[source_ink_mask == 0] = 0.0
            diag["source_boundary_line_pixels_removed"] = int(source_boundary_removed)
        placed_ink_diag = {"valid": True, "reason": "disabled"}
        if bool(getattr(eligibility_cfg, "rigid_container_placed_ink_shape_gate_enabled", True)):
            placed_ink_ok, placed_ink_diag = _support_mask_textlike(source_ink_mask, tm, eligibility_cfg)
            if not placed_ink_ok:
                continue
        diag["placed_ink_shape"] = placed_ink_diag

        # v1.0.7: the target container is geometry truth, but TARGET artwork/
        # colour is also the only background truth. Clear only compact Japanese
        # lettering, never the whole interior, and draw only SOURCE ink opacity as
        # neutral black. SOURCE paper RGB is forbidden from this path.
        border_safe_envelope = _rigid_target_write_envelope(tm, placement_cfg)
        clear = np.zeros(shape, np.uint8)
        full_clear_diag = {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "target_colored_or_disabled",
            "white_full_clear_pixels": 0,
        }
        target_colored = bool(sb.meta.get("target_driven_colored"))
        full_clear_envelope = border_safe_envelope
        if (
            not target_colored
            and bool(getattr(cfg, "white_container_full_clear_enabled", True))
        ):
            paper_mask = white_container_paper_mask(target_reference, tm, source_ink_mask)
            candidate_clear_env, clear_env_diag = white_container_write_envelope(
                target_reference, tm, paper_mask,
                inset_px=max(0, int(getattr(cfg, "white_container_clear_inset_px", 0))),
                border_guard_px=max(0, int(getattr(cfg, "white_container_clear_border_guard_px", 0))),
            )
            if cv2.countNonZero(candidate_clear_env) > 0:
                full_clear_envelope = candidate_clear_env
            full_out, full_mask, full_clear_diag = clear_uniform_white_container_interior(
                rendered, target_reference, full_clear_envelope,
                min_paper_ratio=float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68)),
                max_robust_spread=float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0)),
            )
            full_clear_diag["clear_envelope"] = clear_env_diag
            if bool(full_clear_diag.get("white_full_clear_applied", False)):
                rendered = full_out
                clear = full_mask

        if not bool(full_clear_diag.get("white_full_clear_applied", False)):
            clear = target_text_mask_in_container(target_reference, border_safe_envelope)
            if cv2.countNonZero(clear) > 0:
                d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
                clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
                clear = cv2.bitwise_and(clear, border_safe_envelope)
                cleaned = cv2.inpaint(rendered, clear, 2.5, cv2.INPAINT_TELEA)
                rendered[clear > 0] = cleaned[clear > 0]

        write_alpha8: np.ndarray
        backend = "rigid-container-text-only"
        clarity = "target-background-source-ink-only"
        match_notes = ["rigid_container_text_only", "target_background_preserved", f"uniform_scale={scale:.6f}", f"ink_coverage={ink_cov:.5f}", f"mask_containment={mask_cov:.5f}"]
        gap_fill_diag = {"added_pixels": 0, "enabled": False}
        a = np.clip(alpha, 0.0, 1.0)
        a *= (border_safe_envelope.astype(np.float32) / 255.0)
        # Neutral black preserves the original Chinese raster opacity/topology but
        # cannot carry white/gray SOURCE paper or scan colour into TARGET.
        rendered = np.clip(rendered.astype(np.float32) * (1.0 - a[..., None]), 0, 255).astype(np.uint8)
        # A conservative clear can leave isolated old-Japanese punctuation dots
        # near the inset envelope. Remove only TARGET-origin tiny components that
        # have no SOURCE Chinese support; real Chinese punctuation is protected.
        rendered, speck_mask, speck_diag = cleanup_target_residual_specks(
            rendered, target_reference, border_safe_envelope, source_ink_mask, clear,
            white_container=True, inpaint_radius=2.0,
        )
        if int(speck_diag.get("residual_specks_removed", 0)) > 0:
            match_notes.append(f"residual_specks_removed={int(speck_diag.get('residual_specks_removed', 0))}")
        clear = np.maximum(clear, speck_mask)
        write_alpha8 = np.maximum(clear, (a * 255.0).astype(np.uint8))

        border_diag = {
            "enabled": bool(getattr(cfg, "rigid_container_full_patch_preserve_target_border", True)),
            "protected_pixels": 0,
            "changed_before_restore": 0,
            "changed_after_restore": 0,
        }
        if bool(getattr(placement_cfg, "rigid_container_full_patch_preserve_target_border", True)):
            if bool(full_clear_diag.get("white_full_clear_applied", False)):
                # Restore only the actual HD outline/rules.  Direct's OCR-free
                # completion masks are interior authorities, so the HD outline can
                # sit a few pixels *outside* ``tm`` and be invisible to the old
                # detector. DirectPatchConfig alone opts into a narrow probe ring;
                # Mask/Hybrid have no such field and retain their exact behaviour.
                probe_px=max(0,int(getattr(cfg,"direct_rigid_target_border_probe_dilate_px",0)))
                border_probe=tm
                if probe_px>0:
                    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(probe_px*2+1,probe_px*2+1))
                    border_probe=cv2.dilate((tm>0).astype(np.uint8)*255,k,iterations=1)
                protected_ring = target_container_border_mask(
                    target_reference, border_probe, band_px=max(4,probe_px+1)
                )
                fringe_px=max(0,int(getattr(cfg,"direct_rigid_target_border_restore_fringe_px",0)))
                if probe_px>0 and fringe_px>0 and cv2.countNonZero(protected_ring)>0:
                    fk=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(fringe_px*2+1,fringe_px*2+1))
                    protected_ring=cv2.dilate(protected_ring,fk,iterations=1)
                full_clear_diag["target_border_probe_dilate_px"]=int(probe_px)
                full_clear_diag["target_border_restore_fringe_px"]=int(fringe_px)
            else:
                protected_ring = cv2.bitwise_and(tm, cv2.bitwise_not(border_safe_envelope))
            ring_sel = protected_ring > 0
            border_diag["protected_pixels"] = int(np.count_nonzero(ring_sel))
            if np.any(ring_sel):
                before_delta = np.max(
                    np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
                )
                border_diag["changed_before_restore"] = int(np.count_nonzero(ring_sel & (before_delta > 0)))
                if bool(getattr(placement_cfg, "rigid_container_exact_target_border_restore", True)):
                    rendered[ring_sel] = target_reference[ring_sel]
                    write_alpha8[ring_sel] = 0
                    clear[ring_sel] = 0
                after_delta = np.max(
                    np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
                )
                border_diag["changed_after_restore"] = int(np.count_nonzero(ring_sel & (after_delta > 0)))

        composite = np.maximum(composite, write_alpha8); clear_all = np.maximum(clear_all, clear)
        use = write_alpha8 > 0
        if np.any(use):
            layer[use, :3] = rendered[use][:, ::-1]
        layer[..., 3] = np.maximum(layer[..., 3], write_alpha8)

        matches.append(BubblePatchMatch(
            sb.id, tb.id, conf, 1.0 - conf, mask_cov, 0.0, 1.0,
            match_notes,
        ))
        rec = MaskTransferRecord(sb.id, tb.id, conf, True, "applied_rigid_container_raster")
        rec.sr_backend = backend; rec.sr_scale = scale
        rec.geometry_mode = "rigid_uniform_container"; rec.clarity_mode = clarity
        rec.mask_iou = mask_cov; rec.target_coverage = ink_cov; rec.spill_ratio = max(0.0, 1.0 - ink_cov)
        rec.local_dx = dx; rec.local_dy = dy; rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(tm)))
        if sbox is not None: rec.source_bbox = sbox
        if tbox is not None: rec.target_bbox = tbox
        rec.meta["white_source_clarity"] = white_clarity_diag
        rec.meta["mask_write_gap_fill"] = gap_fill_diag
        rec.meta["target_border_preservation"] = border_diag
        rec.meta["white_container_full_clear"] = full_clear_diag
        rec.meta["source_text_support"] = text_support_diag
        rec.meta["placed_ink_shape"] = placed_ink_diag
        if int(border_diag.get("changed_after_restore", 0)) > 0:
            rec.review_required = True
            rec.review_reason = "protected_target_border_changed"
            rec.restorable = True
            rec.editable = True
        target_ink, _ = _compact_container_ink(target_reference, clear, 190, cfg, gray=target_gray)
        min_cov = float(getattr(cfg, "rigid_container_acceptance_min_source_coverage", 0.985))
        max_res = float(getattr(cfg, "rigid_container_acceptance_max_target_residual", 0.02))
        _evaluate_content_completeness(
            rec,
            source_ink_mask,
            target_ink,
            rendered,
            cfg,
            tolerance_px=3,
            min_source_coverage=min_cov,
            max_target_residual=max_res,
        )
        # The rigid text-only path has no target-sized SOURCE RGB patch.  Older
        # code retained a dead auto-repair branch referencing an undefined
        # ``patch_rgb``; it was guarded by ``patch is not None`` even though
        # ``patch`` was always None.  Keep this route fail-closed: incomplete
        # content is sent to review instead of inventing pixels from an invalid
        # raster source. Other mask routes that own a valid aligned source image
        # still use ``_repair_content_region`` normally.
        # A full-container clear + near-total source-raster containment is a stronger
        # success criterion than legacy "pixels were written".
        if not rec.content_complete:
            rec.review_required = True; rec.review_reason = "rigid_container_content_check_failed"; rec.restorable = True; rec.editable = True
        elif rec.repair_succeeded and rec.review_reason == "rigid_container_content_check_failed":
            rec.review_required = False; rec.review_reason = ""
        records.append(rec)
    return MaskTransferResult(rendered, layer, composite, matches, records, clear_all)


def _fast_dark_pixel_clear(
    image: np.ndarray,
    clear_envelope: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, dict]:
    """Fast white-container clear that removes only dark target glyph pixels.

    Returns ``(image_or_none, actual_clear_mask, diagnostics)``. ``None`` means
    the local region is not paper-like enough and callers should use the normal
    component/inpaint path instead.
    """
    shape = image.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "fast_dark_pixel_clear_enabled", True)):
        return None, empty, {"enabled": False, "reason": "disabled"}
    env = (clear_envelope > 0)
    n = int(np.count_nonzero(env))
    if n < 16:
        return None, empty, {"enabled": True, "reason": "empty_envelope"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    white_ratio = float(np.mean(gray[env] >= 220))
    min_white = float(getattr(cfg, "fast_dark_pixel_clear_min_white_ratio", 0.72))
    if white_ratio < min_white:
        return None, empty, {
            "enabled": True, "reason": "not_white_container",
            "white_ratio": white_ratio, "min_white_ratio": min_white,
        }
    threshold = int(getattr(cfg, "fast_dark_pixel_clear_threshold", 185))
    dark = ((gray <= threshold) & env).astype(np.uint8) * 255
    if cv2.countNonZero(dark) == 0:
        return image.copy(), dark, {
            "enabled": True, "reason": "no_dark_pixels", "white_ratio": white_ratio,
            "cleared_pixels": 0,
        }
    # Keep the operation glyph-local. A tiny dilation catches antialiasing without
    # turning the whole balloon into an inpaint request.
    dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    dark[~env] = 0
    bright = env & (gray >= 220)
    paper = np.median(image[bright], axis=0).astype(np.uint8) if int(np.count_nonzero(bright)) >= 12 else np.array([255, 255, 255], np.uint8)
    base = image.copy()
    base[dark > 0] = paper
    radius = float(getattr(cfg, "fast_dark_pixel_clear_inpaint_radius", 1.5))
    if radius > 0 and cv2.countNonZero(dark) > 0:
        refined = cv2.inpaint(base, dark, radius, cv2.INPAINT_TELEA)
        base[dark > 0] = refined[dark > 0]
    return base, dark, {
        "enabled": True,
        "reason": "applied",
        "white_ratio": white_ratio,
        "threshold": threshold,
        "cleared_pixels": int(cv2.countNonZero(dark)),
        "paper_bgr": paper.tolist(),
        "inpaint_radius": radius,
    }


def transfer_paired_diff_regions(
    aligned_source: np.ndarray,
    target: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    cfg: MaskReplaceConfig | None = None,
    *,
    render_source_label: str = "paired-dense-align",
    render_source_diagnostics: dict | None = None,
) -> MaskTransferResult:
    """Composite paired-diff regions directly in target coordinates.

    ``aligned_source`` is the raster source selected by the Mask policy. Detection
    may use dense/structural alignment, but v2.3.5 normally supplies a global-only
    shape-preserving SOURCE raster here. The target mask defines writable pixels;
    the mask may be refined without bending the Chinese glyph raster itself.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.ndim == 3 and aligned_source.shape[2] >= 3:
        source_sat_p90 = float(np.percentile(cv2.cvtColor(aligned_source, cv2.COLOR_BGR2HSV)[..., 1], 90.0))
    else:
        source_sat_p90 = 0.0
    if aligned_source.shape[:2] != shape:
        raise ValueError("aligned_source must be in target coordinates")

    source_by_id = {b.id: b for b in source_bubbles}
    target_by_id = {b.id: b for b in target_bubbles}
    # Paired-diff ids share the same numeric suffix. Keep explicit one-to-one
    # matches for project/debug compatibility without re-solving Hungarian geometry.
    pairs: list[tuple[BubbleInstance, BubbleInstance]] = []
    for sb in source_bubbles:
        suffix = sb.id.rsplit("-", 1)[-1]
        tb = target_by_id.get(f"diff-dst-{suffix}")
        if tb is not None:
            pairs.append((sb, tb))
    if not pairs and len(source_bubbles) == len(target_bubbles):
        pairs = list(zip(source_bubbles, target_bubbles))

    matches: list[BubblePatchMatch] = []
    records: list[MaskTransferRecord] = []
    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite_mask = np.zeros(shape, np.uint8)
    clear_mask_all = np.zeros(shape, np.uint8)
    saturated_seen = np.zeros(shape, np.uint8)

    def _is_page_furniture(box: tuple[int, int, int, int] | None, kind: str) -> bool:
        """Reject tiny edge text that is page furniture, not translated content."""
        if not bool(getattr(cfg, "paired_diff_protect_page_furniture", True)):
            return False
        if box is None or kind not in {"free_text", "complex_text"}:
            return False
        x0, y0, x1, y1 = box
        h, w = shape
        bw = max(1, x1 - x0); bh = max(1, y1 - y0)
        narrow = bw <= w * float(getattr(cfg, "paired_diff_page_furniture_max_width_ratio", 0.18))
        short = bh <= h * float(getattr(cfg, "paired_diff_page_furniture_max_height_ratio", 0.12))
        edge = (
            y1 <= h * float(getattr(cfg, "paired_diff_page_furniture_top_ratio", 0.10))
            or y0 >= h * float(getattr(cfg, "paired_diff_page_furniture_bottom_ratio", 0.965))
        )
        return bool(edge and narrow and short)

    for sb, tb in pairs:
        tm = _bubble_mask(tb, shape)
        sm = _bubble_mask(sb, sb.mask.shape if sb.mask is not None else shape)
        tbox = _bbox_from_mask(tm)
        sbox = _bbox_from_mask(sm)
        conf = float(min(sb.confidence, tb.confidence))
        match = BubblePatchMatch(
            sb.id, tb.id, conf, 1.0 - conf, 1.0, 0.0, 1.0,
            ["paired-target-space", f"source_mask_iou={tb.meta.get('paired_mask_iou', 0.0):.3f}"],
        )
        matches.append(match)
        rec = MaskTransferRecord(sb.id, tb.id, conf, False, "not_applied")
        photo_source = bool(tb.meta.get("photo_source")) or str(tb.meta.get("paired_diff_method", "")).startswith("photo_")
        rec.geometry_mode = "photo_pair" if photo_source else str(tb.meta.get("paired_diff_method") or "standard")
        if sbox:
            rec.source_bbox = sbox
        if tbox:
            rec.target_bbox = tbox
        rec.meta["source_proxy_geometry"] = paired_proxy_geometry_risk(
            sbox, tbox, float(tb.meta.get("paired_mask_iou", 0.0) or 0.0)
        )
        rec.meta["source_proxy_only"] = bool(rec.meta["source_proxy_geometry"].get("risky", False))
        if photo_source and sbox is not None:
            edge_sides = _edge_touch_sides(
                sbox, sm.shape[:2], int(getattr(cfg, "photo_pair_edge_clip_margin_px", 2))
            )
            rec.source_edge_clipped = bool(edge_sides)
            rec.source_edge_sides = ",".join(edge_sides)
        if tbox is None or cv2.countNonZero(tm) == 0:
            rec.reason = "empty_target_mask"
            records.append(rec)
            continue

        region_kind = str(tb.meta.get("paired_region_kind", "bubble"))
        if _is_page_furniture(tbox, region_kind):
            # This is intentionally not a review candidate: the safest output is
            # the untouched HD target, because the region is outside the dialogue
            # replacement scope. Do not create a clear mask or transfer record.
            matches.pop()
            continue

        # v2.3.6: photographed page edges are not trustworthy free-text sources.
        # A clipped open/complex region frequently contains the book gutter,
        # camera shadow, glare or a truncated translation.  Precise Mask/Hybrid
        # must not copy those pixels into the clean TARGET.  Complete interior
        # open text remains eligible; clipped regions become explicit review
        # records and can only be completed from a non-cropped source/manual text.
        edge_open_reject = bool(
            photo_source
            and rec.source_edge_clipped
            and (
                (region_kind == "free_text" and bool(getattr(cfg, "photo_pair_reject_edge_clipped_open_text", False)))
                or (region_kind == "complex_text" and bool(getattr(cfg, "photo_pair_reject_edge_clipped_complex_text", False)))
            )
        )
        if edge_open_reject:
            rec.reason = "source_open_text_clipped_at_page_edge"
            rec.candidate = False
            rec.review_required = bool(getattr(cfg, "photo_pair_edge_clipped_review_required", True))
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            rec.content_check = "blocked_incomplete_source"
            rec.content_complete = False
            rec.triage_state = "REVIEW"
            records.append(rec)
            continue

        # v2.3.32 fail-closed publication snapshot. Precise Mask and Hybrid may
        # both *attempt* a region, but an incomplete content audit must not leave
        # half-cleared Japanese / blurred Chinese in the published pixel layer.
        # Snapshot only for policies that can roll back; keep_review retains the
        # legacy behaviour without the copy cost.
        incomplete_policy = str(getattr(cfg, "incomplete_pixel_policy", "keep_review") or "keep_review").strip().lower()
        rollback_state = None
        if incomplete_policy in {"restore_target", "defer_to_ocr"}:
            rollback_state = (rendered.copy(), layer.copy(), composite_mask.copy(), clear_mask_all.copy())

        paste_mask = tm.copy()
        # Enclosed bubble masks include the complete interior. A tiny inset keeps
        # the clean HD target outline/tail untouched. Free SFX masks are already
        # local density masks, so they are not eroded.
        if (
            region_kind == "bubble"
            and cfg.preserve_target_border
            and cfg.paired_diff_target_border_inset_px > 0
        ):
            r = int(cfg.paired_diff_target_border_inset_px)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
            eroded = cv2.erode(paste_mask, k)
            if cv2.countNonZero(eroded) > 0:
                paste_mask = eroded

        if cv2.countNonZero(paste_mask) == 0:
            rec.reason = "empty_paste_mask"
            records.append(rec)
            continue

        # v0.8.33: classify the TARGET container before choosing the compositor.
        # A paired-diff candidate may be labelled ``bubble`` even when its mask
        # contains a coloured burst (or a piece of artwork). If that happens, the
        # ordinary whole-raster branch would paste the monochrome SOURCE paper
        # over the TARGET colour. The target-aware compositor must win first for
        # every region kind, not only for pre-labelled free_text/complex_text.
        saturated_gate = None
        saturated_route = False
        if region_kind == "bubble":
            candidate_gate = _dominant_saturated_container_mask(target, tm, cfg)
            if candidate_gate is not None:
                gate_area = max(1, cv2.countNonZero(candidate_gate))
                overlap = float(np.count_nonzero((candidate_gate > 0) & (tm > 0)) / gate_area)
                src_gray_for_sat = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
                sat_sel = candidate_gate > 0
                source_bright_ratio = float(np.mean(src_gray_for_sat[sat_sel] >= 220)) if np.any(sat_sel) else 0.0
                min_overlap = float(getattr(cfg, "paired_diff_saturated_min_region_overlap_ratio", 0.15))
                min_bright = float(getattr(cfg, "paired_diff_saturated_min_source_bright_ratio", 0.75))
                if overlap >= min_overlap and source_bright_ratio >= min_bright:
                    saturated_gate = candidate_gate
                    saturated_route = True

        # v0.8.21: open/complex text must never replace a rectangular coloured
        # background. Clear only Japanese glyph components and composite only the
        # registered Chinese raster ink. This path also covers burst balloons and
        # captions printed directly on artwork.
        if region_kind in {"free_text", "complex_text"} or saturated_route:
            # A legitimate cross-rendition burst has a bright paper-like source
            # container under the registered Chinese glyphs.  If the recovered
            # saturated colour belongs to nearby artwork/signage instead, the
            # aligned source at that location is usually not bright.  Reject that
            # colour route and fall back to bounded component transfer.
            if saturated_gate is None:
                saturated_gate = _dominant_saturated_container_mask(target, tm, cfg)
            if saturated_gate is not None:
                src_gray_for_sat = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
                sat_sel = saturated_gate > 0
                source_bright_ratio = float(np.mean(src_gray_for_sat[sat_sel] >= 220)) if np.any(sat_sel) else 0.0
                if source_bright_ratio < float(getattr(cfg, "paired_diff_saturated_min_source_bright_ratio", 0.75)):
                    saturated_gate = None
            saturated_route = saturated_gate is not None
            if saturated_gate is not None:
                sat_area = max(1, cv2.countNonZero(saturated_gate))
                overlap = int(np.count_nonzero((saturated_gate > 0) & (saturated_seen > 0))) / sat_area
                if overlap >= float(getattr(cfg, "paired_diff_saturated_duplicate_overlap", 0.82)):
                    # Structural splitting can emit two changed-text islands from
                    # the same burst balloon.  The first route already rebuilt the
                    # complete flat-colour container; do not process a subset a
                    # second time or turn it into a fake extra "applied" record.
                    matches.pop()
                    continue
                text_img, write_mask, source_ink_mask, text_diag = _transfer_saturated_text_container(
                    aligned_source, rendered, saturated_gate, cfg
                )
            else:
                text_img, write_mask, source_ink_mask, text_diag = _transfer_open_complex_text_region(
                    aligned_source, rendered, tm, cfg
                )
            rec.geometry_mode = region_kind
            rec.sr_backend = "paired-saturated-container" if saturated_route else "paired-text-components"
            rec.sr_scale = 1.0
            rec.mask_iou = 1.0
            rec.target_coverage = 1.0 if text_img is not None else 0.0
            rec.spill_ratio = 0.0
            ink_gate = saturated_gate if saturated_gate is not None else tm
            rec.ink_ratio = float(cv2.countNonZero(source_ink_mask) / max(1, cv2.countNonZero(ink_gate)))
            if text_img is None:
                rec.reason = str(text_diag.get("reason") or "open_text_component_transfer_failed")
                rec.review_required = True
                rec.review_reason = rec.reason
                rec.restorable = True
                rec.editable = True
                records.append(rec)
                continue
            # Content completeness is a review signal, not a reason to restore
            # Japanese.  If a usable Chinese raster candidate was produced,
            # publish it even when the audit finds missing strokes or residual
            # target ink.  The result remains explicitly reversible/editable.
            if saturated_route:
                rec.clarity_mode = "saturated-container-ink-transfer"
            else:
                rec.clarity_mode = "complex-text-ink-transfer" if region_kind == "complex_text" else "open-text-ink-transfer"
            _evaluate_content_completeness(
                rec, source_ink_mask, text_diag.get("target_ink_mask"), text_img, cfg
            )

            if not bool(getattr(rec, "content_complete", False)) and incomplete_policy in {"restore_target", "defer_to_ocr"}:
                rec.applied = False
                rec.reason = "content_incomplete_deferred_to_ocr" if incomplete_policy == "defer_to_ocr" else "content_incomplete_not_published"
                rec.candidate = True
                rec.review_required = True
                rec.review_reason = rec.reason
                rec.restorable = True
                rec.editable = True
                rec.meta["incomplete_pixel_policy"] = incomplete_policy
                records.append(rec)
                continue

            rendered = text_img
            alpha8 = write_mask.astype(np.uint8)
            composite_mask = np.maximum(composite_mask, alpha8)
            diag_clear = text_diag.get("clear_mask")
            if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
                clear_mask_all = np.maximum(clear_mask_all, diag_clear)
            if saturated_gate is not None:
                saturated_seen = np.maximum(saturated_seen, saturated_gate)
            use = alpha8 > 0
            layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
            layer[..., 3] = np.maximum(layer[..., 3], alpha8)
            rec.applied = True
            if not bool(getattr(rec, "content_complete", False)):
                rec.reason = "applied_incomplete_review_candidate"
                rec.candidate = True
                rec.review_required = True
                rec.review_reason = "content_incomplete_published_per_user_policy"
                rec.restorable = True
                rec.editable = True
                records.append(rec)
                continue
            low_conf = conf < float(getattr(cfg, "paired_diff_low_confidence_candidate_threshold", 0.64))
            boundary_touch = bool(text_diag.get("boundary_touch", False))
            if low_conf or boundary_touch:
                rec.reason = "applied_low_confidence_text_candidate"
                rec.candidate = True
                rec.review_required = True
                rec.review_reason = (
                    "source_text_cluster_touches_candidate_boundary" if boundary_touch
                    else "low_confidence_open_or_complex_text_region"
                )
                rec.restorable = True
                rec.editable = True
            else:
                rec.reason = "applied_registered_text_components"
            records.append(rec)
            continue

        warped_img = aligned_source.copy()
        target_clear_mask = (
            _expand_target_clear_mask_with_text_components(target, tm, cfg)
            if region_kind == "bubble" else tm
        )
        # The geometric write mask may be intentionally inset from the balloon
        # edge.  Japanese glyph components can bridge that inset and survive as
        # fragments.  Clear only those verified target text components first; do
        # not widen the source patch or copy surrounding artwork.
        preclear_extra = cv2.bitwise_and(target_clear_mask, cv2.bitwise_not(paste_mask))
        if region_kind == "bubble" and cv2.countNonZero(preclear_extra) > 0:
            fast_cleared, fast_clear_mask, fast_clear_diag = _fast_dark_pixel_clear(rendered, preclear_extra, cfg)
            if fast_cleared is not None:
                rendered = fast_cleared
                preclear_extra = fast_clear_mask
                rec.meta["fast_dark_pixel_clear"] = fast_clear_diag
            else:
                rendered = cv2.inpaint(rendered, preclear_extra, 2.5, cv2.INPAINT_TELEA)
                rec.meta["fast_dark_pixel_clear"] = fast_clear_diag
        output_mask = paste_mask.copy()
        if cfg.normalize_background:
            warped_img = _normalize_bubble_background(warped_img, tm, target, tm)

        white_clarity_diag = {"enabled": bool(getattr(cfg, "direct_white_clarity_enhance_enabled", True)), "applied": False, "reason": "not_evaluated"}
        if region_kind == "bubble":
            warped_img, white_clarity_diag = _maybe_apply_white_source_clarity(warped_img, target, tm, cfg)
        source_fidelity_lock = bool(white_clarity_diag.get("applied", False))
        if source_fidelity_lock:
            rec.clarity_mode = "source-faithful-white-clarity"
            rec.meta["source_raster_fidelity_lock"] = True

        rec.sr_backend = str(render_source_label or "paired-dense-align")
        rec.meta["render_source"] = dict(render_source_diagnostics or {})
        rec.meta["glyph_dense_warp"] = bool(rec.meta["render_source"].get("glyph_dense_warp", rec.sr_backend == "paired-dense-align"))
        rec.meta["white_source_clarity"] = white_clarity_diag
        rec.meta["dense_flow_geometry_only"] = bool(rec.meta["render_source"].get("dense_flow_geometry_only", False))
        rec.sr_scale = 1.0
        # Geometry QA is defined against the target-driven write mask. Preserve the
        # lower source-outline IoU separately in paired-diff metadata/diagnostics.
        rec.mask_iou = 1.0
        rec.target_coverage = 1.0
        rec.spill_ratio = 0.0
        rec.sharpness = _masked_sharpness(warped_img, paste_mask)
        rec.target_sharpness = _masked_sharpness(target, paste_mask)
        rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        fidelity = (cfg.text_fidelity_mode or "auto").lower().strip()
        # v0.8.34.4 Pixel Enhance: preserve the original source glyph raster and
        # layout, but strengthen soft antialiased edges before falling all the way
        # to binary ink reconstruction/OCR. This is most useful for clean low-res
        # scans mapped onto a higher-resolution target.
        if (not source_fidelity_lock
                and fidelity in {"auto", "pixels"}
                and rec.clarity_mode in {"pixels", "paired-aligned-pixels"}
                and bool(getattr(cfg, "pixel_enhance_enabled", True))
                and (rec.sharpness < float(getattr(cfg, "pixel_enhance_sharpness_trigger", 58.0))
                     or rec.relative_sharpness < float(getattr(cfg, "pixel_enhance_relative_trigger", 0.58)))):
            enhanced, enhance_diag = _pixel_enhance_text_raster(warped_img, paste_mask, cfg)
            rec.meta["pixel_enhance"] = enhance_diag
            if enhanced is not None:
                warped_img = enhanced
                rec.clarity_mode = "pixel-enhance"
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        too_soft = (
            rec.sharpness < cfg.min_pixel_text_sharpness
            or rec.relative_sharpness < cfg.min_relative_text_sharpness
        )
        # v0.8.5 structural supplements still originate from a photographed page.
        # Rebuild their local Chinese ink over the clean target instead of copying
        # camera pixels. Unlike a whole-bubble clear, the supplemental target mask
        # is compact, so this is also safe for open burst bubbles/free text.
        if photo_source and cfg.photo_pair_crisp_text_enabled and not source_fidelity_lock:
            reconstructed = None
            ink_ratio = 0.0
            # v0.8.16: if source glyph pixels cross the target mask boundary, rescue
            # the complete *raster* glyph footprint first. This is deliberately
            # mask-only: it never reads/reflows OCR text and only moves the whole
            # source text block by the minimum amount required for containment.
            if (region_kind == "bubble"
                    and source_sat_p90 <= float(getattr(cfg, "photo_pair_recenter_max_source_saturation_p90", 24.0))):
                reconstructed, ink_ratio, rescue_meta = _reconstruct_photo_glyph_footprint_layer(
                    aligned_source, target, tm, cfg, clear_mask=target_clear_mask
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-glyph-footprint-rescue"
                    rec.local_dx = float(rescue_meta.get("dx", 0.0))
                    rec.local_dy = float(rescue_meta.get("dy", 0.0))
                    rec.sr_scale = float(rescue_meta.get("scale", 1.0))
            # v0.8.7 small-balloon fallback remains for cases where the source text
            # is globally displaced but does not actually straddle the target mask.
            if (reconstructed is None
                    and region_kind == "bubble"
                    and getattr(cfg, "photo_pair_recenter_small_text_enabled", True)
                    and source_sat_p90 <= float(getattr(cfg, "photo_pair_recenter_max_source_saturation_p90", 24.0))):
                reconstructed, ink_ratio = _reconstruct_photo_recentered_ink_layer(
                    aligned_source, target, tm, cfg, clear_mask=target_clear_mask
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-recentered-ink"
            if reconstructed is None:
                reconstructed, ink_ratio = _reconstruct_photo_crisp_layer(
                    warped_img, target, paste_mask, target_clear_mask, cfg
                )
                if reconstructed is not None:
                    rec.clarity_mode = "photo-crisp-ink"
            rec.ink_ratio = ink_ratio
            if reconstructed is not None:
                warped_img = reconstructed
                # The reconstructed image clears the complete clean target bubble
                # before painting source Chinese ink. Blend that complete interior
                # back, otherwise Japanese glyphs near the balloon edge survive the
                # smaller border-inset sampling mask and mix with Chinese.
                if region_kind == "bubble":
                    output_mask = target_clear_mask.copy()
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
            else:
                rec.clarity_mode = "paired-aligned-pixels"
        # Paired transfer already knows where the translated pixels live. For
        # non-photo structural regions, use deterministic ink reconstruction only
        # when the aligned pixels are too soft.
        elif region_kind == "bubble" and not source_fidelity_lock and (fidelity == "ink" or (fidelity == "auto" and too_soft)):
            reconstructed = None
            ink_ratio = 0.0
            if cfg.ink_reconstruction_enabled:
                reconstructed, ink_ratio = _reconstruct_ink_layer(warped_img, target, paste_mask, cfg)
            rec.ink_ratio = ink_ratio
            if reconstructed is not None:
                warped_img = reconstructed
                rec.clarity_mode = "ink-reconstruction"
                rec.sharpness = _masked_sharpness(warped_img, paste_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
            else:
                rec.clarity_mode = "paired-aligned-pixels"
        else:
            if not source_fidelity_lock:
                rec.clarity_mode = "paired-aligned-pixels"

        gap_fill_diag = {"enabled": False, "iterations": 0, "added_pixels": 0}
        if region_kind == "bubble":
            output_mask, gap_fill_diag = _expand_safe_write_mask(output_mask, target_clear_mask, warped_img, target, cfg)
            rec.meta["mask_write_gap_fill"] = gap_fill_diag
        alpha = _alpha_from_mask(output_mask, cfg.feather_px)
        a3 = alpha[..., None]
        rendered = np.clip(
            warped_img.astype(np.float32) * a3 + rendered.astype(np.float32) * (1.0 - a3),
            0, 255,
        ).astype(np.uint8)
        alpha8 = (alpha * 255).astype(np.uint8)
        write_mask = np.maximum(alpha8, preclear_extra)
        composite_mask = np.maximum(composite_mask, write_mask)
        clear_mask_all = np.maximum(clear_mask_all, target_clear_mask)
        use = write_mask > 0
        layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        rec.applied = True
        if photo_source and rec.source_edge_clipped:
            # Review-first policy: target-driven transfer may still recover useful
            # Chinese from an edge-clipped source. Publish it as an explicitly
            # reversible/editable candidate instead of reverting to Japanese.
            rec.reason = "applied_low_confidence_candidate"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = "source_text_region_clipped_at_page_edge"
            rec.restorable = True
            rec.editable = True
        else:
            rec.reason = "applied_paired_target_driven"
        # White speech balloons get the same second-stage content audit.  Use a
        # wider tolerance for photographed/global alignment because a valid glyph
        # rescue may intentionally translate the whole source text block a few px.
        src_audit, _ = _compact_container_ink(aligned_source, tm, 190, cfg)
        tgt_audit, _ = _compact_container_ink(target, tm, 190, cfg)
        audit_tol = max(int(getattr(cfg, "content_completeness_tolerance_px", 2)), 7 if photo_source else 3)
        _evaluate_content_completeness(
            rec, src_audit, tgt_audit, rendered, cfg,
            tolerance_px=audit_tol,
        )
        if region_kind == "bubble" and not rec.content_complete:
            repaired, repaired_mask, repair_diag = _repair_content_region(
                rec, rendered, warped_img, target, write_mask, target_clear_mask,
                src_audit, tgt_audit, cfg, tolerance_px=audit_tol,
            )
            rec.meta["content_auto_repair"] = repair_diag
            if bool(repair_diag.get("improved", False)):
                rendered = repaired
                write_mask = np.maximum(write_mask, repaired_mask)
                composite_mask = np.maximum(composite_mask, write_mask)
                clear_mask_all = np.maximum(clear_mask_all, repaired_mask)
                use = write_mask > 0
                if np.any(use):
                    layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
                layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        if region_kind == "bubble" and not bool(getattr(rec, "content_complete", False)) and incomplete_policy in {"restore_target", "defer_to_ocr"}:
            if rollback_state is not None:
                rendered, layer, composite_mask, clear_mask_all = rollback_state
            rec.applied = False
            rec.reason = "content_incomplete_deferred_to_ocr" if incomplete_policy == "defer_to_ocr" else "content_incomplete_not_published"
            rec.candidate = True
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            rec.meta["incomplete_pixel_policy"] = incomplete_policy
        records.append(rec)

    return MaskTransferResult(rendered, layer, composite_mask, matches, records, clear_mask_all)

def transfer_photo_color_sfx(
    aligned_source: np.ndarray,
    target: np.ndarray,
    cfg: MaskReplaceConfig | None = None,
    *,
    koharu_text_sfx_authority_mask: np.ndarray | None = None,
) -> MaskTransferResult:
    """Transfer large vivid-red translated SFX from a photographed page.

    This intentionally targets only high-saturation red glyph groups that exist
    in both editions but have substantially different shapes. Stable red artwork
    (clothes, hats, scenery) overlaps after registration and therefore fails the
    change gate. The clean target SFX is locally inpainted, then source red fill
    and its immediately adjacent dark outline are rebuilt with crisp masks.
    """
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    if aligned_source.shape[:2] != shape:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])

    def vivid_red(image: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(image.astype(np.float32))
        return ((r > 150.0) & (r > g * 1.55) & (r > b * 1.45)).astype(np.uint8) * 255

    sm = vivid_red(aligned_source); tm = vivid_red(target)
    if cv2.countNonZero(sm) < 300 or cv2.countNonZero(tm) < 300:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])
    group_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    union = cv2.dilate(cv2.bitwise_or(sm, tm), group_k)
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21, 9)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((union > 0).astype(np.uint8), 8)
    td = cv2.dilate(tm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    sd = cv2.dilate(sm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
    page_area = shape[0] * shape[1]
    candidates = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        if area < 500 or area / max(1, page_area) > 0.035:
            continue
        region = labels == i
        sr = int(np.count_nonzero((sm > 0) & region)); tr = int(np.count_nonzero((tm > 0) & region))
        if sr < 500 or tr < 500:
            continue
        matched = int(np.count_nonzero((sm > 0) & td & region) + np.count_nonzero((tm > 0) & sd & region))
        change = float(np.clip(1.0 - matched / max(1, sr + tr), 0.0, 1.0))
        if change < 0.22:
            continue
        candidates.append((i, (x, y, x + bw, y + bh), change, sr, tr))

    if not candidates:
        return MaskTransferResult(target.copy(), np.zeros((shape[0], shape[1], 4), np.uint8), np.zeros(shape, np.uint8), [], [])

    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite = np.zeros(shape, np.uint8)
    clear_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    matches: list[BubblePatchMatch] = []
    src_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    for idx, (lab, box, change, sr, tr) in enumerate(candidates):
        region = labels == lab
        src_red = ((sm > 0) & region).astype(np.uint8) * 255
        tgt_red = ((tm > 0) & region).astype(np.uint8) * 255
        # v2.0.90: colour-SFX recovery is subordinate to the global Koharu
        # semantic authority.  When Layout is available, red artwork may not
        # become writable merely because its shape differs across editions.
        if isinstance(koharu_text_sfx_authority_mask, np.ndarray):
            auth = koharu_text_sfx_authority_mask > 0
            if auth.shape == tgt_red.shape:
                denom = max(1, cv2.countNonZero(tgt_red))
                support = float(np.count_nonzero((tgt_red > 0) & auth) / denom)
                if support < 0.12:
                    continue
        # Include immediately adjacent dark outline/shadow, but not distant panel art.
        near_src = cv2.dilate(src_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        near_tgt = cv2.dilate(tgt_red, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        src_outline = ((src_gray < 70) & near_src & region).astype(np.uint8) * 255
        tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        tgt_outline = ((tgt_gray < 70) & near_tgt & region).astype(np.uint8) * 255
        clear = cv2.bitwise_or(tgt_red, tgt_outline)
        clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        if cv2.countNonZero(clear) == 0:
            continue
        # Inpaint only the old SFX glyph pixels, preserving the rest of the panel.
        rendered = cv2.inpaint(rendered, clear, 3.0, cv2.INPAINT_TELEA)
        clear_all = np.maximum(clear_all, clear)
        red_pixels = target[tgt_red > 0]
        if len(red_pixels):
            red_color = np.median(red_pixels, axis=0).astype(np.uint8)
        else:
            red_color = np.array([35, 35, 205], np.uint8)
        rendered[src_red > 0] = red_color
        rendered[src_outline > 0] = (0, 0, 0)
        write_mask = cv2.bitwise_or(clear, cv2.bitwise_or(src_red, src_outline))
        composite = np.maximum(composite, write_mask)
        use = write_mask > 0
        layer[use, :3] = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], write_mask)
        sid, tid = f"color-sfx-src-{idx:03d}", f"color-sfx-dst-{idx:03d}"
        matches.append(BubblePatchMatch(sid, tid, 0.90, 0.10, 1.0, 0.0, 1.0, [f"red_shape_change={change:.3f}"]))
        rec = MaskTransferRecord(sid, tid, 0.90, True, "applied_color_sfx_rebuild")
        rec.geometry_mode = "photo_pair"
        rec.clarity_mode = "color-sfx-rebuild"
        rec.mask_iou = 1.0; rec.target_coverage = 1.0; rec.spill_ratio = 0.0
        rec.source_bbox = box; rec.target_bbox = box
        rec.ink_ratio = float(sr / max(1, (box[2]-box[0]) * (box[3]-box[1])))
        records.append(rec)
    return MaskTransferResult(rendered, layer, composite, matches, records, clear_all)


def transfer_bubble_patches(
    source: np.ndarray,
    target: np.ndarray,
    source_bubbles: list[BubbleInstance],
    target_bubbles: list[BubbleInstance],
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
) -> MaskTransferResult:
    cfg = cfg or MaskReplaceConfig()
    shape = target.shape[:2]
    matches = match_bubbles(source_bubbles, target_bubbles, registration, shape, cfg)
    source_by_id = {b.id: b for b in source_bubbles}
    target_by_id = {b.id: b for b in target_bubbles}
    rendered = target.copy()
    layer = np.zeros((shape[0], shape[1], 4), np.uint8)
    composite_mask = np.zeros(shape, np.uint8)
    clear_mask_all = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []

    for match in matches:
        sb, tb = source_by_id[match.source_bubble_id], target_by_id[match.target_bubble_id]
        sm = _bubble_mask(sb, source.shape[:2])
        tm = _bubble_mask(tb, shape)
        sbox = _bbox_from_mask(sm); tbox = _bbox_from_mask(tm)
        rec = MaskTransferRecord(sb.id, tb.id, match.confidence, False, "not_applied")
        is_photo_pair = tb.meta.get("paired_diff_method") == "photo_pair"
        rec.geometry_mode = "photo_pair" if is_photo_pair else str(tb.meta.get("paired_diff_method") or "standard")
        if sbox: rec.source_bbox = sbox
        if tbox: rec.target_bbox = tbox
        source_edge_sides = _edge_touch_sides(
            sbox, source.shape[:2], getattr(cfg, "photo_pair_edge_clip_margin_px", 0)
        ) if is_photo_pair else ()
        rec.source_edge_clipped = bool(source_edge_sides)
        rec.source_edge_sides = ",".join(source_edge_sides)
        if _publication_safety_enabled(cfg) and match.confidence < cfg.min_match_confidence:
            rec.reason = "bubble_match_low_confidence"; records.append(rec); continue
        if not sbox or not tbox:
            rec.reason = "empty_bubble_mask"; records.append(rec); continue
        sw, sh = max(1, sbox[2]-sbox[0]), max(1, sbox[3]-sbox[1])
        tw, th = max(1, tbox[2]-tbox[0]), max(1, tbox[3]-tbox[1])
        small_text_photo_pair = bool(is_photo_pair and min(tw, th) < cfg.photo_pair_min_direct_side_px)
        # Compare target size against the *registered* source geometry, not raw
        # source pixels. A 2400px phone photo mapped onto an 850px clean master
        # is expected to have a much larger raw bbox; rejecting that scale change
        # made photographed editions impossible to transfer.
        H_base = transform_to_homography(registration.matrix)
        mapped_box = transform_points(
            [(sbox[0], sbox[1]), (sbox[2], sbox[1]), (sbox[2], sbox[3]), (sbox[0], sbox[3])], H_base
        )
        mx0, my0, mx1, my1 = polygon_bbox(mapped_box)
        mapped_w, mapped_h = max(1e-6, mx1 - mx0), max(1e-6, my1 - my0)
        if _publication_safety_enabled(cfg) and min(tw / mapped_w, th / mapped_h) < cfg.reject_if_target_smaller_ratio:
            rec.reason = "target_bubble_much_smaller_after_registration"; records.append(rec); continue

        # Same-source editions often differ only by translated lettering. When
        # page registration is effectively identity and paired-diff masks agree,
        # copy the translated interior pixel-for-pixel. This avoids needless
        # resampling/feathering and gives a measurable exactness fast path while
        # keeping the Japanese HD page untouched outside the destination mask.
        if cfg.exact_identity_copy and _identity_like(registration, cfg):
            warped_sm = _warp_mask(sm, registration.matrix, shape)
            exact_iou = _mask_iou(warped_sm, tm)
            coverage, spill = _target_coverage(warped_sm, tm)
            if exact_iou >= cfg.exact_identity_mask_iou:
                dest_mask = tm.copy()
                # Paired-diff / white-component masks already represent the interior
                # bounded by the black balloon outline. Do not erode them again.
                interior_mask = bool(tb.meta.get("mask_is_interior"))
                if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
                    ksize = cfg.border_inset_px * 2 + 1
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                    dest_mask = cv2.erode(dest_mask, k, iterations=1)
                paste_mask = dest_mask
                if cfg.exact_identity_changed_fringe_px > 0:
                    # Translation editors often touch a few antialiased pixels just
                    # outside the white interior (ruby cleanup / box-edge AA). In the
                    # strict same-source fast path we can detect those pixels directly
                    # from the paired pages and include only the actually changed fringe.
                    r = int(cfg.exact_identity_changed_fringe_px)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
                    near = cv2.dilate((dest_mask > 0).astype(np.uint8) * 255, k) > 0
                    pair_diff = np.mean(np.abs(source.astype(np.int16) - target.astype(np.int16)), axis=2)
                    changed = pair_diff >= float(cfg.paired_diff_pixel_threshold)
                    fringe = near & (dest_mask == 0) & changed
                    if np.any(fringe):
                        paste_mask = np.maximum(paste_mask, fringe.astype(np.uint8) * 255)
                if cv2.countNonZero(paste_mask):
                    use = paste_mask > 0
                    rendered[use] = source[use]
                    composite_mask = np.maximum(composite_mask, paste_mask)
                    layer[use, :3] = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)[use]
                    layer[..., 3] = np.maximum(layer[..., 3], paste_mask)
                    rec.sr_backend, rec.sr_scale = "pixel-exact", 1.0
                    rec.mask_iou, rec.target_coverage, rec.spill_ratio = exact_iou, coverage, spill
                    rec.sharpness = _masked_sharpness(source, paste_mask)
                    rec.applied = True
                    rec.reason = "applied_exact_identity"
                    src_audit, _ = _compact_container_ink(source, dest_mask, 190, cfg)
                    tgt_audit, _ = _compact_container_ink(target, dest_mask, 190, cfg)
                    _evaluate_content_completeness(rec, src_audit, tgt_audit, rendered, cfg, tolerance_px=3)
                    records.append(rec)
                    continue

        H = H_base.copy()
        if cfg.local_fit in {"bbox", "ecc"}:
            mapped = mapped_box
            mx0,my0,mx1,my1 = polygon_bbox(mapped)
            corr_x = tw / max(1e-6, mx1-mx0); corr_y = th / max(1e-6, my1-my0)
            max_scale_change = cfg.photo_pair_max_local_scale_change if is_photo_pair else cfg.max_local_scale_change
            correction_ok = abs(corr_x-1.0) <= max_scale_change and abs(corr_y-1.0) <= max_scale_change
            if correction_ok:
                if is_photo_pair and getattr(cfg, "photo_pair_uniform_local_fit", True):
                    # Never independently resize X/Y for photographed CJK text.
                    # That produced the visibly squeezed/twisted glyphs reported on
                    # real 007/009 pages. Use one local scale and center alignment.
                    axis_delta = abs(float(corr_x - corr_y))
                    if axis_delta <= float(getattr(cfg, "photo_pair_max_axis_scale_delta", 0.10)):
                        fit_H = _bbox_uniform_fit_matrix(
                            (sbox[0], sbox[1], sbox[2], sbox[3]),
                            (tbox[0], tbox[1], tbox[2], tbox[3]), H,
                        )
                    else:
                        fit_H = H.copy()
                else:
                    fit_H = _bbox_fit_matrix((sbox[0], sbox[1], sbox[2], sbox[3]), (tbox[0], tbox[1], tbox[2], tbox[3]), H)
                # Never accept a local correction merely because it exists. Curved
                # pages may already be better under the global registration.
                global_preview = _warp_mask(sm, H, shape)
                fit_preview = _warp_mask(sm, fit_H, shape)
                if _mask_iou(fit_preview, tm) > _mask_iou(global_preview, tm) + 0.005:
                    H = fit_H
            elif not is_photo_pair and _publication_safety_enabled(cfg):
                rec.reason = "local_scale_correction_too_large"; records.append(rec); continue
            # Photo pairs safely retain the global transform when local scaling is
            # implausible; the target mask remains the publication boundary.

        transfer_sm = sm
        if cfg.source_mask_expand_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
            transfer_sm = cv2.dilate(transfer_sm, k, iterations=cfg.source_mask_expand_px)
        warped_img, warped_mask, sr_backend, sr_scale = _warp_source_patch(source, transfer_sm, H, shape, tbox, cfg)
        rec.sr_backend, rec.sr_scale = sr_backend, sr_scale
        if cfg.local_fit == "ecc":
            dx, dy, _ = _local_translation_ecc(warped_mask, tm, cfg)
            dx, dy, _subpix_score, subpix_diag = _subpixel_translation_refine(warped_mask, tm, dx, dy, cfg)
            if dx or dy:
                warped_img = _shift(warped_img, dx, dy, nearest=False)
                warped_mask = _shift(warped_mask, dx, dy, nearest=True)
            rec.local_dx, rec.local_dy = dx, dy
            rec.meta["local_subpixel_refine"] = subpix_diag

        rec.mask_iou = _mask_iou(warped_mask, tm)
        rec.target_coverage, rec.spill_ratio = _target_coverage(warped_mask, tm)
        min_iou = cfg.photo_pair_min_transfer_iou if is_photo_pair else cfg.min_mask_iou
        min_coverage = cfg.photo_pair_min_transfer_coverage if is_photo_pair else cfg.min_target_coverage
        max_spill = cfg.photo_pair_max_spill_ratio if is_photo_pair else cfg.max_spill_ratio
        if (
            is_photo_pair
            and rec.target_coverage < min_coverage
            and rec.target_coverage >= max(0.0, min_coverage - cfg.photo_pair_salvage_coverage_margin)
            and rec.spill_ratio <= (max_spill + cfg.photo_pair_salvage_spill_extra)
        ):
            salvaged_img, salvaged_mask, salvaged_backend, salvaged_scale = _photo_pair_salvage_warp(
                source, transfer_sm, warped_img, warped_mask, H, shape, tbox, tm, cfg,
            )
            salvage_iou = _mask_iou(salvaged_mask, tm)
            salvage_cov, salvage_spill = _target_coverage(salvaged_mask, tm)
            if (
                salvage_cov > rec.target_coverage + 1e-6
                or (abs(salvage_cov - rec.target_coverage) <= 1e-6 and salvage_iou > rec.mask_iou + 1e-6)
            ):
                warped_img, warped_mask = salvaged_img, salvaged_mask
                rec.sr_backend = salvaged_backend if salvaged_backend != "off" else rec.sr_backend
                rec.sr_scale = salvaged_scale if salvaged_scale != 1.0 else rec.sr_scale
                rec.mask_iou = salvage_iou
                rec.target_coverage, rec.spill_ratio = salvage_cov, salvage_spill

        # A camera-edge-clipped source is fundamentally different from ordinary
        # under-segmentation. Dilating/locally fitting the mask can improve the
        # geometry, but it cannot recover Chinese glyph pixels that were never
        # captured. v0.8.2 incorrectly accepted the real 009 top-right bubble at
        # ~86% coverage and then cleared the whole Japanese target, publishing a
        # visibly truncated translation. Require near-complete coverage whenever
        # the source bubble itself reaches the photo boundary. Bubbles that only
        # lose a little outline but retain their text still pass this stricter gate.
        if (
            is_photo_pair
            and getattr(cfg, "photo_pair_edge_clip_guard_enabled", True)
            and source_edge_sides
            and rec.target_coverage < max(
                min_coverage, float(getattr(cfg, "photo_pair_edge_clip_min_target_coverage", 0.94))
            )
        ):
            # v0.8.6 review-first policy: do not silently leave Japanese. When a
            # meaningful portion of the translated source still exists, publish a
            # recoverable Chinese *candidate* in the automatic preview, clearly
            # flag it for review, and preserve one-click restore/manual reletter.
            candidate_enabled = bool(getattr(cfg, "photo_pair_low_confidence_candidate_enabled", True))
            candidate_min_cov = float(getattr(cfg, "photo_pair_candidate_min_coverage", 0.55))
            if candidate_enabled and rec.target_coverage >= candidate_min_cov:
                dest_mask = tm.copy()
                interior_mask = bool(tb.meta.get("mask_is_interior"))
                if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
                    ksize = cfg.border_inset_px * 2 + 1
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
                    eroded = cv2.erode(dest_mask, k, iterations=1)
                    if cv2.countNonZero(eroded) > 0:
                        dest_mask = eroded
                partial_mask = cv2.bitwise_and(dest_mask, warped_mask)
                candidate_img = None
                candidate_ratio = 0.0
                if cv2.countNonZero(partial_mask) > 0:
                    if cfg.normalize_background:
                        warped_img = _normalize_bubble_background(warped_img, warped_mask, target, tm)
                    if cfg.photo_pair_crisp_text_enabled:
                        candidate_img, candidate_ratio = _reconstruct_photo_crisp_layer(
                            warped_img, target, partial_mask, dest_mask, cfg,
                            allow_nonwhite_target=True,
                        )
                    if candidate_img is None and cfg.photo_pair_normalize_text_pixels:
                        candidate_img = _normalize_photo_text_pixels(
                            warped_img, target, partial_mask, dest_mask, cfg,
                        )
                if candidate_img is not None:
                    use = dest_mask > 0
                    rendered[use] = candidate_img[use]
                    composite_mask = np.maximum(composite_mask, dest_mask)
                    rgb_candidate = cv2.cvtColor(candidate_img, cv2.COLOR_BGR2RGB)
                    layer[use, :3] = rgb_candidate[use]
                    layer[..., 3] = np.maximum(layer[..., 3], dest_mask)
                    rec.ink_ratio = float(candidate_ratio)
                    rec.clarity_mode = "photo-crisp-ink-candidate" if candidate_ratio > 0 else "photo-normalized-candidate"
                    rec.sharpness = _masked_sharpness(candidate_img, dest_mask)
                    rec.target_sharpness = _masked_sharpness(target, dest_mask)
                    rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                    rec.applied = True
                    rec.reason = "applied_low_confidence_candidate"
                    rec.candidate = True
                    rec.review_required = True
                    rec.review_reason = "source_text_region_clipped_at_page_edge"
                    rec.restorable = True
                    rec.editable = True
                    src_audit, _ = _compact_container_ink(warped_img, dest_mask, 190, cfg)
                    tgt_audit, _ = _compact_container_ink(target, dest_mask, 190, cfg)
                    _evaluate_content_completeness(rec, src_audit, tgt_audit, rendered, cfg, tolerance_px=5)
                    records.append(rec)
                    continue
            rec.reason = "source_text_region_clipped_at_page_edge"
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue
        if _publication_safety_enabled(cfg):
            if rec.mask_iou < min_iou:
                rec.reason = "mask_iou_below_threshold"; records.append(rec); continue
            if rec.target_coverage < min_coverage:
                rec.reason = "target_coverage_below_threshold"; records.append(rec); continue
            if rec.spill_ratio > max_spill:
                rec.reason = "source_mask_spills_outside_target"; records.append(rec); continue

        dest_mask = tm.copy()
        if cfg.target_mask_expand_px > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dest_mask = cv2.dilate(dest_mask, k, iterations=cfg.target_mask_expand_px)
        interior_mask = bool(tb.meta.get("mask_is_interior"))
        if cfg.preserve_target_border and cfg.border_inset_px > 0 and not interior_mask:
            ksize = cfg.border_inset_px * 2 + 1
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dest_mask = cv2.erode(dest_mask, k, iterations=1)

        # Only composite where source bubble pixels are valid. Gate above ensures this
        # still covers essentially the whole target interior, so Japanese text cannot
        # silently remain in uncovered holes.
        paste_mask = cv2.bitwise_and(dest_mask, warped_mask)
        if cv2.countNonZero(paste_mask) == 0:
            rec.reason = "empty_paste_mask"; records.append(rec); continue

        if cfg.normalize_background:
            warped_img = _normalize_bubble_background(warped_img, warped_mask, target, tm)
        rec.sharpness = _masked_sharpness(warped_img, paste_mask)
        rec.target_sharpness = _masked_sharpness(target, paste_mask)
        rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        fidelity = (cfg.text_fidelity_mode or "auto").lower().strip()
        if is_photo_pair and cfg.photo_pair_force_ink_reconstruction and fidelity == "auto":
            fidelity = "ink"
        if is_photo_pair and small_text_photo_pair and fidelity == "auto":
            fidelity = "ink"

        # Preferred v0.8.3 path: extract only registered Chinese ink and rebuild it
        # on clean target paper. This removes camera blur/glare and duplicated
        # source balloon outlines without requiring OCR or a font.
        if is_photo_pair and cfg.photo_pair_crisp_text_enabled and fidelity != "reject":
            crisp, crisp_ratio = _reconstruct_photo_crisp_layer(
                warped_img, target, paste_mask, dest_mask, cfg,
            )
            rec.ink_ratio = crisp_ratio
            if crisp is not None:
                warped_img = crisp
                paste_mask = dest_mask
                rec.clarity_mode = "photo-crisp-ink"
                rec.sharpness = _masked_sharpness(warped_img, dest_mask)
                rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                fidelity = "pixels"  # already rebuilt; skip binary reconstruction

        # Photograph-specific fidelity ladder. First remove smooth glare / page
        # shading while retaining real antialiased glyph pixels. Only if the
        # normalized result remains too soft do we use deterministic ink recovery;
        # very soft/tiny text is refused and left for OCR relettering.
        if (is_photo_pair and rec.clarity_mode != "photo-crisp-ink"
                and fidelity in {"auto", "pixels"} and cfg.photo_pair_normalize_text_pixels):
            normalized = _normalize_photo_text_pixels(warped_img, target, paste_mask, dest_mask, cfg)
            if normalized is not None:
                norm_sharp = _masked_sharpness(normalized, dest_mask)
                norm_rel = norm_sharp / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                # Explicit pixels means "preserve glyph pixels", but still allows
                # deterministic illumination normalization because it does not
                # redraw/infer characters.
                if fidelity == "pixels" or norm_rel >= cfg.photo_pair_normalize_min_relative_sharpness:
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.clarity_mode = "photo-normalized-pixels"
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "pixels"
                elif norm_rel < cfg.photo_pair_reletter_below_relative_sharpness:
                    # Do not hard-reject yet. Very soft photographed text can
                    # still be recovered deterministically as crisp black ink on
                    # clean target paper, especially for small bubbles.
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "ink"
                else:
                    # Keep the normalized pixels as the input to ink recovery. It
                    # has no camera background gradient, making thresholding safer.
                    warped_img = normalized
                    paste_mask = dest_mask
                    rec.sharpness = norm_sharp
                    rec.relative_sharpness = norm_rel
                    fidelity = "ink"

        too_soft = (
            rec.sharpness < cfg.min_pixel_text_sharpness
            or rec.relative_sharpness < cfg.min_relative_text_sharpness
        )
        if fidelity == "reject":
            rec.clarity_mode = "rejected"
            rec.reason = "source_text_fidelity_rejected"
            records.append(rec)
            continue

        photo_prefers_ink = bool(
            is_photo_pair and rec.clarity_mode != "photo-crisp-ink" and (
                small_text_photo_pair
                or fidelity == "ink"
                or rec.relative_sharpness < cfg.photo_pair_prefer_ink_below_relative_sharpness
            )
        )
        should_try_ink = bool(
            fidelity == "ink"
            or (fidelity == "auto" and too_soft)
            or photo_prefers_ink
        )
        if should_try_ink:
            reconstructed = None
            ink_ratio = 0.0
            reconstruction_mask = paste_mask
            if cfg.ink_reconstruction_enabled:
                reconstructed, ink_ratio = _reconstruct_ink_layer(
                    warped_img, target, reconstruction_mask, cfg,
                    clear_mask=dest_mask if is_photo_pair else None,
                )
            rec.ink_ratio = ink_ratio
            used_reconstruction = False
            if reconstructed is not None:
                candidate_mask = dest_mask if is_photo_pair else paste_mask
                recon_sharpness = _masked_sharpness(reconstructed, candidate_mask)
                recon_relative = recon_sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0
                if (
                    fidelity == "ink"
                    or small_text_photo_pair
                    or rec.clarity_mode == "reletter-required"
                    or recon_sharpness >= rec.sharpness * cfg.photo_pair_prefer_ink_min_gain
                    or recon_relative >= rec.relative_sharpness + 0.06
                ):
                    warped_img = reconstructed
                    if is_photo_pair:
                        # Japanese glyphs were cleared across the clean target mask,
                        # but Chinese ink was extracted only from valid source pixels.
                        paste_mask = dest_mask
                    rec.clarity_mode = "ink-reconstruction"
                    rec.sharpness = recon_sharpness
                    rec.relative_sharpness = recon_relative
                    used_reconstruction = True
            if not used_reconstruction and (fidelity == "ink" or small_text_photo_pair or (fidelity == "auto" and too_soft)):
                if cfg.reject_blurry_source:
                    rec.clarity_mode = "reletter-required"
                    rec.reason = "source_text_too_blurry_for_pixel_transfer"
                    records.append(rec)
                    continue
        elif not rec.clarity_mode:
            rec.clarity_mode = "pixels"

        alpha = _alpha_from_mask(paste_mask, cfg.feather_px)
        a3 = alpha[..., None]
        rendered = np.clip(warped_img.astype(np.float32) * a3 + rendered.astype(np.float32) * (1.0 - a3), 0, 255).astype(np.uint8)
        composite_mask = np.maximum(composite_mask, (alpha * 255).astype(np.uint8))
        # RGBA layer is the exact transferred patch, useful for ORA/PSD review.
        use = alpha > 0
        layer[use, :3] = cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)[use]
        layer[..., 3] = np.maximum(layer[..., 3], (alpha * 255).astype(np.uint8))
        rec.applied = True
        rec.reason = "applied"
        clear_mask_all = np.maximum(clear_mask_all, dest_mask if is_photo_pair else paste_mask)
        audit_gate = dest_mask if is_photo_pair else paste_mask
        src_audit, _ = _compact_container_ink(warped_img, audit_gate, 190, cfg)
        tgt_audit, _ = _compact_container_ink(target, audit_gate, 190, cfg)
        _evaluate_content_completeness(
            rec, src_audit, tgt_audit, rendered, cfg,
            tolerance_px=max(3, 5 if is_photo_pair else 3),
        )
        records.append(rec)

    return MaskTransferResult(rendered, layer, composite_mask, matches, records, clear_mask_all)
