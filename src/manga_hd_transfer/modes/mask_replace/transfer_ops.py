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
from .text_transfer import source_text_render, changed_text_masks, target_text_mask_in_container, cleanup_target_residual_specks, clear_uniform_white_container_interior, white_container_paper_mask, white_container_write_envelope, target_container_border_mask, remove_container_boundary_line_components
from .source_clarity import enhance_white_source_patch as _enhance_white_source_patch
from ...config import MaskReplaceConfig
from ...geometry import polygon_bbox, rasterize_polygon, transform_points, transform_to_homography
from ...io_utils import read_image, write_image
from ...models import BubbleInstance, RegistrationResult, TextUnit, UnitMatch
from ...text_ink_equivalence import equivalent_registered_ink















































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


def _clear_empty_source_semantic_bubble(
    aligned_source: np.ndarray,
    target: np.ndarray,
    bubble_gate: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, dict]:
    """Clear TARGET lettering when a trusted semantic bubble is empty in SOURCE.

    Koharu can correctly recover a speech bubble that paired-diff missed while the
    translated SOURCE intentionally contains no text. The generic complex-text
    fallback must not turn bubble outlines or other dark structure into fake SOURCE
    ink in that case. Detect emptiness on an eroded bubble interior with the same
    container-aware text selector used by the rigid path, then clear only TARGET
    text components. No SOURCE pixels are written.
    """
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if not bool(getattr(cfg, "koharu_semantic_empty_source_clear_enabled", True)):
        return None, empty, empty, {"reason": "empty_source_clear_disabled"}
    if bubble_gate.shape != shape or cv2.countNonZero(bubble_gate) <= 0:
        return None, empty, empty, {"reason": "empty_source_clear_empty_gate"}

    erode_px = max(1, int(getattr(cfg, "koharu_semantic_empty_source_inner_erode_px", 4)))
    inner = cv2.erode(
        (bubble_gate > 0).astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px * 2 + 1, erode_px * 2 + 1)),
    )
    if cv2.countNonZero(inner) <= 0:
        return None, empty, empty, {"reason": "empty_source_clear_eroded_empty"}

    source_text = target_text_mask_in_container(aligned_source, inner)
    target_text = target_text_mask_in_container(target, inner)
    source_px = int(cv2.countNonZero(source_text))
    target_px = int(cv2.countNonZero(target_text))
    min_target = max(8, int(getattr(cfg, "koharu_semantic_empty_source_min_target_text_pixels", 24)))
    max_source = max(0, int(getattr(cfg, "koharu_semantic_empty_source_max_text_pixels", 12)))
    max_ratio = float(getattr(cfg, "koharu_semantic_empty_source_max_source_target_ratio", 0.08))
    if target_px < min_target:
        return None, empty, empty, {
            "reason": "empty_source_clear_insufficient_target_text",
            "source_text_pixels": source_px,
            "target_text_pixels": target_px,
        }
    if source_px > max_source and source_px / max(1, target_px) > max_ratio:
        return None, empty, empty, {
            "reason": "empty_source_clear_source_not_empty",
            "source_text_pixels": source_px,
            "target_text_pixels": target_px,
            "source_target_ratio": float(source_px / max(1, target_px)),
        }

    dilate_px = max(1, int(getattr(cfg, "koharu_semantic_empty_source_clear_dilate_px", 2)))
    clear = cv2.dilate(
        target_text,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1)),
    )
    clear = cv2.bitwise_and(clear, (bubble_gate > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(clear) <= 0:
        return None, empty, empty, {
            "reason": "empty_source_clear_no_clear_pixels",
            "source_text_pixels": source_px,
            "target_text_pixels": target_px,
        }
    out = cv2.inpaint(target, clear, 2.5, cv2.INPAINT_TELEA)
    final_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    residual = float(
        np.count_nonzero((target_text > 0) & (final_gray <= 205)) / max(1, target_px)
    )
    max_residual = float(getattr(cfg, "koharu_semantic_empty_source_max_residual_ratio", 0.05))
    return out, clear.copy(), empty, {
        "reason": "ok_empty_source_bubble_clear",
        "clear_backend": "koharu-semantic-empty-source-clear",
        "clear_mask": clear,
        "target_ink_mask": target_text,
        "boundary_touch": False,
        "source_text_pixels": source_px,
        "target_text_pixels": target_px,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "target_residual_ratio": residual,
        "max_target_residual_ratio": max_residual,
        "content_complete": bool(residual <= max_residual),
        "empty_source_clear": True,
        "inner_erode_px": erode_px,
        "clear_dilate_px": dilate_px,
    }


def _transfer_open_complex_text_region(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_ink_map: np.ndarray | None = None,
    target_ink_map: np.ndarray | None = None,
    target_clear_hint: np.ndarray | None = None,
    region_pad_px: int | None = None,
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
    if region_pad_px is not None:
        pad = max(0, int(region_pad_px))
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

    src_ink = source_ink_map if isinstance(source_ink_map, np.ndarray) and source_ink_map.shape == shape else _complex_text_ink_map(aligned_source)
    tgt_ink = target_ink_map if isinstance(target_ink_map, np.ndarray) and target_ink_map.shape == shape else _complex_text_ink_map(target)
    tol = max(1, int(getattr(cfg, "paired_diff_ink_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    src_near = cv2.dilate(src_ink, k)
    tgt_near = cv2.dilate(tgt_ink, k)
    src_unique = cv2.bitwise_and(src_ink, cv2.bitwise_not(tgt_near))
    tgt_unique = cv2.bitwise_and(tgt_ink, cv2.bitwise_not(src_near))
    gap = int(getattr(cfg, "paired_diff_complex_group_gap_px", 5))
    src_sel = _select_changed_text_components(src_ink, src_unique, gate, gap)
    tgt_sel = _select_changed_text_components(tgt_ink, tgt_unique, gate, gap)
    # Clear all compact TARGET strokes in trusted detector text scopes,
    # including punctuation/overlapping strokes with no edition-exclusive seed.
    if target_clear_hint is not None and target_clear_hint.shape == shape:
        text_scope = cv2.bitwise_and((target_clear_hint > 0).astype(np.uint8) * 255, gate)
        full_target_text = _select_changed_text_components(tgt_ink, tgt_ink, text_scope, gap)
        tgt_sel = np.maximum(tgt_sel, full_target_text)
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
    if target_clear_hint is not None and target_clear_hint.shape == shape:
        from ...koharu_mask_codec import refine_text_mask
        clear = np.maximum(clear, refine_text_mask(target_clear_hint, gate))
    d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
    clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
    clear = cv2.bitwise_and(clear, gate)
    out = cv2.inpaint(target, clear, 2.5, cv2.INPAINT_TELEA)

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    background = (region_mask > 0) & (hsv[..., 2] >= 160)
    src_rgb = aligned_source[region_mask > 0]
    src_bright = src_rgb[src_rgb.max(axis=1) >= 140]
    color_ink = bool(
        np.count_nonzero(background) >= 64 and src_bright.shape[0] >= 64
        and np.median(hsv[..., 1][background]) >= 58
        and np.percentile(src_bright.max(axis=1).astype(float) - src_bright.min(axis=1), 90) <= 30
    )
    if color_ink:
        # Extract black-ink opacity from the mono scan. Its AA gray already
        # includes white paper; using it as foreground RGB creates a white matte.
        support = cv2.dilate(src_sel, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        support = cv2.bitwise_and(support, gate)
        alpha, ink_gray = source_text_render(aligned_source, support)
        alpha = np.minimum(alpha, (255.0 - ink_gray) / 255.0)
        out = np.clip(out.astype(np.float32) * (1.0 - alpha[..., None]), 0, 255).astype(np.uint8)
    else:
        alpha = _soft_ink_alpha(aligned_source, src_sel, gate)
        a3 = alpha[..., None]
        out = np.clip(
            aligned_source.astype(np.float32) * a3 + out.astype(np.float32) * (1.0 - a3),
            0, 255,
        ).astype(np.uint8)
    write = np.maximum(clear, (alpha * 255).astype(np.uint8))
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
        "text_composite": "neutral_ink_over_target" if color_ink else "source_rgb",
        "source_ink_pixels": src_count,
        "target_ink_pixels": tgt_count,
        "clear_pixels": int(cv2.countNonZero(clear)),
        "boundary_touch": boundary_touch,
        "clear_mask": clear,
        "target_ink_mask": tgt_sel,
    }



_transfer_open_complex_text_region_full = _transfer_open_complex_text_region


def _semantic_local_change_ratio(
    aligned_source: np.ndarray,
    target: np.ndarray,
    gate: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    source_ink_map: np.ndarray | None = None,
    target_ink_map: np.ndarray | None = None,
) -> float:
    """Return a tolerant local ink-shape difference score in ``gate``.

    The score is used only for standalone Koharu text/SFX recovery.  Bubble
    regions already have stronger container semantics; free text printed on
    artwork needs an extra proof that SOURCE and TARGET actually differ before
    any automatic pixel write is attempted.
    """
    if gate is None or cv2.countNonZero(gate) <= 0:
        return 0.0
    shape = gate.shape[:2]
    src_map = source_ink_map if isinstance(source_ink_map, np.ndarray) and source_ink_map.shape == shape else _complex_text_ink_map(aligned_source)
    tgt_map = target_ink_map if isinstance(target_ink_map, np.ndarray) and target_ink_map.shape == shape else _complex_text_ink_map(target)
    src = (src_map > 0) & (gate > 0)
    tgt = (tgt_map > 0) & (gate > 0)
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



def _semantic_page_furniture_gate(gate: np.ndarray, shape: tuple[int, int], cfg) -> bool:
    """Return True for tiny running-header/page-number semantic text."""
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
    bottom_ratio = min(float(getattr(cfg, "paired_diff_page_furniture_bottom_ratio", 0.965)), 0.94)
    return bool((y1 <= h * top_ratio or y0 >= h * bottom_ratio) and narrow and short)

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
    authority_allow = np.zeros(shape, np.uint8)
    records: list[MaskTransferRecord] = []
    patch_matches: list[BubblePatchMatch] = []
    source_ink_cache: np.ndarray | None = None
    semantic_target_ink_cache: np.ndarray | None = None
    rendered_target_ink_cache: np.ndarray | None = None

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

    def standalone_raw_group_mask(row: Any, polygon_gate: np.ndarray) -> tuple[np.ndarray | None, float]:
        """Return a bounded TARGET raw-box rescue for severely undersegmented open text.

        Koharu can emit a narrow polygon tracing only one stroke while ``raw_box``
        spans the whole slanted letter/note.  This is analysis/clear authority only;
        the renderer below still admits SOURCE pixels component-by-component on a
        proven neutral paper surface.
        """
        meta = dict(getattr(row, "meta", {}) or {})
        raw = list(meta.get("raw_box") or [])
        if len(raw) != 4 or polygon_gate.shape != shape:
            return None, 0.0
        x0, y0, x1, y1 = [float(v) for v in raw]
        ix0 = max(0, min(shape[1] - 1, int(np.floor(min(x0, x1)))))
        iy0 = max(0, min(shape[0] - 1, int(np.floor(min(y0, y1)))))
        ix1 = max(ix0 + 1, min(shape[1], int(np.ceil(max(x0, x1)))))
        iy1 = max(iy0 + 1, min(shape[0], int(np.ceil(max(y0, y1)))))
        raw_area = max(1, (ix1 - ix0) * (iy1 - iy0))
        poly_area = max(1, cv2.countNonZero(polygon_gate))
        ratio = float(raw_area / poly_area)
        min_ratio = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_underseg_ratio", 2.0))
        max_page = float(getattr(cfg, "koharu_semantic_paired_raw_group_max_page_area_ratio", 0.22))
        min_conf = float(getattr(cfg, "koharu_semantic_paired_raw_group_min_target_confidence", 0.35))
        if float(getattr(row, "confidence", 0.0) or 0.0) < min_conf:
            return None, ratio
        if ratio < min_ratio or raw_area / max(1.0, float(shape[0] * shape[1])) > max_page:
            return None, ratio
        if (ix1 - ix0) < 40 or (iy1 - iy0) < 40:
            return None, ratio
        out = np.zeros(shape, np.uint8)
        out[iy0:iy1, ix0:ix1] = 255
        return out, ratio

    # First recover missed semantic containers. A text child or a saturated
    # target fill is sufficient evidence that this is a text-bearing region.
    candidates: list[tuple[str, Any, np.ndarray, bool, float]] = []
    for row in sorted(bubbles, key=lambda x: float(getattr(x, "confidence", 0.0)), reverse=True):
        if float(getattr(row, "confidence", 0.0)) < bubble_min:
            continue
        gate = np.asarray(getattr(row, "mask", None), dtype=np.uint8) if getattr(row, "mask", None) is not None else None
        if gate is None or gate.shape != shape or cv2.countNonZero(gate) <= 0:
            continue
        if overlap_ratio(gate, exclude_mask) > max_overlap:
            continue
        has_text_child = any(overlap_ratio(np.asarray(t.mask, dtype=np.uint8), gate) >= 0.35 for t in texts if getattr(t, "mask", None) is not None)
        coloured = _dominant_saturated_container_mask(target, gate, cfg) is not None
        if not has_text_child and not coloured:
            continue
        candidates.append(("bubble", row, gate, False, 0.0))

    # Then recover high-confidence standalone text not already owned by a bubble.
    # These are always marked reviewable and require changed-ink proof.
    for row in sorted(texts, key=lambda x: float(getattr(x, "confidence", 0.0)), reverse=True):
        if float(getattr(row, "confidence", 0.0)) < text_min:
            continue
        gate = np.asarray(getattr(row, "mask", None), dtype=np.uint8) if getattr(row, "mask", None) is not None else None
        if gate is None or gate.shape != shape or cv2.countNonZero(gate) <= 0:
            continue
        if overlap_ratio(gate, exclude_mask) > max_overlap:
            continue
        if any(overlap_ratio(gate, np.asarray(b.mask, dtype=np.uint8)) >= 0.55 for b in bubbles if getattr(b, "mask", None) is not None):
            continue
        if _semantic_page_furniture_gate(gate, shape, cfg):
            continue
        if source_ink_cache is None:
            source_ink_cache = _complex_text_ink_map(aligned_source)
        if semantic_target_ink_cache is None:
            semantic_target_ink_cache = _complex_text_ink_map(target)
        local_change_ratio = _semantic_local_change_ratio(
            aligned_source, target, gate, cfg,
            source_ink_map=source_ink_cache,
            target_ink_map=semantic_target_ink_cache,
        )
        raw_group_gate, _raw_underseg = standalone_raw_group_mask(row, gate)
        if local_change_ratio < change_min and raw_group_gate is None:
            continue
        candidates.append((str(getattr(row, "label", "text") or "text"), row, gate, True, local_change_ratio))

    for index, (kind, row, gate, standalone, local_change_ratio) in enumerate(candidates):
        # Identical numbers/signage in both editions require no replacement.
        # Treating them as translated containers erases unchanged shared glyphs
        # and can blank the burst rays around them (real p-110 regression).
        equivalence_gate = gate
        if kind == "bubble" and not standalone:
            # Burst rays and colour gradients are container artwork. Compare
            # the container's actual compact lettering in both editions.
            inner = cv2.erode(gate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
            shared_text_scope = np.maximum(
                target_text_mask_in_container(aligned_source, inner),
                target_text_mask_in_container(target, inner),
            )
            if cv2.countNonZero(shared_text_scope) > 0:
                equivalence_gate = shared_text_scope
        if equivalent_registered_ink(aligned_source, target, equivalence_gate):
            continue
        active_exclude = np.maximum(exclude_mask, composite)
        if overlap_ratio(gate, active_exclude) > max_overlap:
            continue
        if source_ink_cache is None:
            source_ink_cache = _complex_text_ink_map(aligned_source)
        if rendered_target_ink_cache is None:
            rendered_target_ink_cache = _complex_text_ink_map(rendered)
        raw_group_gate, raw_underseg_ratio = (standalone_raw_group_mask(row, gate) if standalone else (None, 0.0))
        raw_group_used = False
        effective_gate = gate
        text_img = None
        write_mask = np.zeros(shape, np.uint8)
        source_ink_mask = np.zeros(shape, np.uint8)
        diag: dict[str, object] = {"reason": "not_attempted"}
        empty_source_clear_used = False
        if kind == "bubble" and not standalone:
            text_img, write_mask, source_ink_mask, diag = _clear_empty_source_semantic_bubble(
                aligned_source, rendered, gate, cfg
            )
            empty_source_clear_used = bool(
                text_img is not None and str(diag.get("reason") or "") == "ok_empty_source_bubble_clear"
            )
        if raw_group_gate is not None and not empty_source_clear_used:
            # A late expanded raw group must never overwrite a region already
            # claimed by an earlier bubble/caption transfer (p-203 ownership).
            raw_group_gate = raw_group_gate.copy()
            raw_group_gate[active_exclude > 0] = 0
            if cv2.countNonZero(raw_group_gate) > 0:
                text_img, write_mask, source_ink_mask, diag = _strict_raw_group_plain_paper_text_transfer(
                    aligned_source, rendered, raw_group_gate, cfg
                )
                if text_img is not None and cv2.countNonZero(write_mask) > 0:
                    raw_group_used = True
                    effective_gate = raw_group_gate
        if not raw_group_used and not empty_source_clear_used:
            target_text_hint = np.zeros(shape, np.uint8)
            if not standalone:
                for child in texts:
                    child_mask = getattr(child, "mask", None)
                    if child_mask is not None and overlap_ratio(np.asarray(child_mask, dtype=np.uint8), gate) >= 0.35:
                        target_text_hint = np.maximum(target_text_hint, np.asarray(child_mask, dtype=np.uint8))
            text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
                aligned_source, rendered, gate, cfg,
                source_ink_map=source_ink_cache,
                target_ink_map=rendered_target_ink_cache,
                target_clear_hint=target_text_hint,
                region_pad_px=(3 if not standalone else None),
            )
        rid = str((getattr(row, "meta", {}) or {}).get("item_index", index))
        source_id = f"koharu-semantic-src-{kind}-{rid}"
        target_id = f"koharu-semantic-dst-{kind}-{rid}"
        conf = float(getattr(row, "confidence", 0.0))
        rec = MaskTransferRecord(source_id, target_id, conf, False, "not_applied")
        rec.geometry_mode = "free_text" if standalone else "complex_text"
        box = _bbox_from_mask(effective_gate) or (0, 0, 0, 0)
        rec.source_bbox = tuple(int(v) for v in box)
        rec.target_bbox = tuple(int(v) for v in box)
        rec.sr_backend = str(diag.get("clear_backend") or "koharu-semantic-registered-ink")
        rec.meta = {
            "layout_label": str(getattr(row, "label", kind)),
            "layout_confidence": conf,
            "semantic_completion": True,
            "standalone": bool(standalone),
            "local_change_ratio": round(float(local_change_ratio), 4),
            "raw_group_plain_paper": bool(raw_group_used),
            "raw_group_underseg_ratio": round(float(raw_underseg_ratio), 4),
            "raw_group_reason": str(diag.get("reason") or ""),
        }
        patch_matches.append(BubblePatchMatch(
            source_id, target_id, conf, 0.0, 1.0, 0.0, 1.0,
            ["koharu-semantic-first", "registered-raster-ink", "background-preserving"],
        ))
        if text_img is None or cv2.countNonZero(write_mask) <= 0:
            rec.reason = str(diag.get("reason") or "koharu_semantic_component_transfer_failed")
            rec.review_required = True
            rec.review_reason = rec.reason
            rec.restorable = True
            rec.editable = True
            records.append(rec)
            continue

        rendered = text_img
        # The target ink cache describes the previous rendered revision exactly.
        # Invalidate only after a successful pixel write; failed candidates may
        # safely reuse it byte-for-byte.
        rendered_target_ink_cache = None
        composite = np.maximum(composite, write_mask)
        diag_clear = diag.get("clear_mask")
        if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
            clear_all = np.maximum(clear_all, diag_clear)
        if raw_group_used:
            # The TARGET Koharu polygon can under-segment a whole letter into a
            # tiny stroke while its raw_box spans the real typography.  Grant
            # explicit ALLOW only to pixels that the strict plain-paper changed-
            # glyph transfer actually cleared/wrote; never to the raw rectangle.
            authority_allow = np.maximum(authority_allow, write_mask)
            if isinstance(diag_clear, np.ndarray) and diag_clear.shape == shape:
                authority_allow = np.maximum(authority_allow, diag_clear)
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
        if bool(diag.get("empty_source_clear", False)):
            rec.source_ink_coverage = 1.0
            rec.target_residual_ratio = float(diag.get("target_residual_ratio", 1.0))
            rec.content_check = "checked_empty_source"
            rec.content_complete = bool(diag.get("content_complete", False))
            rec.meta["empty_source_clear"] = {
                k: v for k, v in diag.items()
                if k not in {"clear_mask", "target_ink_mask"}
            }
        else:
            _evaluate_content_completeness(
                rec, source_ink_mask, diag.get("target_ink_mask"), rendered, cfg,
                target_invalidation_mask=write_mask,
            )
        if not bool(getattr(rec, "content_complete", False)) and not bool(diag.get("empty_source_clear", False)):
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
        strict_raw_group_safe = bool(
            raw_group_used
            and bool(getattr(rec, "content_complete", False))
            and float(getattr(rec, "source_ink_coverage", 0.0) or 0.0) >= 0.98
            and float(getattr(rec, "target_residual_ratio", 1.0)) <= 0.01
        )
        if (standalone and not strict_raw_group_safe) or boundary_touch or not bool(getattr(rec, "content_complete", False)):
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
    if cv2.countNonZero(authority_allow) > 0:
        result.authority_allow_mask = authority_allow
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
    source_ink_cache: np.ndarray | None = None
    rendered_target_ink_cache: np.ndarray | None = None
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

        if source_ink_cache is None:
            source_ink_cache = _complex_text_ink_map(aligned_source)
        if rendered_target_ink_cache is None:
            rendered_target_ink_cache = _complex_text_ink_map(rendered)
        text_img, write_mask, source_ink_mask, diag = _transfer_open_complex_text_region(
            aligned_source, rendered, gate, cfg,
            source_ink_map=source_ink_cache,
            target_ink_map=rendered_target_ink_cache,
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
        rendered_target_ink_cache = None
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
    # Measure paper uniformity on the detected paper support rather than on the
    # whole bubble. Japanese glyphs are intentionally dark and previously pushed
    # the p10/p90 spread above the white-container threshold, making the white-
    # bubble master switch ineffective on exactly the pages it was meant to own.
    paper_vals = gray[(paper > 0) & (region_u8 > 0)]
    robust_spread = float(np.percentile(paper_vals, 90.0) - np.percentile(paper_vals, 10.0)) if paper_vals.size > 0 else 255.0
    min_ratio = float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68))
    max_spread = float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0))
    # A near-total white-paper support is itself strong evidence. JPEG edges,
    # antialiased JP glyphs and photographed paper can keep the robust spread a
    # little above the strict uniformity threshold even though >94% of the region
    # is detected as white paper (real page 099 is ~99.6%).
    high_paper_ratio = paper_ratio >= max(0.94, min_ratio)
    ok = bool(paper_ratio >= min_ratio and (robust_spread <= max_spread or high_paper_ratio))
    return ok, {
        "is_white": ok,
        "reason": ("paper_ratio_high_confidence" if high_paper_ratio and robust_spread > max_spread else "paper_ratio_and_spread") if ok else "not_uniform_white",
        "paper_ratio": paper_ratio,
        "robust_spread": robust_spread,
        "min_paper_ratio": min_ratio,
        "max_robust_spread": max_spread,
    }


def _white_bubble_enhancement_policy(
    target_image: np.ndarray,
    target_region_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> dict[str, object]:
    """Resolve the white-bubble enhancement master switch for this private mode.

    v2.3.67 makes the UI switch authoritative.  Previously disabling the
    ``white-bubble Chinese enhancement`` path only disabled the newer clarity
    normaliser while the photo-pair crisp/pixel/ink ladders could still rebuild
    exactly the same Chinese glyphs.  A verified white TARGET container now has
    one master policy: OFF means preserve the aligned SOURCE glyph raster and
    bypass every optional glyph enhancement/reconstruction stage.  Japanese
    clearing, mask geometry and TARGET structure restoration remain unaffected.
    """
    enabled = bool(getattr(cfg, "direct_white_clarity_enhance_enabled", False))
    white_ok, white_diag = _target_mask_is_white_container(target_image, target_region_mask, cfg)
    bypass = bool(white_ok and not enabled)
    return {
        "enabled": enabled,
        "target_region": white_diag,
        "master_bypass": bypass,
        "policy": "enhance_enabled" if enabled else ("preserve_source_raster" if bypass else "not_white_container"),
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
    policy = _white_bubble_enhancement_policy(target_image, target_region_mask, cfg)
    enabled = bool(policy["enabled"])
    white_ok = bool((policy.get("target_region") or {}).get("is_white", False))
    diag: dict[str, object] = {
        **policy,
        "applied": False,
        "reason": "disabled_master_bypass" if bool(policy.get("master_bypass")) else ("disabled" if not enabled else "target_not_white"),
    }
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


def _mask_replace_source_structure_guard(
    source: np.ndarray,
    source_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Protect SOURCE lettering while rejecting only actual boundary artwork.

    v2.3.59 fixes an over-conservative failure mode where a bubble outline/ray and
    nearby Chinese glyphs became one connected component.  Older code rejected
    the *entire* component once any part touched the boundary, which silently
    deleted valid SOURCE lettering.  Structural classification may still inspect
    the whole component, but ordinary ambiguous components contribute only their
    boundary-near pixels to the guard.  Only unmistakably thin/long line art may
    be rejected as a whole.  Burst/spiky containers keep their explicit boundary
    annulus so radial artwork remains SOURCE-structure, never lettering.
    """
    out = np.zeros(source_mask.shape, np.uint8)
    enabled = bool(getattr(cfg, "mask_source_structure_guard_enabled", True))
    diag: dict[str, object] = {"enabled": enabled, "guard_pixels": 0, "components": 0}
    if not enabled or source.shape[:2] != source_mask.shape or cv2.countNonZero(source_mask) <= 0:
        return out, diag
    use = (source_mask > 0).astype(np.uint8)
    nz = cv2.findNonZero(use)
    if nz is None:
        return out, diag
    bx, by, bw, bh = cv2.boundingRect(nz)
    min_dim = max(1, min(int(bw), int(bh)))
    ratio = float(getattr(cfg, "mask_source_structure_guard_ratio", 0.035))
    min_px = max(2, int(getattr(cfg, "mask_source_structure_guard_min_px", 5)))
    max_px = max(min_px, int(getattr(cfg, "mask_source_structure_guard_max_px", 14)))
    band_px = int(np.clip(round(min_dim * ratio), min_px, max_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1))
    inner = cv2.erode(use, k, iterations=1)
    boundary = (use > 0) & (inner == 0)
    # A wider annulus is used only for clipping ambiguous connected components.
    # This prevents a border-connected CJK cluster from being deleted deep inside
    # the speech balloon while still removing antialiased outline/tail fragments.
    clip_px = max(band_px, min(max_px * 2, band_px * 2))
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clip_px * 2 + 1, clip_px * 2 + 1))
    clip_inner = cv2.erode(use, ck, iterations=1)
    boundary_clip = (use > 0) & (clip_inner == 0)

    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source.astype(np.uint8)
    vals = gray[use > 0]
    paper = float(np.percentile(vals, 78.0)) if vals.size else 245.0
    dark_thr = int(np.clip(paper - 38.0, 70, 205))
    dark = ((gray <= dark_thr) & (use > 0)).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    region_area = max(1, int(cv2.countNonZero(use)))
    min_area = max(4, int(getattr(cfg, "mask_source_structure_min_component_area", 10)))
    min_aspect = float(getattr(cfg, "mask_source_structure_min_aspect", 2.6))
    min_span = float(getattr(cfg, "mask_source_structure_min_span_ratio", 0.16))
    kept = 0
    clipped_components = 0
    full_line_components = 0
    for lab in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[lab]]
        if area < min_area:
            continue
        comp = labels == lab
        touch = int(np.count_nonzero(comp & boundary))
        if touch <= 0:
            continue
        boundary_fraction = float(touch / max(1, area))
        aspect = max(ww / max(1.0, float(hh)), hh / max(1.0, float(ww)))
        span = max(ww / max(1.0, float(bw)), hh / max(1.0, float(bh)))
        fill = float(area / max(1.0, float(ww * hh)))
        large_art = bool(area >= max(36, int(round(region_area * 0.018))) and boundary_fraction >= 0.12)
        line_art = bool(boundary_fraction >= 0.18 and (aspect >= min_aspect or span >= min_span))
        outline_art = bool(boundary_fraction >= 0.10 and span >= 0.34 and fill <= 0.55)
        if not (line_art or outline_art or large_art):
            continue
        # Only unmistakably thin/long rules/rays are safe to reject wholesale.
        # Large mixed components are clipped to the boundary annulus because they
        # may contain both the bubble outline and legitimate Chinese glyphs.
        strong_line = bool(
            aspect >= max(4.2, min_aspect * 1.55)
            and span >= max(0.24, min_span)
            and fill <= 0.30
            and boundary_fraction >= 0.06
        )
        if strong_line:
            sel = comp
            full_line_components += 1
        else:
            sel = comp & boundary_clip
            clipped_components += 1
        if np.any(sel):
            out[sel] = 255
            kept += 1

    spiky_profile = _mask_replace_spiky_container_profile(source_mask)
    spiky_band_px = 0
    spiky_band_pixels = 0
    if bool(spiky_profile.get("spiky", False)):
        spiky_band_px = max(1, int(getattr(cfg, "mask_source_spiky_boundary_band_px", 14)))
        bk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spiky_band_px * 2 + 1, spiky_band_px * 2 + 1))
        spiky_inner = cv2.erode(use, bk, iterations=1)
        spiky_band = (use > 0) & (spiky_inner == 0)
        out[spiky_band] = 255
        spiky_band_pixels = int(np.count_nonzero(spiky_band))
    if cv2.countNonZero(out) > 0:
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        out[use == 0] = 0

    # SOURCE lettering is an independent authority.  The text selector is run
    # on the original SOURCE raster before any structure veto; compact glyphs
    # are then relieved from the structural guard, including a true burst's
    # boundary annulus.  This preserves edge-near Chinese without copying rays.
    lettering_relief = np.zeros_like(out)
    lettering_relief_enabled = bool(getattr(cfg, "mask_source_lettering_relief_enabled", True))
    lettering_relief_dilate = max(0, int(getattr(cfg, "mask_source_lettering_relief_dilate_px", 1)))
    if lettering_relief_enabled:
        lettering_relief = target_text_mask_in_container(source, source_mask)
        if lettering_relief_dilate > 0 and cv2.countNonZero(lettering_relief) > 0:
            lk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (lettering_relief_dilate * 2 + 1, lettering_relief_dilate * 2 + 1))
            lettering_relief = cv2.dilate(lettering_relief, lk, iterations=1)
            lettering_relief = cv2.bitwise_and(lettering_relief, (use * 255).astype(np.uint8))
        if cv2.countNonZero(lettering_relief) > 0:
            out[lettering_relief > 0] = 0
    diag.update({
        "guard_pixels": int(cv2.countNonZero(out)),
        "components": int(kept),
        "clipped_components": int(clipped_components),
        "full_line_components": int(full_line_components),
        "band_px": int(band_px),
        "component_clip_band_px": int(clip_px),
        "dark_threshold": int(dark_thr),
        "spiky": bool(spiky_profile.get("spiky", False)),
        "spiky_boundary_band_px": int(spiky_band_px),
        "spiky_boundary_band_pixels": int(spiky_band_pixels),
        "source_lettering_relief_enabled": bool(lettering_relief_enabled),
        "source_lettering_relief_pixels": int(cv2.countNonZero(lettering_relief)),
        "source_lettering_relief_dilate_px": int(lettering_relief_dilate),
    })
    return out, diag

def _mask_replace_target_immutable_boundary_band(
    target_mask: np.ndarray,
    source_ink_mask: np.ndarray,
    *,
    spiky: bool,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Freeze a TARGET boundary annulus so antialias/white-paper structure stays exact.

    Dark-line guards alone are insufficient: inpaint/full-clear can alter the gray
    antialias pixels around a balloon/burst outline and create a visible halo.  The
    immutable band preserves *all* TARGET pixels around the container boundary.
    Its inner width is capped by the nearest SOURCE glyph distance so Chinese is
    never clipped merely to protect the border.
    """
    out = np.zeros_like(target_mask, np.uint8)
    enabled = bool(getattr(cfg, "mask_target_immutable_band_enabled", True))
    diag: dict[str, object] = {"enabled": enabled, "band_pixels": 0, "spiky": bool(spiky)}
    use = (target_mask > 0).astype(np.uint8)
    if not enabled or cv2.countNonZero(use) <= 0:
        return out, diag
    requested_inner = max(1, int(getattr(
        cfg,
        "mask_target_spiky_inner_band_px" if spiky else "mask_target_ordinary_inner_band_px",
        18 if spiky else 6,
    )))
    outer_px = max(0, int(getattr(cfg, "mask_target_outer_band_px", 3)))
    text_margin = max(1, int(getattr(cfg, "mask_target_band_text_margin_px", 3)))

    distance_use = (_solidify_container_mask((use * 255).astype(np.uint8), cfg) > 0).astype(np.uint8)
    dist = cv2.distanceTransform(distance_use, cv2.DIST_L2, 3)
    ink_sel = (source_ink_mask > 0) & (distance_use > 0)
    inner_px = requested_inner
    nearest_ink_distance = None
    ink_distance_percentile = 25.0 if spiky else 10.0
    if np.any(ink_sel):
        ink_dist = dist[ink_sel]
        nearest_ink_distance = float(np.percentile(ink_dist, ink_distance_percentile))
        # Ordinary balloons can place punctuation close to the border, so retain
        # the glyph-distance cap.  Burst/spiky SOURCE boundary rays have already
        # been removed by the mode-private SOURCE structure guard; the HD TARGET
        # boundary can therefore be frozen at the full requested width.
        if not spiky:
            safe_cap = max(1, int(np.floor(nearest_ink_distance - text_margin)))
            inner_px = min(inner_px, safe_cap)
    inner_px = max(1, int(inner_px))

    er_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_px * 2 + 1, inner_px * 2 + 1))
    inner = cv2.erode(use, er_k, iterations=1)
    outer = use.copy()
    if outer_px > 0:
        out_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (outer_px * 2 + 1, outer_px * 2 + 1))
        outer = cv2.dilate(use, out_k, iterations=1)
    band = (outer > 0) & (inner == 0)
    out[band] = 255
    diag.update({
        "requested_inner_px": int(requested_inner),
        "effective_inner_px": int(inner_px),
        "outer_px": int(outer_px),
        "text_margin_px": int(text_margin),
        "source_ink_distance_percentile": float(ink_distance_percentile),
        "nearest_source_ink_distance": nearest_ink_distance,
        "band_pixels": int(cv2.countNonZero(out)),
    })
    return out, diag


def _mask_replace_target_structure_guard(
    target: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray, dict[str, object]]:
    """Protect actual TARGET HD structure without freezing Japanese glyphs.

    v2.3.59 separates structure authority from target-language clear authority.
    Boundary-connected dark components are no longer restored wholesale merely
    because they touch the balloon edge: mixed outline+glyph components are
    clipped to a narrow boundary annulus.  Only unmistakably thin/long rules are
    preserved in full.
    """
    out = np.zeros(target_mask.shape, np.uint8)
    enabled = bool(getattr(cfg, "mask_target_structure_guard_enabled", True))
    diag: dict[str, object] = {"enabled": enabled, "guard_pixels": 0, "components": 0}
    if not enabled or target.shape[:2] != target_mask.shape or cv2.countNonZero(target_mask) <= 0:
        return out, diag
    probe_px = max(0, int(getattr(cfg, "mask_target_border_probe_dilate_px", 4)))
    fringe_px = max(0, int(getattr(cfg, "mask_target_border_restore_fringe_px", 2)))
    use = (target_mask > 0).astype(np.uint8) * 255
    probe = use.copy()
    if probe_px > 0:
        pk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (probe_px * 2 + 1, probe_px * 2 + 1))
        probe = cv2.dilate(probe, pk, iterations=1)
    base = target_container_border_mask(target, probe, band_px=max(5, probe_px + 2))

    inner_px = max(2, min(5, probe_px + 1))
    ik = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inner_px * 2 + 1, inner_px * 2 + 1))
    inner = cv2.erode((use > 0).astype(np.uint8), ik, iterations=1)
    outer = probe > 0
    ring = outer & (inner == 0)
    clip_px = max(inner_px + 2, min(12, inner_px * 2))
    ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clip_px * 2 + 1, clip_px * 2 + 1))
    clip_inner = cv2.erode((use > 0).astype(np.uint8), ck, iterations=1)
    ring_clip = outer & (clip_inner == 0)

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target.astype(np.uint8)
    vals = gray[use > 0]
    paper = float(np.percentile(vals, 72.0)) if vals.size else 245.0
    thr = int(np.clip(paper - 46.0, 70, 195))
    dark = ((gray <= thr) & outer).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    nz = cv2.findNonZero((use > 0).astype(np.uint8))
    bx, by, bw, bh = cv2.boundingRect(nz) if nz is not None else (0, 0, target_mask.shape[1], target_mask.shape[0])
    region_area = max(1, int(cv2.countNonZero(use)))
    components = 0
    clipped_components = 0
    full_line_components = 0
    supplement = np.zeros_like(out)
    for lab in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[lab]]
        if area < 5:
            continue
        comp = labels == lab
        touch = int(np.count_nonzero(comp & ring))
        if touch <= 0:
            continue
        aspect = max(ww / max(1.0, float(hh)), hh / max(1.0, float(ww)))
        span = max(ww / max(1.0, float(bw)), hh / max(1.0, float(bh)))
        fill = float(area / max(1.0, float(ww * hh)))
        boundary_fraction = float(touch / max(1, area))
        structural = bool(
            span >= 0.18
            or aspect >= 2.8
            or (area >= max(30, int(round(region_area * 0.010))) and boundary_fraction >= 0.12)
            or (span >= 0.10 and fill <= 0.38 and boundary_fraction >= 0.20)
        )
        if not structural:
            continue
        strong_line = bool(aspect >= 4.5 and span >= 0.24 and fill <= 0.28 and boundary_fraction >= 0.05)
        if strong_line:
            sel = comp
            full_line_components += 1
        else:
            sel = comp & ring_clip
            clipped_components += 1
        if np.any(sel):
            supplement[sel] = 255
            components += 1
    out = cv2.bitwise_or(base, supplement)
    if fringe_px > 0 and cv2.countNonZero(out) > 0:
        fk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (fringe_px * 2 + 1, fringe_px * 2 + 1))
        out = cv2.dilate(out, fk, iterations=1)

    # TARGET lettering and TARGET structure are different authorities.  Relieve
    # compact Japanese glyph support from the border/structure guard so a scan
    # connection to an outline cannot make old text immutable.
    text_relief = np.zeros_like(out)
    text_relief_enabled = bool(getattr(cfg, "mask_target_structure_text_relief_enabled", True))
    text_relief_dilate = max(0, int(getattr(cfg, "mask_target_structure_text_relief_dilate_px", 1)))
    if text_relief_enabled:
        text_relief = target_text_mask_in_container(target, use)
        if text_relief_dilate > 0 and cv2.countNonZero(text_relief) > 0:
            tk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (text_relief_dilate * 2 + 1, text_relief_dilate * 2 + 1))
            text_relief = cv2.dilate(text_relief, tk, iterations=1)
            text_relief = cv2.bitwise_and(text_relief, use)
        if cv2.countNonZero(text_relief) > 0:
            out[text_relief > 0] = 0
    diag.update({
        "guard_pixels": int(cv2.countNonZero(out)),
        "components": int(components),
        "clipped_components": int(clipped_components),
        "full_line_components": int(full_line_components),
        "base_border_pixels": int(cv2.countNonZero(base)),
        "probe_dilate_px": int(probe_px),
        "restore_fringe_px": int(fringe_px),
        "component_clip_band_px": int(clip_px),
        "dark_threshold": int(thr),
        "target_text_relief_enabled": bool(text_relief_enabled),
        "target_text_relief_pixels": int(cv2.countNonZero(text_relief)),
        "target_text_relief_dilate_px": int(text_relief_dilate),
    })
    return out, diag


def _rigid_changed_text_write_gate(
    source: np.ndarray,
    target_reference: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Return SOURCE-crop changed-text authority for rigid raster writes.

    The admission path already proves edition-exclusive compact SOURCE text.
    Preserve that proof at render time so shared illustration ink inside a speech
    container is never reintroduced into a colour TARGET.  The mask is mapped
    back into SOURCE-crop coordinates and slightly dilated for antialiasing.
    """
    diag: dict[str, object] = {"enabled": bool(getattr(cfg, "rigid_container_changed_text_gate_enabled", True)), "applied": False}
    if not bool(diag["enabled"]):
        diag["reason"] = "disabled"
        return None, diag
    sbox = _bbox_from_mask(source_mask); tbox = _bbox_from_mask(target_mask)
    if sbox is None or tbox is None:
        diag["reason"] = "empty_mask"
        return None, diag
    sx0, sy0, sx1, sy1 = [int(v) for v in sbox]
    tx0, ty0, tx1, ty1 = [int(v) for v in tbox]
    source_crop = source[sy0:sy1, sx0:sx1]
    target_crop = target_reference[ty0:ty1, tx0:tx1]
    source_mask_crop = source_mask[sy0:sy1, sx0:sx1]
    target_mask_crop = target_mask[ty0:ty1, tx0:tx1]
    if source_crop.size == 0 or target_crop.size == 0:
        diag["reason"] = "empty_crop"
        return None, diag
    th, tw = target_crop.shape[:2]
    source_to_target = cv2.resize(source_crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    source_mask_to_target = cv2.resize(source_mask_crop, (tw, th), interpolation=cv2.INTER_NEAREST)
    region_mask = np.maximum((source_mask_to_target > 0).astype(np.uint8), (target_mask_crop > 0).astype(np.uint8)) * 255
    source_text, _target_text, text_diag = changed_text_masks(
        source_to_target, target_crop, region_mask,
        tolerance_px=max(1, int(getattr(cfg, "rigid_container_source_text_tolerance_px", 2))),
        min_unique_ratio=float(getattr(cfg, "rigid_container_source_text_unique_ratio", 0.055)),
        max_component_fraction=float(getattr(cfg, "rigid_container_source_text_max_component_fraction", 0.10)),
    )
    if cv2.countNonZero(source_text) <= 0:
        diag.update({"reason": "no_changed_text", "text_diag": text_diag})
        return None, diag
    seed = cv2.resize(source_text, (source_crop.shape[1], source_crop.shape[0]), interpolation=cv2.INTER_NEAREST)
    dilate_px = max(0, int(getattr(cfg, "rigid_container_changed_text_gate_dilate_px", 1)))
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        seed = cv2.dilate(seed, k, iterations=1)
    seed = cv2.bitwise_and(seed, source_mask_crop)

    # `changed_text_masks` proves edition-exclusive lettering but is deliberately
    # subtractive; coincident SOURCE/TARGET strokes are absent from that mask.  Do
    # not render the unique seed itself.  Instead use it to select whole SOURCE
    # text components from the conservative text selector.  This keeps complete
    # Chinese glyphs while still excluding shared illustration ink (p-194 class).
    full_text = target_text_mask_in_container(source_crop, source_mask_crop)
    gate = np.zeros_like(seed)
    selected_components = 0
    total_components = 0
    if cv2.countNonZero(full_text) > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((full_text > 0).astype(np.uint8), 8)
        total_components = max(0, int(n - 1))
        # Seed proximity is slightly more forgiving than the final write fringe: a
        # glyph can share most strokes with JP but still contain one unique radical.
        sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        select_seed = cv2.dilate(seed, sk, iterations=1)
        for idx in range(1, n):
            comp = labels == idx
            if np.any(comp & (select_seed > 0)):
                gate[comp] = 255
                selected_components += 1
    if cv2.countNonZero(gate) <= 0:
        gate = seed
    # Preserve antialiasing around the selected full glyphs without reconnecting
    # unrelated artwork.
    if dilate_px > 0 and cv2.countNonZero(gate) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        gate = cv2.dilate(gate, k, iterations=1)
    gate = cv2.bitwise_and(gate, source_mask_crop)
    diag.update({
        "reason": "ok",
        "authority": "changed_text_seeded_full_source_text_components",
        "changed_text_pixels_target_space": int(cv2.countNonZero(source_text)),
        "seed_pixels_source_space": int(cv2.countNonZero(seed)),
        "gate_pixels_source_space": int(cv2.countNonZero(gate)),
        "full_text_pixels_source_space": int(cv2.countNonZero(full_text)),
        "selected_components": int(selected_components),
        "total_components": int(total_components),
        "dilate_px": int(dilate_px),
        "text_diag": text_diag,
    })
    return gate, diag


def _rigid_target_changed_text_relief(
    source: np.ndarray,
    target_reference: np.ndarray,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    target_bubble: BubbleInstance | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return TARGET-space edition-exclusive text that may be cleared safely."""
    shape = target_reference.shape[:2]
    empty = np.zeros(shape, np.uint8)
    diag: dict[str, object] = {"enabled": bool(getattr(cfg, "rigid_container_target_changed_text_relief_enabled", True)), "pixels": 0}
    if not bool(diag["enabled"]):
        diag["reason"] = "disabled"
        return empty, diag
    sbox = _bbox_from_mask(source_mask); tbox = _bbox_from_mask(target_mask)
    if sbox is None or tbox is None:
        diag["reason"] = "empty_mask"
        return empty, diag
    sx0, sy0, sx1, sy1 = [int(v) for v in sbox]
    tx0, ty0, tx1, ty1 = [int(v) for v in tbox]
    source_crop = source[sy0:sy1, sx0:sx1]
    target_crop = target_reference[ty0:ty1, tx0:tx1]
    source_mask_crop = source_mask[sy0:sy1, sx0:sx1]
    target_mask_crop = target_mask[ty0:ty1, tx0:tx1]
    if source_crop.size == 0 or target_crop.size == 0:
        diag["reason"] = "empty_crop"
        return empty, diag
    th, tw = target_crop.shape[:2]
    source_to_target = cv2.resize(source_crop, (tw, th), interpolation=cv2.INTER_LINEAR)
    source_mask_to_target = cv2.resize(source_mask_crop, (tw, th), interpolation=cv2.INTER_NEAREST)
    region_mask = np.maximum((source_mask_to_target > 0).astype(np.uint8), (target_mask_crop > 0).astype(np.uint8)) * 255
    _source_text, target_text, text_diag = changed_text_masks(
        source_to_target, target_crop, region_mask,
        tolerance_px=max(1, int(getattr(cfg, "rigid_container_source_text_tolerance_px", 2))),
        min_unique_ratio=float(getattr(cfg, "rigid_container_source_text_unique_ratio", 0.055)),
        max_component_fraction=float(getattr(cfg, "rigid_container_source_text_max_component_fraction", 0.10)),
    )
    relief = np.zeros(shape, np.uint8)
    # `changed_text_masks` is subtractive: TARGET strokes that overlap SOURCE
    # disappear from the edition-exclusive mask. Treat those unique TARGET
    # pixels only as a semantic seed, then recover the *whole* TARGET text
    # components touched by the seed. This clears complete Japanese glyphs while
    # shared illustration components remain outside the clear authority.
    unique_seed = cv2.bitwise_and(target_text, target_mask_crop)
    full_target_text = target_text_mask_in_container(target_crop, target_mask_crop)
    local = np.zeros_like(unique_seed)
    selected_components = 0
    total_components = 0
    if cv2.countNonZero(full_target_text) > 0 and cv2.countNonZero(unique_seed) > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats((full_target_text > 0).astype(np.uint8), 8)
        total_components = max(0, int(n - 1))
        select_seed = cv2.dilate(
            unique_seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1
        )
        for idx in range(1, n):
            comp = labels == idx
            if np.any(comp & (select_seed > 0)):
                local[comp] = 255
                selected_components += 1
    if cv2.countNonZero(local) <= 0:
        local = unique_seed.copy()
    dilate_px = max(0, int(getattr(cfg, "rigid_container_target_changed_text_relief_dilate_px", 2)))
    if dilate_px > 0 and cv2.countNonZero(local) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        local = cv2.dilate(local, k, iterations=1)
        local = cv2.bitwise_and(local, target_mask_crop)
    relief[ty0:ty1, tx0:tx1] = local

    # Recover complete TARGET glyph support from Koharu's original semantic text
    # boxes. The postprocessed polygon can collapse to a small core on large bold
    # vertical text; `raw_box` is therefore used only to *clear* dark ink inside
    # the already-verified container. It never enlarges SOURCE write authority.
    raw_relief = np.zeros(shape, np.uint8)
    raw_domain_union = np.zeros(shape, np.uint8)
    raw_diag: dict[str, object] = {"enabled": bool(getattr(cfg, "rigid_container_target_raw_text_clear_enabled", True)), "accepted_boxes": 0}
    bubble_meta = dict((getattr(target_bubble, "meta", {}) or {})) if target_bubble is not None else {}
    hints = list(bubble_meta.get("koharu_layout_text_hints") or [])
    bubble_hints = list(bubble_meta.get("koharu_layout_bubble_hints") or [])
    region_kind = str(bubble_meta.get("paired_region_kind") or "")
    if bool(raw_diag["enabled"]) and hints:
        gray = cv2.cvtColor(target_reference, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(target_reference, cv2.COLOR_BGR2HSV)
        container_area = max(1, int(cv2.countNonZero(target_mask)))
        min_conf = float(getattr(cfg, "rigid_container_target_raw_text_min_confidence", 0.50))
        min_overlap = float(getattr(cfg, "rigid_container_target_raw_text_min_overlap", 0.20))
        max_fraction = float(getattr(cfg, "rigid_container_target_raw_text_max_area_fraction", 0.48))
        min_bright = float(getattr(cfg, "rigid_container_target_raw_text_min_bright_ratio", 0.42))
        max_sat = float(getattr(cfg, "rigid_container_target_raw_text_max_saturation_median", 80.0))
        gray_thr = int(getattr(cfg, "rigid_container_target_raw_text_gray_threshold", 228))
        inset = max(0, int(getattr(cfg, "rigid_container_target_raw_text_inset_px", 1)))
        accepted = []

        # Strongest overlapping semantic bubble.  It is deliberately used only
        # for TARGET cleanup on large complex/free text; SOURCE write authority
        # remains the paired-diff changed-text gate.
        semantic_bubble_mask = None
        semantic_bubble_hint = None
        if region_kind in {"complex_text", "free_text"} and bubble_hints:
            for bh in bubble_hints:
                if float(bh.get("confidence", 0.0) or 0.0) < min_conf:
                    continue
                if float(bh.get("overlap", 0.0) or 0.0) < min_overlap:
                    continue
                poly = list(bh.get("polygon") or [])
                if len(poly) < 3:
                    continue
                candidate = (rasterize_polygon(poly, shape) > 0).astype(np.uint8) * 255
                if cv2.countNonZero(candidate) <= 0:
                    continue
                semantic_bubble_mask = candidate
                semantic_bubble_hint = bh
                break

        semantic_bubble_distance = (
            cv2.distanceTransform((semantic_bubble_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
            if semantic_bubble_mask is not None else None
        )

        for hint in hints:
            if float(hint.get("confidence", 0.0) or 0.0) < min_conf or float(hint.get("overlap", 0.0) or 0.0) < min_overlap:
                continue
            box = list(hint.get("raw_box") or [])
            if len(box) != 4:
                continue
            x0, y0, x1, y1 = [int(round(float(v))) for v in box]
            x0 = max(0, min(shape[1], x0 + inset)); y0 = max(0, min(shape[0], y0 + inset))
            x1 = max(0, min(shape[1], x1 - inset)); y1 = max(0, min(shape[0], y1 - inset))
            if x1 <= x0 or y1 <= y0:
                continue
            raw_rect = np.zeros(shape, np.uint8); raw_rect[y0:y1, x0:x1] = 255
            base_roi = cv2.bitwise_and(raw_rect, target_mask)
            base_area = int(cv2.countNonZero(base_roi))
            if base_area <= 0:
                continue
            area_fraction = float(base_area / container_area)

            # Ordinary semantic boxes stay strictly clipped to paired-diff.  For
            # a large text box inside complex/free text, the paired polygon may be
            # just a notched changed-text lane.  Widen the *clear-only* domain to
            # the verified Koharu bubble polygon, still intersected with raw_box.
            roi = base_roi
            clear_domain = "paired_diff"
            if area_fraction > max_fraction and semantic_bubble_mask is not None:
                expanded = cv2.bitwise_and(raw_rect, semantic_bubble_mask)
                if cv2.countNonZero(expanded) > base_area:
                    roi = expanded
                    clear_domain = "koharu_bubble_polygon"

            pix = roi > 0
            area = int(np.count_nonzero(pix))
            if area <= 0:
                continue
            bright_ratio = float(np.mean(gray[pix] >= 220))
            sat_median = float(np.median(hsv[..., 1][pix]))
            if bright_ratio < min_bright or sat_median > max_sat:
                continue
            dark = ((gray < gray_thr) & pix).astype(np.uint8) * 255
            if cv2.countNonZero(dark) <= 0:
                continue

            # Large raw boxes never authorize rectangle clearing.  Changed-text
            # relief is only a seed; we recover whole TARGET dark components that
            # touch that seed inside the semantic domain.  Unseeded artwork and
            # burst boundaries remain protected.
            raw_policy = "bounded_raw_box"
            selected_dark = dark
            seeded_components = 0
            total_components = 0
            semantic_unseeded = 0
            semantic_unseeded_pixels = 0
            completion_components = 0
            completion_pixels = 0
            semantic_full_components = 0
            semantic_full_pixels = 0
            edge_spill_components = 0
            edge_spill_pixels = 0
            edge_spill_seeded_components = 0
            edge_spill_neighbor_components = 0
            edge_spill_padding_px = 0
            if area_fraction > max_fraction:
                raw_policy = "large_raw_box_seeded_components"
                seed = cv2.bitwise_and(relief, roi)
                seed_dilate = max(1, int(getattr(cfg, "rigid_container_target_raw_text_seed_dilate_px", 3)))
                sk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (seed_dilate * 2 + 1, seed_dilate * 2 + 1))
                seed_support = cv2.dilate((seed > 0).astype(np.uint8) * 255, sk, iterations=1) > 0
                selected_dark = np.zeros_like(dark)
                n, labels, stats, _ = cv2.connectedComponentsWithStats((dark > 0).astype(np.uint8), 8)
                total_components = max(0, int(n - 1))
                max_component_fraction = float(getattr(cfg, "rigid_container_target_raw_text_component_max_fraction", 0.12))
                min_seed_pixels = max(2, int(getattr(cfg, "rigid_container_target_raw_text_component_min_seed_pixels", 3)))
                min_seed_fraction = max(0.0, float(getattr(cfg, "rigid_container_target_raw_text_component_min_seed_fraction", 0.006)))
                rx0, ry0, rw, rh = cv2.boundingRect((roi > 0).astype(np.uint8))
                rw = max(1, int(rw)); rh = max(1, int(rh))
                selected_labels: set[int] = set()
                selected_boxes: list[tuple[int, int, int, int]] = []
                component_limit = max(48, int(round(max(container_area, area) * max_component_fraction)))
                for lab in range(1, n):
                    cx, cy, cw, ch, comp_area = [int(v) for v in stats[lab]]
                    if comp_area < 2 or comp_area > component_limit:
                        continue
                    comp = labels == lab
                    evidence = int(np.count_nonzero(comp & seed_support))
                    if evidence < min_seed_pixels or float(evidence / max(1, comp_area)) < min_seed_fraction:
                        continue
                    span_x = float(cw / rw); span_y = float(ch / rh)
                    if span_x > 0.72 and span_y > 0.38:
                        continue
                    if span_y > 0.82 and span_x > 0.34:
                        continue
                    selected_dark[comp] = 255
                    selected_labels.add(lab)
                    selected_boxes.append((cx, cy, cx + cw, cy + ch))
                    seeded_components += 1

                # High-confidence raw text boxes can contain complete glyphs or
                # punctuation that are spatially separate from every changed-text
                # seed.  On a verified bright Koharu bubble, admit only compact
                # components whose local halo is paper-like and which form a short
                # spatial chain from already seeded text.  This recovers disconnected
                # TARGET Japanese without granting rectangle-level clear authority.
                allow_unseeded = bool(getattr(cfg, "rigid_container_target_raw_text_unseeded_semantic_enabled", True))
                allow_unseeded = bool(
                    allow_unseeded
                    and clear_domain == "koharu_bubble_polygon"
                    and float(hint.get("confidence", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_unseeded_min_confidence", 0.80))
                    and bright_ratio >= float(getattr(cfg, "rigid_container_target_raw_text_unseeded_min_bright_ratio", 0.60))
                    and selected_boxes
                )
                if allow_unseeded:
                    max_span_x = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_max_span_x", 0.58))
                    max_span_y = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_max_span_y", 0.35))
                    halo_px = max(2, int(getattr(cfg, "rigid_container_target_raw_text_unseeded_halo_px", 7)))
                    min_halo_bright = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_min_halo_bright_ratio", 0.72))
                    max_halo_sat = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_max_halo_saturation_median", 45.0))
                    max_gap = max(
                        int(getattr(cfg, "rigid_container_target_raw_text_unseeded_neighbor_gap_min_px", 10)),
                        int(round(min(rw, rh) * float(getattr(cfg, "rigid_container_target_raw_text_unseeded_neighbor_gap_ratio", 0.38)))),
                    )

                    def _box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
                        ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
                        dx = max(0, max(ax0, bx0) - min(ax1, bx1))
                        dy = max(0, max(ay0, by0) - min(ay1, by1))
                        return float((dx * dx + dy * dy) ** 0.5)

                    # Iterate so separated punctuation can be reached through the
                    # preceding character, but cannot jump arbitrarily across art.
                    changed = True
                    while changed:
                        changed = False
                        for lab in range(1, n):
                            if lab in selected_labels:
                                continue
                            cx, cy, cw, ch, comp_area = [int(v) for v in stats[lab]]
                            if comp_area < 3 or comp_area > component_limit:
                                continue
                            span_x = float(cw / rw); span_y = float(ch / rh)
                            if span_x > max_span_x or span_y > max_span_y:
                                continue
                            bbox = (cx, cy, cx + cw, cy + ch)
                            if min((_box_gap(bbox, sbx) for sbx in selected_boxes), default=float("inf")) > max_gap:
                                continue
                            comp = (labels == lab)
                            min_boundary_dist = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_min_boundary_distance_px", 8.0))
                            if semantic_bubble_distance is not None and float(np.min(semantic_bubble_distance[comp])) < min_boundary_dist:
                                continue
                            lx0 = max(0, cx - halo_px); ly0 = max(0, cy - halo_px)
                            lx1 = min(shape[1], cx + cw + halo_px); ly1 = min(shape[0], cy + ch + halo_px)
                            local_comp = comp[ly0:ly1, lx0:lx1].astype(np.uint8)
                            hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_px * 2 + 1, halo_px * 2 + 1))
                            halo = (cv2.dilate(local_comp, hk, iterations=1) > 0) & (local_comp == 0)
                            halo &= (roi[ly0:ly1, lx0:lx1] > 0)
                            halo_pixels = int(np.count_nonzero(halo))
                            if halo_pixels < 8:
                                continue
                            halo_bright = float(np.mean(gray[ly0:ly1, lx0:lx1][halo] >= 220))
                            halo_sat_values = hsv[ly0:ly1, lx0:lx1, 1][halo]
                            halo_sat = float(np.median(halo_sat_values))
                            halo_high_sat = float(np.mean(halo_sat_values > 70))
                            max_halo_high_sat = float(getattr(cfg, "rigid_container_target_raw_text_unseeded_max_halo_high_saturation_ratio", 0.08))
                            if halo_bright < min_halo_bright or halo_sat > max_halo_sat or halo_high_sat > max_halo_high_sat:
                                continue
                            selected_dark[comp] = 255
                            selected_labels.add(lab)
                            selected_boxes.append(bbox)
                            semantic_unseeded += 1
                            semantic_unseeded_pixels += comp_area
                            changed = True

                # High-confidence semantic completion for raw-box edge spill.
                # Koharu raw text boxes can stop a few pixels inside a bold glyph;
                # requiring every disconnected edge fragment to chain from the
                # changed-text seed leaves visible JP slivers (p-194 class).  Under
                # a verified bright/neutral bubble, admit only compact text-like
                # components that intersect the original raw box and remain near
                # already seeded text.  This is TARGET-clear authority only.
                strict_completion = bool(
                    getattr(cfg, "rigid_container_target_raw_text_completion_enabled", True)
                    and clear_domain == "koharu_bubble_polygon"
                    and selected_boxes
                    and float(hint.get("confidence", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_completion_min_confidence", 0.90))
                    and float((semantic_bubble_hint or {}).get("overlap", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_completion_min_bubble_overlap", 0.90))
                    and bright_ratio >= float(getattr(cfg, "rigid_container_target_raw_text_completion_min_bright_ratio", 0.62))
                    and sat_median <= float(getattr(cfg, "rigid_container_target_raw_text_completion_max_saturation_median", 35.0))
                )
                if strict_completion:
                    pad = max(0, int(getattr(cfg, "rigid_container_target_raw_text_completion_padding_px", 10)))
                    padded_rect = np.zeros(shape, np.uint8)
                    px0 = max(0, x0 - pad); py0 = max(0, y0 - pad)
                    px1 = min(shape[1], x1 + pad); py1 = min(shape[0], y1 + pad)
                    padded_rect[py0:py1, px0:px1] = 255
                    completion_roi = cv2.bitwise_and(padded_rect, semantic_bubble_mask)
                    completion_dark = ((gray < gray_thr) & (completion_roi > 0)).astype(np.uint8) * 255
                    cn, clabels, cstats, _ = cv2.connectedComponentsWithStats((completion_dark > 0).astype(np.uint8), 8)
                    crx, cry, crw, crh = cv2.boundingRect((completion_roi > 0).astype(np.uint8))
                    crw = max(1, int(crw)); crh = max(1, int(crh))
                    c_limit = max(48, int(round(max(container_area, int(cv2.countNonZero(completion_roi))) * max_component_fraction)))
                    c_halo_px = max(2, int(getattr(cfg, "rigid_container_target_raw_text_completion_halo_px", 7)))
                    c_min_halo_bright = float(getattr(cfg, "rigid_container_target_raw_text_completion_min_halo_bright_ratio", 0.72))
                    c_max_halo_sat = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_halo_saturation_median", 45.0))
                    c_max_aspect = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_aspect", 14.0))
                    c_min_boundary = float(getattr(cfg, "rigid_container_target_raw_text_completion_min_boundary_distance_px", 4.0))
                    c_gap = max(
                        int(getattr(cfg, "rigid_container_target_raw_text_completion_neighbor_gap_min_px", 14)),
                        int(round(min(crw, crh) * float(getattr(cfg, "rigid_container_target_raw_text_completion_neighbor_gap_ratio", 0.48)))),
                    )
                    for lab in range(1, cn):
                        cx, cy, cw, ch, comp_area = [int(v) for v in cstats[lab]]
                        if comp_area < 2 or comp_area > c_limit or cw < 1 or ch < 1:
                            continue
                        aspect = max(float(cw / max(1, ch)), float(ch / max(1, cw)))
                        if aspect > c_max_aspect:
                            continue
                        span_x = float(cw / crw); span_y = float(ch / crh)
                        if span_x > 0.68 and span_y > 0.34:
                            continue
                        if span_y > 0.72 and span_x > 0.28:
                            continue
                        comp = clabels == lab
                        raw_overlap = int(np.count_nonzero(comp & (raw_rect > 0)))
                        if raw_overlap < max(2, int(round(comp_area * 0.18))):
                            continue
                        bbox = (cx, cy, cx + cw, cy + ch)
                        if min((_box_gap(bbox, sbx) for sbx in selected_boxes), default=float("inf")) > c_gap:
                            continue
                        if semantic_bubble_distance is not None and float(np.min(semantic_bubble_distance[comp])) < c_min_boundary:
                            continue
                        lx0 = max(0, cx - c_halo_px); ly0 = max(0, cy - c_halo_px)
                        lx1 = min(shape[1], cx + cw + c_halo_px); ly1 = min(shape[0], cy + ch + c_halo_px)
                        local_comp = comp[ly0:ly1, lx0:lx1].astype(np.uint8)
                        hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_halo_px * 2 + 1, c_halo_px * 2 + 1))
                        halo = (cv2.dilate(local_comp, hk, iterations=1) > 0) & (local_comp == 0)
                        halo &= (completion_roi[ly0:ly1, lx0:lx1] > 0)
                        if int(np.count_nonzero(halo)) < 8:
                            continue
                        halo_bright = float(np.mean(gray[ly0:ly1, lx0:lx1][halo] >= 220))
                        halo_sat_values = hsv[ly0:ly1, lx0:lx1, 1][halo]
                        halo_sat = float(np.median(halo_sat_values))
                        halo_high_sat = float(np.mean(halo_sat_values > 70))
                        c_max_halo_high_sat = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_halo_high_saturation_ratio", 0.08))
                        if halo_bright < c_min_halo_bright or halo_sat > c_max_halo_sat or halo_high_sat > c_max_halo_high_sat:
                            continue
                        before = int(np.count_nonzero(selected_dark[comp]))
                        selected_dark[comp] = 255
                        if before < comp_area:
                            completion_components += 1
                            completion_pixels += int(comp_area - before)
                    # Final high-confidence raw-box completion: once Koharu has
                    # supplied a strong text raw_box inside a strong semantic bubble,
                    # every compact dark component inside that raw_box is eligible for
                    # TARGET-clear authority when its local halo is neutral/white paper
                    # and it stays away from the bubble boundary.  This intentionally
                    # does *not* require a proximity chain to changed-text seeds: bold
                    # vertical Japanese columns and separated punctuation can be fully
                    # disconnected in the subtractive diff (p-194 class).  The rule is
                    # TARGET-clear-only; it never enlarges SOURCE write authority.
                    allow_full_raw = bool(
                        getattr(cfg, "rigid_container_target_raw_text_full_component_enabled", True)
                        and clear_domain == "koharu_bubble_polygon"
                        and float(hint.get("confidence", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_confidence", 0.90))
                        and float((semantic_bubble_hint or {}).get("overlap", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_bubble_overlap", 0.90))
                        and bright_ratio >= float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_bright_ratio", 0.62))
                        and sat_median <= float(getattr(cfg, "rigid_container_target_raw_text_full_component_max_saturation_median", 35.0))
                    )
                    if allow_full_raw:
                        raw_overlap_fraction = float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_raw_overlap_fraction", 0.55))
                        full_min_boundary = float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_boundary_distance_px", 8.0))
                        full_min_halo_bright = float(getattr(cfg, "rigid_container_target_raw_text_full_component_min_halo_bright_ratio", 0.68))
                        full_max_halo_sat = float(getattr(cfg, "rigid_container_target_raw_text_full_component_max_halo_saturation_median", 35.0))
                        full_max_halo_high_sat = float(getattr(cfg, "rigid_container_target_raw_text_full_component_max_halo_high_saturation_ratio", 0.08))
                        for lab in range(1, cn):
                            cx, cy, cw, ch, comp_area = [int(v) for v in cstats[lab]]
                            if comp_area < 2 or comp_area > c_limit or cw < 1 or ch < 1:
                                continue
                            comp = clabels == lab
                            if int(np.count_nonzero(selected_dark[comp])) >= int(round(comp_area * 0.80)):
                                continue
                            raw_overlap = int(np.count_nonzero(comp & (raw_rect > 0)))
                            if raw_overlap < max(2, int(round(comp_area * raw_overlap_fraction))):
                                continue
                            aspect = max(float(cw / max(1, ch)), float(ch / max(1, cw)))
                            if aspect > c_max_aspect:
                                continue
                            span_x = float(cw / crw); span_y = float(ch / crh)
                            if span_x > 0.68 and span_y > 0.34:
                                continue
                            if span_y > 0.72 and span_x > 0.28:
                                continue
                            if semantic_bubble_distance is not None and float(np.min(semantic_bubble_distance[comp])) < full_min_boundary:
                                continue
                            lx0 = max(0, cx - c_halo_px); ly0 = max(0, cy - c_halo_px)
                            lx1 = min(shape[1], cx + cw + c_halo_px); ly1 = min(shape[0], cy + ch + c_halo_px)
                            local_comp = comp[ly0:ly1, lx0:lx1].astype(np.uint8)
                            hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (c_halo_px * 2 + 1, c_halo_px * 2 + 1))
                            halo = (cv2.dilate(local_comp, hk, iterations=1) > 0) & (local_comp == 0)
                            halo &= (completion_roi[ly0:ly1, lx0:lx1] > 0)
                            halo_pixels = int(np.count_nonzero(halo))
                            if halo_pixels < 8:
                                continue
                            halo_gray = gray[ly0:ly1, lx0:lx1][halo]
                            halo_sat_values = hsv[ly0:ly1, lx0:lx1, 1][halo]
                            halo_bright = float(np.mean(halo_gray >= 220))
                            halo_sat = float(np.median(halo_sat_values))
                            halo_high_sat = float(np.mean(halo_sat_values > 70))
                            if halo_bright < full_min_halo_bright or halo_sat > full_max_halo_sat or halo_high_sat > full_max_halo_high_sat:
                                continue
                            before = int(np.count_nonzero(selected_dark[comp]))
                            selected_dark[comp] = 255
                            added = max(0, int(comp_area - before))
                            if added > 0:
                                semantic_full_components += 1
                                semantic_full_pixels += added
                        if semantic_full_pixels > 0:
                            raw_domain_union = np.maximum(raw_domain_union, completion_roi)

                    if completion_pixels > 0:
                        raw_domain_union = np.maximum(raw_domain_union, completion_roi)
                if cv2.countNonZero(selected_dark) <= 0:
                    continue

            # v2.3.89: raw-box edge spill completion.  A Koharu text raw_box can
            # terminate *inside* the last punctuation/glyph component.  In that
            # case the existing clear mask contains only the few pixels that lie
            # inside raw_box and leaves the rest of the TARGET glyph visible.
            # Follow only the same compact dark component across the raw-box edge,
            # then allow a very short chain of adjacent punctuation components.
            # This authority is TARGET-clear-only and stays inside a verified
            # semantic bubble; SOURCE write geometry is untouched.
            allow_edge_spill = bool(
                getattr(cfg, "rigid_container_target_raw_text_edge_spill_enabled", True)
                and semantic_bubble_mask is not None
                and cv2.countNonZero(selected_dark) > 0
                and float(hint.get("confidence", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_edge_spill_min_confidence", 0.90))
                and float((semantic_bubble_hint or {}).get("overlap", 0.0) or 0.0) >= float(getattr(cfg, "rigid_container_target_raw_text_edge_spill_min_bubble_overlap", 0.90))
                and bright_ratio >= float(getattr(cfg, "rigid_container_target_raw_text_completion_min_bright_ratio", 0.62))
                and sat_median <= float(getattr(cfg, "rigid_container_target_raw_text_completion_max_saturation_median", 35.0))
            )
            if allow_edge_spill:
                raw_w = max(1, x1 - x0); raw_h = max(1, y1 - y0)
                edge_spill_padding_px = int(np.clip(
                    round(min(raw_w, raw_h) * float(getattr(cfg, "rigid_container_target_raw_text_edge_spill_padding_ratio", 0.45))),
                    int(getattr(cfg, "rigid_container_target_raw_text_edge_spill_padding_min_px", 28)),
                    int(getattr(cfg, "rigid_container_target_raw_text_edge_spill_padding_max_px", 112)),
                ))
                ex0 = max(0, x0 - edge_spill_padding_px); ey0 = max(0, y0 - edge_spill_padding_px)
                ex1 = min(shape[1], x1 + edge_spill_padding_px); ey1 = min(shape[0], y1 + edge_spill_padding_px)
                spill_rect = np.zeros(shape, np.uint8)
                spill_rect[ey0:ey1, ex0:ex1] = 255
                spill_roi = cv2.bitwise_and(spill_rect, semantic_bubble_mask)
                spill_dark = ((gray < gray_thr) & (spill_roi > 0)).astype(np.uint8) * 255
                sn, slabels, sstats, _ = cv2.connectedComponentsWithStats((spill_dark > 0).astype(np.uint8), 8)
                spill_area = max(1, int(cv2.countNonZero(spill_roi)))
                spill_component_limit = max(
                    48,
                    int(round(spill_area * float(getattr(cfg, "rigid_container_target_raw_text_component_max_fraction", 0.12)))),
                )
                spill_seed_dilate = max(1, int(getattr(cfg, "rigid_container_target_raw_text_edge_spill_seed_dilate_px", 3)))
                spill_seed = cv2.dilate(
                    (selected_dark > 0).astype(np.uint8) * 255,
                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spill_seed_dilate * 2 + 1, spill_seed_dilate * 2 + 1)),
                    iterations=1,
                ) > 0
                selected_spill_labels: set[int] = set()
                selected_spill_boxes: list[tuple[int, int, int, int]] = []
                # Existing selected text boxes are anchors for punctuation that is
                # split into a bar/dot or dakuten-like disconnected component.
                an, alabels, astats, _ = cv2.connectedComponentsWithStats((selected_dark > 0).astype(np.uint8), 8)
                for alab in range(1, an):
                    ax, ay, aw, ah, aa = [int(v) for v in astats[alab]]
                    if aa >= 2:
                        selected_spill_boxes.append((ax, ay, ax + aw, ay + ah))

                def _edge_box_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
                    ax0, ay0, ax1, ay1 = a; bx0, by0, bx1, by1 = b
                    dx = max(0, max(ax0, bx0) - min(ax1, bx1))
                    dy = max(0, max(ay0, by0) - min(ay1, by1))
                    return float((dx * dx + dy * dy) ** 0.5)

                spill_halo_px = max(2, int(getattr(cfg, "rigid_container_target_raw_text_completion_halo_px", 7)))
                spill_min_halo_bright = float(getattr(cfg, "rigid_container_target_raw_text_completion_min_halo_bright_ratio", 0.72))
                spill_max_halo_sat = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_halo_saturation_median", 45.0))
                spill_max_halo_high_sat = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_halo_high_saturation_ratio", 0.08))
                spill_max_aspect = float(getattr(cfg, "rigid_container_target_raw_text_completion_max_aspect", 14.0))
                spill_min_boundary = float(getattr(cfg, "rigid_container_target_raw_text_edge_spill_min_boundary_distance_px", 8.0))
                spill_neighbor_gap = max(1, int(getattr(cfg, "rigid_container_target_raw_text_edge_spill_neighbor_gap_px", 18)))

                def _spill_component_safe(lab: int) -> tuple[bool, tuple[int, int, int, int] | None, np.ndarray | None]:
                    cx, cy, cw, ch, comp_area = [int(v) for v in sstats[lab]]
                    if comp_area < 2 or comp_area > spill_component_limit or cw < 1 or ch < 1:
                        return False, None, None
                    aspect = max(float(cw / max(1, ch)), float(ch / max(1, cw)))
                    if aspect > spill_max_aspect:
                        return False, None, None
                    span_x = float(cw / max(1, ex1 - ex0)); span_y = float(ch / max(1, ey1 - ey0))
                    if span_x > 0.68 and span_y > 0.34:
                        return False, None, None
                    if span_y > 0.72 and span_x > 0.28:
                        return False, None, None
                    comp = slabels == lab
                    # Only an actual spill may extend authority: at least some of
                    # the component must lie outside the detector raw_box.
                    outside = int(np.count_nonzero(comp & (raw_rect == 0)))
                    if outside <= 0:
                        return False, None, None
                    if semantic_bubble_distance is not None and float(np.min(semantic_bubble_distance[comp])) < spill_min_boundary:
                        return False, None, None
                    lx0 = max(0, cx - spill_halo_px); ly0 = max(0, cy - spill_halo_px)
                    lx1 = min(shape[1], cx + cw + spill_halo_px); ly1 = min(shape[0], cy + ch + spill_halo_px)
                    local_comp = comp[ly0:ly1, lx0:lx1].astype(np.uint8)
                    hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (spill_halo_px * 2 + 1, spill_halo_px * 2 + 1))
                    halo = (cv2.dilate(local_comp, hk, iterations=1) > 0) & (local_comp == 0)
                    halo &= (spill_roi[ly0:ly1, lx0:lx1] > 0)
                    if int(np.count_nonzero(halo)) < 8:
                        return False, None, None
                    halo_gray = gray[ly0:ly1, lx0:lx1][halo]
                    halo_sat_values = hsv[ly0:ly1, lx0:lx1, 1][halo]
                    halo_bright = float(np.mean(halo_gray >= 220))
                    halo_sat = float(np.median(halo_sat_values))
                    halo_high_sat = float(np.mean(halo_sat_values > 70))
                    if halo_bright < spill_min_halo_bright or halo_sat > spill_max_halo_sat or halo_high_sat > spill_max_halo_high_sat:
                        return False, None, None
                    return True, (cx, cy, cx + cw, cy + ch), comp

                # Pass 1: recover complete components that are already clipped by
                # raw_box.  This is the strongest and safest p-194 signal.
                for lab in range(1, sn):
                    comp = slabels == lab
                    if int(np.count_nonzero(comp & spill_seed)) < 2:
                        continue
                    ok, bbox, safe_comp = _spill_component_safe(lab)
                    if not ok or bbox is None or safe_comp is None:
                        continue
                    before = int(np.count_nonzero(selected_dark[safe_comp]))
                    selected_dark[safe_comp] = 255
                    added = max(0, int(np.count_nonzero(safe_comp) - before))
                    if added > 0:
                        selected_spill_labels.add(lab)
                        selected_spill_boxes.append(bbox)
                        edge_spill_components += 1
                        edge_spill_seeded_components += 1
                        edge_spill_pixels += added

                # Pass 2: short punctuation chain (e.g. the detached dot of '!').
                # Iterate only inside the bounded spill ROI and only over neutral
                # compact components, preventing jumps to nearby coloured artwork.
                changed_spill = True
                while changed_spill:
                    changed_spill = False
                    for lab in range(1, sn):
                        if lab in selected_spill_labels:
                            continue
                        ok, bbox, safe_comp = _spill_component_safe(lab)
                        if not ok or bbox is None or safe_comp is None:
                            continue
                        if int(np.count_nonzero(selected_dark[safe_comp])) >= int(round(np.count_nonzero(safe_comp) * 0.80)):
                            continue
                        if min((_edge_box_gap(bbox, anchor) for anchor in selected_spill_boxes), default=float("inf")) > spill_neighbor_gap:
                            continue
                        before = int(np.count_nonzero(selected_dark[safe_comp]))
                        selected_dark[safe_comp] = 255
                        added = max(0, int(np.count_nonzero(safe_comp) - before))
                        if added <= 0:
                            continue
                        selected_spill_labels.add(lab)
                        selected_spill_boxes.append(bbox)
                        edge_spill_components += 1
                        edge_spill_neighbor_components += 1
                        edge_spill_pixels += added
                        changed_spill = True
                if edge_spill_pixels > 0:
                    raw_domain_union = np.maximum(raw_domain_union, spill_roi)
            raw_relief = np.maximum(raw_relief, selected_dark)
            raw_domain_union = np.maximum(raw_domain_union, roi)
            accepted.append({
                "raw_box": [x0, y0, x1, y1],
                "confidence": float(hint.get("confidence", 0.0) or 0.0),
                "overlap": float(hint.get("overlap", 0.0) or 0.0),
                "bright_ratio": round(bright_ratio, 4),
                "saturation_median": round(sat_median, 3),
                "area_fraction": round(area_fraction, 4),
                "clear_domain": clear_domain,
                "semantic_bubble_overlap": round(float((semantic_bubble_hint or {}).get("overlap", 0.0) or 0.0), 4),
                "policy": raw_policy,
                "dark_pixels": int(cv2.countNonZero(dark)),
                "selected_dark_pixels": int(cv2.countNonZero(selected_dark)),
                "seeded_components": int(seeded_components),
                "semantic_unseeded_components": int(semantic_unseeded),
                "semantic_unseeded_pixels": int(semantic_unseeded_pixels),
                "semantic_completion_components": int(completion_components),
                "semantic_completion_pixels": int(completion_pixels),
                "semantic_full_components": int(semantic_full_components),
                "semantic_full_pixels": int(semantic_full_pixels),
                "semantic_edge_spill_components": int(edge_spill_components),
                "semantic_edge_spill_seeded_components": int(edge_spill_seeded_components),
                "semantic_edge_spill_neighbor_components": int(edge_spill_neighbor_components),
                "semantic_edge_spill_pixels": int(edge_spill_pixels),
                "semantic_edge_spill_padding_px": int(edge_spill_padding_px),
                "total_components": int(total_components),
            })
        raw_dilate = max(0, int(getattr(cfg, "rigid_container_target_raw_text_dilate_px", 2)))
        if raw_dilate > 0 and cv2.countNonZero(raw_relief) > 0:
            rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (raw_dilate * 2 + 1, raw_dilate * 2 + 1))
            raw_relief = cv2.dilate(raw_relief, rk, iterations=1)
            if cv2.countNonZero(raw_domain_union) > 0:
                raw_relief = cv2.bitwise_and(raw_relief, raw_domain_union)
            else:
                raw_relief = cv2.bitwise_and(raw_relief, target_mask)
        raw_diag.update({"accepted_boxes": len(accepted), "boxes": accepted, "pixels": int(cv2.countNonZero(raw_relief)), "dilate_px": int(raw_dilate)})
        relief = np.maximum(relief, raw_relief)
    elif not hints:
        raw_diag["reason"] = "no_koharu_text_hints"
    diag.update({
        "reason": "ok" if cv2.countNonZero(relief) > 0 else "no_changed_target_text",
        "authority": "changed_text_seeded_full_target_text_components",
        "pixels": int(cv2.countNonZero(relief)),
        "unique_seed_pixels": int(cv2.countNonZero(unique_seed)),
        "full_target_text_pixels": int(cv2.countNonZero(full_target_text)),
        "selected_components": int(selected_components),
        "total_components": int(total_components),
        "dilate_px": int(dilate_px),
        "text_diag": text_diag,
        "raw_text_clear": raw_diag,
    })
    return relief, diag


def _rigid_source_semantic_text_extension(
    source: np.ndarray,
    target_shape: tuple[int, int],
    cfg: MaskReplaceConfig,
    *,
    source_bubble: BubbleInstance | None,
    target_bubble: BubbleInstance | None,
    scale: float,
    source_pair_origin: tuple[int, int],
    target_pair_origin: tuple[float, float],
    paper: float,
    alpha_floor: float,
    source_gray: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Extrapolate verified SOURCE text pixels clipped by a narrow paired lane.

    This is intentionally SOURCE-text-only.  Koharu raw text geometry chooses the
    candidate zone, a Koharu bubble polygon supplies a semantic container, and
    compact white-paper components away from the bubble boundary become glyph
    authority.  The already-selected rigid scale/translation is merely extended to
    those glyph coordinates; no SOURCE paper/artwork is copied.
    """
    th, tw = [int(v) for v in target_shape]
    empty_alpha = np.zeros((th, tw), np.float32)
    empty_mask = np.zeros((th, tw), np.uint8)
    diag: dict[str, object] = {
        "enabled": bool(getattr(cfg, "rigid_container_source_raw_text_write_enabled", True)),
        "applied": False,
        "selected_components": 0,
        "selected_pixels_source": 0,
        "write_pixels_target": 0,
    }
    if not bool(diag["enabled"]) or source_bubble is None or target_bubble is None:
        diag["reason"] = "disabled_or_missing_pair"
        return empty_alpha, empty_mask, diag
    smeta = dict(getattr(source_bubble, "meta", {}) or {})
    tmeta = dict(getattr(target_bubble, "meta", {}) or {})
    text_hints = list(smeta.get("koharu_layout_text_hints") or [])
    source_bubble_hints = list(smeta.get("koharu_layout_bubble_hints") or [])
    target_bubble_hints = list(tmeta.get("koharu_layout_bubble_hints") or [])
    min_conf = float(getattr(cfg, "rigid_container_source_raw_text_write_min_confidence", 0.80))
    min_overlap = float(getattr(cfg, "rigid_container_source_raw_text_write_min_overlap", 0.20))
    text_hint = next((h for h in text_hints if float(h.get("confidence", 0.0) or 0.0) >= min_conf and float(h.get("overlap", 0.0) or 0.0) >= min_overlap and len(h.get("raw_box") or []) == 4), None)
    sb_hint = next((h for h in source_bubble_hints if float(h.get("confidence", 0.0) or 0.0) >= 0.50 and float(h.get("overlap", 0.0) or 0.0) >= min_overlap and len(h.get("polygon") or []) >= 3), None)
    tb_hint = next((h for h in target_bubble_hints if float(h.get("confidence", 0.0) or 0.0) >= 0.50 and float(h.get("overlap", 0.0) or 0.0) >= min_overlap and len(h.get("polygon") or []) >= 3), None)
    # SOURCE Koharu can miss a burst-balloon polygon even when its high-confidence
    # raw text box is excellent (p-194 class).  Requiring a SOURCE bubble hint here
    # clips punctuation that extends beyond the paired diff lane.  The raw text box
    # itself is already a semantic *text* authority, so allow it to define the SOURCE
    # domain when the bubble polygon is absent.  TARGET still requires a Koharu bubble
    # polygon, which keeps the projected write inside the colour edition's container.
    if text_hint is None or tb_hint is None:
        diag["reason"] = "missing_high_confidence_semantic_geometry"
        return empty_alpha, empty_mask, diag

    sh, sw = source.shape[:2]
    target_domain = (rasterize_polygon(tb_hint["polygon"], (th, tw)) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(target_domain) <= 0:
        diag["reason"] = "empty_target_semantic_domain"
        return empty_alpha, empty_mask, diag
    x0, y0, x1, y1 = [int(round(float(v))) for v in text_hint["raw_box"]]
    x0 = max(0, min(sw, x0)); x1 = max(0, min(sw, x1))
    y0 = max(0, min(sh, y0)); y1 = max(0, min(sh, y1))
    if x1 <= x0 or y1 <= y0:
        diag["reason"] = "invalid_raw_box"
        return empty_alpha, empty_mask, diag
    raw_roi = np.zeros((sh, sw), np.uint8); raw_roi[y0:y1, x0:x1] = 255
    if sb_hint is not None:
        source_domain = (rasterize_polygon(sb_hint["polygon"], (sh, sw)) > 0).astype(np.uint8) * 255
        if cv2.countNonZero(source_domain) <= 0:
            diag["reason"] = "empty_source_semantic_domain"
            return empty_alpha, empty_mask, diag
        raw_roi = cv2.bitwise_and(raw_roi, source_domain)
        source_domain_authority = "koharu_source_bubble_polygon"
    else:
        # Safe fallback: never broaden beyond the high-confidence SOURCE raw text box.
        # This deliberately excludes nearby illustration ink (the p-194 chibi is
        # outside the raw text box) while recovering bottom punctuation clipped by
        # the narrower paired lane.
        source_domain = raw_roi.copy()
        source_domain_authority = "koharu_source_raw_text_box"
    pix = raw_roi > 0
    if not np.any(pix):
        diag["reason"] = "raw_box_outside_semantic_bubble"
        return empty_alpha, empty_mask, diag
    gray_u8 = np.clip(source_gray, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    bright_ratio = float(np.mean(gray_u8[pix] >= 220))
    sat_median = float(np.median(hsv[..., 1][pix]))
    if bright_ratio < float(getattr(cfg, "rigid_container_source_raw_text_write_min_bright_ratio", 0.60)) or sat_median > float(getattr(cfg, "rigid_container_source_raw_text_write_max_saturation_median", 80.0)):
        diag.update({"reason": "not_white_low_saturation_text_zone", "bright_ratio": bright_ratio, "saturation_median": sat_median})
        return empty_alpha, empty_mask, diag

    gray_thr = int(getattr(cfg, "rigid_container_source_raw_text_write_gray_threshold", 228))
    dark = ((gray_u8 < gray_thr) & pix).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    semantic = np.zeros((sh, sw), np.uint8)
    bubble_dist = cv2.distanceTransform((source_domain > 0).astype(np.uint8), cv2.DIST_L2, 5)
    rw, rh = max(1, x1 - x0), max(1, y1 - y0)
    max_component = max(48, int(round(max(1, int(np.count_nonzero(pix))) * float(getattr(cfg, "rigid_container_source_raw_text_write_max_component_fraction", 0.12)))))
    max_span_x = float(getattr(cfg, "rigid_container_source_raw_text_write_max_span_x", 0.65))
    max_span_y = float(getattr(cfg, "rigid_container_source_raw_text_write_max_span_y", 0.35))
    halo_px = max(2, int(getattr(cfg, "rigid_container_source_raw_text_write_halo_px", 7)))
    min_halo_bright = float(getattr(cfg, "rigid_container_source_raw_text_write_min_halo_bright_ratio", 0.72))
    max_halo_sat = float(getattr(cfg, "rigid_container_source_raw_text_write_max_halo_saturation_median", 45.0))
    min_boundary_dist = float(getattr(cfg, "rigid_container_source_raw_text_write_min_boundary_distance_px", 8.0))
    selected = 0
    accepted_boxes: list[tuple[int, int, int, int]] = []
    boundary_candidates: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for lab in range(1, n):
        cx, cy, cw, ch, area = [int(v) for v in stats[lab]]
        if area < 3 or area > max_component:
            continue
        if float(cw / rw) > max_span_x or float(ch / rh) > max_span_y:
            continue
        comp = labels == lab
        lx0 = max(0, cx - halo_px); ly0 = max(0, cy - halo_px)
        lx1 = min(sw, cx + cw + halo_px); ly1 = min(sh, cy + ch + halo_px)
        local_comp = comp[ly0:ly1, lx0:lx1].astype(np.uint8)
        hk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_px * 2 + 1, halo_px * 2 + 1))
        halo = (cv2.dilate(local_comp, hk, iterations=1) > 0) & (local_comp == 0)
        halo &= (raw_roi[ly0:ly1, lx0:lx1] > 0)
        if int(np.count_nonzero(halo)) < 8:
            continue
        halo_bright = float(np.mean(gray_u8[ly0:ly1, lx0:lx1][halo] >= 220))
        halo_sat = float(np.median(hsv[ly0:ly1, lx0:lx1, 1][halo]))
        if halo_bright < min_halo_bright or halo_sat > max_halo_sat:
            continue
        boundary_dist = float(np.min(bubble_dist[comp]))
        if boundary_dist < min_boundary_dist:
            # Do not immediately admit edge-touching dark components: burst rays
            # and panel lines live there too.  Keep only text-like compact candidates
            # for a second pass against already accepted text-column geometry.
            boundary_candidates.append((lab, cx, cy, cw, ch, comp))
            continue
        semantic[comp] = 255
        accepted_boxes.append((cx, cy, cw, ch))
        selected += 1

    edge_recovered = 0
    if (
        bool(getattr(cfg, "rigid_container_source_raw_text_edge_recovery_enabled", True))
        and accepted_boxes and boundary_candidates
    ):
        max_gap = max(12, int(round(rh * float(getattr(cfg, "rigid_container_source_raw_text_edge_recovery_max_gap_ratio", 0.12)))))
        min_overlap = float(getattr(cfg, "rigid_container_source_raw_text_edge_recovery_min_axis_overlap", 0.55))
        max_wr = float(getattr(cfg, "rigid_container_source_raw_text_edge_recovery_max_width_ratio", 0.52))
        max_hr = float(getattr(cfg, "rigid_container_source_raw_text_edge_recovery_max_height_ratio", 0.12))
        for _lab, cx, cy, cw, ch, comp in boundary_candidates:
            if float(cw / rw) > max_wr or float(ch / rh) > max_hr:
                continue
            safe_neighbor = False
            for ax, ay, aw, ah in accepted_boxes:
                overlap_x = max(0, min(cx + cw, ax + aw) - max(cx, ax))
                overlap_ratio = float(overlap_x / max(1, min(cw, aw)))
                vertical_gap = max(0, cy - (ay + ah), ay - (cy + ch))
                if overlap_ratio >= min_overlap and vertical_gap <= max_gap:
                    safe_neighbor = True
                    break
            if not safe_neighbor:
                continue
            semantic[comp] = 255
            accepted_boxes.append((cx, cy, cw, ch))
            selected += 1
            edge_recovered += 1
    if cv2.countNonZero(semantic) <= 0:
        diag["reason"] = "no_safe_source_text_components"
        return empty_alpha, empty_mask, diag

    # Preserve SOURCE antialiasing, but only under the semantic component mask.
    sem_alpha_src = np.clip((float(paper) - source_gray.astype(np.float32)) / max(1.0, float(paper)), 0.0, 1.0)
    sem_alpha_src[sem_alpha_src < float(alpha_floor)] = 0.0
    sem_alpha_src *= (semantic.astype(np.float32) / 255.0)
    ys, xs = np.where(semantic > 0)
    ex0, ey0, ex1, ey1 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    crop = sem_alpha_src[ey0:ey1, ex0:ex1]
    nw = max(1, int(round((ex1 - ex0) * float(scale))))
    nh = max(1, int(round((ey1 - ey0) * float(scale))))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
    warped = cv2.resize(crop, (nw, nh), interpolation=interp)
    sx0, sy0 = [int(v) for v in source_pair_origin]
    px, py = [float(v) for v in target_pair_origin]
    qx = int(round(px + (ex0 - sx0) * float(scale)))
    qy = int(round(py + (ey0 - sy0) * float(scale)))
    xa, ya = max(0, qx), max(0, qy); xb, yb = min(tw, qx + nw), min(th, qy + nh)
    if xb <= xa or yb <= ya:
        diag["reason"] = "semantic_text_maps_outside_target"
        return empty_alpha, empty_mask, diag
    local = warped[ya - qy:yb - qy, xa - qx:xb - qx].copy()
    local *= (target_domain[ya:yb, xa:xb].astype(np.float32) / 255.0)
    out = np.zeros((th, tw), np.float32)
    out[ya:yb, xa:xb] = local
    write_mask = (out >= max(0.08, float(alpha_floor))).astype(np.uint8) * 255
    diag.update({
        "applied": bool(cv2.countNonZero(write_mask) > 0),
        "reason": "semantic_source_text_extrapolated" if cv2.countNonZero(write_mask) > 0 else "no_target_write_pixels",
        "authority": "koharu_source_raw_text_components_inside_semantic_domain",
        "source_domain_authority": source_domain_authority,
        "source_bubble_hint_present": bool(sb_hint is not None),
        "bright_ratio": round(bright_ratio, 4),
        "saturation_median": round(sat_median, 3),
        "selected_components": int(selected),
        "edge_recovered_components": int(edge_recovered),
        "selected_pixels_source": int(cv2.countNonZero(semantic)),
        "write_pixels_target": int(cv2.countNonZero(write_mask)),
        "source_raw_box": [x0, y0, x1, y1],
        "target_origin": [qx, qy],
        "scale": float(scale),
        "min_boundary_distance_px": float(min_boundary_dist),
    })
    return out, write_mask, diag


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
    source_structure_mask: np.ndarray | None = None,
    source_bubble: BubbleInstance | None = None,
    target_bubble: BubbleInstance | None = None,
) -> tuple[np.ndarray, np.ndarray, float, float, float, float, float, dict[str, object], np.ndarray] | None:
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
    # Preserve the admission-time SOURCE-only changed-text authority at render
    # time. This filters shared illustration ink while retaining antialiased
    # translated glyph raster.
    changed_gate, changed_gate_diag = _rigid_changed_text_write_gate(
        source, target_reference, source_mask, target_mask, cfg
    )
    if isinstance(changed_gate, np.ndarray) and changed_gate.shape == alpha.shape:
        raw_ink = alpha >= max(0.08, floor)
        raw_n = int(np.count_nonzero(raw_ink))
        kept_n = int(np.count_nonzero(raw_ink & (changed_gate > 0)))
        retained = float(kept_n / max(1, raw_n))
        min_retained = float(getattr(cfg, "rigid_container_changed_text_gate_min_retained_ratio", 0.65))
        changed_gate_diag["raw_ink_pixels"] = int(raw_n)
        changed_gate_diag["kept_ink_pixels"] = int(kept_n)
        changed_gate_diag["retained_ratio"] = float(retained)
        changed_gate_diag["min_retained_ratio"] = float(min_retained)
        if kept_n >= int(getattr(cfg, "content_completeness_min_ink_pixels", 18)) and retained >= min_retained:
            alpha[changed_gate == 0] = 0.0
            changed_gate_diag["applied"] = True
            changed_gate_diag["reason"] = "applied_changed_text_authority"
        else:
            changed_gate_diag["reason"] = "retention_too_low_fail_open"
    if isinstance(clarity_diag, dict):
        clarity_diag["changed_text_write_gate"] = changed_gate_diag
    # Preserve an audit authority before any SOURCE structure suppression. This is
    # not rendered directly; it exists so a boundary guard can never silently
    # redefine "complete SOURCE text" after it has already deleted pixels.
    raw_alpha_for_audit = alpha.copy()
    structure_mask = source_mask if source_structure_mask is None else source_structure_mask
    source_structure_guard, source_structure_diag = _mask_replace_source_structure_guard(source, structure_mask, cfg)
    local_structure_guard = source_structure_guard[sy0:sy1, sx0:sx1] > 0
    if np.any(local_structure_guard):
        alpha[local_structure_guard] = 0.0
    # Audit the original SOURCE lettering authority itself, including glyphs
    # close to a bubble/burst boundary.  The former eroded-core denominator could
    # hide exactly the failure we need to detect by eroding away clipped letters.
    audit_thr = max(0.08, floor)
    lettering_audit = target_text_mask_in_container(source, structure_mask)[sy0:sy1, sx0:sx1] > 0
    raw_lettering_ink = (raw_alpha_for_audit >= audit_thr) & lettering_audit
    guarded_lettering_ink = (alpha >= audit_thr) & lettering_audit
    raw_core_pixels = int(np.count_nonzero(raw_lettering_ink))
    guarded_core_pixels = int(np.count_nonzero(guarded_lettering_ink))
    # If the conservative selector has no evidence (rare low-contrast source),
    # fall back to all raw alpha rather than silently claiming perfect coverage.
    if raw_core_pixels <= 0:
        raw_lettering_ink = raw_alpha_for_audit >= audit_thr
        guarded_lettering_ink = alpha >= audit_thr
        raw_core_pixels = int(np.count_nonzero(raw_lettering_ink))
        guarded_core_pixels = int(np.count_nonzero(guarded_lettering_ink))
    source_structure_retention = float(guarded_core_pixels / max(1, raw_core_pixels)) if raw_core_pixels > 0 else 1.0
    source_structure_diag["audit_authority"] = "raw_source_lettering_support"
    source_structure_diag["raw_core_ink_pixels"] = int(raw_core_pixels)
    source_structure_diag["guarded_core_ink_pixels"] = int(guarded_core_pixels)
    source_structure_diag["source_structure_retention"] = float(source_structure_retention)
    if isinstance(clarity_diag, dict):
        clarity_diag["source_structure_guard"] = source_structure_diag
        clarity_diag["source_structure_retention"] = float(source_structure_retention)
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
    semantic_alpha, semantic_write_extension, semantic_write_diag = _rigid_source_semantic_text_extension(
        source, target_shape, cfg,
        source_bubble=source_bubble, target_bubble=target_bubble, scale=scale,
        source_pair_origin=(sx0, sy0), target_pair_origin=(px, py),
        paper=paper, alpha_floor=floor, source_gray=gray,
    )
    if cv2.countNonZero(semantic_write_extension) > 0:
        full_alpha = np.maximum(full_alpha, semantic_alpha)
    if isinstance(clarity_diag, dict):
        clarity_diag["source_semantic_text_extension"] = semantic_write_diag
    ink_mask = (full_alpha >= max(0.08, floor)).astype(np.uint8) * 255
    # dx/dy are reported relative to the pure centroid placement at the selected scale.
    base_x = tcx - rel_cx * scale; base_y = tcy - rel_cy * scale
    return full_alpha, ink_mask, scale, float(px - base_x), float(py - base_y), float(ink_cov), float(mask_cov), clarity_diag, semantic_write_extension




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




def _mask_replace_spiky_container_profile(region_mask: np.ndarray) -> dict[str, float | int | bool | str]:
    """Classify burst/spiky container geometry for Precise Mask only.

    A burst balloon has a long, highly non-convex boundary.  Treating it like a
    smooth white speech balloon and blanking the whole interior can erase TARGET
    spike strokes that enter the nominal white region.  This profile is private
    to Mask Replace so changes here cannot alter Direct or Hybrid.
    """
    mask = (np.asarray(region_mask) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "spiky": False, "reason": "empty_mask", "contour_points": 0,
            "solidity": 1.0, "perimeter_ratio": 1.0, "compactness": 1.0,
        }
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    hull_perimeter = float(cv2.arcLength(hull, True))
    solidity = float(area / max(1.0, hull_area))
    perimeter_ratio = float(perimeter / max(1.0, hull_perimeter))
    compactness = float((4.0 * np.pi * area) / max(1.0, perimeter * perimeter))
    contour_points = int(len(contour))
    # Real regression page: ordinary speech balloons cluster around
    # solidity~0.95-0.99 / perimeter_ratio~1.05. Burst balloons are much more
    # concave and have a much longer perimeter. Require all three signals to
    # avoid misclassifying an ordinary speech-tail balloon.
    spiky = bool(
        contour_points >= 28
        and compactness <= 0.45
        and (
            solidity <= 0.88
            or (perimeter_ratio >= 1.55 and solidity <= 0.94)
        )
    )
    return {
        "spiky": spiky,
        "reason": "spiky_burst_geometry" if spiky else "ordinary_container_geometry",
        "contour_points": contour_points,
        "solidity": solidity,
        "perimeter_ratio": perimeter_ratio,
        "compactness": compactness,
        "area": area,
    }


def _mask_replace_spiky_structure_guard(
    target_reference: np.ndarray,
    region_mask: np.ndarray,
    *,
    band_px: int = 7,
) -> np.ndarray:
    """Return TARGET dark structural strokes near a spiky bubble boundary.

    This guard is deliberately derived from TARGET only.  It protects burst
    spikes/outline antialiasing while leaving central Japanese glyphs removable.
    """
    mask = (np.asarray(region_mask) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) <= 0:
        return np.zeros_like(mask)
    boundary = cv2.morphologyEx(
        mask,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    radius = max(2, int(band_px))
    band = cv2.dilate(
        boundary,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
        iterations=1,
    ) > 0
    gray = cv2.cvtColor(target_reference, cv2.COLOR_BGR2GRAY) if target_reference.ndim == 3 else target_reference.astype(np.uint8)
    dark = gray <= 215
    guard = (band & dark).astype(np.uint8) * 255
    # Preserve one antialias fringe around the actual black stroke, but do not
    # turn the whole geometric band into a protected region.
    if cv2.countNonZero(guard) > 0:
        guard = cv2.dilate(
            guard,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        guard = cv2.bitwise_and(guard, cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1))
    return guard


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
        placed = _rigid_source_raster(
            source, target_reference, sm, tm, shape, base_scale, placement_cfg,
            source_gray=source_gray_f32, source_structure_mask=sm_raw,
            source_bubble=sb, target_bubble=tb,
        )
        if placed is None:
            continue
        alpha, source_ink_mask, scale, dx, dy, ink_cov, mask_cov, white_clarity_diag, source_semantic_write_extension = placed
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
        spiky_profile = _mask_replace_spiky_container_profile(tm_raw)
        spiky_structure_guard = (
            _mask_replace_spiky_structure_guard(target_reference, tm_raw)
            if bool(spiky_profile.get("spiky", False)) else np.zeros(shape, np.uint8)
        )
        target_structure_guard, target_structure_diag = _mask_replace_target_structure_guard(
            target_reference, tm_raw, cfg
        )
        immutable_band, immutable_band_diag = _mask_replace_target_immutable_boundary_band(
            tm_raw, source_ink_mask, spiky=bool(spiky_profile.get("spiky", False)), cfg=cfg
        )
        target_changed_text_relief, target_changed_text_relief_diag = _rigid_target_changed_text_relief(
            source, target_reference, sm, tm, cfg, target_bubble=tb
        )
        if cv2.countNonZero(target_changed_text_relief) > 0:
            target_structure_guard[target_changed_text_relief > 0] = 0
            immutable_band[target_changed_text_relief > 0] = 0
            if cv2.countNonZero(spiky_structure_guard) > 0:
                spiky_structure_guard[target_changed_text_relief > 0] = 0
        target_structure_diag["changed_text_relief"] = target_changed_text_relief_diag
        # v2.3.36: a wide immutable burst band must preserve TARGET rays/AA, but
        # it must not freeze Japanese glyphs that happen to sit close to a spike.
        # Relief only compact glyph-like TARGET components; long radial/outline
        # structure is rejected by _compact_container_ink and remains immutable.
        glyph_relief = np.zeros(shape, np.uint8)
        glyph_relief_diag = {"enabled": False, "pixels": 0, "dilate_px": 0}
        if (bool(spiky_profile.get("spiky", False))
                and bool(getattr(cfg, "mask_target_spiky_glyph_relief_enabled", True))):
            compact_target_ink, _ = _compact_container_ink(
                target_reference, tm_raw, 190, cfg, gray=target_gray
            )
            glyph_relief = cv2.bitwise_and(compact_target_ink, immutable_band)
            relief_dilate = max(0, int(getattr(cfg, "mask_target_spiky_glyph_relief_dilate_px", 1)))
            if relief_dilate > 0 and cv2.countNonZero(glyph_relief) > 0:
                rk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (relief_dilate * 2 + 1, relief_dilate * 2 + 1))
                glyph_relief = cv2.dilate(glyph_relief, rk, iterations=1)
                glyph_relief = cv2.bitwise_and(glyph_relief, immutable_band)
            if cv2.countNonZero(glyph_relief) > 0:
                immutable_band[glyph_relief > 0] = 0
            glyph_relief_diag = {
                "enabled": True,
                "pixels": int(cv2.countNonZero(glyph_relief)),
                "dilate_px": int(relief_dilate),
            }
        if cv2.countNonZero(glyph_relief) > 0 and cv2.countNonZero(spiky_structure_guard) > 0:
            spiky_structure_guard[glyph_relief > 0] = 0
        # v2.3.59: immutable boundary is a TARGET-clear boundary, not a SOURCE-write
        # boundary.  Keeping it out of target_structure_guard prevents a wide
        # anti-alias preservation annulus from silently clipping valid Chinese
        # glyphs. Actual dark outline/ray pixels remain protected separately.
        target_structure_diag["immutable_boundary_band"] = immutable_band_diag
        target_structure_diag["spiky_target_glyph_relief"] = glyph_relief_diag
        if cv2.countNonZero(spiky_structure_guard) > 0:
            target_structure_guard = np.maximum(target_structure_guard, spiky_structure_guard)
            target_structure_diag["spiky_guard_pixels"] = int(cv2.countNonZero(spiky_structure_guard))
        clear = np.zeros(shape, np.uint8)
        full_clear_diag = {
            "white_full_clear_applied": False,
            "white_full_clear_reason": "target_colored_or_disabled",
            "white_full_clear_pixels": 0,
        }
        target_colored = bool(sb.meta.get("target_driven_colored"))
        full_clear_envelope = border_safe_envelope.copy()
        # Clear authority is allowed to approach Japanese glyphs but must never
        # cross immutable boundary AA or actual structural TARGET ink.
        # Ordinary balloons keep an immutable AA band.  True burst containers
        # instead clear through the white paper up to the actual ray/outline guard:
        # a blanket immutable annulus can contain legitimate Japanese glyphs (e.g.
        # a final vertical column near the burst tips) and must not make them
        # impossible to remove. Actual TARGET structure is restored byte-exactly.
        if cv2.countNonZero(immutable_band) > 0 and not bool(spiky_profile.get("spiky", False)):
            full_clear_envelope[immutable_band > 0] = 0
        if cv2.countNonZero(target_structure_guard) > 0:
            full_clear_envelope[target_structure_guard > 0] = 0

        if (
            not target_colored
            and bool(getattr(cfg, "white_container_full_clear_enabled", True))
        ):
            clear_env_diag: dict[str, object] = {
                "policy": "spiky_safe_core" if bool(spiky_profile.get("spiky", False)) else "ordinary_white_container",
                "immutable_boundary_pixels": int(cv2.countNonZero(immutable_band)),
                "actual_structure_pixels": int(cv2.countNonZero(target_structure_guard)),
            }
            if not bool(spiky_profile.get("spiky", False)):
                paper_mask = white_container_paper_mask(target_reference, tm, source_ink_mask)
                candidate_clear_env, ordinary_diag = white_container_write_envelope(
                    target_reference, tm, paper_mask,
                    inset_px=max(0, int(getattr(cfg, "white_container_clear_inset_px", 0))),
                    border_guard_px=max(0, int(getattr(cfg, "white_container_clear_border_guard_px", 0))),
                )
                clear_env_diag.update(ordinary_diag)
                if cv2.countNonZero(candidate_clear_env) > 0:
                    full_clear_envelope = candidate_clear_env
                    if cv2.countNonZero(immutable_band) > 0:
                        full_clear_envelope[immutable_band > 0] = 0
                    if cv2.countNonZero(target_structure_guard) > 0:
                        full_clear_envelope[target_structure_guard > 0] = 0
            else:
                clear_env_diag["spiky_profile"] = dict(spiky_profile)
                # A true burst paired-diff polygon can be a narrow/notched text
                # lane rather than the complete white paper interior. Complete
                # TARGET *clear* authority with a conservative bbox ellipse. The
                # ellipse is validated by the same paper-ratio/spread gate below
                # and does not expand SOURCE write authority.
                if bool(getattr(cfg, "mask_target_spiky_safe_core_ellipse_enabled", True)):
                    eb = _bbox_from_mask(tm_raw)
                    if eb is not None:
                        ex0, ey0, ex1, ey1 = eb
                        ew, eh = max(1, ex1 - ex0), max(1, ey1 - ey0)
                        ratio = max(0.0, float(getattr(cfg, "mask_target_spiky_safe_core_inset_ratio", 0.045)))
                        inset = max(
                            int(getattr(cfg, "mask_target_spiky_safe_core_min_inset_px", 6)),
                            int(round(min(ew, eh) * ratio)),
                        )
                        ax = max(2, ew // 2 - inset)
                        ay = max(2, eh // 2 - inset)
                        ellipse = np.zeros(shape, np.uint8)
                        cv2.ellipse(
                            ellipse,
                            (int(round((ex0 + ex1 - 1) * 0.5)), int(round((ey0 + ey1 - 1) * 0.5))),
                            (int(ax), int(ay)), 0.0, 0.0, 360.0, 255, -1, cv2.LINE_8,
                        )
                        full_clear_envelope = np.maximum(full_clear_envelope, ellipse)
                        if cv2.countNonZero(target_structure_guard) > 0:
                            full_clear_envelope[target_structure_guard > 0] = 0
                        clear_env_diag["spiky_safe_core_ellipse_pixels"] = int(cv2.countNonZero(ellipse))
                        clear_env_diag["spiky_safe_core_ellipse_inset_px"] = int(inset)

            # A recovered burst ellipse is only a search envelope. Its white
            # majority must never grant permission to blank coloured pixels at
            # the edge of a star or on the surrounding effect background.
            full_clear_envelope = full_clear_envelope.copy()
            full_clear_envelope[target_hsv[..., 1] > 8] = 0
            full_out, full_mask, full_clear_diag = clear_uniform_white_container_interior(
                rendered, target_reference, full_clear_envelope,
                min_paper_ratio=float(getattr(cfg, "white_container_full_clear_min_paper_ratio", 0.68)),
                max_robust_spread=float(getattr(cfg, "white_container_full_clear_max_robust_spread", 14.0)),
            )
            full_clear_diag["clear_envelope"] = clear_env_diag
            if bool(full_clear_diag.get("white_full_clear_applied", False)):
                rendered = full_out
                clear = full_mask
                if bool(spiky_profile.get("spiky", False)):
                    full_clear_diag["white_full_clear_reason"] = "mask_replace_spiky_safe_core_full_clear"
        elif bool(spiky_profile.get("spiky", False)):
            full_clear_diag["white_full_clear_reason"] = "mask_replace_spiky_colored_or_disabled"
            full_clear_diag["spiky_profile"] = dict(spiky_profile)

        if not bool(full_clear_diag.get("white_full_clear_applied", False)):
            clear = target_text_mask_in_container(target_reference, border_safe_envelope)
            # Include clear-only Koharu semantic text authority.  This recovers
            # large/bold punctuation omitted by the compact component selector.
            if cv2.countNonZero(target_changed_text_relief) > 0:
                clear = np.maximum(clear, target_changed_text_relief)
            if cv2.countNonZero(clear) > 0:
                d = max(1, int(getattr(cfg, "paired_diff_complex_clear_dilate_px", 2)))
                clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1)))
                # TARGET cleanup and SOURCE write have deliberately different
                # authority.  A semantic Koharu relief may extend beyond the
                # narrow paired-diff/write envelope in order to erase the whole
                # original-language glyph.  Do not clip that proven clear-only
                # authority back to border_safe_envelope here.  SOURCE alpha is
                # still clipped to border_safe_envelope below and is never widened.
                clear_envelope = border_safe_envelope
                if cv2.countNonZero(target_changed_text_relief) > 0:
                    clear_envelope = np.maximum(border_safe_envelope, target_changed_text_relief)
                clear = cv2.bitwise_and(clear, clear_envelope)
                if cv2.countNonZero(target_structure_guard) > 0:
                    clear[target_structure_guard > 0] = 0
                cleaned = cv2.inpaint(rendered, clear, 2.5, cv2.INPAINT_TELEA)
                rendered[clear > 0] = cleaned[clear > 0]

        if cv2.countNonZero(target_structure_guard) > 0:
            structure_sel = target_structure_guard > 0
            rendered[structure_sel] = target_reference[structure_sel]
            clear[structure_sel] = 0

        write_alpha8: np.ndarray
        backend = "rigid-container-text-only"
        clarity = "target-background-source-ink-only"
        match_notes = ["rigid_container_text_only", "target_background_preserved", f"uniform_scale={scale:.6f}", f"ink_coverage={ink_cov:.5f}", f"mask_containment={mask_cov:.5f}"]
        gap_fill_diag = {"added_pixels": 0, "enabled": False}
        a = np.clip(alpha, 0.0, 1.0)
        source_authority_pixels = int(cv2.countNonZero(source_ink_mask))
        source_write_envelope = border_safe_envelope
        if cv2.countNonZero(source_semantic_write_extension) > 0:
            source_write_envelope = np.maximum(border_safe_envelope, source_semantic_write_extension)
        a *= (source_write_envelope.astype(np.float32) / 255.0)
        if cv2.countNonZero(target_structure_guard) > 0:
            a[target_structure_guard > 0] = 0.0
        retained_source_mask = ((a >= 0.08) & (source_ink_mask > 0)).astype(np.uint8) * 255
        target_write_pixels = int(cv2.countNonZero(retained_source_mask))
        target_write_retention = float(target_write_pixels / max(1, source_authority_pixels)) if source_authority_pixels > 0 else 1.0
        source_structure_retention = float(white_clarity_diag.get("source_structure_retention", 1.0)) if isinstance(white_clarity_diag, dict) else 1.0
        raw_source_ink_coverage = float(np.clip(source_structure_retention * target_write_retention, 0.0, 1.0))
        raw_source_min_coverage = float(getattr(cfg, "mask_raw_source_min_coverage", 0.965))
        raw_source_fidelity_diag = {
            "enabled": bool(getattr(cfg, "mask_raw_source_completeness_enabled", True)),
            "source_structure_retention": float(source_structure_retention),
            "target_write_retention": float(target_write_retention),
            "raw_source_ink_coverage": float(raw_source_ink_coverage),
            "min_raw_source_coverage": float(raw_source_min_coverage),
            "source_authority_pixels": int(source_authority_pixels),
            "target_write_pixels": int(target_write_pixels),
            "complete": bool(raw_source_ink_coverage >= raw_source_min_coverage),
            "authority_contract": "source-lettering-independent-of-target-clear-and-immutable-boundary",
        }
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
        generic_structure_restore_diag = {
            "enabled": bool(cv2.countNonZero(target_structure_guard) > 0),
            "guard_pixels": int(cv2.countNonZero(target_structure_guard)),
            "changed_before_restore": 0,
            "changed_after_restore": 0,
            "diagnostics": dict(target_structure_diag),
        }
        if cv2.countNonZero(target_structure_guard) > 0:
            guard_sel_all = target_structure_guard > 0
            before_guard_delta = np.max(
                np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
            )
            generic_structure_restore_diag["changed_before_restore"] = int(
                np.count_nonzero(guard_sel_all & (before_guard_delta > 0))
            )
            rendered[guard_sel_all] = target_reference[guard_sel_all]
            write_alpha8[guard_sel_all] = 0
            clear[guard_sel_all] = 0
            after_guard_delta = np.max(
                np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
            )
            generic_structure_restore_diag["changed_after_restore"] = int(
                np.count_nonzero(guard_sel_all & (after_guard_delta > 0))
            )

        spiky_restore_diag = {
            "enabled": bool(spiky_profile.get("spiky", False)),
            "guard_pixels": int(cv2.countNonZero(spiky_structure_guard)),
            "changed_before_restore": 0,
            "changed_after_restore": 0,
        }
        if cv2.countNonZero(spiky_structure_guard) > 0:
            guard_sel = spiky_structure_guard > 0
            before = np.max(
                np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
            )
            spiky_restore_diag["changed_before_restore"] = int(np.count_nonzero(guard_sel & (before > 0)))
            rendered[guard_sel] = target_reference[guard_sel]
            write_alpha8[guard_sel] = 0
            clear[guard_sel] = 0
            after = np.max(
                np.abs(rendered.astype(np.int16) - target_reference.astype(np.int16)), axis=2
            )
            spiky_restore_diag["changed_after_restore"] = int(np.count_nonzero(guard_sel & (after > 0)))

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
            # The paired-diff boundary ring can overlap original-language glyphs
            # when its polygon is a narrow/notched text lane.  Koharu/changed-text
            # relief is stronger semantic evidence for TARGET text and must not be
            # resurrected by exact border restoration.  Real burst outline/rays
            # remain protected because they are outside the semantic relief.
            if cv2.countNonZero(target_changed_text_relief) > 0:
                protected_ring[target_changed_text_relief > 0] = 0
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
        rec.meta["mask_replace_spiky_profile"] = spiky_profile
        rec.meta["mask_replace_spiky_structure_restore"] = spiky_restore_diag
        rec.meta["mask_replace_target_structure_restore"] = generic_structure_restore_diag
        rec.meta["mask_replace_raw_source_fidelity"] = raw_source_fidelity_diag
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
            # The rigid route explicitly records every TARGET pixel it cleared
            # or replaced. A dark SOURCE glyph can legitimately occupy the same
            # coordinates as old Japanese; judge residual TARGET ink only outside
            # this invalidation authority instead of treating new Chinese black as
            # surviving Japanese.
            target_invalidation_mask=write_alpha8,
        )
        if bool(raw_source_fidelity_diag.get("enabled", True)) and not bool(raw_source_fidelity_diag.get("complete", False)):
            rec.content_complete = False
            rec.content_check = "checked_raw_source_fidelity"
            rec.review_required = True
            rec.review_reason = "raw_source_ink_incomplete"
            rec.restorable = True
            rec.editable = True
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
    source_ink_cache: np.ndarray | None = None

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
        if region_kind in {"free_text", "complex_text"} and equivalent_registered_ink(aligned_source, target, tm):
            rec.reason = "preserved_equivalent_source_lettering"
            rec.content_complete = True
            rec.review_required = False
            records.append(rec)
            continue
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
                if source_ink_cache is None:
                    source_ink_cache = _complex_text_ink_map(aligned_source)
                text_img, write_mask, source_ink_mask, text_diag = _transfer_open_complex_text_region(
                    aligned_source, rendered, tm, cfg, source_ink_map=source_ink_cache
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
                rec, source_ink_mask, text_diag.get("target_ink_mask"), text_img, cfg,
                # A cleared dark colour fill is still dark in grayscale. Judge
                # TARGET survival against the actual removal mask, rather than
                # mistaking the restored colour for the old Japanese glyphs.
                target_invalidation_mask=text_diag.get("clear_mask"),
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

        white_policy = (
            _white_bubble_enhancement_policy(target, tm, cfg)
            if region_kind == "bubble"
            else {"enabled": bool(getattr(cfg, "direct_white_clarity_enhance_enabled", False)), "master_bypass": False, "policy": "not_bubble"}
        )
        white_no_enhance_lock = bool(white_policy.get("master_bypass", False))
        white_clarity_diag = {**white_policy, "applied": False, "reason": "disabled_master_bypass" if white_no_enhance_lock else "not_evaluated"}
        if region_kind == "bubble" and not white_no_enhance_lock:
            warped_img, white_clarity_diag = _maybe_apply_white_source_clarity(warped_img, target, tm, cfg)
        source_fidelity_lock = bool(white_clarity_diag.get("applied", False))
        source_enhancement_lock = bool(source_fidelity_lock or white_no_enhance_lock)
        if source_fidelity_lock:
            rec.clarity_mode = "source-faithful-white-clarity"
            rec.meta["source_raster_fidelity_lock"] = True
        elif white_no_enhance_lock:
            rec.clarity_mode = "source-raster-no-enhancement"
            rec.meta["source_raster_no_enhancement_lock"] = True

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
        if (not source_enhancement_lock
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
        if photo_source and cfg.photo_pair_crisp_text_enabled and not source_enhancement_lock:
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
        elif region_kind == "bubble" and not source_enhancement_lock and (fidelity == "ink" or (fidelity == "auto" and too_soft)):
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
            if not source_enhancement_lock:
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
            target_invalidation_mask=write_mask,
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
                # Final CJK raster is shape authority.  Geometry may refine where
                # the destination container lives, but Mask/Hybrid must never
                # independently stretch X/Y merely to increase mask IoU.  The
                # v2.3.57 raw-diff fallback still used _bbox_fit_matrix here,
                # bypassing the shape-preserving contract used by the newer
                # paired-diff route.  Use one uniform local scale for ordinary
                # clean scans as well as photographed pages.
                shape_preserving_fit = (
                    not is_photo_pair
                    or bool(getattr(cfg, "photo_pair_uniform_local_fit", True))
                )
                if shape_preserving_fit:
                    axis_delta = abs(float(corr_x - corr_y))
                    if (not is_photo_pair) or axis_delta <= float(getattr(cfg, "photo_pair_max_axis_scale_delta", 0.10)):
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

        # v2.3.58 fidelity closure for the legacy/raw-diff bubble route.  The
        # v2.3.31 source-raster lock existed in target-driven paired-diff, but
        # raw_diff still fell through to the old sharpness ladder and could
        # replace genuine SOURCE antialiasing with binary ink reconstruction.
        # On a verified ordinary white TARGET container, clean SOURCE paper only
        # with the continuous source-raster normaliser and lock the resulting
        # glyph pixels against all later hardening/reconstruction stages.
        white_policy = _white_bubble_enhancement_policy(target, tm, cfg)
        white_no_enhance_lock = bool(white_policy.get("master_bypass", False))
        white_clarity_diag = {
            **white_policy,
            "applied": False,
            "reason": "disabled_master_bypass" if white_no_enhance_lock else ("photo_pair_uses_photo_fidelity_ladder" if is_photo_pair else "not_evaluated"),
        }
        source_fidelity_lock = bool(white_no_enhance_lock)
        if not is_photo_pair and not white_no_enhance_lock:
            warped_img, white_clarity_diag = _maybe_apply_white_source_clarity(
                warped_img, target, tm, cfg, source_region_mask=warped_mask,
            )
            source_fidelity_lock = bool(white_clarity_diag.get("applied", False))
        rec.meta["white_source_clarity"] = white_clarity_diag
        if bool(white_clarity_diag.get("applied", False)):
            rec.clarity_mode = "source-faithful-white-clarity"
            rec.meta["source_raster_fidelity_lock"] = True
        elif white_no_enhance_lock:
            rec.clarity_mode = "source-raster-no-enhancement"
            rec.meta["source_raster_no_enhancement_lock"] = True

        rec.sharpness = _masked_sharpness(warped_img, paste_mask)
        rec.target_sharpness = _masked_sharpness(target, paste_mask)
        rec.relative_sharpness = rec.sharpness / max(1e-6, rec.target_sharpness) if rec.target_sharpness > 0 else 1.0

        fidelity = (cfg.text_fidelity_mode or "auto").lower().strip()
        if source_fidelity_lock and fidelity in {"auto", "pixels"}:
            fidelity = "pixels"
        if is_photo_pair and not source_fidelity_lock and cfg.photo_pair_force_ink_reconstruction and fidelity == "auto":
            fidelity = "ink"
        if is_photo_pair and not source_fidelity_lock and small_text_photo_pair and fidelity == "auto":
            fidelity = "ink"

        # Preferred v0.8.3 path: extract only registered Chinese ink and rebuild it
        # on clean target paper. This removes camera blur/glare and duplicated
        # source balloon outlines without requiring OCR or a font.
        if is_photo_pair and not source_fidelity_lock and cfg.photo_pair_crisp_text_enabled and fidelity != "reject":
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
        if (is_photo_pair and not source_fidelity_lock and rec.clarity_mode != "photo-crisp-ink"
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
            is_photo_pair and not source_fidelity_lock and rec.clarity_mode != "photo-crisp-ink" and (
                small_text_photo_pair
                or fidelity == "ink"
                or rec.relative_sharpness < cfg.photo_pair_prefer_ink_below_relative_sharpness
            )
        )
        should_try_ink = bool(
            not source_fidelity_lock and (
                fidelity == "ink"
                or (fidelity == "auto" and too_soft)
                or photo_prefers_ink
            )
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
            target_invalidation_mask=paste_mask,
        )
        records.append(rec)

    return MaskTransferResult(rendered, layer, composite_mask, matches, records, clear_mask_all)
