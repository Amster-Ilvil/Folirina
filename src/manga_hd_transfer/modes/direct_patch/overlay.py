from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .text_transfer import (
    changed_text_masks,
    clear_text_components_to_local_paper,
    cleanup_target_residual_specks,
    clear_uniform_white_container_interior,
    source_text_render,
    target_text_mask_in_container,
    white_container_paper_mask,
    white_container_write_envelope,
    transfer_text_only,
)
from .source_clarity import enhance_white_source_patch


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)


def borderless_inner_mask(region_mask: np.ndarray, *, border_guard_px: int = 2, min_pixels: int = 80) -> np.ndarray:
    mask = (region_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return mask
    guard = max(0, int(border_guard_px))
    if guard > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1))
        inner = cv2.erode(mask, k, iterations=1)
        if cv2.countNonZero(inner) >= int(min_pixels):
            return inner
    return mask


def classify_target_region(target: np.ndarray, region_mask: np.ndarray) -> dict[str, float | bool | str]:
    use = region_mask > 0
    if int(np.count_nonzero(use)) == 0:
        return {
            "kind": "empty",
            "white_ratio": 0.0,
            "dark_ratio": 0.0,
            "saturation_mean": 0.0,
            "high_sat_ratio": 0.0,
            "neutral_white": False,
            "colored": False,
        }
    gray = _gray(target)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    white_ratio = float(np.mean(gray[use] >= 212))
    dark_ratio = float(np.mean(gray[use] <= 188))
    saturation_mean = float(np.mean(sat[use]))
    high_sat_ratio = float(np.mean(sat[use] >= 70))
    neutral_white = bool(white_ratio >= 0.54 and saturation_mean <= 42.0 and high_sat_ratio <= 0.28)
    colored = bool(high_sat_ratio >= 0.22 or saturation_mean >= 45.0)
    return {
        "kind": "white" if neutral_white else ("colored" if colored else "mixed"),
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "saturation_mean": saturation_mean,
        "high_sat_ratio": high_sat_ratio,
        "neutral_white": neutral_white,
        "colored": colored,
    }


def restore_target_structural_damage(
    candidate: np.ndarray,
    target_before: np.ndarray,
    clear_geometry: np.ndarray,
    *,
    enabled: bool = True,
    target_dark_max: int = 205,
    min_lighten: int = 18,
    edge_ratio_x: float = 0.16,
    edge_ratio_y: float = 0.12,
    boundary_band_px: int = 6,
    min_area_px: int = 10,
    fringe_px: int = 1,
    fringe_gray_max: int = 238,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Restore TARGET bubble/rule segments actually damaged by a white clear.

    This runs *after* the candidate is built, so it does not have to guess every
    possible outline in advance. It considers only TARGET dark pixels that the
    candidate really lightened, then restores long/thin components near the outer
    registered container edge. Central Japanese glyphs therefore remain cleared.
    """
    geom = (np.asarray(clear_geometry) > 0).astype(np.uint8)
    empty = np.zeros(geom.shape, np.uint8)
    diag: dict[str, Any] = {
        "enabled": bool(enabled),
        "candidate_lost_dark_pixels": 0,
        "restored_components": 0,
        "restored_core_pixels": 0,
        "restored_pixels": 0,
        "reason": "ok",
    }
    if not bool(enabled):
        diag["reason"] = "disabled"
        return candidate, empty, diag
    if candidate.shape != target_before.shape or candidate.shape[:2] != geom.shape or not np.any(geom):
        diag["reason"] = "shape_or_geometry_invalid"
        return candidate, empty, diag
    tg = _gray(target_before)
    cg = _gray(candidate)
    changed = np.any(candidate != target_before, axis=2) if candidate.ndim == 3 else (candidate != target_before)
    lost_dark = changed & (tg <= int(target_dark_max)) & ((cg.astype(np.int16) - tg.astype(np.int16)) >= int(min_lighten))
    diag["candidate_lost_dark_pixels"] = int(np.count_nonzero(lost_dark))
    if not np.any(lost_dark):
        diag["reason"] = "no_lost_target_dark_pixels"
        return candidate, empty, diag

    nz = cv2.findNonZero(geom * 255)
    if nz is None:
        diag["reason"] = "empty_geometry"
        return candidate, empty, diag
    gx, gy, gw, gh = [int(v) for v in cv2.boundingRect(nz)]
    x_margin = max(6, int(round(gw * float(edge_ratio_x))))
    y_margin = max(6, int(round(gh * float(edge_ratio_y))))
    yy, xx = np.indices(geom.shape)
    edge_strip = geom.astype(bool) & (
        (xx <= gx + x_margin) | (xx >= gx + gw - 1 - x_margin)
        | (yy <= gy + y_margin) | (yy >= gy + gh - 1 - y_margin)
    )
    boundary = cv2.morphologyEx(geom * 255, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    radius = max(2, int(boundary_band_px))
    band = cv2.dilate(
        boundary.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
        iterations=1,
    ) > 0
    core_candidates = lost_dark & edge_strip & band
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core_candidates.astype(np.uint8), 8)
    restore_core = np.zeros(geom.shape, np.uint8)
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < max(3, int(min_area_px)):
            continue
        comp = labels == lab
        span_x = bw / max(1.0, float(gw)); span_y = bh / max(1.0, float(gh))
        fill = area / max(1.0, float(bw * bh))
        near_lr = x <= gx + x_margin or (x + bw - 1) >= gx + gw - 1 - x_margin
        near_tb = y <= gy + y_margin or (y + bh - 1) >= gy + gh - 1 - y_margin
        vertical_line = near_lr and bh >= max(10, int(round(bw * 1.8))) and span_y >= 0.12
        horizontal_line = near_tb and bw >= max(10, int(round(bh * 1.8))) and span_x >= 0.12
        curved_outline = (near_lr or near_tb) and (span_x >= 0.16 or span_y >= 0.16) and fill <= 0.46
        if vertical_line or horizontal_line or curved_outline:
            restore_core[comp] = 255
            diag["restored_components"] += 1
    diag["restored_core_pixels"] = int(cv2.countNonZero(restore_core))
    if diag["restored_core_pixels"] <= 0:
        diag["reason"] = "no_structural_component"
        return candidate, empty, diag

    restore = restore_core > 0
    if int(fringe_px) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(fringe_px) + 1, 2 * int(fringe_px) + 1))
        halo = cv2.dilate(restore_core, k, iterations=1) > 0
        restore |= halo & changed & (tg <= int(fringe_gray_max))
    out = candidate.copy(); out[restore] = target_before[restore]
    restore_u8 = restore.astype(np.uint8) * 255
    diag["restored_pixels"] = int(cv2.countNonZero(restore_u8))
    return out, restore_u8, diag


def build_white_source_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    border_guard_px: int = 2,
    clear_target_text: bool = True,
    clear_dilate_px: int = 1,
    target_clear_region_mask: np.ndarray | None = None,
    white_full_clear_enabled: bool = True,
    white_full_clear_min_paper_ratio: float = 0.68,
    white_full_clear_max_robust_spread: float = 14.0,
    white_source_clarity_enabled: bool = True,
    white_source_clarity_alpha_gamma: float = 1.0,
    white_source_clarity_black_boost: int = 0,
    white_source_clarity_pure_white_floor: int = 248,
    white_source_clarity_min_text_pixels: int = 18,
    post_structural_restore_enabled: bool = True,
    post_structural_restore_target_dark_max: int = 205,
    post_structural_restore_min_lighten: int = 18,
    post_structural_restore_edge_ratio_x: float = 0.16,
    post_structural_restore_edge_ratio_y: float = 0.12,
    post_structural_restore_boundary_band_px: int = 6,
    post_structural_restore_min_area_px: int = 10,
    post_structural_restore_fringe_px: int = 1,
    post_structural_restore_fringe_gray_max: int = 238,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Borderless SOURCE overlay with independent TARGET clearing authority.

    ``region_mask`` is the SOURCE/write authority. ``target_clear_region_mask``
    is the registered TARGET-container authority. They must stay independent:
    edition/crop differences can move Japanese strokes outside the SOURCE mask,
    while SOURCE Chinese still needs a small semantic-edge tolerance.

    For a proven uniform white container we blank the complete protected TARGET
    paper interior first, then place SOURCE raster inside the write envelope.
    This removes Japanese by construction instead of hoping per-glyph matching
    finds every shifted stroke.
    """
    region_u8 = (region_mask > 0).astype(np.uint8) * 255
    empty = np.zeros(region_u8.shape, np.uint8)
    if cv2.countNonZero(region_u8) == 0:
        return target_region.copy(), empty.copy(), empty.copy(), {"mode": "white", "reason": "empty_region"}

    clear_region = region_u8.copy()
    if target_clear_region_mask is not None:
        candidate_clear = (np.asarray(target_clear_region_mask) > 0).astype(np.uint8) * 255
        if candidate_clear.shape == region_u8.shape and cv2.countNonZero(candidate_clear) > 0:
            clear_region = candidate_clear

    # Keep a real SOURCE-text mask for diagnostics/completeness.  Returning the
    # whole container here used to make Direct records report 100% coverage even
    # when Chinese edge columns had actually been clipped.
    source_text_strict = target_text_mask_in_container(source_region, region_u8)
    src_changed, tgt_changed, diff_diag = changed_text_masks(
        source_region, target_region, region_u8, tolerance_px=2
    )
    # The strict compact-text detector is useful as a seed, not as a final glyph
    # clipping mask.  Merge the edition-changed SOURCE support so thin CJK edge
    # strokes and punctuation cannot disappear merely because one selector missed
    # them.  The shared clarity helper still rejects long container boundary rules.
    source_text = cv2.bitwise_or(source_text_strict, src_changed)
    diff_diag = dict(diff_diag)
    diff_diag["source_text_strict_pixels"] = int(cv2.countNonZero(source_text_strict))
    diff_diag["source_text_merged_pixels"] = int(cv2.countNonZero(source_text))

    prepared_source, clarity_diag = enhance_white_source_patch(
        source_region,
        region_u8,
        source_text,
        enabled=bool(white_source_clarity_enabled),
        alpha_gamma=float(white_source_clarity_alpha_gamma),
        black_boost=int(white_source_clarity_black_boost),
        pure_white_floor=int(white_source_clarity_pure_white_floor),
        min_text_pixels=int(white_source_clarity_min_text_pixels),
    )

    base = target_region.copy()
    clear_mask = empty.copy()
    clear_diag: dict[str, Any] = {
        "local_paper_components": 0,
        "local_paper_clear_pixels": 0,
        "local_paper_rejected_components": 0,
    }
    full_clear_diag: dict[str, Any] = {
        "white_full_clear_applied": False,
        "white_full_clear_reason": "not_requested",
        "white_full_clear_pixels": 0,
    }
    clear_env_diag: dict[str, Any] = {}

    if clear_target_text and bool(white_full_clear_enabled):
        # Derive the actual neutral TARGET paper inside the independently mapped
        # TARGET container.  The envelope removes long balloon/rule borders, but
        # does not protect ordinary Japanese glyph edges.
        paper = white_container_paper_mask(target_region, clear_region, source_text)
        clear_env, clear_env_diag = white_container_write_envelope(
            target_region, clear_region, paper, inset_px=0, border_guard_px=0
        )
        if cv2.countNonZero(clear_env) > 0:
            base, clear_mask, full_clear_diag = clear_uniform_white_container_interior(
                base, target_region, clear_env,
                min_paper_ratio=float(white_full_clear_min_paper_ratio),
                max_robust_spread=float(white_full_clear_max_robust_spread),
            )

    # If a container is not uniform enough for complete blanking, keep the old
    # conservative component clear.  This preserves coloured/halftone safety.
    if clear_target_text and not bool(full_clear_diag.get("white_full_clear_applied", False)):
        tgt_text = tgt_changed.copy()
        if cv2.countNonZero(tgt_text) > 0:
            if int(clear_dilate_px) > 0:
                k = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (clear_dilate_px * 2 + 1, clear_dilate_px * 2 + 1),
                )
                tgt_text = cv2.dilate(tgt_text, k, iterations=1)
            tgt_text[clear_region == 0] = 0
            base, clear_mask, clear_diag = clear_text_components_to_local_paper(
                base, target_region, tgt_text, clear_region
            )

    write_mask = borderless_inner_mask(region_u8, border_guard_px=border_guard_px)
    candidate = base.copy()
    use = write_mask > 0
    candidate[use] = prepared_source[use]
    post_restore_diag: dict[str, Any] = {"enabled": False, "restored_pixels": 0, "reason": "not_requested"}
    if bool(full_clear_diag.get("white_full_clear_applied", False)):
        candidate, restored_structural, post_restore_diag = restore_target_structural_damage(
            candidate, target_region, clear_region,
            enabled=bool(post_structural_restore_enabled),
            target_dark_max=int(post_structural_restore_target_dark_max),
            min_lighten=int(post_structural_restore_min_lighten),
            edge_ratio_x=float(post_structural_restore_edge_ratio_x),
            edge_ratio_y=float(post_structural_restore_edge_ratio_y),
            boundary_band_px=int(post_structural_restore_boundary_band_px),
            min_area_px=int(post_structural_restore_min_area_px),
            fringe_px=int(post_structural_restore_fringe_px),
            fringe_gray_max=int(post_structural_restore_fringe_gray_max),
        )
        if cv2.countNonZero(restored_structural) > 0:
            source_text[restored_structural > 0] = 0
            clear_mask[restored_structural > 0] = 0
    changed = (np.any(candidate != target_region, axis=2)).astype(np.uint8) * 255
    # SOURCE placement is still bounded by the write authority.  TARGET-only
    # clearing may legitimately extend beyond it inside the mapped HD container.
    changed = cv2.bitwise_or(changed, clear_mask)
    source_text[write_mask == 0] = 0
    diag = {
        "mode": "white",
        "strategy": "borderless_source_overlay_independent_target_clear",
        "source_on_top": True,
        "target_underlay": True,
        "source_border_removed": True,
        "rotation_locked": True,
        "clear_target_text": bool(clear_target_text),
        "changed_text_masks": diff_diag,
        "clear_diag": clear_diag,
        "target_clear_envelope": clear_env_diag,
        "white_source_clarity": clarity_diag,
        "post_structural_restore": post_restore_diag,
        **full_clear_diag,
        "write_pixels": int(cv2.countNonZero(changed)),
        "payload_pixels": int(cv2.countNonZero(write_mask)),
        "source_text_pixels": int(cv2.countNonZero(source_text)),
        "target_text_pixels": int(cv2.countNonZero(tgt_changed)),
        "target_clear_region_pixels": int(cv2.countNonZero(clear_region)),
    }
    return candidate, changed, source_text, diag


def build_colored_text_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    target_clear_region_mask: np.ndarray | None = None,
    clear_dilate_px: int = 1,
    inpaint_radius: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Color-preserving direct mode path for coloured/open areas.

    Keeps the TARGET background and composites only changed SOURCE Chinese glyphs.
    """
    candidate, write_mask, source_text_mask, diag = transfer_text_only(
        target_region.copy(),
        source_region,
        (region_mask > 0).astype(np.uint8) * 255,
        tolerance_px=2,
        clear_dilate_px=max(0, int(clear_dilate_px)),
        inpaint_radius=float(inpaint_radius),
        white_container=False,
        localized_white_text=False,
        white_full_clear_enabled=False,
        target_clear_region_mask=target_clear_region_mask,
    )
    cleaned, residual_mask, residual_diag = cleanup_target_residual_specks(
        candidate, target_region, (region_mask > 0).astype(np.uint8) * 255, source_text_mask, write_mask,
        white_container=False, inpaint_radius=float(inpaint_radius),
    )
    final_mask = cv2.bitwise_or(write_mask, residual_mask)
    diag = dict(diag)
    diag.update({
        "mode": "colored",
        "strategy": "target_background_source_text_overlay",
        "source_on_top": True,
        "target_underlay": True,
        "source_border_removed": True,
        "rotation_locked": True,
        "residual_cleanup": residual_diag,
        "write_pixels": int(cv2.countNonZero(final_mask)),
        "source_text_pixels": int(cv2.countNonZero(source_text_mask)),
    })
    return cleaned, final_mask, source_text_mask, diag


def _refine_region_mask_with_support(
    region_mask: np.ndarray,
    support_mask: np.ndarray | None,
    *,
    white_mode: bool,
    min_keep_ratio: float = 0.45,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Bound Direct writes to semantic support with a tiny raster tolerance.

    Detector masks are geometric evidence, not pixel-perfect glyph alpha.  A
    zero-tolerance intersection clips antialiased/edge-adjacent Chinese strokes.
    We therefore permit only a bounded 4px (white) / 6px (coloured) dilation,
    always intersected with the already accepted Direct candidate.  Empty
    supplied support still fails closed; there is no unbounded fallback.
    """
    mask = (region_mask > 0).astype(np.uint8) * 255
    requested = int(cv2.countNonZero(mask))
    if requested == 0:
        return mask, {"mode": "empty_candidate", "tolerance_px": 0, "requested_pixels": 0, "effective_pixels": 0}
    if support_mask is None:
        return mask, {
            "mode": "candidate_fallback", "tolerance_px": None,
            "requested_pixels": requested, "support_pixels": None,
            "allowed_support_pixels": None, "effective_pixels": requested,
            "clipped_pixels": 0,
        }
    support = (support_mask > 0).astype(np.uint8) * 255
    support_px = int(cv2.countNonZero(support))
    if support_px == 0:
        return np.zeros_like(mask), {
            "mode": "fail_closed_empty_support", "tolerance_px": 0,
            "requested_pixels": requested, "support_pixels": 0,
            "allowed_support_pixels": 0, "effective_pixels": 0,
            "clipped_pixels": requested,
        }
    tolerance = 4 if bool(white_mode) else 6
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tolerance * 2 + 1, tolerance * 2 + 1))
    allowed = cv2.dilate(support, k, iterations=1)
    refined = cv2.bitwise_and(mask, allowed)
    refined_px = int(cv2.countNonZero(refined))
    # Keep fail-closed semantics for badly inconsistent geometry.  This threshold
    # is diagnostic/safety only; no large candidate is restored automatically.
    min_keep = max(12, int(round(requested * float(min_keep_ratio))))
    if refined_px < min_keep:
        refined = np.zeros_like(mask)
        refined_px = 0
    return refined, {
        "mode": "bounded_semantic_support",
        "tolerance_px": int(tolerance),
        "requested_pixels": requested,
        "support_pixels": support_px,
        "allowed_support_pixels": int(cv2.countNonZero(allowed)),
        "effective_pixels": refined_px,
        "clipped_pixels": int(max(0, requested - refined_px)),
    }


def compose_direct_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    white_mode: bool,
    support_mask: np.ndarray | None = None,
    border_guard_px: int = 2,
    clear_target_text: bool = True,
    clear_dilate_px: int = 1,
    inpaint_radius: float = 2.5,
    target_clear_region_mask: np.ndarray | None = None,
    white_source_clarity_enabled: bool = True,
    white_source_clarity_alpha_gamma: float = 1.0,
    white_source_clarity_black_boost: int = 0,
    white_source_clarity_pure_white_floor: int = 248,
    white_source_clarity_min_text_pixels: int = 18,
    post_structural_restore_enabled: bool = True,
    post_structural_restore_target_dark_max: int = 205,
    post_structural_restore_min_lighten: int = 18,
    post_structural_restore_edge_ratio_x: float = 0.16,
    post_structural_restore_edge_ratio_y: float = 0.12,
    post_structural_restore_boundary_band_px: int = 6,
    post_structural_restore_min_area_px: int = 10,
    post_structural_restore_fringe_px: int = 1,
    post_structural_restore_fringe_gray_max: int = 238,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    requested_mask = (region_mask > 0).astype(np.uint8) * 255
    effective_mask, guard_diag = _refine_region_mask_with_support(
        region_mask, support_mask, white_mode=bool(white_mode)
    )
    if bool(white_mode):
        candidate, write_mask, source_payload, diag = build_white_source_overlay(
            target_region, source_region, effective_mask,
            border_guard_px=border_guard_px,
            clear_target_text=clear_target_text,
            clear_dilate_px=clear_dilate_px,
            target_clear_region_mask=target_clear_region_mask,
            white_full_clear_enabled=True,
            white_source_clarity_enabled=white_source_clarity_enabled,
            white_source_clarity_alpha_gamma=white_source_clarity_alpha_gamma,
            white_source_clarity_black_boost=white_source_clarity_black_boost,
            white_source_clarity_pure_white_floor=white_source_clarity_pure_white_floor,
            white_source_clarity_min_text_pixels=white_source_clarity_min_text_pixels,
            post_structural_restore_enabled=post_structural_restore_enabled,
            post_structural_restore_target_dark_max=post_structural_restore_target_dark_max,
            post_structural_restore_min_lighten=post_structural_restore_min_lighten,
            post_structural_restore_edge_ratio_x=post_structural_restore_edge_ratio_x,
            post_structural_restore_edge_ratio_y=post_structural_restore_edge_ratio_y,
            post_structural_restore_boundary_band_px=post_structural_restore_boundary_band_px,
            post_structural_restore_min_area_px=post_structural_restore_min_area_px,
            post_structural_restore_fringe_px=post_structural_restore_fringe_px,
            post_structural_restore_fringe_gray_max=post_structural_restore_fringe_gray_max,
        )
    else:
        candidate, write_mask, source_payload, diag = build_colored_text_overlay(
            target_region, source_region, effective_mask,
            target_clear_region_mask=target_clear_region_mask,
            clear_dilate_px=clear_dilate_px,
            inpaint_radius=inpaint_radius,
        )

    # SOURCE pixels may only enter the bounded semantic write envelope. TARGET
    # clearing is a separate authority and may extend farther inside the mapped
    # white container; do not restore those legitimately cleared pixels here.
    allowed = effective_mask > 0
    source_payload = (source_payload > 0).astype(np.uint8) * 255
    source_payload[~allowed] = 0
    actual_changed = (np.any(candidate != target_region, axis=2)).astype(np.uint8) * 255
    write_mask = cv2.bitwise_and((write_mask > 0).astype(np.uint8) * 255, actual_changed)
    diag = dict(diag)
    diag["strict_support_guard"] = guard_diag
    diag["outside_allowed_source_payload_pixels"] = int(np.count_nonzero(source_payload[~allowed]))
    return candidate, write_mask, source_payload, diag
