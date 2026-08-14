from __future__ import annotations

"""Public safety facade for experimental aligned erase-to-reveal.

The pixel planner/executor lives in :mod:`aligned_overlay_reveal_core`.  This
facade owns the cross-rendition safety contract:

* TARGET remains the background/colour authority.
* Proven SOURCE/TARGET lettering masks may edit pixels even when the underlying
  TARGET background is coloured.  Otherwise open text/SFX on colour artwork
  could never be transferred completely.
* Saturated TARGET pixels outside those proven ink masks, plus a small halo,
  remain byte-for-byte authoritative.
* A narrowly scoped local rescue can recover a region that the core already
  identified as text but rejected only because progressive structural guarding
  reduced both refined masks to almost nothing.  Rescued regions stay REVIEW.
"""

from typing import Any

import cv2
import numpy as np

from .config import AlignedOverlayRevealConfig
from .models import RegistrationResult
from . import aligned_overlay_reveal_core as _core

AlignedOverlayRegion = _core.AlignedOverlayRegion
AlignedOverlayPlan = _core.AlignedOverlayPlan
AlignedOverlayResult = _core.AlignedOverlayResult

_HARD_COLOR_SATURATION = 20
_HARD_COLOR_DILATE_PX = 2


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    binary = (mask > 0).astype(np.uint8) * 255
    if radius <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(binary, kernel)


def _planned_ink_support(plan: AlignedOverlayPlan) -> np.ndarray:
    """Pixels that the core proved to be TARGET erase or SOURCE lettering.

    Full-raster masks are intentionally excluded.  Saturated colour is never
    released merely because a white-container fallback wants to copy a raster.
    """
    support = np.zeros_like(plan.erase_mask)
    for region in plan.regions:
        if region.triage == "REJECT" or region.composite_mode != "ink_only":
            continue
        support = cv2.bitwise_or(support, region.erase_mask)
        support = cv2.bitwise_or(support, region.source_ink_mask)
    return support


def _target_color_guard_masks(
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
    plan: AlignedOverlayPlan | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(protect, text_release, saturated_base)``.

    The old guard protected *every* saturated pixel.  That correctly prevented
    artwork damage but also made legitimate Chinese lettering on coloured
    backgrounds impossible: the new glyph necessarily replaces some coloured
    background pixels.  The refined contract protects colour everywhere except
    exact ink-only masks independently proved by the core planner.
    """
    shape = target.shape[:2]
    if not bool(getattr(cfg, "hard_color_protect_enabled", True)):
        z = np.zeros(shape, dtype=np.uint8)
        return z.copy(), z.copy(), z.copy()
    if target.ndim != 3 or target.shape[2] < 3:
        z = np.zeros(shape, dtype=np.uint8)
        return z.copy(), z.copy(), z.copy()

    hsv = cv2.cvtColor(target[:, :, :3], cv2.COLOR_BGR2HSV)
    threshold = max(0, min(255, int(getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION))))
    saturated = (hsv[:, :, 1] >= threshold).astype(np.uint8) * 255
    radius = max(0, int(getattr(cfg, "hard_color_protect_dilate_px", _HARD_COLOR_DILATE_PX)))
    protect = _dilate(saturated, radius)

    release = np.zeros(shape, dtype=np.uint8)
    if plan is not None:
        support = _planned_ink_support(plan)
        release[(support > 0) & (protect > 0)] = 255
        protect[release > 0] = 0
    return protect, release, saturated


def _strip_protected_pixels(plan: AlignedOverlayPlan, protect: np.ndarray) -> None:
    sel = protect > 0
    if not np.any(sel):
        return
    for mask in (plan.erase_mask, plan.source_ink_mask, plan.full_raster_mask):
        mask[sel] = 0
    page_area = max(1, protect.shape[0] * protect.shape[1])
    for region in plan.regions:
        for mask in (region.erase_mask, region.source_ink_mask, region.full_raster_mask):
            mask[sel] = 0
        region.source_ink_pixels = int(cv2.countNonZero(region.source_ink_mask))
        region.target_ink_pixels = int(cv2.countNonZero(region.erase_mask))
        region.erase_area_ratio = float(region.target_ink_pixels / page_area)


def _local_changed_components(
    dark: np.ndarray,
    exclusive_seed: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    min_area: int,
) -> np.ndarray:
    """Recover compact changed ink locally without re-admitting long artwork.

    This helper is deliberately used only for an already-created core region with
    reason ``empty_refined_mask``.  It therefore has much stronger localisation
    evidence than a page-wide fallback.
    """
    h, w = dark.shape
    x0, y0, x1, y1 = bbox
    x0 = max(0, min(w, int(x0))); x1 = max(x0, min(w, int(x1)))
    y0 = max(0, min(h, int(y0))); y1 = max(y0, min(h, int(y1)))
    out = np.zeros_like(dark, dtype=np.uint8)
    if x1 <= x0 or y1 <= y0:
        return out

    crop = (dark[y0:y1, x0:x1] > 0).astype(np.uint8)
    seed = exclusive_seed[y0:y1, x0:x1] > 0
    if not np.any(crop) or not np.any(seed):
        return out

    n, labels, stats, _ = cv2.connectedComponentsWithStats(crop, 8)
    ch, cw = crop.shape
    kept = np.zeros_like(crop, dtype=np.uint8)
    for label in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[label]]
        if area < max(1, int(min_area)):
            continue
        comp = labels == label
        if not np.any(comp & seed):
            continue
        fill = float(area / max(1, ww * hh))
        span_x = float(ww / max(1, cw))
        span_y = float(hh / max(1, ch))
        touches = x <= 0 or y <= 0 or (x + ww) >= cw or (y + hh) >= ch
        # Long sparse components are panel rules, hair, clothing or balloon
        # boundaries even when a few changed pixels happen to touch them.
        if max(span_x, span_y) > 0.92 and fill < 0.22:
            continue
        if touches and max(span_x, span_y) > 0.78 and fill < 0.18:
            continue
        if area / max(1, cw * ch) > 0.30 and fill < 0.30:
            continue
        kept[comp] = 255
    out[y0:y1, x0:x1] = kept
    return out


def _exclusive_ink(
    plan: AlignedOverlayPlan,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    aligned = plan.aligned_source
    sg = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY) if aligned.ndim == 3 else aligned.astype(np.uint8)
    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target.astype(np.uint8)
    valid = plan.valid_mask > 0
    src_dark = (sg <= int(cfg.source_ink_threshold)) & valid
    tgt_dark = (tg <= int(cfg.target_ink_threshold)) & valid
    tol = max(0, int(cfg.registration_tolerance_px))
    src_near = _dilate(src_dark.astype(np.uint8) * 255, tol) > 0
    tgt_near = _dilate(tgt_dark.astype(np.uint8) * 255, tol) > 0
    delta = max(0, int(cfg.ink_difference_delta))
    src_exclusive = src_dark & (~tgt_near | ((tg.astype(np.int16) - sg.astype(np.int16)) >= delta))
    tgt_exclusive = tgt_dark & (~src_near | ((sg.astype(np.int16) - tg.astype(np.int16)) >= delta))
    return sg, tg, src_dark, tgt_dark, src_exclusive, tgt_exclusive


def _update_manual_candidate(plan: AlignedOverlayPlan, region: AlignedOverlayRegion) -> None:
    items = plan.diagnostics.get("manual_effect_candidates", [])
    if not isinstance(items, list):
        return
    for item in items:
        if isinstance(item, dict) and item.get("id") == region.id:
            item["reason"] = f"aligned_overlay:{region.reason}"
            item["triage"] = region.triage


def _rescue_empty_refined_regions(
    plan: AlignedOverlayPlan,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> None:
    """Recover only core-identified regions rejected by over-aggressive guarding.

    No new page-wide candidate is invented here.  A rescue requires both SOURCE
    and TARGET local exclusive ink, stays inside the original region bbox and
    registered SOURCE validity, obeys per-region/page erase caps, and remains
    REVIEW so the GUI keeps it reversible/editable.
    """
    candidates = [r for r in plan.regions if r.triage == "REJECT" and r.reason == "empty_refined_mask"]
    if not candidates:
        plan.diagnostics.setdefault("rescued_empty_region_count", 0)
        return

    sg, _tg, src_dark, tgt_dark, src_exclusive, tgt_exclusive = _exclusive_ink(plan, target, cfg)
    page_area = max(1, target.shape[0] * target.shape[1])
    rescue_min = max(4, int(cfg.min_region_ink_pixels) // 2)
    rescued: list[str] = []

    for region in candidates:
        src = _local_changed_components(
            src_dark.astype(np.uint8) * 255,
            src_exclusive.astype(np.uint8) * 255,
            region.target_bbox,
            min_area=max(1, int(cfg.min_component_area_px)),
        )
        tgt = _local_changed_components(
            tgt_dark.astype(np.uint8) * 255,
            tgt_exclusive.astype(np.uint8) * 255,
            region.target_bbox,
            min_area=max(1, int(cfg.min_component_area_px)),
        )
        if cv2.countNonZero(src) < rescue_min or cv2.countNonZero(tgt) < rescue_min:
            continue

        x0, y0, x1, y1 = region.target_bbox
        bbox_mask = np.zeros_like(plan.erase_mask)
        bbox_mask[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
        corridor = _dilate(cv2.bitwise_or(src, tgt), max(1, int(cfg.text_corridor_radius_px)))
        corridor = cv2.bitwise_and(corridor, bbox_mask)
        corridor = cv2.bitwise_and(corridor, plan.valid_mask)

        erase = _dilate(tgt, int(cfg.erase_dilate_px))
        erase = cv2.bitwise_and(erase, corridor)
        source_write = cv2.bitwise_and(src, corridor)
        if int(cfg.source_ink_antialias_px) > 0:
            fringe = _dilate(source_write, int(cfg.source_ink_antialias_px))
            source_write = np.where(
                (fringe > 0) & (sg <= int(cfg.source_antialias_threshold)),
                255,
                source_write,
            ).astype(np.uint8)
            source_write = cv2.bitwise_and(source_write, corridor)

        ep = int(cv2.countNonZero(erase))
        sp = int(cv2.countNonZero(source_write))
        if ep < rescue_min or sp < rescue_min:
            continue
        erase_ratio = float(ep / page_area)
        if erase_ratio > float(cfg.max_single_region_area_ratio):
            continue
        candidate_page = cv2.bitwise_or(plan.erase_mask, erase)
        if float(cv2.countNonZero(candidate_page) / page_area) > float(cfg.max_erase_area_ratio_per_page):
            continue

        region.erase_mask = erase
        region.source_ink_mask = source_write
        region.full_raster_mask[:] = 0
        region.composite_mode = "ink_only"
        region.triage = "REVIEW"
        region.reason = "empty_mask_local_exclusive_rescue"
        region.erase_area_ratio = erase_ratio
        region.source_ink_pixels = sp
        region.target_ink_pixels = ep
        region.diagnostics.update({
            "local_exclusive_rescue": True,
            "rescue_min_ink_pixels": rescue_min,
            "rescued_source_pixels": sp,
            "rescued_target_pixels": ep,
        })
        plan.erase_mask = candidate_page
        plan.source_ink_mask = cv2.bitwise_or(plan.source_ink_mask, source_write)
        plan.full_raster_mask[erase > 0] = 0
        _update_manual_candidate(plan, region)
        rescued.append(region.id)

    if rescued:
        plan.accepted = True
        if plan.reason == "no_accepted_regions":
            plan.reason = "ok"
    plan.diagnostics["rescued_empty_regions"] = rescued
    plan.diagnostics["rescued_empty_region_count"] = len(rescued)


def _refresh_plan_diagnostics(plan: AlignedOverlayPlan) -> None:
    page_area = max(1, plan.erase_mask.shape[0] * plan.erase_mask.shape[1])
    plan.diagnostics.update({
        "erase_pixels": int(cv2.countNonZero(plan.erase_mask)),
        "source_ink_pixels": int(cv2.countNonZero(plan.source_ink_mask)),
        "full_raster_pixels": int(cv2.countNonZero(plan.full_raster_mask)),
        "erase_area_ratio": float(cv2.countNonZero(plan.erase_mask) / page_area),
        "region_count": len(plan.regions),
        "applied_region_count": len(plan.applied_regions),
        "triage": plan.page_triage,
    })


def build_aligned_overlay_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayPlan:
    plan = _core.build_aligned_overlay_plan(source, target, registration, cfg)
    _rescue_empty_refined_regions(plan, target, cfg)

    protect, release, saturated = _target_color_guard_masks(target, cfg, plan)
    _strip_protected_pixels(plan, protect)
    _refresh_plan_diagnostics(plan)
    plan.diagnostics.update({
        "hard_color_protected_pixels": int(cv2.countNonZero(protect)),
        "hard_color_saturated_pixels": int(cv2.countNonZero(saturated)),
        "hard_color_text_release_pixels": int(cv2.countNonZero(release)),
        "hard_color_protect_saturation": int(
            getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
        ),
        "hard_color_contract": "target_background_authority_except_proven_text_masks",
    })
    return plan


def execute_aligned_overlay(
    plan: AlignedOverlayPlan,
    source: np.ndarray,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayResult:
    # Recompute from the current plan in case review/manual tooling changed masks
    # between build and execute.  Only ink-only masks get a colour release.
    protect, release, saturated = _target_color_guard_masks(target, cfg, plan)
    _strip_protected_pixels(plan, protect)
    _refresh_plan_diagnostics(plan)
    result = _core.execute_aligned_overlay(plan, source, target, cfg)

    before = np.any(result.image != target, axis=2)
    protected = protect > 0
    restored = int(np.count_nonzero(before & protected))
    if np.any(protected):
        result.image[protected] = target[protected]

    changed = np.any(result.image != target, axis=2)
    layer = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    rgb = cv2.cvtColor(result.image, cv2.COLOR_BGR2RGB)
    layer[changed, :3] = rgb[changed]
    layer[changed, 3] = 255
    result.layer_rgba = layer

    changed_pixels = int(np.count_nonzero(changed))
    background_changed = int(np.count_nonzero(changed & protected))
    saturated_changed = int(np.count_nonzero(changed & (saturated > 0)))
    released_changed = int(np.count_nonzero(changed & (release > 0)))
    nearly_unchanged = bool(
        result.plan.accepted
        and (
            len(result.plan.applied_regions) == 0
            or changed_pixels < max(16, int(target.shape[0] * target.shape[1] * 0.00001))
        )
    )
    result.diagnostics.update({
        "changed_pixels": changed_pixels,
        "nearly_unchanged": nearly_unchanged,
        "result_hint": "accepted_but_almost_no_visible_change" if nearly_unchanged else "ok",
        "source_background_authority": False,
        "hard_color_protected_pixels": int(cv2.countNonZero(protect)),
        "hard_color_saturated_pixels": int(cv2.countNonZero(saturated)),
        "hard_color_text_release_pixels": int(cv2.countNonZero(release)),
        "hard_color_restored_pixels": restored,
        "hard_color_background_changed_after": background_changed,
        "hard_color_saturated_changed_pixels": saturated_changed,
        "hard_color_released_changed_pixels": released_changed,
        "hard_color_protect_saturation": int(
            getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
        ),
        "hard_color_contract": "target_background_authority_except_proven_text_masks",
        "page_triage": "REVIEW" if nearly_unchanged else result.plan.page_triage,
    })
    return result


__all__ = [
    "AlignedOverlayRegion",
    "AlignedOverlayPlan",
    "AlignedOverlayResult",
    "build_aligned_overlay_plan",
    "execute_aligned_overlay",
]
