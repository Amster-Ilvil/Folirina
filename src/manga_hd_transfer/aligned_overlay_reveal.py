from __future__ import annotations

"""Public safety facade for experimental aligned erase-to-reveal.

The pixel planner/executor lives in :mod:`aligned_overlay_reveal_core`.  This
facade owns the cross-rendition safety contract:

* TARGET remains the background/colour authority.
* Saturated TARGET pixels, plus a small halo, are protected by default.
* A saturated pixel is released only when it belongs to an already accepted
  ink-only region with additional container/text evidence.  Merely being a
  SOURCE/TARGET pixel difference is not proof of lettering: hair, mouths, hands
  and clothing can differ strongly between B/W and colour editions too.
* Full SOURCE raster never receives a colour release.
* Core REJECT regions remain rejected.  In particular, ``empty_refined_mask`` is
  not automatically rescued; the real-page regression showed that such a rescue
  can reinterpret character artwork as text.
* REJECT/colour-withheld candidates stay visible in diagnostics but are not
  automatically prefilled into the manual-effect dialog.  Users can still draw a
  region themselves when they know an artwork-like area is actually lettering.
"""

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


def _region_allows_colour_text_release(
    region: AlignedOverlayRegion,
    cfg: AlignedOverlayRevealConfig,
) -> bool:
    """Require evidence beyond cross-rendition pixel difference.

    White/near-white dialogue containers can release a small amount of colour
    caused by antialiasing or palette bleed. True open text on coloured artwork
    is also supported, but only when the changed-ink field is large/dense enough
    to look like lettering rather than a tiny face/detail component.
    """
    if region.triage == "REJECT" or region.composite_mode != "ink_only":
        return False

    source_pixels = int(region.source_ink_pixels)
    target_pixels = int(region.target_ink_pixels)
    min_container_ink = max(20, int(cfg.min_region_ink_pixels) * 2)
    if source_pixels < min_container_ink or target_pixels < min_container_ink:
        return False

    if float(region.white_ratio) >= 0.35:
        return True

    x0, y0, x1, y1 = region.target_bbox
    bw = max(1, int(x1) - int(x0))
    bh = max(1, int(y1) - int(y0))
    bbox_area = bw * bh
    min_open_ink = max(80, int(cfg.min_region_ink_pixels) * 8)
    if source_pixels < min_open_ink or target_pixels < min_open_ink:
        return False
    if bbox_area < 900:
        return False
    density = float(source_pixels / max(1, bbox_area))
    if density < 0.018 or density > 0.32:
        return False
    if float(region.color_ratio) > min(0.30, float(cfg.reject_color_ratio)):
        return False
    return True


def _planned_ink_support(
    plan: AlignedOverlayPlan,
    cfg: AlignedOverlayRevealConfig,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Return colour-release support and auditable region-id decisions."""
    support = np.zeros_like(plan.erase_mask)
    released: list[str] = []
    withheld: list[str] = []
    for region in plan.regions:
        if region.triage == "REJECT" or region.composite_mode != "ink_only":
            continue
        if not _region_allows_colour_text_release(region, cfg):
            withheld.append(region.id)
            continue
        support = cv2.bitwise_or(support, region.erase_mask)
        support = cv2.bitwise_or(support, region.source_ink_mask)
        released.append(region.id)
    return support, released, withheld


def _sync_manual_candidate_actionability(
    plan: AlignedOverlayPlan,
    released_ids: list[str],
    withheld_ids: list[str],
) -> None:
    """Do not auto-prefill candidates that look like character/artwork detail."""
    items = plan.diagnostics.get("manual_effect_candidates", [])
    if not isinstance(items, list):
        return
    regions = {r.id: r for r in plan.regions}
    withheld = set(map(str, withheld_ids))
    released = set(map(str, released_ids))
    actionable: list[str] = []
    safety_withheld: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rid = str(item.get("id", ""))
        region = regions.get(rid)
        blocked_reason = ""
        if region is not None and region.triage == "REJECT":
            blocked_reason = "route_rejected"
        elif rid in withheld:
            blocked_reason = "colour_artwork_risk"
        if blocked_reason:
            item["auto_actionable"] = False
            item["manual_prefill_safety_gate"] = blocked_reason
            safety_withheld.append(rid)
        else:
            if rid in released:
                item["manual_prefill_safety_gate"] = "evidence_gated_text"
            if bool(item.get("auto_actionable", False)):
                actionable.append(rid)
    plan.diagnostics["manual_effect_auto_actionable_ids"] = actionable
    plan.diagnostics["manual_effect_safety_withheld_ids"] = safety_withheld


def _target_color_guard_masks(
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
    plan: AlignedOverlayPlan | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Return ``(protect, text_release, saturated_base, released_ids, withheld_ids)``."""
    shape = target.shape[:2]
    if not bool(getattr(cfg, "hard_color_protect_enabled", True)):
        z = np.zeros(shape, dtype=np.uint8)
        return z.copy(), z.copy(), z.copy(), [], []
    if target.ndim != 3 or target.shape[2] < 3:
        z = np.zeros(shape, dtype=np.uint8)
        return z.copy(), z.copy(), z.copy(), [], []

    hsv = cv2.cvtColor(target[:, :, :3], cv2.COLOR_BGR2HSV)
    threshold = max(0, min(255, int(getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION))))
    saturated = (hsv[:, :, 1] >= threshold).astype(np.uint8) * 255
    radius = max(0, int(getattr(cfg, "hard_color_protect_dilate_px", _HARD_COLOR_DILATE_PX)))
    protect = _dilate(saturated, radius)

    release = np.zeros(shape, dtype=np.uint8)
    released_ids: list[str] = []
    withheld_ids: list[str] = []
    if plan is not None:
        support, released_ids, withheld_ids = _planned_ink_support(plan, cfg)
        release[(support > 0) & (protect > 0)] = 255
        protect[release > 0] = 0
    return protect, release, saturated, released_ids, withheld_ids


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

    protect, release, saturated, released_ids, withheld_ids = _target_color_guard_masks(target, cfg, plan)
    _sync_manual_candidate_actionability(plan, released_ids, withheld_ids)
    _strip_protected_pixels(plan, protect)
    _refresh_plan_diagnostics(plan)
    plan.diagnostics.update({
        "hard_color_protected_pixels": int(cv2.countNonZero(protect)),
        "hard_color_saturated_pixels": int(cv2.countNonZero(saturated)),
        "hard_color_text_release_pixels": int(cv2.countNonZero(release)),
        "hard_color_release_region_ids": released_ids,
        "hard_color_withheld_region_ids": withheld_ids,
        "hard_color_protect_saturation": int(
            getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
        ),
        "hard_color_contract": "target_background_authority_with_evidence_gated_text_release",
        "empty_refined_mask_auto_rescue": False,
    })
    return plan


def execute_aligned_overlay(
    plan: AlignedOverlayPlan,
    source: np.ndarray,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayResult:
    protect, release, saturated, released_ids, withheld_ids = _target_color_guard_masks(target, cfg, plan)
    _sync_manual_candidate_actionability(plan, released_ids, withheld_ids)
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
        "hard_color_release_region_ids": released_ids,
        "hard_color_withheld_region_ids": withheld_ids,
        "hard_color_restored_pixels": restored,
        "hard_color_background_changed_after": background_changed,
        "hard_color_saturated_changed_pixels": saturated_changed,
        "hard_color_released_changed_pixels": released_changed,
        "hard_color_protect_saturation": int(
            getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
        ),
        "hard_color_contract": "target_background_authority_with_evidence_gated_text_release",
        "empty_refined_mask_auto_rescue": False,
        "manual_effect_auto_actionable_ids": plan.diagnostics.get("manual_effect_auto_actionable_ids", []),
        "manual_effect_safety_withheld_ids": plan.diagnostics.get("manual_effect_safety_withheld_ids", []),
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
