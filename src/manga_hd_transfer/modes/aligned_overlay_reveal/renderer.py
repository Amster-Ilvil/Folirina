from __future__ import annotations

"""Public safety facade for experimental aligned erase-to-reveal.

The original pixel planner/executor lives in :mod:`aligned_overlay_reveal_core`.
This facade enforces a page-level invariant that region-ratio triage alone cannot:
when SOURCE is monochrome and TARGET is colour, saturated TARGET artwork and a
small edge neighbourhood remain byte-for-byte authoritative.

Keeping this guard outside the core planner makes the invariant easy to audit and
ensures future mask/triage changes cannot accidentally re-enable colour damage.
"""

from typing import Any

import cv2
import numpy as np

from ...config import AlignedOverlayRevealConfig, BubbleConfig
from ...detector_policy import koharu_is_primary, primary_detector
from ...layout_evidence import collect_koharu_layout_evidence_cached
from ...models import RegistrationResult
from ...pipeline_bubble_service import bubbles_cached


def collect_koharu_layout_evidence(image, bubble_cfg=None, role="page", allow_missing=True, **kwargs):
    return collect_koharu_layout_evidence_cached(
        image, bubble_cfg, role=role, allow_missing=allow_missing, **kwargs
    )
from . import core as _core

AlignedOverlayRegion = _core.AlignedOverlayRegion
AlignedOverlayPlan = _core.AlignedOverlayPlan
AlignedOverlayResult = _core.AlignedOverlayResult

# Conservative defaults are intentionally available through getattr so old saved
# config/schema objects remain compatible without adding a new mandatory field.
_HARD_COLOR_SATURATION = 20
_HARD_COLOR_DILATE_PX = 2


def _target_color_protect_mask(target: np.ndarray, cfg: AlignedOverlayRevealConfig) -> np.ndarray:
    if not bool(getattr(cfg, "hard_color_protect_enabled", True)):
        return np.zeros(target.shape[:2], dtype=np.uint8)
    if target.ndim != 3 or target.shape[2] < 3:
        return np.zeros(target.shape[:2], dtype=np.uint8)
    hsv = cv2.cvtColor(target[:, :, :3], cv2.COLOR_BGR2HSV)
    threshold = max(0, min(255, int(getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION))))
    protect = (hsv[:, :, 1] >= threshold).astype(np.uint8) * 255
    radius = max(0, int(getattr(cfg, "hard_color_protect_dilate_px", _HARD_COLOR_DILATE_PX)))
    if radius > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        protect = cv2.dilate(protect, k)
    return protect


def _strip_protected_pixels(plan: AlignedOverlayPlan, protect: np.ndarray) -> None:
    sel = protect > 0
    if not np.any(sel):
        return
    for mask in (plan.erase_mask, plan.source_ink_mask, plan.full_raster_mask):
        mask[sel] = 0
    for region in plan.regions:
        for mask in (region.erase_mask, region.source_ink_mask, region.full_raster_mask):
            mask[sel] = 0
    plan.diagnostics["hard_color_protected_pixels"] = int(cv2.countNonZero(protect))
    plan.diagnostics["hard_color_protect_saturation"] = int(
        getattr(plan, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
    )



def _mask_overlap_ratio(mask: np.ndarray, support: np.ndarray) -> float:
    area = int(cv2.countNonZero(mask))
    if area <= 0:
        return 0.0
    inter = int(np.count_nonzero((mask > 0) & (support > 0)))
    return float(inter / max(1, area))


def _rebuild_page_masks(plan: AlignedOverlayPlan) -> None:
    plan.erase_mask[:] = 0
    plan.source_ink_mask[:] = 0
    plan.full_raster_mask[:] = 0
    for region in plan.regions:
        if region.triage == "REJECT":
            continue
        plan.erase_mask = np.maximum(plan.erase_mask, region.erase_mask)
        plan.source_ink_mask = np.maximum(plan.source_ink_mask, region.source_ink_mask)
        plan.full_raster_mask = np.maximum(plan.full_raster_mask, region.full_raster_mask)


def apply_koharu_layout_guard(
    plan: AlignedOverlayPlan,
    target: np.ndarray,
    bubble_cfg: BubbleConfig | None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> AlignedOverlayPlan:
    """Constrain experimental erase regions using Koharu text/bubble layout.

    This is a positive-evidence guard only. When Koharu is absent/unavailable the
    experimental mode keeps its current behaviour unchanged. When available, a
    region must overlap text/SFX layout evidence, and full-raster white regions
    additionally need bubble support. This reduces illustration false positives
    without making the route depend on OCR.
    """
    try:
        evidence = collect_koharu_layout_evidence(
            target, bubble_cfg, role="aligned_overlay_target", image_path=target_path,
            cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=True,
        )
    except Exception as exc:  # pragma: no cover - defensive only
        plan.diagnostics["koharu_layout_guard"] = {"available": False, "reason": "error", "error": str(exc)}
        return plan
    if not evidence.available:
        plan.diagnostics["koharu_layout_guard"] = dict(evidence.diagnostics)
        return plan

    text_support = evidence.combined_mask(("text", "sfx"), dilate_px=6)
    bubble_support = evidence.combined_mask(("bubble",), dilate_px=3)
    any_support = np.maximum(text_support, bubble_support)
    changed_regions = 0
    rejected_regions = 0

    for region in plan.regions:
        before_erase = int(cv2.countNonZero(region.erase_mask))
        before_write = int(cv2.countNonZero(region.source_ink_mask))
        before_full = int(cv2.countNonZero(region.full_raster_mask))
        erase_overlap = _mask_overlap_ratio(region.erase_mask, any_support)
        text_overlap = _mask_overlap_ratio(region.source_ink_mask, text_support)
        bubble_overlap = _mask_overlap_ratio(region.full_raster_mask if before_full > 0 else region.erase_mask, bubble_support)

        region.diagnostics["koharu_layout_guard"] = {
            "erase_overlap": erase_overlap,
            "text_overlap": text_overlap,
            "bubble_overlap": bubble_overlap,
            "available": True,
        }

        if before_erase <= 0:
            continue
        # No overlap with any text/bubble evidence -> reject the region.
        if erase_overlap < 0.04 and text_overlap < 0.04 and bubble_overlap < 0.06:
            region.erase_mask[:] = 0
            region.source_ink_mask[:] = 0
            region.full_raster_mask[:] = 0
            region.triage = "REJECT"
            region.reason = f"{region.reason}+koharu_layout_no_support"
            rejected_regions += 1
            continue

        write_support = text_support
        erase_support = text_support.copy()
        if before_full > 0 and bubble_overlap >= 0.06:
            erase_support = np.maximum(erase_support, bubble_support)
            region.full_raster_mask = cv2.bitwise_and(region.full_raster_mask, bubble_support)
        else:
            region.full_raster_mask[:] = 0
        region.erase_mask = cv2.bitwise_and(region.erase_mask, erase_support)
        region.source_ink_mask = cv2.bitwise_and(region.source_ink_mask, write_support)
        if cv2.countNonZero(region.erase_mask) <= 0 or cv2.countNonZero(region.source_ink_mask) <= 0:
            region.erase_mask[:] = 0
            region.source_ink_mask[:] = 0
            region.full_raster_mask[:] = 0
            region.triage = "REJECT"
            region.reason = f"{region.reason}+koharu_layout_empty_after_guard"
            rejected_regions += 1
            continue
        after_erase = int(cv2.countNonZero(region.erase_mask))
        after_write = int(cv2.countNonZero(region.source_ink_mask))
        after_full = int(cv2.countNonZero(region.full_raster_mask))
        if after_erase != before_erase or after_write != before_write or after_full != before_full:
            changed_regions += 1

    _rebuild_page_masks(plan)
    plan.accepted = bool(plan.applied_regions) and cv2.countNonZero(plan.erase_mask) > 0 and cv2.countNonZero(plan.source_ink_mask) > 0
    if not plan.accepted:
        plan.reason = "no_accepted_regions_after_koharu_guard"
    plan.diagnostics["koharu_layout_guard"] = {
        **dict(evidence.diagnostics),
        "changed_regions": changed_regions,
        "rejected_regions": rejected_regions,
        "text_support_pixels": int(cv2.countNonZero(text_support)),
        "bubble_support_pixels": int(cv2.countNonZero(bubble_support)),
    }
    return plan

def apply_detector_policy_guard(
    plan: AlignedOverlayPlan,
    target: np.ndarray,
    bubble_cfg: BubbleConfig | None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> AlignedOverlayPlan:
    """Apply the selected primary detector before aligned reveal writes.

    Koharu primary keeps the full text/SFX/bubble semantic guard. Other primaries
    currently expose bubble geometry only, so they constrain full-raster/container
    writes without pretending to classify open text or panel artwork. This keeps
    the selected main detector first while avoiding hidden Koharu inference.
    """
    cfg = bubble_cfg or BubbleConfig()
    if koharu_is_primary(cfg):
        return apply_koharu_layout_guard(
            plan, target, cfg, stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=target_path, cache_enabled=cache_enabled,
        )

    stats = cache_stats if cache_stats is not None else {}
    try:
        rows = bubbles_cached(
            'target', target, [], target_path or '<aligned-target>', bubble_config=cfg,
            cache=stage_cache, cache_enabled=bool(cache_enabled and stage_cache is not None and target_path),
            stats=stats,
        )
    except Exception as exc:  # pragma: no cover - defensive fail-open
        plan.diagnostics['detector_policy_guard'] = {
            'primary': primary_detector(cfg), 'available': False, 'reason': 'error', 'error': str(exc),
        }
        return plan
    if not rows:
        plan.diagnostics['detector_policy_guard'] = {
            'primary': primary_detector(cfg), 'available': False, 'reason': 'no_primary_or_aux_bubbles',
        }
        return plan

    support = np.zeros(target.shape[:2], dtype=np.uint8)
    for row in rows:
        mask = getattr(row, 'mask', None)
        if mask is not None and mask.shape == support.shape:
            support = np.maximum(support, (mask > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(support) <= 0:
        plan.diagnostics['detector_policy_guard'] = {
            'primary': primary_detector(cfg), 'available': False, 'reason': 'empty_bubble_support',
        }
        return plan

    changed = 0
    for region in plan.regions:
        before_full = int(cv2.countNonZero(region.full_raster_mask))
        if before_full <= 0:
            continue
        overlap = _mask_overlap_ratio(region.full_raster_mask, support)
        region.diagnostics['detector_policy_guard'] = {
            'primary': primary_detector(cfg), 'bubble_overlap': overlap, 'available': True,
        }
        if overlap < 0.06:
            region.full_raster_mask[:] = 0
            changed += 1
        else:
            narrowed = cv2.bitwise_and(region.full_raster_mask, support)
            if int(cv2.countNonZero(narrowed)) != before_full:
                changed += 1
            region.full_raster_mask = narrowed

    _rebuild_page_masks(plan)
    plan.diagnostics['detector_policy_guard'] = {
        'primary': primary_detector(cfg),
        'available': True,
        'bubble_support_pixels': int(cv2.countNonZero(support)),
        'changed_regions': int(changed),
        'semantic_scope': 'bubble_only_non_koharu_primary',
    }
    return plan


def build_aligned_overlay_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayPlan:
    plan = _core.build_aligned_overlay_plan(source, target, registration, cfg)
    protect = _target_color_protect_mask(target, cfg)
    _strip_protected_pixels(plan, protect)
    plan.diagnostics["hard_color_protect_saturation"] = int(
        getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
    )
    return plan


def execute_aligned_overlay(
    plan: AlignedOverlayPlan,
    source: np.ndarray,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayResult:
    # Re-apply before execution in case a caller persisted or manually altered a
    # plan between build and execute.
    protect = _target_color_protect_mask(target, cfg)
    _strip_protected_pixels(plan, protect)
    result = _core.execute_aligned_overlay(plan, source, target, cfg)

    protected = protect > 0
    before = np.any(result.image != target, axis=2)
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
        "hard_color_restored_pixels": restored,
        "hard_color_protect_saturation": int(
            getattr(cfg, "hard_color_protect_saturation", _HARD_COLOR_SATURATION)
        ),
        "page_triage": "REVIEW" if nearly_unchanged else result.plan.page_triage,
    })
    return result


__all__ = [
    "AlignedOverlayRegion",
    "AlignedOverlayPlan",
    "AlignedOverlayResult",
    "apply_koharu_layout_guard", "apply_detector_policy_guard",
    "build_aligned_overlay_plan",
    "execute_aligned_overlay",
]
