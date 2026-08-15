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

from .config import AlignedOverlayRevealConfig
from .models import RegistrationResult
from . import aligned_overlay_reveal_core as _core

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
    "build_aligned_overlay_plan",
    "execute_aligned_overlay",
]
