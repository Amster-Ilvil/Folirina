from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.aligned_overlay_reveal import (
    AlignedOverlayPlan,
    AlignedOverlayRegion,
    _rescue_empty_refined_regions,
)
from manga_hd_transfer.config import PipelineConfig


def test_empty_refined_region_gets_local_exclusive_review_rescue():
    """A core-identified text region may recover from an over-strong line guard.

    The rescue is intentionally not a new detector: it receives an existing
    REJECT/empty_refined_mask bbox, requires both SOURCE and TARGET exclusive ink,
    stays inside that bbox, and returns REVIEW rather than SAFE.
    """
    h, w = 120, 150
    source = np.full((h, w, 3), 248, np.uint8)
    target = source.copy()

    # Common dark structure that can cause the normal progressive guard to become
    # conservative around the candidate.
    cv2.line(source, (25, 28), (112, 28), (25, 25, 25), 2)
    cv2.line(target, (25, 28), (112, 28), (25, 25, 25), 2)
    cv2.line(source, (25, 28), (25, 96), (25, 25, 25), 2)
    cv2.line(target, (25, 28), (25, 96), (25, 25, 25), 2)

    # Different SOURCE Chinese-like and TARGET Japanese-like compact strokes.
    for x in (47, 59):
        cv2.line(source, (x, 45), (x, 76), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(source, (x - 6, 60), (x + 6, 60), (10, 10, 10), 3, cv2.LINE_AA)
    for x in (76, 88):
        cv2.line(target, (x, 43), (x, 78), (10, 10, 10), 4, cv2.LINE_AA)
        cv2.line(target, (x - 6, 54), (x + 6, 54), (10, 10, 10), 3, cv2.LINE_AA)

    empty = np.zeros((h, w), np.uint8)
    region = AlignedOverlayRegion(
        id="aligned_rescue",
        target_bbox=(30, 32, 105, 88),
        source_bbox=(30, 32, 105, 88),
        erase_mask=empty.copy(),
        source_ink_mask=empty.copy(),
        full_raster_mask=empty.copy(),
        composite_mode="ink_only",
        triage="REJECT",
        reason="empty_refined_mask",
        white_ratio=0.70,
        color_ratio=0.0,
        erase_area_ratio=0.0,
        source_ink_pixels=1,
        target_ink_pixels=0,
        border_guard_px=5,
        diagnostics={"outer_dark_ratio": 0.35},
    )
    plan = AlignedOverlayPlan(
        accepted=False,
        reason="no_accepted_regions",
        aligned_source=source.copy(),
        valid_mask=np.full((h, w), 255, np.uint8),
        erase_mask=empty.copy(),
        source_ink_mask=empty.copy(),
        full_raster_mask=empty.copy(),
        regions=[region],
        diagnostics={
            "manual_effect_candidates": [
                {
                    "id": region.id,
                    "reason": "aligned_overlay:empty_refined_mask",
                    "triage": "REJECT",
                }
            ]
        },
    )

    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    _rescue_empty_refined_regions(plan, target, cfg)

    assert plan.accepted is True
    assert plan.reason == "ok"
    assert region.triage == "REVIEW"
    assert region.reason == "empty_mask_local_exclusive_rescue"
    assert region.diagnostics.get("local_exclusive_rescue") is True
    assert region.source_ink_pixels >= 5
    assert region.target_ink_pixels >= 5
    assert cv2.countNonZero(plan.source_ink_mask) >= 5
    assert cv2.countNonZero(plan.erase_mask) >= 5
    assert plan.diagnostics.get("rescued_empty_region_count") == 1
    assert plan.diagnostics.get("rescued_empty_regions") == [region.id]
    candidate = plan.diagnostics["manual_effect_candidates"][0]
    assert candidate["triage"] == "REVIEW"
    assert candidate["reason"] == "aligned_overlay:empty_mask_local_exclusive_rescue"

    # Hard localisation invariant: no rescued write/erase pixel may escape bbox.
    outside = np.ones((h, w), bool)
    outside[32:88, 30:105] = False
    assert int(np.count_nonzero(plan.source_ink_mask[outside])) == 0
    assert int(np.count_nonzero(plan.erase_mask[outside])) == 0
