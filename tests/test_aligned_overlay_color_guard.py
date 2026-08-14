from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.aligned_overlay_reveal import (
    AlignedOverlayPlan,
    AlignedOverlayRegion,
    execute_aligned_overlay,
)
from manga_hd_transfer.config import PipelineConfig


def test_hard_color_guard_restores_small_coloured_art_inside_white_region():
    """A mostly-white candidate must never edit a small coloured face/art detail.

    The real GUI page regression exposed this exact failure mode: region-level
    colour ratios passed because the surrounding balloon was white, while a few
    saturated pixels near a character mouth still entered erase/source masks.
    This test intentionally feeds an over-broad accepted plan to the public
    executor and verifies the page-level TARGET authority guard wins.
    """
    h, w = 120, 160
    target = np.full((h, w, 3), 255, dtype=np.uint8)
    source = np.full_like(target, 255)

    # Small coloured artwork embedded next to an otherwise white text area.
    cv2.circle(target, (82, 62), 15, (80, 145, 225), -1, cv2.LINE_AA)
    cv2.circle(target, (82, 62), 6, (25, 25, 25), 2, cv2.LINE_AA)

    # Japanese-ish target ink to erase and different SOURCE ink to reveal.
    cv2.line(target, (25, 34), (58, 34), (15, 15, 15), 4, cv2.LINE_AA)
    cv2.line(source, (27, 46), (60, 46), (12, 12, 12), 4, cv2.LINE_AA)
    # Deliberately dangerous SOURCE stroke crossing the coloured artwork. The
    # hard guard must remove it even though the plan is already accepted/SAFE.
    cv2.line(source, (70, 55), (94, 69), (10, 10, 10), 4, cv2.LINE_AA)

    erase = np.zeros((h, w), dtype=np.uint8)
    source_write = np.zeros((h, w), dtype=np.uint8)
    cv2.rectangle(erase, (20, 28), (64, 40), 255, -1)
    cv2.circle(erase, (82, 62), 10, 255, -1)
    cv2.rectangle(source_write, (22, 42), (65, 50), 255, -1)
    cv2.circle(source_write, (82, 62), 10, 255, -1)
    full = np.zeros_like(erase)

    region = AlignedOverlayRegion(
        id="colour_guard_regression",
        target_bbox=(16, 24, 102, 80),
        source_bbox=(16, 24, 102, 80),
        erase_mask=erase.copy(),
        source_ink_mask=source_write.copy(),
        full_raster_mask=full.copy(),
        composite_mode="ink_only",
        triage="SAFE",
        reason="synthetic_overbroad_plan",
        white_ratio=0.95,
        color_ratio=0.03,
        erase_area_ratio=float(np.count_nonzero(erase) / (h * w)),
        source_ink_pixels=int(np.count_nonzero(source_write)),
        target_ink_pixels=int(np.count_nonzero(erase)),
        border_guard_px=2,
    )
    plan = AlignedOverlayPlan(
        accepted=True,
        reason="ok",
        aligned_source=source.copy(),
        valid_mask=np.full((h, w), 255, dtype=np.uint8),
        erase_mask=erase.copy(),
        source_ink_mask=source_write.copy(),
        full_raster_mask=full.copy(),
        regions=[region],
        diagnostics={},
    )

    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    result = execute_aligned_overlay(plan, source, target, cfg)

    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    coloured = hsv[..., 1] >= 20
    assert int(np.count_nonzero(coloured)) > 0
    assert np.array_equal(result.image[coloured], target[coloured])

    # The guard must not turn the whole operation into a no-op: ordinary white
    # dialogue pixels still change while colour pixels are protected/restored.
    changed = np.any(result.image != target, axis=2)
    assert int(np.count_nonzero(changed & ~coloured)) > 0
    assert int(np.count_nonzero(changed & coloured)) == 0
    assert int(result.diagnostics.get("hard_color_protected_pixels", 0)) >= int(np.count_nonzero(coloured))
    assert result.diagnostics.get("source_background_authority") is False
