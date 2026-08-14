from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.aligned_overlay_reveal import (
    AlignedOverlayPlan,
    AlignedOverlayRegion,
    execute_aligned_overlay,
)
from manga_hd_transfer.config import PipelineConfig


def _region(
    erase: np.ndarray,
    source_write: np.ndarray,
    full: np.ndarray,
    *,
    composite_mode: str = "ink_only",
) -> AlignedOverlayRegion:
    h, w = erase.shape
    return AlignedOverlayRegion(
        id="colour_guard_regression",
        target_bbox=(0, 0, w, h),
        source_bbox=(0, 0, w, h),
        erase_mask=erase.copy(),
        source_ink_mask=source_write.copy(),
        full_raster_mask=full.copy(),
        composite_mode=composite_mode,
        triage="SAFE",
        reason="synthetic_text_proof",
        white_ratio=0.80,
        color_ratio=0.20,
        erase_area_ratio=float(np.count_nonzero(erase) / max(1, h * w)),
        source_ink_pixels=int(np.count_nonzero(source_write)),
        target_ink_pixels=int(np.count_nonzero(erase)),
        border_guard_px=2,
    )


def test_hard_color_guard_preserves_background_but_allows_proven_text_on_colour():
    """Colour is background authority, not an absolute no-write zone.

    Open manga lettering can sit directly over coloured art.  A Chinese glyph
    therefore has to replace some saturated TARGET pixels.  What must remain
    byte-stable is saturated background *outside* the already-proven ink masks.
    """
    h, w = 120, 160
    target = np.full((h, w, 3), 255, dtype=np.uint8)
    source = np.full_like(target, 255)

    # Coloured artwork.  The intended Chinese stroke crosses its centre while a
    # second coloured lobe is completely unrelated to text.
    cv2.circle(target, (82, 62), 18, (80, 145, 225), -1, cv2.LINE_AA)
    cv2.circle(target, (125, 82), 12, (190, 80, 210), -1, cv2.LINE_AA)

    # Japanese target ink on white paper plus SOURCE Chinese ink over colour.
    cv2.line(target, (25, 34), (58, 34), (15, 15, 15), 4, cv2.LINE_AA)
    cv2.line(source, (27, 46), (60, 46), (12, 12, 12), 4, cv2.LINE_AA)
    cv2.line(source, (70, 55), (94, 69), (10, 10, 10), 4, cv2.LINE_AA)

    erase = np.zeros((h, w), dtype=np.uint8)
    source_write = np.zeros((h, w), dtype=np.uint8)
    cv2.line(erase, (25, 34), (58, 34), 255, 6, cv2.LINE_AA)
    cv2.line(source_write, (27, 46), (60, 46), 255, 6, cv2.LINE_AA)
    cv2.line(source_write, (70, 55), (94, 69), 255, 6, cv2.LINE_AA)
    full = np.zeros_like(erase)

    region = _region(erase, source_write, full)
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
    changed = np.any(result.image != target, axis=2)
    text_support = (erase > 0) | (source_write > 0)

    # No saturated background pixel outside the proven text masks may change.
    assert int(np.count_nonzero(changed & coloured & ~text_support)) == 0
    # But the SOURCE glyph crossing colour must actually be allowed to appear.
    assert int(np.count_nonzero(changed & coloured & (source_write > 0))) > 0
    # The unrelated coloured lobe remains exactly TARGET.
    unrelated = np.zeros((h, w), np.uint8)
    cv2.circle(unrelated, (125, 82), 10, 255, -1)
    assert np.array_equal(result.image[unrelated > 0], target[unrelated > 0])

    assert int(result.diagnostics.get("hard_color_background_changed_after", -1)) == 0
    assert int(result.diagnostics.get("hard_color_text_release_pixels", 0)) > 0
    assert int(result.diagnostics.get("hard_color_saturated_changed_pixels", 0)) > 0
    assert result.diagnostics.get("hard_color_contract") == "target_background_authority_except_proven_text_masks"
    assert result.diagnostics.get("source_background_authority") is False


def test_hard_color_guard_never_releases_full_raster_over_colour():
    """Only ink-only masks may release colour; full SOURCE raster never may."""
    h, w = 100, 140
    target = np.full((h, w, 3), 250, dtype=np.uint8)
    source = np.full_like(target, 255)
    cv2.rectangle(target, (55, 25), (105, 75), (70, 150, 230), -1)
    cv2.rectangle(source, (55, 25), (105, 75), (255, 255, 255), -1)

    erase = np.zeros((h, w), np.uint8)
    cv2.line(target, (20, 20), (40, 20), (15, 15, 15), 3)
    cv2.line(erase, (20, 20), (40, 20), 255, 5)
    source_write = np.zeros_like(erase)
    full = np.zeros_like(erase)
    cv2.rectangle(full, (50, 20), (110, 80), 255, -1)

    region = _region(erase, source_write, full, composite_mode="full_raster_white")
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
    assert np.array_equal(result.image[coloured], target[coloured])
    assert int(result.diagnostics.get("hard_color_background_changed_after", -1)) == 0
