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
    white_ratio: float = 0.80,
    color_ratio: float = 0.20,
    bbox: tuple[int, int, int, int] | None = None,
    region_id: str = "colour_guard_regression",
) -> AlignedOverlayRegion:
    h, w = erase.shape
    if bbox is None:
        bbox = (0, 0, w, h)
    return AlignedOverlayRegion(
        id=region_id,
        target_bbox=bbox,
        source_bbox=bbox,
        erase_mask=erase.copy(),
        source_ink_mask=source_write.copy(),
        full_raster_mask=full.copy(),
        composite_mode=composite_mode,
        triage="SAFE",
        reason="synthetic_text_proof",
        white_ratio=white_ratio,
        color_ratio=color_ratio,
        erase_area_ratio=float(np.count_nonzero(erase) / max(1, h * w)),
        source_ink_pixels=int(np.count_nonzero(source_write)),
        target_ink_pixels=int(np.count_nonzero(erase)),
        border_guard_px=2,
    )


def test_hard_color_guard_preserves_background_but_allows_container_proven_text_on_colour():
    """Paper/container evidence may release exact text pixels over colour bleed."""
    h, w = 120, 160
    target = np.full((h, w, 3), 255, dtype=np.uint8)
    source = np.full_like(target, 255)

    cv2.circle(target, (82, 62), 18, (80, 145, 225), -1, cv2.LINE_AA)
    cv2.circle(target, (125, 82), 12, (190, 80, 210), -1, cv2.LINE_AA)

    cv2.line(target, (25, 34), (58, 34), (15, 15, 15), 4, cv2.LINE_AA)
    cv2.line(source, (27, 46), (60, 46), (12, 12, 12), 4, cv2.LINE_AA)
    cv2.line(source, (70, 55), (94, 69), (10, 10, 10), 4, cv2.LINE_AA)

    erase = np.zeros((h, w), dtype=np.uint8)
    source_write = np.zeros((h, w), dtype=np.uint8)
    cv2.line(erase, (25, 34), (58, 34), 255, 6, cv2.LINE_AA)
    cv2.line(source_write, (27, 46), (60, 46), 255, 6, cv2.LINE_AA)
    cv2.line(source_write, (70, 55), (94, 69), 255, 6, cv2.LINE_AA)
    full = np.zeros_like(erase)

    region = _region(erase, source_write, full, white_ratio=0.80)
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

    assert int(np.count_nonzero(changed & coloured & ~text_support)) == 0
    assert int(np.count_nonzero(changed & coloured & (source_write > 0))) > 0
    unrelated = np.zeros((h, w), np.uint8)
    cv2.circle(unrelated, (125, 82), 10, 255, -1)
    assert np.array_equal(result.image[unrelated > 0], target[unrelated > 0])

    assert int(result.diagnostics.get("hard_color_background_changed_after", -1)) == 0
    assert int(result.diagnostics.get("hard_color_text_release_pixels", 0)) > 0
    assert int(result.diagnostics.get("hard_color_saturated_changed_pixels", 0)) > 0
    assert result.diagnostics.get("hard_color_contract") == "target_background_authority_with_evidence_gated_text_release"
    assert region.id in result.diagnostics.get("hard_color_release_region_ids", [])
    assert result.diagnostics.get("source_background_authority") is False


def test_tiny_high_colour_detail_is_withheld_even_if_core_called_it_ink_only():
    """Small mouth/eye/hair differences must not become open-text releases.

    The real GUI page exposed a 23x25 face region with about 30 changed-ink
    pixels.  Cross-rendition exclusivity alone was enough to repaint the mouth.
    This regression deliberately feeds that shape as an accepted ink-only region
    and requires the colour guard to keep TARGET byte-stable.
    """
    h, w = 100, 120
    target = np.full((h, w, 3), (210, 220, 235), np.uint8)
    source = target.copy()
    cv2.circle(target, (60, 52), 13, (165, 120, 210), -1, cv2.LINE_AA)
    cv2.circle(source, (60, 52), 13, (165, 120, 210), -1, cv2.LINE_AA)
    cv2.rectangle(target, (56, 49), (63, 54), (70, 35, 80), -1)
    cv2.rectangle(source, (57, 50), (64, 55), (45, 45, 45), -1)

    erase = np.zeros((h, w), np.uint8)
    source_write = np.zeros((h, w), np.uint8)
    cv2.rectangle(erase, (56, 49), (63, 54), 255, -1)
    cv2.rectangle(source_write, (57, 50), (64, 55), 255, -1)
    full = np.zeros_like(erase)
    bbox = (49, 41, 72, 66)  # 23x25, mirrors the real false-positive scale.

    region = _region(
        erase, source_write, full,
        white_ratio=0.24,
        color_ratio=0.255,
        bbox=bbox,
        region_id="tiny_face_detail",
    )
    plan = AlignedOverlayPlan(
        accepted=True,
        reason="ok",
        aligned_source=source.copy(),
        valid_mask=np.full((h, w), 255, np.uint8),
        erase_mask=erase.copy(),
        source_ink_mask=source_write.copy(),
        full_raster_mask=full.copy(),
        regions=[region],
        diagnostics={},
    )
    cfg = PipelineConfig().aligned_overlay_reveal.model_copy(deep=True)
    result = execute_aligned_overlay(plan, source, target, cfg)

    assert np.array_equal(result.image[41:66, 49:72], target[41:66, 49:72])
    assert int(result.diagnostics.get("hard_color_text_release_pixels", -1)) == 0
    assert int(result.diagnostics.get("hard_color_saturated_changed_pixels", -1)) == 0
    assert "tiny_face_detail" in result.diagnostics.get("hard_color_withheld_region_ids", [])


def test_hard_color_guard_never_releases_full_raster_over_colour():
    """Only evidence-gated ink-only masks may release colour; full raster never may."""
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
