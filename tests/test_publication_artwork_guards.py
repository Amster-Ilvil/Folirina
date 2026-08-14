from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.bubbles import _registered_ink_change_evidence
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.paired_diff import _supplement_ink_identity_evidence


def _mask(size: int = 96) -> np.ndarray:
    m = np.zeros((size, size), np.uint8)
    m[12:-12, 12:-12] = 255
    return m


def test_unseeded_white_gate_rejects_same_artwork_ink() -> None:
    cfg = PipelineConfig().mask_replace
    s = np.full((96, 96), 255, np.uint8)
    t = np.full((96, 96), 245, np.uint8)
    # Same white-clothing / line-art geometry after registration.
    cv2.line(s, (25, 20), (70, 75), 0, 3)
    cv2.line(t, (25, 20), (70, 75), 0, 3)
    cv2.circle(s, (55, 45), 10, 0, 2)
    cv2.circle(t, (55, 45), 10, 0, 2)
    ev = _registered_ink_change_evidence(s, t, _mask(), cfg)
    assert ev["valid"]
    assert ev["ink_identity_overlap"] > 0.90
    assert ev["passed"] is False


def test_unseeded_white_gate_accepts_translated_ink_geometry() -> None:
    cfg = PipelineConfig().mask_replace
    s = np.full((96, 96), 255, np.uint8)
    t = np.full((96, 96), 255, np.uint8)
    # Comparable amount of ink, but clearly different glyph geometry.
    for x in (28, 42, 56, 70):
        cv2.rectangle(s, (x, 24), (x + 5, 68), 0, -1)
    for y in (28, 42, 56, 70):
        cv2.rectangle(t, (24, y), (68, y + 5), 0, -1)
    ev = _registered_ink_change_evidence(s, t, _mask(), cfg)
    assert ev["source_ink_density_registered"] >= cfg.rigid_container_unseeded_min_source_ink_density
    assert ev["target_ink_density_registered"] >= cfg.rigid_container_unseeded_min_target_ink_density
    assert ev["ink_change_score"] >= cfg.rigid_container_unseeded_min_ink_change_score
    assert ev["passed"] is True


def test_unseeded_white_gate_rejects_extreme_density_imbalance() -> None:
    cfg = PipelineConfig().mask_replace
    s = np.full((96, 96), 255, np.uint8)
    t = np.full((96, 96), 255, np.uint8)
    cv2.rectangle(s, (42, 35), (48, 60), 0, -1)
    for y in range(20, 78, 7):
        for x in range(20, 78, 7):
            cv2.circle(t, (x, y), 2, 0, -1)
    ev = _registered_ink_change_evidence(s, t, _mask(), cfg)
    assert ev["ink_density_ratio"] > cfg.rigid_container_unseeded_max_ink_density_ratio
    assert ev["passed"] is False


def test_structural_supplement_rejects_same_artwork_texture() -> None:
    cfg = PipelineConfig().mask_replace
    s = np.full((96, 96, 3), 245, np.uint8)
    t = np.full((96, 96, 3), 220, np.uint8)
    for x in range(20, 80, 9):
        cv2.line(s, (x, 22), (x + 8, 70), (30, 30, 30), 2)
        cv2.line(t, (x, 22), (x + 8, 70), (30, 30, 30), 2)
    ev = _supplement_ink_identity_evidence(s, t, _mask(), "complex_text", cfg)
    assert ev["enabled"]
    assert ev["ink_change_score"] < cfg.paired_diff_supplement_complex_min_ink_change_score
    assert ev["passed"] is False


def test_structural_free_text_rejects_sparse_source_header_noise() -> None:
    cfg = PipelineConfig().mask_replace
    s = np.full((96, 96, 3), 255, np.uint8)
    t = np.full((96, 96, 3), 255, np.uint8)
    # Tiny SOURCE marks vs a real TARGET-only header: high change alone is not
    # enough evidence that SOURCE contains a translation worth transferring.
    cv2.circle(s, (30, 30), 1, (0, 0, 0), -1)
    cv2.circle(s, (40, 40), 1, (0, 0, 0), -1)
    cv2.putText(t, "ABC", (20, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    ev = _supplement_ink_identity_evidence(s, t, _mask(), "free_text", cfg)
    assert ev["source_ink_density"] < cfg.paired_diff_supplement_free_min_source_ink_density
    assert ev["passed"] is False
