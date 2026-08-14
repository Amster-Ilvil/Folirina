from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig, PipelineConfig
from manga_hd_transfer.mask_transfer import (
    MaskTransferRecord,
    _dominant_saturated_container_mask,
    _evaluate_content_completeness,
)
from manga_hd_transfer.models import PagePair, RegistrationResult
from manga_hd_transfer.qa import run_mask_replace_qa


def _registration() -> RegistrationResult:
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float32), method="test", confidence=0.99,
        inlier_ratio=1.0, reprojection_error=0.0, spatial_coverage=1.0,
        num_matches=100, source_size=(100, 100), target_size=(100, 100),
    )


def test_applied_is_not_content_complete() -> None:
    rec = MaskTransferRecord("src", "dst", 0.99, True, "applied")
    rec.content_check = "checked"
    rec.content_complete = False
    rec.source_ink_coverage = 0.97
    rec.target_residual_ratio = 0.42
    issues = run_mask_replace_qa(
        PagePair("src.png", "dst.png", 0, 0, 1.0, 0.0),
        _registration(), [], [], [rec], PipelineConfig().qa, MaskReplaceConfig(),
    )
    # Newer review-first policy publishes a reversible candidate and records this as a warning.
    assert any(x.code == "mask_replace_content_incomplete" and x.severity in {"warning", "error"} for x in issues)


def test_content_metric_detects_and_clears_target_residual() -> None:
    cfg = MaskReplaceConfig()
    src = np.zeros((80, 80), np.uint8)
    tgt = np.zeros((80, 80), np.uint8)
    cv2.rectangle(src, (14, 18), (19, 54), 255, -1)
    cv2.rectangle(tgt, (48, 18), (53, 54), 255, -1)

    final_bad = np.full((80, 80, 3), 255, np.uint8)
    final_bad[src > 0] = 0
    final_bad[tgt > 0] = 0
    bad = MaskTransferRecord("src", "dst", 1.0, True, "applied")
    _evaluate_content_completeness(bad, src, tgt, final_bad, cfg)
    assert bad.content_check == "checked"
    assert not bad.content_complete
    assert bad.target_residual_ratio > 0.9

    final_good = np.full((80, 80, 3), 255, np.uint8)
    final_good[src > 0] = 0
    good = MaskTransferRecord("src", "dst", 1.0, True, "applied")
    _evaluate_content_completeness(good, src, tgt, final_good, cfg)
    assert good.content_complete
    assert good.source_ink_coverage >= 0.99
    assert good.target_residual_ratio == 0.0


def test_saturated_route_requires_real_region_overlap() -> None:
    cfg = MaskReplaceConfig()
    target = np.full((240, 240, 3), 255, np.uint8)
    # Nearby blue artwork: saturated but not the text region itself.
    target[70:180, 150:225] = (220, 80, 30)
    region = np.zeros((240, 240), np.uint8)
    region[95:175, 80:145] = 255
    assert _dominant_saturated_container_mask(target, region, cfg) is None


def test_saturated_route_accepts_yellow_burst_container() -> None:
    cfg = MaskReplaceConfig()
    target = np.full((260, 260, 3), 255, np.uint8)
    pts = np.array([[40, 40], [220, 40], [205, 95], [235, 130], [205, 165], [220, 220], [40, 220], [55, 165], [25, 130], [55, 95]], np.int32)
    cv2.fillPoly(target, [pts], (0, 240, 255))  # BGR yellow
    # Structural candidate is the central changed-text island, not the full burst.
    region = np.zeros((260, 260), np.uint8)
    region[80:190, 80:180] = 255
    gate = _dominant_saturated_container_mask(target, region, cfg)
    assert gate is not None
    assert cv2.countNonZero(gate) > cv2.countNonZero(region)
