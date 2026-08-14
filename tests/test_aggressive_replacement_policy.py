from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.mask_transfer import _rigid_container_pair_eligible, transfer_bubble_patches
from manga_hd_transfer.models import BubbleInstance, RegistrationResult


def _bubble(mask: np.ndarray, bid: str) -> BubbleInstance:
    ys, xs = np.where(mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return BubbleInstance(
        id=bid,
        polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
        confidence=0.55,
        kind="speech",
        block_ids=[f"{bid}-text"],
        mask=mask,
        safe_mask=mask.copy(),
        meta={},
    )


def test_safety_off_keeps_colored_target_on_target_aware_route() -> None:
    cfg = PipelineConfig().mask_replace
    source = np.full((100, 100, 3), 255, np.uint8)
    target = np.full((100, 100, 3), (80, 80, 235), np.uint8)
    sm = np.zeros((100,100), np.uint8); sm[20:80,20:80] = 255
    tm = sm.copy()
    cv2.putText(source, "CN", (28,60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(target, "JP", (28,60), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,0), 2, cv2.LINE_AA)
    ok, diag = _rigid_container_pair_eligible(source, target, sm, tm, cfg)
    assert ok is False
    assert diag["reason"] == "requires_target_aware_colored_transfer"


def test_mask_aggressive_mode_writes_despite_former_quality_thresholds() -> None:
    source = np.full((96,96,3), 255, np.uint8)
    target = np.full((96,96,3), 255, np.uint8)
    cv2.putText(source, "C", (18,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
    cv2.putText(target, "J", (40,65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,0), 3, cv2.LINE_AA)
    sm = np.zeros((96,96), np.uint8); sm[10:55,10:55] = 255
    tm = np.zeros((96,96), np.uint8); tm[30:80,30:80] = 255
    sb, tb = _bubble(sm, "s"), _bubble(tm, "t")
    reg = RegistrationResult(
        matrix=np.eye(3), method="identity", confidence=.90, inlier_ratio=.90,
        reprojection_error=.2, spatial_coverage=.8, num_matches=30,
        source_size=(96,96), target_size=(96,96), diagnostics={},
    )
    cfg = PipelineConfig().mask_replace
    cfg.publication_safety_enabled = False
    cfg.local_fit = "global"
    cfg.min_match_confidence = .99
    cfg.min_mask_iou = .99
    cfg.min_target_coverage = .99
    cfg.max_spill_ratio = 0.0
    out = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    assert out.records
    assert any(r.applied for r in out.records)


def test_legacy_safety_flag_no_longer_restores_mask_blocking() -> None:
    source = np.full((96,96,3), 255, np.uint8)
    target = np.full((96,96,3), 255, np.uint8)
    sm = np.zeros((96,96), np.uint8); sm[10:55,10:55] = 255
    tm = np.zeros((96,96), np.uint8); tm[30:80,30:80] = 255
    sb, tb = _bubble(sm, "s"), _bubble(tm, "t")
    reg = RegistrationResult(
        matrix=np.eye(3), method="identity", confidence=.90, inlier_ratio=.90,
        reprojection_error=.2, spatial_coverage=.8, num_matches=30,
        source_size=(96,96), target_size=(96,96), diagnostics={},
    )
    cfg = PipelineConfig().mask_replace
    cfg.publication_safety_enabled = True  # legacy value is intentionally ignored
    cfg.local_fit = "global"
    cfg.min_match_confidence = .99
    out = transfer_bubble_patches(source, target, [sb], [tb], reg, cfg)
    assert out.records
    assert any(r.applied for r in out.records)
