from __future__ import annotations

import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig, QAConfig
from manga_hd_transfer.mask_transfer import MaskTransferRecord
from manga_hd_transfer.models import BubbleInstance, PagePair, RegistrationResult
from manga_hd_transfer.qa import run_mask_replace_qa, qa_summary


def _pair() -> PagePair:
    return PagePair('s.png', 't.png', 0, 0, confidence=0.9, score=0.9)


def _registration() -> RegistrationResult:
    return RegistrationResult(
        matrix=np.eye(3, dtype=float),
        method='unit-test',
        confidence=0.95,
        inlier_ratio=1.0,
        reprojection_error=0.0,
        spatial_coverage=1.0,
        num_matches=10,
        source_size=(100, 100),
        target_size=(100, 100),
    )


def _bubble(bid: str) -> BubbleInstance:
    mask = np.zeros((8, 8), np.uint8)
    mask[1:7, 1:7] = 255
    return BubbleInstance(id=bid, polygon=[(1, 1), (6, 1), (6, 6), (1, 6)], mask=mask, safe_mask=mask.copy(), block_ids=['blk'])


def test_rigid_container_safe_patch_uses_relaxed_coverage_and_spill() -> None:
    rec = MaskTransferRecord(
        source_bubble_id='s1',
        target_bubble_id='t1',
        confidence=0.95,
        applied=True,
        reason='applied_rigid_container_raster',
        geometry_mode='rigid_uniform_container',
        clarity_mode='locked-source-container-patch',
        mask_iou=0.93,
        target_coverage=0.936,
        spill_ratio=0.063,
        content_check='checked_ok',
        content_complete=True,
    )
    issues = run_mask_replace_qa(_pair(), _registration(), [], [_bubble('s1')], [rec], QAConfig(), MaskReplaceConfig())
    codes = {i.code for i in issues}
    assert 'mask_replace_low_coverage' not in codes
    assert 'mask_replace_spill' not in codes
    assert qa_summary(issues)['errors'] == 0


def test_edge_clipped_applied_candidate_is_review_warning_not_error() -> None:
    rec = MaskTransferRecord(
        source_bubble_id='s1',
        target_bubble_id='t1',
        confidence=0.71,
        applied=True,
        reason='applied_low_confidence_candidate',
        review_required=True,
        review_reason='source_text_region_clipped_at_page_edge',
        geometry_mode='photo_pair',
        clarity_mode='photo-crisp-ink',
        mask_iou=1.0,
        target_coverage=1.0,
        spill_ratio=0.0,
        content_check='checked_ok',
        content_complete=True,
    )
    issues = run_mask_replace_qa(_pair(), _registration(), [], [_bubble('s1')], [rec], QAConfig(), MaskReplaceConfig())
    candidate = [i for i in issues if i.code == 'mask_replace_low_confidence_candidate']
    assert len(candidate) == 1
    assert candidate[0].severity == 'warning'
    assert qa_summary(issues)['errors'] == 0
