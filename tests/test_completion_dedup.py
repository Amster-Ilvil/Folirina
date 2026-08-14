from __future__ import annotations

import numpy as np

from manga_hd_transfer.mask_transfer import MaskTransferRecord, MaskTransferResult
from manga_hd_transfer.models import BubbleInstance
from manga_hd_transfer.pipeline import _completion_existing_target_bubbles, _mask_transfer_completion_needed


def _bubble(bid: str) -> BubbleInstance:
    mask = np.zeros((8, 8), np.uint8)
    mask[1:7, 1:7] = 255
    return BubbleInstance(id=bid, polygon=[(1, 1), (6, 1), (6, 6), (1, 6)], mask=mask, safe_mask=mask.copy())


def test_completion_needed_false_when_all_records_are_applied_and_safe() -> None:
    result = MaskTransferResult(
        image=np.zeros((4, 4, 3), np.uint8),
        layer_rgba=np.zeros((4, 4, 4), np.uint8),
        composite_mask=np.zeros((4, 4), np.uint8),
        matches=[],
        records=[MaskTransferRecord(source_bubble_id='s1', target_bubble_id='t1', confidence=1.0, applied=True, reason='ok', content_complete=True)],
    )
    assert _mask_transfer_completion_needed(result) is False


def test_completion_existing_target_bubbles_returns_only_safe_unique_targets() -> None:
    result = MaskTransferResult(
        image=np.zeros((4, 4, 3), np.uint8),
        layer_rgba=np.zeros((4, 4, 4), np.uint8),
        composite_mask=np.zeros((4, 4), np.uint8),
        matches=[],
        records=[
            MaskTransferRecord(source_bubble_id='s1', target_bubble_id='t1', confidence=1.0, applied=True, reason='ok', content_complete=True),
            MaskTransferRecord(source_bubble_id='s2', target_bubble_id='t2', confidence=1.0, applied=True, reason='review', review_required=True, content_complete=False),
            MaskTransferRecord(source_bubble_id='s3', target_bubble_id='t3', confidence=1.0, applied=False, reason='reject'),
        ],
    )
    group_a = [_bubble('t1'), _bubble('t2')]
    group_b = [_bubble('t1'), _bubble('t3')]
    existing = _completion_existing_target_bubbles(result, group_a, group_b)
    assert [b.id for b in existing] == ['t1']
