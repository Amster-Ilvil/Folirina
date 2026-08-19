from __future__ import annotations

from typing import Any

HYBRID_SOURCE_INTEGRITY_BLOCK_REASONS = {
    'source_text_region_clipped_at_page_edge',
    'source_open_text_clipped_at_page_edge',
}


def hybrid_source_integrity_blocked(record: Any) -> bool:
    reason = str(getattr(record, 'reason', '') or '')
    review_reason = str(getattr(record, 'review_reason', '') or '')
    return bool(
        reason in HYBRID_SOURCE_INTEGRITY_BLOCK_REASONS
        or review_reason in HYBRID_SOURCE_INTEGRITY_BLOCK_REASONS
    )


def hybrid_mask_result_complete(record: Any) -> bool:
    """True only when the first-stage raster can suppress OCR/reletter fallback.

    An applied mask is not automatically complete. Hybrid should still reletter
    an applied region when the content audit reports missing source ink or target
    residuals. Physically cropped source material is handled separately by the
    integrity block and must never be guessed from OCR.
    """
    if not bool(getattr(record, 'applied', False)):
        return False
    if hybrid_source_integrity_blocked(record):
        return False
    checked = str(getattr(record, 'content_check', '') or '').startswith('checked')
    complete = bool(getattr(record, 'content_complete', False))
    return bool(checked and complete)
