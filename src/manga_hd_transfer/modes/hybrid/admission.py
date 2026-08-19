from __future__ import annotations

from typing import Any


def _confidence(row: Any) -> float:
    try:
        return float(getattr(row, 'confidence', 0.0) or 0.0)
    except Exception:
        return 0.0


def paired_visual_evidence_summary(paired_diff: Any, *, min_confidence: float = 0.55) -> dict:
    """Summarize OCR-independent evidence usable by Hybrid's first stage.

    Hybrid is mask-first. A failed/unavailable OCR backend must not cancel a page
    when Paired Diff already found corresponding translated/target regions.
    """
    if paired_diff is None:
        return {
            'available': False,
            'record_count': 0,
            'strong_record_count': 0,
            'source_bubbles': 0,
            'target_bubbles': 0,
            'method': '',
        }
    records = list(getattr(paired_diff, 'records', []) or [])
    src = list(getattr(paired_diff, 'source_bubbles', []) or [])
    dst = list(getattr(paired_diff, 'target_bubbles', []) or [])
    strong = sum(1 for row in records if _confidence(row) >= float(min_confidence))
    pair_count = min(len(src), len(dst))
    available = bool(pair_count > 0 and (strong > 0 or not records))
    return {
        'available': available,
        'record_count': len(records),
        'strong_record_count': strong,
        'source_bubbles': len(src),
        'target_bubbles': len(dst),
        'method': str(getattr(paired_diff, 'method', '') or ''),
    }


def should_continue_hybrid_after_ocr_negative(config: Any, paired_diff: Any) -> tuple[bool, dict]:
    hybrid = getattr(config, 'hybrid', None)
    enabled = bool(getattr(hybrid, 'continue_with_paired_visual_evidence_when_ocr_unavailable', True))
    summary = paired_visual_evidence_summary(paired_diff)
    return bool(enabled and summary.get('available')), summary
