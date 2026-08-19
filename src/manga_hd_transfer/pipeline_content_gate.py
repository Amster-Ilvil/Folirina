from __future__ import annotations

from .config import PipelineConfig
from .transfer_policy import _has_transferable_source_text


def source_has_no_transferable_text(
    *,
    config: PipelineConfig,
    source_backend,
    source_blocks,
    source_bubbles,
    paired_diff,
    cache_stats: dict,
) -> bool:
    """Return strong OCR-backed negative evidence; ambiguous full-page OCR fails open."""
    ocr_route = str(cache_stats.get("ocr_source", ""))
    configured_source_ocr = str(config.ocr.source_backend or config.ocr.backend or "").strip().lower()
    source_evidence_available = (
        source_backend is not None
        and configured_source_ocr not in {"", "none", "null"}
        and not ocr_route.startswith("skipped")
    )
    real_source_blocks = [b for b in source_blocks if str(getattr(b, "text", "")).strip()]
    region_text_only = bool(source_backend is not None and getattr(source_backend, "region_text_only", False))
    if not source_evidence_available:
        return False
    if not real_source_blocks:
        return True
    if region_text_only:
        return not _has_transferable_source_text(
            source_blocks, source_bubbles, config.mask_replace.enabled_kinds,
            paired_diff.source_bubbles if paired_diff is not None else None,
        )
    # Full-page OCR with text but no reconstructed bubble is ambiguous. Fail open
    # rather than silently skipping a real translation because detection missed an
    # unusual/open container.
    return False
