from __future__ import annotations

"""Semantic authority boundary for destructive transfer completion.

This module owns only evidence collection/classification.  It never renders
pixels.  That makes the selected primary detector's authority explicit and
prevents auxiliary detectors from silently gaining destructive ownership.
"""

from dataclasses import dataclass
from typing import Any

from .detector_policy import koharu_is_primary, primary_detector
from .layout_evidence import collect_koharu_layout_evidence_cached, classify_layout_authority
from .models import BubbleInstance


@dataclass(frozen=True)
class TransferLayoutAuthorityState:
    source: Any | None
    target: Any | None
    status: str


def collect_transfer_layout_authority(
    *, config, source, target, stage_cache=None, cache_stats: dict | None = None,
    source_path: str | None = None, target_path: str | None = None,
) -> TransferLayoutAuthorityState:
    """Collect two-sided Koharu authority only when Koharu is the selected primary.

    Missing evidence stays fail-open for completion candidates, but an auxiliary
    Koharu detector never outranks the user's selected primary detector.
    """
    if not koharu_is_primary(config.bubbles):
        status = f"skipped_primary:{primary_detector(config.bubbles)}"
        if cache_stats is not None:
            cache_stats["transfer_layout_authority"] = status
        return TransferLayoutAuthorityState(None, None, status)

    layout_cache_enabled = bool(getattr(config.cache, "bubbles", True)) and bool(
        getattr(config.bubbles, "koharu_layout_cache_enabled", True)
    )
    source_evidence = collect_koharu_layout_evidence_cached(
        source, config.bubbles, role="transfer_authority_source", image_path=source_path,
        cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
    )
    target_evidence = collect_koharu_layout_evidence_cached(
        target, config.bubbles, role="transfer_authority_target", image_path=target_path,
        cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
    )
    status = (
        "two_sided" if source_evidence.available and target_evidence.available
        else "partial_fail_open" if source_evidence.available or target_evidence.available
        else "unavailable_fail_open"
    )
    if cache_stats is not None:
        cache_stats["transfer_layout_authority"] = status
    return TransferLayoutAuthorityState(source_evidence, target_evidence, status)


def filter_completion_pairs_by_layout_authority(
    source_rows: list[BubbleInstance],
    target_rows: list[BubbleInstance],
    *,
    source_evidence,
    target_evidence,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    cfg,
) -> tuple[list[BubbleInstance], list[BubbleInstance], list[dict]]:
    """Reject completion pairs positively classified as panel/artwork.

    UNKNOWN remains eligible because completion candidates are already
    conservative; PROTECT never reaches a destructive renderer.
    """
    kept_s: list[BubbleInstance] = []
    kept_t: list[BubbleInstance] = []
    audit: list[dict] = []
    for sb, tb in zip(source_rows or [], target_rows or []):
        sd = classify_layout_authority(source_evidence, sb, source_shape, region_kind="bubble", cfg=cfg)
        td = classify_layout_authority(target_evidence, tb, target_shape, region_kind="bubble", cfg=cfg)
        passed = sd.state != "PROTECT" and td.state != "PROTECT"
        decision = {
            "source_id": str(getattr(sb, "id", "")),
            "target_id": str(getattr(tb, "id", "")),
            "passed": bool(passed),
            "source": sd.to_dict(),
            "target": td.to_dict(),
            "reason": "koharu_authority_completion_allowed" if passed else "koharu_layout_panel_only_artwork",
        }
        for row, side in ((sb, sd), (tb, td)):
            meta = dict(getattr(row, "meta", {}) or {})
            meta["koharu_layout_authority"] = side.to_dict()
            row.meta = meta
        audit.append(decision)
        if passed:
            kept_s.append(sb)
            kept_t.append(tb)
    return kept_s, kept_t, audit


# Compatibility name used by older tests/plugins.
_filter_completion_pairs_by_koharu_authority = filter_completion_pairs_by_layout_authority

__all__ = [
    "TransferLayoutAuthorityState",
    "collect_transfer_layout_authority",
    "filter_completion_pairs_by_layout_authority",
    "_filter_completion_pairs_by_koharu_authority",
]
