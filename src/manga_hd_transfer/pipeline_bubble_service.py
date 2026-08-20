from __future__ import annotations

"""Bubble detection and cache restoration service.

v2.0.91 exposes an explicit primary/auxiliary detector policy.  The selected
primary detector always runs first; auxiliaries are multi-select and are either
conditional or always-on according to the policy.  Primary detections keep
priority during deduplication.  When Koharu is the primary, its semantic
ALLOW/PROTECT/UNKNOWN map remains the hard authority for auxiliary candidates.
"""

from pathlib import Path
from typing import Any

import numpy as np

from .bubbles import (
    assign_blocks_to_bubbles,
    detect_koharu_layout_bubbles,
    detect_mangalens_bubbles,
    detect_seeded_white_bubbles,
    load_bubble_sidecar,
)
from .cache import PageStageCache, blocks_signature, image_stage_signature
from .config import MaskReplaceConfig
from .detector_policy import (
    bubble_auxiliary_backends,
    detector_strategy,
    primary_detector,
    should_run_auxiliaries,
    koharu_is_primary,
    uncovered_blocks,
    STRATEGY_ALWAYS,
)
from .layout_evidence import collect_koharu_layout_evidence_cached, filter_candidates_by_layout_authority
from .models import BubbleInstance


def _preferred_koharu_bubbles(
    role: str,
    image: np.ndarray,
    blocks,
    image_path: str | Path,
    *,
    bubble_config: Any,
    cache: PageStageCache | None,
    cache_enabled: bool,
    stats: dict[str, str] | None,
) -> list[BubbleInstance] | None:
    evidence = collect_koharu_layout_evidence_cached(
        image, bubble_config, role=f"bubbles_{role}", image_path=image_path,
        cache=cache, cache_enabled=cache_enabled, stats=stats, allow_missing=True,
    )
    if not evidence.available:
        return None
    rows = evidence.bubble_instances(backend_name="koharu_layout")
    if role == "source":
        for row in rows:
            row.meta["source_only"] = True; row.meta.pop("target_only", None)
    elif role == "target":
        for row in rows:
            row.meta["target_only"] = True; row.meta.pop("source_only", None)
    assign_blocks_to_bubbles(blocks, rows)
    return rows


def _retag_role(rows: list[BubbleInstance], role: str, blocks) -> list[BubbleInstance]:
    for row in rows:
        meta = dict(row.meta or {})
        if role == "source":
            meta["source_only"] = True; meta.pop("target_only", None)
        else:
            meta["target_only"] = True; meta.pop("source_only", None)
        row.meta = meta
    assign_blocks_to_bubbles(blocks, rows)
    return rows


def _detect_policy_provider(
    role: str,
    backend: str,
    image: np.ndarray,
    blocks,
    image_path: str | Path,
    bubble_config: Any,
    *,
    cache: PageStageCache | None,
    cache_enabled: bool,
    stats: dict[str, str] | None,
) -> list[BubbleInstance] | None:
    backend = str(backend or "").lower().strip()
    if backend == "koharu_layout":
        return _preferred_koharu_bubbles(
            role, image, blocks, image_path, bubble_config=bubble_config,
            cache=cache, cache_enabled=cache_enabled, stats=stats,
        )
    if backend in {"geometry_white", "seeded_white"}:
        # Keep the legacy adapter observable for plugins/tests when the stored
        # backend is the classic seeded-white detector.
        if str(getattr(bubble_config, "backend", "") or "").lower() == "seeded_white":
            return _retag_role(detect_bubbles(image, blocks, image_path, bubble_config), role, blocks)
        return _retag_role(detect_seeded_white_bubbles(image, blocks, bubble_config), role, blocks)
    if backend == "mangalens":
        return _retag_role(detect_mangalens_bubbles(image, blocks, bubble_config), role, blocks)
    if backend == "sidecar":
        return _retag_role(load_bubble_sidecar(image, image_path, blocks, bubble_config), role, blocks)
    if backend == "ysg_obb":
        from .source_detectors import detect_source_ysg_obb
        rows = [r for r in detect_source_ysg_obb(image, MaskReplaceConfig(), bubble_config, existing=[]) if str((r.meta or {}).get("region_kind") or "") == "bubble"]
        return _retag_role(rows, role, blocks)
    if backend == "rtdetr_v2":
        from .source_detectors import detect_source_rtdetr_v2
        rows = detect_source_rtdetr_v2(image, MaskReplaceConfig(), bubble_config, existing=[])
        return _retag_role(rows, role, blocks)
    if backend == "sam2":
        from .source_detectors import detect_source_sam2
        rows = detect_source_sam2(image, MaskReplaceConfig(), bubble_config, existing=[])
        return _retag_role(rows, role, blocks)
    return []


def detect_bubbles(
    image: np.ndarray,
    blocks,
    image_path: str | Path,
    bubble_config: Any,
) -> list[BubbleInstance]:
    """Legacy single-backend adapter retained for plugins and old callers."""
    backend = str(bubble_config.backend).lower()
    if backend == "none":
        return []
    if backend == "seeded_white":
        return detect_seeded_white_bubbles(image, blocks, bubble_config)
    if backend == "mangalens":
        return detect_mangalens_bubbles(image, blocks, bubble_config)
    if backend == "koharu_layout":
        return detect_koharu_layout_bubbles(image, blocks, bubble_config)
    if backend == "sidecar":
        return load_bubble_sidecar(image, image_path, blocks, bubble_config)
    if backend == "ysg_obb":
        from .source_detectors import detect_source_ysg_obb
        return [r for r in detect_source_ysg_obb(image, MaskReplaceConfig(), bubble_config, existing=[]) if str((r.meta or {}).get("region_kind") or "") == "bubble"]
    if backend == "rtdetr_v2":
        from .source_detectors import detect_source_rtdetr_v2
        return detect_source_rtdetr_v2(image, MaskReplaceConfig(), bubble_config, existing=[])
    if backend == "sam2":
        from .source_detectors import detect_source_sam2
        return detect_source_sam2(image, MaskReplaceConfig(), bubble_config, existing=[])
    raise ValueError(f"Unsupported bubble backend: {bubble_config.backend}")


def _mask_iou(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    aa = a > 0; bb = b > 0
    union = int(np.count_nonzero(aa | bb))
    return float(np.count_nonzero(aa & bb)) / max(1, union)


def primary_bubbles_cached(
    role: str,
    image: np.ndarray,
    image_path: str | Path,
    *,
    bubble_config: Any,
    cache: PageStageCache | None,
    cache_enabled: bool,
    stats: dict[str, str],
) -> list[BubbleInstance]:
    """Run/cache only the selected primary detector.

    This canonical primary stage lets page-flow establish the user's selected
    detector before Paired Diff/OCR/mode-specific discovery without also running
    auxiliary detectors.  Later bubble stages can reuse this result and merely
    attach OCR block relationships.
    """
    primary = primary_detector(bubble_config)
    cache_role = f"primary_{role}"
    cache_enabled = bool(cache_enabled and cache is not None and getattr(cache, "enabled", True))
    sig = None
    if cache_enabled and cache is not None:
        sig = image_stage_signature(
            image_path, bubble_config,
            {"role": role, "detector_policy_primary": primary, "primary_only_stage": True},
        )
        hit = cache.load_bubbles(cache_role, sig)
        if hit is not None:
            # v2.3.22: an empty Koharu primary-bubble cache must not outrank a
            # positive canonical LayoutEvidence cache for the exact same page.
            # A transient/older primary stage could persist [] while the shared
            # SOURCE/TARGET layout cache already contains valid bubble geometry.
            # Direct used to consume that poisoned empty result, drop to a partial
            # pseudo-text fallback, and leave Japanese pixels in otherwise known
            # text boxes. Reconcile only this contradictory state; a genuinely
            # textless page (LayoutEvidence also has zero bubbles) remains empty.
            if primary == "koharu_layout" and len(hit) == 0:
                preferred = _preferred_koharu_bubbles(
                    role, image, [], image_path, bubble_config=bubble_config,
                    cache=cache,
                    cache_enabled=cache_enabled and bool(getattr(bubble_config, "koharu_layout_cache_enabled", True)),
                    stats=stats,
                )
                preferred_rows = _retag_role(list(preferred or []), role, [])
                if preferred_rows:
                    cache.save_bubbles(cache_role, sig, preferred_rows)
                    stats[f"primary_detector_{role}"] = f"recovered_layout:{primary}:{len(preferred_rows)}"
                    stats[f"primary_detector_{role}_cache_reconciled"] = "empty_primary_to_positive_layout"
                    return preferred_rows
            stats[f"primary_detector_{role}"] = f"hit:{primary}:{len(hit)}"
            return _retag_role(hit, role, [])
    rows = _detect_policy_provider(
        role, primary, image, [], image_path, bubble_config,
        cache=cache,
        cache_enabled=cache_enabled and bool(getattr(bubble_config, "koharu_layout_cache_enabled", True)),
        stats=stats,
    )
    rows = _retag_role(list(rows or []), role, [])
    if cache_enabled and cache is not None and sig is not None:
        cache.save_bubbles(cache_role, sig, rows)
    stats[f"primary_detector_{role}"] = f"miss:{primary}:{len(rows)}"
    return rows


def bubbles_cached(
    role: str,
    image: np.ndarray,
    blocks,
    image_path: str | Path,
    *,
    bubble_config: Any,
    cache: PageStageCache,
    cache_enabled: bool,
    stats: dict[str, str],
) -> list[BubbleInstance]:
    primary = primary_detector(bubble_config)
    strategy = detector_strategy(bubble_config)
    aux = bubble_auxiliary_backends(bubble_config)
    cache_enabled = bool(cache_enabled and cache is not None and getattr(cache, "enabled", True))
    policy_payload = {"primary": primary, "strategy": strategy, "aux": aux}

    sig = None
    if cache_enabled:
        sig = image_stage_signature(
            image_path,
            bubble_config,
            {"role": role, "blocks": blocks_signature(blocks), "detector_policy": policy_payload},
        )
        hit = cache.load_bubbles(role, sig)
        if hit is not None:
            by_block_id = {block.id: block for block in blocks}
            for bubble in hit:
                for block_id in bubble.block_ids:
                    block = by_block_id.get(block_id)
                    if block is not None:
                        block.bubble_id = bubble.id
                        if block.kind == "unknown":
                            block.kind = bubble.kind
            # Preserve the legacy cache-status token for downstream UI/tests;
            # expose the richer policy separately.
            stats[f"bubbles_{role}"] = "hit_fallback_filtered"
            stats[f"bubbles_{role}_policy"] = f"{strategy}:{primary}"
            return hit

    primary_rows = primary_bubbles_cached(
        role, image, image_path, bubble_config=bubble_config, cache=cache,
        cache_enabled=cache_enabled, stats=stats,
    )
    primary_rows = _retag_role(list(primary_rows or []), role, blocks)
    accepted: list[BubbleInstance] = list(primary_rows)
    missing = uncovered_blocks(blocks, primary_rows)
    primary_sufficient = bool(primary_rows) and (not list(blocks) or missing == 0)
    stats[f"bubbles_{role}_primary"] = f"{primary}:{len(primary_rows)}:uncovered={missing}"

    if should_run_auxiliaries(bubble_config, primary_sufficient=primary_sufficient):
        authority = None
        if koharu_is_primary(bubble_config):
            authority = collect_koharu_layout_evidence_cached(
                image, bubble_config, role=f"bubble_aux_authority_{role}", image_path=image_path,
                cache=cache,
                cache_enabled=cache_enabled and bool(getattr(bubble_config, "koharu_layout_cache_enabled", True)),
                stats=stats, allow_missing=True,
            )
        for backend in aux:
            if backend == primary:
                continue
            rows = _detect_policy_provider(
                role, backend, image, blocks, image_path, bubble_config,
                cache=cache,
                cache_enabled=cache_enabled and bool(getattr(bubble_config, "koharu_layout_cache_enabled", True)),
                stats=stats,
            )
            rows = list(rows or [])
            authority_audit: list[dict] = []
            if authority is not None:
                rows, authority_audit = filter_candidates_by_layout_authority(
                    rows, authority, image.shape[:2], region_kind="bubble", cfg=None,
                    allow_unknown=True, meta_key="koharu_layout_authority",
                )
                stats[f"bubbles_{role}_authority"] = "koharu_first"
                stats[f"bubbles_{role}_authority_rejected"] = str(
                    int(stats.get(f"bubbles_{role}_authority_rejected", "0") or 0)
                    + sum(1 for x in authority_audit if not x.get("accepted"))
                )
            before = len(accepted)
            for row in sorted(rows, key=lambda x: float(x.confidence), reverse=True):
                if any(_mask_iou(row.mask, old.mask) >= 0.72 for old in accepted):
                    continue
                accepted.append(row)
            stats[f"bubbles_{role}_aux_{backend}"] = (
                f"added={len(accepted)-before};rejected={sum(1 for x in authority_audit if not x.get('accepted'))}"
            )
            if strategy != STRATEGY_ALWAYS and uncovered_blocks(blocks, accepted) == 0 and accepted:
                break

    if cache_enabled and sig is not None:
        cache.save_bubbles(role, sig, accepted)
    stats[f"bubbles_{role}"] = f"miss_policy:{strategy}:{primary}:{len(accepted)}"
    return accepted


__all__ = ["detect_bubbles", "primary_bubbles_cached", "bubbles_cached"]
