from __future__ import annotations

"""Cache-aware paired-difference geometry stage."""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from .cache import PageStageCache, paired_diff_stage_signature
from .geometry import rasterize_polygon
from .layout_evidence import collect_koharu_layout_evidence_cached
from .detector_policy import koharu_is_primary, primary_detector
from .models import PagePair
from .paired_diff import extract_paired_diff_bubbles

logger = logging.getLogger(__name__)


def _mask_for_bubble(bubble, shape: tuple[int, int]) -> np.ndarray:
    mask = getattr(bubble, "mask", None)
    if mask is not None and tuple(mask.shape[:2]) == tuple(shape):
        return (mask > 0).astype(np.uint8)
    return (rasterize_polygon(getattr(bubble, "polygon", []) or [], shape) > 0).astype(np.uint8)


def _max_layout_overlap(mask: np.ndarray, items) -> float:
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return 0.0
    best = 0.0
    for item in items:
        im = (item.mask > 0)
        if im.shape != mask.shape:
            continue
        ia = int(np.count_nonzero(im))
        if ia <= 0:
            continue
        inter = int(np.count_nonzero((mask > 0) & im))
        best = max(best, float(inter / max(1, min(area, ia))))
    return best


def _annotate_side_with_layout(bubbles, evidence, shape: tuple[int, int]) -> dict[str, int]:
    if evidence is None or not evidence.available:
        return {"count": len(bubbles or []), "supported": 0, "unsupported": 0}
    bubble_items = evidence.by_label("bubble")
    text_items = evidence.by_label("text")
    sfx_items = evidence.by_label("sfx")
    panel_items = evidence.by_label("panel")
    supported = 0
    for bubble in bubbles or []:
        mask = _mask_for_bubble(bubble, shape)
        bubble_overlap = _max_layout_overlap(mask, bubble_items)
        text_overlap = _max_layout_overlap(mask, text_items)
        sfx_overlap = _max_layout_overlap(mask, sfx_items)
        panel_overlap = _max_layout_overlap(mask, panel_items)
        is_supported = bool(bubble_overlap >= 0.25 or text_overlap >= 0.35 or sfx_overlap >= 0.35)
        meta = dict(getattr(bubble, "meta", {}) or {})
        meta["koharu_layout_evidence"] = {
            "supported": is_supported,
            "bubble_overlap": round(bubble_overlap, 4),
            "text_overlap": round(text_overlap, 4),
            "sfx_overlap": round(sfx_overlap, 4),
            "panel_overlap": round(panel_overlap, 4),
        }
        bubble.meta = meta
        supported += int(is_supported)
    count = len(bubbles or [])
    return {"count": count, "supported": supported, "unsupported": max(0, count - supported)}


def _semantic_layout_support(bubble, region_kind: str, cfg: Any) -> tuple[bool, str, dict[str, float]]:
    """Return region-kind-aware Koharu support for one paired-diff proposal.

    Koharu Layout is positive semantic evidence, not an artwork detector.  The
    safety gate therefore asks a stricter question than the generic annotation:
    closed speech containers need bubble/text evidence, while open/free/complex
    text needs actual text/SFX evidence.  A panel hit alone is explicitly *not*
    transfer authority.
    """
    ev = dict((getattr(bubble, "meta", {}) or {}).get("koharu_layout_evidence") or {})
    bubble_overlap = float(ev.get("bubble_overlap", 0.0) or 0.0)
    text_overlap = float(ev.get("text_overlap", 0.0) or 0.0)
    sfx_overlap = float(ev.get("sfx_overlap", 0.0) or 0.0)
    panel_overlap = float(ev.get("panel_overlap", 0.0) or 0.0)
    bubble_min = float(getattr(cfg, "paired_diff_koharu_layout_bubble_min_overlap", 0.25))
    text_min = float(getattr(cfg, "paired_diff_koharu_layout_text_min_overlap", 0.35))
    sfx_min = float(getattr(cfg, "paired_diff_koharu_layout_sfx_min_overlap", 0.35))
    panel_min = float(getattr(cfg, "paired_diff_koharu_layout_panel_only_min_overlap", 0.20))
    kind = str(region_kind or "bubble")
    if kind in {"complex_text", "free_text"}:
        passed = bool(text_overlap >= text_min or sfx_overlap >= sfx_min)
    elif kind == "bubble":
        passed = bool(bubble_overlap >= bubble_min or text_overlap >= text_min)
    else:
        # Unknown/plugin region kinds retain the generic positive-evidence rule;
        # this avoids silently breaking extension records we do not understand.
        passed = bool(bubble_overlap >= bubble_min or text_overlap >= text_min or sfx_overlap >= sfx_min)
    if passed:
        reason = "koharu_layout_semantic_support"
    elif panel_overlap >= panel_min:
        reason = "koharu_layout_panel_only_artwork"
    else:
        reason = "koharu_layout_no_semantic_support"
    return passed, reason, {
        "bubble_overlap": round(bubble_overlap, 4),
        "text_overlap": round(text_overlap, 4),
        "sfx_overlap": round(sfx_overlap, 4),
        "panel_overlap": round(panel_overlap, 4),
    }


def _apply_layout_safety_gate(result, *, source_available: bool, target_available: bool, cfg: Any, route: str) -> dict[str, Any]:
    """Hard-gate paired-diff renderer authority using two-sided Koharu evidence.

    The gate intentionally fails open when either Layout inference is unavailable:
    model/runtime failure must not disable the established visual transfer route.
    When both sides are available, however, a proposal must prove region-kind
    appropriate semantic support on *both* source and target before any renderer
    sees it.  Rejected proposals remain in diagnostics but are removed from the
    renderable bubble/record lists.
    """
    records = list(getattr(result, "records", []) or [])
    enabled = bool(getattr(cfg, "paired_diff_koharu_layout_safety_gate_enabled", True))
    both_available = bool(source_available and target_available)
    if not enabled or not both_available or not records:
        return {
            "enabled": enabled,
            "applied": False,
            "route": str(route),
            "source_available": bool(source_available),
            "target_available": bool(target_available),
            "input_regions": len(records),
            "kept_regions": len(records),
            "rejected_regions": 0,
            "fail_open": bool(enabled and not both_available),
            "rejected": [],
        }

    original_src = list(getattr(result, "source_bubbles", []) or [])
    original_dst = list(getattr(result, "target_bubbles", []) or [])
    src_by = {str(b.id): b for b in original_src}
    dst_by = {str(b.id): b for b in original_dst}
    referenced_src = {str(getattr(r, "source_id", "")) for r in records}
    referenced_dst = {str(getattr(r, "target_id", "")) for r in records}
    keep_records = []
    keep_src_ids: set[str] = set()
    keep_dst_ids: set[str] = set()
    rejected = []
    for rec in records:
        sb = src_by.get(str(getattr(rec, "source_id", "")))
        tb = dst_by.get(str(getattr(rec, "target_id", "")))
        if sb is None or tb is None:
            # Preserve malformed/plugin rows for legacy handling rather than
            # converting a bookkeeping mismatch into data loss.
            keep_records.append(rec)
            if sb is not None:
                keep_src_ids.add(str(sb.id))
            if tb is not None:
                keep_dst_ids.add(str(tb.id))
            continue
        kind = str(getattr(rec, "region_kind", "bubble") or "bubble")
        src_ok, src_reason, src_ev = _semantic_layout_support(sb, kind, cfg)
        tgt_ok, tgt_reason, tgt_ev = _semantic_layout_support(tb, kind, cfg)
        pair_ok = bool(src_ok and tgt_ok)
        if pair_ok:
            decision = {
                "passed": True,
                "route": str(route),
                "region_kind": kind,
                "reason": "koharu_layout_two_sided_semantic_support",
                "source": {"supported": True, **src_ev},
                "target": {"supported": True, **tgt_ev},
            }
            for b in (sb, tb):
                meta = dict(getattr(b, "meta", {}) or {})
                meta["koharu_layout_safety_gate"] = decision
                b.meta = meta
            keep_records.append(rec)
            keep_src_ids.add(str(sb.id)); keep_dst_ids.add(str(tb.id))
            continue

        if (not src_ok and src_reason == "koharu_layout_panel_only_artwork") or (not tgt_ok and tgt_reason == "koharu_layout_panel_only_artwork"):
            reason = "koharu_layout_panel_only_artwork"
        else:
            reason = "koharu_layout_missing_two_sided_semantic_support"
        decision = {
            "passed": False,
            "route": str(route),
            "region_kind": kind,
            "reason": reason,
            "source": {"supported": bool(src_ok), "reason": src_reason, **src_ev},
            "target": {"supported": bool(tgt_ok), "reason": tgt_reason, **tgt_ev},
        }
        for b in (sb, tb):
            meta = dict(getattr(b, "meta", {}) or {})
            meta["koharu_layout_safety_gate"] = decision
            b.meta = meta
        rejected.append({
            "source_id": str(getattr(rec, "source_id", "")),
            "target_id": str(getattr(rec, "target_id", "")),
            "region_kind": kind,
            "reason": reason,
            "source": decision["source"],
            "target": decision["target"],
        })

    result.records = keep_records
    # Preserve unreferenced/plugin bubbles verbatim; only proposals explicitly
    # represented by a rejected record are removed from renderer authority.
    result.source_bubbles = [
        b for b in original_src
        if str(b.id) in keep_src_ids or str(b.id) not in referenced_src
    ]
    result.target_bubbles = [
        b for b in original_dst
        if str(b.id) in keep_dst_ids or str(b.id) not in referenced_dst
    ]
    return {
        "enabled": True,
        "applied": True,
        "route": str(route),
        "source_available": True,
        "target_available": True,
        "input_regions": len(records),
        "kept_regions": len(keep_records),
        "rejected_regions": len(rejected),
        "fail_open": False,
        "rejected": rejected,
    }


def _fuse_layout_evidence(
    pair: PagePair, source: np.ndarray, target: np.ndarray, paired_diff, *,
    config: Any, cache: PageStageCache, stats: dict[str, str],
) -> None:
    if paired_diff is None:
        return
    # The two-sided semantic hard gate is a Koharu-specific capability because
    # it requires text/SFX/bubble/panel instance masks.  When the user selects a
    # different main detector, do not silently load Koharu or let an auxiliary
    # override the chosen primary.  Paired Diff then keeps its established visual
    # safety rules and the selected primary still controls normal bubble routing.
    if not koharu_is_primary(config.bubbles):
        diag = {
            "policy": "selected_primary_without_koharu_semantic_gate",
            "primary_detector": primary_detector(config.bubbles),
            "safety_gate": {"enabled": False, "applied": False, "reason": "koharu_not_primary"},
        }
        paired_diff.diagnostics = {**dict(getattr(paired_diff, "diagnostics", {}) or {}), "detector_policy": diag}
        supplemental = getattr(paired_diff, "supplemental", None)
        if supplemental is not None:
            supplemental.diagnostics = {**dict(getattr(supplemental, "diagnostics", {}) or {}), "detector_policy": diag}
        stats["paired_diff_layout"] = f"primary:{primary_detector(config.bubbles)}:no_koharu_semantic_gate"
        return
    cache_enabled = bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True))
    src_ev = collect_koharu_layout_evidence_cached(
        source, config.bubbles, role="paired_diff_source", image_path=pair.source_path,
        cache=cache, cache_enabled=cache_enabled, stats=stats, allow_missing=True,
    )
    tgt_ev = collect_koharu_layout_evidence_cached(
        target, config.bubbles, role="paired_diff_target", image_path=pair.target_path,
        cache=cache, cache_enabled=cache_enabled, stats=stats, allow_missing=True,
    )
    diag = {
        "source": _annotate_side_with_layout(paired_diff.source_bubbles, src_ev, source.shape[:2]),
        "target": _annotate_side_with_layout(paired_diff.target_bubbles, tgt_ev, target.shape[:2]),
        "source_available": bool(src_ev.available),
        "target_available": bool(tgt_ev.available),
        "policy": "two_sided_region_kind_semantic_renderer_gate",
    }
    main_gate = _apply_layout_safety_gate(
        paired_diff,
        source_available=bool(src_ev.available), target_available=bool(tgt_ev.available),
        cfg=config.mask_replace, route="primary",
    )
    diag["safety_gate"] = main_gate
    paired_diff.diagnostics = {**dict(getattr(paired_diff, "diagnostics", {}) or {}), "koharu_layout_evidence": diag}
    supplemental = getattr(paired_diff, "supplemental", None)
    if supplemental is not None:
        supp_diag = {
            "source": _annotate_side_with_layout(supplemental.source_bubbles, src_ev, source.shape[:2]),
            "target": _annotate_side_with_layout(supplemental.target_bubbles, tgt_ev, target.shape[:2]),
            "source_available": bool(src_ev.available),
            "target_available": bool(tgt_ev.available),
            "policy": "two_sided_region_kind_semantic_renderer_gate",
        }
        supp_gate = _apply_layout_safety_gate(
            supplemental,
            source_available=bool(src_ev.available), target_available=bool(tgt_ev.available),
            cfg=config.mask_replace, route="supplemental",
        )
        supp_diag["safety_gate"] = supp_gate
        supplemental.diagnostics = {**dict(getattr(supplemental, "diagnostics", {}) or {}), "koharu_layout_evidence": supp_diag}
    stats["paired_diff_layout"] = "koharu_layout_guarded" if (src_ev.available and tgt_ev.available) else (
        "koharu_layout_partial_fail_open" if (src_ev.available or tgt_ev.available) else "fallback_no_layout"
    )


@dataclass
class PairedDiffStageResult:
    paired_diff: Any | None
    use_paired_diff: bool
    gate: float


def run_paired_diff_stage(
    pair: PagePair,
    source: np.ndarray,
    target: np.ndarray,
    registration: Any,
    *,
    mode: str,
    direct_container_fast: bool,
    config: Any,
    cache: PageStageCache,
    stats: dict[str, str],
) -> PairedDiffStageResult:
    cfg = config.mask_replace
    paired_gate = min(
        cfg.paired_diff_min_registration_confidence,
        cfg.photo_pair_min_registration_confidence,
        cfg.paired_diff_structural_min_registration_confidence,
    )
    paired_diff = None
    use_paired_diff = False
    eligible = bool(
        not direct_container_fast
        and mode in {"auto", "mask_replace", "hybrid", "reletter"}
        and cfg.paired_diff_enabled
        and registration.confidence >= paired_gate
    )
    if eligible:
        sig = paired_diff_stage_signature(pair, registration, cfg)
        paired_diff = cache.load_paired_diff(sig)
        if paired_diff is not None:
            stats["paired_diff"] = "hit"
            use_paired_diff = bool(paired_diff.source_bubbles and paired_diff.target_bubbles)
        else:
            try:
                paired_diff = extract_paired_diff_bubbles(source, target, registration, cfg)
                cache.save_paired_diff(sig, paired_diff)
                stats["paired_diff"] = "miss"
                use_paired_diff = bool(paired_diff.source_bubbles and paired_diff.target_bubbles)
            except Exception as exc:
                stats["paired_diff"] = "failed"
                logger.warning(
                    "Paired-difference bubble extraction failed; falling back to OCR/bubble pipeline: %s",
                    exc,
                )
    if paired_diff is not None:
        _fuse_layout_evidence(
            pair, source, target, paired_diff, config=config, cache=cache, stats=stats
        )
        # Layout safety may remove renderer-unsafe paired-diff proposals. Keep
        # downstream routing in sync with the post-gate candidate set.
        use_paired_diff = bool(paired_diff.source_bubbles and paired_diff.target_bubbles)
    return PairedDiffStageResult(paired_diff, use_paired_diff, float(paired_gate))


__all__ = ["PairedDiffStageResult", "run_paired_diff_stage"]
