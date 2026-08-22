from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .modes.aligned_overlay_reveal.route import execute_isolated_hole_route
from .modes.aligned_overlay_reveal.renderer import apply_detector_policy_guard, build_aligned_overlay_plan, execute_aligned_overlay
from .modes.aligned_overlay_reveal.persist import persist_aligned_hole_page
from .config import PipelineConfig
from .models import PagePair, PageProject, QAItem, RegistrationResult
from .page_management import PageMark
from .pipeline_direct_arbitration import arbitrate_direct_source
from .pipeline_passthrough import emit_passthrough_page
from .modes.transparent_bubble_reveal.persist import persist_transparent_page
from .modes.transparent_bubble_reveal.route import execute_isolated_transparent_route
from .transfer_planner import TransferDecision, choose_transfer_strategy
from .pipeline_ocr_service import build_ocr_backend_soft
from .ocr import NullOCRBackend

logger = logging.getLogger(__name__)


def _build_transparent_target_ocr(config: PipelineConfig, cache_stats: dict) -> object | None:
    tcfg = config.transparent_bubble_reveal
    if not bool(getattr(tcfg, "target_text_presence_ocr_enabled", False)):
        cache_stats["transparent_ocr_target"] = "disabled"
        return None
    backend_name = str(config.ocr.target_backend or config.ocr.backend or "none")
    failures: list[str] = []
    backend = build_ocr_backend_soft(
        config.ocr, str(config.ocr.target_lang or "japan"), backend_name,
        role="target", soft_failures=failures,
    )
    if isinstance(backend, NullOCRBackend):
        cache_stats["transparent_ocr_target"] = "unavailable"
        if failures:
            cache_stats["transparent_ocr_target_error"] = failures[-1]
        return None
    cache_stats["transparent_ocr_target"] = f"ready:{backend_name}"
    return backend


@dataclass(slots=True)
class PageRouteState:
    source: np.ndarray
    registration: RegistrationResult
    pair_check: Any
    source_path_local: str
    direct_container_plan: Any | None
    direct_container_fast: bool
    selected_source_kind: str
    selected_secondary_source: bool
    dual_source_arbitration: list[dict]
    selected_arbitration_evidence: dict
    decision: TransferDecision | None
    early_project: PageProject | None = None


def run_page_route_stage(
    *,
    config: PipelineConfig,
    mode: str,
    mode_contract,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None,
    mark: PageMark,
    authority_source: np.ndarray,
    authority_source_path: str | Path,
    source: np.ndarray,
    source_path_local: str | Path,
    target: np.ndarray,
    registration: RegistrationResult,
    pair_check,
    replace_source_specs: list[dict],
    secondary_source_available: bool,
    cache_stats: dict,
    stage_cache=None,
) -> PageRouteState:
    """Resolve explicit Reveal/Direct/Auto routing without owning renderer algorithms."""
    base = dict(
        source=source,
        registration=registration,
        pair_check=pair_check,
        source_path_local=str(source_path_local),
        direct_container_plan=None,
        direct_container_fast=False,
        selected_source_kind="primary",
        selected_secondary_source=False,
        dual_source_arbitration=[],
        selected_arbitration_evidence={},
    )

    # Explicit whole-page transparent reveal returns before Direct/Mask/OCR.
    if mode_contract.transparent_reveal:
        t_cfg = config.transparent_bubble_reveal
        transparent_target_ocr = None
        if bool(pair_check.same_page):
            transparent_target_ocr = _build_transparent_target_ocr(config, cache_stats)
        transparent_result = execute_isolated_transparent_route(
            mode, same_page=bool(pair_check.same_page), source=source, target=target,
            registration=registration, config=t_cfg, bubble_config=config.bubbles,
            stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=str(pair.target_path), source_path=str(source_path_local),
            target_text_ocr=transparent_target_ocr, semantic_config=config.semantic,
        )
        transparent_decision = choose_transfer_strategy(
            mode, same_page=bool(pair_check.same_page),
            same_page_confidence=float(pair_check.confidence),
            direct_plan_available=False, direct_plan_safe=False,
        )
        early = persist_transparent_page(
            config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, pair_check=pair_check, result=transparent_result,
            planner_decision=transparent_decision, cache_stats=cache_stats,
        )
        return PageRouteState(**base, decision=transparent_decision, early_project=early)

    # Explicit aligned whole-page mode is a genuinely independent renderer.
    # It must never alias/fall through to Transparent Reveal.  The primary TARGET
    # bubble detector is already warmed by page-flow before this point.
    if mode_contract.aligned_reveal:
        aligned_result = execute_isolated_hole_route(
            mode, same_page=bool(pair_check.same_page), source=source, target=target,
            registration=registration, config=config.aligned_overlay_reveal, bubble_config=config.bubbles,
            stage_cache=stage_cache, cache_stats=cache_stats,
            source_path=str(source_path_local), target_path=str(pair.target_path),
        )
        aligned_available = bool(aligned_result.accepted and aligned_result.applied_count > 0)
        aligned_safe = bool(aligned_available and str(aligned_result.page_triage).upper() == "SAFE")
        aligned_decision = choose_transfer_strategy(
            mode, same_page=bool(pair_check.same_page),
            same_page_confidence=float(pair_check.confidence),
            direct_plan_available=False, direct_plan_safe=False,
            aligned_plan_available=aligned_available, aligned_plan_safe=aligned_safe,
        )
        early = persist_aligned_hole_page(
            config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, pair_check=pair_check, result=aligned_result,
            requested_mode=mode, planner_decision=aligned_decision, cache_stats=cache_stats,
        )
        return PageRouteState(**base, decision=aligned_decision, early_project=early)

    direct_requested = bool(mode_contract.direct)
    arbitration = arbitrate_direct_source(
        mode=mode, direct_requested=direct_requested, config=config,
        authority_source=authority_source, authority_source_path=str(authority_source_path),
        source=source, source_path_local=str(source_path_local), target=target,
        registration=registration, pair_check=pair_check,
        replace_source_specs=replace_source_specs,
        secondary_source_available=bool(secondary_source_available),
        stage_cache=stage_cache, cache_enabled=bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True)), cache_stats=cache_stats,
    )
    source = arbitration.source
    registration = arbitration.registration
    pair_check = arbitration.pair_check
    source_path_local = arbitration.source_path_local
    direct_container_plan = arbitration.direct_container_plan
    direct_container_fast = arbitration.direct_container_fast
    selected_source_kind = arbitration.selected_source_kind
    selected_secondary_source = arbitration.selected_secondary_source
    dual_source_arbitration = list(arbitration.arbitration)
    selected_arbitration_evidence = dict(arbitration.selected_arbitration_evidence)

    # Experimental aligned erase-to-reveal may participate in Auto only under
    # its explicit allow-in-auto contract and only after Direct arbitration.
    aligned_auto_result = None
    aligned_auto_allowed = bool(
        mode == "auto"
        and config.aligned_overlay_reveal.enabled
        and config.aligned_overlay_reveal.allow_in_auto
        and not config.aligned_overlay_reveal.require_explicit_mode
        and bool(pair_check.same_page)
        and not direct_container_fast
    )
    if aligned_auto_allowed:
        try:
            aligned_auto_plan = build_aligned_overlay_plan(
                source, target, registration, config.aligned_overlay_reveal
            )
            aligned_auto_plan = apply_detector_policy_guard(
                aligned_auto_plan, target, config.bubbles, stage_cache=stage_cache,
                cache_stats=cache_stats, target_path=str(pair.target_path),
                cache_enabled=bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True)),
            )
            aligned_auto_result = execute_aligned_overlay(
                aligned_auto_plan, source, target, config.aligned_overlay_reveal
            )
        except Exception as exc:
            logger.warning("Aligned overlay reveal auto candidate failed: %s", exc)
            aligned_auto_result = None

    aligned_auto_available = bool(
        aligned_auto_result is not None
        and aligned_auto_result.accepted
        and aligned_auto_result.applied_count > 0
    )
    aligned_auto_safe = bool(
        aligned_auto_available
        and str(aligned_auto_result.page_triage).upper() == "SAFE"
    )
    decision = choose_transfer_strategy(
        mode,
        same_page=bool(pair_check.same_page),
        same_page_confidence=float(pair_check.confidence),
        direct_plan_available=bool(direct_container_plan is not None and direct_container_plan.result.applied_count > 0),
        direct_plan_safe=bool(direct_container_plan is not None and direct_container_plan.safe_to_skip_other_paths),
        secondary_source_available=bool(secondary_source_available),
        secondary_source_selected=bool(selected_secondary_source),
        aligned_plan_available=aligned_auto_available,
        aligned_plan_safe=aligned_auto_safe,
        aligned_auto_allowed=aligned_auto_allowed,
    )

    if decision.strategy == "aligned_overlay_reveal" and aligned_auto_result is not None:
        early = persist_aligned_hole_page(
            config, pair, page_root, final_path, mark, source=source, target=target,
            registration=registration, pair_check=pair_check,
            result=aligned_auto_result, requested_mode=mode,
            planner_decision=decision, cache_stats=cache_stats,
        )
        return PageRouteState(
            source, registration, pair_check, str(source_path_local), direct_container_plan,
            direct_container_fast, selected_source_kind, selected_secondary_source,
            dual_source_arbitration, selected_arbitration_evidence, decision, early,
        )

    # Direct Patch is strict: rejection preserves TARGET unchanged and never
    # silently falls through to Mask Replace.
    if mode == "direct_patch" and not direct_container_fast:
        reject_mark = PageMark(
            page_type="content", origin=mark.origin, confidence=float(pair_check.confidence),
            reason=f"direct_patch_rejected:{decision.reason}",
            bubble_regions=0, free_text_regions=0,
            registration_confidence=float(registration.confidence),
            source_name=Path(pair.source_path).name, target_name=Path(pair.target_path).name,
        )
        early = emit_passthrough_page(
            config, pair, page_root, final_path, reject_mark, source=source, target=target,
            registration=registration, passthrough_reason="direct_patch_rejected",
            extra_meta={
                "transfer_mode": mode,
                "transfer_planner": decision.to_dict(),
                "page_pairing_check": pair_check.to_dict(),
                "direct_patch": {
                    "used": False,
                    "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                    "manual_effect_candidates": list(((direct_container_plan.diagnostics if direct_container_plan is not None else {}) or {}).get("manual_effect_candidates", []) or []),
                    "review_required": True,
                },
            },
            qa=[QAItem(
                "direct_patch_rejected", "warning",
                "Direct Patch could not prove a safe same-layout whole-container transfer; target page was kept unchanged. Use Mask Transfer/Auto for non-identical layouts.",
                meta={"reason": decision.reason, "pairing": pair_check.to_dict()},
            )],
        )
        return PageRouteState(
            source, registration, pair_check, str(source_path_local), direct_container_plan,
            direct_container_fast, selected_source_kind, selected_secondary_source,
            dual_source_arbitration, selected_arbitration_evidence, decision, early,
        )

    return PageRouteState(
        source, registration, pair_check, str(source_path_local), direct_container_plan,
        direct_container_fast, selected_source_kind, selected_secondary_source,
        dual_source_arbitration, selected_arbitration_evidence, decision, None,
    )
