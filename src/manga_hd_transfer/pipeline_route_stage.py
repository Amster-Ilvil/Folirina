from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Callable

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


class _LazyBudgetTargetOCR:
    """Create the optional TARGET OCR backend only when a weak candidate needs it."""

    def __init__(self, config: PipelineConfig, cache_stats: dict) -> None:
        self.config = config
        self.cache_stats = cache_stats
        self.backend = None
        self.calls = 0
        self.max_calls = max(0, int(getattr(
            config.transparent_bubble_reveal,
            "target_text_presence_ocr_max_regions",
            6,
        )))
        self.cache_stats["transparent_ocr_target"] = "deferred"
        self.cache_stats["transparent_ocr_target_queries"] = 0

    def _ensure(self):
        if self.backend is not None:
            return self.backend
        backend_name = str(self.config.ocr.target_backend or self.config.ocr.backend or "none")
        failures: list[str] = []
        backend = build_ocr_backend_soft(
            self.config.ocr,
            str(self.config.ocr.target_lang or "japan"),
            backend_name,
            role="target",
            soft_failures=failures,
        )
        if isinstance(backend, NullOCRBackend):
            self.cache_stats["transparent_ocr_target"] = "unavailable"
            if failures:
                self.cache_stats["transparent_ocr_target_error"] = failures[-1]
            raise RuntimeError("transparent target OCR unavailable")
        self.backend = backend
        self.cache_stats["transparent_ocr_target"] = f"ready:{type(backend).__name__}"
        return backend

    def recognize_region(self, *args, **kwargs):
        if self.max_calls <= 0 or self.calls >= self.max_calls:
            self.cache_stats["transparent_ocr_target_budget_exhausted"] = True
            raise RuntimeError("transparent target OCR candidate budget exhausted")
        self.calls += 1
        self.cache_stats["transparent_ocr_target_queries"] = int(self.calls)
        return self._ensure().recognize_region(*args, **kwargs)


def _build_transparent_target_ocr(config: PipelineConfig, cache_stats: dict) -> object | None:
    tcfg = config.transparent_bubble_reveal
    if not bool(getattr(tcfg, "target_text_presence_ocr_enabled", False)):
        cache_stats["transparent_ocr_target"] = "disabled"
        return None
    return _LazyBudgetTargetOCR(config, cache_stats)


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
    inpaint_callback: Callable | None = None,
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
            inpaint_callback=inpaint_callback,
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
        target_path_local=str(pair.target_path), registration=registration, pair_check=pair_check,
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

    # Direct Patch remains strict when its whole-container planner rejects the page:
    # never fall through to Mask Replace / OCR-guided / photo-SFX transfer.  However,
    # a rejected Direct plan must not short-circuit Koharu semantic completion.  The
    # latter is a bounded, registered-source-ink-only recovery path backed by the
    # primary SOURCE/TARGET layout evidence, and is exactly what recovers translated
    # open text / coloured semantic regions such as p-115 and p-116.
    #
    # Keep routing alive and let the Direct-private pixel stage enter its
    # ``direct_semantic_only`` branch.  If semantic completion finds nothing safe,
    # the result is still byte-identical TARGET with an explicit review warning.
    if mode == "direct_patch" and not direct_container_fast:
        rejected_reason = str(decision.reason)
        cache_stats["direct_patch_rejected_continue_semantic"] = "1"
        cache_stats["direct_patch_rejected_reason"] = rejected_reason
        # The renderer ownership remains Direct: only its bounded semantic-only
        # subroute may execute.  Marking this as ``direct_reject`` would make
        # workspace validation treat legitimate Direct artifacts as stale
        # passthrough files.
        decision = TransferDecision(
            requested_mode="direct_patch", strategy="direct_patch", fallback_allowed=False,
            reason=f"semantic_completion_only_after:{rejected_reason}",
            same_page_confidence=float(pair_check.confidence), direct_plan_safe=False,
            evidence={**dict(getattr(decision, "evidence", {}) or {}),
                      "semantic_completion_only": True,
                      "whole_container_rejected_reason": rejected_reason},
            force_actions=list(getattr(decision, "force_actions", []) or []),
        )

    return PageRouteState(
        source, registration, pair_check, str(source_path_local), direct_container_plan,
        direct_container_fast, selected_source_kind, selected_secondary_source,
        dual_source_arbitration, selected_arbitration_evidence, decision, None,
    )
