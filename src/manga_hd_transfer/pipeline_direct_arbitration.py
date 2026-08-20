from __future__ import annotations

"""Direct SOURCE arbitration extracted from the page orchestrator.

This service owns candidate discovery/evidence/scoring only.  It does not own
page persistence, OCR, Mask/Reletter rendering, or GUI state.  The translated
SOURCE remains text/raster authority; TARGET is used only for page identity and
placement evidence.
"""

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np

from .modes.direct_patch.container_renderer import build_source_direct_container_plan
from .dual_source import build_direct_source_evidence, select_direct_source_candidate
from .io_utils import read_image
from .pipeline_registration_service import register_page, verify_same_page_strict
from .plugins import REGISTRY as PROVIDER_REGISTRY
from .source_detectors import run_source_detector_chain
from .detector_policy import (
    detector_strategy, source_auxiliary_providers, expensive_provider,
    STRATEGY_PRIMARY_ONLY, STRATEGY_ALWAYS,
)
from .transfer_policy import _blocking_direct_invariant_issues

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class DirectArbitrationResult:
    source: np.ndarray
    registration: Any
    pair_check: Any
    source_path_local: str
    direct_container_plan: Any | None
    direct_container_fast: bool
    selected_source_kind: str
    selected_secondary_source: bool
    arbitration: list[dict]
    selected_arbitration_evidence: dict


def _annotate_plan(plan, provider_audit: list[dict], *, source_spec: dict | None = None) -> bool:
    if plan is None:
        return False
    plan.diagnostics["source_detector_provider_audit"] = provider_audit
    if source_spec is not None:
        plan.diagnostics["replace_translation_additional_source"] = dict(source_spec)
    qa_provider = PROVIDER_REGISTRY.get("qa_check", "source_direct_invariants")
    invariant_issues = qa_provider(plan) if qa_provider is not None else []
    plan.diagnostics["plugin_invariant_issues"] = invariant_issues
    blocking = _blocking_direct_invariant_issues(invariant_issues)
    plan.diagnostics["plugin_blocking_invariant_issues"] = blocking
    if blocking:
        plan.safe_to_skip_other_paths = False
    return not bool(blocking)


def arbitrate_direct_source(
    *,
    mode: str,
    direct_requested: bool,
    config,
    authority_source: np.ndarray,
    authority_source_path: str,
    source: np.ndarray,
    source_path_local: str,
    target: np.ndarray,
    registration,
    pair_check,
    replace_source_specs: list[dict],
    secondary_source_available: bool,
    stage_cache=None,
    cache_enabled: bool = True,
    cache_stats: dict | None = None,
) -> DirectArbitrationResult:
    """Build the primary Direct plan and, when useful, arbitrate alternates.

    The implementation intentionally preserves v2.0.61 semantics: explicit
    Direct may keep a reviewable partial primary plan, while Auto accepts only a
    publication-safe winner.
    """
    direct_container_plan = None
    direct_container_fast = False
    selected_source_kind = "primary"
    selected_secondary_source = False
    arbitration_rows: list[dict] = []

    if direct_requested and (
        bool(pair_check.same_page)
        or not bool(getattr(config.direct_patch, "require_same_page_precheck", True))
    ):
        try:
            # v2.0.91 explicit detector policy: one selected primary runs first.
            # Selected auxiliaries run only when the strategy asks for them;
            # expensive auxiliaries are never inferred merely because they are
            # installed.
            detector_mode = detector_strategy(config.bubbles)
            selected_aux = source_auxiliary_providers(config.bubbles, include_refiner=True)
            selected_expensive = any(expensive_provider(name) for name in selected_aux)
            source_hints, provider_audit = run_source_detector_chain(
                source, config.direct_patch, config.bubbles,
                existing=[], source_path=source_path_local, allow_expensive=False, primary_only=True,
                cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
            )
            all_source_hints = list(source_hints)
            direct_container_plan = build_source_direct_container_plan(
                source, target, registration, config.direct_patch,
                source_hint_bubbles=all_source_hints,
            )
            direct_plan_safe = bool(
                direct_container_plan is not None
                and direct_container_plan.safe_to_skip_other_paths
                and direct_container_plan.result.applied_count > 0
            )
            run_aux = detector_mode != STRATEGY_PRIMARY_ONLY and (detector_mode == STRATEGY_ALWAYS or not direct_plan_safe)
            if run_aux:
                fallback_hints, fallback_audit = run_source_detector_chain(
                    source, config.direct_patch, config.bubbles,
                    existing=all_source_hints, source_path=source_path_local,
                    allow_expensive=False, fallback_only=True,
                    cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
                )
                provider_audit.extend(fallback_audit)
                if fallback_hints:
                    all_source_hints.extend(fallback_hints)
                    direct_container_plan = build_source_direct_container_plan(
                        source, target, registration, config.direct_patch,
                        source_hint_bubbles=all_source_hints,
                    )
                    direct_plan_safe = bool(
                        direct_container_plan is not None
                        and direct_container_plan.safe_to_skip_other_paths
                        and direct_container_plan.result.applied_count > 0
                    )
            run_expensive = bool(
                selected_expensive and detector_mode != STRATEGY_PRIMARY_ONLY
                and (detector_mode == STRATEGY_ALWAYS or not direct_plan_safe)
            )
            if run_expensive:
                expensive_hints, expensive_audit = run_source_detector_chain(
                    source, config.direct_patch, config.bubbles,
                    existing=(direct_container_plan.source_bubbles if direct_container_plan is not None else all_source_hints),
                    source_path=source_path_local, allow_expensive=True, only_expensive=True,
                    cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
                )
                provider_audit.extend(expensive_audit)
                if expensive_hints:
                    all_source_hints.extend(expensive_hints)
                    direct_container_plan = build_source_direct_container_plan(
                        source, target, registration, config.direct_patch,
                        source_hint_bubbles=all_source_hints,
                    )
                    direct_plan_safe = bool(
                        direct_container_plan is not None
                        and direct_container_plan.safe_to_skip_other_paths
                        and direct_container_plan.result.applied_count > 0
                    )
            if direct_container_plan is not None:
                invariant_ok = _annotate_plan(direct_container_plan, provider_audit)
                if not invariant_ok:
                    direct_plan_safe = False
                direct_container_plan.diagnostics["provider_registry"] = PROVIDER_REGISTRY.snapshot()
                direct_container_plan.diagnostics["source_completion_plan_strategy"] = f"{detector_mode}:primary_then_selected_aux"
            direct_container_fast = bool(
                direct_container_plan is not None
                and direct_container_plan.result.applied_count > 0
                and (mode == "direct_patch" or direct_plan_safe)
            )
        except Exception as exc:
            logger.warning("Direct Patch plan failed: %s", exc)
            direct_container_plan = None
            direct_container_fast = False

    dual_prefer = bool(getattr(config.dual_source, "enabled", False) and getattr(config.dual_source, "prefer_secondary_for_direct", True))
    allow_replace_retry = bool(getattr(config.replace_translation, "additional_source_retry_direct", True))
    allow_secondary_retry = bool(getattr(config.dual_source, "enabled", False) and getattr(config.dual_source, "accept_secondary_direct", True))
    arbitration_enabled = bool(getattr(config.dual_source, "arbitration_enabled", True))
    arbitration_candidates: list[tuple[Any, dict]] = []

    if direct_requested and direct_container_plan is not None:
        primary_ev = build_direct_source_evidence(
            path=str(authority_source_path), kind="primary", is_secondary=False,
            source=authority_source, registration=registration, pair_check=pair_check,
            plan=direct_container_plan, config=config.dual_source,
        )
        arbitration_rows.append(primary_ev.to_dict())
        arbitration_candidates.append((primary_ev, {
            "spec": {"path": str(authority_source_path), "kind": "primary", "origin": "primary"},
            "source": authority_source, "registration": registration, "pair_check": pair_check,
            "plan": direct_container_plan, "is_secondary": False,
        }))

    should_try_alternates = bool(
        direct_requested and replace_source_specs and (
            not direct_container_fast
            or (allow_secondary_retry and secondary_source_available)
            or (not arbitration_enabled and dual_prefer and secondary_source_available)
        )
    )
    legacy_best_alt = None
    if should_try_alternates:
        for spec in replace_source_specs:
            is_secondary = str(spec.get("origin", "")) == "dual_source" or str(spec.get("kind", "")) == "secondary_dir"
            if is_secondary and not allow_secondary_retry:
                continue
            if (not is_secondary) and not allow_replace_retry:
                continue
            try:
                alt_source = read_image(spec["path"])
                alt_registration = register_page(alt_source, target, config.registration)
                alt_check = verify_same_page_strict(alt_source, target, alt_registration, config.pairing)
                if not alt_check.same_page and bool(getattr(config.direct_patch, "require_same_page_precheck", True)):
                    alt_plan = None
                else:
                    alt_hints, alt_audit = run_source_detector_chain(
                        alt_source, config.direct_patch, config.bubbles,
                        existing=[], source_path=spec["path"], allow_expensive=False, primary_only=True,
                        cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
                    )
                    all_alt_hints = list(alt_hints)
                    alt_plan = build_source_direct_container_plan(
                        alt_source, target, alt_registration, config.direct_patch,
                        source_hint_bubbles=all_alt_hints,
                    )
                    alt_safe = bool(
                        alt_plan is not None and alt_plan.safe_to_skip_other_paths
                        and alt_plan.result.applied_count > 0
                    )
                    detector_mode = detector_strategy(config.bubbles)
                    selected_aux = source_auxiliary_providers(config.bubbles, include_refiner=True)
                    if detector_mode != STRATEGY_PRIMARY_ONLY and (detector_mode == STRATEGY_ALWAYS or not alt_safe):
                        fallback_hints, fallback_audit = run_source_detector_chain(
                            alt_source, config.direct_patch, config.bubbles,
                            existing=all_alt_hints, source_path=spec["path"], allow_expensive=False, fallback_only=True,
                            cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
                        )
                        alt_audit.extend(fallback_audit)
                        if fallback_hints:
                            all_alt_hints.extend(fallback_hints)
                            alt_plan = build_source_direct_container_plan(
                                alt_source, target, alt_registration, config.direct_patch,
                                source_hint_bubbles=all_alt_hints,
                            )
                            alt_safe = bool(alt_plan is not None and alt_plan.safe_to_skip_other_paths and alt_plan.result.applied_count > 0)
                    if (
                        detector_mode != STRATEGY_PRIMARY_ONLY
                        and any(expensive_provider(name) for name in selected_aux)
                        and (detector_mode == STRATEGY_ALWAYS or not alt_safe)
                    ):
                        heavy_hints, heavy_audit = run_source_detector_chain(
                            alt_source, config.direct_patch, config.bubbles,
                            existing=all_alt_hints, source_path=spec["path"], allow_expensive=True, only_expensive=True,
                            cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats,
                        )
                        alt_audit.extend(heavy_audit)
                        if heavy_hints:
                            all_alt_hints.extend(heavy_hints)
                            alt_plan = build_source_direct_container_plan(
                                alt_source, target, alt_registration, config.direct_patch,
                                source_hint_bubbles=all_alt_hints,
                            )
                    if alt_plan is not None:
                        _annotate_plan(alt_plan, alt_audit, source_spec=spec)
                alt_ev = build_direct_source_evidence(
                    path=spec["path"], kind=str(spec.get("kind", "alternate")), is_secondary=is_secondary,
                    source=alt_source, registration=alt_registration, pair_check=alt_check,
                    plan=alt_plan, config=config.dual_source,
                )
                arbitration_rows.append(alt_ev.to_dict())
                payload = {
                    "spec": spec, "source": alt_source, "registration": alt_registration,
                    "pair_check": alt_check, "plan": alt_plan, "is_secondary": is_secondary,
                }
                arbitration_candidates.append((alt_ev, payload))
                legacy_score = (
                    1 if alt_ev.safe else 0,
                    2 if (is_secondary and dual_prefer) else 1,
                    alt_ev.applied_count,
                    int(alt_source.shape[0] * alt_source.shape[1]),
                    alt_ev.registration_confidence,
                )
                if legacy_best_alt is None or legacy_score > legacy_best_alt[0]:
                    legacy_best_alt = (legacy_score, alt_ev, payload)
            except Exception as exc:
                logger.warning("Alternate replace_translation source failed (%s): %s", spec.get("path"), exc)

    selected_candidate = select_direct_source_candidate(arbitration_candidates) if arbitration_enabled else None
    if selected_candidate is not None:
        selected_ev, payload = selected_candidate
        if selected_ev.path != str(authority_source_path):
            source = payload["source"]
            registration = payload["registration"]
            pair_check = payload["pair_check"]
            direct_container_plan = payload["plan"]
            source_path_local = str(payload["spec"]["path"])
            selected_source_kind = str(payload["spec"].get("kind", "alternate"))
            selected_secondary_source = bool(payload["is_secondary"])
            direct_container_fast = True
        elif direct_container_plan is not None:
            source = authority_source
            source_path_local = str(authority_source_path)
            selected_source_kind = "primary"
            selected_secondary_source = False
            if mode == "auto":
                direct_container_fast = bool(
                    direct_container_plan.safe_to_skip_other_paths
                    and direct_container_plan.result.applied_count > 0
                )
    elif (not arbitration_enabled) and legacy_best_alt is not None:
        _legacy_score, ev, payload = legacy_best_alt
        if ev.safe and (not direct_container_fast or (payload["is_secondary"] and dual_prefer)):
            source = payload["source"]
            registration = payload["registration"]
            pair_check = payload["pair_check"]
            direct_container_plan = payload["plan"]
            source_path_local = str(payload["spec"]["path"])
            selected_source_kind = str(payload["spec"].get("kind", "alternate"))
            selected_secondary_source = bool(payload["is_secondary"])
            direct_container_fast = True

    selected_evidence = next(
        (row for row in arbitration_rows if str(row.get("path")) == str(source_path_local)),
        {},
    )
    return DirectArbitrationResult(
        source=source,
        registration=registration,
        pair_check=pair_check,
        source_path_local=str(source_path_local),
        direct_container_plan=direct_container_plan,
        direct_container_fast=bool(direct_container_fast),
        selected_source_kind=selected_source_kind,
        selected_secondary_source=bool(selected_secondary_source),
        arbitration=arbitration_rows,
        selected_arbitration_evidence=dict(selected_evidence or {}),
    )


__all__ = ["DirectArbitrationResult", "arbitrate_direct_source"]
