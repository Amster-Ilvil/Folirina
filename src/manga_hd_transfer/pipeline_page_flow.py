from __future__ import annotations

"""Single-page application flow orchestration.

This module owns only the sequencing between already-isolated pipeline stages.
It deliberately does not implement registration, OCR, matching, rendering,
composition, persistence algorithms, or renderer internals.
"""

from pathlib import Path
from typing import Callable
import time
import cv2
import numpy as np

from .config import PipelineConfig
from .layout_evidence import (
    prepare_page_layout_evidence, collect_koharu_layout_evidence_cached, classify_layout_authority,
    koharu_semantic_text_ink_mask,
)
from .detector_policy import (
    STRATEGY_ALWAYS, auxiliary_detectors, detector_strategy, primary_detector, koharu_is_primary,
)
from .mode_contracts import get_mode_contract, mode_ocr_execution_violations
from .models import PagePair, PageProject, QAItem
from .page_management import PageMark
from .pipeline_page_prep import prepare_workspace, load_page_inputs
from .pipeline_bubble_service import primary_bubbles_cached
from .pipeline_route_stage import run_page_route_stage
from .pipeline_content_gate import source_has_no_transferable_text
from .ocr import NullOCRBackend
from .modes.hybrid.admission import should_continue_hybrid_after_ocr_negative
from .pipeline_registration_service import register_page_cached, verify_same_page
from .page_pairing import refine_pairing_with_layout_evidence
from .pipeline_paired_diff_service import run_paired_diff_stage
from .pipeline_text_stage import run_text_stage
from .pipeline_match_service import run_text_matching_stage
from .pipeline_transfer_execution import run_transfer_execution_stage
from .pipeline_transfer_composition import compose_transfer_state
from .pipeline_project_assembly import assemble_page_project, finalize_page_project
from .pipeline_artifact_export import export_page_artifacts
from .koharu_sfx_cleanup import clean_koharu_text_on_flat_backgrounds
from .lama_manga_runtime import inpaint_koharu_text, status as lama_manga_status
from .colored_target_composition import compose_coloured_containers




def _paired_ocr_authority_rescue(mode: str, match, source_unit, target_unit, source_decision, target_decision, registration_confidence: float) -> bool:
    """Rescue a strong paired OCR match from a one-sided false PROTECT.

    This is intentionally restricted to the two OCR product flows.  It does not
    weaken Direct, pure Mask or Reveal authority.  The common failure is a
    target-driven SOURCE crop/text detector covering only the first vertical
    glyphs of an otherwise valid bubble: Koharu then sees lots of panel area and
    marks the SOURCE side PROTECT even though the paired TARGET region is an
    explicit ALLOW.
    """
    if str(mode or "") not in {"reletter", "hybrid"}:
        return False
    reasons = [str(x or "") for x in (getattr(match, "reasons", None) or [])]
    if not any(r.startswith("route=paired_id_binding") for r in reasons):
        return False
    if float(getattr(match, "confidence", 0.0) or 0.0) < 0.92:
        return False
    if float(registration_confidence or 0.0) < 0.70:
        return False
    if not str(getattr(source_unit, "bubble_id", "") or "").strip():
        return False
    if not str(getattr(target_unit, "bubble_id", "") or "").strip():
        return False

    source_meta = dict(getattr(source_unit, "meta", {}) or {})
    source_recovered = bool(source_meta.get("source_crop_recovered"))
    pairs = ((source_decision, target_decision), (target_decision, source_decision))
    for protected, supported in pairs:
        if str(getattr(protected, "state", "")) != "PROTECT" or str(getattr(supported, "state", "")) != "ALLOW":
            continue
        if str(getattr(protected, "reason", "")) != "koharu_layout_panel_only_artwork":
            continue
        bubble_overlap = float(getattr(protected, "bubble_overlap", 0.0) or 0.0)
        text_overlap = float(getattr(protected, "text_overlap", 0.0) or 0.0)
        supported_text = float(getattr(supported, "text_overlap", 0.0) or 0.0)
        supported_bubble = float(getattr(supported, "bubble_overlap", 0.0) or 0.0)
        # A recovered SOURCE crop plus positive TARGET semantic support is the
        # strongest case.  The looser overlap clause is retained for already-
        # complete paired OCR blocks created before v2.3.16.
        if source_recovered and supported is target_decision and bubble_overlap >= 0.45:
            return True
        if bubble_overlap >= 0.55 and text_overlap >= 0.10 and (supported_text >= 0.35 or supported_bubble >= 0.55):
            return True
    return False

def _layout_region_kind_for_unit(unit) -> str:
    """Map OCR/reletter units to Koharu's bubble vs free-text authority domain.

    Target-driven Reletter regions use geometry=reletter_text_region even when
    they are narration/speech inside a detected bubble. Treating that geometry
    as free_text raises the text-overlap threshold and can incorrectly PROTECT
    a valid matched bubble. Bubble membership and speech/narration semantics are
    therefore authoritative before the low-level geometry tag.
    """
    meta = dict(getattr(unit, "meta", {}) or {})
    geometry = str(meta.get("geometry") or "").strip().lower()
    kind = str(getattr(unit, "kind", "") or meta.get("kind") or "").strip().lower()
    bubble_id = str(getattr(unit, "bubble_id", "") or meta.get("bubble_id") or "").strip()
    if bubble_id:
        return "bubble"
    if geometry == "bubble":
        return "bubble"
    if kind in {"speech", "narration", "dialogue", "dialog"}:
        return "bubble"
    return "free_text"


def run_page_flow(
    *,
    config: PipelineConfig,
    pair: PagePair,
    page_root: str | Path,
    final_path: str | Path | None = None,
    page_mark: PageMark | dict | None = None,
    cancel_cb=None,
    progress_cb=None,
    prefetched_images=None,
    trace=None,
    check_cancel: Callable | None = None,
    passthrough_page: Callable,
    get_source_backend: Callable,
    get_target_backend: Callable,
    get_reletter_executor: Callable,
) -> PageProject:
    """Run one already-transaction-wrapped page without owning domain algorithms."""

    if check_cancel is None:
        check_cancel = lambda _cb, _stage="": None

    flow_started = time.perf_counter()
    stage_timings: dict[str, float] = {}

    def _stage_done(name: str, started: float) -> None:
        stage_timings[str(name)] = round(max(0.0, time.perf_counter() - started), 6)

    def _attach_timings(project: PageProject) -> PageProject:
        payload = {
            "schema": "folirina.stage_timings.v1",
            "seconds": dict(stage_timings),
            "total_through_current_stage_seconds": round(max(0.0, time.perf_counter() - flow_started), 6),
        }
        if isinstance(getattr(project, "meta", None), dict):
            project.meta["stage_timings"] = payload
        if trace is not None:
            trace.event("stage_timings", **payload)
        return project

    def emit_progress(percent: int, stage: str, message: str) -> None:
        if progress_cb is None:
            return
        try:
            progress_cb(int(max(0, min(100, percent))), str(stage), str(message))
        except Exception:
            # UI progress must never be able to fail a page transaction.
            pass

    mode = config.transfer.mode.lower().strip()
    stage_started = time.perf_counter()
    emit_progress(2, "workspace", "准备页面工作区")
    mode_contract = get_mode_contract(mode)
    prepared = prepare_workspace(pair, page_root, page_mark, mode=mode)
    page_root = prepared.page_root
    mark = prepared.mark
    review_mode_archive = prepared.review_mode_archive
    stale_mode_cleanup = prepared.stale_mode_cleanup
    if trace is not None:
        trace.event(
            "workspace_prepared", selected_mode=mode,
            stale_mode_cleanup=prepared.stale_mode_cleanup,
            review_mode_archive=prepared.review_mode_archive,
        )
    check_cancel(cancel_cb, "before_page")
    if not mark.should_process:
        emit_progress(100, "skip", "页面已标记跳过")
        _stage_done("workspace_decode", stage_started)
        return _attach_timings(passthrough_page(pair, page_root, final_path, mark))

    inputs = load_page_inputs(pair, page_root, config, prefetched_images=prefetched_images)
    authority_source_path = inputs.authority_source_path
    source_path_local = inputs.source_path_local
    target_path_local = inputs.target_path_local
    authority_source = inputs.authority_source
    source = inputs.source
    target = inputs.target
    replace_source_specs = inputs.replace_source_specs
    secondary_source_available = inputs.secondary_source_available
    stage_cache = inputs.stage_cache
    cache_stats = inputs.cache_stats
    if trace is not None:
        trace.event(
            "images_loaded", source_shape=list(source.shape), target_shape=list(target.shape),
            source_path=str(authority_source_path), target_path=str(target_path_local),
        )
    check_cancel(cancel_cb, "after_decode")
    emit_progress(9, "decode", "已读取 SOURCE / TARGET")
    _stage_done("workspace_decode", stage_started)

    stage_started = time.perf_counter()
    emit_progress(12, "registration", "正在配准 SOURCE / TARGET")
    registration = register_page_cached(
        pair, source, target, config.registration, cache=stage_cache,
        cache_enabled=bool(config.cache.registration), stats=cache_stats,
    )
    if trace is not None:
        trace.event(
            "registration_done", registration_confidence=float(registration.confidence),
            registration_method=str(registration.method), cache=cache_stats.get("registration", ""),
        )
    check_cancel(cancel_cb, "after_registration")
    emit_progress(24, "registration", f"配准完成 · {registration.method} · {registration.confidence:.3f}")
    pair_check = verify_same_page(source, target, registration, config.pairing)
    _stage_done("registration", stage_started)
    stage_started = time.perf_counter()
    emit_progress(27, "layout", "正在运行主布局检测器")

    # v2.0.91: run the selected primary detector before any mode-specific
    # detector/OCR route.  Koharu receives its richer SOURCE+TARGET layout
    # prefetch when it is primary.  A non-Koharu primary is cached as bubble
    # geometry first; Koharu auxiliary evidence is never allowed to run before it.
    selected_primary = primary_detector(config.bubbles)
    koharu_target_evidence = None
    koharu_source_evidence = None
    if selected_primary == "koharu_layout":
        layout_prefetch = prepare_page_layout_evidence(
            mode, source, target, source_path=source_path_local, target_path=target_path_local,
            bubble_cfg=config.bubbles, cache=stage_cache,
            cache_enabled=bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True)),
            stats=cache_stats,
        )
        koharu_target_evidence = layout_prefetch.get("target")
        koharu_source_evidence = layout_prefetch.get("source")
        pair_check = refine_pairing_with_layout_evidence(
            pair_check, registration, layout_prefetch.get("source"), layout_prefetch.get("target"),
        )
        if bool((pair_check.diagnostics or {}).get("semantic_layout_rescue")):
            cache_stats["same_page_semantic_rescue"] = "koharu_layout"
        # Also materialize the canonical primary-bubble cache so later generic
        # bubble stages do not need a second primary adapter call.
        primary_bubbles_cached(
            "source", source, source_path_local, bubble_config=config.bubbles, cache=stage_cache,
            cache_enabled=bool(config.cache.bubbles), stats=cache_stats,
        )
        primary_bubbles_cached(
            "target", target, target_path_local, bubble_config=config.bubbles, cache=stage_cache,
            cache_enabled=bool(config.cache.bubbles), stats=cache_stats,
        )
    else:
        primary_bubbles_cached(
            "source", source, source_path_local, bubble_config=config.bubbles, cache=stage_cache,
            cache_enabled=bool(config.cache.bubbles), stats=cache_stats,
        )
        primary_bubbles_cached(
            "target", target, target_path_local, bubble_config=config.bubbles, cache=stage_cache,
            cache_enabled=bool(config.cache.bubbles), stats=cache_stats,
        )
        cache_stats["layout_prefetch"] = f"skipped_primary:{selected_primary}"
        # In 'always auxiliary' mode a selected Koharu auxiliary may be warmed,
        # but only *after* the chosen primary has completed. Conditional mode
        # stays lazy and invokes it only when an actual gap requires fallback.
        if detector_strategy(config.bubbles) == STRATEGY_ALWAYS and "koharu_layout" in auxiliary_detectors(config.bubbles):
            prepare_page_layout_evidence(
                mode, source, target, source_path=source_path_local, target_path=target_path_local,
                bubble_cfg=config.bubbles, cache=stage_cache,
                cache_enabled=bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True)),
                stats=cache_stats,
            )

    if trace is not None:
        trace.event(
            "layout_evidence_ready",
            route=str(cache_stats.get("layout_prefetch", "")),
            source=str(cache_stats.get("layout_source_cache", "")),
            target=str(cache_stats.get("layout_target_cache", "")),
            primary_detector=primary_detector(config.bubbles),
            detector_strategy=detector_strategy(config.bubbles),
            device_receipts=cache_stats.get("layout_device_receipts", {}),
        )
    check_cancel(cancel_cb, "after_layout_evidence")
    emit_progress(42, "layout", "主布局检测完成")
    _stage_done("layout", stage_started)

    stage_started = time.perf_counter()
    emit_progress(45, "route", "正在规划当前模式路线")
    lama_model_path = getattr(config.inpainting, "lama_model_path", None)
    lama_status = lama_manga_status(lama_model_path)
    lama_inpaint_callback = (
        (lambda image, mask: inpaint_koharu_text(image, mask, model_path=lama_model_path))
        if lama_status.ready else None
    )
    route = run_page_route_stage(
        config=config, mode=mode, mode_contract=mode_contract, pair=pair,
        page_root=page_root, final_path=final_path, mark=mark,
        authority_source=authority_source, authority_source_path=authority_source_path,
        source=source, source_path_local=source_path_local, target=target,
        registration=registration, pair_check=pair_check,
        replace_source_specs=replace_source_specs,
        secondary_source_available=bool(secondary_source_available), cache_stats=cache_stats,
        stage_cache=stage_cache, inpaint_callback=lama_inpaint_callback,
    )
    if route.early_project is not None:
        if trace is not None and mode == "aligned_overlay_reveal":
            reveal_meta = route.early_project.meta.get("aligned_overlay_reveal", {})
            decisions = (reveal_meta.get("diagnostics") or {}).get("candidate_decisions", [])
            trace.event("aligned_hole_candidate_decisions", candidates=decisions)
        emit_progress(100, "route", "页面按模式策略直接完成")
        _stage_done("route", stage_started)
        return _attach_timings(route.early_project)
    source = route.source
    registration = route.registration
    pair_check = route.pair_check
    source_path_local = route.source_path_local
    direct_container_plan = route.direct_container_plan
    direct_container_fast = route.direct_container_fast
    selected_source_kind = route.selected_source_kind
    selected_secondary_source = route.selected_secondary_source
    dual_source_arbitration = route.dual_source_arbitration
    selected_arbitration_evidence = route.selected_arbitration_evidence
    decision = route.decision
    _stage_done("route", stage_started)

    stage_started = time.perf_counter()
    emit_progress(50, "paired_diff", "正在分析 SOURCE / TARGET 差异")
    paired_stage = run_paired_diff_stage(
        pair, source, target, registration, mode=mode,
        direct_container_fast=direct_container_fast, config=config,
        cache=stage_cache, stats=cache_stats,
    )
    paired_diff = paired_stage.paired_diff
    use_paired_diff = paired_stage.use_paired_diff
    if trace is not None:
        trace.event(
            "paired_diff_done", paired_used=bool(use_paired_diff),
            paired_method=str(getattr(paired_diff, "method", "") or "") if paired_diff is not None else "",
            source_bubbles=len(getattr(paired_diff, "source_bubbles", []) or []) if paired_diff is not None else 0,
            target_bubbles=len(getattr(paired_diff, "target_bubbles", []) or []) if paired_diff is not None else 0,
            cache=cache_stats.get("paired_diff", ""),
        )
    check_cancel(cancel_cb, "after_paired_diff")
    emit_progress(60, "paired_diff", "版本差异分析完成")
    _stage_done("paired_diff", stage_started)

    stage_started = time.perf_counter()
    emit_progress(63, "text", "正在处理文字 / 气泡候选")
    text_stage = run_text_stage(
        config=config, mode=mode,
        direct_container_fast=direct_container_fast,
        direct_container_plan=direct_container_plan,
        use_paired_diff=use_paired_diff, paired_diff=paired_diff,
        source=source, target=target,
        source_path=source_path_local, target_path=target_path_local,
        registration=registration, cache=stage_cache, stats=cache_stats,
        get_source_backend=get_source_backend,
        get_target_backend=get_target_backend,
        get_reletter_executor=get_reletter_executor,
    )
    source_backend = text_stage.source_backend
    source_blocks = text_stage.source_blocks
    target_blocks = text_stage.target_blocks
    source_bubbles = text_stage.source_bubbles
    target_bubbles = text_stage.target_bubbles
    target_driven_reletter_regions = text_stage.target_driven_reletter_regions
    target_driven_reletter_diagnostics = text_stage.target_driven_reletter_diagnostics

    ocr_contract_violations = mode_ocr_execution_violations(mode, cache_stats)
    if ocr_contract_violations:
        if trace is not None:
            trace.event(
                "mode_ocr_contract_violation",
                selected_mode=mode,
                violations=list(ocr_contract_violations),
                ocr_source=str(cache_stats.get("ocr_source", "")),
                ocr_target=str(cache_stats.get("ocr_target", "")),
            )
        raise RuntimeError(
            "OCR mode isolation violation: " + ", ".join(ocr_contract_violations)
        )

    # Reletter is transcript-authoritative.  A missing SOURCE OCR runtime is not
    # evidence that the source page contains no text; silently publishing an
    # unchanged TARGET would look like a successful reletter run while no OCR
    # ever happened.  Fail before the generic no-text passthrough gate so the
    # lifecycle records a real failure and restores any previous published page.
    if mode == "reletter" and isinstance(source_backend, NullOCRBackend):
        configured_backend = str(config.ocr.source_backend or config.ocr.backend or "none")
        detail = next((
            str(x) for x in reversed(getattr(config, "_ocr_soft_failures", []) or [])
            if "source" in str(x).lower() or configured_backend.lower() in str(x).lower()
        ), "")
        raise RuntimeError(
            f"Reletter requires the configured SOURCE OCR backend '{configured_backend}', "
            "but that backend is unavailable. OCR fallback is disabled for Reletter; "
            "install/enable the selected OCR backend before processing."
            + (f" Detail: {detail}" if detail else "")
        )

    no_transferable_source_text = source_has_no_transferable_text(
        config=config, source_backend=source_backend, source_blocks=source_blocks,
        source_bubbles=source_bubbles, paired_diff=paired_diff, cache_stats=cache_stats,
    )
    if mode == "hybrid" and no_transferable_source_text:
        continue_hybrid, hybrid_visual = should_continue_hybrid_after_ocr_negative(config, paired_diff)
        if continue_hybrid:
            no_transferable_source_text = False
            cache_stats["hybrid_admission"] = "paired_visual_evidence_fail_open"
            cache_stats["hybrid_admission_method"] = str(hybrid_visual.get("method") or "")
            cache_stats["hybrid_admission_pairs"] = str(min(int(hybrid_visual.get("source_bubbles") or 0), int(hybrid_visual.get("target_bubbles") or 0)))
            cache_stats["hybrid_admission_strong_records"] = str(int(hybrid_visual.get("strong_record_count") or 0))
            if trace is not None:
                trace.event(
                    "hybrid_admission_continue_without_ocr",
                    visual_evidence=hybrid_visual,
                    reason="paired_visual_evidence_available",
                )
    if (bool(config.page_management.skip_transfer_when_source_has_no_text_boxes)
            and no_transferable_source_text):
        no_text_mark = PageMark(
            page_type="content", origin=mark.origin if mark.origin in {"default", "manual"} else "default",
            confidence=1.0, reason="source_no_transferable_chinese_text_box;keep_target_unchanged",
            bubble_regions=0, free_text_regions=0,
            registration_confidence=float(registration.confidence),
            source_name=Path(pair.source_path).name, target_name=Path(pair.target_path).name,
        )
        emit_progress(100, "content_gate", "SOURCE 无可迁移文字，保持 TARGET")
        _stage_done("text_ocr", stage_started)
        return _attach_timings(passthrough_page(
            pair, page_root, final_path, no_text_mark, source=source, target=target,
            registration=registration, passthrough_reason="source_no_transferable_text",
        ))

    if trace is not None:
        trace.event(
            "ocr_and_bubbles_done",
            ocr_source=cache_stats.get("ocr_source", ""), ocr_target=cache_stats.get("ocr_target", ""),
            source_blocks=len(source_blocks), target_blocks=len(target_blocks),
            source_bubbles=len(source_bubbles), target_bubbles=len(target_bubbles),
            target_driven_regions=bool(target_driven_reletter_regions),
            recognized_regions=sum(bool(str(b.text or "").strip()) for b in source_blocks),
        )
    check_cancel(cancel_cb, "after_ocr_and_bubbles")
    emit_progress(73, "text", "文字 / 气泡候选完成")
    _stage_done("text_ocr", stage_started)
    mask_source_bubbles = paired_diff.source_bubbles if use_paired_diff else source_bubbles
    mask_target_bubbles = paired_diff.target_bubbles if use_paired_diff else target_bubbles

    stage_started = time.perf_counter()
    emit_progress(76, "matching", "正在建立 SOURCE / TARGET 文字对应")
    match_stage = run_text_matching_stage(
        pair, registration, source_blocks, target_blocks, source_bubbles, target_bubbles,
        config=config, mode_contract=mode_contract, mode=mode,
        use_paired_diff=use_paired_diff,
        target_driven_reletter_regions=target_driven_reletter_regions,
    )
    source_units = match_stage.source_units
    target_units = match_stage.target_units
    match_result = match_stage.match_result
    matches = match_stage.matches
    accepted = match_stage.accepted

    # v2.0.90: OCR/text matching remains a content layer underneath Koharu's
    # first semantic authority.  OCR-confirmed matches may fill UNKNOWN gaps,
    # but they may not destructively override a positive panel/artwork PROTECT
    # decision on either edition.
    if accepted and bool(mode_contract.may_use_ocr) and koharu_is_primary(config.bubbles):
        cache_enabled_layout = bool(config.cache.bubbles) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True))
        src_authority = collect_koharu_layout_evidence_cached(
            source, config.bubbles, role="ocr_match_authority_source", image_path=source_path_local,
            cache=stage_cache, cache_enabled=cache_enabled_layout, stats=cache_stats, allow_missing=True,
        )
        tgt_authority = collect_koharu_layout_evidence_cached(
            target, config.bubbles, role="ocr_match_authority_target", image_path=target_path_local,
            cache=stage_cache, cache_enabled=cache_enabled_layout, stats=cache_stats, allow_missing=True,
        )
        su_by = {u.id: u for u in source_units}
        tu_by = {u.id: u for u in target_units}
        authority_kept = []
        authority_rejected = []
        for m in accepted:
            su = su_by.get(m.source_unit_id); tu = tu_by.get(m.target_unit_id)
            if su is None or tu is None:
                continue
            sk = _layout_region_kind_for_unit(su)
            tk = _layout_region_kind_for_unit(tu)
            sd = classify_layout_authority(src_authority, su, source.shape[:2], region_kind=sk, cfg=config.mask_replace)
            td = classify_layout_authority(tgt_authority, tu, target.shape[:2], region_kind=tk, cfg=config.mask_replace)
            if sd.state == "PROTECT" or td.state == "PROTECT":
                if _paired_ocr_authority_rescue(
                    mode, m, su, tu, sd, td, float(registration.confidence)
                ):
                    authority_kept.append(m)
                    authority_rejected.append({
                        "source_unit_id": su.id, "target_unit_id": tu.id,
                        "source": sd.to_dict(), "target": td.to_dict(),
                        "rescued": True,
                        "reason": "paired_ocr_false_protect_recovered",
                    })
                    continue
                authority_rejected.append({
                    "source_unit_id": su.id, "target_unit_id": tu.id,
                    "source": sd.to_dict(), "target": td.to_dict(),
                    "rescued": False,
                })
                continue
            authority_kept.append(m)
        accepted = authority_kept
        cache_stats["ocr_match_authority"] = "koharu_first"
        rejected_rows = [x for x in authority_rejected if not bool(x.get("rescued"))]
        rescued_rows = [x for x in authority_rejected if bool(x.get("rescued"))]
        cache_stats["ocr_match_authority_rejected"] = str(len(rejected_rows))
        cache_stats["ocr_match_authority_rescued"] = str(len(rescued_rows))
        if trace is not None and rejected_rows:
            trace.event("ocr_match_authority_rejected", count=len(rejected_rows), rows=rejected_rows[:12])
        if trace is not None and rescued_rows:
            trace.event("ocr_match_authority_rescued", count=len(rescued_rows), rows=rescued_rows[:12])

    if trace is not None:
        trace.event(
            "matching_done", route=str((getattr(match_result, "diagnostics", {}) or {}).get("route") or "geometric"),
            total_matches=len(matches), accepted_matches=len(accepted),
            paired_reletter_binding=bool(match_stage.paired_reletter_binding),
            pair_confidence_bypassed=bool(match_stage.pair_confidence_bypassed),
            pair_confidence=float(pair.confidence),
            registration_confidence=float(registration.confidence),
            unmatched_source=len(getattr(match_result, "unmatched_source", []) or []),
            unmatched_target=len(getattr(match_result, "unmatched_target", []) or []),
        )

    _stage_done("matching", stage_started)
    stage_started = time.perf_counter()
    emit_progress(82, "transfer", "正在生成安全迁移 / 蒙版")
    transfer_stage = run_transfer_execution_stage(
        config=config, pair=pair, registration=registration, pair_check=pair_check,
        mode=mode, mode_contract=mode_contract, source=source, target=target,
        source_blocks=source_blocks, target_blocks=target_blocks,
        source_units=source_units, target_units=target_units, matches=matches, accepted=accepted,
        paired_diff=paired_diff, direct_container_fast=direct_container_fast,
        direct_container_plan=direct_container_plan,
        mask_source_bubbles=mask_source_bubbles, mask_target_bubbles=mask_target_bubbles,
        source_bubbles=source_bubbles, target_bubbles=target_bubbles,
        target_driven_reletter_regions=target_driven_reletter_regions,
        target_driven_reletter_diagnostics=target_driven_reletter_diagnostics,
        check_cancel=check_cancel, cancel_cb=cancel_cb, trace=trace,
        stage_cache=stage_cache, cache_stats=cache_stats,
        source_path=str(source_path_local), target_path=str(target_path_local),
    )
    mask_transfer = transfer_stage.mask_transfer
    unseeded_white_pair_count = transfer_stage.unseeded_white_pair_count
    completion_display_source = transfer_stage.completion_display_source
    completion_display_target = transfer_stage.completion_display_target
    transfer_rgba = transfer_stage.transfer_rgba
    fallback_matches = transfer_stage.fallback_matches
    rendered = transfer_stage.rendered
    inpaint_result = transfer_stage.inpaint_result
    mask_result = transfer_stage.mask_result
    lettering = transfer_stage.lettering
    lettering_masks = transfer_stage.lettering_masks
    constrained_layout_units = transfer_stage.constrained_layout_units
    target_layout_hint_units = transfer_stage.target_layout_hint_units
    qa = transfer_stage.qa
    # Keep the mode-owned Chinese foreground before any TARGET-based cleanup
    # can be baked into a transfer layer. Otherwise missed Japanese can later
    # be mistaken for translated foreground during final composition.
    white_bubble_foreground = transfer_rgba.copy() if transfer_rgba is not None else None

    # Koharu's disconnected foreground text masks survive OCR-free and OCR modes
    # alike. Apply only the flat-background cleanup policy here, after each
    # mode's own renderer and before the common artifact/export boundary. This
    # does not call an OCR backend or alter the mode's matching/transfer logic.
    koharu_text_blocks = []
    if koharu_target_evidence is not None and bool(getattr(koharu_target_evidence, "available", False)):
        koharu_text_blocks = [
            block for block in koharu_target_evidence.text_blocks(
                include_sfx=True, backend_name="koharu_layout", target_only=True,
            )
            if str(getattr(block, "kind", "") or "").strip().lower() in {"speech", "sfx"}
        ]
    # SOURCE text remains authoritative even when a common CJK glyph or numeral
    # happens to have identical TARGET pixels. Pixel difference alone cannot
    # distinguish an untranslated glyph from already correct shared lettering.
    source_ink = koharu_semantic_text_ink_mask(
        source, koharu_source_evidence, labels=("text", "sfx"),
        min_confidence=0.50, pad_px=3, dilate_px=2,
    )
    protected_source_ink = cv2.warpPerspective(
        source_ink, np.asarray(registration.matrix, dtype=np.float64),
        (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    # Protect lettering, not the SOURCE detector's surrounding paper corridor.
    protected_source_ink[cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY) > 210] = 0
    koharu_cleanup = clean_koharu_text_on_flat_backgrounds(
        target, rendered, inpaint_result.image, koharu_text_blocks,
        protected_mask=protected_source_ink,
        aligned_source=cv2.warpPerspective(
            source, np.asarray(registration.matrix, dtype=np.float64),
            (target.shape[1], target.shape[0]), borderValue=(255, 255, 255),
        ),
        inpaint_callback=(
            lama_inpaint_callback
            if lama_status.ready else None
        ),
    )
    rendered = koharu_cleanup.rendered
    if int(koharu_cleanup.diagnostics.get("applied", 0) or 0) > 0:
        inpaint_result.image = koharu_cleanup.editable_base
        cache_stats["koharu_text_cleanup"] = str(koharu_cleanup.diagnostics["applied"])
        if trace is not None:
            trace.event("koharu_text_cleanup", diagnostics=koharu_cleanup.diagnostics)
    elif koharu_cleanup.diagnostics.get("candidates"):
        cache_stats["koharu_text_cleanup"] = "deferred"
        if trace is not None:
            trace.event("koharu_text_cleanup", diagnostics=koharu_cleanup.diagnostics)
    if int(koharu_cleanup.diagnostics.get("needs_model_inpainting", 0) or 0) > 0:
        # Do not silently report a clean page when Koharu found high-confidence
        # foreground text over a textured/color background that this conservative
        # stage intentionally left untouched. The dedicated inpainting model is
        # optional and may not be installed on the host.
        qa.append(QAItem(
            "koharu_text_cleanup_deferred", "warning",
            "Koharu detected Japanese text/SFX on a non-uniform background, but no safe flat fill was applied; Japanese may remain. Use a configured inpainting model or review this region manually.",
            meta={
                "needs_model_inpainting": int(koharu_cleanup.diagnostics.get("needs_model_inpainting", 0) or 0),
                "unresolved_foreground_pixels": int(koharu_cleanup.diagnostics.get("unresolved_foreground_pixels", 0) or 0),
                "model_inpainting_error": str(koharu_cleanup.diagnostics.get("model_inpainting_error", "") or ""),
                "runner_status": lama_status.detail,
                "rows": [
                    row for row in koharu_cleanup.diagnostics.get("rows", [])
                    if str(row.get("status", "")) == "skipped_non_uniform_background"
                ],
            },
        ))

    target_first = None
    if mode in {"direct_patch", "mask_replace", "auto"} and mask_transfer is not None:
        source_letters = koharu_semantic_text_ink_mask(
            source, koharu_source_evidence, labels=("text",),
            min_confidence=0.35, pad_px=4, dilate_px=2,
        )
        matrix = np.asarray(registration.matrix, dtype=np.float64)
        aligned_letters = cv2.warpPerspective(source_letters, matrix,
            (target.shape[1], target.shape[0]), flags=cv2.INTER_NEAREST)
        aligned_raster = cv2.warpPerspective(source, matrix,
            (target.shape[1], target.shape[0]), borderValue=(255, 255, 255))
        target_first = compose_coloured_containers(
            target, rendered, aligned_raster, aligned_letters,
            mask_transfer.records, lama_inpaint_callback,
        )
        if target_first.diagnostics.get("applied", 0):
            rendered = target_first.image
            use = target_first.authority > 0
            inpaint_result.image[use] = target_first.clean_base[use]
            mask_transfer.image = rendered
            rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
            mask_transfer.layer_rgba[use, :3] = rgb[use]
            mask_transfer.layer_rgba[use, 3] = 255
            mask_transfer.composite_mask[use] = 255
            if mask_transfer.clear_mask is not None:
                mask_transfer.clear_mask[use] = target_first.clear_mask[use]
            transfer_rgba = mask_transfer.layer_rgba
            koharu_cleanup.clear_mask[use] = target_first.clear_mask[use]
        colour_rows = target_first.diagnostics.get("rows", [])
        unresolved = [row for row in colour_rows if str(row.get("status", "")).startswith("deferred_")]
        residual = [row for row in colour_rows if row.get("status") == "applied_target_first_lama"
                    and (int(row.get("unchanged_target_dark_pixels_outside_source_ink", 0)) > 16
                         or int(row.get("clean_base_dark_pixels", 0)) > 16)]
        if unresolved or residual:
            qa.append(QAItem(
                "target_first_colour_review", "error",
                "Some coloured containers could not be independently cleaned or still have dark foreground in the clean TARGET base; manual review is required.",
                meta={"deferred": unresolved, "possible_residual": residual},
            ))
        if trace is not None:
            trace.event("target_first_colour", diagnostics=target_first.diagnostics)

    _stage_done("transfer", stage_started)
    stage_started = time.perf_counter()
    emit_progress(91, "composition", "正在合成迁移结果")
    composition = compose_transfer_state(
        pair=pair, registration=registration, pair_check=pair_check, decision=decision,
        mode=mode, config=config, cache_stats=cache_stats,
        source_blocks=source_blocks, target_blocks=target_blocks,
        source_units=source_units, target_units=target_units, matches=matches,
        accepted=accepted, match_result=match_result, paired_diff=paired_diff,
        mask_transfer=mask_transfer, direct_container_fast=direct_container_fast,
        direct_container_plan=direct_container_plan,
        unseeded_white_pair_count=unseeded_white_pair_count,
        fallback_matches=fallback_matches, lettering=lettering,
        source_bubbles=source_bubbles, target_bubbles=target_bubbles,
        completion_display_source=completion_display_source,
        completion_display_target=completion_display_target, qa=qa,
    )
    source_bubbles = composition.source_bubbles
    target_bubbles = composition.target_bubbles

    project = assemble_page_project(
        config=config, pair=pair, mark=mark, registration=registration,
        source_blocks=source_blocks, target_blocks=target_blocks,
        source_units=source_units, target_units=target_units, matches=matches,
        lettering=lettering, qa=qa, mode=mode, mode_contract=mode_contract,
        stale_mode_cleanup=stale_mode_cleanup, review_mode_archive=review_mode_archive,
        fallback_matches=fallback_matches, mask_transfer=mask_transfer,
        direct_container_fast=direct_container_fast, direct_container_plan=direct_container_plan,
        use_paired_diff=use_paired_diff, paired_diff=paired_diff,
        target_driven_reletter_regions=target_driven_reletter_regions,
        target_driven_reletter_diagnostics=target_driven_reletter_diagnostics,
        constrained_layout_units=constrained_layout_units,
        target_layout_hint_units=target_layout_hint_units,
        inpaint_result=inpaint_result, mask_result=mask_result, match_result=match_result,
        decision=decision, pair_check=pair_check, cache_stats=cache_stats,
        authority_source_path=authority_source_path, source_path_local=source_path_local,
        selected_source_kind=selected_source_kind,
        secondary_source_available=secondary_source_available,
        selected_secondary_source=selected_secondary_source,
        dual_source_arbitration=dual_source_arbitration,
        selected_arbitration_evidence=selected_arbitration_evidence,
        replace_source_specs=replace_source_specs, composition=composition,
    )
    project.meta["koharu_text_cleanup"] = koharu_cleanup.diagnostics
    from .white_bubble_composite import compose_white_bubbles
    from .export import make_text_layer_rgba
    lcfg = config.hybrid.lettering if mode == "hybrid" else config.reletter.lettering if mode == "reletter" else config.lettering
    white_bubbles = compose_white_bubbles(
        target, rendered, white_bubble_foreground, target_bubbles,
        text_rgba=make_text_layer_rgba(target.shape[:2], lettering_masks, color=lcfg.fill),
    )
    project.meta["white_bubble_blank_first"] = white_bubbles.diagnostics
    if white_bubbles.diagnostics["applied"]:
        rendered = white_bubbles.image
        use = white_bubbles.clear_mask > 0
        inpaint_result.image[use] = 255
        mask_result.mask[use] = 255
        koharu_cleanup.clear_mask[use] = 255
        if white_bubbles.layer_rgba is not None:
            if transfer_rgba is None:
                transfer_rgba = white_bubbles.layer_rgba
            else:
                transfer_rgba[use] = white_bubbles.layer_rgba[use]
        if mask_transfer is not None:
            mask_transfer.image = rendered
            if white_bubbles.layer_rgba is not None:
                mask_transfer.layer_rgba[use] = white_bubbles.layer_rgba[use]
                mask_transfer.composite_mask[use] = 255
            if mask_transfer.clear_mask is not None:
                mask_transfer.clear_mask[use] = 255
        if trace is not None:
            trace.event("white_bubble_blank_first", diagnostics=white_bubbles.diagnostics)
    if target_first is not None:
        project.meta["target_first_colour"] = target_first.diagnostics
        if target_first.diagnostics.get("applied", 0):
            from .io_utils import write_image
            write_image(page_root / "target_clean_base.png", inpaint_result.image)
            write_image(page_root / "target_removal_mask.png", target_first.clear_mask)
            write_image(page_root / "source_ink_layer.png", target_first.source_layer)
            for name in ("target_clean_base", "target_removal_mask", "source_ink_layer"):
                project.artifacts[name] = str(page_root / (name + ".png"))

    _stage_done("composition_assembly", stage_started)
    check_cancel(cancel_cb, "before_export")
    stage_started = time.perf_counter()
    emit_progress(96, "export", "正在写入页面结果与 QA")
    export_page_artifacts(
        config=config, project=project, pair=pair, page_root=page_root,
        final_path=final_path, source=source, authority_source=authority_source,
        target=target, rendered=rendered, inpaint_result=inpaint_result,
        koharu_text_clear_mask=koharu_cleanup.clear_mask,
        koharu_text_cleanup_diagnostics=koharu_cleanup.diagnostics,
        mask_result=mask_result, lettering_masks=lettering_masks,
        mask_transfer=mask_transfer, direct_container_fast=direct_container_fast,
        transfer_rgba=transfer_rgba, transfer_audit=composition.transfer_audit,
        target_bubbles=target_bubbles, source_units=source_units, target_units=target_units,
        matches=matches, source_blocks=source_blocks, target_blocks=target_blocks,
        registration=registration, paired_diff=paired_diff, decision=decision,
        pair_check=pair_check, direct_container_plan=direct_container_plan,
        authority_source_path=authority_source_path, source_path_local=source_path_local,
        selected_source_kind=selected_source_kind,
        secondary_source_available=secondary_source_available,
        selected_secondary_source=selected_secondary_source,
        dual_source_arbitration=dual_source_arbitration,
        selected_arbitration_evidence=selected_arbitration_evidence,
        target_path_local=target_path_local, match_result=match_result,
    )
    _stage_done("export", stage_started)
    _attach_timings(project)
    stage_started = time.perf_counter()
    finalized = finalize_page_project(
        config=config, project=project, page_root=page_root, mode=mode,
        decision=decision, direct_container_fast=direct_container_fast,
        mask_transfer=mask_transfer, lettering=lettering,
    )
    _stage_done("finalize", stage_started)
    _attach_timings(finalized)
    emit_progress(100, "done", "页面处理完成")
    return finalized
