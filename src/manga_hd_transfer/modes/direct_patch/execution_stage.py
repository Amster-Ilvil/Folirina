from __future__ import annotations

"""Transfer renderer orchestration extracted from the main page pipeline.

Only sequencing, completion/fallback policy, clear-mask/inpaint coordination and
mode QA live here. Stable Direct/Mask/Reletter pixel renderer implementations
remain in their original modules and are called without algorithm rewrites.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...geometry import rasterize_polygon
from ...models import BubbleInstance, QAItem
from ...qa import run_direct_patch_qa, run_mask_replace_qa, run_page_qa
from .pixel_stage import run_pixel_transfer_stage, _merge_mask_transfer
from ...pipeline_transfer_authority import (
    collect_transfer_layout_authority,
    _filter_completion_pairs_by_koharu_authority,
)
from ...pipeline_transfer_qa import (
    _append_koharu_semantic_coverage_qa,
    append_photo_pair_evidence_qa,
)




def _resolve_reletter_layout_policy(mode: str):
    def _apply(*args, **kwargs):
        return None
    def _project(*args, **kwargs):
        return None
    def _orientation(value, *args, **kwargs):
        return value
    return _apply, _project, _orientation


def _resolve_mode_runtime_primitives(mode: str, *, direct_container_fast: bool):
    from . import masking_ops, inpainting_ops
    return "direct_patch", masking_ops, inpainting_ops, None, None


@dataclass
class TransferExecutionState:
    mask_transfer: Any
    unseeded_white_pair_count: int
    completion_display_source: list[BubbleInstance]
    completion_display_target: list[BubbleInstance]
    transfer_rgba: np.ndarray
    fallback_matches: list[Any]
    rendered: np.ndarray
    inpaint_result: Any
    mask_result: Any
    lettering: list[Any]
    lettering_masks: list[np.ndarray]
    constrained_layout_units: int
    target_layout_hint_units: int
    qa: list[QAItem]






# Compatibility alias for tests/plugins written against the v2.3.11 symbol.
# Semantics are now the stronger OCR paper-first implementation.










def run_transfer_execution_stage(
    *, config, pair, registration, pair_check, mode, mode_contract, source, target,
    source_blocks, target_blocks, source_units, target_units, matches, accepted,
    paired_diff, direct_container_fast, direct_container_plan, mask_source_bubbles,
    mask_target_bubbles, source_bubbles, target_bubbles,
    target_driven_reletter_regions, target_driven_reletter_diagnostics,
    check_cancel, cancel_cb=None, trace=None, stage_cache=None, cache_stats: dict | None = None,
    source_path: str | None = None, target_path: str | None = None,
) -> TransferExecutionState:
    if str(mode or "").strip().lower() not in {"direct_patch", "auto"}:
        raise RuntimeError("direct_patch execution stage cannot execute mode=" + str(mode))
    _apply_target_layout_hints, _project_source_profile_mask, _reletter_orientation = _resolve_reletter_layout_policy(mode)
    runtime_owner, masking_ops, inpainting_ops, lettering_ops, ocr_cleanup_ops = _resolve_mode_runtime_primitives(
        mode, direct_container_fast=direct_container_fast
    )
    MaskBuildResult = masking_ops.MaskBuildResult
    build_clear_mask = masking_ops.build_clear_mask
    InpaintResult = inpainting_ops.InpaintResult
    inpaint_image = inpainting_ops.inpaint_image
    if lettering_ops is not None:
        composite_text = lettering_ops.composite_text
        fit_text = lettering_ops.fit_text
        polygon_safe_mask = lettering_ops.polygon_safe_mask
        textbox_safe_mask = lettering_ops.textbox_safe_mask
    else:
        composite_text = fit_text = polygon_safe_mask = textbox_safe_mask = None
    _ocr_paper_first_clear = getattr(ocr_cleanup_ops, "_ocr_paper_first_clear", None) if ocr_cleanup_ops is not None else None
    if cache_stats is not None:
        cache_stats["active_runtime_capsule"] = runtime_owner
    # Semantic authority is evidence-only and lives behind a dedicated boundary.
    # Auxiliary detectors never gain destructive ownership.
    authority = collect_transfer_layout_authority(
        config=config, source=source, target=target, stage_cache=stage_cache,
        cache_stats=cache_stats, source_path=source_path, target_path=target_path,
    )
    source_layout_authority = authority.source
    target_layout_authority = authority.target

    check_cancel(cancel_cb, "before_transfer")
    pixel_state = run_pixel_transfer_stage(
        config=config, pair_check=pair_check, registration=registration, mode=mode,
        mode_contract=mode_contract, source=source, target=target, source_blocks=source_blocks,
        target_blocks=target_blocks, source_units=source_units, target_units=target_units,
        matches=matches, accepted=accepted, paired_diff=paired_diff,
        direct_container_fast=direct_container_fast, direct_container_plan=direct_container_plan,
        mask_source_bubbles=mask_source_bubbles, mask_target_bubbles=mask_target_bubbles,
        target_bubbles=target_bubbles, source_layout_authority=source_layout_authority,
        target_layout_authority=target_layout_authority, stage_cache=stage_cache,
        cache_stats=cache_stats, target_path=target_path,
    )
    mask_transfer = pixel_state.mask_transfer
    unseeded_white_pair_count = pixel_state.unseeded_white_pair_count
    completion_display_source = pixel_state.completion_display_source
    completion_display_target = pixel_state.completion_display_target
    transfer_rgba = pixel_state.transfer_rgba
    fallback_matches = pixel_state.fallback_matches
    semantic_layout_evidence = pixel_state.semantic_layout_evidence

    constrained_layout_units = 0
    target_layout_hint_units = 0
    # Strict pixel modes never become OCR text renderers just because no safe
    # pixel transfer was found. Auto may orchestrate Direct→Mask, but Reletter
    # is a separate explicit mode. Preserve TARGET and surface QA/review instead.
    if mode in {"auto", "mask_replace"} and mask_transfer is None:
        fallback_matches = []
        operation_mask = np.zeros(target.shape[:2], np.uint8)
        rendered = target.copy()
        mask_result = MaskBuildResult(
            mask=operation_mask, per_unit={}, clipped_pixels=0, source_pixels=0,
        )
        inpaint_result = InpaintResult(target.copy(), f"{mode}-no-safe-pixel-transfer", {"applied": 0})
        lettering = []
        lettering_masks = []
        qa = run_mask_replace_qa(
            pair, registration, source_units, mask_source_bubbles, [],
            config.qa, config.mask_replace,
        )
        qa.append(QAItem(
            "strict_mode_no_safe_transfer", "warning",
            "No safe pixel/mask transfer was found; TARGET was preserved. Reletter was not invoked because the selected mode is isolated.",
            meta={"requested_mode": mode},
        ))
    elif mode in {"auto", "direct_patch", "mask_replace"} and not fallback_matches:
        assert mask_transfer is not None
        rendered = mask_transfer.image.copy()
        operation_mask = mask_transfer.composite_mask
        mask_result = MaskBuildResult(
            mask=operation_mask,
            per_unit={},
            clipped_pixels=0,
            source_pixels=int(cv2.countNonZero(operation_mask)),
        )
        # No Japanese inpainting is needed: the aligned Chinese bubble interior
        # itself covers the Japanese text. Keep the untouched HD page as base layer.
        inpaint_method = "direct-patch-no-inpaint" if direct_container_fast else "mask-replace-no-inpaint"
        inpaint_result = InpaintResult(target.copy(), inpaint_method, {"applied": mask_transfer.applied_count})
        lettering = []
        lettering_masks: list[np.ndarray] = []
        if direct_container_fast:
            qa = run_direct_patch_qa(
                pair, registration, source_units, mask_source_bubbles, mask_transfer.records,
                config.qa, config.direct_patch,
            )
        else:
            qa = run_mask_replace_qa(
                pair, registration, source_units, mask_source_bubbles, mask_transfer.records,
                config.qa, config.mask_replace,
            )
        if mode == "direct_patch" and direct_container_plan is not None and not direct_container_plan.safe_to_skip_other_paths:
            qa.append(QAItem(
                "direct_patch_partial_review", "warning",
                "Direct Patch applied only regions that passed the whole-raster safety gate, but one or more container-like regions were skipped for review.",
                meta={"diagnostics": dict(direct_container_plan.diagnostics)},
            ))
    else:
        base = mask_transfer.image.copy() if mask_transfer is not None else target
        mask_result = build_clear_mask(
            target.shape[:2],
            target_blocks,
            target_units,
            target_bubbles,
            fallback_matches,
            config.masking,
            min_match_confidence=config.matching.review_confidence,
            allow_relations={"one_to_one"},
            target_image=target, current_image=base,
        )
        clear_coverage_rows = []
        if mode == "reletter" and target_driven_reletter_regions:
            tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
            for block in target_blocks:
                if not bool((block.meta or {}).get("synthetic_region_only")):
                    continue
                region_diag = (block.meta or {}).get("region_diagnostics") or {}
                component_mask = np.zeros(target.shape[:2], np.uint8)
                for box in list(region_diag.get("component_bboxes") or []):
                    if not isinstance(box, (list, tuple)) or len(box) != 4:
                        continue
                    x0, y0, x1, y1 = [int(v) for v in box]
                    x0=max(0,min(target.shape[1],x0)); x1=max(0,min(target.shape[1],x1))
                    y0=max(0,min(target.shape[0],y0)); y1=max(0,min(target.shape[0],y1))
                    if x1>x0 and y1>y0:
                        component_mask[y0:y1, x0:x1] = 255
                if cv2.countNonZero(component_mask) == 0:
                    component_mask = rasterize_polygon(block.polygon, target.shape[:2])
                dark = (tg < 190) & (component_mask > 0)
                dark_n = int(np.count_nonzero(dark))
                covered_n = int(np.count_nonzero(dark & (mask_result.mask > 0)))
                missing_ratio = float(max(0, dark_n - covered_n) / max(1, dark_n))
                clear_coverage_rows.append({
                    "region_id": str((block.meta or {}).get("reletter_region_id") or block.id),
                    "target_text_dark_pixels": dark_n, "covered_target_text_dark_pixels": covered_n,
                    "missing_target_text_ratio": round(missing_ratio, 4),
                })
            if isinstance(target_driven_reletter_diagnostics, dict):
                target_driven_reletter_diagnostics["clear_coverage"] = clear_coverage_rows
        if trace is not None:
            trace.event(
                "clear_mask_built", clear_pixels=int(cv2.countNonZero(mask_result.mask)),
                per_unit_masks=len(mask_result.per_unit), clipped_ratio=float(mask_result.clipped_ratio),
                clear_coverage=clear_coverage_rows,
                max_missing_target_text_ratio=max([float(r.get("missing_target_text_ratio", 0.0)) for r in clear_coverage_rows] or [0.0]),
            )
        if mode in {"reletter", "hybrid"}:
            # OCR product routes only: restore proven balloon/text-box paper
            # directly from TARGET instead of asking interpolation to invent it.
            # This prevents grey shadows while keeping coloured/artwork regions on
            # the configured inpainting backend. No other transfer mode enters here.
            paper_base, remaining_inpaint_mask, paper_diag = _ocr_paper_first_clear(
                base, target, mask_result, target_units, target_bubbles,
            )
            inpaint_result = inpaint_image(paper_base, remaining_inpaint_mask, config.inpainting)
            inpaint_result.diagnostics.update({"ocr_paper_first": True, "ocr_paper_mode": mode, **paper_diag})
            if trace is not None:
                trace.event("ocr_paper_first_clear", mode=mode, **paper_diag)
        else:
            inpaint_result = inpaint_image(base, mask_result.mask, config.inpainting)
        rendered = inpaint_result.image.copy()

        source_by_id = {u.id: u for u in source_units}
        target_by_id = {u.id: u for u in target_units}
        bubbles_by_id = {b.id: b for b in target_bubbles}
        lettering = []
        lettering_masks = []
        constrained_layout_units = 0
        target_layout_hint_units = 0
        for match in fallback_matches:
            check_cancel(cancel_cb, "reletter_render_region")
            src = source_by_id[match.source_unit_id]
            dst = target_by_id[match.target_unit_id]
            safe = None
            if dst.bubble_id and dst.bubble_id in bubbles_by_id:
                safe = bubbles_by_id[dst.bubble_id].safe_mask
            if safe is None or cv2.countNonZero(safe) == 0:
                safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, config.bubbles.safe_margin_px // 2))
            lcfg = config.lettering.model_copy(deep=True)
            source_block_by_id = {b.id: b for b in source_blocks}
            target_block_by_id = {b.id: b for b in target_blocks}
            lcfg.orientation = _reletter_orientation(lcfg.orientation, src, source_block_by_id)
            base_safe = None if safe is None else safe.copy()
            target_driven_unit = bool((dst.meta or {}).get("geometry") == "reletter_text_region")
            if target_driven_unit and base_safe is not None:
                # Hard layout fence: text may expand around the original JP
                # glyph island, but never drift across the parent balloon.
                region_limit = rasterize_polygon(dst.polygon, target.shape[:2])
                if cv2.countNonZero(region_limit) > 0:
                    rb = dst.bbox
                    pad = max(3, int(round(min(max(1.0, rb[2]-rb[0]), max(1.0, rb[3]-rb[1])) * 0.18)))
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*pad+1, 2*pad+1))
                    region_limit = cv2.dilate(region_limit, k, iterations=1)
                    limited = cv2.bitwise_and(base_safe, region_limit)
                    if cv2.countNonZero(limited) > 0:
                        base_safe = limited
                        safe = limited.copy()
            clear_mask = mask_result.per_unit.get(dst.id)
            source_profile_mask = _project_source_profile_mask(src, dst, source_block_by_id, target.shape[:2])
            clear_for_layout = clear_mask
            if source_profile_mask is not None and cv2.countNonZero(source_profile_mask) > 0:
                if clear_for_layout is None or cv2.countNonZero(clear_for_layout) == 0:
                    clear_for_layout = source_profile_mask
                elif base_safe is not None:
                    safe_area = max(1, cv2.countNonZero(base_safe))
                    clear_area = cv2.countNonZero(clear_for_layout)
                    # Synthetic target-mask recovery can be missing or overly broad
                    # on photo pairs; fall back to the source-projected textbox when
                    # the target clear mask does not localize the original text box.
                    if clear_area >= safe_area * 0.78:
                        clear_for_layout = source_profile_mask
                    else:
                        union = cv2.bitwise_or(clear_for_layout, source_profile_mask)
                        if cv2.countNonZero(union) > 0:
                            clear_for_layout = union
            constrained = textbox_safe_mask(base_safe, clear_for_layout, orientation=lcfg.orientation)
            layout_mode = str(getattr(lcfg, "layout_mode", "smart_scaling") or "smart_scaling").lower()
            if constrained is not None and cv2.countNonZero(constrained) > 0:
                if base_safe is None or cv2.countNonZero(cv2.bitwise_xor((constrained > 0).astype(np.uint8) * 255, (base_safe > 0).astype(np.uint8) * 255)) > 0:
                    constrained_layout_units += 1
                safe = constrained
            # Layout policy is independent from line breaking. Strict keeps the
            # original TARGET text box as a hard fence. Smart Scaling starts
            # there and may expand only on failure. Balloon Fill uses the parent
            # bubble-safe mask up front, but keeps TARGET anchor/shape hints so
            # the text does not simply drift to the bubble centroid.
            if layout_mode == "balloon_fill" and base_safe is not None and cv2.countNonZero(base_safe) > 0:
                safe = base_safe.copy()
            # Recover typography from the translated source scan. OCR identifies
            # Unicode only; it must not invent a new font scale/column count.
            profiles = [
                source_block_by_id[bid].meta.get("source_layout_profile")
                for bid in src.block_ids if bid in source_block_by_id
                and source_block_by_id[bid].meta.get("source_layout_profile")
            ]
            source_predicted_font = 0
            if profiles:
                profile = profiles[0]
                if lcfg.orientation == "vertical" and int(profile.get("columns") or 0) > 0:
                    lcfg.preferred_columns = int(profile["columns"])
                pitch = float(profile.get("glyph_pitch_px") or 0.0)
                if pitch > 0:
                    sx0, sy0, sx1, sy1 = src.bbox
                    sbw, sbh = max(1.0, sx1-sx0), max(1.0, sy1-sy0)
                    if target_driven_unit:
                        dx0, dy0, dx1, dy1 = dst.bbox
                        dw, dh = max(1.0, dx1-dx0), max(1.0, dy1-dy0)
                        scale = min(max(0.20, dw / sbw), max(0.20, dh / sbh))
                        source_predicted_font = int(round(pitch * scale))
                    else:
                        safe_box = cv2.boundingRect((safe > 0).astype(np.uint8))
                        _x, _y, sw, sh = safe_box
                        scale = min(max(0.25, sw / sbw), max(0.25, sh / sbh))
                        predicted = int(round(pitch * scale))
                        lcfg.preferred_font_size = int(np.clip(predicted, lcfg.min_font_size, lcfg.max_font_size))
            # v2.0.35: on target-driven regions, the clean Japanese master is
            # authoritative for intended column count and approximate glyph
            # pitch. This is more stable than estimating typography from a
            # photographed Chinese source. Source hints remain the fallback.
            if target_driven_unit:
                tdiags = []
                for bid in dst.block_ids:
                    block = target_block_by_id.get(bid)
                    if block is None:
                        continue
                    d = (block.meta or {}).get("region_diagnostics") or {}
                    if isinstance(d, dict):
                        tdiags.append(d)
                if tdiags:
                    td = tdiags[0]
                    tcols = int(td.get("estimated_columns") or 0)
                    tpitch = float(td.get("target_glyph_pitch_px") or 0.0)
                    if lcfg.orientation == "vertical" and tcols > 0:
                        # Use target columns first, but allow fit_text to explore
                        # neighbouring column counts instead of hard-locking.
                        lcfg.preferred_columns = int(np.clip(tcols, 1, lcfg.max_lines))
                    candidates = []
                    if tpitch > 0:
                        candidates.append(int(round(tpitch * 1.10)))
                    if source_predicted_font > 0:
                        candidates.append(int(round(source_predicted_font * 0.92)))
                    if candidates:
                        predicted = max(candidates)
                        # Never let a noisy source-photo estimate exceed most of
                        # the target text-region short side.
                        dx0, dy0, dx1, dy1 = dst.bbox
                        regional_cap = max(lcfg.min_font_size, int(round(min(max(1.0, dx1-dx0), max(1.0, dy1-dy0)) * 0.72)))
                        predicted = min(predicted, regional_cap)
                        lcfg.preferred_font_size = int(np.clip(predicted, lcfg.min_font_size, lcfg.max_font_size))
                    target_layout_hint_units += 1
            if target_driven_unit:
                _apply_target_layout_hints(lcfg, dst, safe)
            result = fit_text(target.shape[:2], safe, dst, src.text, lcfg)
            if (layout_mode != "strict" and not result.success and base_safe is not None and safe is not None
                    and cv2.countNonZero(cv2.bitwise_xor((safe > 0).astype(np.uint8) * 255, (base_safe > 0).astype(np.uint8) * 255)) > 0):
                retry = fit_text(target.shape[:2], base_safe, dst, src.text, lcfg)
                if retry.success:
                    result = retry
            lettering.append(result)
            if trace is not None:
                trace.event(
                    "reletter_region", region_id=str((dst.meta or {}).get("reletter_region_id") or dst.id),
                    target_unit_id=str(dst.id), source_unit_id=str(src.id),
                    success=bool(result.success), reason=str(getattr(result, "reason", "") or ""),
                    text=str(src.text), orientation=str(lcfg.orientation),
                    font_size=int(getattr(result, "font_size", 0) or 0),
                    bbox=list(getattr(result, "bbox", ())) if getattr(result, "bbox", None) else [],
                )
            if result.success and result.text_mask is not None:
                rendered = composite_text(rendered, result, config.lettering)
                lettering_masks.append(result.text_mask)
            elif mask_transfer is not None:
                # Never turn a valid Chinese transfer into an empty balloon just
                # because transcript re-layout could not fit. Restore the exact
                # pre-reletter candidate under the clear mask and keep QA/review
                # visible. This preserves a usable Chinese page for manual edit.
                restore_mask = mask_result.per_unit.get(dst.id)
                if restore_mask is not None and cv2.countNonZero(restore_mask) > 0:
                    use = restore_mask > 0
                    rendered[use] = mask_transfer.image[use]

        if mode == "reletter":
            qa = run_page_qa(
                pair, registration, source_units, target_units, matches, lettering,
                mask_result, inpaint_result.image, config.qa,
            )
        else:
            assert mask_transfer is not None
            mask_qa = run_mask_replace_qa(
                pair, registration, source_units, source_bubbles, mask_transfer.records,
                config.qa, config.mask_replace,
            )
            # A rejected mask patch is expected to fall back to OCR re-lettering in
            # hybrid mode. Keep it visible, but do not make that fallback itself a
            # publication-blocking error.
            for item in mask_qa:
                if item.code in {"mask_replace_rejected", "source_bubble_unmatched"}:
                    item.severity = "warning"
            fsids = {m.source_unit_id for m in fallback_matches}
            ftids = {m.target_unit_id for m in fallback_matches}
            fallback_qa = run_page_qa(
                pair, registration,
                [u for u in source_units if u.id in fsids],
                [u for u in target_units if u.id in ftids],
                fallback_matches, lettering, mask_result, inpaint_result.image, config.qa,
            )
            merged = []
            seen = set()
            for item in mask_qa + fallback_qa:
                key = (item.code, item.unit_id, item.message)
                if key not in seen:
                    seen.add(key); merged.append(item)
            qa = merged

    # v2.0.93: QA must validate semantic *coverage*, not only the subset of
    # regions that earlier matchers happened to accept.  This makes a strong
    # Koharu hit that was silently dropped by Paired Diff visible as incomplete.
    qa_layout_evidence = semantic_layout_evidence or target_layout_authority
    if (
        mask_transfer is not None
        and qa_layout_evidence is not None
        and bool(getattr(qa_layout_evidence, "available", False))
        and mode in {"auto", "direct_patch", "mask_replace", "hybrid"}
    ):
        semantic_qa_cfg = config.direct_patch if direct_container_fast else config.mask_replace
        _append_koharu_semantic_coverage_qa(
            qa, evidence=qa_layout_evidence, mask_transfer=mask_transfer,
            shape=target.shape[:2], cfg=semantic_qa_cfg, stats=cache_stats,
        )

    append_photo_pair_evidence_qa(
        qa, paired_diff=paired_diff, config=config, mode=mode, mode_contract=mode_contract,
        registration=registration, source_blocks=source_blocks, target_blocks=target_blocks,
        mask_transfer=mask_transfer,
    )

    return TransferExecutionState(
        mask_transfer=mask_transfer,
        unseeded_white_pair_count=unseeded_white_pair_count,
        completion_display_source=completion_display_source,
        completion_display_target=completion_display_target,
        transfer_rgba=transfer_rgba,
        fallback_matches=fallback_matches,
        rendered=rendered,
        inpaint_result=inpaint_result,
        mask_result=mask_result,
        lettering=lettering,
        lettering_masks=lettering_masks,
        constrained_layout_units=constrained_layout_units,
        target_layout_hint_units=target_layout_hint_units,
        qa=qa,
    )


__all__ = ["TransferExecutionState", "run_transfer_execution_stage"]
