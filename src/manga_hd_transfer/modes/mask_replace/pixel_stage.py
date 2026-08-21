from __future__ import annotations

"""Pixel-transfer stage for Direct / Precise Mask / Hybrid pixel candidates.

This module owns pixel-transfer sequencing and fallback candidate selection only.
It never performs OCR cleanup, inpainting, lettering, or final QA. Stable pixel
renderers remain in their existing mode-owned modules.
"""

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from ...bubbles import pair_unseeded_white_containers
from ...detector_policy import auxiliary_detectors, detector_strategy, STRATEGY_ALWAYS
from ...geometry import transform_to_homography
from ...layout_evidence import collect_ysg_obb_layout_evidence_cached, merge_positive_layout_evidence
from ...models import BubbleInstance
from ...pipeline_transfer_authority import _filter_completion_pairs_by_koharu_authority
from ...source_candidate_service import _cross_rendition_monochrome_source
from ...transfer_completion import (
    remaining_paired_bubbles as _remaining_paired_bubbles,
    mask_transfer_completion_needed as _mask_transfer_completion_needed,
    completion_existing_target_bubbles as _completion_existing_target_bubbles,
    completion_review_regions as _completion_review_regions,
    filter_uncovered_white_completion_pairs as _filter_uncovered_white_completion_pairs,
    completion_filter_pairs_to_review_regions as _completion_filter_pairs_to_review_regions,
)


def _active_mask_config(config: Any, mode: str):
    return config.mask_replace


def _resolve_hybrid_fallback_policy(mode: str):
    return frozenset(), None


def _resolve_active_pixel_mode_modules(*, mode: str, direct_container_fast: bool):
    from . import transfer_ops
    from .raster_policy import global_registered_raster
    return "mask_replace", transfer_ops, global_registered_raster


@dataclass
class PixelTransferState:
    mask_transfer: Any
    unseeded_white_pair_count: int
    completion_display_source: list[BubbleInstance]
    completion_display_target: list[BubbleInstance]
    transfer_rgba: np.ndarray
    fallback_matches: list[Any]
    semantic_layout_evidence: Any | None


def _semantic_completion_exclude_mask(base_mask: np.ndarray, direct_container_plan: Any | None) -> np.ndarray:
    """Exclude complete Direct-owned TARGET regions from semantic completion.

    ``base_mask`` contains pixels that actually changed.  That is insufficient for
    Direct because a correctly cleared white bubble can leave most paper pixels
    unchanged.  Semantic completion must therefore respect region ownership from
    successfully applied Direct target bubbles, otherwise the same bubble can be
    painted twice with a second alignment.
    """
    out = np.array(base_mask, copy=True)
    if direct_container_plan is None:
        return out
    for owned in list(getattr(direct_container_plan, "target_bubbles", []) or []):
        owned_mask = getattr(owned, "safe_mask", None)
        if not isinstance(owned_mask, np.ndarray) or owned_mask.shape != out.shape:
            owned_mask = getattr(owned, "mask", None)
        if isinstance(owned_mask, np.ndarray) and owned_mask.shape == out.shape:
            out = np.maximum(out, (owned_mask > 0).astype(np.uint8) * 255)
    return out


def _transfer_owned_region_mask(
    base_mask: np.ndarray,
    transfer_result: Any | None,
    bubble_groups: list[list[BubbleInstance] | tuple[BubbleInstance, ...]],
) -> np.ndarray:
    """Return full TARGET-region ownership for verified pixel results.

    A renderer owns the *whole accepted semantic/mask region*, not only pixels
    whose RGB changed. White paper often remains byte-identical, so changed-pixel
    masks alone allow later semantic/OCR completion stages to re-enter the same
    bubble and repaint Chinese with a different registration. That was the root
    cause of the v2.3.31 Mask double-image/blur regression.

    Only applied, content-complete records claim full ownership. Incomplete rows
    remain available to Hybrid fallback or Precise-Mask review.
    """
    out = np.array(base_mask, copy=True)
    if transfer_result is None:
        return out
    by_id: dict[str, BubbleInstance] = {}
    for group in bubble_groups:
        for bubble in list(group or []):
            by_id[str(getattr(bubble, "id", ""))] = bubble
    for rec in list(getattr(transfer_result, "records", []) or []):
        if not bool(getattr(rec, "applied", False)):
            continue
        check = str(getattr(rec, "content_check", "") or "")
        if check.startswith("checked") and not bool(getattr(rec, "content_complete", False)):
            continue
        bubble = by_id.get(str(getattr(rec, "target_bubble_id", "")))
        if bubble is None:
            continue
        owned = getattr(bubble, "safe_mask", None)
        if not isinstance(owned, np.ndarray) or owned.shape != out.shape:
            owned = getattr(bubble, "mask", None)
        if isinstance(owned, np.ndarray) and owned.shape == out.shape:
            out = np.maximum(out, (owned > 0).astype(np.uint8) * 255)
    return out


def _merge_mask_transfer(base, extra):
    """Merge a sequential transfer result whose input image was ``base.image``."""
    if extra is None:
        return base
    if base is None:
        return extra
    base.image = extra.image
    use = extra.layer_rgba[..., 3] > 0
    base.layer_rgba[use, :3] = extra.layer_rgba[use, :3]
    base.layer_rgba[..., 3] = np.maximum(base.layer_rgba[..., 3], extra.layer_rgba[..., 3])
    base.composite_mask = np.maximum(base.composite_mask, extra.composite_mask)
    if extra.clear_mask is not None:
        if base.clear_mask is None:
            base.clear_mask = np.zeros_like(base.composite_mask)
        base.clear_mask = np.maximum(base.clear_mask, extra.clear_mask)
    base.matches.extend(extra.matches)
    base.records.extend(extra.records)
    return base


def run_pixel_transfer_stage(
    *, config, pair_check, registration, mode, mode_contract, source, target,
    source_blocks, target_blocks, source_units, target_units, matches, accepted,
    paired_diff, direct_container_fast, direct_container_plan, mask_source_bubbles,
    mask_target_bubbles, target_bubbles, source_layout_authority, target_layout_authority,
    stage_cache=None, cache_stats: dict | None = None, target_path: str | None = None,
) -> PixelTransferState:
    if str(mode or "").strip().lower() not in {"mask_replace", "auto"}:
        raise RuntimeError("mask_replace pixel stage cannot execute mode=" + str(mode))
    # Private mode capsule: this stage file is never shared with another active mode.
    # Hybrid must never borrow the user-facing Precise Mask namespace.
    mask_cfg = _active_mask_config(config, mode)
    active_pixel_owner, transfer_ops, global_registered_raster = _resolve_active_pixel_mode_modules(
        mode=mode, direct_container_fast=direct_container_fast
    )
    hybrid_integrity_block_reasons, hybrid_mask_result_complete = _resolve_hybrid_fallback_policy(mode)
    finalize_transfer_records = transfer_ops.finalize_transfer_records
    transfer_bubble_patches = transfer_ops.transfer_bubble_patches
    transfer_ocr_guided_text_units = transfer_ops.transfer_ocr_guided_text_units
    transfer_paired_diff_regions = transfer_ops.transfer_paired_diff_regions
    transfer_photo_color_sfx = transfer_ops.transfer_photo_color_sfx
    transfer_koharu_semantic_regions = transfer_ops.transfer_koharu_semantic_regions
    transfer_rigid_container_rasters = transfer_ops.transfer_rigid_container_rasters
    if cache_stats is not None:
        cache_stats["active_pixel_owner"] = active_pixel_owner

    mask_transfer = None
    unseeded_white_pair_count = 0
    completion_display_source: list[BubbleInstance] = []
    completion_display_target: list[BubbleInstance] = []
    transfer_rgba = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    fallback_matches = accepted
    semantic_layout_evidence = target_layout_authority
    if mode_contract.direct or mode_contract.mask_replace:
        if direct_container_fast and direct_container_plan is not None:
            # Direct remains the primary renderer, but a publication-safe
            # OCR-free completion pass is now allowed for *isolated ordinary
            # white balloons* that Direct did not discover.  This fixes the
            # silent-missing-bubble failure where all Direct records were SAFE
            # yet one or two plain speech balloons were absent from the plan.
            mask_transfer = direct_container_plan.result
            if (bool(getattr(config.direct_patch, "rigid_container_unseeded_completion_enabled", True))
                    and bool(pair_check.same_page)
                    and registration.confidence >= float(getattr(config.direct_patch, "rigid_container_unseeded_min_registration_confidence", 0.72))):
                extra_src, extra_dst = pair_unseeded_white_containers(
                    source, target, registration, config.direct_patch, config.bubbles,
                    existing_target_bubbles=direct_container_plan.target_bubbles,
                )
                extra_src, extra_dst, authority_audit = _filter_completion_pairs_by_koharu_authority(
                    extra_src, extra_dst, source_evidence=source_layout_authority, target_evidence=target_layout_authority,
                    source_shape=source.shape[:2], target_shape=target.shape[:2], cfg=config.direct_patch,
                )
                if cache_stats is not None:
                    cache_stats["direct_completion_authority_rejected"] = str(sum(1 for row in authority_audit if not row.get("passed")))
                existing_boxes = [tuple(map(int, r.target_bbox)) for r in list(mask_transfer.records or [])
                                  if bool(getattr(r, "applied", False)) and getattr(r, "target_bbox", None)]
                extra_src, extra_dst = _filter_uncovered_white_completion_pairs(
                    extra_src, extra_dst, existing_boxes, config.direct_patch,
                )
                unseeded_white_pair_count = len(extra_src)
                if extra_src and extra_dst:
                    recovered = transfer_rigid_container_rasters(
                        source, target, mask_transfer.image, extra_src, extra_dst, config.direct_patch,
                    )
                    if recovered.records:
                        applied_ids = {str(r.target_bubble_id) for r in recovered.records if bool(getattr(r, "applied", False))}
                        completion_display_target.extend([b for b in extra_dst if str(b.id) in applied_ids])
                        completion_display_source.extend([s for s, t in zip(extra_src, extra_dst) if str(t.id) in applied_ids])
                        mask_transfer = _merge_mask_transfer(mask_transfer, recovered)
        elif (
            paired_diff is not None
            and paired_diff.aligned_source is not None
            and mask_cfg.paired_diff_target_driven_transfer
            and (
                paired_diff.method == "structural_v08"
                or (paired_diff.method == "photo_pair"
                    and mask_cfg.photo_pair_target_driven_enabled
                    and _cross_rendition_monochrome_source(source, target))
            )
        ):
            # v0.8.23: same-layout white containers are rendered from the
            # ORIGINAL source page with one uniform local scale.  The affine
            # page registration remains useful for pairing/detection, but its
            # anisotropic X/Y correction is never inherited by final CJK
            # raster glyphs.  Anything that fails this strict white-container
            # gate continues through the established component/saturated path.
            rigid = transfer_rigid_container_rasters(
                source, target, target, mask_source_bubbles, mask_target_bubbles,
                mask_cfg,
            )
            handled = {r.target_bubble_id for r in rigid.records if r.applied}
            rem_s, rem_t = _remaining_paired_bubbles(mask_source_bubbles, mask_target_bubbles, handled)
            mask_transfer = rigid
            paired_render_source = paired_diff.aligned_source
            paired_render_diag = {
                "policy": "paired_detector_aligned_source",
                "dense_flow_geometry_only": False,
                "glyph_dense_warp": True,
            }
            if mode in {"mask_replace", "hybrid"} and bool(getattr(mask_cfg, "paired_diff_render_use_global_raster", True)):
                paired_render_source, paired_render_diag = global_registered_raster(
                    source, target.shape[:2], registration
                )
                if cache_stats is not None:
                    cache_stats["mask_render_source"] = "global_registration_raster_only"
            if rem_s and rem_t:
                legacy = transfer_paired_diff_regions(
                    paired_render_source, mask_transfer.image, rem_s, rem_t,
                    mask_cfg,
                    render_source_label="paired-global-raster" if paired_render_diag.get("glyph_dense_warp") is False else "paired-dense-align",
                    render_source_diagnostics=paired_render_diag,
                )
                mask_transfer = _merge_mask_transfer(mask_transfer, legacy)
        else:
            mask_transfer = transfer_bubble_patches(
                source,
                target,
                mask_source_bubbles,
                mask_target_bubbles,
                registration,
                mask_cfg,
            )

        # v0.8.5: photographed pages can carry a tightly filtered v0.8
        # structural supplement for open burst bubbles/free text that the
        # conservative closed-container route intentionally misses. Apply it
        # only after the main photo transfer, using target-driven masks and the
        # locally aligned source, then merge the editable layers/QA records.
        supplement = getattr(paired_diff, "supplemental", None) if paired_diff is not None else None
        if (not direct_container_fast and supplement is not None and supplement.aligned_source is not None
                and supplement.source_bubbles and supplement.target_bubbles):
            # Structural local/dense flow is excellent for *detecting* changed
            # islands but it can bend CJK strokes on B/W -> colour editions.
            # Keep that geometry for the masks, while sourcing final Chinese
            # pixels from the primary/global registration whenever possible.
            supplement_transfer_source = supplement.aligned_source
            supplement_render_diag = {
                "policy": "structural_detector_aligned_source",
                "dense_flow_geometry_only": False,
                "glyph_dense_warp": True,
            }
            if mode in {"mask_replace", "hybrid"} and bool(getattr(mask_cfg, "paired_diff_render_use_global_raster", True)):
                supplement_transfer_source, supplement_render_diag = global_registered_raster(
                    source, target.shape[:2], registration
                )
                if cache_stats is not None:
                    cache_stats["mask_supplement_render_source"] = "global_registration_raster_only"
            elif (paired_diff is not None and paired_diff.aligned_source is not None
                    and paired_diff.aligned_source.shape[:2] == target.shape[:2]):
                supplement_transfer_source = paired_diff.aligned_source
            # The structural detector is also allowed to discover ordinary
            # white bubbles that the closed-container pass missed.  Try the
            # same locked whole-raster route first; only coloured/open regions
            # fall through to component/saturated reconstruction.
            rigid_extra = transfer_rigid_container_rasters(
                source, target, mask_transfer.image,
                supplement.source_bubbles, supplement.target_bubbles,
                mask_cfg,
            )
            rigid_handled = {r.target_bubble_id for r in rigid_extra.records if r.applied}
            if rigid_extra.records:
                mask_transfer = _merge_mask_transfer(mask_transfer, rigid_extra)
            rem_s, rem_t = _remaining_paired_bubbles(
                supplement.source_bubbles, supplement.target_bubbles, rigid_handled
            )
            if rem_s and rem_t:
                extra = transfer_paired_diff_regions(
                    supplement_transfer_source, mask_transfer.image,
                    rem_s, rem_t, mask_cfg,
                    render_source_label="paired-global-raster" if supplement_render_diag.get("glyph_dense_warp") is False else "paired-dense-align",
                    render_source_diagnostics=supplement_render_diag,
                )
                if extra.records:
                    mask_transfer = _merge_mask_transfer(mask_transfer, extra)
        # v0.8.25 completion pass: OCR-free white-container pairing. The
        # source and target editions often have identical dialogue balloons
        # even when paired-diff misses one entirely (for example a lower-right
        # speech box whose text changed very little). Detect full enclosed
        # white containers on both pages, pair them through page registration,
        # then send *only* those pairs through the rigid uniform-raster gate.
        # Failed eligibility is ignored rather than falling into component
        # transfer, so architecture/panel false positives cannot be published.
        completion_needed = _mask_transfer_completion_needed(mask_transfer)
        if (not direct_container_fast
                and bool(getattr(mask_cfg, "rigid_container_unseeded_completion_enabled", True))
                and bool(pair_check.same_page)
                and registration.confidence >= float(getattr(mask_cfg, "rigid_container_unseeded_min_registration_confidence", 0.72))):
            existing_completion_targets = _completion_existing_target_bubbles(
                mask_transfer,
                mask_target_bubbles,
                supplement.target_bubbles if supplement is not None else [],
            )
            extra_src, extra_dst = pair_unseeded_white_containers(
                source, target, registration, mask_cfg, config.bubbles,
                # Only already-applied target containers block the OCR-free
                # completion retry. Rejected proposals remain eligible, but
                # successfully transferred bubbles must not be rediscovered
                # and written a second time.
                existing_target_bubbles=existing_completion_targets,
            )
            extra_src, extra_dst, authority_audit = _filter_completion_pairs_by_koharu_authority(
                extra_src, extra_dst, source_evidence=source_layout_authority, target_evidence=target_layout_authority,
                source_shape=source.shape[:2], target_shape=target.shape[:2], cfg=mask_cfg,
            )
            if cache_stats is not None:
                cache_stats["mask_completion_authority_rejected"] = str(sum(1 for row in authority_audit if not row.get("passed")))
            review_boxes = _completion_review_regions(mask_transfer)
            extra_src, extra_dst = _completion_filter_pairs_to_review_regions(extra_src, extra_dst, review_boxes, mask_cfg)
            existing_boxes = [tuple(map(int, r.target_bbox)) for r in list(mask_transfer.records or [])
                              if bool(getattr(r, "applied", False)) and getattr(r, "target_bbox", None)]
            extra_src, extra_dst = _filter_uncovered_white_completion_pairs(extra_src, extra_dst, existing_boxes, mask_cfg)
            unseeded_white_pair_count = len(extra_src)
            if extra_src and extra_dst:
                recovered = transfer_rigid_container_rasters(
                    source, target, mask_transfer.image, extra_src, extra_dst, mask_cfg,
                )
                if recovered.records:
                    applied_ids = {str(r.target_bubble_id) for r in recovered.records if bool(getattr(r, "applied", False))}
                    completion_display_target.extend([b for b in extra_dst if str(b.id) in applied_ids])
                    completion_display_source.extend([s for s, t in zip(extra_src, extra_dst) if str(t.id) in applied_ids])
                    mask_transfer = _merge_mask_transfer(mask_transfer, recovered)

        # v2.0.93 semantic completion: Koharu can correctly classify a coloured
        # burst/open-text container even when Paired Diff intentionally refuses
        # to treat it as a rigid white balloon.  Promote those TARGET semantic
        # regions to first-class completion candidates and transfer only the
        # registered SOURCE glyph ink.  This preserves purple/coloured artwork
        # while removing the Japanese glyph components inside the missed region.
        semantic_layout_evidence = target_layout_authority
        selected_aux = set(auxiliary_detectors(config.bubbles))
        ysg_selected = "ysg_obb" in selected_aux and detector_strategy(config.bubbles) != "primary_only"
        ysg_should_run = bool(
            ysg_selected and (
                detector_strategy(config.bubbles) == STRATEGY_ALWAYS
                or semantic_layout_evidence is None
                or not bool(getattr(semantic_layout_evidence, "available", False))
            )
        )
        if ysg_should_run:
            ysg_target = collect_ysg_obb_layout_evidence_cached(
                target, config.bubbles, role="semantic_target", image_path=target_path,
                cache=stage_cache, cache_enabled=bool(getattr(config.cache, "bubbles", True)),
                stats=cache_stats, allow_missing=True,
            )
            semantic_layout_evidence = merge_positive_layout_evidence(
                semantic_layout_evidence, ysg_target, target.shape[:2], cfg=mask_cfg,
            )
            if cache_stats is not None:
                cache_stats["semantic_aux_ysg_obb"] = "used" if bool(getattr(ysg_target,"available",False)) else "unavailable"

        # Semantic completion is renderer-agnostic and OCR-free.  Direct mode
        # used to skip this pass entirely, which meant a Koharu-detected coloured
        # / open-text region could remain Japanese even though the same evidence
        # was recovered correctly in Mask/Hybrid.  Use the active renderer's
        # safety config, but keep the exact same registered-ink-only algorithm.
        semantic_cfg = config.direct_patch if direct_container_fast else mask_cfg
        if (
            bool(getattr(semantic_cfg, "koharu_semantic_recovery_enabled", True))
            and semantic_layout_evidence is not None
            and bool(getattr(semantic_layout_evidence, "available", False))
            and bool(pair_check.same_page)
            and registration.confidence >= float(getattr(semantic_cfg, "ocr_guided_min_registration_confidence", 0.62))
        ):
            aligned_for_semantic = None
            if mode in {"mask_replace", "hybrid"} and bool(getattr(mask_cfg, "paired_diff_render_use_global_raster", True)):
                aligned_for_semantic, semantic_render_diag = global_registered_raster(
                    source, target.shape[:2], registration
                )
                if cache_stats is not None:
                    cache_stats["semantic_render_source"] = "global_registration_raster_only"
            elif (
                paired_diff is not None
                and paired_diff.aligned_source is not None
                and paired_diff.aligned_source.shape[:2] == target.shape[:2]
            ):
                aligned_for_semantic = paired_diff.aligned_source
            if aligned_for_semantic is None:
                H = transform_to_homography(registration.matrix)
                aligned_for_semantic = cv2.warpPerspective(
                    source, H, (target.shape[1], target.shape[0]),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
            semantic_exclude_mask = mask_transfer.composite_mask
            # Completion is region-ownership based, not changed-pixel based.
            # White paper is often unchanged, so changed-pixel masks let a later
            # renderer re-enter the same accepted bubble and paint it with another
            # registration. v2.3.32 applies this ownership rule to Mask/Hybrid as
            # well as Direct.
            if direct_container_fast and direct_container_plan is not None:
                semantic_exclude_mask = _semantic_completion_exclude_mask(
                    mask_transfer.composite_mask, direct_container_plan
                )
                if cache_stats is not None:
                    cache_stats["koharu_semantic_direct_owned_exclude_pixels"] = str(int(cv2.countNonZero(semantic_exclude_mask)))
            else:
                ownership_groups = [mask_target_bubbles, target_bubbles, completion_display_target]
                if supplement is not None:
                    ownership_groups.append(list(getattr(supplement, "target_bubbles", []) or []))
                semantic_exclude_mask = _transfer_owned_region_mask(
                    mask_transfer.composite_mask, mask_transfer, ownership_groups
                )
                if cache_stats is not None:
                    cache_stats["koharu_semantic_mask_owned_exclude_pixels"] = str(int(cv2.countNonZero(semantic_exclude_mask)))
            semantic_extra = transfer_koharu_semantic_regions(
                aligned_for_semantic,
                mask_transfer.image,
                semantic_layout_evidence,
                semantic_cfg,
                exclude_mask=semantic_exclude_mask,
                include_sfx=bool(getattr(semantic_cfg, "koharu_semantic_include_sfx", False)),
            )
            if cache_stats is not None:
                cache_stats["koharu_semantic_candidates"] = str(len(semantic_extra.records))
                cache_stats["koharu_semantic_applied"] = str(sum(1 for r in semantic_extra.records if bool(getattr(r, "applied", False))))
            if semantic_extra.records:
                applied_semantic_indexes = set()
                for rec in semantic_extra.records:
                    if not bool(getattr(rec, "applied", False)):
                        continue
                    meta = dict(getattr(rec, "meta", {}) or {})
                    if str(meta.get("layout_label") or "") == "bubble":
                        token = str(getattr(rec, "target_bubble_id", ""))
                        try:
                            applied_semantic_indexes.add(int(token.rsplit("-", 1)[-1]))
                        except (TypeError, ValueError):
                            pass
                if applied_semantic_indexes:
                    for bubble in semantic_layout_evidence.bubble_instances(
                        backend_name=str(getattr(semantic_layout_evidence, "backend", "semantic_layout")), target_only=True
                    ):
                        item_index = (getattr(bubble, "meta", {}) or {}).get("item_index")
                        if item_index in applied_semantic_indexes:
                            completion_display_target.append(bubble)
                mask_transfer = _merge_mask_transfer(mask_transfer, semantic_extra)

        # v0.8.21 completeness fallback: OCR may confirm a source/target text
        # correspondence that paired-diff/container geometry missed.  In
        # Precise Mask mode OCR is geometry/evidence only: clear concrete
        # Japanese glyph components and copy registered source raster ink; it
        # never re-typesets or substitutes OCR text.  Low-confidence matches
        # become reversible review candidates instead of disappearing.
        if (not direct_container_fast
                and bool(getattr(mask_cfg, "ocr_guided_component_transfer_enabled", True))
                and registration.confidence >= float(getattr(mask_cfg, "ocr_guided_min_registration_confidence", 0.62))
                and source_units and target_units and matches):
            aligned_for_ocr = None
            if mode in {"mask_replace", "hybrid"} and bool(getattr(mask_cfg, "paired_diff_render_use_global_raster", True)):
                aligned_for_ocr, ocr_render_diag = global_registered_raster(
                    source, target.shape[:2], registration
                )
                if cache_stats is not None:
                    cache_stats["ocr_guided_render_source"] = "global_registration_raster_only"
            elif (paired_diff is not None and paired_diff.aligned_source is not None
                    and paired_diff.aligned_source.shape[:2] == target.shape[:2]):
                aligned_for_ocr = paired_diff.aligned_source
            if aligned_for_ocr is None:
                H = transform_to_homography(registration.matrix)
                aligned_for_ocr = cv2.warpPerspective(
                    source, H, (target.shape[1], target.shape[0]),
                    flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(255, 255, 255),
                )
            ocr_ownership_groups = [mask_target_bubbles, target_bubbles, completion_display_target]
            if supplement is not None:
                ocr_ownership_groups.append(list(getattr(supplement, "target_bubbles", []) or []))
            ocr_exclude_mask = _transfer_owned_region_mask(
                mask_transfer.composite_mask, mask_transfer, ocr_ownership_groups
            )
            ocr_extra = transfer_ocr_guided_text_units(
                aligned_for_ocr, mask_transfer.image, source_units, target_units,
                matches, registration, mask_cfg,
                exclude_mask=ocr_exclude_mask,
            )
            if ocr_extra.records:
                mask_transfer.image = ocr_extra.image
                use = ocr_extra.layer_rgba[..., 3] > 0
                mask_transfer.layer_rgba[use, :3] = ocr_extra.layer_rgba[use, :3]
                mask_transfer.layer_rgba[..., 3] = np.maximum(
                    mask_transfer.layer_rgba[..., 3], ocr_extra.layer_rgba[..., 3]
                )
                mask_transfer.composite_mask = np.maximum(
                    mask_transfer.composite_mask, ocr_extra.composite_mask
                )
                if ocr_extra.clear_mask is not None:
                    if mask_transfer.clear_mask is None:
                        mask_transfer.clear_mask = np.zeros_like(mask_transfer.composite_mask)
                    mask_transfer.clear_mask = np.maximum(mask_transfer.clear_mask, ocr_extra.clear_mask)
                mask_transfer.matches.extend(ocr_extra.matches)
                mask_transfer.records.extend(ocr_extra.records)

        # Large vivid-red translated SFX are not speech bubbles and can be
        # missed by both container and OCR geometry. Rebuild only red groups
        # whose registered shapes differ strongly between source and target.
        color_source = (getattr(getattr(paired_diff, "supplemental", None), "aligned_source", None)
                        if paired_diff is not None else None)
        if color_source is None and paired_diff is not None:
            color_source = paired_diff.aligned_source
        if (not direct_container_fast and color_source is not None and paired_diff is not None and paired_diff.method == "photo_pair"
                and mask_cfg.photo_pair_color_sfx_enabled):
            koharu_color_authority = (
                target_layout_authority.combined_mask(("text", "sfx"), dilate_px=4)
                if target_layout_authority.available else None
            )
            color_extra = transfer_photo_color_sfx(
                color_source, mask_transfer.image, mask_cfg,
                koharu_text_sfx_authority_mask=koharu_color_authority,
            )
            if color_extra.records:
                mask_transfer.image = color_extra.image
                use = color_extra.layer_rgba[..., 3] > 0
                mask_transfer.layer_rgba[use, :3] = color_extra.layer_rgba[use, :3]
                mask_transfer.layer_rgba[..., 3] = np.maximum(mask_transfer.layer_rgba[..., 3], color_extra.layer_rgba[..., 3])
                mask_transfer.composite_mask = np.maximum(mask_transfer.composite_mask, color_extra.composite_mask)
                if color_extra.clear_mask is not None:
                    if mask_transfer.clear_mask is None:
                        mask_transfer.clear_mask = np.zeros_like(mask_transfer.composite_mask)
                    mask_transfer.clear_mask = np.maximum(mask_transfer.clear_mask, color_extra.clear_mask)
                mask_transfer.matches.extend(color_extra.matches)
                mask_transfer.records.extend(color_extra.records)
        transfer_rgba = mask_transfer.layer_rgba
        applied_source_bubbles = {r.source_bubble_id for r in mask_transfer.records if r.applied}
        applied_target_bubbles = {r.target_bubble_id for r in mask_transfer.records if r.applied}
        source_unit_by_id = {u.id: u for u in source_units}
        target_unit_by_id = {u.id: u for u in target_units}
        mask_source_by_id = {b.id: b for b in mask_source_bubbles}
        mask_target_by_id = {b.id: b for b in mask_target_bubbles}

        def _center_in_boxes(unit, boxes):
            if unit is None:
                return False
            cx, cy = unit.centroid
            return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)

        applied_source_boxes = [mask_source_by_id[i].bbox for i in applied_source_bubbles if i in mask_source_by_id]
        applied_target_boxes = [mask_target_by_id[i].bbox for i in applied_target_bubbles if i in mask_target_by_id]

        # Source-integrity rejection is stronger than an OCR/reletter fallback.
        # A camera-edge-clipped photographed bubble may still yield *some* OCR
        # text, but those characters are necessarily only a prefix/suffix of
        # the translation that existed outside the frame. Relettering that
        # partial OCR string into the complete HD target would recreate the
        # exact v0.8.2 failure through a different code path. Keep these
        # regions blocked until a complete source or a manual full translation
        # is supplied.
        integrity_block_reasons = set(hybrid_integrity_block_reasons)
        def _integrity_blocked_record(r):
            return (
                r.reason in integrity_block_reasons
                or (bool(getattr(r, "review_required", False)) and getattr(r, "review_reason", "") in integrity_block_reasons)
            )
        integrity_blocked_src = {
            r.source_bubble_id for r in mask_transfer.records if _integrity_blocked_record(r)
        }
        integrity_blocked_dst = {
            r.target_bubble_id for r in mask_transfer.records if _integrity_blocked_record(r)
        }
        integrity_blocked_source_boxes = [
            mask_source_by_id[i].bbox for i in integrity_blocked_src if i in mask_source_by_id
        ]
        integrity_blocked_target_boxes = [
            mask_target_by_id[i].bbox for i in integrity_blocked_dst if i in mask_target_by_id
        ]

        if mode == "hybrid":
            # Hybrid is mask-first, not mask-only. A geometrically applied raster
            # suppresses OCR/reletter fallback only when its content audit is
            # actually complete. Applied-but-incomplete regions remain eligible
            # for OCR relettering; physically cropped SOURCE regions are filtered
            # by the integrity block below and can never be guessed/recreated.
            assert hybrid_mask_result_complete is not None
            complete_src_ids = {
                r.source_bubble_id for r in mask_transfer.records if hybrid_mask_result_complete(r)
            }
            complete_dst_ids = {
                r.target_bubble_id for r in mask_transfer.records if hybrid_mask_result_complete(r)
            }
            complete_source_boxes = [
                mask_source_by_id[i].bbox for i in complete_src_ids if i in mask_source_by_id
            ]
            complete_target_boxes = [
                mask_target_by_id[i].bbox for i in complete_dst_ids if i in mask_target_by_id
            ]
            fallback_matches = [
                m for m in accepted
                if source_unit_by_id.get(m.source_unit_id) is not None
                and target_unit_by_id.get(m.target_unit_id) is not None
                and not _center_in_boxes(source_unit_by_id[m.source_unit_id], complete_source_boxes)
                and not _center_in_boxes(target_unit_by_id[m.target_unit_id], complete_target_boxes)
            ]
            if cache_stats is not None:
                cache_stats["hybrid_mask_complete_records"] = str(len(complete_dst_ids))
                cache_stats["hybrid_reletter_fallback_candidates"] = str(len(fallback_matches))
        elif (mode == "mask_replace" and paired_diff is not None
                and paired_diff.method == "photo_pair"
                and mask_cfg.photo_pair_fallback_reletter_missing):
            # A photographed pair is deliberately conservative: direct mask
            # transfer handles only geometrically safe containers. Use OCR
            # re-lettering for every accepted text match not already covered,
            # including missed/open/clipped burst bubbles and blurry patches.
            fallback_matches = [
                m for m in accepted
                if source_unit_by_id.get(m.source_unit_id) is not None
                and target_unit_by_id.get(m.target_unit_id) is not None
                and not _center_in_boxes(source_unit_by_id[m.source_unit_id], applied_source_boxes)
                and not _center_in_boxes(target_unit_by_id[m.target_unit_id], applied_target_boxes)
            ]
        elif mode == "mask_replace" and mask_cfg.fallback_reletter_on_blur:
            blur_rejected_src = {
                r.source_bubble_id for r in mask_transfer.records
                if (not r.applied and r.reason in {
                    "source_text_too_blurry_for_pixel_transfer",
                    "source_text_fidelity_rejected",
                })
            }
            blur_rejected_dst = {
                r.target_bubble_id for r in mask_transfer.records
                if (not r.applied and r.reason in {
                    "source_text_too_blurry_for_pixel_transfer",
                    "source_text_fidelity_rejected",
                })
            }
            rejected_source_boxes = [mask_source_by_id[i].bbox for i in blur_rejected_src if i in mask_source_by_id]
            rejected_target_boxes = [mask_target_by_id[i].bbox for i in blur_rejected_dst if i in mask_target_by_id]
            fallback_matches = [
                m for m in accepted
                if source_unit_by_id.get(m.source_unit_id) is not None
                and target_unit_by_id.get(m.target_unit_id) is not None
                and _center_in_boxes(source_unit_by_id[m.source_unit_id], rejected_source_boxes)
                and _center_in_boxes(target_unit_by_id[m.target_unit_id], rejected_target_boxes)
            ]
        else:
            fallback_matches = []

        # Apply the source-integrity block *after* every fallback branch so
        # hybrid mode, photo-pair OCR fallback and future fallback routes all
        # inherit the same publication guarantee.
        if integrity_blocked_source_boxes or integrity_blocked_target_boxes:
            fallback_matches = [
                m for m in fallback_matches
                if source_unit_by_id.get(m.source_unit_id) is not None
                and target_unit_by_id.get(m.target_unit_id) is not None
                and not _center_in_boxes(source_unit_by_id[m.source_unit_id], integrity_blocked_source_boxes)
                and not _center_in_boxes(target_unit_by_id[m.target_unit_id], integrity_blocked_target_boxes)
            ]

        # v0.8.15 strict mode contract. "精准蒙版替换" means exactly that:
        # preserve the translated source glyph pixels/ink, including punctuation
        # and stylized symbols. OCR can be used as evidence for detection/review,
        # but must never rewrite final text in this mode. This intentionally
        # overrides legacy saved configs whose old fallback flags may still be true.
        strict_mask_only = bool(
            (mode == "mask_replace")
            or (mode == "direct_patch")
            or (mode == "auto" and direct_container_fast)
        )
        if strict_mask_only:
            fallback_matches = []

    # Pixel-owned triage/fallback pruning happens before the stage result leaves
    # this boundary. OCR/reletter orchestration only receives candidates that
    # are genuinely uncovered or rejected by the pixel route.
    if mask_transfer is not None:
        triage_cfg = config.direct_patch if direct_container_fast else mask_cfg
        finalize_transfer_records(mask_transfer.records, triage_cfg)
        if (
            mode == "auto"
            and bool(getattr(mask_cfg, "auto_preserve_safe_and_review_pixel_results", True))
            and fallback_matches
        ):
            protected_target_bubbles = {
                str(getattr(r, "target_bubble_id", ""))
                for r in mask_transfer.records
                if bool(getattr(r, "applied", False))
                and str(getattr(r, "triage_state", "")) in {"SAFE", "REVIEW"}
            }
            if protected_target_bubbles:
                target_unit_by_id = {u.id: u for u in target_units}
                kept_fallback = []
                for m in fallback_matches:
                    tu = target_unit_by_id.get(m.target_unit_id)
                    bubble_id = str(getattr(tu, "bubble_id", "") or "") if tu is not None else ""
                    if bubble_id and bubble_id in protected_target_bubbles:
                        continue
                    kept_fallback.append(m)
                fallback_matches = kept_fallback

    return PixelTransferState(
        mask_transfer=mask_transfer,
        unseeded_white_pair_count=unseeded_white_pair_count,
        completion_display_source=completion_display_source,
        completion_display_target=completion_display_target,
        transfer_rgba=transfer_rgba,
        fallback_matches=fallback_matches,
        semantic_layout_evidence=semantic_layout_evidence,
    )


__all__ = ["PixelTransferState", "run_pixel_transfer_stage", "_merge_mask_transfer"]
