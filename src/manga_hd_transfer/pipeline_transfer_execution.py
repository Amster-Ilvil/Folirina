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

from .bubbles import pair_unseeded_white_containers
from .detector_policy import koharu_is_primary, primary_detector, auxiliary_detectors, detector_strategy, STRATEGY_ALWAYS, STRATEGY_CONDITIONAL
from .geometry import rasterize_polygon, transform_to_homography
from .inpainting import InpaintResult, inpaint_image
from .lettering import composite_text, fit_text, polygon_safe_mask, textbox_safe_mask
from .layout_evidence import collect_koharu_layout_evidence_cached, collect_ysg_obb_layout_evidence_cached, merge_positive_layout_evidence, classify_layout_authority
from .masking import MaskBuildResult, build_clear_mask
from .modes.mask_replace.raster_policy import global_registered_raster
from .modes.hybrid.fallback_policy import (
    HYBRID_SOURCE_INTEGRITY_BLOCK_REASONS,
    hybrid_mask_result_complete,
    hybrid_source_integrity_blocked,
)
from .mask_transfer import (
    finalize_transfer_records,
    transfer_bubble_patches, transfer_ocr_guided_text_units,
    transfer_paired_diff_regions, transfer_photo_color_sfx,
    transfer_koharu_semantic_regions,
    transfer_rigid_container_rasters,
)
from .models import BubbleInstance, QAItem
from .qa import run_direct_patch_qa, run_mask_replace_qa, run_page_qa
from .reletter_binding import masked_layout_profile as _masked_layout_profile
from .reletter_layout import (
    apply_target_layout_hints as _apply_target_layout_hints,
    project_source_profile_mask as _project_source_profile_mask,
    preserved_layout_looks_complete as _preserved_layout_looks_complete,
    reletter_orientation as _reletter_orientation,
)
from .source_candidate_service import _cross_rendition_monochrome_source
from .transfer_completion import (
    remaining_paired_bubbles as _remaining_paired_bubbles,
    mask_transfer_completion_needed as _mask_transfer_completion_needed,
    completion_existing_target_bubbles as _completion_existing_target_bubbles,
    completion_review_regions as _completion_review_regions,
    filter_uncovered_white_completion_pairs as _filter_uncovered_white_completion_pairs,
    completion_filter_pairs_to_review_regions as _completion_filter_pairs_to_review_regions,
)
from .transfer_policy import _should_preserve_transferred_layout
from .text_only_transfer import clear_text_components_to_local_paper, clear_broad_neutral_paper_components


@dataclass
class TransferExecutionState:
    mask_transfer: Any
    unseeded_white_pair_count: int
    completion_display_source: list[BubbleInstance]
    completion_display_target: list[BubbleInstance]
    transfer_rgba: np.ndarray
    fallback_matches: list[Any]
    rendered: np.ndarray
    inpaint_result: InpaintResult
    mask_result: MaskBuildResult
    lettering: list[Any]
    lettering_masks: list[np.ndarray]
    constrained_layout_units: int
    target_layout_hint_units: int
    qa: list[QAItem]


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


def _ocr_paper_first_clear(
    base: np.ndarray,
    target: np.ndarray,
    mask_result: MaskBuildResult,
    target_units: list[Any],
    target_bubbles: list[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Clear OCR/reletter text on proven TARGET paper before interpolation.

    Two paper proofs are deliberately combined:

    1. **Broad neutral component proof** handles OCR text-box rectangles.  These
       broad masks were the source of p-005's triangular grey shadows because
       inpainting was asked to invent an already-white rectangle.
    2. **Per-unit local ring proof** retains the older conservative path for
       smaller/irregular glyph masks whose local surroundings prove white paper.

    Only the OCR product routes call this helper; Direct, pure Mask and Reveal
    never enter it.
    """
    cleaned, broad_handled, broad_changed, broad_diag = clear_broad_neutral_paper_components(
        base, target, mask_result.mask,
    )
    accepted = broad_handled.copy()
    unit_by_id = {str(getattr(u, "id", "")): u for u in target_units}
    bubble_by_id = {str(getattr(b, "id", "")): b for b in target_bubbles}
    kept_components = rejected_components = 0

    broad_inv = cv2.bitwise_not(broad_handled)
    for unit_id, unit_clear in (mask_result.per_unit or {}).items():
        if unit_clear is None or cv2.countNonZero(unit_clear) == 0:
            continue
        unit_pending = cv2.bitwise_and(unit_clear, broad_inv)
        if cv2.countNonZero(unit_pending) == 0:
            continue
        unit = unit_by_id.get(str(unit_id))
        if unit is None:
            continue
        region = None
        bubble_id = str(getattr(unit, "bubble_id", "") or "")
        if bubble_id and bubble_id in bubble_by_id:
            region = getattr(bubble_by_id[bubble_id], "safe_mask", None)
        if region is None or region.shape[:2] != target.shape[:2] or cv2.countNonZero(region) == 0:
            region = rasterize_polygon(getattr(unit, "polygon", []) or [], target.shape[:2])
            if cv2.countNonZero(region) > 0:
                # Give the local-paper detector a small TARGET ring around text
                # regions that do not have an explicit parent balloon.
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                region = cv2.dilate(region, k, iterations=1)
        if region is None or cv2.countNonZero(region) == 0:
            continue
        cleaned, local, diag = clear_text_components_to_local_paper(
            cleaned, target, unit_pending, region,
        )
        if local is not None and cv2.countNonZero(local) > 0:
            accepted = cv2.bitwise_or(accepted, local)
        kept_components += int(diag.get("local_paper_components", 0) or 0)
        rejected_components += int(diag.get("local_paper_rejected_components", 0) or 0)

    remaining = mask_result.mask.copy()
    if cv2.countNonZero(accepted) > 0:
        remaining[accepted > 0] = 0
    return cleaned, remaining, {
        **broad_diag,
        "paper_clear_pixels": int(cv2.countNonZero(accepted)),
        "paper_changed_pixels": int(cv2.countNonZero(broad_changed)),
        "remaining_inpaint_pixels": int(cv2.countNonZero(remaining)),
        "paper_components": int(kept_components) + int(broad_diag.get("broad_paper_components", 0) or 0),
        "paper_rejected_components": int(rejected_components) + int(broad_diag.get("broad_paper_rejected_components", 0) or 0),
    }


# Compatibility alias for tests/plugins written against the v2.3.11 symbol.
# Semantics are now the stronger OCR paper-first implementation.
def _reletter_paper_first_clear(
    base: np.ndarray, target: np.ndarray, mask_result: MaskBuildResult,
    target_units: list[Any], target_bubbles: list[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _ocr_paper_first_clear(base, target, mask_result, target_units, target_bubbles)


def _mask_fraction(mask: np.ndarray, other: np.ndarray) -> float:
    if mask is None or other is None or mask.shape != other.shape:
        return 0.0
    area = max(1, int(cv2.countNonZero(mask)))
    return float(np.count_nonzero((mask > 0) & (other > 0)) / area)


def _rect_mask(shape: tuple[int, int], box) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return out
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    h, w = shape
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def _append_koharu_semantic_coverage_qa(
    qa: list[QAItem], *, evidence, mask_transfer, shape: tuple[int, int], cfg, stats: dict | None = None,
) -> None:
    """Audit whether strong Koharu semantic regions actually reached a renderer.

    A detector hit is not useful if a later container matcher silently drops it.
    This audit deliberately reasons from the *target* semantic map and applied
    transfer records.  It therefore catches the p-044 failure where Koharu saw
    the coloured open-text region but Paired Diff only retained four white
    containers.  SFX remains opt-in because stylised effect text can be a valid
    intentional preserve case.
    """
    if evidence is None or not bool(getattr(evidence, "available", False)):
        return
    rows = list(getattr(evidence, "items", []) or [])
    if not rows:
        return
    bubble_min = float(getattr(cfg, "koharu_semantic_bubble_min_confidence", 0.70))
    text_min = float(getattr(cfg, "koharu_semantic_text_min_confidence", 0.75))
    include_sfx = bool(getattr(cfg, "koharu_semantic_include_sfx", False))
    applied = [r for r in list(getattr(mask_transfer, "records", []) or []) if bool(getattr(r, "applied", False))]
    handled = np.zeros(shape, np.uint8)
    for rec in applied:
        handled = np.maximum(handled, _rect_mask(shape, getattr(rec, "target_bbox", None)))
    # Pixel masks are narrower than semantic bubble masks but are valuable for
    # standalone text whose record bbox may intentionally be tight.
    composite = getattr(mask_transfer, "composite_mask", None)
    if isinstance(composite, np.ndarray) and composite.shape == shape:
        handled_pixels = composite
    else:
        handled_pixels = np.zeros(shape, np.uint8)

    bubbles = [r for r in rows if str(getattr(r, "label", "")) == "bubble" and float(getattr(r, "confidence", 0.0)) >= bubble_min]
    texts = [r for r in rows if str(getattr(r, "label", "")) == "text" and float(getattr(r, "confidence", 0.0)) >= text_min]
    if include_sfx:
        texts += [r for r in rows if str(getattr(r, "label", "")) == "sfx" and float(getattr(r, "confidence", 0.0)) >= text_min]

    uncovered_bubbles = 0
    uncovered_text = 0
    bubble_masks: list[np.ndarray] = []
    for row in bubbles:
        mask = getattr(row, "mask", None)
        if not isinstance(mask, np.ndarray) or mask.shape != shape or cv2.countNonZero(mask) <= 0:
            continue
        bubble_masks.append(mask)
        # Only demand transfer coverage for text-bearing semantic containers.
        child = False
        for tr in texts:
            tm = getattr(tr, "mask", None)
            if isinstance(tm, np.ndarray) and tm.shape == shape and _mask_fraction(tm, mask) >= 0.35:
                child = True
                break
        if not child:
            continue
        record_coverage = _mask_fraction(mask, handled)
        pixel_coverage = _mask_fraction(mask, handled_pixels)
        if max(record_coverage, pixel_coverage) >= 0.08:
            continue
        uncovered_bubbles += 1
        conf = float(getattr(row, "confidence", 0.0))
        severity = "error" if conf >= 0.85 else "warning"
        qa.append(QAItem(
            "koharu_semantic_uncovered_bubble", severity,
            "Koharu detected a strong text-bearing bubble/open container, but no automatic transfer record covered it. The page is semantically incomplete.",
            unit_id=f"koharu-bubble-{(getattr(row, 'meta', {}) or {}).get('item_index', '?')}",
            value=conf, threshold=bubble_min,
            meta={
                "bbox": list(getattr(row, "box", ())),
                "record_coverage": round(record_coverage, 4),
                "pixel_coverage": round(pixel_coverage, 4),
                "verification_scope": "koharu_semantic_coverage",
            },
        ))

    for row in texts:
        mask = getattr(row, "mask", None)
        if not isinstance(mask, np.ndarray) or mask.shape != shape or cv2.countNonZero(mask) <= 0:
            continue
        # Text already inside a strong bubble is covered by the bubble audit.
        if any(_mask_fraction(mask, bm) >= 0.55 for bm in bubble_masks):
            continue
        record_coverage = _mask_fraction(mask, handled)
        pixel_coverage = _mask_fraction(mask, handled_pixels)
        if max(record_coverage, pixel_coverage) >= 0.12:
            continue
        uncovered_text += 1
        conf = float(getattr(row, "confidence", 0.0))
        qa.append(QAItem(
            "koharu_semantic_uncovered_text", "warning",
            "Koharu detected high-confidence standalone/open text that was not covered by any automatic transfer. Review or recover this region before export.",
            unit_id=f"koharu-{getattr(row, 'label', 'text')}-{(getattr(row, 'meta', {}) or {}).get('item_index', '?')}",
            value=conf, threshold=text_min,
            meta={
                "bbox": list(getattr(row, "box", ())),
                "record_coverage": round(record_coverage, 4),
                "pixel_coverage": round(pixel_coverage, 4),
                "verification_scope": "koharu_semantic_coverage",
            },
        ))
    if stats is not None:
        stats["koharu_semantic_qa_uncovered_bubbles"] = str(uncovered_bubbles)
        stats["koharu_semantic_qa_uncovered_text"] = str(uncovered_text)


def _filter_completion_pairs_by_koharu_authority(
    source_rows: list[BubbleInstance],
    target_rows: list[BubbleInstance],
    *,
    source_evidence,
    target_evidence,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    cfg,
) -> tuple[list[BubbleInstance], list[BubbleInstance], list[dict]]:
    """Reject completion pairs that Koharu positively classifies as artwork.

    UNKNOWN remains eligible because these are already conservative completion
    candidates; PROTECT never reaches a destructive renderer.
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
            kept_s.append(sb); kept_t.append(tb)
    return kept_s, kept_t, audit


def run_transfer_execution_stage(
    *, config, pair, registration, pair_check, mode, mode_contract, source, target,
    source_blocks, target_blocks, source_units, target_units, matches, accepted,
    paired_diff, direct_container_fast, direct_container_plan, mask_source_bubbles,
    mask_target_bubbles, source_bubbles, target_bubbles,
    target_driven_reletter_regions, target_driven_reletter_diagnostics,
    check_cancel, cancel_cb=None, trace=None, stage_cache=None, cache_stats: dict | None = None,
    source_path: str | None = None, target_path: str | None = None,
) -> TransferExecutionState:
    mask_transfer = None
    unseeded_white_pair_count = 0
    # Successful OCR-free completion boxes are also merged into the persisted
    # bubble/debug view. Previously they could be transferred correctly but
    # remained invisible in debug_structure/GUI, making users think the box
    # was never recognized.
    completion_display_source: list[BubbleInstance] = []
    completion_display_target: list[BubbleInstance] = []
    transfer_rgba = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    fallback_matches = accepted

    # Only a Koharu *primary* owns the panel/artwork hard-veto semantics.
    # Selecting MangaLens/RT-DETR as primary must not secretly invoke Koharu in
    # transfer completion or let an auxiliary outrank the selected main model.
    source_layout_authority = None
    target_layout_authority = None
    if koharu_is_primary(config.bubbles):
        layout_cache_enabled = bool(getattr(config.cache, "bubbles", True)) and bool(getattr(config.bubbles, "koharu_layout_cache_enabled", True))
        source_layout_authority = collect_koharu_layout_evidence_cached(
            source, config.bubbles, role="transfer_authority_source", image_path=source_path,
            cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
        )
        target_layout_authority = collect_koharu_layout_evidence_cached(
            target, config.bubbles, role="transfer_authority_target", image_path=target_path,
            cache=stage_cache, cache_enabled=layout_cache_enabled, stats=cache_stats, allow_missing=True,
        )
        if cache_stats is not None:
            cache_stats["transfer_layout_authority"] = (
                "two_sided" if source_layout_authority.available and target_layout_authority.available
                else "partial_fail_open" if source_layout_authority.available or target_layout_authority.available
                else "unavailable_fail_open"
            )
    elif cache_stats is not None:
        cache_stats["transfer_layout_authority"] = f"skipped_primary:{primary_detector(config.bubbles)}"

    check_cancel(cancel_cb, "before_transfer")
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
            and config.mask_replace.paired_diff_target_driven_transfer
            and (
                paired_diff.method == "structural_v08"
                or (paired_diff.method == "photo_pair"
                    and config.mask_replace.photo_pair_target_driven_enabled
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
                config.mask_replace,
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
            if mode in {"mask_replace", "hybrid"} and bool(getattr(config.mask_replace, "paired_diff_render_use_global_raster", True)):
                paired_render_source, paired_render_diag = global_registered_raster(
                    source, target.shape[:2], registration
                )
                if cache_stats is not None:
                    cache_stats["mask_render_source"] = "global_registration_raster_only"
            if rem_s and rem_t:
                legacy = transfer_paired_diff_regions(
                    paired_render_source, mask_transfer.image, rem_s, rem_t,
                    config.mask_replace,
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
                config.mask_replace,
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
            if mode in {"mask_replace", "hybrid"} and bool(getattr(config.mask_replace, "paired_diff_render_use_global_raster", True)):
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
                config.mask_replace,
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
                    rem_s, rem_t, config.mask_replace,
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
                and bool(getattr(config.mask_replace, "rigid_container_unseeded_completion_enabled", True))
                and bool(pair_check.same_page)
                and registration.confidence >= float(getattr(config.mask_replace, "rigid_container_unseeded_min_registration_confidence", 0.72))):
            existing_completion_targets = _completion_existing_target_bubbles(
                mask_transfer,
                mask_target_bubbles,
                supplement.target_bubbles if supplement is not None else [],
            )
            extra_src, extra_dst = pair_unseeded_white_containers(
                source, target, registration, config.mask_replace, config.bubbles,
                # Only already-applied target containers block the OCR-free
                # completion retry. Rejected proposals remain eligible, but
                # successfully transferred bubbles must not be rediscovered
                # and written a second time.
                existing_target_bubbles=existing_completion_targets,
            )
            extra_src, extra_dst, authority_audit = _filter_completion_pairs_by_koharu_authority(
                extra_src, extra_dst, source_evidence=source_layout_authority, target_evidence=target_layout_authority,
                source_shape=source.shape[:2], target_shape=target.shape[:2], cfg=config.mask_replace,
            )
            if cache_stats is not None:
                cache_stats["mask_completion_authority_rejected"] = str(sum(1 for row in authority_audit if not row.get("passed")))
            review_boxes = _completion_review_regions(mask_transfer)
            extra_src, extra_dst = _completion_filter_pairs_to_review_regions(extra_src, extra_dst, review_boxes, config.mask_replace)
            existing_boxes = [tuple(map(int, r.target_bbox)) for r in list(mask_transfer.records or [])
                              if bool(getattr(r, "applied", False)) and getattr(r, "target_bbox", None)]
            extra_src, extra_dst = _filter_uncovered_white_completion_pairs(extra_src, extra_dst, existing_boxes, config.mask_replace)
            unseeded_white_pair_count = len(extra_src)
            if extra_src and extra_dst:
                recovered = transfer_rigid_container_rasters(
                    source, target, mask_transfer.image, extra_src, extra_dst, config.mask_replace,
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
                semantic_layout_evidence, ysg_target, target.shape[:2], cfg=config.mask_replace,
            )
            if cache_stats is not None:
                cache_stats["semantic_aux_ysg_obb"] = "used" if bool(getattr(ysg_target,"available",False)) else "unavailable"

        # Semantic completion is renderer-agnostic and OCR-free.  Direct mode
        # used to skip this pass entirely, which meant a Koharu-detected coloured
        # / open-text region could remain Japanese even though the same evidence
        # was recovered correctly in Mask/Hybrid.  Use the active renderer's
        # safety config, but keep the exact same registered-ink-only algorithm.
        semantic_cfg = config.direct_patch if direct_container_fast else config.mask_replace
        if (
            bool(getattr(semantic_cfg, "koharu_semantic_recovery_enabled", True))
            and semantic_layout_evidence is not None
            and bool(getattr(semantic_layout_evidence, "available", False))
            and bool(pair_check.same_page)
            and registration.confidence >= float(getattr(semantic_cfg, "ocr_guided_min_registration_confidence", 0.62))
        ):
            aligned_for_semantic = None
            if mode in {"mask_replace", "hybrid"} and bool(getattr(config.mask_replace, "paired_diff_render_use_global_raster", True)):
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
            semantic_extra = transfer_koharu_semantic_regions(
                aligned_for_semantic,
                mask_transfer.image,
                semantic_layout_evidence,
                semantic_cfg,
                exclude_mask=mask_transfer.composite_mask,
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
                and bool(getattr(config.mask_replace, "ocr_guided_component_transfer_enabled", True))
                and registration.confidence >= float(getattr(config.mask_replace, "ocr_guided_min_registration_confidence", 0.62))
                and source_units and target_units and matches):
            aligned_for_ocr = None
            if mode in {"mask_replace", "hybrid"} and bool(getattr(config.mask_replace, "paired_diff_render_use_global_raster", True)):
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
            ocr_extra = transfer_ocr_guided_text_units(
                aligned_for_ocr, mask_transfer.image, source_units, target_units,
                matches, registration, config.mask_replace,
                exclude_mask=mask_transfer.composite_mask,
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
                and config.mask_replace.photo_pair_color_sfx_enabled):
            koharu_color_authority = (
                target_layout_authority.combined_mask(("text", "sfx"), dilate_px=4)
                if target_layout_authority.available else None
            )
            color_extra = transfer_photo_color_sfx(
                color_source, mask_transfer.image, config.mask_replace,
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
        integrity_block_reasons = set(HYBRID_SOURCE_INTEGRITY_BLOCK_REASONS)
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
                and config.mask_replace.photo_pair_fallback_reletter_missing):
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
        elif mode == "mask_replace" and config.mask_replace.fallback_reletter_on_blur:
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

        # Publication-quality photo pages should prefer HD re-lettering when OCR
        # evidence is available. Mask transfer still establishes the region and
        # safely covers OCR-missing areas, but accepted OCR dialogue matches are
        # re-rendered sharply instead of publishing photographed glyph pixels.
        if (not strict_mask_only and mode == "mask_replace" and paired_diff is not None
                and paired_diff.method == "photo_pair"
                and config.mask_replace.photo_pair_prefer_reletter_with_ocr
                and source_blocks and target_blocks and applied_target_boxes):
            min_ocr = float(config.mask_replace.photo_pair_prefer_reletter_min_confidence)
            min_match = float(min(config.matching.review_confidence, max(0.0, min_ocr - 0.08)))
            preferred = []
            transfer_record_by_target = {
                str(r.target_bubble_id): r for r in (mask_transfer.records if mask_transfer is not None else [])
            }
            source_block_by_id = {b.id: b for b in source_blocks}
            target_bubble_by_id = {b.id: b for b in target_bubbles}
            for m in accepted:
                su = source_unit_by_id.get(m.source_unit_id)
                tu = target_unit_by_id.get(m.target_unit_id)
                if su is None or tu is None:
                    continue
                if not str(su.text).strip():
                    continue
                if su.confidence < min_ocr or tu.confidence < min_ocr or m.confidence < min_match:
                    continue
                if _center_in_boxes(su, integrity_blocked_source_boxes) or _center_in_boxes(tu, integrity_blocked_target_boxes):
                    continue
                if not _center_in_boxes(tu, applied_target_boxes):
                    continue
                # v0.8.11: when a clean translated scan already produced a
                # sharp, geometry-preserving glyph transfer, OCR should remain
                # evidence/review only. Re-typesetting a short transcript from
                # scratch is what created oversized words and lost source columns
                # on real Macs. Only low-quality/blurred transfers are promoted
                # to OCR re-lettering automatically.
                rec = transfer_record_by_target.get(str(tu.bubble_id or ""))
                if _should_preserve_transferred_layout(rec, config.mask_replace):
                    profiles = [
                        source_block_by_id[bid].meta.get("source_layout_profile")
                        for bid in su.block_ids if bid in source_block_by_id
                        and source_block_by_id[bid].meta.get("source_layout_profile")
                    ]
                    profile = profiles[0] if profiles else {}
                    safe = None
                    if tu.bubble_id and tu.bubble_id in target_bubble_by_id:
                        safe = target_bubble_by_id[tu.bubble_id].safe_mask
                    if safe is None or cv2.countNonZero(safe) == 0:
                        safe = polygon_safe_mask(tu, target.shape[:2], margin=max(2, config.bubbles.safe_margin_px // 2))
                    transferred_profile = _masked_layout_profile(mask_transfer.image, safe, su.text, _reletter_orientation("auto", su, source_block_by_id)) if mask_transfer is not None else {}
                    if profile and _preserved_layout_looks_complete(profile, transferred_profile):
                        continue
                preferred.append(m)
            if preferred:
                seen = {(m.source_unit_id, m.target_unit_id) for m in fallback_matches}
                for m in preferred:
                    key = (m.source_unit_id, m.target_unit_id)
                    if key not in seen:
                        fallback_matches.append(m)
                        seen.add(key)

    if mask_transfer is not None:
        triage_cfg = config.direct_patch if direct_container_fast else config.mask_replace
        finalize_transfer_records(mask_transfer.records, triage_cfg)
        # v0.8.34.4: Auto must preserve a usable pixel/mask result. OCR may
        # still provide evidence, but SAFE/REVIEW regions are not re-typeset
        # over the source raster merely because a text match exists. Only
        # REJECT or genuinely uncovered regions continue to heavy fallback.
        if (mode == "auto"
                and bool(getattr(config.mask_replace, "auto_preserve_safe_and_review_pixel_results", True))
                and fallback_matches):
            protected_target_bubbles = {
                str(getattr(r, "target_bubble_id", ""))
                for r in mask_transfer.records
                if bool(getattr(r, "applied", False))
                and str(getattr(r, "triage_state", "")) in {"SAFE", "REVIEW"}
            }
            if protected_target_bubbles:
                kept_fallback = []
                for m in fallback_matches:
                    tu = target_unit_by_id.get(m.target_unit_id)
                    bubble_id = str(getattr(tu, "bubble_id", "") or "") if tu is not None else ""
                    if bubble_id and bubble_id in protected_target_bubbles:
                        continue
                    kept_fallback.append(m)
                fallback_matches = kept_fallback

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
    qa_layout_evidence = locals().get("semantic_layout_evidence") or target_layout_authority
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

    if (paired_diff is not None and paired_diff.method == "photo_pair"
            and config.mask_replace.photo_pair_require_ocr_evidence
            and not source_blocks and not target_blocks):
        photo_records = list(mask_transfer.records) if mask_transfer is not None else []
        # Judge the conservative photo route on its own records. Structural
        # supplements may legitimately add/reject open-text candidates and
        # should not turn a successfully transferred photographed pair into
        # a publication-blocking "OCR missing" error.
        photo_only = [
            r for r in photo_records
            if getattr(r, "geometry_mode", "") in {"photo_pair", "rigid_uniform_container"}
            or getattr(r, "sr_backend", "") == "rigid-container-raster"
        ]
        content_verified = bool(
            photo_records
            and all(
                (not bool(getattr(r, "applied", False)))
                or (str(getattr(r, "content_check", "")).startswith("checked")
                    and bool(getattr(r, "content_complete", False)))
                for r in photo_records
            )
        )
        # OCR absence is a review warning in the user's preferred policy:
        # once a raster candidate was detected and published, incomplete
        # content must not turn the page back into Japanese.
        fully_applied_photo = bool(
            photo_only
            and all(r.applied for r in photo_only)
            and registration.confidence >= 0.78
        )
        # v2.0.83 explicit Mask is intentionally a zero-OCR contract.  Do not
        # turn the *absence* of OCR into a blocking error for the very mode that
        # forbids OCR.  Actual raster failures remain independently blocking via
        # mask_replace_rejected / mask_replace_content_incomplete.  This item is
        # therefore only a scope warning for explicit Mask, while Auto/Hybrid
        # keep their historical evidence policy because they are allowed to use
        # OCR as corroboration.
        explicit_visual_mask = bool(mode == "mask_replace" and not mode_contract.may_use_ocr)
        if explicit_visual_mask:
            qa.append(QAItem(
                "photo_pair_visual_only_contract",
                "warning",
                (
                    "Precise Mask is running under the explicit zero-OCR contract. "
                    "Detected photographed-page regions were judged only by visual/raster evidence; "
                    "review for completely undiscovered open/SFX text."
                ),
                meta={
                    "detected_regions": len(photo_records),
                    "content_verified_detected_regions": content_verified,
                    "verification_scope": "visual_detected_regions_only",
                    "registration_confidence": registration.confidence,
                    "ocr_intentionally_disabled": True,
                },
            ))
        else:
            qa.append(QAItem(
                "photo_pair_ocr_evidence_missing",
                "warning" if fully_applied_photo else "error",
                (
                    "All detected photographed-page regions passed the independent raster-content check under strong registration. OCR is unavailable, so this verifies detected regions only; review the page for any entirely undiscovered open/SFX text."
                    if fully_applied_photo else
                    "Photographed-edition extraction is conservative and OCR evidence is unavailable; at least one detected region is not independently content-verified, or page registration is weak."
                ),
                meta={
                    "detected_regions": len(photo_records),
                    "content_verified_detected_regions": content_verified,
                    "verification_scope": "detected_regions_only",
                    "registration_confidence": registration.confidence,
                },
            ))

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
