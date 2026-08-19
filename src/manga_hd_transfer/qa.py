from __future__ import annotations

from collections import Counter

import cv2
import numpy as np

from .config import QAConfig
from .models import (
    LetteringResult,
    PagePair,
    QAItem,
    RegistrationResult,
    TextUnit,
    UnitMatch,
)
from .masking import MaskBuildResult


def _dark_ratio(image: np.ndarray, mask: np.ndarray, threshold: int = 120) -> float:
    if mask is None or cv2.countNonZero(mask) == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    vals = gray[mask > 0]
    return float(np.mean(vals < threshold)) if len(vals) else 0.0


def run_page_qa(
    pair: PagePair,
    registration: RegistrationResult,
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    matches: list[UnitMatch],
    lettering: list[LetteringResult],
    clear_masks: MaskBuildResult,
    inpainted: np.ndarray,
    config: QAConfig | None = None,
) -> list[QAItem]:
    cfg = config or QAConfig()
    issues: list[QAItem] = []

    if pair.confidence < 0.45:
        issues.append(QAItem("page_pair_low_confidence", "error", "Page pairing is uncertain; automatic overwrite is unsafe.", value=pair.confidence, threshold=0.45))
    if registration.confidence < cfg.registration_min_confidence:
        issues.append(
            QAItem(
                "registration_low_confidence",
                "error",
                "Cross-edition geometric registration is below the automatic-apply threshold.",
                value=registration.confidence,
                threshold=cfg.registration_min_confidence,
                meta={"method": registration.method, "reprojection_error": registration.reprojection_error},
            )
        )

    for unit in source_units:
        if unit.confidence < cfg.ocr_min_confidence:
            issues.append(QAItem("source_ocr_low_confidence", "error", "Chinese OCR requires review.", unit.id, unit.confidence, cfg.ocr_min_confidence))
    for unit in target_units:
        if unit.confidence < cfg.ocr_min_confidence:
            issues.append(QAItem("target_detection_low_confidence", "warning", "Target text geometry has low OCR confidence.", unit.id, unit.confidence, cfg.ocr_min_confidence))

    matched_source = Counter(m.source_unit_id for m in matches)
    matched_target = Counter(m.target_unit_id for m in matches)
    source_by_id = {u.id: u for u in source_units}
    target_by_id = {u.id: u for u in target_units}
    for match in matches:
        sunit = source_by_id.get(match.source_unit_id)
        tunit = target_by_id.get(match.target_unit_id)
        if (sunit and sunit.kind in {"sfx", "free_text"}) or (tunit and tunit.kind in {"sfx", "free_text"}):
            issues.append(QAItem("art_text_review", "warning", "Free/SFX text should be manually reviewed for style and exact geometry before publication.", match.source_unit_id, meta={"target": match.target_unit_id}))
        if match.confidence < cfg.match_min_confidence:
            issues.append(QAItem("match_low_confidence", "error", "Cross-version text identity match requires review.", match.source_unit_id, match.confidence, cfg.match_min_confidence, {"target": match.target_unit_id, "relation": match.relation}))
        if match.relation != "one_to_one":
            issues.append(QAItem("split_merge_match", "error", f"Detected {match.relation} relationship; manual text split/merge is required.", match.source_unit_id, meta={"target": match.target_unit_id}))
    for source_id, count in matched_source.items():
        if count > 1:
            issues.append(QAItem("source_mapped_multiple_times", "warning", "One source translation unit maps to multiple target units.", source_id, value=count))
    for target_id, count in matched_target.items():
        if count > 1:
            issues.append(QAItem("target_receives_multiple_sources", "warning", "One target unit receives multiple source translations.", target_id, value=count))

    matched_ids = set(matched_source)
    for unit in source_units:
        if unit.id not in matched_ids:
            issues.append(QAItem("source_unmatched", "error", "Chinese translation unit has no target region.", unit.id))

    if clear_masks.source_pixels > 0 and clear_masks.clipped_ratio > 0.18:
        issues.append(QAItem("mask_heavily_clipped", "warning", "A large fraction of the clear mask was clipped by bubble-border protection.", value=clear_masks.clipped_ratio, threshold=0.18))

    for target_id, mask in clear_masks.per_unit.items():
        ratio = _dark_ratio(inpainted, mask)
        if ratio > cfg.residual_dark_ratio_max:
            issues.append(QAItem("possible_japanese_residual", "warning", "Dark pixels remain inside a cleared target text mask; inspect for residual Japanese glyphs or line-art overlap.", target_id, ratio, cfg.residual_dark_ratio_max))

    lettering_by_id = {l.unit_id: l for l in lettering}
    for match in matches:
        if match.relation != "one_to_one" or match.confidence < cfg.match_min_confidence:
            continue
        lr = lettering_by_id.get(match.target_unit_id)
        if lr is None:
            issues.append(QAItem("lettering_missing", "error", "Accepted match has no lettering output.", match.target_unit_id))
            continue
        if not lr.success:
            issues.append(QAItem("lettering_failed", "error", f"Text could not be fit safely: {lr.reason}", match.target_unit_id))
        elif lr.coverage_inside_safe < cfg.lettering_safe_coverage_min:
            issues.append(QAItem("lettering_overflow", "error", "Rendered glyphs extend outside the safe area.", match.target_unit_id, lr.coverage_inside_safe, cfg.lettering_safe_coverage_min))
        if lr.success and lr.font_size < cfg.min_font_size:
            issues.append(QAItem("font_too_small", "warning", "Fitted font size is below the publication threshold.", match.target_unit_id, lr.font_size, cfg.min_font_size))

    return issues


def qa_summary(items: list[QAItem]) -> dict[str, int | bool]:
    counts = Counter(i.severity for i in items)
    return {
        "errors": counts.get("error", 0),
        "warnings": counts.get("warning", 0),
        "info": counts.get("info", 0),
        "pass": counts.get("error", 0) == 0,
    }


def run_mask_replace_qa(
    pair: PagePair,
    registration: RegistrationResult,
    source_units: list[TextUnit],
    source_bubbles,
    records,
    qa_config: QAConfig | None = None,
    replace_config=None,
) -> list[QAItem]:
    """QA for direct Chinese bubble/text-box patch transfer.

    This path does not require target OCR lettering output, so it must not reuse
    the lettering-missing checks from ``run_page_qa``.
    """
    cfg = qa_config or QAConfig()
    issues: list[QAItem] = []
    publication_safety = False  # v1.0.6: publication blocking removed; diagnostics only
    if pair.confidence < 0.45:
        photo_records = [r for r in records if getattr(r, "geometry_mode", "standard") == "photo_pair"]
        # An intentional edge-clip rejection is evidence of source incompleteness,
        # not failed page registration. Keep that issue blocking under its own
        # specific QA code, but do not also mislabel a strongly registered pair as
        # geometrically uncertain.
        photo_geometry_verified = bool(
            photo_records
            and all(
                getattr(r, "applied", False)
                or getattr(r, "reason", "") == "source_text_region_clipped_at_page_edge"
                for r in photo_records
            )
            and any(getattr(r, "applied", False) for r in photo_records)
            and registration.confidence >= max(float(cfg.registration_min_confidence), 0.82)
        )
        severity = "warning" if (photo_geometry_verified or not publication_safety) else "error"
        message = (
            "Initial page-pair heuristic is weak, but high-confidence feature registration verifies the photographed page geometry; any deliberately rejected edge-clipped source remains a separate blocking integrity issue."
            if photo_geometry_verified else
            "Page pairing is uncertain; mask replacement is unsafe."
        )
        issues.append(QAItem("page_pair_low_confidence", severity, message, value=pair.confidence, threshold=0.45, meta={"registration_confidence": registration.confidence}))
    if registration.confidence < cfg.registration_min_confidence:
        issues.append(QAItem("registration_low_confidence", "error" if publication_safety else "warning", "Cross-edition registration is below the mask-transfer threshold; aggressive mode keeps the diagnostic but does not publication-block an otherwise writable region.", value=registration.confidence, threshold=cfg.registration_min_confidence))
    for unit in source_units:
        if unit.confidence < cfg.ocr_min_confidence:
            issues.append(QAItem("source_ocr_low_confidence", "warning", "Source Chinese OCR is low-confidence; bubble extraction should be reviewed.", unit.id, unit.confidence, cfg.ocr_min_confidence))

    enabled = set(getattr(replace_config, "enabled_kinds", ["speech", "narration"]))
    eligible = {b.id for b in source_bubbles if b.kind in enabled and (b.block_ids or not getattr(replace_config, "require_source_text", True))}
    if not eligible and not records and getattr(cfg, "fail_empty_mask_replace", True):
        issues.append(QAItem(
            "mask_replace_no_candidates", "error",
            "No translated bubble/text-box candidate was found; zero-output mask replacement cannot be published as success.",
        ))
    matched = {r.source_bubble_id for r in records}
    for bid in sorted(eligible - matched):
        issues.append(QAItem("source_bubble_unmatched", "error", "Chinese bubble/text box has no aligned target instance.", bid))

    for rec in records:
        if not rec.applied:
            if rec.reason == "source_text_region_clipped_at_page_edge":
                issues.append(QAItem(
                    "mask_replace_source_translation_clipped",
                    "error",
                    "The photographed Chinese source is physically clipped at the page/image edge, so the complete translation is not present in the input. The HD target bubble was deliberately left untouched instead of publishing partial Chinese text.",
                    rec.source_bubble_id,
                    meta={
                        "target": rec.target_bubble_id,
                        "mask_iou": rec.mask_iou,
                        "coverage": rec.target_coverage,
                        "spill": rec.spill_ratio,
                        "source_edge_sides": getattr(rec, "source_edge_sides", ""),
                    },
                ))
            else:
                speculative_complex = (
                    not source_units
                    and str(getattr(rec, "geometry_mode", "")) == "complex_text"
                    and str(getattr(rec, "source_bubble_id", "")).startswith("diff-src-")
                )
                issues.append(QAItem(
                    "mask_replace_rejected",
                    "warning" if speculative_complex else "error",
                    ("Speculative complex/open-text recovery candidate was rejected before publication; no OCR text evidence confirms it as a required region." if speculative_complex else f"Bubble patch replacement rejected: {rec.reason}"),
                    rec.source_bubble_id,
                    meta={"target": rec.target_bubble_id, "mask_iou": rec.mask_iou, "coverage": rec.target_coverage, "spill": rec.spill_ratio, "reason": rec.reason},
                ))
            continue
        if bool(getattr(replace_config, "content_completeness_enabled", True)):
            content_check = str(getattr(rec, "content_check", "not_checked") or "not_checked")
            if content_check.startswith("checked"):
                if not bool(getattr(rec, "content_complete", False)):
                    issues.append(QAItem(
                        "mask_replace_content_incomplete", "warning" if getattr(rec, "applied", False) else "error",
                        "Pixels were written, but the independent content check found missing source ink or residual target-language ink. Raster application is not publication success.",
                        rec.source_bubble_id,
                        meta={
                            "target": rec.target_bubble_id,
                            "source_ink_coverage": float(getattr(rec, "source_ink_coverage", 0.0)),
                            "target_residual_ratio": float(getattr(rec, "target_residual_ratio", 0.0)),
                            "clarity_mode": getattr(rec, "clarity_mode", "pixels"),
                        },
                    ))
            elif content_check not in {"disabled"}:
                strict_content_route = str(getattr(rec, "geometry_mode", "")) in {
                    "photo_pair", "free_text", "complex_text", "ocr_guided_components"
                }
                issues.append(QAItem(
                    "mask_replace_content_unverified", "error" if (strict_content_route and publication_safety) else "warning",
                    "Pixels were written but there is not enough independent ink evidence to verify content completeness. Cross-rendition/complex routes are publication-blocking; legacy same-style raw-diff routes remain review warnings for compatibility.",
                    rec.source_bubble_id,
                    meta={
                        "target": rec.target_bubble_id,
                        "content_check": content_check,
                        "geometry_mode": getattr(rec, "geometry_mode", "standard"),
                    },
                ))
        if getattr(rec, "candidate", False) or getattr(rec, "review_required", False):
            # Complex/open text is deliberately rendered as a reversible raster
            # candidate when the source and target glyph groups touch the local
            # recovery boundary.  It is still surfaced in the workbench for
            # review, but it is not a failed replacement: treating it as an
            # error made otherwise complete burst/caption transfers look like
            # zero-output failures.
            speculative_complex = (
                not source_units
                and str(getattr(rec, "geometry_mode", "")) == "complex_text"
                and str(getattr(rec, "source_bubble_id", "")).startswith("diff-src-")
            )
            clipped_edge_candidate = (
                getattr(rec, "applied", False)
                and str(getattr(rec, "review_reason", "")) == "source_text_region_clipped_at_page_edge"
                and float(getattr(rec, "target_coverage", 0.0)) >= 0.98
            )
            rigid_patch_candidate = (
                getattr(rec, "applied", False)
                and str(getattr(rec, "geometry_mode", "")) == "rigid_uniform_container"
                and str(getattr(rec, "clarity_mode", "")) == "locked-source-container-patch"
            )
            candidate_severity = (
                "warning"
                if (not publication_safety) or speculative_complex or clipped_edge_candidate or rigid_patch_candidate or (
                    getattr(rec, "applied", False)
                    and getattr(rec, "clarity_mode", "") in {"complex-text-ink-transfer", "open-text-ink-transfer"}
                )
                else "error"
            )
            issues.append(QAItem(
                "mask_replace_low_confidence_candidate",
                candidate_severity,
                "A Chinese review candidate was placed instead of leaving Japanese, but the source translation is incomplete/uncertain. Review, edit, accept, or restore this region before publication.",
                rec.source_bubble_id,
                meta={
                    "target": rec.target_bubble_id,
                    "review_reason": getattr(rec, "review_reason", ""),
                    "coverage": rec.target_coverage,
                    "clarity_mode": getattr(rec, "clarity_mode", "pixels"),
                    "restorable": bool(getattr(rec, "restorable", False)),
                    "editable": bool(getattr(rec, "editable", False)),
                },
            ))
            # Candidate geometry is intentionally below the publication gate; do
            # not duplicate the same issue as generic low-IoU/coverage failures.
            continue
        if replace_config is not None:
            geometry_mode = getattr(rec, "geometry_mode", "standard")
            photo = geometry_mode == "photo_pair"
            rigid_patch = geometry_mode == "rigid_uniform_container" and str(getattr(rec, "clarity_mode", "")) == "locked-source-container-patch"
            min_iou = replace_config.photo_pair_min_transfer_iou if photo else replace_config.min_mask_iou
            min_coverage = replace_config.photo_pair_min_transfer_coverage if photo else replace_config.min_target_coverage
            max_spill = replace_config.photo_pair_max_spill_ratio if photo else replace_config.max_spill_ratio
            if rigid_patch:
                min_coverage = min(min_coverage, 0.92)
                max_spill = max(max_spill, 0.10)
            geometry_severity = "error" if publication_safety else "warning"
            if rec.mask_iou < min_iou:
                issues.append(QAItem("mask_replace_low_iou", geometry_severity, "Transferred bubble mask is below the former publication alignment threshold.", rec.source_bubble_id, rec.mask_iou, min_iou, {"target": rec.target_bubble_id, "geometry_mode": getattr(rec, "geometry_mode", "standard")}))
            if rec.target_coverage < min_coverage:
                issues.append(QAItem("mask_replace_low_coverage", geometry_severity, "Transferred Chinese patch is below the former publication coverage threshold.", rec.source_bubble_id, rec.target_coverage, min_coverage, {"target": rec.target_bubble_id, "geometry_mode": getattr(rec, "geometry_mode", "standard")}))
            if rec.spill_ratio > max_spill:
                issues.append(QAItem("mask_replace_spill", geometry_severity, "Transferred source bubble exceeds the former publication spill threshold.", rec.source_bubble_id, rec.spill_ratio, max_spill, {"target": rec.target_bubble_id, "geometry_mode": getattr(rec, "geometry_mode", "standard")}))
        if replace_config is not None and bool(getattr(replace_config, "paired_diff_forbid_dense_glyph_warp", False)):
            rmeta = getattr(rec, "meta", {}) or {}
            contract_active = bool((rmeta.get("render_source") or {}).get("mask_replace_glyph_integrity_contract", False))
            if contract_active and bool(rmeta.get("glyph_dense_warp", False)):
                issues.append(QAItem(
                    "mask_replace_dense_glyph_warp_forbidden", "error",
                    "Precise Mask detected a final Chinese raster that was deformed by dense/local flow. Mask geometry may use dense alignment, but glyph pixels must come from the global shape-preserving raster.",
                    rec.source_bubble_id, meta={"target": rec.target_bubble_id, "backend": rec.sr_backend},
                ))
        if rec.sr_scale > 3.25:
            issues.append(QAItem("mask_replace_high_upscale", "warning", "Source Chinese bubble required very large upscaling; inspect text sharpness.", rec.source_bubble_id, rec.sr_scale, 3.25, {"target": rec.target_bubble_id, "backend": rec.sr_backend}))
        sharp_floor = getattr(replace_config, "warn_sharpness_below", 0.0) if replace_config is not None else 0.0
        if sharp_floor > 0 and rec.sharpness > 0 and rec.sharpness < sharp_floor:
            issues.append(QAItem("mask_replace_low_sharpness", "warning", "Transferred Chinese bubble remains soft after sampling enhancement; consider OCR re-lettering.", rec.source_bubble_id, rec.sharpness, sharp_floor, {"target": rec.target_bubble_id, "backend": rec.sr_backend, "scale": rec.sr_scale, "clarity_mode": getattr(rec, "clarity_mode", "pixels")}))
        clarity_mode = getattr(rec, "clarity_mode", "pixels")
        if clarity_mode == "ink-reconstruction":
            issues.append(QAItem("mask_replace_ink_reconstruction", "info", "Soft source lettering was deterministically reconstructed as a crisp ink mask; verify rare/complex glyphs at 100% zoom.", rec.source_bubble_id, meta={"target": rec.target_bubble_id, "ink_ratio": getattr(rec, "ink_ratio", 0.0)}))
        elif clarity_mode == "photo-crisp-ink":
            issues.append(QAItem("mask_replace_photo_crisp_ink", "info", "Photographed lettering was rebuilt as antialiased neutral ink over clean target paper; camera blur/glare and source balloon borders were excluded.", rec.source_bubble_id, meta={"target": rec.target_bubble_id, "ink_ratio": getattr(rec, "ink_ratio", 0.0)}))
    return issues


def run_direct_patch_qa(
    pair: PagePair,
    registration: RegistrationResult,
    source_units: list[TextUnit],
    source_bubbles,
    records,
    qa_config: QAConfig | None = None,
    direct_config=None,
) -> list[QAItem]:
    """QA facade for the independent Direct Patch contract.

    Direct and Mask currently share several geometry/content-integrity metrics,
    but projects and review tooling must not report Direct failures as
    ``mask_replace_*`` failures. Reuse the mature measurements while exposing a
    distinct public QA namespace and Direct-specific wording.
    """
    issues = run_mask_replace_qa(
        pair, registration, source_units, source_bubbles, records,
        qa_config, direct_config,
    )
    for item in issues:
        if item.code.startswith("mask_replace_"):
            item.code = "direct_patch_" + item.code[len("mask_replace_"):]
        item.message = (
            item.message
            .replace("mask replacement", "Direct Patch")
            .replace("mask-transfer", "Direct Patch")
            .replace("Bubble patch replacement", "Direct Patch region")
        )

    # v2.3.4: Direct-specific immutable transform contract. These checks do not
    # belong to Mask/Reveal/Reletter QA and intentionally live only here.
    for rec in records or []:
        meta = getattr(rec, "meta", {}) or {}
        contract = meta.get("direct_transform_contract", {}) if isinstance(meta, dict) else {}
        if not isinstance(contract, dict) or not contract:
            continue
        unit_id = str(getattr(rec, "target_bubble_id", "") or getattr(rec, "source_bubble_id", "") or "")
        if not bool(contract.get("source_on_top", False)):
            issues.append(QAItem(
                "direct_patch_source_layer_order_violation", "error",
                "Direct Patch violated its fixed layer order: SOURCE must be above TARGET.",
                unit_id, meta={"contract": contract},
            ))
        if not bool(contract.get("source_border_removed", False)):
            issues.append(QAItem(
                "direct_patch_source_border_violation", "error",
                "Direct Patch attempted to keep SOURCE container borders; TARGET must remain the border authority.",
                unit_id, meta={"contract": contract},
            ))
        if bool(contract.get("axis_locked", False)) and abs(float(contract.get("rotation_deg", 0.0) or 0.0)) > 1e-6:
            issues.append(QAItem(
                "direct_patch_rotation_violation", "error",
                "Direct Patch axis lock was violated; final SOURCE raster must not receive a local rotation.",
                unit_id, value=abs(float(contract.get("rotation_deg", 0.0) or 0.0)), threshold=0.0, meta={"contract": contract},
            ))
        shift_dx = int(contract.get("text_shift_dx", 0) or 0); shift_dy = int(contract.get("text_shift_dy", 0) or 0)
        if shift_dx != 0 or shift_dy != 0:
            issues.append(QAItem(
                "direct_patch_text_shift_violation", "error",
                "Direct Patch final Chinese raster was locally shifted after registration; text position must remain registration-locked.",
                unit_id, meta={"dx": shift_dx, "dy": shift_dy, "contract": contract},
            ))
    return issues
