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
