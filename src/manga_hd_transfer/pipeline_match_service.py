from __future__ import annotations

"""Text-unit construction, identity matching and auto-acceptance policy."""

from dataclasses import dataclass
from typing import Any

from .bubbles import build_reletter_units, build_text_units
from .matching import match_units, match_units_by_paired_ids
from .models import PagePair, TextUnit, UnitMatch


@dataclass
class TextMatchStageResult:
    source_units: list[TextUnit]
    target_units: list[TextUnit]
    match_result: Any
    matches: list[UnitMatch]
    accepted: list[UnitMatch]
    paired_reletter_binding: bool
    pair_confidence_bypassed: bool


def accepted_matches(
    pair: PagePair,
    registration_confidence: float,
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    matches: list[UnitMatch],
    *,
    config: Any,
    paired_reletter_authority: bool = False,
) -> list[UnitMatch]:
    # A restored photographed-page pair can carry a weak historical pairing
    # heuristic even after SIFT/ECC registration has proven the geometry and
    # Reletter has built deterministic SOURCE↔TARGET region IDs.  Treating the
    # old page-pair score as a second hard veto silently dropped *all* OCR
    # lettering (the supplied page 009 had 5 paired matches and 0 accepted).
    # Only explicit target-driven Reletter may bypass this one heuristic gate;
    # registration, OCR confidence, kind, relation and match confidence remain
    # mandatory below. Direct/Mask/Hybrid semantics are unchanged.
    if pair.confidence < config.pairing.confidence_floor and not paired_reletter_authority:
        return []
    if registration_confidence < config.qa.registration_min_confidence:
        return []
    su = {u.id: u for u in source_units}
    tu = {u.id: u for u in target_units}
    accepted: list[UnitMatch] = []
    for match in matches:
        s, t = su.get(match.source_unit_id), tu.get(match.target_unit_id)
        if s is None or t is None:
            continue
        if match.relation != "one_to_one":
            continue
        if match.confidence < config.matching.review_confidence:
            continue
        if s.confidence < config.ocr.min_confidence or t.confidence < config.ocr.min_confidence:
            continue
        if s.kind not in config.matching.auto_apply_kinds or t.kind not in config.matching.auto_apply_kinds:
            continue
        if not s.text.strip():
            continue
        accepted.append(match)
    return accepted


def run_text_matching_stage(
    pair: PagePair,
    registration: Any,
    source_blocks,
    target_blocks,
    source_bubbles,
    target_bubbles,
    *,
    config: Any,
    mode_contract: Any,
    mode: str,
    use_paired_diff: bool,
    target_driven_reletter_regions: bool,
) -> TextMatchStageResult:
    if mode_contract.reletter and mode == "reletter" and target_driven_reletter_regions:
        source_units = build_reletter_units(source_blocks, "src")
        target_units = build_reletter_units(target_blocks, "dst")
    else:
        source_units = build_text_units(source_blocks, source_bubbles, "src")
        target_units = build_text_units(target_blocks, target_bubbles, "dst")

    paired_reletter_binding = bool(
        mode_contract.reletter
        and mode == "reletter"
        and use_paired_diff
        and source_units
        and target_units
        and any(
            str((u.meta or {}).get("paired_target_region_id") or "")
            or str((u.meta or {}).get("paired_target_id") or "")
            for u in source_units
        )
    )
    if paired_reletter_binding:
        match_result = match_units_by_paired_ids(
            source_units, target_units, registration, config.matching
        )
    else:
        match_result = match_units(
            source_units, target_units, registration, config.matching
        )
    matches = match_result.matches
    pair_confidence_bypassed = bool(
        float(pair.confidence) < float(config.pairing.confidence_floor)
        and paired_reletter_binding
        and target_driven_reletter_regions
        and float(registration.confidence) >= float(config.qa.registration_min_confidence)
    )
    accepted = accepted_matches(
        pair,
        float(registration.confidence),
        source_units,
        target_units,
        matches,
        config=config,
        paired_reletter_authority=pair_confidence_bypassed,
    )
    return TextMatchStageResult(
        source_units=source_units,
        target_units=target_units,
        match_result=match_result,
        matches=matches,
        accepted=accepted,
        paired_reletter_binding=paired_reletter_binding,
        pair_confidence_bypassed=pair_confidence_bypassed,
    )


__all__ = ["TextMatchStageResult", "accepted_matches", "run_text_matching_stage"]
