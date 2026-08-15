from __future__ import annotations

"""Publication-oriented arbitration for primary vs secondary translated sources.

A secondary source is evidence, not translation authority.  This module scores
only whether a candidate is a *better raster/geometry source* for Direct Patch.
It never compares or chooses translation text.
"""

from dataclasses import asdict, dataclass
import math
from typing import Any

import cv2
import numpy as np


@dataclass(slots=True)
class DirectSourceEvidence:
    path: str
    kind: str
    is_secondary: bool
    safe: bool
    same_page_confidence: float
    registration_confidence: float
    reprojection_error: float
    applied_count: int
    candidate_count: int
    coverage: float
    median_boundary_distance: float | None
    boundary_score: float
    sharpness: float
    sharpness_score: float
    residual_risk: float
    arbitration_score: float
    reject_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clamp01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _masked_direct_sharpness(source: np.ndarray, plan) -> float:
    if source is None or source.size == 0:
        return 0.0
    gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source.astype(np.uint8)
    masks: list[np.ndarray] = []
    if plan is not None:
        for bubble in list(getattr(plan, "source_bubbles", []) or []):
            mask = getattr(bubble, "mask", None)
            if mask is not None and mask.shape == gray.shape:
                masks.append(mask > 0)
    if masks:
        use = np.zeros(gray.shape, bool)
        for mask in masks:
            use |= mask
        ys, xs = np.where(use)
        if len(xs) >= 64:
            x0, x1 = max(0, int(xs.min()) - 2), min(gray.shape[1], int(xs.max()) + 3)
            y0, y1 = max(0, int(ys.min()) - 2), min(gray.shape[0], int(ys.max()) + 3)
            roi = gray[y0:y1, x0:x1]
            roi_mask = use[y0:y1, x0:x1]
            lap = cv2.Laplacian(roi, cv2.CV_32F)
            vals = lap[roi_mask]
            if vals.size >= 32:
                return float(np.var(vals))
    # Fallback is deliberately low-frequency tolerant but still rewards a sharper
    # scan. It is only one term in arbitration, never the sole selector.
    max_side = 1400
    h, w = gray.shape
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        gray = cv2.resize(gray, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def build_direct_source_evidence(
    *,
    path: str,
    kind: str,
    is_secondary: bool,
    source: np.ndarray,
    registration,
    pair_check,
    plan,
    config,
) -> DirectSourceEvidence:
    diagnostics = dict(getattr(plan, "diagnostics", {}) or {}) if plan is not None else {}
    applied = int(getattr(getattr(plan, "result", None), "applied_count", 0)) if plan is not None else 0
    candidates = int(diagnostics.get("candidate_count", 0) or 0)
    coverage = float(applied / max(1, candidates)) if candidates else (1.0 if applied > 0 else 0.0)
    boundary = diagnostics.get("median_boundary_distance")
    boundary = float(boundary) if boundary is not None else None
    boundary_norm = float(getattr(config, "arbitration_boundary_distance_good_px", 2.5))
    boundary_score = _clamp01(1.0 - ((boundary if boundary is not None else boundary_norm * 2.0) / max(0.25, boundary_norm * 2.0)))
    sharpness = _masked_direct_sharpness(source, plan)
    sharp_ref = max(1.0, float(getattr(config, "arbitration_sharpness_reference", 180.0)))
    sharpness_score = _clamp01(math.log1p(max(0.0, sharpness)) / math.log1p(sharp_ref))
    same_page = _clamp01(float(getattr(pair_check, "confidence", 0.0)))
    reg_conf = _clamp01(float(getattr(registration, "confidence", 0.0)))
    reproj = max(0.0, float(getattr(registration, "reprojection_error", 999.0)))
    reproj_ref = max(0.25, float(getattr(config, "arbitration_reprojection_good_px", 3.0)))
    reproj_score = float(math.exp(-reproj / reproj_ref))

    review_count = sum(1 for r in list(getattr(getattr(plan, "result", None), "records", []) or []) if bool(getattr(r, "review_required", False)))
    incomplete_count = sum(1 for r in list(getattr(getattr(plan, "result", None), "records", []) or []) if str(getattr(r, "content_check", "")).startswith("checked") and not bool(getattr(r, "content_complete", False)))
    review_candidates = int(diagnostics.get("review_candidates_skipped", 0) or 0)
    rejected_alignment = int(diagnostics.get("rejected_alignment", 0) or 0)
    risk_events = review_count + incomplete_count + review_candidates + rejected_alignment
    residual_risk = _clamp01(risk_events / max(1.0, float(applied + risk_events)))

    safe = bool(plan is not None and applied > 0 and bool(getattr(plan, "safe_to_skip_other_paths", False)))
    reject_reasons: list[str] = []
    min_pair = float(getattr(config, "arbitration_min_same_page_confidence", 0.72))
    min_reg = float(getattr(config, "arbitration_min_registration_confidence", 0.78))
    max_reproj = float(getattr(config, "arbitration_max_reprojection_error_px", 7.0))
    if same_page < min_pair:
        reject_reasons.append("same_page_confidence_below_gate")
    if reg_conf < min_reg:
        reject_reasons.append("registration_confidence_below_gate")
    if reproj > max_reproj:
        reject_reasons.append("reprojection_error_above_gate")
    if not safe:
        reject_reasons.append("direct_plan_not_publication_safe")

    # Weights intentionally emphasize identity/geometry before sharpness. A very
    # sharp wrong scan must never beat a slightly softer correctly aligned scan.
    weights = {
        "same_page": float(getattr(config, "arbitration_same_page_weight", 0.24)),
        "registration": float(getattr(config, "arbitration_registration_weight", 0.20)),
        "reprojection": float(getattr(config, "arbitration_reprojection_weight", 0.12)),
        "coverage": float(getattr(config, "arbitration_coverage_weight", 0.18)),
        "boundary": float(getattr(config, "arbitration_boundary_weight", 0.10)),
        "sharpness": float(getattr(config, "arbitration_sharpness_weight", 0.12)),
        "risk": float(getattr(config, "arbitration_risk_weight", 0.04)),
    }
    denom = max(1e-9, sum(max(0.0, v) for v in weights.values()))
    score = (
        weights["same_page"] * same_page
        + weights["registration"] * reg_conf
        + weights["reprojection"] * reproj_score
        + weights["coverage"] * _clamp01(coverage)
        + weights["boundary"] * boundary_score
        + weights["sharpness"] * sharpness_score
        + weights["risk"] * (1.0 - residual_risk)
    ) / denom
    if reject_reasons:
        score *= float(getattr(config, "arbitration_rejected_score_multiplier", 0.25))

    return DirectSourceEvidence(
        path=str(path), kind=str(kind), is_secondary=bool(is_secondary), safe=safe,
        same_page_confidence=same_page, registration_confidence=reg_conf,
        reprojection_error=reproj, applied_count=applied, candidate_count=candidates,
        coverage=_clamp01(coverage), median_boundary_distance=boundary,
        boundary_score=boundary_score, sharpness=sharpness,
        sharpness_score=sharpness_score, residual_risk=residual_risk,
        arbitration_score=_clamp01(score), reject_reasons=reject_reasons,
    )


def select_direct_source_candidate(candidates: list[tuple[DirectSourceEvidence, Any]]) -> tuple[DirectSourceEvidence, Any] | None:
    """Choose the highest-scoring publication-safe candidate.

    Unsafe candidates stay in diagnostics but can never win arbitration. Ties are
    resolved in favor of the primary source to preserve the authority principle.
    """
    eligible = [(ev, payload) for ev, payload in candidates if ev.safe and not ev.reject_reasons]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[0].arbitration_score, not item[0].is_secondary), reverse=True)
    return eligible[0]
