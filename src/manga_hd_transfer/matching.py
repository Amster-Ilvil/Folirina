from __future__ import annotations

from dataclasses import replace
from math import exp, log
import re

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    def linear_sum_assignment(cost_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        costs = np.asarray(cost_matrix, dtype=float)
        pairs = sorted((float(costs[i, j]), i, j) for i in range(costs.shape[0]) for j in range(costs.shape[1]))
        used_rows: set[int] = set(); used_cols: set[int] = set(); selected: list[tuple[int, int]] = []
        for _, i, j in pairs:
            if i not in used_rows and j not in used_cols:
                selected.append((i, j)); used_rows.add(i); used_cols.add(j)
        selected.sort()
        return np.array([i for i, _ in selected], dtype=int), np.array([j for _, j in selected], dtype=int)

from .config import MatchingConfig
from .geometry import (
    bbox_iou,
    overlap_over_smaller,
    polygon_area,
    polygon_bbox,
    transform_points,
)
from .models import RegistrationResult, TextUnit, UnitMatch


class MatchResult:
    def __init__(
        self,
        matches: list[UnitMatch],
        unmatched_source: list[str],
        unmatched_target: list[str],
        ambiguous_source: list[str] | None = None,
        diagnostics: dict | None = None,
    ) -> None:
        self.matches = matches
        self.unmatched_source = unmatched_source
        self.unmatched_target = unmatched_target
        self.ambiguous_source = ambiguous_source or []
        self.diagnostics = diagnostics or {}


def project_source_units(units: list[TextUnit], registration: RegistrationResult) -> list[TextUnit]:
    return [replace(u, polygon=transform_points(u.polygon, registration.matrix), meta={**u.meta, "projected": True}) for u in units]


def _shape_cost(a: TextUnit, b: TextUnit) -> float:
    aa = max(1.0, polygon_area(a.polygon))
    ba = max(1.0, polygon_area(b.polygon))
    area_cost = min(1.0, abs(log(aa / ba)) / 2.0)
    ab = a.bbox
    bb = b.bbox
    ar_a = max(1e-4, (ab[2] - ab[0]) / max(1.0, ab[3] - ab[1]))
    ar_b = max(1e-4, (bb[2] - bb[0]) / max(1.0, bb[3] - bb[1]))
    aspect_cost = min(1.0, abs(log(ar_a / ar_b)) / 2.0)
    return 0.55 * area_cost + 0.45 * aspect_cost


def _kind_cost(a: TextUnit, b: TextUnit) -> float:
    if a.kind == b.kind:
        return 0.0
    bubble_kinds = {"speech", "narration"}
    if a.kind in bubble_kinds and b.kind in bubble_kinds:
        return 0.30
    if "unknown" in {a.kind, b.kind}:
        return 0.20
    return 0.85


def _normalized_text_len(text: str) -> int:
    if not text:
        return 0
    compact = re.sub(r"\s+", "", str(text))
    return len(compact)


def _text_length_cost(a: TextUnit, b: TextUnit) -> float:
    la = _normalized_text_len(getattr(a, 'text', ''))
    lb = _normalized_text_len(getattr(b, 'text', ''))
    if la <= 1 or lb <= 1:
        return 0.0
    return float(min(1.0, abs(log(max(1.0, la) / max(1.0, lb))) / 2.0))


def _pair_cost(
    source: TextUnit,
    target: TextUnit,
    source_rank: int,
    target_rank: int,
    source_count: int,
    target_count: int,
    target_size: tuple[int, int],
    cfg: MatchingConfig,
    registration_confidence: float = 1.0,
) -> tuple[float, list[str]]:
    sw, sh = target_size
    sx, sy = source.centroid
    tx, ty = target.centroid
    diag = max(1.0, float(np.hypot(sw, sh)))
    centroid = min(1.0, float(np.hypot(sx - tx, sy - ty) / diag) * 4.0)
    overlap = overlap_over_smaller(source.bbox, target.bbox)
    overlap_cost = 1.0 - overlap
    projected_iou = bbox_iou(source.bbox, target.bbox)
    projected_iou_cost = 1.0 - projected_iou
    text_length = _text_length_cost(source, target)
    shape = _shape_cost(source, target)
    sr = source_rank / max(1, source_count - 1)
    tr = target_rank / max(1, target_count - 1)
    order = min(1.0, abs(sr - tr) * 2.0)
    kind = _kind_cost(source, target)
    registration_penalty = (1.0 - float(np.clip(registration_confidence, 0.0, 1.0))) * float(
        getattr(cfg, "registration_confidence_penalty_weight", 0.12)
    )
    cost = (
        cfg.centroid_weight * centroid
        + cfg.overlap_weight * overlap_cost
        + cfg.projected_iou_weight * projected_iou_cost
        + cfg.text_length_weight * text_length
        + cfg.shape_weight * shape
        + cfg.order_weight * order
        + cfg.kind_weight * kind
        + registration_penalty
    )
    if overlap >= float(getattr(cfg, 'replace_translation_overlap_gate', 0.30)):
        cost = max(0.0, cost - float(getattr(cfg, 'replace_translation_overlap_bonus', 0.05)))
    reasons = [
        f"centroid={centroid:.3f}",
        f"overlap={overlap:.3f}",
        f"projected_iou={projected_iou:.3f}",
        f"text_length={text_length:.3f}",
        f"shape={shape:.3f}",
        f"order={order:.3f}",
        f"kind={kind:.3f}",
        f"registration_penalty={registration_penalty:.3f}",
    ]
    return float(cost), reasons


def match_units(
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    registration: RegistrationResult,
    config: MatchingConfig | None = None,
) -> MatchResult:
    cfg = config or MatchingConfig()
    if not source_units or not target_units:
        return MatchResult(
            [], [u.id for u in source_units], [u.id for u in target_units], diagnostics={
                "registration_confidence": float(registration.confidence),
                "top_candidates": {},
                "rejected_over_max_cost": [],
                "ambiguities": [],
                "force_actions": ["force_match", "skip_unit", "force_direct_patch", "force_mask_replace"],
            }
        )

    projected = project_source_units(source_units, registration)
    n, m = len(projected), len(target_units)
    costs = np.zeros((n, m), dtype=np.float64)
    reasons: dict[tuple[int, int], list[str]] = {}
    for i, s in enumerate(projected):
        for j, t in enumerate(target_units):
            c, r = _pair_cost(
                s, t, i, j, n, m, registration.target_size, cfg,
                registration_confidence=float(registration.confidence),
            )
            costs[i, j] = c
            reasons[(i, j)] = r

    rows, cols = linear_sum_assignment(costs)
    matches: list[UnitMatch] = []
    used_s: set[int] = set()
    used_t: set[int] = set()
    ambiguous: list[str] = []
    rejected_over_max_cost: list[dict] = []

    for i, j in zip(rows, cols):
        cost = float(costs[i, j])
        if cost > cfg.max_cost:
            rejected_over_max_cost.append({
                "source_unit_id": source_units[i].id,
                "target_unit_id": target_units[j].id,
                "cost": cost,
                "max_cost": float(cfg.max_cost),
                "reasons": list(reasons[(i, j)]),
            })
            continue
        row_sorted = np.sort(costs[i])
        margin = float(row_sorted[1] - row_sorted[0]) if m > 1 else 1.0
        base_conf = float(np.clip(1.0 - cost / max(cfg.max_cost, 1e-6), 0.0, 1.0))
        ambiguity_factor = float(np.clip(0.45 + margin / 0.22, 0.45, 1.0))
        confidence = base_conf * ambiguity_factor
        rs = list(reasons[(i, j)]) + [f"assignment_margin={margin:.3f}"]
        if confidence < cfg.review_confidence:
            rs.append("review_recommended")
            ambiguous.append(source_units[i].id)
        matches.append(
            UnitMatch(
                source_unit_id=source_units[i].id,
                target_unit_id=target_units[j].id,
                confidence=confidence,
                cost=cost,
                relation="one_to_one",
                reasons=rs,
            )
        )
        used_s.add(i)
        used_t.add(j)

    # Explicitly discover split/merge candidates. They are reported but intentionally
    # low-confidence: publication mode must not silently duplicate or concatenate text.
    for i, s in enumerate(projected):
        if i in used_s:
            # One source unit can geometrically cover a second target unit.
            primary = next((x for x in matches if x.source_unit_id == source_units[i].id), None)
            if primary is None:
                continue
            for j, t in enumerate(target_units):
                if j in used_t:
                    continue
                overlap = overlap_over_smaller(s.bbox, t.bbox)
                if overlap >= float(getattr(cfg, "replace_translation_many_to_one_overlap", 0.58)):
                    matches.append(
                        UnitMatch(
                            source_unit_id=source_units[i].id,
                            target_unit_id=t.id,
                            confidence=min(0.55, overlap * 0.65),
                            cost=1.0 - overlap,
                            relation="one_to_many",
                            reasons=[f"projected_overlap={overlap:.3f}", "manual_split_required"],
                        )
                    )
                    used_t.add(j)
                    ambiguous.append(source_units[i].id)
        else:
            # Multiple source units can land in the same target bubble.
            best_j = int(np.argmin(costs[i]))
            if best_j in used_t and costs[i, best_j] <= min(cfg.max_cost, 0.60):
                matches.append(
                    UnitMatch(
                        source_unit_id=source_units[i].id,
                        target_unit_id=target_units[best_j].id,
                        confidence=0.48,
                        cost=float(costs[i, best_j]),
                        relation="many_to_one",
                        reasons=reasons[(i, best_j)] + ["manual_merge_required"],
                    )
                )
                used_s.add(i)
                ambiguous.append(source_units[i].id)

    unmatched_s = [source_units[i].id for i in range(n) if i not in used_s]
    unmatched_t = [target_units[j].id for j in range(m) if j not in used_t]
    top_k = max(1, int(getattr(cfg, "diagnostics_top_k", 3)))
    top_candidates: dict[str, list[dict]] = {}
    for i, unit in enumerate(source_units):
        ranked = np.argsort(costs[i])[: min(top_k, m)]
        top_candidates[unit.id] = [
            {
                "target_unit_id": target_units[int(j)].id,
                "cost": float(costs[i, int(j)]),
                "reasons": list(reasons[(i, int(j))]),
            }
            for j in ranked
        ]
    ambiguity_rows = [m.to_dict() for m in matches if m.relation != "one_to_one" or m.source_unit_id in ambiguous]
    diagnostics = {
        "registration_confidence": float(registration.confidence),
        "top_candidates": {k: v for k, v in top_candidates.items() if k in unmatched_s or k in set(ambiguous)},
        "rejected_over_max_cost": rejected_over_max_cost,
        "ambiguities": ambiguity_rows,
        "force_actions": ["force_match", "skip_unit", "force_direct_patch", "force_mask_replace"],
    }
    return MatchResult(matches, unmatched_s, unmatched_t, sorted(set(ambiguous)), diagnostics)
