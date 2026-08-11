from __future__ import annotations

from dataclasses import replace
from math import exp, log

import numpy as np
from scipy.optimize import linear_sum_assignment

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
    ) -> None:
        self.matches = matches
        self.unmatched_source = unmatched_source
        self.unmatched_target = unmatched_target
        self.ambiguous_source = ambiguous_source or []


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


def _pair_cost(
    source: TextUnit,
    target: TextUnit,
    source_rank: int,
    target_rank: int,
    source_count: int,
    target_count: int,
    target_size: tuple[int, int],
    cfg: MatchingConfig,
) -> tuple[float, list[str]]:
    sw, sh = target_size
    sx, sy = source.centroid
    tx, ty = target.centroid
    diag = max(1.0, float(np.hypot(sw, sh)))
    centroid = min(1.0, float(np.hypot(sx - tx, sy - ty) / diag) * 4.0)
    overlap = overlap_over_smaller(source.bbox, target.bbox)
    overlap_cost = 1.0 - overlap
    shape = _shape_cost(source, target)
    sr = source_rank / max(1, source_count - 1)
    tr = target_rank / max(1, target_count - 1)
    order = min(1.0, abs(sr - tr) * 2.0)
    kind = _kind_cost(source, target)
    cost = (
        cfg.centroid_weight * centroid
        + cfg.overlap_weight * overlap_cost
        + cfg.shape_weight * shape
        + cfg.order_weight * order
        + cfg.kind_weight * kind
    )
    reasons = [
        f"centroid={centroid:.3f}",
        f"overlap={overlap:.3f}",
        f"shape={shape:.3f}",
        f"order={order:.3f}",
        f"kind={kind:.3f}",
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
        return MatchResult([], [u.id for u in source_units], [u.id for u in target_units])

    projected = project_source_units(source_units, registration)
    n, m = len(projected), len(target_units)
    costs = np.zeros((n, m), dtype=np.float64)
    reasons: dict[tuple[int, int], list[str]] = {}
    for i, s in enumerate(projected):
        for j, t in enumerate(target_units):
            c, r = _pair_cost(s, t, i, j, n, m, registration.target_size, cfg)
            costs[i, j] = c
            reasons[(i, j)] = r

    rows, cols = linear_sum_assignment(costs)
    matches: list[UnitMatch] = []
    used_s: set[int] = set()
    used_t: set[int] = set()
    ambiguous: list[str] = []

    for i, j in zip(rows, cols):
        cost = float(costs[i, j])
        if cost > cfg.max_cost:
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
                if overlap >= 0.62:
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
    return MatchResult(matches, unmatched_s, unmatched_t, sorted(set(ambiguous)))
