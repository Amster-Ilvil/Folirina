from __future__ import annotations

"""Optional joined-balloon flow-cell geometry for Reletter mode.

This is a clean-room Python implementation inspired by Koharu's high-level idea
of assigning one non-overlapping layout cell to each independent text flow in a
joined manga balloon.  It does not copy Koharu's Rust implementation.

Safety rules:
- input is an already-trusted bubble/safe mask;
- cells never extend outside that mask;
- cells are mutually exclusive and cover the original mask;
- a physical neck cut is accepted only when it actually separates the anchors;
- otherwise the function falls back to an anchor Voronoi partition.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np


Point = tuple[float, float]


@dataclass(slots=True)
class FlowCellPartition:
    masks: list[np.ndarray]
    anchors: list[Point]
    route: str
    diagnostics: dict = field(default_factory=dict)


def _binary(mask: np.ndarray) -> np.ndarray:
    if mask is None or getattr(mask, "size", 0) == 0:
        return np.zeros((0, 0), np.uint8)
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return (mask > 0).astype(np.uint8) * 255


def _nearest_foreground(mask: np.ndarray, point: Point) -> tuple[int, int] | None:
    h, w = mask.shape[:2]
    if h == 0 or w == 0 or cv2.countNonZero(mask) == 0:
        return None
    x = int(np.clip(round(float(point[0])), 0, w - 1))
    y = int(np.clip(round(float(point[1])), 0, h - 1))
    if mask[y, x] > 0:
        return x, y
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    d2 = (xs.astype(np.float64) - x) ** 2 + (ys.astype(np.float64) - y) ** 2
    idx = int(np.argmin(d2))
    return int(xs[idx]), int(ys[idx])


def _seed_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    work = (mask > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(work, 8)
    if n <= 1:
        return work * 255
    x, y = seed
    label = int(labels[int(np.clip(y, 0, labels.shape[0] - 1)), int(np.clip(x, 0, labels.shape[1] - 1))])
    if label <= 0:
        label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == label).astype(np.uint8) * 255


def _anchor_voronoi(mask: np.ndarray, seeds: list[tuple[int, int]]) -> list[np.ndarray]:
    """Euclidean anchor partition clipped to the trusted bubble mask."""
    ys, xs = np.where(mask > 0)
    out = [np.zeros_like(mask) for _ in seeds]
    if len(xs) == 0 or not seeds:
        return out
    # Number of flows in a manga balloon is tiny.  Computing the distances only
    # for foreground pixels avoids a H*W*N tensor for a full page.
    coords_x = xs.astype(np.float64)[:, None]
    coords_y = ys.astype(np.float64)[:, None]
    ax = np.asarray([s[0] for s in seeds], dtype=np.float64)[None, :]
    ay = np.asarray([s[1] for s in seeds], dtype=np.float64)[None, :]
    labels = np.argmin((coords_x - ax) ** 2 + (coords_y - ay) ** 2, axis=1)
    for idx in range(len(seeds)):
        sel = labels == idx
        out[idx][ys[sel], xs[sel]] = 255
        out[idx] = _seed_component(out[idx], seeds[idx])

    # If a clipped Voronoi cell had a disconnected island, assign the abandoned
    # pixels to the nearest surviving cell. This restores complete coverage while
    # keeping cells mutually exclusive.
    occupied = np.zeros_like(mask)
    for cell in out:
        occupied = cv2.bitwise_or(occupied, cell)
    missing_y, missing_x = np.where((mask > 0) & (occupied == 0))
    if len(missing_x):
        mx = missing_x.astype(np.float64)[:, None]
        my = missing_y.astype(np.float64)[:, None]
        labels = np.argmin((mx - ax) ** 2 + (my - ay) ** 2, axis=1)
        for idx in range(len(seeds)):
            sel = labels == idx
            out[idx][missing_y[sel], missing_x[sel]] = 255
    return out


def _line_mask(shape: tuple[int, int], center: tuple[float, float], direction: tuple[float, float], span: float, thickness: int) -> np.ndarray:
    h, w = shape
    cx, cy = center
    dx, dy = direction
    p0 = (int(round(cx - dx * span)), int(round(cy - dy * span)))
    p1 = (int(round(cx + dx * span)), int(round(cy + dy * span)))
    cut = np.zeros((h, w), np.uint8)
    cv2.line(cut, p0, p1, 255, max(1, int(thickness)), cv2.LINE_8)
    return cut


def _try_two_anchor_neck(mask: np.ndarray, seeds: list[tuple[int, int]]) -> tuple[list[np.ndarray] | None, dict]:
    """Find a narrow cross-section between two text anchors.

    Instead of reproducing Koharu's polygon/reflex-vertex algorithm, this
    clean-room version scans cross-sections between the two anchors in raster
    space. A candidate is accepted only if cutting it splits the trusted bubble
    into two meaningful components and the selected cross-section is distinctly
    narrower than typical alternatives.
    """
    if len(seeds) != 2:
        return None, {"reason": "requires_two_anchors"}
    (x0, y0), (x1, y1) = seeds
    vx, vy = float(x1 - x0), float(y1 - y0)
    length = float(np.hypot(vx, vy))
    if length < 8.0:
        return None, {"reason": "anchors_too_close", "anchor_distance": round(length, 3)}
    ux, uy = vx / length, vy / length
    # Cross-section is perpendicular to anchor flow.
    px, py = -uy, ux
    h, w = mask.shape
    span = float(np.hypot(w, h) + 4.0)
    scale = max(1.0, min(w, h))
    thickness = max(1, int(round(scale * 0.0025)))
    area = max(1, cv2.countNonZero(mask))

    candidates: list[dict] = []
    for t in np.linspace(0.22, 0.78, 29):
        cx = x0 + vx * float(t)
        cy = y0 + vy * float(t)
        cut = _line_mask((h, w), (cx, cy), (px, py), span, thickness)
        cross_pixels = int(cv2.countNonZero(cv2.bitwise_and(mask, cut)))
        if cross_pixels <= 0:
            continue
        cut_mask = mask.copy()
        cut_mask[cut > 0] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats((cut_mask > 0).astype(np.uint8), 8)
        l0 = int(labels[y0, x0]) if 0 <= x0 < w and 0 <= y0 < h else 0
        l1 = int(labels[y1, x1]) if 0 <= x1 < w and 0 <= y1 < h else 0
        if l0 <= 0 or l1 <= 0 or l0 == l1:
            continue
        a0 = int(stats[l0, cv2.CC_STAT_AREA]); a1 = int(stats[l1, cv2.CC_STAT_AREA])
        if min(a0, a1) < area * 0.08:
            continue
        candidates.append({
            "t": float(t), "cross_pixels": cross_pixels, "labels": labels,
            "label0": l0, "label1": l1, "area0": a0, "area1": a1,
        })

    if not candidates:
        return None, {"reason": "no_separating_cross_section"}
    widths = np.asarray([c["cross_pixels"] for c in candidates], dtype=np.float64)
    typical = float(np.median(widths))
    best = min(candidates, key=lambda c: (c["cross_pixels"], abs(c["t"] - 0.5)))
    bottleneck_ratio = float(best["cross_pixels"] / max(1.0, typical))
    # Uniform rectangles/ellipses can be split by any line but do not have a
    # physical neck. Requiring a clear bottleneck prevents topology hallucination.
    if bottleneck_ratio > 0.78:
        return None, {
            "reason": "no_distinct_neck",
            "best_cross_pixels": int(best["cross_pixels"]),
            "median_cross_pixels": round(typical, 3),
            "bottleneck_ratio": round(bottleneck_ratio, 4),
        }

    labels = best["labels"]
    cells = [
        (labels == int(best["label0"])).astype(np.uint8) * 255,
        (labels == int(best["label1"])).astype(np.uint8) * 255,
    ]
    occupied = cv2.bitwise_or(cells[0], cells[1])
    missing = (mask > 0) & (occupied == 0)
    if np.any(missing):
        ys, xs = np.where(missing)
        d0 = (xs - x0) ** 2 + (ys - y0) ** 2
        d1 = (xs - x1) ** 2 + (ys - y1) ** 2
        first = d0 <= d1
        cells[0][ys[first], xs[first]] = 255
        cells[1][ys[~first], xs[~first]] = 255
    return cells, {
        "reason": "accepted",
        "anchor_distance": round(length, 3),
        "cut_fraction": round(float(best["t"]), 4),
        "best_cross_pixels": int(best["cross_pixels"]),
        "median_cross_pixels": round(typical, 3),
        "bottleneck_ratio": round(bottleneck_ratio, 4),
        "cell_areas": [int(cv2.countNonZero(c)) for c in cells],
    }


def partition_flow_cells(mask: np.ndarray, anchors: list[Point], *, prefer_neck: bool = True) -> FlowCellPartition:
    """Partition a trusted bubble mask into one non-overlapping cell per anchor."""
    binary = _binary(mask)
    if binary.size == 0 or cv2.countNonZero(binary) == 0 or not anchors:
        return FlowCellPartition([], [], "empty", {"reason": "empty_input"})

    resolved: list[tuple[int, int]] = []
    kept_anchors: list[Point] = []
    # A malformed duplicate anchor must not create a zero-area phantom flow.
    min_dim = max(1.0, float(min(binary.shape[:2])))
    duplicate_d2 = (min_dim * 0.003) ** 2
    for anchor in anchors[:8]:
        seed = _nearest_foreground(binary, anchor)
        if seed is None:
            continue
        if any((seed[0] - sx) ** 2 + (seed[1] - sy) ** 2 <= duplicate_d2 for sx, sy in resolved):
            continue
        resolved.append(seed)
        kept_anchors.append((float(anchor[0]), float(anchor[1])))

    if not resolved:
        return FlowCellPartition([], [], "empty", {"reason": "no_valid_anchors"})
    if len(resolved) == 1:
        return FlowCellPartition([binary.copy()], kept_anchors, "single", {"cell_areas": [int(cv2.countNonZero(binary))]})

    neck_diag = None
    if prefer_neck and len(resolved) == 2:
        cells, neck_diag = _try_two_anchor_neck(binary, resolved)
        if cells is not None:
            return FlowCellPartition(cells, kept_anchors, "neck_cut", {"neck": neck_diag})

    cells = _anchor_voronoi(binary, resolved)
    return FlowCellPartition(
        cells,
        kept_anchors,
        "anchor_voronoi",
        {
            "neck": neck_diag or {"reason": "not_attempted"},
            "cell_areas": [int(cv2.countNonZero(c)) for c in cells],
        },
    )


__all__ = ["FlowCellPartition", "partition_flow_cells"]
