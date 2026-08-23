from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from ...models import BubbleInstance
from .flow_cells import FlowCellPartition, partition_flow_cells


@dataclass(slots=True)
class TargetTextRegion:
    id: str
    target_bubble_id: str
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[float, float]]
    text_mask: np.ndarray
    orientation: str
    component_count: int
    confidence: float
    diagnostics: dict


def _protected_mask(bubble: BubbleInstance, shape: tuple[int, int]) -> np.ndarray | None:
    h, w = shape
    candidate = bubble.safe_mask if bubble.safe_mask is not None else bubble.mask
    if candidate is None:
        return None
    mask = (candidate > 0).astype(np.uint8) * 255
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    if cv2.countNonZero(mask) == 0:
        return None
    return mask


def _component_rows(image: np.ndarray, bubble: BubbleInstance) -> tuple[list[dict], np.ndarray, dict]:
    h, w = image.shape[:2]
    protected = _protected_mask(bubble, (h, w))
    if protected is None:
        return [], np.zeros((h, w), np.uint8), {"reason": "empty_bubble_mask"}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    vals = gray[protected > 0]
    if vals.size == 0:
        return [], np.zeros((h, w), np.uint8), {"reason": "empty_bubble_values"}
    paper = float(np.percentile(vals, 80))
    threshold = int(np.clip(paper - 48.0, 115, 205))
    raw = ((gray < threshold) & (protected > 0)).astype(np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    safe_area = float(max(1, cv2.countNonZero(protected)))
    bx0, by0, bx1, by1 = bubble.bbox
    bw, bh = max(1.0, bx1 - bx0), max(1.0, by1 - by0)
    # Distance to the trusted bubble edge is a much stronger discriminator than
    # raw component size. Manga balloon contours are dark connected components
    # too; on small balloons they can outnumber the actual glyph components and
    # poison median-size heuristics. Keep edge distance per component so the
    # target text detector can reason about *interior glyphs* instead of borders.
    distance = cv2.distanceTransform((protected > 0).astype(np.uint8), cv2.DIST_L2, 5)
    raw_rows: list[dict] = []
    for idx in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        if area < 2:
            continue
        if area > safe_area * 0.05 or cw > bw * 0.45 or ch > bh * 0.45:
            continue
        component_pixels = labels == idx
        distances = distance[component_pixels]
        boundary_distance = float(np.median(distances)) if distances.size else 0.0
        raw_rows.append({
            "label": idx, "x": x, "y": y, "w": cw, "h": ch, "area": area,
            "cx": float(centroids[idx][0]), "cy": float(centroids[idx][1]),
            "fill_ratio": float(area / max(1, cw * ch)),
            "boundary_distance_px": boundary_distance,
            "_mask": (labels[y:y + ch, x:x + cw] == idx).astype(np.uint8) * 255,
        })

    boundary_margin = float(np.clip(min(bw, bh) * 0.035, 2.0, 6.0))

    def glyph_like(row: dict, *, satellite: bool = False) -> bool:
        cw = int(row["w"]); ch = int(row["h"]); area = int(row["area"])
        min_dim = min(cw, ch); max_dim = max(cw, ch)
        aspect = max_dim / max(1.0, float(min_dim))
        min_boundary = boundary_margin * (0.58 if satellite else 1.0)
        return bool(
            min_dim >= (2 if satellite else 3)
            and area >= (3 if satellite else 5)
            and float(row.get("fill_ratio", 0.0)) >= (0.055 if satellite else 0.08)
            and aspect <= (6.5 if satellite else 5.5)
            and float(row.get("boundary_distance_px", 0.0)) >= max(1.35, min_boundary)
        )

    core = [row for row in raw_rows if glyph_like(row)]
    # Punctuation and detached kana strokes can be smaller than the core glyphs.
    # Admit them only when they are spatially close to a proven interior glyph;
    # this retains dakuten / punctuation without re-admitting long balloon edges.
    selected = list(core)
    if core:
        med_extent = float(np.median([max(int(r["w"]), int(r["h"])) for r in core]))
        satellite_radius = float(np.clip(med_extent * 1.85, 12.0, 34.0))
        core_xy = np.asarray([[float(r["cx"]), float(r["cy"])] for r in core], dtype=np.float32)
        selected_labels = {int(r["label"]) for r in selected}
        for row in raw_rows:
            if int(row["label"]) in selected_labels or not glyph_like(row, satellite=True):
                continue
            pt = np.asarray([float(row["cx"]), float(row["cy"])], dtype=np.float32)
            if float(np.min(np.linalg.norm(core_xy - pt, axis=1))) <= satellite_radius:
                selected.append(row)
                selected_labels.add(int(row["label"]))

    # Degenerate tiny balloons (e.g. 「チッ」) can contain only two or three
    # meaningful connected glyph pieces. If strict core selection yielded too
    # little, retain the strongest interior compact pieces rather than declaring
    # the whole bubble text-free.
    if len(selected) < 2:
        fallback = [
            row for row in raw_rows
            if min(int(row["w"]), int(row["h"])) >= 2
            and int(row["area"]) >= 4
            and float(row.get("boundary_distance_px", 0.0)) >= max(1.5, boundary_margin * 0.55)
            and max(int(row["w"]), int(row["h"])) / max(1.0, float(min(int(row["w"]), int(row["h"])))) <= 6.0
        ]
        fallback.sort(key=lambda r: (float(r.get("boundary_distance_px", 0.0)), int(r["area"])), reverse=True)
        selected = fallback[: max(2, min(8, len(fallback)))]

    selected.sort(key=lambda r: (float(r["cy"]), float(r["cx"])))
    kept = np.zeros_like(raw)
    for row in selected:
        x, y, cw, ch = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
        kept[y:y + ch, x:x + cw] = cv2.bitwise_or(
            kept[y:y + ch, x:x + cw], row["_mask"]
        )
    diag = {
        "threshold": threshold,
        "component_count": len(selected),
        "raw_component_count": len(raw_rows),
        "selected_component_count": len(selected),
        "core_component_count": len(core),
        "boundary_margin_px": round(boundary_margin, 3),
        "safe_pixels": int(safe_area),
        "white_ratio": float(np.mean(vals > 220)),
        "gray_std": float(np.std(vals)),
    }
    if selected:
        diag.update({
            "median_component_w": float(np.median([r["w"] for r in selected])),
            "median_component_h": float(np.median([r["h"] for r in selected])),
            "median_component_area": float(np.median([r["area"] for r in selected])),
            "median_boundary_distance_px": float(np.median([r["boundary_distance_px"] for r in selected])),
        })
    return selected, kept, diag


def _kmeans_diagonal_split(rows: list[dict]) -> tuple[list[list[int]], dict]:
    """Split a compound manga balloon only when two diagonal text islands are clear.

    Normal vertical text has several columns (large X separation but similar Y
    centre); normal horizontal text has several rows (large Y separation but
    similar X centre).  A compound balloon such as two attached lobes tends to
    have *both* X and Y centre separation.  This gate avoids turning ordinary
    columns into independent translations.
    """
    if len(rows) < 10:
        return [list(range(len(rows)))], {"split": False, "reason": "too_few_components"}
    med_w = max(1.0, float(np.median([r["w"] for r in rows])))
    med_h = max(1.0, float(np.median([r["h"] for r in rows])))
    pts = np.asarray([[r["cx"] / med_w, r["cy"] / med_h] for r in rows], dtype=np.float32)
    mean = pts.mean(axis=0)
    sse1 = float(np.sum((pts - mean) ** 2))
    if sse1 <= 1e-6:
        return [list(range(len(rows)))], {"split": False, "reason": "zero_spread"}
    compactness, labels, centers = cv2.kmeans(
        pts, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4),
        12, cv2.KMEANS_PP_CENTERS,
    )
    lab = labels.reshape(-1)
    groups = [np.flatnonzero(lab == k).tolist() for k in range(2)]
    improvement = float(1.0 - float(compactness) / sse1)
    dx = float(abs(centers[0, 0] - centers[1, 0]))
    dy = float(abs(centers[0, 1] - centers[1, 1]))
    group_boxes = []
    for group in groups:
        gx0 = min(rows[i]["x"] for i in group); gy0 = min(rows[i]["y"] for i in group)
        gx1 = max(rows[i]["x"] + rows[i]["w"] for i in group); gy1 = max(rows[i]["y"] + rows[i]["h"] for i in group)
        group_boxes.append((gx0, gy0, gx1, gy1))
    a, b = group_boxes
    gap_x = float(max(0, max(a[0], b[0]) - min(a[2], b[2])))
    gap_y = float(max(0, max(a[1], b[1]) - min(a[3], b[3])))
    gap_x_glyphs = gap_x / med_w
    gap_y_glyphs = gap_y / med_h
    accepted = (
        improvement >= 0.58
        and dx >= 3.2
        and dy >= 3.2
        and min(len(groups[0]), len(groups[1])) >= 4
        # A real two-flow compound balloon has a visible whitespace corridor.
        # Curved/diagonal normal columns can separate their centres in both axes
        # without being independent text islands; do not split those.
        and max(gap_x_glyphs, gap_y_glyphs) >= 1.8
    )
    diag = {
        "split": bool(accepted), "kmeans_improvement": improvement,
        "center_dx_glyphs": dx, "center_dy_glyphs": dy,
        "whitespace_gap_x_glyphs": round(gap_x_glyphs, 3),
        "whitespace_gap_y_glyphs": round(gap_y_glyphs, 3),
        "cluster_sizes": [len(groups[0]), len(groups[1])],
    }
    return groups if accepted else [list(range(len(rows)))], diag


def _koharu_neck_split(
    rows: list[dict], protected: np.ndarray
) -> tuple[list[list[int]], dict, FlowCellPartition | None]:
    """Recover attached-lobe text flows only when bubble topology proves a neck.

    The legacy splitter intentionally requires diagonal separation so normal manga
    columns/rows are not mistaken for independent dialogue.  With the optional
    Koharu-inspired path enabled we may consider side-by-side or stacked islands,
    but accept them only when a raster neck cut physically separates the trusted
    bubble mask.
    """
    if len(rows) < 6:
        return [list(range(len(rows)))], {"split": False, "reason": "too_few_components_for_neck"}, None
    med_w = max(1.0, float(np.median([r["w"] for r in rows])))
    med_h = max(1.0, float(np.median([r["h"] for r in rows])))
    pts = np.asarray([[r["cx"] / med_w, r["cy"] / med_h] for r in rows], dtype=np.float32)
    mean = pts.mean(axis=0)
    sse1 = float(np.sum((pts - mean) ** 2))
    if sse1 <= 1e-6:
        return [list(range(len(rows)))], {"split": False, "reason": "zero_spread"}, None
    compactness, labels, centers = cv2.kmeans(
        pts, 2, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-4),
        12, cv2.KMEANS_PP_CENTERS,
    )
    lab = labels.reshape(-1)
    groups = [np.flatnonzero(lab == k).tolist() for k in range(2)]
    improvement = float(1.0 - float(compactness) / sse1)
    center_distance = float(np.linalg.norm(centers[0] - centers[1]))
    diag = {
        "split": False,
        "reason": "candidate_rejected",
        "kmeans_improvement": improvement,
        "center_distance_glyphs": center_distance,
        "cluster_sizes": [len(groups[0]), len(groups[1])],
    }
    if improvement < 0.46 or center_distance < 3.0 or min(len(groups[0]), len(groups[1])) < 3:
        return [list(range(len(rows)))], diag, None

    anchors = [
        (float(np.mean([rows[i]["cx"] for i in group])), float(np.mean([rows[i]["cy"] for i in group])))
        for group in groups
    ]
    flow = partition_flow_cells(protected, anchors, prefer_neck=True)
    diag["flow_route"] = flow.route
    diag["flow_diagnostics"] = dict(flow.diagnostics or {})
    if flow.route != "neck_cut" or len(flow.masks) != 2:
        diag["reason"] = "no_physical_neck"
        return [list(range(len(rows)))], diag, None

    # The topology split must agree with the proposed text clusters.  This extra
    # gate prevents a random narrow tail or bubble border from validating a bad
    # k-means split.
    purities = []
    for group_index, group in enumerate(groups):
        cell = flow.masks[group_index]
        inside = 0
        for row_index in group:
            x = int(round(rows[row_index]["cx"])); y = int(round(rows[row_index]["cy"]))
            if 0 <= y < cell.shape[0] and 0 <= x < cell.shape[1] and cell[y, x] > 0:
                inside += 1
        purities.append(float(inside / max(1, len(group))))
    diag["cluster_cell_purity"] = [round(v, 4) for v in purities]
    if min(purities) < 0.86:
        diag["reason"] = "cluster_topology_disagree"
        return [list(range(len(rows)))], diag, None
    diag["split"] = True
    diag["reason"] = "physical_neck_confirmed"
    return groups, diag, flow


def _estimate_region_grid(rows: list[dict], indices: Iterable[int], orientation: str) -> dict:
    selected = [rows[i] for i in indices]
    if not selected:
        return {}
    med_w = max(1.0, float(np.median([r["w"] for r in selected])))
    med_h = max(1.0, float(np.median([r["h"] for r in selected])))
    xs = np.asarray([r["cx"] for r in selected], dtype=np.float32)
    ys = np.asarray([r["cy"] for r in selected], dtype=np.float32)

    def cluster_axis(values: np.ndarray, threshold: float) -> list[float]:
        vals = sorted(float(v) for v in values)
        if not vals:
            return []
        groups: list[list[float]] = [[vals[0]]]
        for v in vals[1:]:
            if v - float(np.mean(groups[-1])) <= threshold:
                groups[-1].append(v)
            else:
                groups.append([v])
        return [float(np.mean(g)) for g in groups]

    if orientation == "vertical":
        col_centers = cluster_axis(xs, max(4.0, med_w * 1.55))
        row_centers = cluster_axis(ys, max(4.0, med_h * 1.35))
        columns = max(1, len(col_centers))
        rows_n = max(1, int(round(len(selected) / columns)))
        if len(row_centers) > 1:
            pitch = float(np.median(np.diff(sorted(row_centers))))
        else:
            pitch = med_h * 1.25
    else:
        row_centers = cluster_axis(ys, max(4.0, med_h * 1.55))
        col_centers = cluster_axis(xs, max(4.0, med_w * 1.35))
        rows_n = max(1, len(row_centers))
        columns = max(1, int(round(len(selected) / rows_n)))
        if len(col_centers) > 1:
            pitch = float(np.median(np.diff(sorted(col_centers))))
        else:
            pitch = med_w * 1.25
    return {
        "estimated_columns": int(np.clip(columns, 1, 10)),
        "estimated_rows": int(np.clip(rows_n, 1, 32)),
        "target_glyph_pitch_px": round(max(4.0, pitch), 3),
        "median_component_w": round(med_w, 3),
        "median_component_h": round(med_h, 3),
    }


def _bbox_for_rows(rows: list[dict], indices: Iterable[int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    selected = [rows[i] for i in indices]
    h, w = shape
    x0 = min(r["x"] for r in selected); y0 = min(r["y"] for r in selected)
    x1 = max(r["x"] + r["w"] for r in selected); y1 = max(r["y"] + r["h"] for r in selected)
    med_w = max(2.0, float(np.median([r["w"] for r in selected])))
    med_h = max(2.0, float(np.median([r["h"] for r in selected])))
    pad_x = max(3, int(round(med_w * 0.85)))
    pad_y = max(3, int(round(med_h * 0.65)))
    return (
        max(0, x0 - pad_x), max(0, y0 - pad_y),
        min(w, x1 + pad_x), min(h, y1 + pad_y),
    )


def detect_target_text_regions(
    image: np.ndarray, bubble: BubbleInstance, *, koharu_flow_cells: bool = False
) -> list[TargetTextRegion]:
    """Detect target Japanese text islands inside one already-paired bubble.

    This is intentionally reletter-only geometry. It never participates in Direct,
    Mask Replace, Hybrid, or Auto raster transfer.
    """
    h, w = image.shape[:2]
    rows, kept, diag = _component_rows(image, bubble)
    if not rows:
        return []
    med_w = float(diag.get("median_component_w", 0.0))
    med_h = float(diag.get("median_component_h", 0.0))
    med_area = float(diag.get("median_component_area", 0.0))
    # Structural/photo difference can occasionally return an arm edge or panel
    # artifact as a pseudo balloon. Real printed glyphs at this page scale have a
    # meaningful two-dimensional component size; the false arm region in the
    # regression page is only 4x2px median.
    # These statistics are now computed from interior glyph candidates rather
    # than balloon-edge noise. Keep the gate permissive enough for short sound
    # effects / two-glyph balloons while still rejecting single stray strokes.
    if len(rows) < 2 or med_w < 2.5 or med_h < 2.5 or med_area < 4.0:
        return []

    groups, split_diag = _kmeans_diagonal_split(rows)
    protected = _protected_mask(bubble, (h, w))
    assert protected is not None
    flow_partition: FlowCellPartition | None = None
    if koharu_flow_cells:
        if len(groups) > 1:
            anchors = [
                (float(np.mean([rows[i]["cx"] for i in group])), float(np.mean([rows[i]["cy"] for i in group])))
                for group in groups
            ]
            flow_partition = partition_flow_cells(protected, anchors, prefer_neck=True)
            split_diag = {
                **split_diag,
                "koharu_flow_cells": True,
                "flow_route": flow_partition.route,
                "flow_diagnostics": dict(flow_partition.diagnostics or {}),
            }
        else:
            groups, neck_diag, flow_partition = _koharu_neck_split(rows, protected)
            split_diag = {**split_diag, "koharu_flow_cells": True, "koharu_neck": neck_diag}
    out: list[TargetTextRegion] = []
    for gi, group in enumerate(groups):
        if not group:
            continue
        bbox = _bbox_for_rows(rows, group, (h, w))
        x0, y0, x1, y1 = bbox
        region_rect = np.zeros((h, w), np.uint8)
        region_rect[y0:y1, x0:x1] = 255
        region_rect = cv2.bitwise_and(region_rect, protected)
        flow_route = "disabled"
        if flow_partition is not None and gi < len(flow_partition.masks):
            cell = flow_partition.masks[gi]
            if cell.shape[:2] == region_rect.shape[:2] and cv2.countNonZero(cell) > 0:
                region_rect = cv2.bitwise_and(region_rect, cell)
                flow_route = flow_partition.route
        # Reconstruct only the exact connected glyph components belonging to this
        # subregion. Do not pull every dark pixel inside the expanded region bbox:
        # that is how balloon borders leaked into v2.3.87/88 Reletter masks.
        region_ink = np.zeros((h, w), np.uint8)
        for ri in group:
            row = rows[ri]
            rx, ry, rwc, rhc = int(row["x"]), int(row["y"]), int(row["w"]), int(row["h"])
            roi = region_ink[ry:ry + rhc, rx:rx + rwc]
            region_ink[ry:ry + rhc, rx:rx + rwc] = cv2.bitwise_or(roi, row["_mask"])
        region_ink = cv2.bitwise_and(region_ink, region_rect)
        if cv2.countNonZero(region_ink) == 0:
            continue
        region_ink = cv2.dilate(region_ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        region_ink = cv2.bitwise_and(region_ink, region_rect)
        rw, rh = max(1, x1 - x0), max(1, y1 - y0)
        orientation = "vertical" if rh >= rw * 0.82 else "horizontal"
        grid_diag = _estimate_region_grid(rows, group, orientation)
        poly = [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]
        if flow_route != "disabled":
            contours, _ = cv2.findContours(region_rect, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                epsilon = max(1.0, 0.006 * cv2.arcLength(contour, True))
                approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
                if len(approx) >= 3:
                    poly = [(float(x), float(y)) for x, y in approx]
        confidence = float(np.clip(0.72 + min(0.2, len(group) / 80.0), 0.0, 0.96))
        component_bboxes = [
            [int(rows[i]["x"]), int(rows[i]["y"]), int(rows[i]["x"] + rows[i]["w"]), int(rows[i]["y"] + rows[i]["h"])]
            for i in group
        ]
        out.append(TargetTextRegion(
            id=f"{bubble.id}-region-{gi:02d}",
            target_bubble_id=bubble.id,
            bbox=bbox,
            polygon=poly,
            text_mask=region_ink,
            orientation=orientation,
            component_count=len(group),
            confidence=confidence,
            diagnostics={
                **diag, **split_diag, **grid_diag, "group_index": gi, "bbox": list(bbox),
                "component_bboxes": component_bboxes,
                "glyph_component_bboxes": component_bboxes,
                "component_area_sum": int(sum(int(rows[i]["area"]) for i in group)),
                "text_mask_pixels": int(cv2.countNonZero(region_ink)),
                "flow_cell_route": flow_route,
                "flow_cell_enabled": bool(koharu_flow_cells),
            },
        ))

    # Manga vertical reading order is right-to-left; for diagonal compound lobes,
    # right/top region should be recognized before left/bottom region. Horizontal
    # regions retain top-to-bottom order.
    out.sort(key=lambda r: ((r.bbox[1] + r.bbox[3]) * 0.20, -(r.bbox[0] + r.bbox[2])))
    for i, region in enumerate(out):
        region.id = f"{bubble.id}-region-{i:02d}"
    return out


def normalized_map_bbox(
    target_bbox: tuple[int, int, int, int],
    target_bubble_bbox: tuple[float, float, float, float],
    source_bubble_bbox: tuple[float, float, float, float],
    source_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map one target text island into its paired source bubble by normalized position."""
    tx0, ty0, tx1, ty1 = [float(v) for v in target_bubble_bbox]
    sx0, sy0, sx1, sy1 = [float(v) for v in source_bubble_bbox]
    bx0, by0, bx1, by1 = [float(v) for v in target_bbox]
    tw = max(1.0, tx1 - tx0); th = max(1.0, ty1 - ty0)
    sw = max(1.0, sx1 - sx0); sh = max(1.0, sy1 - sy0)
    rx0 = np.clip((bx0 - tx0) / tw, 0.0, 1.0); ry0 = np.clip((by0 - ty0) / th, 0.0, 1.0)
    rx1 = np.clip((bx1 - tx0) / tw, 0.0, 1.0); ry1 = np.clip((by1 - ty0) / th, 0.0, 1.0)
    x0 = sx0 + float(rx0) * sw; y0 = sy0 + float(ry0) * sh
    x1 = sx0 + float(rx1) * sw; y1 = sy0 + float(ry1) * sh
    h, w = source_shape
    pad = max(4, int(round(min(max(1.0, x1 - x0), max(1.0, y1 - y0)) * 0.08)))
    return (
        max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad),
        min(w, int(np.ceil(x1)) + pad), min(h, int(np.ceil(y1)) + pad),
    )
