from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import BubbleConfig
from .geometry import bbox_polygon, mask_to_largest_polygon, polygon_bbox, polygon_centroid, rasterize_polygon, union_bbox
from .models import BubbleInstance, TextBlock, TextUnit

_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]")


def _nearest_white_seed(mask: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int] | None:
    h, w = mask.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    if mask[y, x] > 0:
        return x, y
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    ys, xs = np.where(mask[y0:y1, x0:x1] > 0)
    if len(xs) == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    dist2 = (xs - x) ** 2 + (ys - y) ** 2
    idx = int(np.argmin(dist2))
    return int(xs[idx]), int(ys[idx])


def _safe_mask(component: np.ndarray, seed: tuple[int, int], margin: int) -> np.ndarray:
    binary = (component > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    safe = (dist >= max(1, margin)).astype(np.uint8) * 255
    if not np.any(safe):
        # Relax instead of returning an unusable region.
        safe = (dist >= max(1, margin // 2)).astype(np.uint8) * 255
    if not np.any(safe):
        return component.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats((safe > 0).astype(np.uint8), 8)
    sx, sy = seed
    wanted = labels[int(np.clip(sy, 0, labels.shape[0] - 1)), int(np.clip(sx, 0, labels.shape[1] - 1))]
    if wanted == 0:
        if count <= 1:
            return safe
        wanted = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == wanted).astype(np.uint8) * 255


def detect_seeded_white_bubbles(
    image: np.ndarray, blocks: list[TextBlock], config: BubbleConfig | None = None
) -> list[BubbleInstance]:
    cfg = config or BubbleConfig()
    if not blocks:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # White-ish interior. A mild close fills tiny raster/compression gaps without deleting borders.
    white = (gray >= cfg.white_threshold).astype(np.uint8) * 255
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    h, w = gray.shape
    page_area = h * w
    label_to_bubble: dict[int, BubbleInstance] = {}

    for block in blocks:
        cx, cy = block.centroid
        seed = _nearest_white_seed(white, round(cx), round(cy), cfg.search_radius)
        if seed is None:
            continue
        sx, sy = seed
        label = int(labels[sy, sx])
        if label <= 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        ratio = area / max(1, page_area)
        if ratio < cfg.min_area_ratio or ratio > cfg.max_area_ratio:
            continue
        if label not in label_to_bubble:
            raw_component = (labels == label).astype(np.uint8) * 255
            # Text glyphs are dark holes inside an otherwise white bubble. For layout
            # geometry they must be considered usable interior after clearing, so fill
            # the external contour instead of preserving glyph-shaped holes.
            contours, _ = cv2.findContours(raw_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            outer = max(contours, key=cv2.contourArea)
            component = np.zeros_like(raw_component)
            cv2.drawContours(component, [outer], -1, 255, thickness=cv2.FILLED)
            polygon = mask_to_largest_polygon(component)
            if len(polygon) < 3:
                continue
            x, y, bw, bh, _ = stats[label]
            margin = max(cfg.safe_margin_px, int(min(bw, bh) * cfg.safe_margin_ratio))
            safe = _safe_mask(component, seed, margin)
            contour_area = max(1.0, float(cv2.countNonZero(component)))
            rect_fill = contour_area / max(1.0, bw * bh)
            approx = cv2.approxPolyDP(
                max(cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea),
                0.02 * (bw + bh),
                True,
            )
            kind = "narration" if len(approx) <= 5 and rect_fill > 0.82 else "speech"
            label_to_bubble[label] = BubbleInstance(
                id=f"bubble-{len(label_to_bubble):04d}",
                polygon=polygon,
                confidence=float(np.clip(0.55 + 0.35 * min(1.0, rect_fill), 0.0, 0.95)),
                kind=kind,
                mask=component,
                safe_mask=safe,
                meta={"area_ratio": ratio, "rect_fill": rect_fill, "seed_label": label, "safe_margin": margin},
            )
        bubble = label_to_bubble[label]
        bubble.block_ids.append(block.id)
        block.bubble_id = bubble.id
        if block.kind == "unknown":
            block.kind = bubble.kind

    return list(label_to_bubble.values())


def load_bubble_sidecar(
    image: np.ndarray,
    image_path: str | Path,
    blocks: list[TextBlock],
    config: BubbleConfig | None = None,
) -> list[BubbleInstance]:
    cfg = config or BubbleConfig()
    p = Path(image_path)
    sidecar = p.with_suffix(cfg.sidecar_suffix) if cfg.sidecar_suffix.startswith(".") else p.parent / f"{p.stem}{cfg.sidecar_suffix}"
    if not sidecar.exists():
        raise FileNotFoundError(f"Bubble sidecar not found: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    rows = payload.get("bubbles", []) if isinstance(payload, dict) else payload
    h, w = image.shape[:2]
    out: list[BubbleInstance] = []
    for i, row in enumerate(rows):
        poly = row.get("polygon")
        mask = None
        mp = row.get("mask_path")
        if mp:
            path = Path(mp)
            if not path.is_absolute():
                path = sidecar.parent / path
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if mask is not None and not poly:
                poly = mask_to_largest_polygon(mask)
        if not poly:
            continue
        poly = [(float(x), float(y)) for x, y in poly]
        if mask is None:
            mask = rasterize_polygon(poly, (h, w))
        safe = None
        sp = row.get("safe_mask_path")
        if sp:
            path = Path(sp)
            if not path.is_absolute():
                path = sidecar.parent / path
            safe = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if safe is not None and safe.shape != (h, w):
                safe = cv2.resize(safe, (w, h), interpolation=cv2.INTER_NEAREST)
        if safe is None:
            x0, y0, x1, y1 = polygon_bbox(poly)
            margin = max(cfg.safe_margin_px, int(min(x1 - x0, y1 - y0) * cfg.safe_margin_ratio))
            # Use largest component after erosion as safe area; it naturally removes narrow tails.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            safe = cv2.erode((mask > 0).astype(np.uint8) * 255, k)
            if cv2.countNonZero(safe) == 0:
                safe = mask.copy()
        out.append(BubbleInstance(
            id=str(row.get("id") or f"bubble-{i:04d}"),
            polygon=poly,
            confidence=float(row.get("confidence", 1.0)),
            kind=str(row.get("kind", "speech")),
            block_ids=list(row.get("block_ids", [])),
            mask=(mask > 0).astype(np.uint8) * 255,
            safe_mask=(safe > 0).astype(np.uint8) * 255,
            meta=dict(row.get("meta", {})),
        ))
    assign_blocks_to_bubbles(blocks, out)
    return out


def assign_blocks_to_bubbles(blocks: list[TextBlock], bubbles: list[BubbleInstance]) -> None:
    lookup = {b.id: b for b in bubbles}
    for bubble in bubbles:
        bubble.block_ids = []
    for block in blocks:
        cx, cy = block.centroid
        candidates: list[tuple[float, BubbleInstance]] = []
        for bubble in bubbles:
            if bubble.mask is not None:
                y, x = round(cy), round(cx)
                if 0 <= y < bubble.mask.shape[0] and 0 <= x < bubble.mask.shape[1] and bubble.mask[y, x] > 0:
                    candidates.append((1.0, bubble))
            else:
                bx0, by0, bx1, by1 = bubble.bbox
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    candidates.append((0.5, bubble))
        if candidates:
            bubble = max(candidates, key=lambda x: x[0])[1]
            block.bubble_id = bubble.id
            if block.id not in bubble.block_ids:
                bubble.block_ids.append(block.id)
            if block.kind == "unknown":
                block.kind = bubble.kind
        elif block.bubble_id not in lookup:
            block.bubble_id = None
            if block.kind == "unknown":
                block.kind = "free_text"


def _join_fragments(fragments: list[str]) -> str:
    fragments = [f.strip() for f in fragments if f and f.strip()]
    if not fragments:
        return ""
    combined = "".join(fragments)
    cjk_ratio = len(_CJK_RE.findall(combined)) / max(1, len(combined))
    return "".join(fragments) if cjk_ratio >= 0.25 else " ".join(fragments)


def build_text_units(blocks: list[TextBlock], bubbles: list[BubbleInstance], prefix: str) -> list[TextUnit]:
    by_id = {b.id: b for b in blocks}
    units: list[TextUnit] = []
    used: set[str] = set()

    for bubble in bubbles:
        members = [by_id[i] for i in bubble.block_ids if i in by_id]
        if not members:
            continue
        members.sort(key=lambda x: x.reading_order)
        used.update(m.id for m in members)
        text = _join_fragments([m.text for m in members])
        conf = float(np.mean([m.confidence for m in members]))
        units.append(
            TextUnit(
                id=f"{prefix}-unit-{len(units):04d}",
                polygon=list(bubble.polygon),
                block_ids=[m.id for m in members],
                text=text,
                confidence=conf,
                kind=bubble.kind,
                reading_order=min(m.reading_order for m in members),
                bubble_id=bubble.id,
                meta={"geometry": "bubble"},
            )
        )

    for block in blocks:
        if block.id in used:
            continue
        units.append(
            TextUnit(
                id=f"{prefix}-unit-{len(units):04d}",
                polygon=list(block.polygon),
                block_ids=[block.id],
                text=block.text,
                confidence=block.confidence,
                kind=block.kind if block.kind != "unknown" else "free_text",
                reading_order=block.reading_order,
                bubble_id=None,
                meta={"geometry": "text_block"},
            )
        )

    units.sort(key=lambda u: u.reading_order)
    for i, unit in enumerate(units):
        unit.reading_order = i
    return units


def bubble_by_id(bubbles: list[BubbleInstance]) -> dict[str, BubbleInstance]:
    return {b.id: b for b in bubbles}
