from __future__ import annotations

import cv2
import numpy as np

from .schema import SemanticBlock, SemanticLayoutResult
from .router import route_blocks


def _poly(box: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x0, y0, x1, y1 = box
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _confidence(count: int, bbox: tuple[int, int, int, int]) -> float:
    x0, y0, x1, y1 = bbox
    area = max(1, (x1 - x0) * (y1 - y0))
    density = min(1.0, count / 8.0)
    compact = min(1.0, 18000.0 / area)
    return float(np.clip(0.42 + 0.34 * density + 0.18 * compact, 0.35, 0.94))


def analyze_heuristic(image: np.ndarray, cfg, *, strategy: str = "auto") -> SemanticLayoutResult:
    from ..source_detectors import _compact_character_components, _cluster_text_components

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    comps = _compact_character_components(gray)
    groups = _cluster_text_components(gray, comps, min_components=max(1, int(getattr(cfg, "heuristic_min_components", 2))))
    blocks: list[SemanticBlock] = []
    top_ratio = float(getattr(cfg, "header_top_ratio", 0.14))
    bottom_ratio = float(getattr(cfg, "footer_bottom_ratio", 0.94))
    for idx, group in enumerate(groups[: int(getattr(cfg, "max_blocks", 128))]):
        x0, y0, x1, y1 = [int(v) for v in group["bbox"]]
        bw, bh = max(1, x1 - x0), max(1, y1 - y0)
        count = int(group.get("count", 0))
        inside = [c for c in comps if x0 <= c[5][0] < x1 and y0 <= c[5][1] < y1]
        widths = [c[2] for c in inside] or [max(1, bw)]
        heights = [c[3] for c in inside] or [max(1, bh)]
        centers_x = [c[5][0] for c in inside]
        centers_y = [c[5][1] for c in inside]
        mw = float(np.median(widths)); mh = float(np.median(heights))
        x_span = float(max(centers_x) - min(centers_x)) if len(centers_x) >= 2 else 0.0
        y_span = float(max(centers_y) - min(centers_y)) if len(centers_y) >= 2 else 0.0
        vertical = bh >= bw * 1.12
        label = "vertical_text" if vertical else "text"
        # Small groups must look like an actual line/column.  This rejects face,
        # sweat, nipples and anatomy strokes that the generic dark-component
        # clusterer can otherwise mistake for manga text. Large multi-glyph text
        # blocks are allowed more spatial freedom because they may contain several
        # CJK columns.
        if count < 4:
            label = "unknown"
        elif count <= 5:
            coherent = (x_span <= max(30.0, mw * 3.2)) if vertical else (y_span <= max(26.0, mh * 3.0))
            if (not coherent) or min(mw, mh) < 18.0:
                label = "unknown"
        elif count < 10:
            coherent = (x_span <= max(30.0, mw * 3.2)) if vertical else (y_span <= max(26.0, mh * 3.0))
            if not coherent:
                label = "unknown"
        # Semantic page furniture classification is deliberately stricter than a
        # plain y-coordinate rule: header/footer must also be shallow and compact.
        if y1 <= int(round(h * top_ratio)) and bh <= max(52, int(round(h * 0.06))) and bw <= int(round(w * 0.48)):
            label = "header"
        elif y0 >= int(round(h * bottom_ratio)) and bh <= max(42, int(round(h * 0.04))):
            label = "number" if bw <= max(72, int(round(w * 0.12))) else "footer"
        elif count <= 1 and bw <= 42 and bh <= 42:
            label = "unknown"
        conf = _confidence(count, (x0, y0, x1, y1))
        blocks.append(SemanticBlock(
            id=f"heur-layout-{idx:04d}", source="heuristic_manga_layout", raw_label=label,
            semantic_type="unknown", confidence=conf, bbox=(x0, y0, x1, y1), polygon=_poly((x0, y0, x1, y1)),
            reading_order=idx, meta={
                "component_count": count, "dilated_bbox": list(group.get("dilated_bbox", (x0, y0, x1, y1))),
                "median_component_width": mw, "median_component_height": mh,
                "component_x_span": x_span, "component_y_span": y_span,
            },
        ))
    route_blocks(blocks, strategy)
    return SemanticLayoutResult(True, "heuristic_manga_layout", blocks, diagnostics={
        "status": "ok", "component_count": len(comps), "group_count": len(groups), "block_count": len(blocks),
        "note": "fallback semantic layout; PP-DocLayoutV3 runtime unavailable",
    })
