from __future__ import annotations

"""External OCR result adapter.

Supports structured PaddleOCR-VL/PP-Structure JSON exports and simple Markdown
transcripts.  The structured JSON route is authoritative because it preserves
page-local polygons/bboxes and reading order.  Markdown is accepted as a
companion/fallback, but a multi-page Markdown file without page markers cannot
safely reconstruct coordinates and is therefore treated as transcript-only.
"""

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .geometry import bbox_polygon
from .models import TextBlock

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_DEFAULT_IGNORED = {
    "number", "footnote", "header", "header_image", "footer", "footer_image",
    "aside_text", "image", "figure", "chart", "formula", "table", "seal",
}


def _natural_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.lower())
    return tuple(int(x) if x.isdigit() else x for x in parts)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("text") or value.get("content") or value.get("markdown") or ""
    text = str(value).replace("\r\n", "\n").replace("\r", "\n")
    # Paddle document-parsing exports often insert blank lines inside one
    # vertical block. Keep real line breaks but collapse empty spacer lines.
    lines = [line.strip() for line in text.split("\n")]
    compact: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if compact and not previous_blank:
                compact.append("")
            previous_blank = True
            continue
        compact.append(line)
        previous_blank = False
    return "\n".join(compact).strip()


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _page_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        pages = payload.get("pages")
        if isinstance(pages, list):
            return [x for x in pages if isinstance(x, dict)]
        return [payload]
    return []


def _polygon_from_row(row: dict[str, Any], image_shape: tuple[int, int]) -> list[tuple[float, float]]:
    poly = row.get("block_polygon_points") or row.get("polygon_points") or row.get("polygon")
    if isinstance(poly, list) and len(poly) >= 3:
        try:
            return [(float(p[0]), float(p[1])) for p in poly]
        except Exception:
            pass
    bbox = row.get("block_bbox") or row.get("bbox") or row.get("coordinate")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        try:
            return [(float(x), float(y)) for x, y in bbox_polygon(tuple(float(v) for v in bbox[:4]))]
        except Exception:
            pass
    h, w = image_shape
    return [(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))]


def _layout_score_map(page: dict[str, Any]) -> dict[tuple[int, int, int, int], float]:
    boxes = _as_dict(_as_dict(page.get("prunedResult")).get("layout_det_res")).get("boxes")
    if not isinstance(boxes, list):
        boxes = _as_dict(page.get("layout_det_res")).get("boxes")
    out: dict[tuple[int, int, int, int], float] = {}
    for row in boxes or []:
        if not isinstance(row, dict):
            continue
        coord = row.get("coordinate") or row.get("bbox")
        if not isinstance(coord, (list, tuple)) or len(coord) < 4:
            continue
        try:
            key = tuple(int(round(float(v))) for v in coord[:4])
            out[key] = float(row.get("score", 0.95))
        except Exception:
            continue
    return out


def _structured_rows(page: dict[str, Any]) -> list[dict[str, Any]]:
    pruned = _as_dict(page.get("prunedResult"))
    rows = pruned.get("parsing_res_list") or page.get("parsing_res_list")
    if isinstance(rows, list):
        return [x for x in rows if isinstance(x, dict)]
    # PP-Structure/VL variants may nest the parsing payload one level deeper.
    for value in page.values():
        if isinstance(value, dict):
            rows = value.get("parsing_res_list")
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
    return []


def _normalise_md_stem(stem: str) -> str:
    return re.sub(r"\s*\(\d+\)$", "", stem).strip().lower()


def find_json_companion(md_path: Path) -> Path | None:
    direct = md_path.with_suffix(".json")
    if direct.is_file():
        return direct
    wanted = _normalise_md_stem(md_path.stem)
    candidates = []
    for path in md_path.parent.glob("*.json"):
        stem = _normalise_md_stem(path.stem)
        if stem == wanted or stem.startswith(wanted) or wanted.startswith(stem):
            candidates.append(path)
    return sorted(candidates, key=lambda p: (abs(len(p.stem) - len(md_path.stem)), p.name))[0] if candidates else None


@dataclass(slots=True)
class ExternalOCRDocument:
    path: Path
    pages: list[dict[str, Any]]
    format: str
    companion_json: Path | None = None
    reference_size: tuple[float, float] | None = None  # (width, height) of external coordinate canvas

    def __post_init__(self) -> None:
        if self.reference_size is None:
            self.reference_size = self._estimate_reference_size()

    def _estimate_reference_size(self) -> tuple[float, float] | None:
        """Infer the external export canvas from geometry across all pages.

        Paddle online/VL JSON does not always serialize image width/height. Across
        a whole book, however, layout polygons normally reach close to page edges.
        Using the global maximum and rounding to the nearest 10 px reproduces the
        export canvas for the supplied VL 1.6 sample (1190×1870) while avoiding
        per-page crop drift. The scale is only used when the local image size is
        materially different.
        """
        max_x=max_y=0.0
        for page in self.pages:
            for row in _structured_rows(page):
                poly=row.get("block_polygon_points") or row.get("polygon_points") or row.get("polygon")
                if isinstance(poly,list):
                    for pt in poly:
                        try: max_x=max(max_x,float(pt[0])); max_y=max(max_y,float(pt[1]))
                        except Exception: pass
                bbox=row.get("block_bbox") or row.get("bbox") or row.get("coordinate")
                if isinstance(bbox,(list,tuple)) and len(bbox)>=4:
                    try: max_x=max(max_x,float(bbox[2])); max_y=max(max_y,float(bbox[3]))
                    except Exception: pass
        if max_x < 64 or max_y < 64:
            return None
        width=max(1.0, round(max_x/10.0)*10.0)
        height=max(1.0, round(max_y/10.0)*10.0)
        return (width,height)

    @classmethod
    def load(cls, path: str | Path) -> "ExternalOCRDocument":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"外部 OCR 文件不存在：{p}")
        suffix = p.suffix.lower()
        if suffix == ".json":
            payload = json.loads(p.read_text(encoding="utf-8-sig"))
            pages = _page_list(payload)
            if not pages:
                raise ValueError("外部 OCR JSON 不包含可识别页面。")
            return cls(p, pages, "json", None)
        if suffix in {".md", ".markdown"}:
            companion = find_json_companion(p)
            if companion is not None:
                payload = json.loads(companion.read_text(encoding="utf-8-sig"))
                pages = _page_list(payload)
                if pages:
                    return cls(p, pages, "md+json", companion)
            # Standalone Markdown cannot reliably infer multi-page geometry.
            text = p.read_text(encoding="utf-8-sig")
            return cls(p, [{"markdown": {"text": text, "images": {}}}], "markdown", None)
        raise ValueError("外部 OCR 仅支持 .json / .md / .markdown。")

    def page_count(self) -> int:
        return len(self.pages)

    def blocks_for_page(
        self,
        page_index: int,
        image_shape: tuple[int, int],
        *,
        ignored_labels: Iterable[str] = _DEFAULT_IGNORED,
    ) -> list[TextBlock]:
        if page_index < 0 or page_index >= len(self.pages):
            return []
        page = self.pages[page_index]
        ignored = {str(x).strip().lower() for x in ignored_labels}
        scores = _layout_score_map(page)
        rows = _structured_rows(page)
        blocks: list[TextBlock] = []
        if rows:
            for i, row in enumerate(rows):
                label = str(row.get("block_label") or row.get("label") or row.get("type") or "text").strip().lower()
                if label in ignored:
                    continue
                text = _clean_text(row.get("block_content") or row.get("text") or row.get("content"))
                if not text:
                    continue
                poly = _polygon_from_row(row, image_shape)
                scale_x=scale_y=1.0
                if self.reference_size is not None:
                    ref_w,ref_h=self.reference_size
                    local_h,local_w=image_shape
                    if ref_w > 0 and ref_h > 0:
                        raw_sx=float(local_w)/float(ref_w); raw_sy=float(local_h)/float(ref_h)
                        # Only auto-scale when the observed document-wide geometry
                        # plausibly spans the page. A one-block JSON may have no
                        # edge evidence, in which case raw coordinates are safer.
                        has_canvas_evidence = ref_w >= local_w * 0.55 and ref_h >= local_h * 0.55
                        if has_canvas_evidence and (abs(raw_sx-1.0) > 0.025 or abs(raw_sy-1.0) > 0.025):
                            scale_x,scale_y=raw_sx,raw_sy
                            poly=[(float(x)*scale_x,float(y)*scale_y) for x,y in poly]
                bbox = row.get("block_bbox") or row.get("bbox") or row.get("coordinate")
                score = 0.95
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    try:
                        score = scores.get(tuple(int(round(float(v))) for v in bbox[:4]), score)
                    except Exception:
                        pass
                order = row.get("block_order")
                try:
                    order_i = int(order) if order is not None else i
                except Exception:
                    order_i = i
                orientation = "vertical" if "vertical" in label else ("horizontal" if "text" in label else "auto")
                blocks.append(TextBlock(
                    id=f"external-{page_index:04d}-{i:04d}",
                    polygon=poly,
                    text=text,
                    confidence=float(score),
                    kind=label or "text",
                    reading_order=order_i,
                    meta={
                        "backend": "external_ocr",
                        "external_format": self.format,
                        "external_file": str(self.path),
                        "external_companion_json": str(self.companion_json) if self.companion_json else "",
                        "external_page_index": page_index,
                        "block_label": label,
                        "block_id": row.get("block_id"),
                        "group_id": row.get("group_id"),
                        "orientation_hint": orientation,
                        "coordinates_authoritative": True,
                        "external_reference_size": list(self.reference_size) if self.reference_size else None,
                        "external_coordinate_scale": [scale_x, scale_y],
                    },
                ))
            blocks.sort(key=lambda b: (b.reading_order, b.id))
            for i, block in enumerate(blocks):
                block.reading_order = i
            return blocks

        markdown = _as_dict(page.get("markdown"))
        text = _clean_text(markdown.get("text") or page.get("text") or page.get("content"))
        if not text:
            return []
        h, w = image_shape
        return [TextBlock(
            id=f"external-{page_index:04d}-transcript",
            polygon=[(0.0, 0.0), (float(w), 0.0), (float(w), float(h)), (0.0, float(h))],
            text=text,
            confidence=0.85,
            kind="transcript",
            reading_order=0,
            meta={
                "backend": "external_ocr",
                "external_format": self.format,
                "external_file": str(self.path),
                "external_page_index": page_index,
                "transcript_only": True,
                "coordinates_authoritative": False,
            },
        )]


class ExternalOCRBackend:
    """Use a whole-book external OCR export as an OCRBackend-compatible source."""

    supports_crop_recognition = False
    supports_region_query = True
    supports_rectified_input = False

    def __init__(self, path: str | Path, *, start_page: int = 1, ignored_labels: Iterable[str] = _DEFAULT_IGNORED) -> None:
        self.document = ExternalOCRDocument.load(path)
        self.start_page = max(1, int(start_page))
        self.ignored_labels = tuple(ignored_labels)
        self._page_index_cache: dict[str, int] = {}

    def _local_ordinal(self, image_path: str | Path) -> int:
        p = Path(image_path).resolve()
        key = str(p)
        if key in self._page_index_cache:
            return self._page_index_cache[key]
        siblings = sorted(
            [x.resolve() for x in p.parent.iterdir() if x.is_file() and x.suffix.lower() in _IMAGE_SUFFIXES],
            key=_natural_key,
        )
        try:
            ordinal = siblings.index(p) + 1
        except ValueError:
            ordinal = 1
        self._page_index_cache[key] = ordinal
        return ordinal

    def external_page_index(self, image_path: str | Path) -> int:
        return self._local_ordinal(image_path) - self.start_page

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if image_path is None:
            raise ValueError("ExternalOCRBackend 需要原始 image_path 以确定外部结果页码。")
        idx = self.external_page_index(image_path)
        return self.document.blocks_for_page(idx, image.shape[:2], ignored_labels=self.ignored_labels)

    @staticmethod
    def _overlap_ratio(block: TextBlock, bbox: tuple[int, int, int, int]) -> float:
        bx0, by0, bx1, by1 = block.bbox
        x0, y0, x1, y1 = [float(v) for v in bbox]
        ix0, iy0 = max(bx0, x0), max(by0, y0)
        ix1, iy1 = min(bx1, x1), min(by1, y1)
        inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
        area = max(1.0, (bx1 - bx0) * (by1 - by0))
        return inter / area

    def recognize_region(
        self,
        page_image: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        image_path: str | Path | None = None,
        min_overlap: float = 0.30,
    ) -> list[TextBlock]:
        if image_path is None:
            raise ValueError("ExternalOCRBackend region query 需要原始 image_path。")
        all_blocks = self.recognize(page_image, image_path=image_path)
        selected = []
        x0, y0, x1, y1 = [float(v) for v in bbox]
        for block in all_blocks:
            cx, cy = block.centroid
            center_inside = x0 <= cx <= x1 and y0 <= cy <= y1
            if center_inside or self._overlap_ratio(block, bbox) >= min_overlap:
                selected.append(block)
        selected.sort(key=lambda b: b.reading_order)
        return selected


__all__ = ["ExternalOCRDocument", "ExternalOCRBackend", "find_json_companion"]
