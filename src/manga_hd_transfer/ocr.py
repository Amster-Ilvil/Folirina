from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import OCRConfig
from .geometry import bbox_polygon, polygon_centroid
from .models import TextBlock

logger = logging.getLogger(__name__)


class OCRBackend(ABC):
    @abstractmethod
    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        raise NotImplementedError


class NullOCRBackend(OCRBackend):
    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        return []


class SidecarOCRBackend(OCRBackend):
    """Reads deterministic OCR data from `<image>.ocr.json` or a configured sidecar.

    Accepted payloads:
      {"blocks": [{"polygon": [[x,y],...], "text": "...", "confidence": .99}]}
      [{...}, {...}]
    """

    def __init__(self, suffix: str = ".ocr.json") -> None:
        self.suffix = suffix

    def sidecar_path(self, image_path: str | Path) -> Path:
        p = Path(image_path)
        # page.png -> page.ocr.json
        if self.suffix.startswith("."):
            return p.with_suffix(self.suffix)
        return p.parent / f"{p.stem}{self.suffix}"

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if image_path is None:
            raise ValueError("SidecarOCRBackend requires image_path")
        sidecar = self.sidecar_path(image_path)
        if not sidecar.exists():
            raise FileNotFoundError(f"OCR sidecar not found: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        rows = payload.get("blocks", []) if isinstance(payload, dict) else payload
        blocks: list[TextBlock] = []
        for i, row in enumerate(rows):
            poly = row.get("polygon")
            if not poly and row.get("bbox"):
                poly = bbox_polygon(row["bbox"])
            if not poly:
                continue
            blocks.append(
                TextBlock(
                    id=str(row.get("id") or f"ocr-{i:04d}"),
                    polygon=[(float(x), float(y)) for x, y in poly],
                    text=str(row.get("text", "")),
                    confidence=float(row.get("confidence", 1.0)),
                    kind=str(row.get("kind", "unknown")),
                    reading_order=int(row.get("reading_order", i)),
                    bubble_id=row.get("bubble_id"),
                    meta=dict(row.get("meta", {})),
                )
            )
            mask_path = row.get("mask_path") or blocks[-1].meta.get("mask_path")
            if mask_path:
                mp = Path(mask_path)
                if not mp.is_absolute():
                    mp = sidecar.parent / mp
                blocks[-1].meta["mask_path"] = str(mp)
        return blocks


def _find_ocr_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if "rec_texts" in obj and ("rec_polys" in obj or "dt_polys" in obj or "rec_boxes" in obj):
            return obj
        for v in obj.values():
            found = _find_ocr_dict(v)
            if found is not None:
                return found
    return None


def _result_to_dict(res: Any) -> dict[str, Any]:
    candidates = []
    for attr in ("json", "res", "result"):
        if hasattr(res, attr):
            value = getattr(res, attr)
            try:
                value = value() if callable(value) else value
            except Exception:
                pass
            candidates.append(value)
    if hasattr(res, "to_dict"):
        try:
            candidates.append(res.to_dict())
        except Exception:
            pass
    candidates.append(res)
    for candidate in candidates:
        found = _find_ocr_dict(candidate)
        if found is not None:
            return found
    raise ValueError(f"Unsupported PaddleOCR result structure: {type(res)!r}")


class PaddleOCRBackend(OCRBackend):
    def __init__(self, lang: str, config: OCRConfig | None = None, device: str | None = None) -> None:
        self.config = config or OCRConfig()
        try:
            from paddleocr import PaddleOCR
        except Exception as e:  # pragma: no cover - optional dependency
            raise RuntimeError("PaddleOCR is not installed. Install the 'ocr' extra or use sidecar OCR.") from e
        kwargs: dict[str, Any] = {
            "lang": lang,
            "ocr_version": self.config.ocr_version,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if device:
            kwargs["device"] = device
        self.engine = PaddleOCR(**kwargs)

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        results = self.engine.predict(image)
        blocks: list[TextBlock] = []
        offset = 0
        for page_result in results:
            data = _result_to_dict(page_result)
            texts = list(data.get("rec_texts", []))
            scores = list(data.get("rec_scores", [1.0] * len(texts)))
            polys = data.get("rec_polys")
            if polys is None or len(polys) != len(texts):
                polys = data.get("dt_polys")
            if polys is None or len(polys) != len(texts):
                boxes = data.get("rec_boxes")
                if boxes is not None:
                    polys = [bbox_polygon(box) for box in boxes]
            if polys is None:
                continue
            for i, (text, score, poly) in enumerate(zip(texts, scores, polys)):
                arr = np.asarray(poly, dtype=float).reshape(-1, 2)
                if len(arr) < 3:
                    continue
                blocks.append(
                    TextBlock(
                        id=f"ocr-{offset+i:04d}",
                        polygon=[(float(x), float(y)) for x, y in arr],
                        text=str(text),
                        confidence=float(score),
                        reading_order=offset + i,
                        meta={"backend": "paddle", "ocr_version": self.config.ocr_version},
                    )
                )
            offset += len(texts)
        return sort_reading_order(blocks)


class RetryingOCRBackend(OCRBackend):
    """Re-runs low-confidence crops at higher scale and keeps evidence for arbitration."""

    def __init__(self, primary: OCRBackend, threshold: float = 0.80, scale: float = 2.0) -> None:
        self.primary = primary
        self.threshold = threshold
        self.scale = max(1.0, scale)

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        blocks = self.primary.recognize(image, image_path=image_path)
        h, w = image.shape[:2]
        for block in blocks:
            if block.confidence >= self.threshold:
                continue
            x0, y0, x1, y1 = block.bbox
            pad = max(3, round(min(x1 - x0, y1 - y0) * 0.12))
            ix0, iy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ix1, iy1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            crop = image[iy0:iy1, ix0:ix1]
            up = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            sharpen = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.7, 0)
            variants = [up, cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)]
            candidates = [{"text": block.text, "confidence": block.confidence, "variant": "original"}]
            for vi, variant in enumerate(variants):
                try:
                    reread = self.primary.recognize(variant, image_path=None)
                except Exception:
                    continue
                if not reread:
                    continue
                # Prefer the most confident, text-bearing region in the crop.
                cand = max(reread, key=lambda b: (b.confidence, len(b.text.strip())))
                candidates.append({"text": cand.text, "confidence": cand.confidence, "variant": f"retry_{vi}"})
            best = max(candidates, key=lambda x: (x["confidence"], len(x["text"].strip())))
            block.meta["ocr_candidates"] = candidates
            if best["confidence"] > block.confidence + 0.015 and best["text"].strip():
                block.meta["ocr_original"] = {"text": block.text, "confidence": block.confidence}
                block.text = str(best["text"])
                block.confidence = float(best["confidence"])
                block.meta["ocr_selected"] = best["variant"]
        return blocks


class InjectedOCRBackend(OCRBackend):
    """Test/embedding adapter that returns caller-provided blocks by image path or one static list."""

    def __init__(self, blocks: list[TextBlock] | dict[str, list[TextBlock]]) -> None:
        self.blocks = blocks

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if isinstance(self.blocks, dict):
            key = str(image_path) if image_path is not None else ""
            values = self.blocks.get(key, [])
        else:
            values = self.blocks
        return [
            TextBlock(
                id=b.id,
                polygon=list(b.polygon),
                text=b.text,
                confidence=b.confidence,
                kind=b.kind,
                reading_order=b.reading_order,
                bubble_id=b.bubble_id,
                meta=dict(b.meta),
            )
            for b in values
        ]


def sort_reading_order(blocks: list[TextBlock], vertical: bool = False) -> list[TextBlock]:
    if vertical:
        ordered = sorted(blocks, key=lambda b: (-b.centroid[0], b.centroid[1]))
    else:
        # Row-aware sort: quantize Y by median block height to avoid tiny OCR jitter.
        heights = [max(1.0, b.bbox[3] - b.bbox[1]) for b in blocks]
        band = float(np.median(heights)) * 0.7 if heights else 16.0
        ordered = sorted(blocks, key=lambda b: (round(b.centroid[1] / max(1.0, band)), b.centroid[0]))
    for i, block in enumerate(ordered):
        block.reading_order = i
    return ordered


def filter_blocks(blocks: list[TextBlock], min_confidence: float) -> tuple[list[TextBlock], list[TextBlock]]:
    accepted, review = [], []
    for b in blocks:
        (accepted if b.confidence >= min_confidence else review).append(b)
    return accepted, review


def build_backend(config: OCRConfig, lang: str, backend: str | None = None) -> OCRBackend:
    name = (backend or config.backend).lower()
    if name == "paddle":
        return PaddleOCRBackend(lang, config)
    if name == "sidecar":
        return SidecarOCRBackend(config.sidecar_suffix)
    if name == "none":
        return NullOCRBackend()
    raise ValueError(f"Unknown OCR backend: {name}")
