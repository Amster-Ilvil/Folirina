from __future__ import annotations

"""Reletter-only OCR execution service.

This module owns the two OCR routes that operate on already-paired manga
containers.  It intentionally does not import ``pipeline`` or any raster transfer
renderer: TARGET geometry remains authoritative, SOURCE contributes text only,
and the caller supplies a cancellation callback and optional trace sink.
"""

from pathlib import Path
from typing import Callable, Any

import cv2
import numpy as np

from .cache import PageStageCache, image_stage_signature
from .config import PipelineConfig
from .models import BubbleInstance, TextBlock
from .ocr import OCRBackend
from .reletter_regions import detect_target_text_regions, normalized_map_bbox
from .reletter_binding import (
    infer_region_orientation,
    source_layout_profile,
    pair_reletter_bubbles,
    filter_region_query_blocks,
    match_paired_bubble_regions,
    normalize_region_ocr_text,
)


CancelCheck = Callable[[str], None]


def _noop_cancel(_stage: str) -> None:
    return None


def _bbox_intersection_ratio(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
    """Return how much of *outer* is covered by *inner*.

    Target-driven OCR uses this to detect a truncated independently-detected
    SOURCE text subregion.  The TARGET projection is the expected full text
    envelope; a matched SOURCE subregion that covers only a small fraction of
    it must not be allowed to crop away glyphs.
    """
    ax0, ay0, ax1, ay1 = [int(v) for v in inner]
    bx0, by0, bx1, by1 = [int(v) for v in outer]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    area = max(1, (bx1 - bx0) * (by1 - by0))
    return float(inter) / float(area)


def _clamp_bbox_to_image(box: tuple[int, int, int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = [int(v) for v in shape]
    x0, y0, x1, y1 = [int(v) for v in box]
    x0, x1 = sorted((max(0, min(w, x0)), max(0, min(w, x1))))
    y0, y1 = sorted((max(0, min(h, y0)), max(0, min(h, y1))))
    return x0, y0, x1, y1


def _recover_source_crop_bbox(
    matched_bbox: tuple[int, int, int, int] | None,
    projected_bbox: tuple[int, int, int, int],
    source_bubble_bbox: tuple[int, int, int, int],
    source_shape: tuple[int, int],
) -> tuple[tuple[int, int, int, int], bool, float]:
    """Choose a complete SOURCE OCR crop for a TARGET-driven text region.

    Independent SOURCE text-region detection is useful when accurate, but on
    manga it can stop at the first few vertical glyphs.  When that happens the
    old Reletter path OCRs a visibly truncated crop even though TARGET and the
    paired SOURCE bubble already provide a complete geometry envelope.

    The projected TARGET region therefore acts as the completion authority.
    A matched SOURCE region is retained only when it covers at least 78% of the
    projected envelope.  Otherwise we recover to the projection, lightly padded
    and bounded to the paired SOURCE bubble.
    """
    px0, py0, px1, py1 = [int(v) for v in projected_bbox]
    cover = 1.0
    if matched_bbox is not None:
        cover = _bbox_intersection_ratio(tuple(int(v) for v in matched_bbox), projected_bbox)
        if cover >= 0.78:
            return _clamp_bbox_to_image(tuple(int(v) for v in matched_bbox), source_shape), False, cover

    # Recovery path: target-projected geometry plus a small text-safe pad.
    pw, ph = max(1, px1 - px0), max(1, py1 - py0)
    pad_x = max(4, int(round(pw * 0.035)))
    pad_y = max(4, int(round(ph * 0.035)))
    bx0, by0, bx1, by1 = [int(round(v)) for v in source_bubble_bbox]
    # Allow a tiny margin outside the detected bubble because detector polygons
    # often stop exactly on the outermost antialiased glyph/container edge.
    bubble_pad_x = max(3, int(round(max(1, bx1 - bx0) * 0.025)))
    bubble_pad_y = max(3, int(round(max(1, by1 - by0) * 0.025)))
    rx0 = max(bx0 - bubble_pad_x, px0 - pad_x)
    ry0 = max(by0 - bubble_pad_y, py0 - pad_y)
    rx1 = min(bx1 + bubble_pad_x, px1 + pad_x)
    ry1 = min(by1 + bubble_pad_y, py1 + pad_y)
    recovered = _clamp_bbox_to_image((rx0, ry0, rx1, ry1), source_shape)
    return recovered, matched_bbox is not None, cover


class ReletterExecutor:
    """Execute reletter OCR without owning page orchestration or rendering."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        trace: Any | None = None,
        cancel_check: CancelCheck | None = None,
        detect_regions=detect_target_text_regions,
    ) -> None:
        self.config = config
        self.trace = trace
        self.cancel_check = cancel_check or _noop_cancel
        self.detect_regions = detect_regions

    def recognize_target_driven_regions(
        self,
        backend: OCRBackend,
        source: np.ndarray,
        target: np.ndarray,
        source_path: str | Path,
        source_bubbles: list[BubbleInstance],
        target_bubbles: list[BubbleInstance],
        stats: dict,
    ) -> tuple[list[TextBlock], list[TextBlock], list[BubbleInstance], list[BubbleInstance], dict]:
        """Create immutable TARGET text regions, then OCR only their SOURCE peers."""
        target_to_source, bubble_pair_diag = pair_reletter_bubbles(
            source_bubbles, target_bubbles, source.shape[:2], target.shape[:2]
        )

        source_blocks: list[TextBlock] = []
        target_blocks: list[TextBlock] = []
        region_rows: list[dict] = []
        bubble_rows: list[dict] = []
        rejected_bubbles: list[dict] = []
        order = 0
        sh, sw = source.shape[:2]

        for tb in target_bubbles:
            self.cancel_check("reletter_target_bubble")
            sb = target_to_source.get(tb.id)
            if sb is None:
                rejected_bubbles.append({"target_bubble_id": tb.id, "reason": "missing_paired_source"})
                continue
            regions = self.detect_regions(target, tb)
            if not regions:
                rejected_bubbles.append({"target_bubble_id": tb.id, "reason": "no_printed_text_region"})
                tb.block_ids = []
                sb.block_ids = []
                continue

            source_regions = self.detect_regions(source, sb)
            source_region_map, source_region_diag = match_paired_bubble_regions(
                source_regions, regions, sb.bbox, tb.bbox
            )
            bubble_diag = {
                "target_bubble_id": tb.id,
                "source_bubble_id": sb.id,
                "target_region_count": len(regions),
                "source_region_count": len(source_regions),
                "region_matching": dict(source_region_diag or {}),
            }
            bubble_rows.append(bubble_diag)
            if self.trace is not None:
                self.trace.event("reletter_bubble_binding", **bubble_diag)

            tb.block_ids = []
            sb.block_ids = []
            for region in regions:
                self.cancel_check("reletter_source_region_ocr")
                matched_source_region = source_region_map.get(str(region.id))
                projected_bbox = normalized_map_bbox(
                    region.bbox, tb.bbox, sb.bbox, (sh, sw)
                )
                matched_bbox = (
                    tuple(int(v) for v in matched_source_region.bbox)
                    if matched_source_region is not None else None
                )
                (sx0, sy0, sx1, sy1), recovered_from_truncation, source_projection_coverage = _recover_source_crop_bbox(
                    matched_bbox, tuple(int(v) for v in projected_bbox), tuple(int(round(v)) for v in sb.bbox), (sh, sw)
                )
                if recovered_from_truncation:
                    source_crop_route = "target_projection_recovered_truncated_source_region"
                elif matched_source_region is not None:
                    source_crop_route = "paired_source_subregion"
                else:
                    source_crop_route = "normalized_target_projection"
                if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                    continue

                crop = source[sy0:sy1, sx0:sx1].copy()
                local_mask = None
                # A recovered crop must not reuse the truncated SOURCE text mask;
                # doing so would whiten the newly recovered lower/right glyphs
                # before OCR sees them.  The paired bubble mask remains available
                # below as a broad, safe container constraint.
                if (not recovered_from_truncation and matched_source_region is not None
                        and getattr(matched_source_region, "text_mask", None) is not None):
                    sm = matched_source_region.text_mask
                    if sm.shape[:2] == source.shape[:2]:
                        local_mask = sm[sy0:sy1, sx0:sx1]
                if (
                    (local_mask is None or not local_mask.size or cv2.countNonZero(local_mask) == 0)
                    and sb.mask is not None
                    and sb.mask.shape[:2] == source.shape[:2]
                ):
                    local_mask = sb.mask[sy0:sy1, sx0:sx1]
                if local_mask is not None and local_mask.size and cv2.countNonZero(local_mask) > 0:
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    local_mask = cv2.dilate((local_mask > 0).astype(np.uint8) * 255, k, iterations=1)
                    crop[local_mask == 0] = 255

                # Local engines must see the masked crop. Coordinate-query backends
                # (external JSON/MD) instead query the original page and are then
                # filtered back down to the matched SOURCE region.
                if bool(getattr(backend, "supports_crop_recognition", True)) and not bool(
                    getattr(backend, "supports_region_query", False)
                ):
                    raw = backend.recognize(crop, image_path=None)
                    ocr_route = "masked_crop"
                    query_diag = {
                        "route": "masked_crop",
                        "input_blocks": len(raw),
                        "accepted_blocks": len(raw),
                    }
                else:
                    raw_query = backend.recognize_region(
                        source, (sx0, sy0, sx1, sy1), image_path=source_path
                    )
                    full_region_mask = (
                        getattr(matched_source_region, "text_mask", None)
                        if matched_source_region is not None
                        else None
                    )
                    raw, query_diag = filter_region_query_blocks(
                        raw_query, (sx0, sy0, sx1, sy1), full_region_mask
                    )
                    ocr_route = "region_query"

                text_value, ocr_norm_diag = normalize_region_ocr_text(raw, region.orientation)
                ocr_norm_diag["region_query"] = query_diag
                if not text_value:
                    region_rows.append(
                        {
                            "region_id": region.id,
                            "target_bubble_id": tb.id,
                            "source_bbox": [sx0, sy0, sx1, sy1],
                            "status": "source_ocr_empty",
                        }
                    )
                    continue

                confs = [float(x.confidence) for x in raw if str(x.text).strip()]
                conf = float(np.mean(confs)) if confs else 0.88
                orientation = region.orientation
                layout_profile = source_layout_profile(crop, text_value, orientation)
                region_id = str(region.id)
                sid = f"reletter-source-{order:04d}"
                tid = f"reletter-target-{order:04d}"
                source_poly = [
                    (float(sx0), float(sy0)),
                    (float(sx1), float(sy0)),
                    (float(sx1), float(sy1)),
                    (float(sx0), float(sy1)),
                ]
                source_blocks.append(
                    TextBlock(
                        id=sid,
                        polygon=source_poly,
                        text=text_value,
                        confidence=conf,
                        kind=tb.kind,
                        reading_order=order,
                        bubble_id=sb.id,
                        meta={
                            "backend": type(backend).__name__,
                            "target_driven_reletter": True,
                            "reletter_region_id": region_id,
                            "paired_target_region_id": region_id,
                            "paired_target_id": tb.id,
                            "source_region_bbox": [sx0, sy0, sx1, sy1],
                            "source_crop_route": source_crop_route,
                            "source_projected_bbox": [int(v) for v in projected_bbox],
                            "source_projection_coverage": float(source_projection_coverage),
                            "source_crop_recovered": bool(recovered_from_truncation),
                            "orientation_hint": orientation,
                            "source_layout_profile": layout_profile,
                            "ocr_normalization": dict(ocr_norm_diag),
                            "ocr_route": ocr_route,
                        },
                    )
                )
                target_blocks.append(
                    TextBlock(
                        id=tid,
                        polygon=list(region.polygon),
                        text="□",
                        confidence=region.confidence,
                        kind=tb.kind,
                        reading_order=order,
                        bubble_id=tb.id,
                        meta={
                            "backend": "target_text_region",
                            "target_driven_reletter": True,
                            "synthetic_geometry_only": True,
                            "synthetic_region_only": True,
                            "reletter_region_id": region_id,
                            "paired_source_region_id": region_id,
                            "paired_source_id": sb.id,
                            "orientation_hint": orientation,
                            "target_region_bbox": list(region.bbox),
                            "component_count": int(region.component_count),
                            "region_diagnostics": dict(region.diagnostics),
                        },
                    )
                )
                sb.block_ids.append(sid)
                tb.block_ids.append(tid)
                region_rows.append(
                    {
                        "region_id": region_id,
                        "target_bubble_id": tb.id,
                        "source_bubble_id": sb.id,
                        "target_bbox": list(region.bbox),
                        "source_bbox": [sx0, sy0, sx1, sy1],
                        "orientation": orientation,
                        "component_count": region.component_count,
                        "status": "recognized",
                        "source_crop_route": source_crop_route,
                        "source_projected_bbox": [int(v) for v in projected_bbox],
                        "source_projection_coverage": float(source_projection_coverage),
                        "source_crop_recovered": bool(recovered_from_truncation),
                        "ocr_normalization": dict(ocr_norm_diag),
                        "ocr_route": ocr_route,
                        "source_region_bbox": (
                            list(matched_source_region.bbox)
                            if matched_source_region is not None
                            else None
                        ),
                    }
                )
                order += 1

        stats["ocr_source"] = "target_driven_regions"
        stats["ocr_target"] = "target_text_regions"
        diagnostics = {
            "route": "target_driven_reletter",
            "regions": region_rows,
            "recognized_regions": len(source_blocks),
            "rejected_bubbles": rejected_bubbles,
            "bubble_pairing": bubble_pair_diag,
            "source_region_matching": [
                r
                for r in region_rows
                if str(r.get("source_crop_route", "")) == "paired_source_subregion"
            ],
        }
        return source_blocks, target_blocks, source_bubbles, target_bubbles, diagnostics

    def recognize_paired_regions_text_only(
        self,
        backend: OCRBackend,
        source: np.ndarray,
        source_path: str | Path,
        source_bubbles: list[BubbleInstance],
        target_bubbles: list[BubbleInstance],
        cache: PageStageCache,
        stats: dict,
    ) -> tuple[list[TextBlock], list[TextBlock], list[BubbleInstance], list[BubbleInstance]]:
        """OCR each already-paired SOURCE bubble using a transcript-only backend."""
        source_to_target: dict[str, str] = {}
        target_to_source: dict[str, str] = {}
        for sb in source_bubbles:
            tid = str(sb.meta.get("paired_target_id") or "")
            if tid:
                source_to_target[sb.id] = tid
                target_to_source[tid] = sb.id
        for tb in target_bubbles:
            sid = str(tb.meta.get("paired_source_id") or "")
            if sid:
                target_to_source[tb.id] = sid
                source_to_target[sid] = tb.id

        region_rows = [
            {
                "id": b.id,
                "bbox": [round(v, 2) for v in b.bbox],
                "paired": source_to_target.get(b.id, ""),
            }
            for b in source_bubbles
        ]
        sig = image_stage_signature(
            source_path,
            self.config.ocr,
            {
                "role": "source_paired_regions",
                "backend": type(backend).__name__,
                "lang": self.config.ocr.source_lang,
                "regions": region_rows,
                "orientation_policy": "source_ink_v1",
            },
        )
        if self.config.cache.ocr:
            cached = cache.load_blocks("source_paired_regions", sig)
            if cached is not None:
                by_bubble = {
                    str(b.meta.get("paired_region_source_id")): b.id for b in cached
                }
                for bubble in source_bubbles:
                    bid = by_bubble.get(bubble.id)
                    bubble.block_ids = [bid] if bid else []
                target_blocks: list[TextBlock] = []
                for i, bubble in enumerate(target_bubbles):
                    source_id = target_to_source.get(
                        bubble.id, str(bubble.meta.get("paired_source_id") or "")
                    )
                    if source_id not in by_bubble:
                        bubble.block_ids = []
                        continue
                    block_id = f"apple-target-geometry-{i:04d}"
                    bubble.block_ids = [block_id]
                    target_blocks.append(
                        TextBlock(
                            id=block_id,
                            polygon=list(bubble.polygon),
                            text="□",
                            confidence=1.0,
                            kind=bubble.kind,
                            reading_order=i,
                            bubble_id=bubble.id,
                            meta={
                                "backend": "paired_geometry",
                                "synthetic_geometry_only": True,
                            },
                        )
                    )
                stats["ocr_source"] = "hit_paired_regions"
                stats["ocr_target"] = "geometry_only"
                return cached, target_blocks, source_bubbles, target_bubbles

        h, w = source.shape[:2]
        pad_ratio = float(
            getattr(self.config.ocr, "apple_live_text_region_padding_ratio", 0.08)
        )
        min_side = int(getattr(self.config.ocr, "apple_live_text_region_min_side_px", 28))
        whiten = bool(
            getattr(self.config.ocr, "apple_live_text_region_whiten_outside_mask", True)
        )
        blocks: list[TextBlock] = []
        recognized_source_ids: set[str] = set()

        for i, bubble in enumerate(source_bubbles):
            self.cancel_check("paired_region_ocr")
            x0, y0, x1, y1 = bubble.bbox
            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            pad = max(3, int(round(max(bw, bh) * pad_ratio)))
            ix0 = max(0, int(np.floor(x0)) - pad)
            iy0 = max(0, int(np.floor(y0)) - pad)
            ix1 = min(w, int(np.ceil(x1)) + pad)
            iy1 = min(h, int(np.ceil(y1)) + pad)
            if ix1 - ix0 < min_side or iy1 - iy0 < min_side:
                bubble.block_ids = []
                continue

            crop = source[iy0:iy1, ix0:ix1].copy()
            layout_crop = crop.copy()
            if whiten and bubble.mask is not None and bubble.mask.shape[:2] == source.shape[:2]:
                raw_local_mask = bubble.mask[iy0:iy1, ix0:ix1]
                k = max(3, int(round(min(bw, bh) * 0.025)) | 1)
                ocr_mask = cv2.dilate(raw_local_mask, np.ones((k, k), np.uint8), iterations=1)
                crop[ocr_mask == 0] = 255
                inner_k = max(3, min(11, k | 1))
                inner = cv2.erode(
                    raw_local_mask, np.ones((inner_k, inner_k), np.uint8), iterations=1
                )
                layout_crop[inner == 0] = 255

            raw = backend.recognize(crop, image_path=None)
            text = "\n".join(
                str(b.text).strip() for b in raw if str(b.text).strip()
            ).strip()
            if not text:
                bubble.block_ids = []
                continue

            orientation_hint, orientation_meta = infer_region_orientation(
                crop, text, bubble.kind
            )
            layout_profile = source_layout_profile(layout_crop, text, orientation_hint)
            confs = [float(b.confidence) for b in raw if str(b.text).strip()]
            conf = (
                float(np.mean(confs))
                if confs
                else float(
                    getattr(self.config.ocr, "apple_live_text_assumed_confidence", 0.88)
                )
            )
            block_id = f"apple-region-{i:04d}"
            meta = {
                "backend": str(raw[0].meta.get("backend") if raw else "apple_live_text"),
                "paired_region_ocr": True,
                "paired_region_source_id": bubble.id,
                "paired_target_id": source_to_target.get(
                    bubble.id, str(bubble.meta.get("paired_target_id") or "")
                ),
                "ocr_region_bbox": [ix0, iy0, ix1, iy1],
                "text_only_geometry": "paired_bubble",
                "orientation_hint": orientation_hint,
                "orientation_evidence": orientation_meta,
                "source_layout_profile": layout_profile,
            }
            if raw:
                for key in (
                    "apple_auto_route",
                    "apple_live_text_fallback_reason",
                    "languages",
                ):
                    if key in raw[0].meta:
                        meta[key] = raw[0].meta[key]
            block = TextBlock(
                id=block_id,
                polygon=list(bubble.polygon),
                text=text,
                confidence=conf,
                kind=bubble.kind,
                reading_order=i,
                bubble_id=bubble.id,
                meta=meta,
            )
            blocks.append(block)
            bubble.block_ids = [block_id]
            recognized_source_ids.add(bubble.id)

        target_blocks: list[TextBlock] = []
        for i, bubble in enumerate(target_bubbles):
            source_id = target_to_source.get(
                bubble.id, str(bubble.meta.get("paired_source_id") or "")
            )
            if source_id not in recognized_source_ids:
                bubble.block_ids = []
                continue
            block_id = f"apple-target-geometry-{i:04d}"
            bubble.block_ids = [block_id]
            target_blocks.append(
                TextBlock(
                    id=block_id,
                    polygon=list(bubble.polygon),
                    text="□",
                    confidence=1.0,
                    kind=bubble.kind,
                    reading_order=i,
                    bubble_id=bubble.id,
                    meta={
                        "backend": "paired_geometry",
                        "synthetic_geometry_only": True,
                    },
                )
            )

        if self.config.cache.ocr:
            cache.save_blocks("source_paired_regions", sig, blocks)
        stats["ocr_source"] = "miss_paired_regions"
        stats["ocr_target"] = "geometry_only"
        return blocks, target_blocks, source_bubbles, target_bubbles


__all__ = ["ReletterExecutor"]
