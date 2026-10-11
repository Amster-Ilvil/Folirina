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

from ...cache import PageStageCache, image_stage_signature
from ...config import PipelineConfig
from ...models import BubbleInstance, TextBlock
from ...ocr import OCRBackend
from .regions import detect_target_text_regions, normalized_map_bbox
from .binding import (
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
                source_crop_padding_applied = False
                if recovered_from_truncation:
                    source_crop_route = "target_projection_recovered_truncated_source_region"
                elif matched_source_region is not None:
                    source_crop_route = "paired_source_subregion"
                else:
                    source_crop_route = "normalized_target_projection"
                # One transcript belongs to the whole paired SOURCE container,
                # not the shorter Japanese ink envelope. Multi-region bubbles
                # retain subregion binding to avoid duplicated transcripts.
                if (len(regions) == 1 and len(source_regions) <= 1
                        and bool(getattr(backend, "region_text_only", False))):
                    # Transcript-only fallback OCR needs a small amount of paper
                    # around the detected container. Cropping exactly on the
                    # balloon polygon lets border strokes touch the OCR frame and
                    # sharply lowers confidence on otherwise clean vertical CJK.
                    # Use a bounded padding based on the *short* container side so
                    # context is restored without admitting nearby panel art.
                    sbw = max(1.0, float(sb.bbox[2] - sb.bbox[0]))
                    sbh = max(1.0, float(sb.bbox[3] - sb.bbox[1]))
                    context_pad = max(4, min(12, int(round(min(sbw, sbh) * 0.04))))
                    sx0, sy0, sx1, sy1 = (
                        max(0, int(np.floor(sb.bbox[0])) - context_pad),
                        max(0, int(np.floor(sb.bbox[1])) - context_pad),
                        min(sw, int(np.ceil(sb.bbox[2])) + context_pad),
                        min(sh, int(np.ceil(sb.bbox[3])) + context_pad),
                    )
                    recovered_from_truncation = True
                    source_crop_padding_applied = True
                    source_crop_route = "single_region_source_container"
                if sx1 - sx0 < 4 or sy1 - sy0 < 4:
                    continue

                crop = source[sy0:sy1, sx0:sx1].copy()
                unmasked_crop = crop.copy()
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
                    mask_scale = min(
                        max(1.0, float(sb.bbox[2] - sb.bbox[0])),
                        max(1.0, float(sb.bbox[3] - sb.bbox[1])),
                    )
                    mask_k = max(3, int(round(mask_scale * 0.025)) | 1)
                    mask_k = min(mask_k, 11)
                    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (mask_k, mask_k))
                    local_mask = cv2.dilate((local_mask > 0).astype(np.uint8) * 255, k, iterations=1)
                    crop[local_mask == 0] = 255

                # Local engines must see the masked crop. Coordinate-query backends
                # (external JSON/MD) instead query the original page and are then
                # filtered back down to the matched SOURCE region.
                if bool(getattr(backend, "supports_crop_recognition", True)) and not bool(
                    getattr(backend, "supports_region_query", False)
                ):
                    if hasattr(backend, "recognize_manga_region"):
                        raw = backend.recognize_manga_region(crop, orientation=region.orientation)
                        ocr_route = "masked_crop_manga_orientation"
                    else:
                        raw = backend.recognize(crop, image_path=None)
                        ocr_route = "masked_crop"
                    query_diag = {
                        "route": ocr_route,
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

                if not raw and getattr(backend, "_all_unavailable_reason", ""):
                    raise RuntimeError("Apple OCR 后端不可用（不是页面无文字）：" + backend._all_unavailable_reason)
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
                raw_backend = str((raw[0].meta or {}).get("backend", "") if raw else "").lower()
                # Tesseract vertical manga OCR is column-based. One difficult
                # glyph column should not drag an otherwise stable transcript
                # below the global acceptance threshold, so use the robust
                # median for this fallback only. Other OCR engines keep their
                # historical mean-confidence semantics.
                conf = (
                    float(np.median(confs)) if confs and raw_backend == "tesseract"
                    else float(np.mean(confs)) if confs
                    else 0.88
                )
                if raw_backend == "tesseract":
                    cjk_count = sum(
                        1 for ch in str(text_value)
                        if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                    )
                    # Masking protects nearby artwork, but on some photographed
                    # balloons it clips antialiased edge strokes and depresses OCR
                    # confidence. Retry the same bounded crop without whitening only
                    # when the masked result is below the publication threshold,
                    # then keep whichever transcript has stronger CJK evidence.
                    compare_unmasked = bool(
                        hasattr(backend, "recognize_manga_region")
                        and (
                            conf < float(getattr(self.config.qa, "ocr_min_confidence", 0.66))
                            or float(source_projection_coverage) < float(getattr(
                                self.config.ocr, "tesseract_unmasked_compare_projection_coverage", 0.72
                            ))
                        )
                    )
                    masked_text_before_compare = str(text_value or "")
                    masked_conf_before_compare = float(conf)
                    masked_unmasked_material_disagreement = False
                    if compare_unmasked:
                        alt_raw = backend.recognize_manga_region(
                            unmasked_crop, orientation=region.orientation
                        )
                        alt_text, alt_norm = normalize_region_ocr_text(alt_raw, region.orientation)
                        alt_confs = [float(x.confidence) for x in alt_raw if str(x.text).strip()]
                        alt_conf = float(np.mean(alt_confs)) if alt_confs else 0.0
                        alt_cjk = sum(
                            1 for ch in str(alt_text)
                            if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                        )
                        current_score = (int(cjk_count > 0), min(cjk_count, 24), float(conf))
                        alt_score = (int(alt_cjk > 0), min(alt_cjk, 24), float(alt_conf))
                        if alt_text and masked_text_before_compare and alt_text != masked_text_before_compare:
                            import difflib
                            def _cjk_only(value: str) -> str:
                                return "".join(
                                    ch for ch in str(value or "")
                                    if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                                )
                            masked_cjk_text = _cjk_only(masked_text_before_compare)
                            alt_cjk_text = _cjk_only(alt_text)
                            ratio = difflib.SequenceMatcher(
                                None, masked_cjk_text, alt_cjk_text, autojunk=False
                            ).ratio() if masked_cjk_text and alt_cjk_text else 0.0
                            same_length_substitution = bool(
                                masked_cjk_text and alt_cjk_text
                                and len(masked_cjk_text) == len(alt_cjk_text)
                                and masked_cjk_text != alt_cjk_text
                            )
                            masked_unmasked_material_disagreement = bool(
                                same_length_substitution
                                or ratio < float(getattr(
                                    self.config.ocr, "tesseract_unmasked_disagreement_min_ratio", 0.96
                                ))
                            )
                            alt_review_flag = any(
                                bool((getattr(child, "meta", {}) or {}).get("column_consensus_review_required"))
                                for child in alt_raw
                            )
                            alt_clearly_dominates = bool(
                                alt_conf >= masked_conf_before_compare + float(getattr(
                                    self.config.ocr, "tesseract_unmasked_override_confidence_gain", 0.10
                                ))
                                and not alt_review_flag
                            )
                            ocr_norm_diag["masked_unmasked_compare"] = {
                                "masked_text": masked_text_before_compare,
                                "unmasked_text": str(alt_text),
                                "masked_confidence": masked_conf_before_compare,
                                "unmasked_confidence": float(alt_conf),
                                "cjk_similarity": float(ratio),
                                "source_projection_coverage": float(source_projection_coverage),
                                "unmasked_clearly_dominates": bool(alt_clearly_dominates),
                            }
                            if masked_unmasked_material_disagreement and not alt_clearly_dominates:
                                ocr_norm_diag.setdefault("fallback_review_reasons", [])
                                if "masked_unmasked_ocr_disagreement" not in ocr_norm_diag["fallback_review_reasons"]:
                                    ocr_norm_diag["fallback_review_reasons"].append("masked_unmasked_ocr_disagreement")
                        if alt_text and (alt_conf >= conf + 0.03 or alt_score > current_score):
                            raw = alt_raw
                            text_value = alt_text
                            conf = alt_conf
                            cjk_count = alt_cjk
                            previous_review_reasons = list(ocr_norm_diag.get("fallback_review_reasons", []) or [])
                            previous_compare = ocr_norm_diag.get("masked_unmasked_compare")
                            ocr_norm_diag = alt_norm
                            if previous_review_reasons:
                                ocr_norm_diag["fallback_review_reasons"] = previous_review_reasons
                            if previous_compare is not None:
                                ocr_norm_diag["masked_unmasked_compare"] = previous_compare
                            ocr_norm_diag["region_query"] = {
                                **dict(query_diag),
                                "route": str(query_diag.get("route", ocr_route)) + "_unmasked_retry",
                                "masked_confidence": float(current_score[2]),
                                "unmasked_confidence": float(alt_conf),
                            }
                            ocr_route = str(ocr_route) + "_unmasked_retry"
                    # Do not let a median region confidence hide one structurally
                    # unreliable CJK column.  Tesseract is a cloud/runtime fallback,
                    # so incomplete consensus or a near-zero glyph column must be
                    # review-only instead of silently entering publication pixels.
                    cjk_child_confidences = []
                    consensus_coverages = []
                    consensus_agreements = []
                    for child in raw:
                        child_text = str(getattr(child, "text", "") or "")
                        child_cjk = sum(
                            1 for ch in child_text
                            if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                        )
                        if child_cjk > 0:
                            cjk_child_confidences.append(float(getattr(child, "confidence", 0.0) or 0.0))
                        child_meta = dict(getattr(child, "meta", {}) or {})
                        if bool(child_meta.get("column_consensus")):
                            consensus_coverages.append(float(child_meta.get("consensus_char_coverage", 0.0) or 0.0))
                            consensus_agreements.append(float(child_meta.get("consensus_pairwise_agreement", 0.0) or 0.0))
                        if bool(child_meta.get("column_consensus_review_required")):
                            ocr_norm_diag.setdefault("fallback_review_reasons", [])
                            if "unresolved_column_consensus" not in ocr_norm_diag["fallback_review_reasons"]:
                                ocr_norm_diag["fallback_review_reasons"].append("unresolved_column_consensus")
                    min_child = min(cjk_child_confidences) if cjk_child_confidences else 1.0
                    min_consensus_coverage = min(consensus_coverages) if consensus_coverages else 1.0
                    min_consensus_agreement = min(consensus_agreements) if consensus_agreements else 1.0
                    review_reasons = list(ocr_norm_diag.get("fallback_review_reasons", []) or [])
                    if min_child < float(getattr(self.config.ocr, "tesseract_region_review_min_child_confidence", 0.35)):
                        review_reasons.append("weak_cjk_column")
                    if min_consensus_coverage < float(getattr(self.config.ocr, "tesseract_region_review_min_consensus_coverage", 0.90)):
                        review_reasons.append("incomplete_column_consensus")
                    if min_consensus_agreement < float(getattr(self.config.ocr, "tesseract_region_review_min_consensus_agreement", 0.55)):
                        review_reasons.append("low_column_consensus_agreement")
                    if review_reasons:
                        conf = min(
                            conf,
                            float(getattr(self.config.ocr, "tesseract_region_review_confidence_cap", 0.64)),
                        )
                        ocr_norm_diag["fallback_review_reasons"] = list(review_reasons)
                        ocr_norm_diag["fallback_min_child_confidence"] = float(min_child)
                        ocr_norm_diag["fallback_min_consensus_coverage"] = float(min_consensus_coverage)
                        ocr_norm_diag["fallback_min_consensus_agreement"] = float(min_consensus_agreement)
                    reject_fallback = (
                        cjk_count <= 0
                        or (cjk_count == 1 and conf < 0.82)
                        or (2 <= cjk_count <= 5 and conf < 0.35)
                        or (cjk_count >= 6 and conf < 0.25)
                    )
                    if reject_fallback:
                        stats["ocr_source_fallback_quality_rejected"] = int(
                            stats.get("ocr_source_fallback_quality_rejected", 0) or 0
                        ) + 1
                        region_rows.append({
                            "region_id": region.id,
                            "target_bubble_id": tb.id,
                            "source_bbox": [sx0, sy0, sx1, sy1],
                            "status": "fallback_ocr_quality_rejected",
                            "confidence": float(conf),
                            "cjk_count": int(cjk_count),
                        })
                        continue
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
                            "source_crop_padding_applied": bool(source_crop_padding_applied),
                            "source_projected_bbox": [int(v) for v in projected_bbox],
                            "source_projection_coverage": float(source_projection_coverage),
                            "source_crop_recovered": bool(recovered_from_truncation),
                            "orientation_hint": orientation,
                            "source_layout_profile": layout_profile,
                            "ocr_normalization": dict(ocr_norm_diag),
                            "ocr_route": ocr_route,
                            "ocr_backend_receipts": [dict(b.meta or {}) for b in raw],
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
                        "source_crop_padding_applied": bool(source_crop_padding_applied),
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
                "recognition_policy": "manga_vertical_region_v3",
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

            # Determine manga text direction from the isolated ink before OCR.
            # Transcript-only fallbacks such as Tesseract need this hint up front:
            # a wide multi-column speech bubble is still vertical manga text and
            # must not be treated as a horizontal paragraph merely from aspect.
            orientation_hint, orientation_meta = infer_region_orientation(
                layout_crop, "中文", bubble.kind
            )
            if hasattr(backend, "recognize_manga_region"):
                raw = backend.recognize_manga_region(crop, orientation=orientation_hint)
                # Ambiguous region-direction estimates are allowed one opposite-
                # direction retry only when the primary route produced no CJK.
                primary_text, _primary_diag = normalize_region_ocr_text(raw, orientation_hint)
                if not primary_text:
                    alt_orientation = "horizontal" if orientation_hint == "vertical" else "vertical"
                    alt_raw = backend.recognize_manga_region(crop, orientation=alt_orientation)
                    alt_text, _alt_diag = normalize_region_ocr_text(alt_raw, alt_orientation)
                    if alt_text:
                        raw = alt_raw
                        orientation_hint = alt_orientation
                        orientation_meta = dict(orientation_meta or {})
                        orientation_meta["fallback_orientation_retry"] = True
            else:
                raw = backend.recognize(crop, image_path=None)
            if not raw and getattr(backend, "_all_unavailable_reason", ""):
                raise RuntimeError("Apple OCR 后端不可用（不是页面无文字）：" + backend._all_unavailable_reason)
            text, norm_diag = normalize_region_ocr_text(raw, orientation_hint)
            if not text:
                bubble.block_ids = []
                continue

            layout_profile = source_layout_profile(layout_crop, text, orientation_hint)
            confs = [float(b.confidence) for b in raw if str(b.text).strip()]
            raw_backend = str((raw[0].meta or {}).get("backend", "") if raw else "").lower()
            conf = (
                float(np.median(confs)) if confs and raw_backend == "tesseract"
                else float(np.mean(confs)) if confs
                else float(
                    getattr(self.config.ocr, "apple_live_text_assumed_confidence", 0.88)
                )
            )
            # Cloud/headless Tesseract is a safety fallback, not semantic
            # authority. Large artwork fragments occasionally look like one or
            # two CJK glyphs. Reject those low-information transcripts before
            # they can create a huge Hybrid/Reletter clear region. Real short
            # dialogue remains allowed when recognition confidence is strong.
            if raw_backend == "tesseract":
                cjk_count = sum(
                    1 for ch in str(text)
                    if "\u3040" <= ch <= "\u30ff" or "\u3400" <= ch <= "\u9fff"
                )
                reject_fallback = (
                    cjk_count <= 0
                    or (cjk_count == 1 and conf < 0.82)
                    or (2 <= cjk_count <= 5 and conf < 0.35)
                    or (cjk_count >= 6 and conf < 0.25)
                )
                if reject_fallback:
                    bubble.block_ids = []
                    stats["ocr_source_fallback_quality_rejected"] = int(
                        stats.get("ocr_source_fallback_quality_rejected", 0) or 0
                    ) + 1
                    continue
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
                "ocr_normalization": dict(norm_diag),
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
