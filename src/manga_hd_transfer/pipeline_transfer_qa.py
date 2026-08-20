from __future__ import annotations

"""QA-only policies for transfer execution.

No renderer calls live here.  The module consumes completed transfer state and
adds semantic/evidence QA without being able to mutate pixels.
"""

import cv2
import numpy as np

from .models import QAItem


def _mask_fraction(mask: np.ndarray, other: np.ndarray) -> float:
    if mask is None or other is None or mask.shape != other.shape:
        return 0.0
    area = max(1, int(cv2.countNonZero(mask)))
    return float(np.count_nonzero((mask > 0) & (other > 0)) / area)


def _rect_mask(shape: tuple[int, int], box) -> np.ndarray:
    out = np.zeros(shape, np.uint8)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return out
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    h, w = shape
    x0 = max(0, min(w, x0)); x1 = max(0, min(w, x1))
    y0 = max(0, min(h, y0)); y1 = max(0, min(h, y1))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def append_semantic_coverage_qa(
    qa: list[QAItem], *, evidence, mask_transfer, shape: tuple[int, int], cfg,
    stats: dict | None = None,
) -> None:
    """Audit whether strong semantic regions actually reached a renderer."""
    if evidence is None or not bool(getattr(evidence, "available", False)):
        return
    rows = list(getattr(evidence, "items", []) or [])
    if not rows:
        return
    bubble_min = float(getattr(cfg, "koharu_semantic_bubble_min_confidence", 0.70))
    text_min = float(getattr(cfg, "koharu_semantic_text_min_confidence", 0.75))
    include_sfx = bool(getattr(cfg, "koharu_semantic_include_sfx", False))
    applied = [r for r in list(getattr(mask_transfer, "records", []) or []) if bool(getattr(r, "applied", False))]
    handled = np.zeros(shape, np.uint8)
    for rec in applied:
        handled = np.maximum(handled, _rect_mask(shape, getattr(rec, "target_bbox", None)))
    composite = getattr(mask_transfer, "composite_mask", None)
    handled_pixels = composite if isinstance(composite, np.ndarray) and composite.shape == shape else np.zeros(shape, np.uint8)

    bubbles = [
        r for r in rows
        if str(getattr(r, "label", "")) == "bubble"
        and float(getattr(r, "confidence", 0.0)) >= bubble_min
    ]
    texts = [
        r for r in rows
        if str(getattr(r, "label", "")) == "text"
        and float(getattr(r, "confidence", 0.0)) >= text_min
    ]
    if include_sfx:
        texts += [
            r for r in rows
            if str(getattr(r, "label", "")) == "sfx"
            and float(getattr(r, "confidence", 0.0)) >= text_min
        ]

    uncovered_bubbles = 0
    uncovered_text = 0
    bubble_masks: list[np.ndarray] = []
    for row in bubbles:
        mask = getattr(row, "mask", None)
        if not isinstance(mask, np.ndarray) or mask.shape != shape or cv2.countNonZero(mask) <= 0:
            continue
        bubble_masks.append(mask)
        child = any(
            isinstance(getattr(tr, "mask", None), np.ndarray)
            and getattr(tr, "mask").shape == shape
            and _mask_fraction(getattr(tr, "mask"), mask) >= 0.35
            for tr in texts
        )
        if not child:
            continue
        record_coverage = _mask_fraction(mask, handled)
        pixel_coverage = _mask_fraction(mask, handled_pixels)
        if max(record_coverage, pixel_coverage) >= 0.08:
            continue
        uncovered_bubbles += 1
        conf = float(getattr(row, "confidence", 0.0))
        qa.append(QAItem(
            "koharu_semantic_uncovered_bubble", "error" if conf >= 0.85 else "warning",
            "Koharu detected a strong text-bearing bubble/open container, but no automatic transfer record covered it. The page is semantically incomplete.",
            unit_id=f"koharu-bubble-{(getattr(row, 'meta', {}) or {}).get('item_index', '?')}",
            value=conf, threshold=bubble_min,
            meta={
                "bbox": list(getattr(row, "box", ())),
                "record_coverage": round(record_coverage, 4),
                "pixel_coverage": round(pixel_coverage, 4),
                "verification_scope": "koharu_semantic_coverage",
            },
        ))

    for row in texts:
        mask = getattr(row, "mask", None)
        if not isinstance(mask, np.ndarray) or mask.shape != shape or cv2.countNonZero(mask) <= 0:
            continue
        if any(_mask_fraction(mask, bm) >= 0.55 for bm in bubble_masks):
            continue
        record_coverage = _mask_fraction(mask, handled)
        pixel_coverage = _mask_fraction(mask, handled_pixels)
        if max(record_coverage, pixel_coverage) >= 0.12:
            continue
        uncovered_text += 1
        conf = float(getattr(row, "confidence", 0.0))
        qa.append(QAItem(
            "koharu_semantic_uncovered_text", "warning",
            "Koharu detected high-confidence standalone/open text that was not covered by any automatic transfer. Review or recover this region before export.",
            unit_id=f"koharu-{getattr(row, 'label', 'text')}-{(getattr(row, 'meta', {}) or {}).get('item_index', '?')}",
            value=conf, threshold=text_min,
            meta={
                "bbox": list(getattr(row, "box", ())),
                "record_coverage": round(record_coverage, 4),
                "pixel_coverage": round(pixel_coverage, 4),
                "verification_scope": "koharu_semantic_coverage",
            },
        ))
    if stats is not None:
        stats["koharu_semantic_qa_uncovered_bubbles"] = str(uncovered_bubbles)
        stats["koharu_semantic_qa_uncovered_text"] = str(uncovered_text)


def append_photo_pair_evidence_qa(
    qa: list[QAItem], *, paired_diff, config, mode: str, mode_contract,
    registration, source_blocks, target_blocks, mask_transfer,
) -> None:
    """Append photo-pair OCR scope QA after rendering has finished."""
    if not (
        paired_diff is not None
        and paired_diff.method == "photo_pair"
        and config.mask_replace.photo_pair_require_ocr_evidence
        and not source_blocks and not target_blocks
    ):
        return

    photo_records = list(mask_transfer.records) if mask_transfer is not None else []
    photo_only = [
        r for r in photo_records
        if getattr(r, "geometry_mode", "") in {"photo_pair", "rigid_uniform_container"}
        or getattr(r, "sr_backend", "") == "rigid-container-raster"
    ]
    content_verified = bool(
        photo_records
        and all(
            (not bool(getattr(r, "applied", False)))
            or (
                str(getattr(r, "content_check", "")).startswith("checked")
                and bool(getattr(r, "content_complete", False))
            )
            for r in photo_records
        )
    )
    fully_applied_photo = bool(
        photo_only
        and all(r.applied for r in photo_only)
        and registration.confidence >= 0.78
    )
    explicit_visual_mask = bool(mode == "mask_replace" and not mode_contract.may_use_ocr)
    if explicit_visual_mask:
        qa.append(QAItem(
            "photo_pair_visual_only_contract", "warning",
            (
                "Precise Mask is running under the explicit zero-OCR contract. "
                "Detected photographed-page regions were judged only by visual/raster evidence; "
                "review for completely undiscovered open/SFX text."
            ),
            meta={
                "detected_regions": len(photo_records),
                "content_verified_detected_regions": content_verified,
                "verification_scope": "visual_detected_regions_only",
                "registration_confidence": registration.confidence,
                "ocr_intentionally_disabled": True,
            },
        ))
    else:
        qa.append(QAItem(
            "photo_pair_ocr_evidence_missing", "warning" if fully_applied_photo else "error",
            (
                "All detected photographed-page regions passed the independent raster-content check under strong registration. OCR is unavailable, so this verifies detected regions only; review the page for any entirely undiscovered open/SFX text."
                if fully_applied_photo else
                "Photographed-edition extraction is conservative and OCR evidence is unavailable; at least one detected region is not independently content-verified, or page registration is weak."
            ),
            meta={
                "detected_regions": len(photo_records),
                "content_verified_detected_regions": content_verified,
                "verification_scope": "detected_regions_only",
                "registration_confidence": registration.confidence,
            },
        ))


# Compatibility name retained for older imports.
_append_koharu_semantic_coverage_qa = append_semantic_coverage_qa

__all__ = [
    "append_semantic_coverage_qa",
    "append_photo_pair_evidence_qa",
    "_append_koharu_semantic_coverage_qa",
]
