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




def _box_overlap_over_min(a, b) -> float:
    if not isinstance(a, (list, tuple)) or len(a) != 4 or not isinstance(b, (list, tuple)) or len(b) != 4:
        return 0.0
    ax0, ay0, ax1, ay1 = [float(v) for v in a]
    bx0, by0, bx1, by1 = [float(v) for v in b]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    aa = max(1.0, max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0))
    ba = max(1.0, max(0.0, bx1 - bx0) * max(0.0, by1 - by0))
    return float(inter / max(1.0, min(aa, ba)))


def _project_source_item_box(item, registration) -> tuple[float, float, float, float] | None:
    polygon = list(getattr(item, "polygon", []) or [])
    if len(polygon) < 3 or registration is None:
        return None
    try:
        matrix = np.asarray(getattr(registration, "matrix", None), dtype=np.float64)
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        if matrix.shape == (3, 3):
            mapped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
        elif matrix.shape == (2, 3):
            mapped = cv2.transform(pts, matrix).reshape(-1, 2)
        else:
            return None
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            return None
        x0, y0 = mapped.min(axis=0); x1, y1 = mapped.max(axis=0)
        return float(x0), float(y0), float(x1), float(y1)
    except Exception:
        return None


def _has_projected_source_text_support(target_row, source_evidence, registration, *, min_overlap: float = 0.18) -> bool | None:
    """Return whether SOURCE has semantic text at the same registered location.

    ``None`` means the evidence is unavailable, in which case QA keeps the old
    conservative behaviour.  ``False`` is meaningful only when SOURCE Koharu
    evidence is available and no projected text/SFX item supports the TARGET
    standalone text.
    """
    if source_evidence is None or not bool(getattr(source_evidence, "available", False)) or registration is None:
        return None
    tbox = getattr(target_row, "box", None)
    if not isinstance(tbox, (list, tuple)) or len(tbox) != 4:
        return None
    tmeta = getattr(target_row, "meta", {}) or {}
    traw = tmeta.get("raw_box") or tbox
    for src in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(src, "label", "")).strip().lower() not in {"text", "sfx", "onomatopoeia"}:
            continue
        if float(getattr(src, "confidence", 0.0)) < 0.20:
            continue
        # First use the post-processed polygon box.  Koharu can heavily
        # under-segment a long vertical/open-text run, though, so a narrow
        # polygon must not erase strong same-location evidence carried by the
        # detector's raw box (Vol.04 p0023 class).
        pbox = _project_source_item_box(src, registration)
        if pbox is not None and _box_overlap_over_min(tbox, pbox) >= float(min_overlap):
            return True
        smeta = getattr(src, "meta", {}) or {}
        sraw = smeta.get("raw_box") or getattr(src, "box", None)
        if not isinstance(sraw, (list, tuple)) or len(sraw) != 4:
            continue
        try:
            matrix = np.asarray(getattr(registration, "matrix", None), dtype=np.float64)
            pts = np.asarray([
                [float(sraw[0]), float(sraw[1])], [float(sraw[2]), float(sraw[1])],
                [float(sraw[2]), float(sraw[3])], [float(sraw[0]), float(sraw[3])],
            ], dtype=np.float32).reshape(-1, 1, 2)
            if matrix.shape == (3, 3):
                mapped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
            elif matrix.shape == (2, 3):
                mapped = cv2.transform(pts, matrix).reshape(-1, 2)
            else:
                continue
            projected_raw = [
                float(mapped[:, 0].min()), float(mapped[:, 1].min()),
                float(mapped[:, 0].max()), float(mapped[:, 1].max()),
            ]
            if _box_overlap_over_min(traw, projected_raw) >= max(float(min_overlap), 0.60):
                return True
        except Exception:
            continue
    return False


def _has_projected_source_dialogue_support(target_row, source_evidence, registration, *, min_overlap: float = 0.18) -> bool | None:
    """Return whether SOURCE has bubble/text evidence at the registered TARGET container.

    A TARGET-only decorative/SFX container can be detected as ``bubble+text`` by
    Koharu even when the translated SOURCE intentionally leaves that shape blank.
    In that case preserving TARGET is correct and semantic QA must not report a
    missed translation.  Keep the old conservative behaviour when SOURCE evidence
    or registration is unavailable/weak.
    """
    if source_evidence is None or not bool(getattr(source_evidence, "available", False)) or registration is None:
        return None
    try:
        if float(getattr(registration, "confidence", 0.0) or 0.0) < 0.60:
            return None
    except Exception:
        return None
    tbox = getattr(target_row, "box", None)
    if not isinstance(tbox, (list, tuple)) or len(tbox) != 4:
        return None
    for src in list(getattr(source_evidence, "items", []) or []):
        label = str(getattr(src, "label", "")).strip().lower()
        if label not in {"bubble", "text"}:
            continue
        if float(getattr(src, "confidence", 0.0)) < 0.20:
            continue
        pbox = _project_source_item_box(src, registration)
        if pbox is not None and _box_overlap_over_min(tbox, pbox) >= float(min_overlap):
            return True
    return False


def _registered_text_raster_equivalent(
    target_row, source_evidence, registration, *, source_image=None, target_image=None,
    min_raw_overlap: float = 0.80, min_corr: float = 0.55, min_dark_f1: float = 0.82,
) -> bool:
    """Conservatively prove that registered SOURCE/TARGET scene text is unchanged.

    This is intentionally local and bilateral. It suppresses only a strong uncovered
    TARGET text item whose best registered SOURCE text has a strongly overlapping
    raw box *and* essentially the same local raster. It prevents immutable scene
    signage (Vol.04 p0021 class) from being escalated as a missing translation while
    still blocking true translated text such as p0023.
    """
    if source_image is None or target_image is None or registration is None:
        return False
    if not isinstance(source_image, np.ndarray) or not isinstance(target_image, np.ndarray):
        return False
    if source_image.ndim != 3 or target_image.ndim != 3:
        return False
    if source_evidence is None or not bool(getattr(source_evidence, "available", False)):
        return False
    try:
        if float(getattr(registration, "confidence", 0.0) or 0.0) < 0.80:
            return False
        matrix=np.asarray(getattr(registration, "matrix", None), dtype=np.float64)
        if matrix.shape != (3,3):
            return False
    except Exception:
        return False
    tmeta=getattr(target_row, "meta", {}) or {}
    traw=list(tmeta.get("raw_box") or getattr(target_row, "box", ()) or ())
    if len(traw) != 4:
        return False
    best=None
    for src in list(getattr(source_evidence, "items", []) or []):
        if str(getattr(src, "label", "")).strip().lower() != "text":
            continue
        smeta=getattr(src, "meta", {}) or {}
        sraw=list(smeta.get("raw_box") or getattr(src, "box", ()) or ())
        if len(sraw) != 4:
            continue
        pts=np.asarray([[float(sraw[0]),float(sraw[1])],[float(sraw[2]),float(sraw[1])],
                        [float(sraw[2]),float(sraw[3])],[float(sraw[0]),float(sraw[3])]],
                       np.float32).reshape(-1,1,2)
        mapped=cv2.perspectiveTransform(pts,matrix).reshape(-1,2)
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            continue
        sp=[float(mapped[:,0].min()),float(mapped[:,1].min()),float(mapped[:,0].max()),float(mapped[:,1].max())]
        overlap=_box_overlap_over_min(traw,sp)
        if best is None or overlap > best[0]:
            best=(overlap,sp)
    if best is None or float(best[0]) < float(min_raw_overlap):
        return False
    sp=best[1]
    h,w=target_image.shape[:2]
    x0=max(0,int(np.floor(min(float(traw[0]),sp[0])))-4); y0=max(0,int(np.floor(min(float(traw[1]),sp[1])))-4)
    x1=min(w,int(np.ceil(max(float(traw[2]),sp[2])))+4); y1=min(h,int(np.ceil(max(float(traw[3]),sp[3])))+4)
    if x1-x0 < 6 or y1-y0 < 6:
        return False
    aligned=cv2.warpPerspective(source_image,matrix,(w,h),flags=cv2.INTER_LINEAR,borderValue=(255,255,255))
    ag=cv2.cvtColor(aligned[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg=cv2.cvtColor(target_image[y0:y1,x0:x1],cv2.COLOR_BGR2GRAY).astype(np.float32)
    if ag.size < 64 or bg.size != ag.size:
        return False
    # Remove local background/illumination so cross-rendition gray levels do not
    # dominate the comparison; compare only local stroke structure.
    ah=ag-cv2.GaussianBlur(ag,(0,0),4)
    bh=bg-cv2.GaussianBlur(bg,(0,0),4)
    sa=float(ah.std()); sb=float(bh.std())
    if sa < 1.0 or sb < 1.0:
        return False
    corr=float(np.mean(((ah-float(ah.mean()))/sa)*((bh-float(bh.mean()))/sb)))
    ad=ah < -18.0; bd=bh < -18.0
    ac=int(np.count_nonzero(ad)); bc=int(np.count_nonzero(bd))
    if min(ac,bc) < 20:
        return False
    inter=int(np.count_nonzero(ad & bd))
    dark_f1=float(2.0*inter/max(1,ac+bc))
    dark_ratio=float(min(ac,bc)/max(1,max(ac,bc)))
    return bool(corr >= float(min_corr) and dark_f1 >= float(min_dark_f1) and dark_ratio >= 0.75)


def _is_micro_standalone_text(row, shape: tuple[int, int]) -> bool:
    box = getattr(row, "box", None)
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    h, w = shape
    x0, y0, x1, y1 = [float(v) for v in box]
    bw=max(0.0,x1-x0); bh=max(0.0,y1-y0)
    area_ratio=(bw*bh)/max(1.0,float(w*h))
    thin_ratio=min(bw/max(1.0,float(w)), bh/max(1.0,float(h)))
    return bool(area_ratio <= 0.00065 or thin_ratio <= 0.012)

def append_semantic_coverage_qa(
    qa: list[QAItem], *, evidence, mask_transfer, shape: tuple[int, int], cfg,
    stats: dict | None = None, source_evidence=None, registration=None,
    source_image=None, target_image=None,
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
    suppressed_target_only_bubbles = 0
    downgraded_target_only_rejections = 0
    suppressed_target_only_micro_text = 0
    suppressed_equivalent_scene_text = 0
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
        source_support = _has_projected_source_dialogue_support(
            row, source_evidence, registration, min_overlap=0.18,
        )
        if source_support is False:
            suppressed_target_only_bubbles += 1
            # Semantic completion may already have created a synthetic Direct/Mask
            # candidate for this TARGET-only container.  When the registered
            # translation SOURCE has no bubble/text support here and the candidate
            # failed specifically because there is no SOURCE ink, preserving TARGET
            # is intentional, not a publication-blocking rejection.  Downgrade only
            # the matching synthetic semantic record; ordinary renderer rejections
            # remain untouched.
            rid = str((getattr(row, "meta", {}) or {}).get("item_index", "?"))
            target_token = f"koharu-semantic-dst-bubble-{rid}"
            for issue in qa:
                meta = getattr(issue, "meta", {}) or {}
                if (
                    str(getattr(issue, "code", "")).endswith("_rejected")
                    and str(meta.get("target") or "") == target_token
                    and str(meta.get("reason") or "") == "insufficient_source_ink"
                ):
                    issue.code = str(getattr(issue, "code", "")).replace("_rejected", "_target_only_preserved")
                    issue.severity = "info"
                    issue.message = (
                        "TARGET-only semantic container has no registered SOURCE dialogue support; "
                        "preserving the original TARGET region is intentional."
                    )
                    meta["source_support"] = False
                    meta["verification_scope"] = "registered_source_semantic_support"
                    issue.meta = meta
                    downgraded_target_only_rejections += 1
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
        # Tiny standalone TARGET text is often immutable print on books, signs,
        # appliances or other scene props.  Do not call it a missed translation
        # merely because Koharu can read its geometry.  If the aligned SOURCE
        # contains semantic text at the same location, keep the warning; if the
        # SOURCE has no such text, the translation edition itself chose not to
        # replace it, so TARGET remains authoritative.  Larger standalone text
        # stays conservative even when SOURCE detection is weak.
        source_support = _has_projected_source_text_support(
            row, source_evidence, registration, min_overlap=0.18,
        )
        if source_support is False and _is_micro_standalone_text(row, shape):
            suppressed_target_only_micro_text += 1
            continue
        if source_support is True and _registered_text_raster_equivalent(
            row, source_evidence, registration, source_image=source_image, target_image=target_image,
        ):
            suppressed_equivalent_scene_text += 1
            continue
        uncovered_text += 1
        conf = float(getattr(row, "confidence", 0.0))
        # Strong TARGET text with registered SOURCE text at the same location is
        # not optional scene decoration: the translated edition contains content
        # here and an uncovered region means the automatic transfer is incomplete.
        # Escalate only this bilateral case; target-only/unknown open text keeps
        # the conservative warning behaviour.
        error_floor = float(getattr(cfg, "koharu_semantic_uncovered_text_error_confidence", 0.85))
        severity = "error" if source_support is True and conf >= error_floor else "warning"
        qa.append(QAItem(
            "koharu_semantic_uncovered_text", severity,
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
        stats["koharu_semantic_qa_suppressed_target_only_bubbles"] = str(suppressed_target_only_bubbles)
        stats["koharu_semantic_qa_downgraded_target_only_rejections"] = str(downgraded_target_only_rejections)
        stats["koharu_semantic_qa_suppressed_target_only_micro_text"] = str(suppressed_target_only_micro_text)
        stats["koharu_semantic_qa_suppressed_equivalent_scene_text"] = str(suppressed_equivalent_scene_text)


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
