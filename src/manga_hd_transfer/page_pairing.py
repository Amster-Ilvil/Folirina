from __future__ import annotations

"""OCR-free verification that a SOURCE/TARGET pair depicts the same manga page.

Filename/order pairing is intentionally not enough for destructive transfer.  This
module verifies the already-registered pair from structure only.  Text is
suppressed by heavy blur/downsampling so Japanese/Chinese glyph differences carry
far less weight than panel borders, faces, clothing, and background line art.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import transform_to_homography
from .models import RegistrationResult


@dataclass(slots=True)
class PagePairingCheck:
    same_page: bool
    confidence: float
    structure_score: float
    edge_score: float
    registration_score: float
    diagnostics: dict

    def to_dict(self) -> dict:
        return {
            "same_page": bool(self.same_page),
            "confidence": float(self.confidence),
            "structure_score": float(self.structure_score),
            "edge_score": float(self.edge_score),
            "registration_score": float(self.registration_score),
            "diagnostics": dict(self.diagnostics),
        }


def _resize_pair(a: np.ndarray, b: np.ndarray, max_side: int) -> tuple[np.ndarray, np.ndarray, float]:
    h, w = b.shape[:2]
    scale = min(1.0, float(max_side) / max(1.0, float(max(h, w))))
    if scale >= 0.999:
        return a, b, 1.0
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return (
        cv2.resize(a, size, interpolation=cv2.INTER_AREA),
        cv2.resize(b, size, interpolation=cv2.INTER_AREA),
        scale,
    )


def _robust_corr(a: np.ndarray, b: np.ndarray, valid: np.ndarray) -> float:
    use = valid > 0
    if int(np.count_nonzero(use)) < 256:
        return 0.0
    av = a[use].astype(np.float32)
    bv = b[use].astype(np.float32)
    av -= float(np.mean(av)); bv -= float(np.mean(bv))
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(float(np.dot(av, bv)) / denom, -1.0, 1.0) * 0.5 + 0.5)



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


def _project_item_box(item, registration: RegistrationResult):
    polygon = list(getattr(item, "polygon", []) or [])
    if len(polygon) < 3:
        return None
    try:
        pts = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        H = transform_to_homography(registration.matrix)
        mapped = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        if mapped.size == 0 or not np.all(np.isfinite(mapped)):
            return None
        x0, y0 = mapped.min(axis=0); x1, y1 = mapped.max(axis=0)
        return (float(x0), float(y0), float(x1), float(y1))
    except Exception:
        return None


def _rescue_pairing_with_registered_text_anchor(
    check: PagePairingCheck,
    registration: RegistrationResult,
    source_evidence,
    target_evidence,
) -> PagePairingCheck:
    """Rescue a colour/greyscale borderline pair from one large registered text anchor.

    Some full-colour TARGET pages expose a stylized/open caption as one large
    Koharu text region while the greyscale translated SOURCE detects the same
    caption as a tighter text region and no reliable bubbles.  Bubble-only rescue
    therefore cannot fire (p-115).  A single text anchor is accepted only under
    much stronger page-level registration/structure constraints and near-exact
    registered placement; tiny header/page-number text is explicitly excluded.
    """
    if float(check.confidence) < 0.65:
        return check
    if float(check.structure_score) < 0.90:
        return check
    if float(getattr(registration, "confidence", 0.0) or 0.0) < 0.82:
        return check
    if float(getattr(registration, "inlier_ratio", 0.0) or 0.0) < 0.75:
        return check
    if float(getattr(registration, "reprojection_error", 999.0) or 999.0) > 1.10:
        return check
    if float(getattr(registration, "spatial_coverage", 0.0) or 0.0) < 0.50:
        return check

    src = [
        x for x in list(getattr(source_evidence, "items", []) or [])
        if str(getattr(x, "label", "")) == "text"
        and float(getattr(x, "confidence", 0.0) or 0.0) >= 0.50
    ]
    tgt = [
        x for x in list(getattr(target_evidence, "items", []) or [])
        if str(getattr(x, "label", "")) == "text"
        and float(getattr(x, "confidence", 0.0) or 0.0) >= 0.30
    ]
    if not src or not tgt:
        return check

    target_size = tuple(getattr(registration, "target_size", ()) or ())
    if len(target_size) == 2:
        page_area = max(1.0, float(target_size[0]) * float(target_size[1]))
    else:
        page_area = 1.0
    best = None
    for row in src:
        pbox = _project_item_box(row, registration)
        if pbox is None:
            continue
        px0, py0, px1, py1 = pbox
        parea = max(1.0, (px1 - px0) * (py1 - py0))
        # Reject page furniture / micro labels: this rescue is for a substantive
        # translated caption, not a page number or tiny header.
        if page_area > 1.0 and parea / page_area < 0.008:
            continue
        pcx = (px0 + px1) * 0.5; pcy = (py0 + py1) * 0.5
        for dst in tgt:
            box = getattr(dst, "box", None)
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue
            tx0, ty0, tx1, ty1 = [float(v) for v in box]
            tarea = max(1.0, (tx1 - tx0) * (ty1 - ty0))
            ratio = max(parea, tarea) / max(1.0, min(parea, tarea))
            if ratio > 3.25:
                continue
            overlap = _box_overlap_over_min(pbox, box)
            if overlap < 0.92:
                continue
            tcx = (tx0 + tx1) * 0.5; tcy = (ty0 + ty1) * 0.5
            diag = max(1.0, ((tx1 - tx0) ** 2 + (ty1 - ty0) ** 2) ** 0.5)
            center_delta = ((pcx - tcx) ** 2 + (pcy - tcy) ** 2) ** 0.5 / diag
            if center_delta > 0.18:
                continue
            score = overlap - 0.15 * center_delta - 0.02 * (ratio - 1.0)
            if best is None or score > best[0]:
                best = (score, overlap, center_delta, ratio, pbox, box)
    if best is None:
        return check

    _, overlap, center_delta, area_ratio, pbox, tbox = best
    rescued_conf = float(max(
        check.confidence,
        min(0.95, 0.47 * float(check.structure_score) + 0.33 * float(registration.confidence) + 0.20 * float(overlap)),
    ))
    diagnostics = dict(check.diagnostics or {})
    diagnostics.update({
        "semantic_layout_rescue": True,
        "semantic_layout_provider": "koharu_layout",
        "semantic_text_anchor_rescue": True,
        "semantic_text_anchor_overlap": round(float(overlap), 4),
        "semantic_text_anchor_center_delta": round(float(center_delta), 4),
        "semantic_text_anchor_area_ratio": round(float(area_ratio), 4),
        "semantic_text_anchor_projected_source_box": [round(float(v), 2) for v in pbox],
        "semantic_text_anchor_target_box": [round(float(v), 2) for v in tbox],
        "pre_rescue_confidence": float(check.confidence),
        "ocr_used": False,
    })
    return PagePairingCheck(
        True, rescued_conf, float(check.structure_score), float(check.edge_score),
        float(check.registration_score), diagnostics,
    )


def refine_pairing_with_layout_evidence(
    check: PagePairingCheck,
    registration: RegistrationResult,
    source_evidence,
    target_evidence,
    *,
    min_registration: float = 0.70,
    min_structure: float = 0.80,
    min_bubble_confidence: float = 0.70,
    min_overlap: float = 0.78,
    min_anchor_count: int = 2,
    min_match_ratio: float = 0.75,
) -> PagePairingCheck:
    """Conservatively rescue a borderline same-page decision from Koharu geometry.

    Colour-vs-greyscale editions can depress the edge term even when registration
    and page structure are strong.  A destructive route may be rescued only when
    *both* Koharu sides independently detect multiple high-confidence bubbles and
    the registered SOURCE bubble geometry overlaps TARGET bubbles nearly exactly.
    This never turns weak registration/structure into a pass and never uses OCR.
    """
    if bool(check.same_page):
        return check
    if float(getattr(registration, "confidence", 0.0) or 0.0) < float(min_registration):
        return check
    if float(check.structure_score) < float(min_structure):
        return check
    if source_evidence is None or target_evidence is None:
        return check
    if not bool(getattr(source_evidence, "available", False)) or not bool(getattr(target_evidence, "available", False)):
        return check
    src = [
        x for x in list(getattr(source_evidence, "items", []) or [])
        if str(getattr(x, "label", "")) == "bubble"
        and float(getattr(x, "confidence", 0.0) or 0.0) >= float(min_bubble_confidence)
    ]
    tgt = [
        x for x in list(getattr(target_evidence, "items", []) or [])
        if str(getattr(x, "label", "")) == "bubble"
        and float(getattr(x, "confidence", 0.0) or 0.0) >= float(min_bubble_confidence)
    ]
    if min(len(src), len(tgt)) < int(min_anchor_count):
        return _rescue_pairing_with_registered_text_anchor(
            check, registration, source_evidence, target_evidence
        )
    available = set(range(len(tgt)))
    overlaps = []
    for row in sorted(src, key=lambda x: float(getattr(x, "confidence", 0.0) or 0.0), reverse=True):
        pbox = _project_item_box(row, registration)
        if pbox is None or not available:
            continue
        best_j = None; best = 0.0
        for j in list(available):
            score = _box_overlap_over_min(pbox, getattr(tgt[j], "box", None))
            if score > best:
                best = score; best_j = j
        if best_j is not None and best >= float(min_overlap):
            available.remove(best_j); overlaps.append(float(best))
    need = max(int(min_anchor_count), int(np.ceil(float(min_match_ratio) * min(len(src), len(tgt)))))
    if len(overlaps) < need:
        return _rescue_pairing_with_registered_text_anchor(
            check, registration, source_evidence, target_evidence
        )
    semantic_score = float(np.mean(overlaps))
    rescued_conf = float(max(
        check.confidence,
        min(0.97, 0.45 * float(check.structure_score) + 0.30 * float(registration.confidence) + 0.25 * semantic_score),
    ))
    diagnostics = dict(check.diagnostics or {})
    diagnostics.update({
        "semantic_layout_rescue": True,
        "semantic_layout_provider": "koharu_layout",
        "semantic_bubble_matches": int(len(overlaps)),
        "semantic_bubble_required": int(need),
        "semantic_bubble_mean_overlap": round(semantic_score, 4),
        "pre_rescue_confidence": float(check.confidence),
        "ocr_used": False,
    })
    return PagePairingCheck(
        True, rescued_conf, float(check.structure_score), float(check.edge_score),
        float(check.registration_score), diagnostics,
    )

def verify_registered_page_pair(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    *,
    max_side: int = 720,
    min_confidence: float = 0.72,
    min_valid_ratio: float = 0.45,
) -> PagePairingCheck:
    """Verify same-page identity after registration without OCR.

    The source is warped only for *verification*.  This transform is never used as
    the final Chinese raster transform.  Low-frequency structure and tolerant edge
    overlap dominate the score; differing language glyphs therefore do not decide
    page identity.
    """
    if source.ndim != 3 or target.ndim != 3:
        return PagePairingCheck(False, 0.0, 0.0, 0.0, float(registration.confidence), {"reason": "invalid_image"})

    th, tw = target.shape[:2]
    H = transform_to_homography(registration.matrix)
    src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    tgt_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    warped = cv2.warpPerspective(src_gray, H, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    valid_src = np.full(source.shape[:2], 255, np.uint8)
    valid = cv2.warpPerspective(valid_src, H, (tw, th), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    warped, tgt_gray, scale = _resize_pair(warped, tgt_gray, max_side)
    if scale != 1.0:
        valid = cv2.resize(valid, (tgt_gray.shape[1], tgt_gray.shape[0]), interpolation=cv2.INTER_NEAREST)

    valid_ratio = float(np.mean(valid > 0))
    if valid_ratio < float(min_valid_ratio):
        return PagePairingCheck(False, 0.0, 0.0, 0.0, float(registration.confidence), {
            "reason": "insufficient_registered_overlap", "valid_ratio": valid_ratio,
        })

    # Heavy blur suppresses translated glyph differences and scan/JPEG grain.
    sigma = max(2.0, max(tgt_gray.shape[:2]) / 180.0)
    sa = cv2.GaussianBlur(warped, (0, 0), sigma)
    ta = cv2.GaussianBlur(tgt_gray, (0, 0), sigma)
    structure = _robust_corr(sa, ta, valid)

    # Tolerant edge overlap emphasizes panel/character/background structure.  A
    # 2px dilation means tiny registration residuals do not become false rejects.
    es = cv2.Canny(sa, 45, 125)
    et = cv2.Canny(ta, 45, 125)
    es[valid == 0] = 0; et[valid == 0] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    esd = cv2.dilate(es, k) > 0
    etd = cv2.dilate(et, k) > 0
    e1 = es > 0; e2 = et > 0
    n1 = int(np.count_nonzero(e1)); n2 = int(np.count_nonzero(e2))
    if n1 + n2:
        matched = int(np.count_nonzero(e1 & etd)) + int(np.count_nonzero(e2 & esd))
        edge = float(np.clip(matched / max(1, n1 + n2), 0.0, 1.0))
    else:
        edge = 0.0

    reg = float(np.clip(registration.confidence, 0.0, 1.0))
    # Geometric mean penalizes a catastrophic disagreement in any one dimension.
    confidence = float(np.clip((max(structure, 1e-6) ** 0.50) * (max(edge, 1e-6) ** 0.30) * (max(reg, 1e-6) ** 0.20), 0.0, 1.0))
    same = bool(confidence >= float(min_confidence) and structure >= 0.58 and edge >= 0.34)
    return PagePairingCheck(same, confidence, structure, edge, reg, {
        "valid_ratio": valid_ratio,
        "max_side": int(max_side),
        "downscale": float(scale),
        "threshold": float(min_confidence),
        "ocr_used": False,
    })
