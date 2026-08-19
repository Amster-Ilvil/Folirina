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
