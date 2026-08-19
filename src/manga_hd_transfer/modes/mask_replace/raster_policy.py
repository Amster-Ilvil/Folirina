from __future__ import annotations

import cv2
import numpy as np

from ...geometry import transform_to_homography


def global_registered_raster(source: np.ndarray, target_shape: tuple[int, int], registration) -> tuple[np.ndarray, dict]:
    """Return SOURCE in TARGET coordinates without local/dense deformation.

    Mask Replace may use dense/structural alignment to *discover/refine masks*, but
    final Chinese glyph pixels must come from a shape-preserving page transform.
    This prevents CJK strokes from bending under optical-flow remapping.
    """
    h, w = target_shape
    H = transform_to_homography(registration.matrix)
    raster = cv2.warpPerspective(
        source,
        H,
        (w, h),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    # Report affine shape properties for QA.  The page registration may include
    # small anisotropy; the important guard here is that no *additional* dense
    # deformation is applied to the glyph raster after registration.
    A = H[:2, :2].astype(np.float64)
    sx = float(np.linalg.norm(A[:, 0]))
    sy = float(np.linalg.norm(A[:, 1]))
    det = float(np.linalg.det(A))
    return raster, {
        "policy": "global_registration_raster_only",
        "mask_replace_glyph_integrity_contract": True,
        "dense_flow_geometry_only": True,
        "glyph_dense_warp": False,
        "page_scale_x": sx,
        "page_scale_y": sy,
        "page_scale_ratio": (sx / sy) if sy > 1e-9 else 1.0,
        "orientation_preserved": bool(det > 0),
    }


def paired_proxy_geometry_risk(source_bbox: tuple[int, int, int, int] | None,
                               target_bbox: tuple[int, int, int, int] | None,
                               paired_mask_iou: float) -> dict:
    """Diagnose when a paired-diff SOURCE mask is only a bookkeeping proxy.

    A tiny source proxy must never be stretched to fill a large target balloon.
    The raster policy therefore treats such source masks as detection metadata,
    not as a final raster transform authority.
    """
    if not source_bbox or not target_bbox:
        return {"risky": True, "reason": "missing_bbox", "paired_mask_iou": float(paired_mask_iou)}
    sx0, sy0, sx1, sy1 = source_bbox
    tx0, ty0, tx1, ty1 = target_bbox
    sa = max(1, (sx1 - sx0) * (sy1 - sy0))
    ta = max(1, (tx1 - tx0) * (ty1 - ty0))
    area_ratio = float(sa / ta)
    risky = bool(float(paired_mask_iou) < 0.08 or area_ratio < 0.02)
    return {
        "risky": risky,
        "reason": "tiny_or_low_iou_source_proxy" if risky else "ok",
        "paired_mask_iou": float(paired_mask_iou),
        "source_proxy_area": int(sa),
        "target_area": int(ta),
        "raw_area_ratio": area_ratio,
    }
