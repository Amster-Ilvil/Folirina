from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig, RegistrationConfig
from manga_hd_transfer.direct_containers import build_source_direct_container_plan
from manga_hd_transfer.models import RegistrationResult
from manga_hd_transfer.registration import register_images
from .helpers import make_art_page


def _star(center: tuple[int, int], r1: int, r2: int, n: int = 18) -> np.ndarray:
    cx, cy = center
    pts = []
    for i in range(n):
        a = 2.0 * np.pi * i / n
        r = r1 if i % 2 == 0 else r2
        pts.append([int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))])
    return np.asarray(pts, np.int32)


def test_auto_registration_quick_handles_different_size_crop_and_small_rotation():
    src = make_art_page(900, 1200)
    # Different output canvas, uniform scale + small rotation + crop/translation.
    center = (src.shape[1] * 0.5, src.shape[0] * 0.5)
    a = cv2.getRotationMatrix2D(center, 0.8, 0.79)
    a[:, 2] += (22.0, 31.0)
    dst = cv2.warpAffine(src, a, (760, 980), flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))

    reg = register_images(src, dst, RegistrationConfig(backend="auto", feature="sift"))
    assert reg.confidence >= 0.72
    assert reg.num_matches >= 10
    assert reg.spatial_coverage >= 0.18
    assert reg.diagnostics.get("route") in {"opencv_quick", "opencv", "fast_identity"}

    pts = np.asarray([[[110, 120], [650, 260], [430, 850]]], np.float32)
    expected = cv2.transform(pts, a)[0]
    actual = cv2.perspectiveTransform(pts, reg.matrix)[0]
    err = np.median(np.linalg.norm(expected - actual, axis=1))
    assert err < 7.0


def test_source_direct_affine_location_never_affine_warps_final_text_raster():
    h, w = 720, 520
    source = np.full((h, w, 3), 255, np.uint8)
    # Add artwork structure so the page does not look like only two text boxes.
    cv2.rectangle(source, (20, 20), (500, 700), (0, 0, 0), 3)
    cv2.line(source, (30, 330), (490, 330), (0, 0, 0), 2)
    cv2.circle(source, (390, 210), 55, (80, 80, 80), 3)

    cv2.ellipse(source, (145, 180), (58, 43), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(source, "CN", (118, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    star = _star((330, 520), 68, 43)
    cv2.fillPoly(source, [star], (255, 255, 255))
    cv2.polylines(source, [star], True, (0, 0, 0), 3)
    cv2.putText(source, "CN", (302, 530), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Deliberately anisotropic page mapping. This is valid for *location* only.
    H = np.array([
        [0.78, -0.002, 19.0],
        [0.004, 0.82, 24.0],
        [0.0, 0.0, 1.0],
    ], np.float64)
    target = cv2.warpPerspective(source, H, (450, 620), flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))

    # Replace target lettering so a transfer is required. Build target-space masks
    # from the known source shapes, keeping target outlines as the alignment guide.
    ellipse_mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(ellipse_mask, (145, 180), (54, 39), 0, 0, 360, 255, cv2.FILLED)
    em = cv2.warpPerspective(ellipse_mask, H, (450, 620), flags=cv2.INTER_NEAREST)
    target[em > 0] = (255, 255, 255)
    ec = cv2.perspectiveTransform(np.asarray([[[145.0, 180.0]]], np.float32), H)[0, 0]
    cv2.putText(target, "JP", (int(ec[0] - 22), int(ec[1] + 9)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    star_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(star_mask, [star], 255)
    sm = cv2.warpPerspective(star_mask, H, (450, 620), flags=cv2.INTER_NEAREST)
    target[sm > 0] = (0, 235, 255)
    warped_star = cv2.perspectiveTransform(star.astype(np.float32)[None, ...], H)[0].astype(np.int32)
    cv2.polylines(target, [warped_star], True, (0, 0, 0), 3)
    sc = cv2.perspectiveTransform(np.asarray([[[330.0, 520.0]]], np.float32), H)[0, 0]
    cv2.putText(target, "JP", (int(sc[0] - 22), int(sc[1] + 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    reg = RegistrationResult(
        matrix=H,
        method="test-affine",
        confidence=0.99,
        inlier_ratio=1.0,
        reprojection_error=0.5,
        spatial_coverage=0.9,
        num_matches=100,
        source_size=(w, h),
        target_size=(450, 620),
        diagnostics={},
    )
    cfg = MaskReplaceConfig(
        source_direct_min_registration_confidence=0.8,
        source_direct_max_axis_scale_delta=0.2,
        source_direct_max_local_mapping_anisotropy=0.2,
    )
    plan = build_source_direct_container_plan(source, target, reg, cfg)
    assert plan is not None
    assert plan.result.applied_count >= 2
    assert plan.diagnostics["auto_alignment_mode"] == "A2_affine_location_local_similarity_raster"
    assert plan.diagnostics["shape_preserving_raster"] is True
    assert plan.diagnostics["final_raster_transform"] == "local_similarity_only"
    assert plan.diagnostics["border_pixels_written"] == 0
    assert plan.diagnostics["max_local_mapping_anisotropy"] > 0.02
    assert all("local_rotation_deg" in b.meta for b in plan.target_bubbles)
