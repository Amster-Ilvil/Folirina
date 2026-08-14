from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig, RegistrationConfig
from manga_hd_transfer.direct_containers import build_source_direct_container_plan
from manga_hd_transfer.models import BubbleInstance, RegistrationResult
from manga_hd_transfer.registration import register_images


def _reg(h: int, w: int) -> RegistrationResult:
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float64), method="test-identity", confidence=0.99,
        inlier_ratio=1.0, reprojection_error=0.0, spatial_coverage=1.0, num_matches=100,
        source_size=(w, h), target_size=(w, h), diagnostics={},
    )


def _star(cx: int, cy: int, r1: int, r2: int, n: int = 18) -> np.ndarray:
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        r = r1 if i % 2 == 0 else r2
        pts.append([int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))])
    return np.asarray(pts, np.int32)


def test_colored_source_direct_preserves_target_fill_and_accepts_source_only_hint():
    h, w = 640, 480
    source = np.full((h, w, 3), 255, np.uint8)
    target = np.full((h, w, 3), 245, np.uint8)
    star = _star(320, 430, 60, 40)
    cv2.fillPoly(source, [star], (255, 255, 255))
    cv2.polylines(source, [star], True, (0, 0, 0), 3)
    cv2.putText(source, "CN", (298, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.fillPoly(target, [star], (0, 230, 255))
    cv2.polylines(target, [star], True, (0, 0, 0), 3)
    cv2.putText(target, "JP", (298, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    hint_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hint_mask, [star], 255)
    hint = BubbleInstance(id="hint", polygon=[tuple(map(float, p)) for p in star.tolist()], mask=hint_mask)
    plan = build_source_direct_container_plan(
        source, target, _reg(h, w), MaskReplaceConfig(), source_hint_bubbles=[hint]
    )
    assert plan is not None
    assert plan.result.applied_count == 1
    assert plan.diagnostics["source_detector_hint_count"] == 1
    assert plan.diagnostics["accepted_colored_spiky"] == 1
    assert plan.diagnostics["border_pixels_written"] == 0
    # A fill-only point well away from glyphs keeps the target's saturation.
    y, x = 465, 320
    before = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[y, x, 1]
    after = cv2.cvtColor(plan.result.image, cv2.COLOR_BGR2HSV)[y, x, 1]
    assert int(before) > 150
    assert abs(int(after) - int(before)) <= 8


def test_quick_registration_structure_refine_is_strictly_residual():
    h, w = 760, 560
    src = np.full((h, w, 3), 255, np.uint8)
    # Stable manga-like geometry spread over the page.
    cv2.rectangle(src, (25, 25), (535, 735), (0, 0, 0), 3)
    cv2.line(src, (30, 250), (530, 250), (0, 0, 0), 2)
    cv2.line(src, (280, 30), (280, 730), (50, 50, 50), 2)
    for cx, cy in [(120, 130), (420, 120), (135, 470), (420, 530)]:
        cv2.circle(src, (cx, cy), 45, (30, 30, 30), 3)
        cv2.line(src, (cx - 30, cy), (cx + 30, cy + 16), (80, 80, 80), 2)
    cv2.putText(src, "CN", (90, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    a = cv2.getRotationMatrix2D((w / 2, h / 2), 0.45, 0.82)
    a[:, 2] += (17.0, 22.0)
    dst = cv2.warpAffine(src, a, (490, 650), flags=cv2.INTER_LINEAR, borderValue=(250, 245, 230))
    # Palette/text rendition differences should not dominate structure refinement.
    hsv = cv2.cvtColor(dst, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.maximum(hsv[..., 1], 25)
    dst = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    cv2.putText(dst, "JP", (80, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    reg = register_images(src, dst, RegistrationConfig(structure_refine_enabled=True, structure_refine_max_side=700))
    assert reg.confidence >= 0.72
    if reg.diagnostics.get("structure_refine_applied"):
        assert reg.diagnostics["structure_corr_after"] >= reg.diagnostics["structure_corr_before"] + 0.001 - 1e-7
        assert abs(reg.diagnostics["structure_residual_dx"]) <= 3.0
        assert abs(reg.diagnostics["structure_residual_dy"]) <= 3.0
        assert abs(reg.diagnostics["structure_residual_rotation_deg"]) <= 0.35
