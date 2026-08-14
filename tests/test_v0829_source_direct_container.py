import cv2
import numpy as np

from manga_hd_transfer.config import MaskReplaceConfig
from manga_hd_transfer.direct_containers import build_source_direct_container_plan
from manga_hd_transfer.models import RegistrationResult


def _reg(shape):
    h, w = shape
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float64), method="test-identity", confidence=0.99,
        inlier_ratio=1.0, reprojection_error=0.0, spatial_coverage=1.0, num_matches=100,
        source_size=(w, h), target_size=(w, h), diagnostics={},
    )


def _jagged(poly_center, r1, r2, n=16):
    cx, cy = poly_center
    pts = []
    for i in range(n):
        a = 2 * np.pi * i / n
        r = r1 if i % 2 == 0 else r2
        pts.append([int(round(cx + r * np.cos(a))), int(round(cy + r * np.sin(a)))])
    return np.asarray(pts, np.int32)


def test_source_direct_copies_interior_without_source_border_and_handles_colored_burst():
    h, w = 640, 480
    source = np.full((h, w, 3), 255, np.uint8)
    target = np.full((h, w, 3), 245, np.uint8)

    # White oval speech bubble.
    cv2.ellipse(source, (150, 180), (52, 40), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(source, "CN", (125, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.ellipse(target, (150, 180), (52, 40), 0, 0, 360, (0, 0, 0), 3)
    cv2.putText(target, "JP", (125, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    # Coloured jagged target, white translated source.  Keep a black target
    # outline so the detector can align, but final source border must not replace it.
    star = _jagged((320, 430), 60, 40, 18)
    cv2.fillPoly(source, [star], (255, 255, 255)); cv2.polylines(source, [star], True, (0, 0, 0), 3)
    cv2.putText(source, "CN", (298, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.fillPoly(target, [star], (0, 230, 255)); cv2.polylines(target, [star], True, (0, 0, 0), 3)
    cv2.putText(target, "JP", (298, 438), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    plan = build_source_direct_container_plan(source, target, _reg((h, w)), MaskReplaceConfig())
    assert plan is not None
    assert plan.safe_to_skip_other_paths
    assert plan.result.applied_count == 2
    assert plan.diagnostics["accepted_white"] == 1
    assert plan.diagnostics["accepted_colored_spiky"] == 1
    assert plan.diagnostics["border_pixels_written"] == 0

    # Interior is source content, border itself remains target black outline.
    assert np.all(plan.result.image[430, 320] == source[430, 320])
    assert np.max(plan.result.image[430, 380]) < 40
    assert all(r.sr_backend == "source-direct-container" for r in plan.result.records)
    assert all(r.content_complete for r in plan.result.records)
