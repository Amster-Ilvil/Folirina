from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.bubbles import pair_unseeded_white_containers
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.mask_transfer import transfer_rigid_container_rasters
from manga_hd_transfer.models import RegistrationResult


def _reg() -> RegistrationResult:
    return RegistrationResult(
        matrix=np.eye(3, dtype=np.float32), method='identity', confidence=0.99,
        inlier_ratio=1.0, reprojection_error=0.0, spatial_coverage=1.0,
        num_matches=100, source_size=(320, 320), target_size=(320, 320),
    )


def test_inverse_target_container_recovers_leaky_source_white_region() -> None:
    cfg = PipelineConfig()
    src = np.full((320, 320, 3), 40, np.uint8)
    tgt = np.full((320, 320, 3), 40, np.uint8)
    # Target has a clean enclosed white dialogue box.
    cv2.rectangle(tgt, (180, 185), (270, 270), (255, 255, 255), -1)
    cv2.rectangle(tgt, (180, 185), (270, 270), (0, 0, 0), 3)
    for x in (204, 224, 244):
        for y in (204, 224, 244):
            cv2.rectangle(tgt, (x, y), (x + 8, y + 10), (0, 0, 0), -1)
    # Source translation enlarged/leaked the white region upward, but the actual
    # Chinese lettering still sits inside the target-mapped container.
    cv2.rectangle(src, (178, 120), (272, 272), (255, 255, 255), -1)
    cv2.rectangle(src, (178, 120), (272, 272), (0, 0, 0), 3)
    for x in (202, 220, 238, 256):
        for y in (198, 220, 242):
            cv2.rectangle(src, (x, y), (x + 8, y + 10), (0, 0, 0), -1)
    cv2.circle(src, (228, 260), 3, (0, 0, 0), -1)

    ss, tt = pair_unseeded_white_containers(src, tgt, _reg(), cfg.mask_replace, cfg.bubbles, [])
    assert ss and tt
    assert any(x.meta.get('source_mask_mode') == 'inverse_target_container' for x in ss)
    result = transfer_rigid_container_rasters(src, tgt, tgt, ss, tt, cfg.mask_replace)
    assert result.applied_count >= 1
    assert any(r.content_complete for r in result.records if r.applied)
