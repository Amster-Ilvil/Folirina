from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.config import RegistrationConfig
from manga_hd_transfer.registration import register_images
from .helpers import make_art_page


def test_sift_registration_recovers_affine_transform():
    src = make_art_page(900, 1200)
    center = (src.shape[1] / 2, src.shape[0] / 2)
    a = cv2.getRotationMatrix2D(center, 1.7, 1.08)
    a[:, 2] += (24, -17)
    dst = cv2.warpAffine(src, a, (1000, 1300), borderValue=(255, 255, 255))
    cfg = RegistrationConfig(backend="opencv", feature="sift", min_matches=8, review_confidence=0.4)
    reg = register_images(src, dst, cfg)
    assert reg.num_matches >= 8
    assert reg.confidence > 0.45
    pts = np.array([[[100, 100], [700, 200], [450, 900]]], dtype=np.float32)
    expected = cv2.transform(pts, a)[0]
    actual = cv2.perspectiveTransform(pts, reg.matrix)[0]
    err = np.median(np.linalg.norm(expected - actual, axis=1))
    assert err < 6.0
