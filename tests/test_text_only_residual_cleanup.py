from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.text_only_transfer import cleanup_target_residual_specks


def test_target_only_black_dot_removed_but_source_punctuation_protected():
    h, w = 96, 128
    target = np.full((h, w, 3), 248, np.uint8)
    # Old TARGET-only punctuation / antialias residue.
    cv2.circle(target, (88, 48), 2, (20, 20, 20), -1)
    # A second dark dot at a location occupied by legitimate SOURCE punctuation.
    cv2.circle(target, (52, 70), 2, (20, 20, 20), -1)
    image = target.copy()  # simulate imperfect first clear that left both dots
    region = np.zeros((h, w), np.uint8); region[12:86, 12:116] = 255
    source_mask = np.zeros((h, w), np.uint8)
    cv2.circle(source_mask, (52, 70), 3, 255, -1)  # protect Chinese punctuation
    clear = np.zeros((h, w), np.uint8)
    cv2.circle(clear, (88, 48), 4, 255, -1)

    out, removed, diag = cleanup_target_residual_specks(
        image, target, region, source_mask, clear, white_container=True,
    )
    assert diag["residual_specks_removed"] > 0
    assert cv2.countNonZero(removed) > 0
    # TARGET-only residue is restored to white paper.
    assert float(np.mean(out[46:51, 86:91])) > 220.0
    # SOURCE-supported punctuation is untouched.
    assert int(out[70, 52, 0]) < 60


def test_target_only_dot_just_outside_strict_white_mask_is_cleaned():
    h, w = 80, 120
    target = np.full((h, w, 3), 250, np.uint8)
    # Strict write mask ends at x=80; old punctuation sits 5px beyond it.
    region = np.zeros((h, w), np.uint8); region[10:70, 20:81] = 255
    cv2.circle(target, (85, 38), 2, (10, 10, 10), -1)
    image = target.copy()
    source_mask = np.zeros((h, w), np.uint8)
    clear = np.zeros((h, w), np.uint8); clear[25:55, 60:80] = 255
    out, removed, diag = cleanup_target_residual_specks(
        image, target, region, source_mask, clear, white_container=True,
    )
    assert diag["residual_specks_removed"] > 0
    assert int(removed[38, 85]) > 0
    assert int(out[38, 85, 0]) > 220
