from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.review_apply import _load_effective_clear_mask


def _write(path: Path, arr: np.ndarray) -> None:
    assert cv2.imwrite(str(path), arr)


def test_additive_japanese_clear_mask_merges_with_auto(tmp_path: Path):
    shape = (32, 40)
    auto = np.zeros(shape, np.uint8); auto[5:10, 5:10] = 255
    extra = np.zeros(shape, np.uint8); extra[20:24, 27:31] = 255
    _write(tmp_path / 'target_clear_mask.png', auto)
    _write(tmp_path / 'manual_japanese_clear_mask.png', extra)

    mask, source = _load_effective_clear_mask(tmp_path, shape)
    assert source == 'target_clear_mask+manual_japanese_clear_mask'
    assert np.all(mask[5:10, 5:10] == 255)
    assert np.all(mask[20:24, 27:31] == 255)
    assert cv2.countNonZero(mask) == cv2.countNonZero(auto) + cv2.countNonZero(extra)


def test_legacy_manual_clear_remains_authoritative_then_additive_brush_is_added(tmp_path: Path):
    shape = (30, 36)
    auto = np.zeros(shape, np.uint8); auto[2:8, 2:8] = 255
    legacy = np.zeros(shape, np.uint8); legacy[10:15, 10:15] = 255
    extra = np.zeros(shape, np.uint8); extra[20:23, 25:29] = 255
    _write(tmp_path / 'target_clear_mask.png', auto)
    _write(tmp_path / 'manual_clear_mask.png', legacy)
    _write(tmp_path / 'manual_japanese_clear_mask.png', extra)

    mask, source = _load_effective_clear_mask(tmp_path, shape)
    assert source == 'manual_clear_mask+manual_japanese_clear_mask'
    assert np.all(mask[10:15, 10:15] == 255)
    assert np.all(mask[20:23, 25:29] == 255)
    assert np.count_nonzero(mask[2:8, 2:8]) == 0
