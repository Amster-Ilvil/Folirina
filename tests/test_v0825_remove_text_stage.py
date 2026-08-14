from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.review_apply import generate_remove_text_preview


def _write(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', arr)
    assert ok
    data.tofile(path)


def test_remove_text_stage_runs_without_rerunning_transfer(tmp_path: Path) -> None:
    target = np.full((120, 160, 3), 255, np.uint8)
    target[45:78, 68:88] = 0
    mask = np.zeros((120, 160), np.uint8)
    mask[40:84, 62:94] = 255
    _write(tmp_path / 'target_original.png', target)
    _write(tmp_path / 'target_clear_mask.png', mask)

    cfg = PipelineConfig()
    cfg.inpainting.backend = 'solid'
    out = generate_remove_text_preview(tmp_path, cfg)
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    assert float(result[50:72, 72:84].mean()) > 245.0
    meta = json.loads((tmp_path / 'remove_text_stage.json').read_text(encoding='utf-8'))
    assert meta['mask_source'] == 'target_clear_mask'
    assert meta['mask_pixels'] == int(cv2.countNonZero(mask))


def test_manual_clear_mask_is_authoritative(tmp_path: Path) -> None:
    target = np.full((100, 140, 3), 255, np.uint8)
    target[25:45, 25:45] = 0
    target[55:78, 90:112] = 0
    auto = np.zeros((100, 140), np.uint8); auto[20:50, 20:50] = 255
    manual = np.zeros((100, 140), np.uint8); manual[50:82, 85:117] = 255
    _write(tmp_path / 'target_original.png', target)
    _write(tmp_path / 'target_clear_mask.png', auto)
    _write(tmp_path / 'manual_clear_mask.png', manual)

    cfg = PipelineConfig(); cfg.inpainting.backend = 'solid'
    out = generate_remove_text_preview(tmp_path, cfg)
    result = cv2.imread(str(out), cv2.IMREAD_COLOR)
    assert result is not None
    # Auto-only region stays untouched; manual region is removed.
    assert float(result[30:40, 30:40].mean()) < 20.0
    assert float(result[60:72, 94:108].mean()) > 245.0
    meta = json.loads((tmp_path / 'remove_text_stage.json').read_text(encoding='utf-8'))
    assert meta['mask_source'] == 'manual_clear_mask'
