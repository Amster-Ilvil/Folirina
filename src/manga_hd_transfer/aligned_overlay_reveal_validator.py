
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np


def _load_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def _imread(path: Path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f'Unable to read image: {path}')
    return img


def validate_aligned_overlay_reveal(page_dir: str | Path) -> Dict[str, Any]:
    page = Path(page_dir)
    target = _imread(page / 'target_original.png', cv2.IMREAD_COLOR)
    final = _imread(page / 'final.png', cv2.IMREAD_COLOR)
    mask = _imread(page / 'aligned_overlay_reveal_mask.png', cv2.IMREAD_GRAYSCALE)
    meta = _load_json(page / 'aligned_overlay_reveal.json')

    same_shape = tuple(target.shape[:2]) == tuple(final.shape[:2]) == tuple(mask.shape[:2])
    outside = mask == 0
    outside_unchanged = bool(np.array_equal(final[outside], target[outside])) if same_shape else False
    changed_mask = np.any(final != target, axis=2).astype(np.uint8) * 255
    inside_change = int(np.count_nonzero((changed_mask > 0) & (mask > 0)))
    outside_change = int(np.count_nonzero((changed_mask > 0) & (mask == 0)))
    total_change = int(np.count_nonzero(changed_mask > 0))
    changed_ratio = total_change / float(target.shape[0] * target.shape[1])
    issues = []
    if not same_shape:
        issues.append('shape_mismatch')
    if not meta.get('used'):
        issues.append('mode_not_used')
    if not meta.get('accepted'):
        issues.append('mode_not_accepted')
    if outside_change:
        issues.append('outside_mask_changed')
    if total_change <= 0:
        issues.append('no_effect')
    return {
        'schema': 'manga_hd_translation_transfer.aligned_overlay_reveal.validation.v1',
        'page_dir': str(page),
        'pass': not issues,
        'issues': issues,
        'same_shape': same_shape,
        'outside_mask_unchanged': outside_unchanged,
        'inside_change_pixels': inside_change,
        'outside_change_pixels': outside_change,
        'total_change_pixels': total_change,
        'changed_ratio': changed_ratio,
        'meta_requested_mode': meta.get('requested_mode'),
        'meta_used': meta.get('used'),
        'meta_accepted': meta.get('accepted'),
    }
