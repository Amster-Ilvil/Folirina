"""Low-level helpers for persisted page registration geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema_compat import as_dict


def normalize_registration_matrix(value: Any) -> np.ndarray:
    """Validate a persisted transform before it can authorise raster writes."""
    try:
        raw = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("配准矩阵损坏，请重新处理当前页后再替换") from exc
    if raw.shape == (2, 3):
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = raw
        raw = matrix
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        raise ValueError("配准矩阵形状或数值无效，请重新处理当前页后再替换")
    scale = float(np.max(np.abs(raw)))
    if scale == 0 or np.linalg.slogdet(raw / scale)[0] == 0:
        raise ValueError("配准矩阵不可逆，请重新处理当前页后再替换")
    if abs(float(raw[2, 2])) > scale * 1e-12:
        raw = raw / float(raw[2, 2])
    else:
        raw = raw / scale
    return raw


def registration_homography(project: dict[str, Any]) -> np.ndarray:
    """Keep missing legacy geometry compatible, but reject explicit corrupt data."""
    row = as_dict(as_dict(project).get("registration"))
    return normalize_registration_matrix(row.get("matrix", np.eye(3)))
