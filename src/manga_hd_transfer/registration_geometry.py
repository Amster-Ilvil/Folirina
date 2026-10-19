"""Low-level helpers for persisted page registration geometry."""

from __future__ import annotations

from typing import Any

import numpy as np

from .schema_compat import as_dict


def registration_homography(project: dict[str, Any]) -> np.ndarray:
    """Read and normalize the persisted SOURCE->TARGET page transform."""
    root = as_dict(project)
    row = as_dict(root.get("registration"))
    raw = np.asarray(row.get("matrix", np.eye(3)), dtype=np.float64)
    if raw.shape == (2, 3):
        matrix = np.eye(3, dtype=np.float64)
        matrix[:2, :] = raw
        raw = matrix
    if raw.shape != (3, 3) or not np.all(np.isfinite(raw)):
        return np.eye(3, dtype=np.float64)
    if abs(float(raw[2, 2])) > 1e-12:
        raw = raw / float(raw[2, 2])
    return raw
