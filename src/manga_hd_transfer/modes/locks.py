from __future__ import annotations

import hashlib
from pathlib import Path

from .registry import get_mode_spec


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def owned_code_hashes(project_root: str | Path, mode: str) -> dict[str, str]:
    """Hash only mode-owned code. Shared primitives are intentionally excluded."""
    root = Path(project_root)
    out: dict[str, str] = {}
    for raw in get_mode_spec(mode).owned_paths:
        p = root / raw
        if p.is_file():
            out[raw] = _sha256(p)
        elif p.is_dir():
            for f in sorted(p.rglob('*.py')):
                out[f.relative_to(root).as_posix()] = _sha256(f)
    return out
