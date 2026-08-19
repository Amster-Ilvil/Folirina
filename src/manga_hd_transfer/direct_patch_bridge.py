from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .direct_patch_mode import run_direct_patch_mode


class ModeIsolationError(RuntimeError):
    """Raised when the isolated direct mode is invoked incorrectly."""


def dispatch_isolated_direct_mode(requested_mode: str, page_dir: str | Path) -> Dict[str, Any]:
    if requested_mode != 'direct_patch':
        raise ModeIsolationError(
            f'isolated direct dispatcher only accepts direct_patch, got: {requested_mode!r}'
        )
    result = run_direct_patch_mode(page_dir)
    return {
        'accepted': result.accepted,
        'reason': result.reason,
        'used': result.used,
        'requested_mode': result.requested_mode,
        'strategy': result.strategy,
        'registration_confidence': result.registration_confidence,
        'mask_pixels': result.mask_pixels,
        'changed_pixels': result.changed_pixels,
        'changed_ratio': result.changed_ratio,
        'outside_mask_unchanged': result.outside_mask_unchanged,
        'target_shape': result.target_shape,
        'regions': [r.__dict__ for r in result.regions],
    }
