
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .hole_renderer import run_aligned_overlay_reveal


class ModeIsolationError(RuntimeError):
    """Raised when the isolated aligned whole-page mode is invoked incorrectly."""


def dispatch_isolated_aligned_mode(requested_mode: str, page_dir: str | Path) -> Dict[str, Any]:
    if requested_mode != 'aligned_overlay_reveal':
        raise ModeIsolationError(
            f'isolated aligned whole-page dispatcher only accepts aligned_overlay_reveal, got: {requested_mode!r}'
        )
    result = run_aligned_overlay_reveal(page_dir)
    return {
        'accepted': result.accepted,
        'reason': result.reason,
        'page_triage': result.page_triage,
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
