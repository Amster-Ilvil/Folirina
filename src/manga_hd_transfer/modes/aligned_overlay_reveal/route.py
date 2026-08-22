from __future__ import annotations

"""Private execution capsule for the explicit whole-page hole mode."""

from typing import Any

import numpy as np

from ...config import AlignedOverlayRevealConfig, BubbleConfig
from ...models import RegistrationResult
from .hole_renderer import AlignedOverlayResult, build_production_aligned_hole_result

MODE_KEY = "aligned_overlay_reveal"


class AlignedHoleModeIsolationError(RuntimeError):
    pass


def execute_isolated_hole_route(
    requested_mode: str,
    *,
    same_page: bool,
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    config: AlignedOverlayRevealConfig,
    bubble_config: BubbleConfig,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    source_path: str | None = None,
    target_path: str | None = None,
) -> AlignedOverlayResult:
    mode = str(requested_mode or "").strip().lower()
    if mode != MODE_KEY:
        raise AlignedHoleModeIsolationError(
            f"isolated hole dispatcher only accepts {MODE_KEY}, got: {requested_mode!r}"
        )
    stats = cache_stats if cache_stats is not None else {}
    if not bool(same_page):
        result = build_production_aligned_hole_result(
            source, target, registration, config, bubble_config,
            stage_cache=stage_cache, cache_stats=stats, source_path=source_path,
            target_path=target_path, target_bubbles=[],
        )
        result.plan.accepted = False
        result.plan.reason = "rejected_page_pair_verification"
        result.diagnostics["reason"] = "rejected_page_pair_verification"
        return result
    return build_production_aligned_hole_result(
        source, target, registration, config, bubble_config, stage_cache=stage_cache,
        cache_stats=stats, source_path=source_path, target_path=target_path,
    )


__all__ = ["MODE_KEY", "AlignedHoleModeIsolationError", "execute_isolated_hole_route"]
