from __future__ import annotations

"""Private execution capsule for the explicit whole-page transparent mode.

This module is deliberately separate from aligned_overlay_reveal.  It may call
only the renderer owned by transparent_bubble_reveal and returns before any
Direct/Mask/Hybrid/Reletter route is considered.
"""

from typing import Any

import numpy as np

from ...config import BubbleConfig, TransparentBubbleRevealConfig
from ...models import RegistrationResult
from ...transparent_bubble_reveal import (
    TransparentBubbleResult,
    build_transparent_bubble_plan,
    execute_transparent_bubble,
    reject_transparent_bubble_plan,
)


MODE_KEY = "transparent_bubble_reveal"


class TransparentModeIsolationError(RuntimeError):
    pass


def execute_isolated_transparent_route(
    requested_mode: str,
    *,
    same_page: bool,
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    config: TransparentBubbleRevealConfig,
    bubble_config: BubbleConfig,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    source_path: str | None = None,
    target_text_ocr=None,
    semantic_config=None,
) -> TransparentBubbleResult:
    """Build and execute only the transparent route.

    A wrong mode key is a hard error instead of an alias/fallback.  This prevents
    aligned-hole and transparent workflows from silently calling each other.
    """
    mode = str(requested_mode or "").strip().lower()
    if mode != MODE_KEY:
        raise TransparentModeIsolationError(
            f"isolated transparent dispatcher only accepts {MODE_KEY}, got: {requested_mode!r}"
        )
    stats = cache_stats if cache_stats is not None else {}
    if not bool(same_page):
        plan = reject_transparent_bubble_plan(
            source, target, registration, "rejected_page_pair_verification"
        )
    else:
        plan = build_transparent_bubble_plan(
            source, target, registration, config, bubble_config=bubble_config,
            stage_cache=stage_cache, cache_stats=stats, target_path=target_path,
            source_path=source_path, target_text_ocr=target_text_ocr,
            semantic_config=semantic_config,
        )
    return execute_transparent_bubble(plan, source, target, config)


__all__ = [
    "MODE_KEY", "TransparentModeIsolationError", "execute_isolated_transparent_route",
]
