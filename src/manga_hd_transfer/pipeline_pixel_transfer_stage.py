from __future__ import annotations

"""Thin dispatcher for mode-private pixel stages.

v2.3.35 removes active pixel sequencing from this shared module. Direct,
Precise Mask, Hybrid and Reletter each own a complete stage implementation
under ``modes/<mode>/pixel_stage.py``. The shared file performs routing only.
"""

from typing import Any

# Compatibility exports for older tests/plugins. They are aliases to one frozen
# mode-private implementation; the active runtime dispatcher below never uses
# these aliases to route another mode.
from .modes.direct_patch.pixel_stage import _semantic_completion_exclude_mask
from .modes.mask_replace.pixel_stage import _transfer_owned_region_mask, _merge_mask_transfer


def _active_mask_config(config: Any, mode: str):
    """Compatibility resolver for tests/legacy callers only.

    Active runtime dispatch no longer uses this shared resolver.
    """
    return config.hybrid.mask if str(mode or "").strip().lower() == "hybrid" else config.mask_replace


def _resolve_private_pixel_stage(*, mode: str, direct_container_fast: bool):
    key = str(mode or "").strip().lower()
    if direct_container_fast and key not in {"direct_patch", "auto"}:
        raise RuntimeError(
            f"mode isolation violation: direct_container_fast cannot route mode={key or mode}"
        )
    if key == "direct_patch" or (key == "auto" and direct_container_fast):
        from .modes.direct_patch import pixel_stage
        return "direct_patch", pixel_stage
    if key == "hybrid":
        from .modes.hybrid import pixel_stage
        return "hybrid", pixel_stage
    if key == "reletter":
        from .modes.reletter import pixel_stage
        return "reletter", pixel_stage
    if key in {"mask_replace", "auto"}:
        from .modes.mask_replace import pixel_stage
        return "mask_replace", pixel_stage
    raise ValueError(f"Unsupported pixel stage mode: {mode}")


def run_pixel_transfer_stage(*, mode: str, direct_container_fast: bool, cache_stats: dict | None = None, **kwargs):
    owner, module = _resolve_private_pixel_stage(mode=mode, direct_container_fast=direct_container_fast)
    if cache_stats is not None:
        cache_stats["active_pixel_stage_capsule"] = owner
    return module.run_pixel_transfer_stage(
        mode=mode,
        direct_container_fast=direct_container_fast,
        cache_stats=cache_stats,
        **kwargs,
    )


# Kept as a loose compatibility type alias. Concrete instances are returned by
# the selected private module and only rely on attribute protocol at callers.
PixelTransferState = Any

__all__ = [
    "PixelTransferState", "run_pixel_transfer_stage", "_merge_mask_transfer", "_active_mask_config",
    "_semantic_completion_exclude_mask", "_transfer_owned_region_mask",
]
