from __future__ import annotations

"""Thin dispatcher for mode-private transfer execution stages.

v2.3.35 removes pixel-changing orchestration from this shared module. Direct,
Precise Mask, Hybrid and Reletter each own their clear/inpaint/lettering sequence
under ``modes/<mode>/execution_stage.py``. This file only selects the capsule.
"""

from typing import Any

# Compatibility exports used by older tests/plugins. Active runtime execution
# does not use these shared aliases; OCR modes import their own private cleanup.
from .modes.reletter.ocr_cleanup import (
    _ocr_paper_first_clear,
    _reletter_paper_first_clear,
)


def _resolve_private_execution_stage(*, mode: str, direct_container_fast: bool):
    key = str(mode or "").strip().lower()
    if direct_container_fast and key not in {"direct_patch", "auto"}:
        raise RuntimeError(
            f"mode isolation violation: direct_container_fast cannot route mode={key or mode}"
        )
    if key == "direct_patch" or (key == "auto" and direct_container_fast):
        from .modes.direct_patch import execution_stage
        return "direct_patch", execution_stage
    if key == "hybrid":
        from .modes.hybrid import execution_stage
        return "hybrid", execution_stage
    if key == "reletter":
        from .modes.reletter import execution_stage
        return "reletter", execution_stage
    if key in {"mask_replace", "auto"}:
        from .modes.mask_replace import execution_stage
        return "mask_replace", execution_stage
    raise ValueError(f"Unsupported transfer execution mode: {mode}")


def run_transfer_execution_stage(*, mode: str, direct_container_fast: bool, cache_stats: dict | None = None, **kwargs):
    owner, module = _resolve_private_execution_stage(mode=mode, direct_container_fast=direct_container_fast)
    if cache_stats is not None:
        cache_stats["active_execution_capsule"] = owner
    return module.run_transfer_execution_stage(
        mode=mode,
        direct_container_fast=direct_container_fast,
        cache_stats=cache_stats,
        **kwargs,
    )


TransferExecutionState = Any

__all__ = [
    "TransferExecutionState", "run_transfer_execution_stage",
    "_ocr_paper_first_clear", "_reletter_paper_first_clear",
]
