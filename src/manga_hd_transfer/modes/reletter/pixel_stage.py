from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
from ...models import BubbleInstance

@dataclass
class PixelTransferState:
    mask_transfer: Any
    unseeded_white_pair_count: int
    completion_display_source: list[BubbleInstance]
    completion_display_target: list[BubbleInstance]
    transfer_rgba: np.ndarray
    fallback_matches: list[Any]
    semantic_layout_evidence: Any | None

def _merge_mask_transfer(base, extra):
    return extra if base is None else base

def run_pixel_transfer_stage(*, mode, target, accepted, target_layout_authority, **kwargs) -> PixelTransferState:
    if str(mode or "").strip().lower() != "reletter":
        raise RuntimeError("reletter pixel stage cannot execute mode=" + str(mode))
    return PixelTransferState(
        mask_transfer=None, unseeded_white_pair_count=0,
        completion_display_source=[], completion_display_target=[],
        transfer_rgba=np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8),
        fallback_matches=list(accepted or []), semantic_layout_evidence=target_layout_authority,
    )

__all__=["PixelTransferState","run_pixel_transfer_stage","_merge_mask_transfer"]
