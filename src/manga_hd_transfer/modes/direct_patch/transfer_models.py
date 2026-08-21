from __future__ import annotations

"""Renderer-independent result models for Mask/Direct transfer.

These dataclasses are intentionally kept outside ``mask_transfer.py`` so callers
that only need to exchange transfer results do not import the 2k+ line raster
renderer.  The compatibility module still re-exports the same names.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from .geometry_ops import BubblePatchMatch


@dataclass(slots=True)
class MaskTransferRecord:
    source_bubble_id: str
    target_bubble_id: str
    confidence: float
    applied: bool
    reason: str
    sr_backend: str = "off"
    sr_scale: float = 1.0
    mask_iou: float = 0.0
    target_coverage: float = 0.0
    spill_ratio: float = 0.0
    local_dx: float = 0.0
    local_dy: float = 0.0
    sharpness: float = 0.0
    target_sharpness: float = 0.0
    relative_sharpness: float = 0.0
    clarity_mode: str = "pixels"
    geometry_mode: str = "standard"
    ink_ratio: float = 0.0
    source_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    target_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)
    source_edge_clipped: bool = False
    source_edge_sides: str = ""
    candidate: bool = False
    review_required: bool = False
    review_reason: str = ""
    restorable: bool = False
    editable: bool = False
    # Geometry/raster application and content verification are separate.
    # ``applied`` means pixels were written; ``content_complete`` means the
    # expected SOURCE ink survived and TARGET-language ink was removed.
    content_check: str = "not_checked"
    source_ink_coverage: float = 0.0
    target_residual_ratio: float = 0.0
    content_complete: bool = False
    repair_attempted: bool = False
    repair_succeeded: bool = False
    triage_state: str = "UNSET"  # SAFE|REVIEW|REJECT|UNSET
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class MaskTransferResult:
    image: np.ndarray
    layer_rgba: np.ndarray
    composite_mask: np.ndarray
    matches: list[BubblePatchMatch]
    records: list[MaskTransferRecord]
    clear_mask: np.ndarray | None = None

    @property
    def applied_count(self) -> int:
        return sum(1 for row in self.records if row.applied)


__all__ = ["MaskTransferRecord", "MaskTransferResult"]
