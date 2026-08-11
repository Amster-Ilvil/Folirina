from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import MaskingConfig
from .geometry import rasterize_polygon
from .models import BubbleInstance, TextBlock, TextUnit, UnitMatch


@dataclass(slots=True)
class MaskBuildResult:
    mask: np.ndarray
    per_unit: dict[str, np.ndarray]
    clipped_pixels: int
    source_pixels: int

    @property
    def clipped_ratio(self) -> float:
        return self.clipped_pixels / max(1, self.source_pixels)


def _dilation_pixels(block: TextBlock, cfg: MaskingConfig) -> int:
    x0, y0, x1, y1 = block.bbox
    scale = max(1.0, min(x1 - x0, y1 - y0))
    return int(np.clip(round(scale * cfg.dilation_ratio), cfg.min_dilation_px, cfg.max_dilation_px))


def _protected_bubble_mask(bubble: BubbleInstance, cfg: MaskingConfig) -> np.ndarray | None:
    if bubble.mask is None:
        return None
    if cfg.bubble_border_protection_px <= 0:
        return bubble.mask
    r = cfg.bubble_border_protection_px
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    eroded = cv2.erode((bubble.mask > 0).astype(np.uint8) * 255, k, iterations=1)
    return eroded


def build_clear_mask(
    image_shape: tuple[int, int],
    target_blocks: list[TextBlock],
    target_units: list[TextUnit],
    target_bubbles: list[BubbleInstance],
    matches: list[UnitMatch],
    config: MaskingConfig | None = None,
    min_match_confidence: float = 0.0,
    allow_relations: set[str] | None = None,
) -> MaskBuildResult:
    cfg = config or MaskingConfig()
    h, w = image_shape
    block_by_id = {b.id: b for b in target_blocks}
    unit_by_id = {u.id: u for u in target_units}
    bubble_by_id = {b.id: b for b in target_bubbles}
    allowed_relations = allow_relations or {"one_to_one"}

    final = np.zeros((h, w), dtype=np.uint8)
    per_unit: dict[str, np.ndarray] = {}
    source_pixels = 0
    clipped_pixels = 0

    for match in matches:
        if match.confidence < min_match_confidence or match.relation not in allowed_relations:
            continue
        unit = unit_by_id.get(match.target_unit_id)
        if unit is None:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        for block_id in unit.block_ids:
            block = block_by_id.get(block_id)
            if block is None:
                continue
            block_mask = None
            mask_path = block.meta.get("mask_path") if block.meta else None
            if mask_path and Path(mask_path).exists():
                pixel_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if pixel_mask is not None:
                    if pixel_mask.shape != (h, w):
                        pixel_mask = cv2.resize(pixel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    block_mask = (pixel_mask > 0).astype(np.uint8) * 255
            if block_mask is None:
                block_mask = rasterize_polygon(block.polygon, (h, w))
            d = _dilation_pixels(block, cfg)
            if d > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
                block_mask = cv2.dilate(block_mask, k, iterations=1)
            mask = cv2.bitwise_or(mask, block_mask)

        before = cv2.countNonZero(mask)
        source_pixels += before
        if cfg.clip_to_bubble and unit.bubble_id and unit.bubble_id in bubble_by_id:
            protected = _protected_bubble_mask(bubble_by_id[unit.bubble_id], cfg)
            if protected is not None:
                clipped = cv2.bitwise_and(mask, protected)
                clipped_pixels += max(0, before - cv2.countNonZero(clipped))
                mask = clipped
        if cv2.countNonZero(mask) > 0:
            per_unit[unit.id] = mask
            final = cv2.bitwise_or(final, mask)

    return MaskBuildResult(final, per_unit, clipped_pixels, source_pixels)
