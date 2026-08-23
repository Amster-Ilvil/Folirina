from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...config import MaskingConfig
from ...geometry import rasterize_polygon
from ...models import BubbleInstance, TextBlock, TextUnit, UnitMatch


@dataclass(slots=True)
class MaskBuildResult:
    mask: np.ndarray
    per_unit: dict[str, np.ndarray]
    clipped_pixels: int
    source_pixels: int

    @property
    def clipped_ratio(self) -> float:
        return self.clipped_pixels / max(1, self.source_pixels)


def _dilation_pixels(block: TextBlock, cfg: MaskingConfig, *, has_pixel_mask: bool = False) -> int:
    if has_pixel_mask and bool(getattr(cfg, "pixel_mask_priority_no_extra_dilation", True)):
        return 0
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






def _border_guard(mask: np.ndarray, bubble: BubbleInstance | None, target_image: np.ndarray | None, cfg: MaskingConfig) -> np.ndarray:
    if not bool(getattr(cfg, "border_guard_enabled", True)) or bubble is None or bubble.mask is None:
        return mask
    protected = _protected_bubble_mask(bubble, cfg)
    if protected is None:
        return mask
    border_ring = cv2.bitwise_and((bubble.mask > 0).astype(np.uint8) * 255, cv2.bitwise_not(protected))
    if cv2.countNonZero(border_ring) == 0:
        return mask
    out = mask.copy()
    max_erode = max(0, int(getattr(cfg, "border_guard_max_erode_px", 2)))
    overlap_limit = float(getattr(cfg, "border_guard_overlap_ratio_max", 0.12))
    dark_limit = float(getattr(cfg, "border_guard_dark_ratio_min", 0.22))
    gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY) if target_image is not None and target_image.ndim == 3 else None
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for _ in range(max_erode + 1):
        overlap = cv2.bitwise_and(out, border_ring)
        overlap_n = int(cv2.countNonZero(overlap))
        if overlap_n == 0:
            break
        overlap_ratio = overlap_n / max(1, int(cv2.countNonZero(out)))
        dark_ratio = 0.0
        if gray is not None:
            dark_ratio = float(np.count_nonzero((overlap > 0) & (gray <= 165)) / max(1, overlap_n))
        if overlap_ratio <= overlap_limit or dark_ratio < dark_limit:
            break
        out = cv2.erode(out, kernel, iterations=1)
    return out


def _synthetic_target_text_mask(
    image: np.ndarray, bubble: BubbleInstance | None, cfg: MaskingConfig,
    current_image: np.ndarray | None = None,
) -> np.ndarray | None:
    """Detect Japanese glyph ink inside a known target bubble.

    Transcript-only Apple OCR gives us reliable bubble identity but no target text
    polygon.  Clearing the whole synthetic bubble creates conspicuous white blocks.
    Instead, use the clean target scan itself: threshold dark ink *inside the protected
    bubble interior*, reject implausibly large components, then dilate slightly to
    include antialiasing.  Bubble outlines and surrounding artwork remain untouched.
    """
    if image is None or bubble is None or bubble.mask is None:
        return None
    protected = _protected_bubble_mask(bubble, cfg)
    if protected is None or cv2.countNonZero(protected) == 0:
        return None
    safe_area = float(max(1, cv2.countNonZero(protected)))

    def detect_ink(src: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY) if src.ndim == 3 else src.copy()
        vals = gray[protected > 0]
        if vals.size == 0:
            return np.zeros_like(protected)
        paper = float(np.percentile(vals, 80))
        threshold = int(np.clip(paper - 48.0, 115, 205))
        raw = ((gray < threshold) & (protected > 0)).astype(np.uint8) * 255
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
        kept = np.zeros_like(raw)
        for idx in range(1, count):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < 2:
                continue
            if area > safe_area * 0.08 or w * h > safe_area * 0.18:
                continue
            kept[labels == idx] = 255
        return kept

    # Clear both the original Japanese glyphs and any already-transferred Chinese
    # glyphs underneath.  v0.8.10 only cleared the Japanese geometry while using
    # mask_transfer.image as the base, which left dark ghost strokes behind the
    # newly typeset OCR text.
    kept = detect_ink(image)
    if current_image is not None and current_image.shape[:2] == image.shape[:2]:
        kept = cv2.bitwise_or(kept, detect_ink(current_image))
    if cv2.countNonZero(kept) == 0:
        return None
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kept = cv2.dilate(kept, k, iterations=2)
    return cv2.bitwise_and(kept, protected)

def _complete_reletter_region_mask(
    image: np.ndarray, bubble: BubbleInstance | None, region_clip: np.ndarray,
    baseline: np.ndarray, cfg: MaskingConfig,
) -> np.ndarray:
    """Complete Japanese text clearing inside one target-driven Region.

    The detector polygon is already the identity boundary for the Japanese text
    island.  Component thresholding can miss thin kana, dakuten and antialiasing,
    especially after resampling.  On a uniform white speech/narration Region we
    therefore clear the whole trusted Region interior.  On non-uniform Regions we
    only add compact dark components and continue to protect long borders/rules.
    """
    if image is None or region_clip is None or cv2.countNonZero(region_clip) == 0:
        return baseline
    clip = (region_clip > 0).astype(np.uint8) * 255
    protected = _protected_bubble_mask(bubble, cfg) if bubble is not None else None
    if protected is not None and protected.shape == clip.shape and cv2.countNonZero(protected) > 0:
        clip = cv2.bitwise_and(clip, protected)
    if cv2.countNonZero(clip) == 0:
        return baseline
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    vals = gray[clip > 0]
    if vals.size == 0:
        return baseline
    white_ratio = float(np.count_nonzero(vals >= 220) / max(1, vals.size))
    spread = float(np.percentile(vals, 90) - np.percentile(vals, 20))
    # Target-driven Reletter regions are generated from detected Japanese glyph
    # islands. Treating their rectangular extent as disposable paper caused large
    # white holes and could eat balloon outlines. Keep legacy full-region clearing
    # available only behind an explicit compatibility opt-out; the normal Reletter
    # path is glyph/component based.
    allow_full_region = (
        bool(getattr(cfg, "reletter_region_full_clear_enabled", True))
        and not bool(getattr(cfg, "reletter_target_region_glyph_only", True))
    )
    if (allow_full_region
            and white_ratio >= float(getattr(cfg, "reletter_region_full_clear_min_white_ratio", 0.68))
            and spread <= float(getattr(cfg, "reletter_region_full_clear_max_spread", 48.0))):
        # Keep one-pixel safety around the Region edge; the parent bubble border
        # has already been excluded by ``protected``.
        inner = cv2.erode(clip, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        if cv2.countNonZero(inner) > 0:
            return cv2.bitwise_or(baseline, inner)

    paper = float(np.percentile(vals, 82))
    threshold = int(np.clip(paper - 44.0, 118, 208))
    raw = ((gray < threshold) & (clip > 0)).astype(np.uint8) * 255
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(raw, 8)
    added = np.zeros_like(raw)
    area_limit = max(18, int(round(cv2.countNonZero(clip) * 0.16)))
    ys, xs = np.where(clip > 0)
    rw = max(1, int(xs.max() - xs.min() + 1)) if xs.size else 1
    rh = max(1, int(ys.max() - ys.min() + 1)) if ys.size else 1
    for idx in range(1, count):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 2 or area > area_limit:
            continue
        # Reject long/thin panel or balloon rules. Small punctuation remains.
        if (w >= int(rw * 0.72) and h <= 3) or (h >= int(rh * 0.72) and w <= 3):
            continue
        cx, cy = centroids[idx]
        if not (0 <= int(round(cy)) < clip.shape[0] and 0 <= int(round(cx)) < clip.shape[1] and clip[int(round(cy)), int(round(cx))] > 0):
            continue
        added[labels == idx] = 255
    if cv2.countNonZero(added) > 0:
        added = cv2.dilate(added, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        added = cv2.bitwise_and(added, clip)
    return cv2.bitwise_or(baseline, added)


def _glyph_component_clip(block: TextBlock, shape: tuple[int, int]) -> np.ndarray | None:
    """Return a compact clip around detector-proven glyph components.

    The block polygon is intentionally a layout envelope, not a destructive mask.
    Destructive clearing must stay close to the component evidence stored by the
    target text-region detector. A tiny pad keeps antialiasing/dakuten while the
    protected bubble mask still owns the final boundary.
    """
    h, w = shape
    diag = (block.meta or {}).get("region_diagnostics") or {}
    boxes = list(diag.get("glyph_component_bboxes") or diag.get("component_bboxes") or [])
    if not boxes:
        return None
    clip = np.zeros((h, w), dtype=np.uint8)
    for box in boxes:
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        short = max(1, min(x1 - x0, y1 - y0))
        pad = int(np.clip(round(short * 0.20), 1, 3))
        x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
        x1 = min(w, x1 + pad); y1 = min(h, y1 + pad)
        if x1 > x0 and y1 > y0:
            clip[y0:y1, x0:x1] = 255
    return clip if cv2.countNonZero(clip) > 0 else None


def build_clear_mask(
    image_shape: tuple[int, int],
    target_blocks: list[TextBlock],
    target_units: list[TextUnit],
    target_bubbles: list[BubbleInstance],
    matches: list[UnitMatch],
    config: MaskingConfig | None = None,
    min_match_confidence: float = 0.0,
    allow_relations: set[str] | None = None,
    target_image: np.ndarray | None = None,
    current_image: np.ndarray | None = None,
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
            synthetic_region_pixel_mask = False
            if block_mask is None and bool(block.meta.get("synthetic_geometry_only")) and target_image is not None:
                block_mask = _synthetic_target_text_mask(
                    target_image, bubble_by_id.get(unit.bubble_id) if unit.bubble_id else None, cfg,
                    current_image=current_image,
                )
                if block_mask is not None and bool(block.meta.get("synthetic_region_only")):
                    # Target-driven reletter regions represent one text island
                    # inside a possibly compound balloon. Restrict synthetic ink
                    # recovery to this Region and then complete thin/missed target
                    # glyphs from the authoritative Japanese image.
                    region_clip = rasterize_polygon(block.polygon, (h, w))
                    if cv2.countNonZero(region_clip) > 0:
                        bubble = bubble_by_id.get(unit.bubble_id) if unit.bubble_id else None
                        protected = _protected_bubble_mask(bubble, cfg) if bubble is not None else None
                        if protected is not None and cv2.countNonZero(protected) > 0:
                            region_clip = cv2.bitwise_and(region_clip, protected)
                        component_clip = _glyph_component_clip(block, (h, w))
                        if component_clip is not None:
                            component_clip = cv2.bitwise_and(component_clip, region_clip)
                            if protected is not None and cv2.countNonZero(protected) > 0:
                                component_clip = cv2.bitwise_and(component_clip, protected)
                        destructive_clip = component_clip if component_clip is not None and cv2.countNonZero(component_clip) > 0 else region_clip
                        block_mask = cv2.bitwise_and(block_mask, destructive_clip)
                        block_mask = _complete_reletter_region_mask(target_image, bubble, destructive_clip, block_mask, cfg)
                        synthetic_region_pixel_mask = block_mask is not None and cv2.countNonZero(block_mask) > 0
            if block_mask is None:
                block_mask = rasterize_polygon(block.polygon, (h, w))
            has_pixel_mask = bool(mask_path and Path(mask_path).exists()) or synthetic_region_pixel_mask
            d = _dilation_pixels(block, cfg, has_pixel_mask=has_pixel_mask)
            if d > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
                block_mask = cv2.dilate(block_mask, k, iterations=1)
            if unit.bubble_id and unit.bubble_id in bubble_by_id:
                block_mask = _border_guard(block_mask, bubble_by_id.get(unit.bubble_id), target_image, cfg)
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
