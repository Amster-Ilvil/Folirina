from __future__ import annotations

"""OCR-owned background cleanup.

This module is intentionally imported only by OCR-capable transfer routes.
Direct, pure Mask, and Reveal do not depend on this cleanup policy.
"""

from typing import Any

import cv2
import numpy as np

from ...geometry import rasterize_polygon
from .masking_ops import MaskBuildResult
from .text_transfer import clear_text_components_to_local_paper, clear_broad_neutral_paper_components

def _ocr_paper_first_clear(
    base: np.ndarray,
    target: np.ndarray,
    mask_result: MaskBuildResult,
    target_units: list[Any],
    target_bubbles: list[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Clear OCR/reletter text on proven TARGET paper before interpolation.

    Two paper proofs are deliberately combined:

    1. **Broad neutral component proof** handles OCR text-box rectangles.  These
       broad masks were the source of p-005's triangular grey shadows because
       inpainting was asked to invent an already-white rectangle.
    2. **Per-unit local ring proof** retains the older conservative path for
       smaller/irregular glyph masks whose local surroundings prove white paper.

    Only the OCR product routes call this helper; Direct, pure Mask and Reveal
    never enter it.
    """
    cleaned, broad_handled, broad_changed, broad_diag = clear_broad_neutral_paper_components(
        base, target, mask_result.mask,
    )
    accepted = broad_handled.copy()
    unit_by_id = {str(getattr(u, "id", "")): u for u in target_units}
    bubble_by_id = {str(getattr(b, "id", "")): b for b in target_bubbles}
    kept_components = rejected_components = 0

    broad_inv = cv2.bitwise_not(broad_handled)
    for unit_id, unit_clear in (mask_result.per_unit or {}).items():
        if unit_clear is None or cv2.countNonZero(unit_clear) == 0:
            continue
        unit_pending = cv2.bitwise_and(unit_clear, broad_inv)
        if cv2.countNonZero(unit_pending) == 0:
            continue
        unit = unit_by_id.get(str(unit_id))
        if unit is None:
            continue
        region = None
        bubble_id = str(getattr(unit, "bubble_id", "") or "")
        if bubble_id and bubble_id in bubble_by_id:
            region = getattr(bubble_by_id[bubble_id], "safe_mask", None)
        if region is None or region.shape[:2] != target.shape[:2] or cv2.countNonZero(region) == 0:
            region = rasterize_polygon(getattr(unit, "polygon", []) or [], target.shape[:2])
            if cv2.countNonZero(region) > 0:
                # Give the local-paper detector a small TARGET ring around text
                # regions that do not have an explicit parent balloon.
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
                region = cv2.dilate(region, k, iterations=1)
        if region is None or cv2.countNonZero(region) == 0:
            continue
        cleaned, local, diag = clear_text_components_to_local_paper(
            cleaned, target, unit_pending, region,
        )
        if local is not None and cv2.countNonZero(local) > 0:
            accepted = cv2.bitwise_or(accepted, local)
        kept_components += int(diag.get("local_paper_components", 0) or 0)
        rejected_components += int(diag.get("local_paper_rejected_components", 0) or 0)

    remaining = mask_result.mask.copy()
    if cv2.countNonZero(accepted) > 0:
        remaining[accepted > 0] = 0
    return cleaned, remaining, {
        **broad_diag,
        "paper_clear_pixels": int(cv2.countNonZero(accepted)),
        "paper_changed_pixels": int(cv2.countNonZero(broad_changed)),
        "remaining_inpaint_pixels": int(cv2.countNonZero(remaining)),
        "paper_components": int(kept_components) + int(broad_diag.get("broad_paper_components", 0) or 0),
        "paper_rejected_components": int(rejected_components) + int(broad_diag.get("broad_paper_rejected_components", 0) or 0),
    }


# Compatibility alias for tests/plugins written against the v2.3.11 symbol.
# Semantics are now the stronger OCR paper-first implementation.
def _reletter_paper_first_clear(
    base: np.ndarray, target: np.ndarray, mask_result: MaskBuildResult,
    target_units: list[Any], target_bubbles: list[Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    return _ocr_paper_first_clear(base, target, mask_result, target_units, target_bubbles)

__all__ = ["_ocr_paper_first_clear", "_reletter_paper_first_clear"]
