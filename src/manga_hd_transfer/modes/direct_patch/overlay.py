from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ...text_only_transfer import (
    changed_text_masks,
    clear_text_components_to_local_paper,
    cleanup_target_residual_specks,
    transfer_text_only,
)


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)


def borderless_inner_mask(region_mask: np.ndarray, *, border_guard_px: int = 2, min_pixels: int = 80) -> np.ndarray:
    mask = (region_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(mask) == 0:
        return mask
    guard = max(0, int(border_guard_px))
    if guard > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard * 2 + 1, guard * 2 + 1))
        inner = cv2.erode(mask, k, iterations=1)
        if cv2.countNonZero(inner) >= int(min_pixels):
            return inner
    return mask


def classify_target_region(target: np.ndarray, region_mask: np.ndarray) -> dict[str, float | bool | str]:
    use = region_mask > 0
    if int(np.count_nonzero(use)) == 0:
        return {
            "kind": "empty",
            "white_ratio": 0.0,
            "dark_ratio": 0.0,
            "saturation_mean": 0.0,
            "high_sat_ratio": 0.0,
            "neutral_white": False,
            "colored": False,
        }
    gray = _gray(target)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1]
    white_ratio = float(np.mean(gray[use] >= 212))
    dark_ratio = float(np.mean(gray[use] <= 188))
    saturation_mean = float(np.mean(sat[use]))
    high_sat_ratio = float(np.mean(sat[use] >= 70))
    neutral_white = bool(white_ratio >= 0.54 and saturation_mean <= 42.0 and high_sat_ratio <= 0.28)
    colored = bool(high_sat_ratio >= 0.22 or saturation_mean >= 45.0)
    return {
        "kind": "white" if neutral_white else ("colored" if colored else "mixed"),
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "saturation_mean": saturation_mean,
        "high_sat_ratio": high_sat_ratio,
        "neutral_white": neutral_white,
        "colored": colored,
    }


def build_white_source_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    border_guard_px: int = 2,
    clear_target_text: bool = True,
    clear_dilate_px: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Borderless source-on-top overlay for white bubbles/textboxes.

    The source patch is applied only inside the container interior, so the target
    outline survives. Japanese glyphs are cleared first, then the already aligned
    SOURCE interior (paper + Chinese text) is copied on top of TARGET.
    """
    region_u8 = (region_mask > 0).astype(np.uint8) * 255
    empty = np.zeros(region_u8.shape, np.uint8)
    if cv2.countNonZero(region_u8) == 0:
        return target_region.copy(), empty.copy(), empty.copy(), {"mode": "white", "reason": "empty_region"}

    # Identify changed SOURCE/TARGET text so TARGET cleanup can remove JP even
    # when the final pasted white paper almost but not entirely covers it.
    src_text, tgt_text, diff_diag = changed_text_masks(source_region, target_region, region_u8, tolerance_px=2)
    if clear_target_text and cv2.countNonZero(tgt_text) > 0:
        if int(clear_dilate_px) > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clear_dilate_px * 2 + 1, clear_dilate_px * 2 + 1))
            tgt_text = cv2.dilate(tgt_text, k, iterations=1)
        base, clear_mask, clear_diag = clear_text_components_to_local_paper(
            target_region.copy(), target_region, tgt_text, region_u8
        )
    else:
        base = target_region.copy()
        clear_mask = empty.copy()
        clear_diag = {"local_paper_components": 0, "local_paper_clear_pixels": 0, "local_paper_rejected_components": 0}

    write_mask = borderless_inner_mask(region_u8, border_guard_px=border_guard_px)
    candidate = base.copy()
    use = write_mask > 0
    candidate[use] = source_region[use]
    changed = (np.any(candidate != target_region, axis=2)).astype(np.uint8) * 255
    # Keep target-border protection strict: writes happen only inside write_mask.
    changed[write_mask == 0] = 0
    source_payload = write_mask.copy()
    diag = {
        "mode": "white",
        "strategy": "borderless_source_overlay",
        "source_on_top": True,
        "target_underlay": True,
        "source_border_removed": True,
        "rotation_locked": True,
        "clear_target_text": bool(clear_target_text),
        "changed_text_masks": diff_diag,
        "clear_diag": clear_diag,
        "write_pixels": int(cv2.countNonZero(changed)),
        "payload_pixels": int(cv2.countNonZero(source_payload)),
        "source_text_pixels": int(cv2.countNonZero(src_text)),
        "target_text_pixels": int(cv2.countNonZero(tgt_text)),
    }
    return candidate, changed, source_payload, diag


def build_colored_text_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    target_clear_region_mask: np.ndarray | None = None,
    clear_dilate_px: int = 1,
    inpaint_radius: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Color-preserving direct mode path for coloured/open areas.

    Keeps the TARGET background and composites only changed SOURCE Chinese glyphs.
    """
    candidate, write_mask, source_text_mask, diag = transfer_text_only(
        target_region.copy(),
        source_region,
        (region_mask > 0).astype(np.uint8) * 255,
        tolerance_px=2,
        clear_dilate_px=max(0, int(clear_dilate_px)),
        inpaint_radius=float(inpaint_radius),
        white_container=False,
        localized_white_text=False,
        white_full_clear_enabled=False,
        target_clear_region_mask=target_clear_region_mask,
    )
    cleaned, residual_mask, residual_diag = cleanup_target_residual_specks(
        candidate, target_region, (region_mask > 0).astype(np.uint8) * 255, source_text_mask, write_mask,
        white_container=False, inpaint_radius=float(inpaint_radius),
    )
    final_mask = cv2.bitwise_or(write_mask, residual_mask)
    diag = dict(diag)
    diag.update({
        "mode": "colored",
        "strategy": "target_background_source_text_overlay",
        "source_on_top": True,
        "target_underlay": True,
        "source_border_removed": True,
        "rotation_locked": True,
        "residual_cleanup": residual_diag,
        "write_pixels": int(cv2.countNonZero(final_mask)),
        "source_text_pixels": int(cv2.countNonZero(source_text_mask)),
    })
    return cleaned, final_mask, source_text_mask, diag


def _refine_region_mask_with_support(
    region_mask: np.ndarray,
    support_mask: np.ndarray | None,
    *,
    white_mode: bool,
    min_keep_ratio: float = 0.45,
) -> np.ndarray:
    mask = (region_mask > 0).astype(np.uint8) * 255
    if support_mask is None or cv2.countNonZero(mask) == 0:
        return mask
    support = (support_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(support) == 0:
        return mask
    ksz = 9 if white_mode else 13
    grow = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz)), iterations=1)
    refined = cv2.bitwise_and(mask, grow)
    refined_px = int(cv2.countNonZero(refined))
    if refined_px < max(80, int(round(cv2.countNonZero(mask) * float(min_keep_ratio)))):
        return mask
    return refined


def compose_direct_overlay(
    target_region: np.ndarray,
    source_region: np.ndarray,
    region_mask: np.ndarray,
    *,
    white_mode: bool,
    support_mask: np.ndarray | None = None,
    border_guard_px: int = 2,
    clear_target_text: bool = True,
    clear_dilate_px: int = 1,
    inpaint_radius: float = 2.5,
    target_clear_region_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    effective_mask = _refine_region_mask_with_support(region_mask, support_mask, white_mode=bool(white_mode))
    if bool(white_mode):
        return build_white_source_overlay(
            target_region, source_region, effective_mask,
            border_guard_px=border_guard_px,
            clear_target_text=clear_target_text,
            clear_dilate_px=clear_dilate_px,
        )
    return build_colored_text_overlay(
        target_region, source_region, effective_mask,
        target_clear_region_mask=target_clear_region_mask,
        clear_dilate_px=clear_dilate_px,
        inpaint_radius=inpaint_radius,
    )
