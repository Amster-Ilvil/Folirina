from __future__ import annotations

"""Blank-first replacement for verified white speech-bubble interiors."""

from typing import Any
from dataclasses import dataclass

import cv2
import numpy as np

from .registration_geometry import registration_homography
from .region_alignment import aligned_source_roi


def _bubble_points(row: Any) -> np.ndarray | None:
    if hasattr(row, "to_dict"):
        row = row.to_dict()
    if not isinstance(row, dict):
        return None
    raw = row.get("polygon")
    if raw is None:
        raw = row.get("points")
    try:
        points = np.asarray(raw, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return None
    return points


def _polygon_mask(points: np.ndarray, shape: tuple[int, int], *, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), np.uint8)
    local = np.asarray(points, np.float32).copy()
    local[:, 0] -= int(offset[0]); local[:, 1] -= int(offset[1])
    cv2.fillPoly(out, [np.rint(local).astype(np.int32)], 255)
    return out


def _kind(row: dict[str, Any]) -> str:
    return str(row.get("kind") or row.get("type") or "speech").strip().lower()


def white_bubble_interior(target: np.ndarray, bubble: Any, *, inset: int = 3) -> tuple[np.ndarray | None, dict]:
    """Prove neutral white paper inside a semantic balloon, preserving its rim.

    Bright yellow, skin, gradients and arbitrary rectangular selections do not
    establish white-paper authority. Geometry alone never authorises a fill.
    """
    bubble = bubble.to_dict() if hasattr(bubble, "to_dict") else bubble
    if not isinstance(bubble, dict) or _kind(bubble) not in {"speech", "bubble", "dialogue", "text_bubble"}:
        return None, {"reason": "not_speech_bubble"}
    points = _bubble_points(bubble)
    confidence = float(bubble.get("confidence", 1.0) or 0.0)
    if points is None or confidence < 0.62:
        return None, {"reason": "unproven_bubble_geometry"}
    full = _polygon_mask(points, target.shape[:2])
    area = int(cv2.countNonZero(full))
    if area < 64 or area > target.shape[0] * target.shape[1] * 0.22:
        return None, {"reason": "invalid_bubble_area"}
    radius = max(2, min(12, int(inset)))
    inner = cv2.erode(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    pixels = target[inner > 0]
    if len(pixels) < 64:
        return None, {"reason": "empty_bubble_interior"}
    paper = (pixels.min(axis=1) >= 240) & (np.ptp(pixels, axis=1) <= 10)
    ratio = float(np.mean(paper))
    median = np.median(pixels, axis=0)
    diag = {"target_bubble_id": str(bubble.get("id") or ""), "white_ratio": ratio,
            "paper_median_bgr": median.tolist(), "inset_px": radius}
    if ratio < 0.68 or float(median.min()) < 248 or float(np.ptp(median)) > 4:
        return None, {**diag, "reason": "not_flat_neutral_white"}
    return inner, {**diag, "reason": "verified_white_bubble"}


def verified_white_bubble_mask(target: np.ndarray, bubbles: list, *, inset: int = 3) -> np.ndarray:
    union = np.zeros(target.shape[:2], np.uint8)
    for bubble in bubbles or ():
        inner, _ = white_bubble_interior(target, bubble, inset=inset)
        if inner is not None:
            union = cv2.bitwise_or(union, inner)
    return union


def fill_verified_white_under_mask(target: np.ndarray, image: np.ndarray, mask: np.ndarray, bubbles: list) -> np.ndarray:
    white = verified_white_bubble_mask(target, bubbles)
    use = (white > 0) & (np.asarray(mask) > 0)
    out = image.copy()
    out[use] = 255
    return out


def clean_white_mask_if_proven(target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict] | None:
    """Exact white fill for a manual erase mask inside closed white paper.

    This primitive never expands the caller's mask. It is also usable by older
    review projects without semantic bubble metadata.
    """
    if not np.any(mask):
        return None
    x, y, w, h = cv2.boundingRect(np.asarray(mask, np.uint8))
    pad = max(8, min(24, int(min(w, h) * .25)))
    seed = np.zeros(target.shape[:2], np.uint8)
    seed[max(0,y-pad):min(target.shape[0],y+h+pad),max(0,x-pad):min(target.shape[1],x+w+pad)] = 255
    from .target_text_removal import coloured_container_removal
    plan = coloured_container_removal(target, seed)
    if plan.diagnostics.get("reason") != "target_white_container_holes":
        return None
    contours, _ = cv2.findContours(plan.domain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist()
    inner, _ = white_bubble_interior(target, {"polygon": polygon, "confidence": 1.0, "kind": "speech"}, inset=2)
    if inner is None or np.any((mask > 0) & (inner == 0)):
        return None
    out = target.copy(); out[mask > 0] = 255
    return out, {"mode": "verified_white_mask_fill", "pixels": int(cv2.countNonZero(mask)), "model_invoked": False}


@dataclass
class WhiteBubbleComposition:
    image: np.ndarray
    clean_base: np.ndarray
    clear_mask: np.ndarray
    layer_rgba: np.ndarray | None
    diagnostics: dict


def compose_white_bubbles(target: np.ndarray, proposed: np.ndarray, raster_rgba: np.ndarray | None,
                          bubbles: list, *, text_rgba: np.ndarray | None = None,
                          scope: np.ndarray | None = None) -> WhiteBubbleComposition:
    """Blank verified white interiors before the mode's own translated layers.

    OCR keeps its generated text mask and font; raster modes keep their actual
    transformed SOURCE layer. No Japanese pixels from ``proposed`` are reused
    inside a rewritten balloon. A balloon without translated ink stays intact.
    """
    shape = target.shape[:2]
    out, base = proposed.copy(), target.copy()
    clear = np.zeros(shape, np.uint8)
    layer = raster_rgba.copy() if raster_rgba is not None else None
    for foreground in (raster_rgba, text_rgba):
        if foreground is not None and foreground.shape != (*shape, 4):
            raise ValueError("white bubble foreground dimensions differ")
    rows = []
    for bubble in bubbles or ():
        inner, diag = white_bubble_interior(target, bubble)
        if inner is None:
            continue
        if scope is not None:
            inner = cv2.bitwise_and(inner, np.asarray(scope, np.uint8))
        if not np.any(inner):
            continue
        x, y, w, h = cv2.boundingRect(inner)
        use = inner[y:y+h, x:x+w] > 0
        foregrounds = []
        ink_pixels = 0
        for foreground in (raster_rgba, text_rgba):
            if foreground is None:
                foregrounds.append(None)
                continue
            fg = foreground[y:y+h, x:x+w]
            # SOURCE paper is never a foreground. Keep glyph antialiasing and
            # the mode's opacity, while making the clean background pure white.
            active = use & (fg[..., 3] > 0) & (fg[..., :3].min(axis=2) < 249)
            ink_pixels += int(np.count_nonzero(active & (fg[..., :3].min(axis=2) < 210)))
            foregrounds.append((fg, active))
        if ink_pixels < 8:
            continue
        rendered = np.full((h, w, 3), 255, np.uint8)
        pixel_rendered = rendered.copy()
        for index, item in enumerate(foregrounds):
            if item is None:
                continue
            fg, active = item
            alpha = fg[..., 3:4].astype(np.float32) / 255.0
            top = cv2.cvtColor(fg[..., :3], cv2.COLOR_RGB2BGR)
            blend = np.clip(top.astype(np.float32) * alpha + rendered.astype(np.float32) * (1-alpha), 0, 255).astype(np.uint8)
            rendered[active] = blend[active]
            if index == 0:
                pixel_rendered = rendered.copy()
        out_roi = out[y:y+h, x:x+w]; out_roi[use] = rendered[use]
        base_roi = base[y:y+h, x:x+w]; base_roi[use] = 255
        clear = cv2.bitwise_or(clear, inner)
        if layer is not None:
            lr = layer[y:y+h, x:x+w]
            rgb = cv2.cvtColor(pixel_rendered, cv2.COLOR_BGR2RGB)
            lr[use, :3] = rgb[use]; lr[use, 3] = 255
        rows.append({**diag, "translated_ink_pixels": ink_pixels, "cleared_interior_pixels": int(np.count_nonzero(use)),
                     "target_pixels_reused_in_interior": False})
    return WhiteBubbleComposition(out, base, clear, layer, {
        "schema": "folirina.white_bubble_blank_first.v1", "applied": len(rows), "rows": rows,
        "policy": "verified_white_interior_then_mode_owned_chinese_layer",
    })


def compose_ocr_white_bubbles(target: np.ndarray, proposed: np.ndarray, text_mask: np.ndarray,
                              fill: tuple, bubbles: list) -> tuple[np.ndarray, dict]:
    foreground = np.zeros((*target.shape[:2], 4), np.uint8)
    foreground[..., :3] = fill
    foreground[..., 3] = text_mask
    result = compose_white_bubbles(target, proposed, None, bubbles, text_rgba=foreground)
    contours, _ = cv2.findContours(result.clear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    diag = dict(result.diagnostics)
    diag["white_bubble_polygons"] = [c.reshape(-1, 2).astype(int).tolist() for c in contours if len(c) >= 3]
    return result.image, diag


def _choose_target_bubble(
    project: dict[str, Any], selection: np.ndarray, row: dict[str, Any], shape: tuple[int, int], inset: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, dict[str, Any]]:
    bubbles = project.get("target_bubbles")
    if not isinstance(bubbles, list):
        bubbles = []
    requested_id = str(row.get("target_bubble_id") or row.get("bubble_id") or "").strip()
    selection_area = max(1, int(cv2.countNonZero(selection)))
    candidates: list[tuple[float, float, dict[str, Any], np.ndarray, np.ndarray]] = []
    for bubble in bubbles:
        if not isinstance(bubble, dict) or _kind(bubble) not in {"speech", "bubble", "dialogue", "text_bubble"}:
            continue
        bubble_id = str(bubble.get("id") or "")
        if requested_id and bubble_id != requested_id:
            continue
        confidence = float(bubble.get("confidence", 1.0) or 0.0)
        if confidence < 0.62:
            continue
        points = _bubble_points(bubble)
        if points is None:
            continue
        full = _polygon_mask(points, shape)
        area = int(cv2.countNonZero(full))
        fraction = area / float(max(1, shape[0] * shape[1]))
        if fraction < 0.00035 or fraction > 0.22:
            continue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
        inner = cv2.erode(full, k, iterations=1) if inset else full.copy()
        inner_area = int(cv2.countNonZero(inner))
        if inner_area < max(64, int(area * 0.72)):
            continue
        if requested_id:
            containment = 1.0
        else:
            contained = cv2.bitwise_and(selection, inner)
            containment = cv2.countNonZero(contained) / float(selection_area)
            # A hand-drawn box may include a little of the balloon outline, but
            # most of it must lie inside exactly one detected speech bubble.
            if containment < 0.62:
                continue
        score = 0.72 * containment + 0.28 * confidence
        candidates.append((score, containment, bubble, full, inner))
    if not candidates:
        suffix = f" ({requested_id})" if requested_id else ""
        raise ValueError("选区内没有可确认的完整白底气泡" + suffix)
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not requested_id and len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.035:
        raise ValueError("选区同时覆盖多个气泡，请缩小选区后再应用")
    score, containment, chosen, full, inner = candidates[0]
    return chosen, full, inner, {
        "target_bubble_id": str(chosen.get("id") or ""),
        "target_bubble_confidence": float(chosen.get("confidence", 1.0) or 0.0),
        "selection_containment": float(containment),
        "selection_score": float(score),
        "target_polygon_pixels": int(cv2.countNonZero(full)),
        "target_interior_pixels": int(cv2.countNonZero(inner)),
    }


def _match_source_bubble(
    project: dict[str, Any], target_bubble: dict[str, Any], target_mask: np.ndarray,
    source_shape: tuple[int, int], target_shape: tuple[int, int], row: dict[str, Any],
) -> tuple[dict[str, Any], np.ndarray, float]:
    bubbles = project.get("source_bubbles")
    if not isinstance(bubbles, list):
        bubbles = []
    requested_id = str(row.get("source_bubble_id") or "").strip()
    H = registration_homography(project).copy()
    dx = int(row.get("source_offset_x", 0) or 0); dy = int(row.get("source_offset_y", 0) or 0)
    if dx or dy:
        H = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]]) @ H
    best: tuple[float, dict[str, Any], np.ndarray] | None = None
    target_area = max(1, int(cv2.countNonZero(target_mask)))
    for bubble in bubbles:
        if not isinstance(bubble, dict) or _kind(bubble) not in {"speech", "bubble", "dialogue", "text_bubble"}:
            continue
        if requested_id and str(bubble.get("id") or "") != requested_id:
            continue
        points = _bubble_points(bubble)
        if points is None:
            continue
        projected = cv2.perspectiveTransform(points.reshape(1, -1, 2), H).reshape(-1, 2)
        mapped = _polygon_mask(projected, target_shape)
        intersection = int(cv2.countNonZero(cv2.bitwise_and(mapped, target_mask)))
        union = int(cv2.countNonZero(cv2.bitwise_or(mapped, target_mask)))
        iou = intersection / float(max(1, union))
        target_coverage = intersection / float(target_area)
        score = 0.75 * iou + 0.25 * target_coverage
        if best is None or score > best[0]:
            best = (score, bubble, mapped)
    if best is None or best[0] < 0.38:
        raise ValueError("SOURCE 中没有与选中气泡可靠配对的中文气泡")
    return best[1], best[2], float(best[0])


def apply_white_bubble_blank_first(
    current: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    project: dict[str, Any],
    row: dict[str, Any],
    selection: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Clear a verified white TARGET speech balloon, then place only SOURCE ink.

    The persisted TARGET bubble polygon is the write boundary.  A manual
    selection chooses one bubble; it does not constrain writes to the old small
    Japanese text box.  TARGET pixels are replaced with white inside the
    inset polygon, then registered dark SOURCE ink is composited on top.
    """
    if target.ndim != 3 or target.shape[2] != 3 or current.shape != target.shape:
        raise ValueError("白底气泡处理需要尺寸一致的彩色 TARGET 与当前结果")
    h, w = target.shape[:2]
    if np.asarray(selection).shape != (h, w):
        raise ValueError("选区尺寸与 TARGET 不一致")
    inset = max(2, min(12, int(row.get("inset_px", 4) or 4)))
    target_bubble, target_full, target_inner, target_diag = _choose_target_bubble(
        project, np.asarray(selection, np.uint8), row, (h, w), inset
    )
    target_inner, white_diag = white_bubble_interior(target, target_bubble, inset=inset)
    if target_inner is None:
        raise ValueError("选中的气泡不是经过验证的纯白底气泡，未修改")
    white_ratio = float(white_diag["white_ratio"])

    source_container_fallback = False
    try:
        source_bubble, source_projected, match_score = _match_source_bubble(
            project, target_bubble, target_full, source.shape[:2], target.shape[:2], row
        )
    except ValueError:
        # SOURCE detectors occasionally miss a short translation or ellipsis.
        # Prove its closed white container from the registered raster instead
        # of treating missing detector metadata as missing Chinese content.
        from .target_text_removal import coloured_container_removal
        aligned_page, _ = aligned_source_roi(
            source, (h, w), project, (0, 0, w, h),
            source_offset_x=int(row.get("source_offset_x", 0) or 0),
            source_offset_y=int(row.get("source_offset_y", 0) or 0),
        )
        plan = coloured_container_removal(aligned_page, target_full)
        intersection = cv2.countNonZero(cv2.bitwise_and(plan.domain, target_full))
        union = cv2.countNonZero(cv2.bitwise_or(plan.domain, target_full))
        match_score = intersection / float(max(1, union))
        if plan.diagnostics.get("reason") != "target_white_container_holes" or match_score < 0.50:
            raise ValueError("SOURCE 检测及闭合白底轮廓均未确认对应中文气泡，未修改")
        source_bubble = {"id": "registered-white-source:" + str(target_bubble.get("id") or "")}
        source_projected = plan.domain
        source_container_fallback = True
    x, y, bw, bh = cv2.boundingRect(target_inner)
    x1, y1 = min(w, x + bw), min(h, y + bh)
    roi = (x, y, x1, y1)
    aligned, identity = aligned_source_roi(
        source, (h, w), project, roi,
        source_offset_x=int(row.get("source_offset_x", 0) or 0),
        source_offset_y=int(row.get("source_offset_y", 0) or 0),
    )
    inner_roi = target_inner[y:y1, x:x1]
    projected_roi = source_projected[y:y1, x:x1].copy()
    source_safe = cv2.erode(
        projected_roi,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1)),
        iterations=1,
    )
    support = (inner_roi > 0) & (source_safe > 0)
    if int(np.count_nonzero(support)) < max(64, int(np.count_nonzero(inner_roi) * 0.55)):
        raise ValueError("SOURCE/TARGET 气泡内区重叠不足，未修改")

    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    core = (gray <= int(row.get("ink_core_threshold", 198) or 198)) & support
    if int(np.count_nonzero(core)) < 8:
        raise ValueError("配准后的 SOURCE 气泡内没有足够中文笔画，未修改")
    soft_near = cv2.dilate(core.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
    source_ink = support & soft_near & (gray <= int(row.get("ink_aa_threshold", 248) or 248))
    ink_pixels = int(np.count_nonzero(source_ink))
    if ink_pixels < 8:
        raise ValueError("没有提取到可用 SOURCE 中文笔画，未修改")

    out = current.copy()
    roi_out = out[y:y1, x:x1]
    roi_out[inner_roi > 0] = (255, 255, 255)
    roi_out[source_ink] = aligned[source_ink]
    clear = np.zeros((h, w), np.uint8)
    clear[y:y1, x:x1] = inner_roi
    layer = np.zeros((h, w, 4), np.uint8)
    layer_roi = layer[y:y1, x:x1]
    layer_roi[source_ink, :3] = aligned[source_ink]
    layer_roi[source_ink, 3] = 255

    out_gray = cv2.cvtColor(roi_out, cv2.COLOR_BGR2GRAY)
    residual_dark = int(np.count_nonzero((out_gray < 190) & (inner_roi > 0) & (~source_ink)))
    white_outside_ink = int(np.count_nonzero((inner_roi > 0) & (~source_ink) & np.all(roi_out == 255, axis=2)))
    bx, by, bwidth, bheight = cv2.boundingRect(target_full)
    diag = {
        **target_diag,
        "mode": "white_bubble_blank_first",
        "source_bubble_id": str(source_bubble.get("id") or ""),
        "source_target_polygon_match_score": float(match_score),
        "source_closed_white_container_fallback": source_container_fallback,
        "target_white_ratio": white_ratio,
        "inset_px": inset,
        "source_ink_core_pixels": int(np.count_nonzero(core)),
        "source_ink_pixels": ink_pixels,
        "target_dark_pixels_before": int(np.count_nonzero((cv2.cvtColor(target[y:y1, x:x1], cv2.COLOR_BGR2GRAY) < 190) & (inner_roi > 0))),
        "residual_target_dark_pixels_after": residual_dark,
        "white_fill_pixels": white_outside_ink,
        "source_to_target_identity": bool(identity),
        "write_scope": "inset_verified_white_speech_bubble_polygon",
        "bubble_bbox": [int(bx), int(by), int(bx + bwidth), int(by + bheight)],
        "bubble_interior_bbox": [int(x), int(y), int(x1), int(y1)],
    }
    return out, layer, clear, target_inner, diag


__all__ = ["apply_white_bubble_blank_first", "white_bubble_interior", "verified_white_bubble_mask", "compose_white_bubbles"]
