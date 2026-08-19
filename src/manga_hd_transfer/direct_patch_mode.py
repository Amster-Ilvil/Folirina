from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .modes.direct_patch.overlay import classify_target_region, compose_direct_overlay


@dataclass
class RegionResult:
    id: str
    source_id: str
    target_id: str
    confidence: float
    region_kind: str
    bbox_target: List[int]
    mode: str
    applied: bool
    reason: str
    mask_pixels: int
    changed_pixels: int
    white_ratio: float = 0.0
    saturation_mean: float = 0.0


@dataclass
class ModeResult:
    accepted: bool
    reason: str
    used: bool
    requested_mode: str
    strategy: str
    registration_confidence: float
    mask_pixels: int
    changed_pixels: int
    changed_ratio: float
    outside_mask_unchanged: bool
    target_shape: List[int]
    regions: List[RegionResult]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _imread(path: Path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return img


def _imwrite(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise RuntimeError(f"Unable to encode image for: {path}")
    encoded.tofile(str(path))


def _polygon_to_mask(shape: Tuple[int, int], polygon: Sequence[Sequence[float]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if not polygon:
        return mask
    pts = np.array([[int(round(x)), int(round(y))] for x, y in polygon], dtype=np.int32)
    if pts.size:
        cv2.fillPoly(mask, [pts], 255)
    return mask


def _mask_bbox(mask: np.ndarray) -> List[int]:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    union = int(np.count_nonzero((a > 0) | (b > 0)))
    return inter / union if union else 0.0


def _warp_polygon(polygon: Sequence[Sequence[float]], matrix: np.ndarray) -> List[List[float]]:
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if pts.size == 0:
        return []
    warped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return warped.astype(float).tolist()


def _load_registration(page_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    reg = _load_json(page_dir / ".cache" / "registration.json")["payload"]
    matrix = np.array(reg["matrix"], dtype=np.float32)
    return matrix, reg


def _load_layout(page_dir: Path, role: str) -> Dict[str, Any]:
    return _load_json(page_dir / ".cache" / f"layout_{role}.json")["payload"]


def _bubble_pairs(source_layout: Dict[str, Any], target_layout: Dict[str, Any], matrix: np.ndarray, shape: Tuple[int, int]) -> List[Dict[str, Any]]:
    src_items = [it for it in source_layout.get("items", []) if it.get("label") == "bubble" and float(it.get("confidence", 0.0)) >= 0.70]
    dst_items = [it for it in target_layout.get("items", []) if it.get("label") == "bubble" and float(it.get("confidence", 0.0)) >= 0.70]
    warped_src = []
    for i, s in enumerate(src_items):
        poly = _warp_polygon(s.get("polygon", []), matrix)
        smask = _polygon_to_mask(shape, poly)
        warped_src.append((i, s, poly, smask))
    used_src: set[int] = set()
    pairs: List[Dict[str, Any]] = []
    for di, d in enumerate(dst_items):
        dmask = _polygon_to_mask(shape, d.get("polygon", []))
        best = None
        for si, s, poly, smask in warped_src:
            if si in used_src:
                continue
            iou = _mask_iou(dmask, smask)
            if best is None or iou > best[0]:
                best = (iou, si, s, poly, smask)
        if best is None or best[0] < 0.20:
            continue
        iou, si, s, poly, smask = best
        used_src.add(si)
        pairs.append({
            "id": f"direct-pair-{di:03d}",
            "source_id": f"source-bubble-{si:03d}",
            "target_id": f"target-bubble-{di:03d}",
            "confidence": float(min(float(s.get("confidence", 0.0)), float(d.get("confidence", 0.0)), 0.45 + 0.55 * iou)),
            "target_mask": dmask,
            "source_mask_warped": smask,
        })
    return pairs


def run_direct_patch_mode(page_dir: str | Path) -> ModeResult:
    page_dir = Path(page_dir)
    source = _imread(page_dir / "source_original.png")
    target = _imread(page_dir / "target_original.png")
    th, tw = target.shape[:2]
    matrix, reg = _load_registration(page_dir)
    source_layout = _load_layout(page_dir, "source")
    target_layout = _load_layout(page_dir, "target")
    warped_source = cv2.warpPerspective(source, matrix, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    pairs = _bubble_pairs(source_layout, target_layout, matrix, (th, tw))
    final = target.copy()
    layer = np.zeros((th, tw, 4), dtype=np.uint8)
    union_mask = np.zeros((th, tw), dtype=np.uint8)
    regions: List[RegionResult] = []

    for pair in pairs:
        mask = pair["target_mask"]
        if cv2.countNonZero(mask) <= 0:
            continue
        x0, y0, x1, y1 = _mask_bbox(mask)
        if x1 <= x0 or y1 <= y0:
            continue
        local_mask = mask[y0:y1, x0:x1]
        src_crop = warped_source[y0:y1, x0:x1]
        tgt_crop = target[y0:y1, x0:x1]
        appearance = classify_target_region(tgt_crop, local_mask)
        is_white = bool(appearance.get("neutral_white", False))
        candidate, write_mask, source_mask, diag = compose_direct_overlay(
            tgt_crop,
            src_crop,
            local_mask,
            white_mode=is_white,
            support_mask=pair["source_mask_warped"][y0:y1, x0:x1],
            border_guard_px=2,
            clear_target_text=True,
            clear_dilate_px=1,
            inpaint_radius=2.5,
            target_clear_region_mask=local_mask,
        )
        if cv2.countNonZero(write_mask) <= 0:
            continue
        local_use = write_mask > 0
        dst = final[y0:y1, x0:x1]
        before = dst.copy()
        dst[local_use] = candidate[local_use]
        final[y0:y1, x0:x1] = dst
        changed = (np.any(dst != before, axis=2)).astype(np.uint8) * 255
        if cv2.countNonZero(changed) <= 0:
            continue
        union_mask[y0:y1, x0:x1] = cv2.bitwise_or(union_mask[y0:y1, x0:x1], changed)
        layer_crop = layer[y0:y1, x0:x1]
        layer_crop[changed > 0, :3] = dst[changed > 0][:, ::-1]
        layer_crop[..., 3] = np.maximum(layer_crop[..., 3], changed)
        layer[y0:y1, x0:x1] = layer_crop
        diag_support_pixels = int(cv2.countNonZero((pair["source_mask_warped"][y0:y1, x0:x1] > 0).astype(np.uint8) * 255))
        regions.append(RegionResult(
            id=pair["id"],
            source_id=pair["source_id"],
            target_id=pair["target_id"],
            confidence=float(pair["confidence"]),
            region_kind=str(appearance.get("kind", "mixed")),
            bbox_target=[x0, y0, x1, y1],
            mode=str(diag.get("mode", "colored")),
            applied=True,
            reason=str(diag.get("strategy", "direct_overlay")),
            mask_pixels=int(cv2.countNonZero(local_mask)),
            changed_pixels=int(cv2.countNonZero(changed)),
            white_ratio=float(appearance.get("white_ratio", 0.0)),
            saturation_mean=float(appearance.get("saturation_mean", 0.0)),
        ))
        regions[-1].reason = f"{regions[-1].reason};support_pixels={diag_support_pixels}"

    _imwrite(page_dir / "direct_patch_layer.png", layer)
    _imwrite(page_dir / "direct_patch_regions.png", union_mask)
    _imwrite(page_dir / "final.png", final)
    payload = {
        "schema": "manga_hd_translation_transfer.direct_patch.v2",
        "contract": "borderless_source_overlay_target_underlay",
        "requested_mode": "direct_patch",
        "strategy": "direct_borderless_overlay",
        "registration_confidence": float(reg.get("confidence", 0.0)),
        "regions": [r.__dict__ for r in regions],
    }
    _save_json(page_dir / "direct_patch.json", payload)

    mask_pixels = int(cv2.countNonZero(union_mask))
    changed_pixels = int(np.count_nonzero(np.any(final != target, axis=2)))
    outside = union_mask == 0
    outside_unchanged = bool(np.array_equal(final[outside], target[outside]))
    return ModeResult(
        accepted=bool(regions),
        reason="ok" if regions else "no_regions_applied",
        used=bool(regions),
        requested_mode="direct_patch",
        strategy="direct_borderless_overlay",
        registration_confidence=float(reg.get("confidence", 0.0)),
        mask_pixels=mask_pixels,
        changed_pixels=changed_pixels,
        changed_ratio=float(changed_pixels / max(1, th * tw)),
        outside_mask_unchanged=outside_unchanged,
        target_shape=[th, tw],
        regions=regions,
    )
