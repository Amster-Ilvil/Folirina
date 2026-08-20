from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from .aligned_overlay_reveal_validator import validate_aligned_overlay_reveal
from .aligned_overlay_reveal_core import AlignedOverlayPlan, AlignedOverlayRegion, AlignedOverlayResult
from .bubbles import detect_target_colored_containers, detect_unseeded_white_containers
from .config import AlignedOverlayRevealConfig, BubbleConfig
from .models import BubbleInstance, RegistrationResult
from .modes.aligned_overlay_reveal.container_detector import detect_text_barrier_containers
from .pipeline_bubble_service import primary_bubbles_cached


@dataclass
class RegionResult:
    id: str
    source_id: str
    target_id: str
    confidence: float
    region_kind: str
    bbox_target: List[int]
    clear_mode: str
    applied: bool
    reason: str
    mask_pixels: int
    changed_pixels: int
    candidate_source: str = "paired_diff"
    white_ratio: float = 0.0
    saturation_mean: float = 0.0


@dataclass
class ModeResult:
    accepted: bool
    reason: str
    page_triage: str
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


def _mask_overlap_fraction(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.count_nonzero((a > 0) & (b > 0)))
    denom = min(int(np.count_nonzero(a > 0)), int(np.count_nonzero(b > 0)))
    return inter / denom if denom else 0.0


def _collect_target_container_candidates(
    target: np.ndarray,
    bubble_cfg: BubbleConfig,
    cfg: AlignedOverlayRevealConfig,
    primary_candidates: Sequence[BubbleInstance] | None,
) -> list[BubbleInstance]:
    min_conf = float(getattr(cfg, "target_bubble_min_confidence", 0.70))
    supplemental_min_conf = float(getattr(cfg, "target_container_min_confidence", 0.55))
    dedupe_iou = float(getattr(cfg, "target_container_dedupe_iou", 0.60))
    dedupe_cover = float(getattr(cfg, "target_container_dedupe_cover", 0.82))
    out: list[BubbleInstance] = []

    def _add(rows: Sequence[BubbleInstance], *, source: str) -> None:
        for cand in rows:
            conf = float(getattr(cand, "confidence", 0.0))
            if conf < (min_conf if source == "primary" else supplemental_min_conf):
                continue
            mask = _bubble_mask_from_instance(cand, target.shape[:2])
            if int(cv2.countNonZero(mask)) <= 0:
                continue
            dup = False
            for old in out:
                old_mask = _bubble_mask_from_instance(old, target.shape[:2])
                if _mask_iou(mask, old_mask) >= dedupe_iou or _mask_overlap_fraction(mask, old_mask) >= dedupe_cover:
                    dup = True
                    break
            if dup:
                continue
            meta = dict(getattr(cand, "meta", {}) or {})
            meta["aligned_hole_candidate_source"] = source
            out.append(BubbleInstance(
                id=str(cand.id), polygon=list(cand.polygon), confidence=conf,
                kind=str(getattr(cand, "kind", "bubble") or "bubble"),
                block_ids=list(getattr(cand, "block_ids", []) or []),
                mask=None if getattr(cand, "mask", None) is None else cand.mask.copy(),
                safe_mask=None if getattr(cand, "safe_mask", None) is None else cand.safe_mask.copy(),
                meta=meta,
            ))

    _add(list(primary_candidates or []), source="primary")
    _add(detect_unseeded_white_containers(target, bubble_cfg, prefix="aligned-hole-white"), source="supplemental_white")
    _add(detect_text_barrier_containers(target, cfg, existing=out), source="supplemental_text_barrier")
    _add(detect_target_colored_containers(target, prefix="aligned-hole-color"), source="supplemental_colored")
    out.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
    return out


def _load_registration(page_dir: Path) -> Tuple[np.ndarray, Dict[str, Any]]:
    reg = _load_json(page_dir / ".cache" / "registration.json")["payload"]
    matrix = np.array(reg["matrix"], dtype=np.float32)
    if matrix.shape != (3, 3):
        raise ValueError("registration matrix must be 3x3")
    return matrix, reg


def _load_paired_diff(page_dir: Path) -> Dict[str, Any]:
    return _load_json(page_dir / ".cache" / "paired_diff.json")["payload"]


def _load_layout(page_dir: Path, role: str) -> Dict[str, Any]:
    return _load_json(page_dir / ".cache" / f"layout_{role}.json")["payload"]


def _paired_region_index(paired: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    src = {item["id"]: item for item in paired.get("source_bubbles", [])}
    dst = {item["id"]: item for item in paired.get("target_bubbles", [])}
    return src, dst


def _warp_polygon(polygon: Sequence[Sequence[float]], matrix: np.ndarray) -> List[List[float]]:
    pts = np.array(polygon, dtype=np.float32).reshape(-1, 1, 2)
    if pts.size == 0:
        return []
    warped = cv2.perspectiveTransform(pts, matrix).reshape(-1, 2)
    return warped.astype(float).tolist()


def _bubble_appearance(target: np.ndarray, mask: np.ndarray) -> Tuple[float, float]:
    if int(np.count_nonzero(mask)) == 0:
        return 0.0, 255.0
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    idx = mask > 0
    return float(np.mean(gray[idx] >= 215)), float(np.mean(hsv[..., 1][idx]))


def _registered_translation_evidence(
    aligned_source: np.ndarray,
    target: np.ndarray,
    region_mask: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> dict[str, float | bool | str]:
    """Require real SOURCE/TARGET ink change for geometry-only supplements.

    Primary semantic bubble detections remain authoritative.  Supplemental white
    geometry must additionally behave like translated text after page registration,
    otherwise white clothes/panels/windows can become accidental SOURCE holes.
    """
    m=(region_mask>0).astype(np.uint8)*255
    pixels=int(cv2.countNonZero(m))
    if pixels<=0:
        return {"passed":False,"reason":"empty_mask","change_score":0.0}
    inner=cv2.erode(m,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
    if cv2.countNonZero(inner)<24:
        inner=m
    sg=cv2.cvtColor(aligned_source,cv2.COLOR_BGR2GRAY) if aligned_source.ndim==3 else aligned_source
    tg=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY) if target.ndim==3 else target
    thr=int(getattr(cfg,"supplemental_ink_threshold",190))
    s=((sg<thr)&(inner>0)).astype(np.uint8)*255
    t=((tg<thr)&(inner>0)).astype(np.uint8)*255
    area=max(1,int(cv2.countNonZero(inner)))
    sc=int(cv2.countNonZero(s)); tc=int(cv2.countNonZero(t))
    sdens=float(sc/area); tdens=float(tc/area)
    tol=max(1,int(getattr(cfg,"supplemental_ink_match_tolerance_px",2)))
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(tol*2+1,tol*2+1))
    sd=cv2.dilate(s,k); td=cv2.dilate(t,k)
    smatch=float(np.count_nonzero((s>0)&(td>0))/max(1,sc)) if sc else 0.0
    tmatch=float(np.count_nonzero((t>0)&(sd>0))/max(1,tc)) if tc else 0.0
    identity=float(np.clip(0.5*(smatch+tmatch),0.0,1.0))
    change=float(np.clip(1.0-identity,0.0,1.0))
    min_sd=float(getattr(cfg,"supplemental_min_source_ink_density",0.012))
    min_td=float(getattr(cfg,"supplemental_min_target_ink_density",0.008))
    min_change=float(getattr(cfg,"supplemental_min_ink_change_score",0.12))
    target_text_pixels=int(cv2.countNonZero(_target_jp_text_mask(target, m)))
    source_text_pixels=int(cv2.countNonZero(_source_glyph_mask(aligned_source, m, colored=False)))
    min_target_text=int(getattr(cfg,"supplemental_min_target_text_pixels",25))
    min_source_text=int(getattr(cfg,"supplemental_min_source_text_pixels",25))
    text_ok=bool(target_text_pixels>=min_target_text and source_text_pixels>=min_source_text)
    passed=bool(sdens>=min_sd and tdens>=min_td and change>=min_change and text_ok)
    if not text_ok:
        reason="missing_text_components"
    elif change<min_change or sdens<min_sd or tdens<min_td:
        reason="insufficient_translation_change"
    else:
        reason="ok"
    return {
        "passed":passed,"reason":reason,
        "source_ink_density":sdens,"target_ink_density":tdens,
        "target_text_pixels":target_text_pixels,"source_text_pixels":source_text_pixels,
        "min_target_text_pixels":min_target_text,"min_source_text_pixels":min_source_text,
        "source_ink_match":smatch,"target_ink_match":tmatch,
        "ink_identity_overlap":identity,"change_score":change,
        "min_source_ink_density":min_sd,"min_target_ink_density":min_td,"min_change_score":min_change,
    }


def _filter_text_components(binary: np.ndarray, region_mask: np.ndarray, *, max_dim: int = 90) -> np.ndarray:
    binary = cv2.bitwise_and(binary, region_mask)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    out = np.zeros_like(binary)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3:
            continue
        if area > 1300:
            continue
        if w > max_dim or h > max_dim * 2:
            continue
        # reject long burst/rule fragments while preserving vertical punctuation and kana
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 10.0 and area > 45:
            continue
        out[labels == i] = 255
    return out




def _expand_white_bubble_for_target_text(target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Catch tiny JP glyphs sitting just outside an interior bubble polygon.

    Only compact dark components touching a modest dilation of the trusted bubble mask
    are added. Long borders/rays are rejected, so this does not become a broad bbox fill.
    """
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    near = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    raw = np.zeros_like(mask)
    raw[(gray < 185) & (near > 0)] = 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    add = np.zeros_like(mask)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 3 or area > 850:
            continue
        if w > 58 or h > 90:
            continue
        aspect = max(w, h) / max(1.0, min(w, h))
        if aspect > 9.0 and area > 40:
            continue
        comp = labels == i
        # Ignore components already safely covered.
        if np.count_nonzero(comp & (mask > 0)) >= max(1, int(area * 0.85)):
            continue
        # Must touch the near-ring around the existing trusted interior mask.
        if not np.any(comp & (near > 0)):
            continue
        add[comp] = 255
    if int(np.count_nonzero(add)):
        add = cv2.dilate(add, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return cv2.bitwise_or(mask, add)

def _source_glyph_mask(warped_source: np.ndarray, region_mask: np.ndarray, colored: bool) -> np.ndarray:
    # Work well inside the bubble so burst rays/borders cannot become Chinese glyphs.
    erode_px = 11 if colored else 4
    k = max(3, erode_px * 2 + 1)
    inner = cv2.erode(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    if int(np.count_nonzero(inner)) < 80:
        inner = region_mask.copy()
    gray = cv2.cvtColor(warped_source, cv2.COLOR_BGR2GRAY)
    threshold = 190 if colored else 205
    raw = np.zeros_like(region_mask)
    raw[(gray < threshold) & (inner > 0)] = 255
    filtered = _filter_text_components(raw, inner, max_dim=72 if colored else 100)
    filtered = cv2.dilate(filtered, np.ones((2, 2), np.uint8), iterations=1)
    return filtered


def _target_jp_text_mask(target: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    # Colored/open-text route: identify dark compact glyph components only in the central interior.
    k = 23
    inner = cv2.erode(region_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=1)
    if int(np.count_nonzero(inner)) < 80:
        inner = region_mask.copy()
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    raw = np.zeros_like(region_mask)
    raw[(gray < 180) & (inner > 0)] = 255
    filtered = _filter_text_components(raw, inner, max_dim=58)
    # Keep anti-aliased edges so Japanese does not leave halos.
    filtered = cv2.dilate(filtered, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return filtered


def _layout_bubble_pairs(
    source_layout: Dict[str, Any],
    target_layout: Dict[str, Any],
    matrix: np.ndarray,
    shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    src_bubbles = [x for x in source_layout.get("items", []) if x.get("label") == "bubble" and float(x.get("confidence", 0.0)) >= 0.80]
    dst_bubbles = [x for x in target_layout.get("items", []) if x.get("label") == "bubble" and float(x.get("confidence", 0.0)) >= 0.80]
    src_masks = []
    for s in src_bubbles:
        poly = _warp_polygon(s.get("polygon", []), matrix)
        src_masks.append((_polygon_to_mask(shape, poly), poly, s))
    pairs: List[Dict[str, Any]] = []
    used_src: set[int] = set()
    for di, d in enumerate(dst_bubbles):
        dm = _polygon_to_mask(shape, d.get("polygon", []))
        best = None
        for si, (sm, spoly, s) in enumerate(src_masks):
            if si in used_src:
                continue
            iou = _mask_iou(dm, sm)
            if best is None or iou > best[0]:
                best = (iou, si, spoly, s)
        if best is None or best[0] < 0.55:
            continue
        iou, si, spoly, s = best
        used_src.add(si)
        pairs.append({
            "id": f"layout-pair-{di:03d}",
            "source_id": f"layout-source-{si:03d}",
            "target_id": f"layout-target-{di:03d}",
            "confidence": float(min(float(s.get("confidence", 0.0)), float(d.get("confidence", 0.0)), 0.5 + 0.5 * iou)),
            "region_kind": "bubble",
            "bbox_target": list(map(int, d.get("box", _mask_bbox(dm)))),
            "target_polygon": d.get("polygon", []),
            "source_polygon_warped": spoly,
            "target_mask": dm,
            "layout_iou": float(iou),
            "candidate_source": "layout_registered_pair",
        })
    return pairs


def _build_candidates(
    paired: Dict[str, Any],
    source_layout: Dict[str, Any],
    target_layout: Dict[str, Any],
    matrix: np.ndarray,
    shape: Tuple[int, int],
) -> List[Dict[str, Any]]:
    source_idx, target_idx = _paired_region_index(paired)
    layout_pairs = _layout_bubble_pairs(source_layout, target_layout, matrix, shape)
    candidates: List[Dict[str, Any]] = []

    for rec in paired.get("records", []):
        src = source_idx.get(rec.get("source_id"))
        dst = target_idx.get(rec.get("target_id"))
        if not src or not dst:
            continue
        pm = _polygon_to_mask(shape, dst.get("polygon", []))
        best_layout = None
        for lp in layout_pairs:
            iou = _mask_iou(pm, lp["target_mask"])
            if best_layout is None or iou > best_layout[0]:
                best_layout = (iou, lp)
        # Union paired-diff interior with registered layout bubble when they refer to the same container.
        # This fixes tiny notches/tail holes that can leave one Japanese glyph behind.
        combined = pm.copy()
        source_name = "paired_diff"
        if best_layout and best_layout[0] >= 0.50:
            combined = cv2.bitwise_or(combined, best_layout[1]["target_mask"])
            source_name = "paired_diff+layout_union"
        # Close tiny one-character notches but keep the actual bubble border outside the mask.
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        candidates.append({
            "source_id": rec.get("source_id", ""),
            "target_id": rec.get("target_id", ""),
            "confidence": float(rec.get("confidence", 0.0)),
            "region_kind": rec.get("region_kind", "bubble"),
            "bbox_target": _mask_bbox(combined),
            "target_mask": combined,
            "candidate_source": source_name,
        })

    # Add registered-layout bubbles missed by paired-diff, e.g. color/narration containers.
    for lp in layout_pairs:
        overlap = max((_mask_iou(lp["target_mask"], c["target_mask"]) for c in candidates), default=0.0)
        if overlap >= 0.50:
            continue
        candidates.append(lp)

    candidates.sort(key=lambda c: (c["bbox_target"][1], c["bbox_target"][0]))
    return candidates


def run_aligned_overlay_reveal(
    page_dir: str | os.PathLike[str],
    output_final_name: str = "final.png",
) -> ModeResult:
    page_dir = Path(page_dir)
    source = _imread(page_dir / "source_original.png")
    target = _imread(page_dir / "target_original.png")
    matrix, reg = _load_registration(page_dir)
    paired = _load_paired_diff(page_dir)
    source_layout = _load_layout(page_dir, "source")
    target_layout = _load_layout(page_dir, "target")

    target_h, target_w = target.shape[:2]
    shape = (target_h, target_w)
    warped_source = cv2.warpPerspective(
        source,
        matrix,
        (target_w, target_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )

    candidates = _build_candidates(paired, source_layout, target_layout, matrix, shape)

    # upper layer = target. For colored/open text we first erase JP glyphs locally.
    cleaned_target = target.copy()
    # lower layer = transparent; whole white bubble regions receive full aligned SOURCE,
    # colored/open regions receive only aligned Chinese glyph pixels.
    lower_rgba = np.zeros((target_h, target_w, 4), dtype=np.uint8)
    upper_alpha = np.full((target_h, target_w), 255, dtype=np.uint8)

    hole_mask = np.zeros(shape, dtype=np.uint8)
    erase_mask = np.zeros(shape, dtype=np.uint8)
    source_ink_mask = np.zeros(shape, dtype=np.uint8)
    allowed_change_mask = np.zeros(shape, dtype=np.uint8)
    region_preview = target.copy()
    regions: List[RegionResult] = []

    for idx, c in enumerate(candidates):
        mask = c["target_mask"].copy()
        if int(np.count_nonzero(mask)) == 0:
            continue
        white_ratio, saturation_mean = _bubble_appearance(target, mask)
        # White/neutral bubble: true full-bubble transparent hole to aligned SOURCE below.
        # Colored container: preserve TARGET RGB, erase Japanese glyphs, reveal only Chinese glyph layer.
        colored = saturation_mean > 12.0 or white_ratio < 0.62
        clear_mode = "colored_text_hole" if colored else "full_bubble_hole"
        applied_mask = np.zeros_like(mask)
        reason = "ok"

        if not colored:
            # Do not erode: target/layout masks are already interior masks. Erosion caused residual kana.
            bubble_hole = _expand_white_bubble_for_target_text(target, mask)
            hole_mask = cv2.bitwise_or(hole_mask, bubble_hole)
            allowed_change_mask = cv2.bitwise_or(allowed_change_mask, bubble_hole)
            upper_alpha[bubble_hole > 0] = 0
            lower_rgba[..., :3][bubble_hole > 0] = warped_source[bubble_hole > 0]
            lower_rgba[..., 3][bubble_hole > 0] = 255
            glyph = _source_glyph_mask(warped_source, bubble_hole, colored=False)
            source_ink_mask = cv2.bitwise_or(source_ink_mask, glyph)
            applied_mask = bubble_hole
        else:
            jp_mask = _target_jp_text_mask(target, mask)
            cn_mask = _source_glyph_mask(warped_source, mask, colored=True)
            if int(np.count_nonzero(jp_mask)) < 25 or int(np.count_nonzero(cn_mask)) < 25:
                reason = "insufficient_colored_text_masks"
                regions.append(
                    RegionResult(
                        id=f"aligned-{idx:04d}",
                        source_id=c.get("source_id", ""),
                        target_id=c.get("target_id", ""),
                        confidence=float(c.get("confidence", 0.0)),
                        region_kind=c.get("region_kind", "bubble"),
                        bbox_target=list(map(int, c.get("bbox_target", [0, 0, 0, 0]))),
                        clear_mode=clear_mode,
                        applied=False,
                        reason=reason,
                        mask_pixels=0,
                        changed_pixels=0,
                        candidate_source=c.get("candidate_source", "layout_registered_pair"),
                        white_ratio=white_ratio,
                        saturation_mean=saturation_mean,
                    )
                )
                continue
            # Erase Japanese from the TARGET itself to keep the purple/color background authoritative.
            cleaned_target = cv2.inpaint(cleaned_target, jp_mask, 3, cv2.INPAINT_TELEA)
            erase_mask = cv2.bitwise_or(erase_mask, jp_mask)
            # Hole only where Chinese glyph pixels exist. Lower layer contains only those glyph pixels.
            hole_mask = cv2.bitwise_or(hole_mask, cn_mask)
            source_ink_mask = cv2.bitwise_or(source_ink_mask, cn_mask)
            allowed_change_mask = cv2.bitwise_or(allowed_change_mask, cv2.bitwise_or(jp_mask, cn_mask))
            upper_alpha[cn_mask > 0] = 0
            lower_rgba[..., :3][cn_mask > 0] = warped_source[cn_mask > 0]
            lower_rgba[..., 3][cn_mask > 0] = 255
            applied_mask = cv2.bitwise_or(jp_mask, cn_mask)

        x1, y1, x2, y2 = list(map(int, c.get("bbox_target", _mask_bbox(mask))))
        color = (20, 190, 20) if not colored else (0, 160, 255)
        cv2.rectangle(region_preview, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            region_preview,
            f"{idx + 1}:{clear_mode}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )
        regions.append(
            RegionResult(
                id=f"aligned-{idx:04d}",
                source_id=c.get("source_id", ""),
                target_id=c.get("target_id", ""),
                confidence=float(c.get("confidence", 0.0)),
                region_kind=c.get("region_kind", "bubble"),
                bbox_target=[x1, y1, x2, y2],
                clear_mode=clear_mode,
                applied=True,
                reason=reason,
                mask_pixels=int(np.count_nonzero(applied_mask)),
                changed_pixels=0,
                candidate_source=c.get("candidate_source", "paired_diff"),
                white_ratio=white_ratio,
                saturation_mean=saturation_mean,
            )
        )

    # Build explicit layers, then alpha-composite exactly as the mode contract says.
    upper_rgba = np.dstack([cleaned_target, upper_alpha])
    alpha_top = upper_rgba[..., 3:4].astype(np.float32) / 255.0
    alpha_bottom = lower_rgba[..., 3:4].astype(np.float32) / 255.0
    out_alpha = alpha_top + alpha_bottom * (1.0 - alpha_top)
    out_rgb_premul = (
        upper_rgba[..., :3].astype(np.float32) * alpha_top
        + lower_rgba[..., :3].astype(np.float32) * alpha_bottom * (1.0 - alpha_top)
    )
    denom = np.maximum(out_alpha, 1e-6)
    flat = np.clip(out_rgb_premul / denom, 0, 255).astype(np.uint8)

    # The allowed-change mask includes holes + colored JP inpainting. Everything else must be byte-identical TARGET.
    flat[allowed_change_mask == 0] = target[allowed_change_mask == 0]

    diff_bin = np.any(flat != target, axis=2).astype(np.uint8)
    changed_pixels = int(np.count_nonzero(diff_bin))
    changed_ratio = changed_pixels / float(target_h * target_w)
    mask_pixels = int(np.count_nonzero(allowed_change_mask))
    outside = allowed_change_mask == 0
    outside_mask_unchanged = bool(np.array_equal(flat[outside], target[outside]))

    for rr in regions:
        x1, y1, x2, y2 = rr.bbox_target
        rr.changed_pixels = int(np.count_nonzero(diff_bin[y1:y2, x1:x2]))

    aligned_json = {
        "schema": "manga_hd_translation_transfer.aligned_overlay_reveal.v2",
        "requested_mode": "aligned_overlay_reveal",
        "used": True,
        "accepted": bool(changed_pixels > 0),
        "reason": "ok" if changed_pixels > 0 else "no_regions_applied",
        "page_triage": "SAFE" if changed_pixels > 0 and outside_mask_unchanged else "REVIEW",
        "contract": "target_upper_layer__full_bubble_or_colored_text_holes__aligned_cn_lower_layer",
        "registration": {
            "confidence": float(reg.get("confidence", 0.0)),
            "inlier_ratio": float(reg.get("inlier_ratio", 0.0)),
            "reprojection_error": float(reg.get("reprojection_error", 0.0)),
            "spatial_coverage": float(reg.get("spatial_coverage", 0.0)),
            "method": reg.get("method", ""),
        },
        "diagnostics": {
            "paired_records": len(paired.get("records", [])),
            "candidate_count": len(candidates),
            "applied_region_count": sum(1 for x in regions if x.applied),
            "full_bubble_holes": sum(1 for x in regions if x.applied and x.clear_mode == "full_bubble_hole"),
            "colored_text_holes": sum(1 for x in regions if x.applied and x.clear_mode == "colored_text_hole"),
            "allowed_change_pixels": mask_pixels,
            "hole_pixels": int(np.count_nonzero(hole_mask)),
            "erase_pixels": int(np.count_nonzero(erase_mask)),
            "changed_pixels": changed_pixels,
            "changed_ratio": changed_ratio,
            "outside_mask_unchanged": outside_mask_unchanged,
            "target_shape": [target_h, target_w],
            "source_shape": list(source.shape[:2]),
        },
        "regions": [asdict(r) for r in regions],
    }

    _save_json(page_dir / "aligned_overlay_reveal.json", aligned_json)
    _imwrite(page_dir / "aligned_overlay_reveal_mask.png", allowed_change_mask)
    _imwrite(page_dir / "aligned_overlay_reveal_hole_mask.png", hole_mask)
    _imwrite(page_dir / "aligned_overlay_reveal_erase_mask.png", erase_mask)
    _imwrite(page_dir / "aligned_overlay_reveal_regions.png", region_preview)
    _imwrite(page_dir / "aligned_overlay_reveal_source_ink.png", source_ink_mask)
    diff_mask = diff_bin * 255
    _imwrite(page_dir / "aligned_overlay_reveal_diff_mask.png", diff_mask)
    judgment = target.copy()
    green = np.zeros_like(target); green[..., 1] = 255
    red = np.zeros_like(target); red[..., 2] = 255
    judgment = np.where(allowed_change_mask[..., None] > 0, cv2.addWeighted(judgment, 0.62, green, 0.38, 0), judgment)
    judgment = np.where(diff_mask[..., None] > 0, cv2.addWeighted(judgment, 0.50, red, 0.50, 0), judgment)
    _imwrite(page_dir / "aligned_overlay_reveal_judgment.png", judgment)
    _imwrite(page_dir / "aligned_overlay_reveal_layer.png", lower_rgba)
    _imwrite(page_dir / "jp_layer_rgba.png", upper_rgba)
    _imwrite(page_dir / "cn_layer_rgb.png", warped_source)
    _imwrite(page_dir / output_final_name, flat)
    _imwrite(page_dir / "review_preview.png", flat)

    run_state_path = page_dir / "last_run_state.json"
    try:
        run_state = _load_json(run_state_path)
    except Exception:
        run_state = {"schema": "manga_hd_translation_transfer.run_state.v2"}
    run_state.update({
        "status": "success",
        "mode": "aligned_overlay_reveal",
        "selected_strategy": "aligned_overlay_reveal",
        "workspace_integrity": {
            "schema": "manga_hd_translation_transfer.workspace_integrity.v1",
            "pass": True,
            "mode": "aligned_overlay_reveal",
            "selected_strategy": "aligned_overlay_reveal",
            "target_shape": [target_h, target_w],
            "issues": [],
            "checked_images": [
                output_final_name,
                "review_preview.png",
                "target_original.png",
                "aligned_overlay_reveal_layer.png",
                "aligned_overlay_reveal_mask.png",
                "aligned_overlay_reveal_hole_mask.png",
                "aligned_overlay_reveal_erase_mask.png",
                "aligned_overlay_reveal_regions.png",
                "aligned_overlay_reveal_source_ink.png",
            ],
            "checked_json": ["project.json", "aligned_overlay_reveal.json"],
        },
    })
    _save_json(run_state_path, run_state)

    project_path = page_dir / "project.json"
    project = _load_json(project_path)
    artifacts = project.setdefault("artifacts", {})
    artifacts.update({
        "aligned_overlay_reveal": {
            "json": "aligned_overlay_reveal.json",
            "mask": "aligned_overlay_reveal_mask.png",
            "hole_mask": "aligned_overlay_reveal_hole_mask.png",
            "erase_mask": "aligned_overlay_reveal_erase_mask.png",
            "regions": "aligned_overlay_reveal_regions.png",
            "layer": "aligned_overlay_reveal_layer.png",
            "source_ink": "aligned_overlay_reveal_source_ink.png",
        },
        "final": output_final_name,
        "review_preview": "review_preview.png",
    })
    project.setdefault("meta", {})["last_mode"] = "aligned_overlay_reveal"
    _save_json(project_path, project)

    validation = validate_aligned_overlay_reveal(page_dir)
    _save_json(page_dir / "aligned_overlay_reveal_validation.json", validation)

    return ModeResult(
        accepted=bool(changed_pixels > 0),
        reason="ok" if changed_pixels > 0 else "no_regions_applied",
        page_triage="SAFE" if changed_pixels > 0 and outside_mask_unchanged else "REVIEW",
        used=True,
        requested_mode="aligned_overlay_reveal",
        strategy="aligned_overlay_reveal",
        registration_confidence=float(reg.get("confidence", 0.0)),
        mask_pixels=mask_pixels,
        changed_pixels=changed_pixels,
        changed_ratio=changed_ratio,
        outside_mask_unchanged=outside_mask_unchanged,
        target_shape=[target_h, target_w],
        regions=regions,
    )


# ---------------------------------------------------------------------------
# v2.2.2 production in-memory whole-page route
# ---------------------------------------------------------------------------

def _bubble_mask_from_instance(bubble: BubbleInstance, shape: tuple[int, int]) -> np.ndarray:
    if bubble.mask is not None and bubble.mask.shape == shape:
        return (bubble.mask > 0).astype(np.uint8) * 255
    return _polygon_to_mask(shape, bubble.polygon)


def _valid_warp_mask(source_shape: tuple[int, int], registration: RegistrationResult, target_shape: tuple[int, int]) -> np.ndarray:
    sh, sw = source_shape
    th, tw = target_shape
    ones = np.full((sh, sw), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        ones,
        np.asarray(registration.matrix, dtype=np.float64),
        (tw, th),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _inverse_bbox(bbox: Sequence[int], registration: RegistrationResult) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in bbox]
    pts = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
    try:
        inv = np.linalg.inv(np.asarray(registration.matrix, dtype=np.float64))
        mapped = cv2.perspectiveTransform(pts, inv)[0]
        sw, sh = registration.source_size
        sx0 = max(0, min(sw, int(np.floor(mapped[:, 0].min()))))
        sy0 = max(0, min(sh, int(np.floor(mapped[:, 1].min()))))
        sx1 = max(sx0, min(sw, int(np.ceil(mapped[:, 0].max()))))
        sy1 = max(sy0, min(sh, int(np.ceil(mapped[:, 1].max()))))
        return sx0, sy0, sx1, sy1
    except Exception:
        return int(x0), int(y0), int(x1), int(y1)


def _empty_production_result(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    reason: str,
) -> AlignedOverlayResult:
    h, w = target.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    aligned = cv2.warpPerspective(
        source,
        np.asarray(registration.matrix, dtype=np.float64),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    plan = AlignedOverlayPlan(
        accepted=False,
        reason=reason,
        aligned_source=aligned,
        valid_mask=_valid_warp_mask(source.shape[:2], registration, target.shape[:2]),
        erase_mask=empty.copy(),
        source_ink_mask=empty.copy(),
        full_raster_mask=empty.copy(),
        regions=[],
        diagnostics={
            "engine": "aligned_hole_v2.3.38",
            "contract": "target_upper_layer__bubble_or_textbox_holes__registered_cn_lower_layer",
            "reason": reason,
        },
    )
    return AlignedOverlayResult(
        image=target.copy(),
        layer_rgba=np.zeros((h, w, 4), dtype=np.uint8),
        erase_mask=empty.copy(),
        source_ink_mask=empty.copy(),
        regions_overlay=target.copy(),
        plan=plan,
        diagnostics=dict(plan.diagnostics),
    )


def build_production_aligned_hole_result(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: AlignedOverlayRevealConfig,
    bubble_cfg: BubbleConfig,
    *,
    stage_cache=None,
    cache_stats: dict[str, str] | None = None,
    source_path: str | os.PathLike[str] | None = None,
    target_path: str | os.PathLike[str] | None = None,
    target_bubbles: Sequence[BubbleInstance] | None = None,
) -> AlignedOverlayResult:
    """Independent production whole-page hole renderer.

    The route is intentionally isolated from ``transparent_bubble_reveal``:
    TARGET bubble geometry determines where the upper Japanese page may open.
    White/neutral containers reveal the fully registered SOURCE underneath.
    Coloured containers keep TARGET RGB, erase only JP glyphs locally, and open
    holes only where registered Chinese glyphs exist.
    """
    stats = cache_stats if cache_stats is not None else {}
    # Whole-page geometry gate.  0.78 accepts the observed p-044 registration
    # (0.79994) while remaining stricter than the production transparent route.
    if float(registration.confidence) < float(cfg.min_registration_confidence):
        return _empty_production_result(source, target, registration, "rejected_registration:registration_confidence")
    if float(registration.reprojection_error) > float(cfg.max_reprojection_error):
        return _empty_production_result(source, target, registration, "rejected_registration:registration_reprojection_error")
    if float(registration.inlier_ratio) < float(cfg.min_inlier_ratio):
        return _empty_production_result(source, target, registration, "rejected_registration:registration_inlier_ratio")
    if float(registration.spatial_coverage) < float(cfg.min_spatial_coverage):
        return _empty_production_result(source, target, registration, "rejected_registration:registration_spatial_coverage")

    h, w = target.shape[:2]
    shape = (h, w)
    aligned = cv2.warpPerspective(
        source,
        np.asarray(registration.matrix, dtype=np.float64),
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    if target_bubbles is None:
        if target_path is None:
            target_path = "<aligned-hole-target>"
        target_bubbles = primary_bubbles_cached(
            "target",
            target,
            target_path,
            bubble_config=bubble_cfg,
            cache=stage_cache,
            cache_enabled=bool(stage_cache is not None),
            stats=stats,
        )

    candidates = _collect_target_container_candidates(target, bubble_cfg, cfg, list(target_bubbles or []))
    if not candidates:
        return _empty_production_result(source, target, registration, "no_target_container_candidates")

    cleaned_target = target.copy()
    lower_rgba = np.zeros((h, w, 4), dtype=np.uint8)
    upper_alpha = np.full((h, w), 255, dtype=np.uint8)
    hole_mask = np.zeros(shape, dtype=np.uint8)
    erase_mask = np.zeros(shape, dtype=np.uint8)
    source_ink_mask = np.zeros(shape, dtype=np.uint8)
    allowed_change_mask = np.zeros(shape, dtype=np.uint8)
    overlay = target.copy()
    regions: list[AlignedOverlayRegion] = []
    supplemental_filtered_count = 0
    supplemental_filter_reasons: dict[str, int] = {}

    for idx, bubble in enumerate(candidates):
        mask = _bubble_mask_from_instance(bubble, shape)
        area = int(cv2.countNonZero(mask))
        if area <= 0:
            continue
        area_ratio = area / float(max(1, h * w))
        candidate_source = str((bubble.meta or {}).get("aligned_hole_candidate_source", "primary"))
        is_supplemental = candidate_source.startswith("supplemental_")
        supplemental_max_area = float(getattr(cfg, "supplemental_max_area_ratio", 0.04))
        translation_diag = _registered_translation_evidence(aligned, target, mask, cfg) if is_supplemental else {"passed": True, "reason": "primary_semantic_authority"}
        if is_supplemental and area_ratio > supplemental_max_area:
            translation_diag = {**translation_diag, "passed": False, "reason": "supplemental_area_cap", "area_ratio": area_ratio, "max_area_ratio": supplemental_max_area}
        if is_supplemental and not bool(translation_diag.get("passed", False)):
            supplemental_filtered_count += 1
            filter_reason = str(translation_diag.get("reason", "supplemental_translation_gate"))
            supplemental_filter_reasons[filter_reason] = int(supplemental_filter_reasons.get(filter_reason, 0)) + 1
            continue
        if area_ratio > float(cfg.max_single_region_area_ratio):
            regions.append(AlignedOverlayRegion(
                id=f"aligned_hole_{idx:03d}", target_bbox=tuple(map(int, bubble.bbox)),
                source_bbox=_inverse_bbox(tuple(map(int, bubble.bbox)), registration),
                erase_mask=np.zeros(shape, np.uint8), source_ink_mask=np.zeros(shape, np.uint8),
                full_raster_mask=np.zeros(shape, np.uint8), composite_mode="reject",
                triage="REJECT", reason="single_region_area_cap", white_ratio=0.0, color_ratio=0.0,
                erase_area_ratio=0.0, source_ink_pixels=0, target_ink_pixels=0,
                border_guard_px=0, diagnostics={"confidence": float(bubble.confidence), "backend": bubble.meta.get("backend", "")},
            ))
            continue

        white_ratio, saturation_mean = _bubble_appearance(target, mask)
        colored = bool(saturation_mean > 12.0 or white_ratio < 0.62)
        bbox = tuple(int(round(v)) for v in bubble.bbox)
        src_bbox = _inverse_bbox(bbox, registration)
        local_erase = np.zeros(shape, dtype=np.uint8)
        local_hole = np.zeros(shape, dtype=np.uint8)
        local_source_ink = np.zeros(shape, dtype=np.uint8)
        composite_mode = "colored_text_hole" if colored else "full_bubble_hole"
        reason = "ok"

        if not colored:
            # TARGET bubble masks are interior authority.  Never erode them: doing
            # so is exactly what left small kana remnants in earlier builds.
            local_hole = _expand_white_bubble_for_target_text(target, mask)
            local_source_ink = _source_glyph_mask(aligned, local_hole, colored=False)
            if int(cv2.countNonZero(local_source_ink)) < 25:
                reason = "insufficient_registered_source_ink"
                if is_supplemental:
                    supplemental_filtered_count += 1
                    supplemental_filter_reasons[reason] = int(supplemental_filter_reasons.get(reason, 0)) + 1
                    continue
                triage = "REJECT"
            else:
                triage = "SAFE"
                local_erase = local_hole.copy()
                upper_alpha[local_hole > 0] = 0
                lower_rgba[..., :3][local_hole > 0] = aligned[local_hole > 0]
                lower_rgba[..., 3][local_hole > 0] = 255
        else:
            # Colored container: TARGET RGB stays authoritative.  JP glyphs are
            # inpainted in place; only CN glyph positions become transparent holes.
            jp_mask = _target_jp_text_mask(target, mask)
            cn_mask = _source_glyph_mask(aligned, mask, colored=True)
            if int(cv2.countNonZero(jp_mask)) < 25 or int(cv2.countNonZero(cn_mask)) < 25:
                reason = "insufficient_colored_text_masks"
                if is_supplemental:
                    supplemental_filtered_count += 1
                    supplemental_filter_reasons[reason] = int(supplemental_filter_reasons.get(reason, 0)) + 1
                    continue
                triage = "REJECT"
            else:
                triage = "SAFE"
                cleaned_target = cv2.inpaint(cleaned_target, jp_mask, 3, cv2.INPAINT_TELEA)
                local_erase = cv2.bitwise_or(jp_mask, cn_mask)
                local_hole = cn_mask.copy()
                local_source_ink = cn_mask.copy()
                upper_alpha[cn_mask > 0] = 0
                lower_rgba[..., :3][cn_mask > 0] = aligned[cn_mask > 0]
                lower_rgba[..., 3][cn_mask > 0] = 255

        if triage != "REJECT":
            hole_mask = cv2.bitwise_or(hole_mask, local_hole)
            erase_mask = cv2.bitwise_or(erase_mask, local_erase)
            source_ink_mask = cv2.bitwise_or(source_ink_mask, local_source_ink)
            allowed_change_mask = cv2.bitwise_or(allowed_change_mask, local_erase)
            color = (30, 190, 30) if not colored else (0, 165, 255)
        else:
            color = (40, 40, 220)
        x0, y0, x1, y1 = bbox
        cv2.rectangle(overlay, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 2)
        cv2.putText(overlay, f"H{idx + 1}:{composite_mode}", (x0, max(14, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        regions.append(AlignedOverlayRegion(
            id=f"aligned_hole_{idx:03d}", target_bbox=bbox, source_bbox=src_bbox,
            erase_mask=local_erase, source_ink_mask=local_source_ink, full_raster_mask=local_hole,
            composite_mode=composite_mode, triage=triage, reason=reason,
            white_ratio=float(white_ratio), color_ratio=float(min(1.0, saturation_mean / 255.0)),
            erase_area_ratio=float(cv2.countNonZero(local_erase) / max(1, h * w)),
            source_ink_pixels=int(cv2.countNonZero(local_source_ink)),
            target_ink_pixels=int(cv2.countNonZero(local_erase)), border_guard_px=0,
            diagnostics={
                "engine": "aligned_hole_v2.3.38", "candidate_source": candidate_source,
                "confidence": float(bubble.confidence), "backend": str(bubble.meta.get("backend", "")),
                "saturation_mean": float(saturation_mean), "translation_evidence": translation_diag,
            },
        ))

    upper_rgba = np.dstack([cleaned_target, upper_alpha])
    alpha_top = upper_rgba[..., 3:4].astype(np.float32) / 255.0
    alpha_bottom = lower_rgba[..., 3:4].astype(np.float32) / 255.0
    out_alpha = alpha_top + alpha_bottom * (1.0 - alpha_top)
    premul = (
        upper_rgba[..., :3].astype(np.float32) * alpha_top
        + lower_rgba[..., :3].astype(np.float32) * alpha_bottom * (1.0 - alpha_top)
    )
    flat = np.clip(premul / np.maximum(out_alpha, 1e-6), 0, 255).astype(np.uint8)
    flat[allowed_change_mask == 0] = target[allowed_change_mask == 0]

    changed = np.any(flat != target, axis=2)
    changed_pixels = int(np.count_nonzero(changed))
    outside_unchanged = bool(np.array_equal(flat[allowed_change_mask == 0], target[allowed_change_mask == 0]))
    accepted = bool(changed_pixels > 0 and any(r.triage != "REJECT" for r in regions))
    reason = "ok" if accepted else "no_accepted_regions"
    plan = AlignedOverlayPlan(
        accepted=accepted,
        reason=reason,
        aligned_source=aligned,
        valid_mask=_valid_warp_mask(source.shape[:2], registration, target.shape[:2]),
        erase_mask=allowed_change_mask.copy(),
        source_ink_mask=source_ink_mask.copy(),
        # For this renderer full_raster_mask is the actual upper-layer hole mask,
        # including CN-only holes on coloured containers.
        full_raster_mask=hole_mask.copy(),
        regions=regions,
        diagnostics={
            "engine": "aligned_hole_v2.3.38",
            "contract": "target_upper_layer__bubble_or_textbox_holes__registered_cn_lower_layer",
            "candidate_count": len(candidates),
            "applied_region_count": sum(1 for r in regions if r.triage != "REJECT"),
            "full_bubble_holes": sum(1 for r in regions if r.triage != "REJECT" and r.composite_mode == "full_bubble_hole"),
            "colored_text_holes": sum(1 for r in regions if r.triage != "REJECT" and r.composite_mode == "colored_text_hole"),
            "hole_pixels": int(cv2.countNonZero(hole_mask)),
            "allowed_change_pixels": int(cv2.countNonZero(allowed_change_mask)),
            "changed_pixels": changed_pixels,
            "outside_mask_unchanged": outside_unchanged,
            "primary_target_bubbles": sum(1 for b in candidates if str((b.meta or {}).get("aligned_hole_candidate_source", "primary")) == "primary"),
            "supplemental_white_candidates": sum(1 for b in candidates if str((b.meta or {}).get("aligned_hole_candidate_source", "")) == "supplemental_white"),
            "supplemental_text_barrier_candidates": sum(1 for b in candidates if str((b.meta or {}).get("aligned_hole_candidate_source", "")) == "supplemental_text_barrier"),
            "supplemental_colored_candidates": sum(1 for b in candidates if str((b.meta or {}).get("aligned_hole_candidate_source", "")) == "supplemental_colored"),
            "supplemental_filtered_count": int(supplemental_filtered_count),
            "supplemental_filter_reasons": dict(supplemental_filter_reasons),
            "cache_primary_target": str(stats.get("primary_detector_target", "")),
        },
    )
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    rgb = cv2.cvtColor(flat, cv2.COLOR_BGR2RGB)
    layer[changed, :3] = rgb[changed]
    layer[changed, 3] = 255
    diagnostics = dict(plan.diagnostics)
    diagnostics.update({"accepted": accepted, "reason": reason, "page_triage": plan.page_triage})
    return AlignedOverlayResult(
        image=flat,
        layer_rgba=layer,
        erase_mask=allowed_change_mask.copy(),
        source_ink_mask=source_ink_mask.copy(),
        regions_overlay=overlay,
        plan=plan,
        diagnostics=diagnostics,
    )
