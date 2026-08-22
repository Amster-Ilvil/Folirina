from __future__ import annotations

"""Region-level compositing adapters for the review workbench.

These actions are explicitly manual. They do not change or dispatch the page's
automatic transfer mode. Each action is clipped by a TARGET-space selection mask
and can therefore be stacked with actions from other algorithms in review order.
"""

from copy import deepcopy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import TextUnit
from .region_selection import selection_mask_from_row, bbox_from_mask
from . import manual_effect as legacy_ops
from .modes.mask_replace import open_text_manual as mask_ops
from .modes.mask_replace import transfer_ops as mask_transfer_ops
from .modes.hybrid import open_text_manual as hybrid_mask_ops
from .modes.hybrid import transfer_ops as hybrid_transfer_ops
from .modes.hybrid.lettering_ops import fit_text, composite_text
from .review_artifacts import safe_page_artifact_path
from .text_only_transfer import target_text_mask_in_container

REGION_MODES = {
    "region_direct_patch",
    "region_precise_mask",
    "region_hole_reveal",
    "region_transparent",
    "region_ocr",
    "region_brush_reveal",
}


def _project_mode(project: dict[str, Any]) -> str:
    return str(((project.get("meta") or {}).get("transfer_mode") or "")).strip().lower()


def _precise_region_engine(project: dict[str, Any], cfg: Any):
    """Keep regional Precise semantics aligned with the page's private mode.

    Hybrid owns a private copy of the Precise renderer/config namespace.  The
    old region adapter always called Mask's renderer/config, so a Hybrid page
    could behave differently inside the Region workbench even after the two
    automatic modes had been synchronized.  Direct/Reveal/Reletter pages still
    use Mask as the standalone manual Precise engine.
    """
    if _project_mode(project) == "hybrid":
        return hybrid_transfer_ops._transfer_open_complex_text_region, getattr(getattr(cfg, "hybrid", None), "mask", cfg), "hybrid"
    return mask_transfer_ops._transfer_open_complex_text_region_full, getattr(cfg, "mask_replace", cfg), "mask_replace"


def _transparent_region_ops(project: dict[str, Any]):
    return hybrid_mask_ops if _project_mode(project) == "hybrid" else mask_ops


def is_region_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in REGION_MODES


def _layer_from_rgb(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    out = np.zeros((rgb.shape[0], rgb.shape[1], 4), np.uint8)
    out[:, :, :3] = rgb
    out[:, :, 3] = np.asarray(alpha, np.uint8)
    return out


def _alpha_composite(base: np.ndarray, top_rgb: np.ndarray, alpha_u8: np.ndarray) -> np.ndarray:
    a = np.asarray(alpha_u8, np.float32)[:, :, None] / 255.0
    return np.clip(base.astype(np.float32) * (1.0 - a) + top_rgb.astype(np.float32) * a, 0, 255).astype(np.uint8)


def _feather(mask: np.ndarray, amount: int) -> np.ndarray:
    px = max(0, min(8, int(amount)))
    if px <= 0:
        return np.asarray(mask, np.uint8)
    alpha = np.asarray(mask, np.float32) / 255.0
    sigma = max(0.45, px * 0.65)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
    alpha[mask > 0] = np.maximum(alpha[mask > 0], 0.96)
    # Feather inward only: the manual selection is a hard authority boundary.
    alpha[mask == 0] = 0.0
    return np.clip(alpha * 255.0, 0, 255).astype(np.uint8)


def _polygon_mask(shape: tuple[int, int], polygons: list[Any]) -> np.ndarray:
    h, w = shape; out = np.zeros((h, w), np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1)); pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], 255)
    if cv2.countNonZero(out):
        out = cv2.dilate(out, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return out


def _roi_bounds(mask: np.ndarray, *, halo: int = 0) -> tuple[int, int, int, int]:
    box = bbox_from_mask(mask)
    if len(box) != 4:
        raise ValueError("区域工具没有有效选区")
    h, w = mask.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in box]
    pad = max(0, int(halo))
    return max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)


def _aligned_source_roi(source: np.ndarray, target_shape: tuple[int, int], project: dict[str, Any],
                        roi: tuple[int, int, int, int], *, source_offset_x: int = 0,
                        source_offset_y: int = 0) -> tuple[np.ndarray, bool]:
    """Warp SOURCE directly into a TARGET-space ROI instead of a full page."""
    th, tw = map(int, target_shape)
    xa, ya, xb, yb = [int(v) for v in roi]
    rw, rh = max(1, xb - xa), max(1, yb - ya)
    H = mask_ops.registration_homography(project).copy()
    dx, dy = int(source_offset_x), int(source_offset_y)
    identity = bool(
        source.shape[:2] == (th, tw) and not dx and not dy
        and np.max(np.abs(np.asarray(H, np.float64) - np.eye(3))) <= 1e-7
    )
    if identity:
        return source[ya:yb, xa:xb].copy(), True
    if dx or dy:
        T = np.asarray([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]], np.float64)
        H = T @ H
    C = np.asarray([[1.0, 0.0, -float(xa)], [0.0, 1.0, -float(ya)], [0.0, 0.0, 1.0]], np.float64)
    crop = cv2.warpPerspective(
        source, C @ H, (rw, rh), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return crop, False


def _manual_precise_relaxed_cfg(cfg_obj: Any) -> Any:
    """Clone one Precise config for a human-confirmed Region retry.

    Automatic page processing deliberately requires a minimum amount of unique
    SOURCE/TARGET ink before it trusts a candidate.  A Region selection already
    carries explicit human authority, so tiny punctuation, thin strokes and
    low-contrast glyphs must not be rejected only because they are below the
    automatic candidate threshold.  This clone is local to the manual action;
    mode-owned configuration and automatic renderers are never mutated.
    """
    try:
        local = cfg_obj.model_copy(deep=True)
    except Exception:
        local = deepcopy(cfg_obj)
    for name in ("paired_diff_complex_min_source_ink_pixels", "paired_diff_complex_min_target_ink_pixels"):
        try:
            setattr(local, name, 1)
        except Exception:
            # Config-like objects used by plugins/tests may expose attributes via
            # an immutable proxy.  The subsequent transparent fallback still
            # provides a detector-independent manual path in that case.
            pass
    return local


def _precise_manual_transparent_fallback(
    current: np.ndarray,
    target: np.ndarray,
    source: np.ndarray,
    project: dict[str, Any],
    row: dict[str, Any],
    cfg: Any,
    *,
    page_dir: str | Path | None,
    original_reason: str,
    original_diag: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Fallback for a *human-selected* Precise region with too little auto ink.

    Reuse the mature Region-transparent text extractor rather than copying the
    whole SOURCE rectangle.  TARGET background therefore remains authoritative,
    SOURCE glyph antialiasing is preserved on neutral paper, and the explicit
    selection remains the hard write boundary.  Returning ``None`` means even the
    manual text extractor found no usable SOURCE/TARGET authority.
    """
    fallback_row = deepcopy(row)
    fallback_row["mode"] = "region_transparent"
    fallback_row["feather_px"] = int(row.get("feather_px", 0) or 0)
    try:
        fb_out, fb_layer, fb_clear, fb_rec = apply_region_action(
            current, target, source, project, fallback_row, cfg, page_dir=page_dir
        )
    except Exception:
        return None
    if not bool(fb_rec.get("success")) or int(fb_rec.get("authority_pixels", 0) or 0) <= 0:
        return None
    fb_rec = dict(fb_rec)
    fb_diag = dict(fb_rec.get("diagnostics") or {})
    fb_diag.update({
        "manual_precise_authority_fallback": True,
        "manual_precise_fallback_mode": "region_transparent",
        "manual_precise_original_reason": str(original_reason or ""),
        "manual_precise_original_diagnostics": {
            k: v for k, v in dict(original_diag or {}).items() if not isinstance(v, np.ndarray)
        },
        "automatic_thresholds_unchanged": True,
    })
    fb_rec["mode"] = "region_precise_mask"
    fb_rec["precise_manual_fallback"] = True
    fb_rec["fallback_mode"] = "region_transparent"
    fb_rec["diagnostics"] = fb_diag
    return fb_out, fb_layer, fb_clear, fb_rec


def _precise_mask_roi_halo(selection: np.ndarray, cfg: Any) -> int:
    box = bbox_from_mask(selection)
    if len(box) != 4:
        return 32
    x0, y0, x1, y1 = box
    mc = getattr(cfg, "mask_replace", cfg)
    base = max(1, int(max(x1 - x0, y1 - y0)))
    inner = max(6, int(round(base * float(getattr(mc, "paired_diff_complex_region_pad_ratio", 0.16)))))
    tol = max(1, int(getattr(mc, "paired_diff_ink_tolerance_px", 2)))
    gap = max(1, int(getattr(mc, "paired_diff_complex_group_gap_px", 5)))
    clear = max(1, int(getattr(mc, "paired_diff_complex_clear_dilate_px", 2)))
    return max(224, inner + tol * 3 + gap * 2 + clear * 2 + 14)


def _expand_roi_mask(value: Any, shape: tuple[int, int], roi: tuple[int, int, int, int]) -> np.ndarray:
    h, w = shape; xa, ya, xb, yb = roi
    out = np.zeros((h, w), np.uint8)
    if isinstance(value, np.ndarray) and value.shape[:2] == (yb - ya, xb - xa):
        out[ya:yb, xa:xb] = np.asarray(value, np.uint8)
    return out


def _fallback_text_mask(target: np.ndarray, safe: np.ndarray) -> np.ndarray:
    ys, xs = np.where(safe > 0)
    out = np.zeros(safe.shape, np.uint8)
    if xs.size == 0:
        return out
    x0, x1 = int(xs.min()), int(xs.max()) + 1; y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = target[y0:y1, x0:x1]; gate = safe[y0:y1, x0:x1] > 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    values = gray[gate]
    if values.size == 0:
        return out
    threshold = min(180, int(np.percentile(values, 35)) + 24)
    cand = ((gray < threshold) & gate).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    keep = np.zeros_like(cand); area_total = max(1, int(np.count_nonzero(gate)))
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2 or area > area_total * 0.10:
            continue
        if bw > crop.shape[1] * 0.78 or bh > crop.shape[0] * 0.78:
            continue
        fill = float(area / max(1, bw * bh))
        # OCR fallback must prefer glyph-like compact strokes over filled artwork.
        # A dense triangular/face shadow can sit inside a manual selection and
        # otherwise looks like dark text to a simple threshold.  Reject only large
        # dense blobs; normal punctuation and kanji components stay below this
        # combined area/fill gate.
        if area >= max(420, int(area_total * 0.018)) and min(bw, bh) >= 18 and fill >= 0.46:
            continue
        keep[labels == lab] = 1
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    out[y0:y1, x0:x1] = keep * 255; out[safe == 0] = 0
    return out


def _refine_ocr_clear_mask(target: np.ndarray, selection: np.ndarray, polygon_mask: np.ndarray) -> np.ndarray:
    """Keep manual OCR cleanup tightly attached to actual text ink.

    OCR polygons sometimes include skewed quadrilateral corners or padding that
    leaves a dark triangular artifact after cleanup.  Combine the OCR geometry
    with a compact TARGET-ink heuristic: the heuristic becomes the authority and
    the polygon mask only contributes a small antialias halo around proven text.
    """
    poly = cv2.bitwise_and(np.asarray(polygon_mask, np.uint8), np.asarray(selection, np.uint8))
    heuristic = _fallback_text_mask(target, selection)
    if cv2.countNonZero(heuristic) <= 0:
        return poly
    band = cv2.dilate(heuristic, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    if cv2.countNonZero(poly) > 0:
        poly = cv2.bitwise_and(poly, band)
        clear = cv2.bitwise_or(heuristic, poly)
    else:
        clear = heuristic
    clear = cv2.bitwise_and(clear, selection)
    return clear


def _strengthen_region_transparent_clear_mask(
    target: np.ndarray,
    aligned_source: np.ndarray,
    source_mask: np.ndarray,
    target_clear_mask: np.ndarray,
    selection: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover TARGET lettering fragments missed by the first text-diff pass.

    Region "transparent text" is intentionally *not* a rectangular SOURCE
    reveal: TARGET colour/artwork must remain authoritative.  The old adapter,
    however, treated ``target_clear_mask`` as complete authority.  If the paired
    text detector returned only a subset of one Japanese glyph (AA fringe, a
    dakuten dot, punctuation or a disconnected stroke), those pixels survived
    after the SOURCE Chinese delta was composited.

    The manual selection plus confirmed SOURCE text lane gives us stronger local
    authority than the whole-page automatic renderer has.  Inside that lane we
    can safely recover compact TARGET lettering components, while still refusing
    remote panel lines / hair / artwork.  Work is deliberately performed on the
    tight selection ROI so a 4K/6K manga page does not allocate another stack of
    full-page temporary masks merely for one manual region.
    """
    selection_u8 = np.asarray(selection, np.uint8)
    out = np.zeros_like(selection_u8)
    box = bbox_from_mask(selection_u8)
    if len(box) != 4:
        return out, {
            "base_target_clear_pixels": 0, "source_authority_pixels": 0,
            "recovered_target_text_pixels": 0, "aa_halo_pixels": 0,
            "association_radius": 0, "candidate_target_text_pixels": 0,
            "reason": "empty_selection",
        }
    x0, y0, x1, y1 = [int(v) for v in box]
    sel = selection_u8[y0:y1, x0:x1] > 0
    src = (np.asarray(source_mask[y0:y1, x0:x1], np.uint8) > 0) & sel
    base = (np.asarray(target_clear_mask[y0:y1, x0:x1], np.uint8) > 0) & sel
    base_pixels = int(np.count_nonzero(base)); src_pixels = int(np.count_nonzero(src))
    diag: dict[str, Any] = {
        "base_target_clear_pixels": base_pixels,
        "source_authority_pixels": src_pixels,
        "recovered_target_text_pixels": 0,
        "aa_halo_pixels": 0,
        "association_radius": 0,
        "candidate_target_text_pixels": 0,
        "roi_bbox": [x0, y0, x1, y1],
        "roi_fraction": float(((x1 - x0) * (y1 - y0)) / max(1, selection_u8.shape[0] * selection_u8.shape[1])),
    }
    if src_pixels <= 0 or not np.any(sel):
        out[y0:y1, x0:x1] = base.astype(np.uint8) * 255
        return out, diag

    sw = max(1, x1 - x0); sh = max(1, y1 - y0)
    # Manual text can shift between editions, especially vertical lettering.
    # Keep the corridor generous enough to join disconnected glyph pieces, but
    # hard-cap it so a large freehand selection cannot turn into an artwork erase.
    radius = max(12, min(56, int(round(min(sw, sh) * 0.38))))
    anchor = (src | base).astype(np.uint8)
    near = cv2.dilate(
        anchor,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)),
        iterations=1,
    ) > 0
    near &= sel

    target_crop = target[y0:y1, x0:x1]
    aligned_crop = aligned_source[y0:y1, x0:x1]
    selection_crop = sel.astype(np.uint8) * 255
    # The shared compact-text extractor already rejects broad fills and region
    # boundary components.  The spatial corridor then rejects remote artwork.
    candidates = target_text_mask_in_container(target_crop, selection_crop) > 0
    diff = np.max(
        np.abs(np.asarray(aligned_crop, np.int16) - np.asarray(target_crop, np.int16)),
        axis=2,
    )
    recovered = candidates & near & (diff >= 12)
    strengthened = base | recovered

    # One-pixel antialias/dakuten fringe.  Requiring an actual SOURCE/TARGET
    # difference keeps common panel/hair structure out even when it touches text.
    halo = cv2.dilate(
        strengthened.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    ) > 0
    halo &= near & sel & (diff >= 7)
    final = strengthened | halo
    out[y0:y1, x0:x1] = final.astype(np.uint8) * 255

    diag.update({
        "association_radius": int(radius),
        "candidate_target_text_pixels": int(np.count_nonzero(candidates)),
        "recovered_target_text_pixels": int(np.count_nonzero(recovered & ~base)),
        "aa_halo_pixels": int(np.count_nonzero(final & ~strengthened)),
        "final_target_clear_pixels": int(np.count_nonzero(final)),
        "authority_policy": "manual_source_lane_compact_target_text_v2",
        "roi_fast_path": True,
    })
    return out, diag


def _neutral_paper_selection(target: np.ndarray, selection: np.ndarray) -> tuple[bool, dict[str, Any]]:
    """Return whether a manual selection is a bright/neutral paper container.

    In that case the registered SOURCE raster is already the authoritative
    antialiased Chinese lettering.  Copying those pixels verbatim is both more
    faithful and safer than reconstructing the glyph from a luminance delta.
    Dark TARGET lettering is intentionally included in the statistics; it does
    not make neutral paper look saturated.
    """
    use = np.asarray(selection, np.uint8) > 0
    if not np.any(use):
        return False, {"reason": "empty_selection"}
    pixels = np.asarray(target, np.uint8)[use]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    value = hsv[:, 2].astype(np.float32)
    sat = hsv[:, 1].astype(np.float32)
    bright_neutral = (value >= 210.0) & (sat <= 45.0)
    ratio = float(np.mean(bright_neutral)) if bright_neutral.size else 0.0
    median_v = float(np.median(value)) if value.size else 0.0
    p90_sat = float(np.percentile(sat, 90.0)) if sat.size else 255.0
    safe = bool(ratio >= 0.72 and median_v >= 235.0 and p90_sat <= 55.0)
    return safe, {
        "bright_neutral_ratio": ratio,
        "median_value": median_v,
        "saturation_p90": p90_sat,
        "policy": "exact_source_raster" if safe else "target_relative_delta",
    }


def _clean_region_ocr_target(target: np.ndarray, clear: np.ndarray, selection: np.ndarray, bbox: list[int]) -> tuple[np.ndarray, dict[str, Any]]:
    """Restore OCR-cleared TARGET text without pulling dark artwork into paper.

    Telea is useful on textured artwork, but on a white speech bubble a glyph
    close to a panel line/hair stroke can inherit that dark neighbour and form a
    black wedge/triangle.  Detect bright neutral paper and fill each requested
    glyph component from its own local bright ring.  Non-paper regions keep the
    mature manual TARGET cleanup path.
    """
    mask=(np.asarray(clear,np.uint8)>0).astype(np.uint8)*255
    sel=(np.asarray(selection,np.uint8)>0)
    if cv2.countNonZero(mask)<=0:
        return target.copy(), {"mode":"none","pixels":0}
    gray=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    ring=cv2.dilate(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(17,17)),iterations=1)>0
    ring &= sel & (mask==0)
    vals=gray[ring]; sats=hsv[...,1][ring]
    paper_like=bool(
        vals.size>=32
        and float(np.median(vals))>=210.0
        and float(np.mean(vals>=195))>=0.76
        and float(np.percentile(sats,90.0))<=48.0
    )
    if not paper_like:
        cleaned,diag=mask_ops.clean_manual_target_text(target,mask,bbox=bbox)
        return cleaned,{**dict(diag or {}),"ocr_paper_restore":False}
    out=target.copy(); count,labels,stats,_=cv2.connectedComponentsWithStats((mask>0).astype(np.uint8),8)
    restored=0; fallback=0
    for lab in range(1,count):
        x,y,bw,bh,area=[int(v) for v in stats[lab]]
        if area<=0: continue
        comp=labels==lab
        radius=max(6,min(20,int(round(max(bw,bh)*0.75))))
        local=cv2.dilate(comp.astype(np.uint8),cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,radius*2+1)),iterations=1)>0
        local &= sel & (~comp)
        # Bright/neutral samples reject panel lines, hair and manga ink.
        good=local & (gray>=195) & (hsv[...,1]<=55)
        pixels=target[good]
        if len(pixels)<8:
            pixels=target[local & (gray>=180) & (hsv[...,1]<=70)]
        if len(pixels)<4:
            fallback+=area; continue
        color=np.median(pixels,axis=0).astype(np.uint8)
        out[comp]=color; restored+=area
    if fallback:
        missing=(mask>0)&np.all(out==target,axis=2)
        if np.any(missing):
            # Fall back only for components without a safe paper ring; paste only
            # the requested pixels so no triangular expansion can escape mask.
            repaired,_=mask_ops.clean_manual_target_text(target,missing.astype(np.uint8)*255,bbox=bbox)
            out[missing]=repaired[missing]
    return out,{
        "mode":"ocr_local_paper_restore",
        "pixels":int(cv2.countNonZero(mask)),
        "restored_pixels":int(restored),
        "fallback_pixels":int(fallback),
        "ocr_paper_restore":True,
    }


def apply_region_action(current: np.ndarray, target: np.ndarray, source: np.ndarray,
                        project: dict[str, Any], row: dict[str, Any], cfg: Any,
                        *, page_dir: str | Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    mode = str(row.get("mode") or "").strip().lower()
    if mode not in REGION_MODES:
        raise ValueError(f"unknown region action: {mode}")
    h, w = target.shape[:2]
    out = current.copy(); clear = np.zeros((h, w), np.uint8); layer = np.zeros((h, w, 4), np.uint8)

    # Brush reveal commits are sparse exact-preview patches.  They deliberately
    # do not require a rectangle selection: the painted alpha itself is the
    # authority mask and may span arbitrary disconnected parts of the page.
    if mode == "region_brush_reveal":
        root = Path(page_dir) if page_dir is not None else None
        name = str(row.get("reveal_patch_file") or "").strip()
        box = [int(v) for v in list(row.get("reveal_patch_bbox") or row.get("target_bbox") or [])]
        if root is None or not name or len(box) != 4:
            raise ValueError("涂抹揭示补丁缺少文件或范围")
        x0, y0, x1, y1 = box
        x0=max(0,min(w,x0)); x1=max(0,min(w,x1)); y0=max(0,min(h,y0)); y1=max(0,min(h,y1))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("涂抹揭示补丁范围无效")
        patch_path = safe_page_artifact_path(root, name)
        if patch_path is None:
            raise ValueError("涂抹揭示补丁路径无效")
        patch = cv2.imread(str(patch_path), cv2.IMREAD_UNCHANGED)
        if patch is None or patch.ndim != 3 or patch.shape[2] != 4 or patch.shape[:2] != (y1-y0, x1-x0):
            raise ValueError("涂抹揭示补丁缺失或尺寸不一致")
        alpha = np.asarray(patch[:, :, 3], np.uint8).copy()
        mask_name = str(row.get("reveal_mask_file") or "").strip()
        if mask_name:
            mask_path = safe_page_artifact_path(root, mask_name)
            if mask_path is None:
                raise ValueError("涂抹揭示 authority mask 路径无效")
            authority = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if authority is None or authority.shape != alpha.shape:
                raise ValueError("涂抹揭示 authority mask 缺失或尺寸不一致")
            alpha[authority == 0] = 0
            patch = patch.copy(); patch[:, :, 3] = alpha
        if cv2.countNonZero(alpha) <= 0:
            raise ValueError("涂抹揭示补丁为空")
        base_roi = out[y0:y1, x0:x1].copy()
        out[y0:y1, x0:x1] = _alpha_composite(base_roi, patch[:, :, :3], alpha)
        layer[y0:y1, x0:x1] = patch
        # Do not report the brush alpha as a TARGET inpaint mask.  The sparse
        # patch already represents the exact top-layer transparency/cut-out;
        # feeding this alpha into review_base inpainting would unnecessarily
        # alter manga background underneath a reveal stroke.
        changed = np.any(out[y0:y1, x0:x1] != base_roi, axis=2)
        rec = {
            "id": str(row.get("id") or "region-brush-reveal"),
            "success": True,
            "mode": mode,
            "region_composite": True,
            "brush_reveal": True,
            "target_bbox": [x0,y0,x1,y1],
            "write_pixels": int(cv2.countNonZero(alpha)),
            "authority_pixels": int(cv2.countNonZero(alpha)),
            "changed_pixels": int(np.count_nonzero(changed)),
            "pixel_changed": bool(np.any(changed)),
            "idempotent_success": bool(cv2.countNonZero(alpha) > 0 and not np.any(changed)),
            "transparent_pixels": int(row.get("transparent_pixels", 0) or 0),
            "hole_pixels": int(row.get("hole_pixels", 0) or 0),
            "preview_patch_replayed": True,
            "diagnostics": {"sparse_patch": True, "patch_shape": [int(y1-y0), int(x1-x0)]},
        }
        return out, layer, clear, rec

    selection = selection_mask_from_row(row, (h, w))
    selected_px = int(cv2.countNonZero(selection))
    if selected_px <= 0:
        raise ValueError("区域工具没有有效选区")
    bbox = bbox_from_mask(selection)
    diag: dict[str, Any] = {"selection_pixels": selected_px, "selection_bbox": bbox}

    if mode in {"region_direct_patch", "region_hole_reveal"}:
        roi = _roi_bounds(selection, halo=max(10, int(row.get("feather_px", 0) or 0) * 4))
        xa, ya, xb, yb = roi
        aligned_crop, identity = _aligned_source_roi(
            source, target.shape[:2], project, roi,
            source_offset_x=int(row.get("source_offset_x", 0) or 0),
            source_offset_y=int(row.get("source_offset_y", 0) or 0),
        )
        effective = selection[ya:yb, xa:xb].copy()
        if mode == "region_hole_reveal":
            inset = max(0, min(12, int(row.get("inset_px", 1) or 0)))
            if inset:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (inset * 2 + 1, inset * 2 + 1))
                effective = cv2.erode(effective, k, iterations=1)
        alpha_crop = _feather(effective, int(row.get("feather_px", 0 if mode == "region_direct_patch" else 1) or 0))
        current_crop = out[ya:yb, xa:xb].copy()
        out[ya:yb, xa:xb] = _alpha_composite(current_crop, aligned_crop, alpha_crop)
        layer_crop = _layer_from_rgb(aligned_crop, alpha_crop); layer[ya:yb, xa:xb] = layer_crop
        diag.update({
            "identity_pixel_lock": bool(identity), "write_pixels": int(cv2.countNonZero(alpha_crop)),
            "inset_px": int(row.get("inset_px", 0) or 0), "roi_fast_path": True,
            "roi_bbox": [xa, ya, xb, yb], "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
        })

    elif mode == "region_precise_mask":
        halo = _precise_mask_roi_halo(selection, cfg)
        roi = _roi_bounds(selection, halo=halo); xa, ya, xb, yb = roi
        aligned_crop, identity = _aligned_source_roi(
            source, target.shape[:2], project, roi,
            source_offset_x=int(row.get("source_offset_x", 0) or 0),
            source_offset_y=int(row.get("source_offset_y", 0) or 0),
        )
        target_crop = target[ya:yb, xa:xb]
        selection_crop = selection[ya:yb, xa:xb]
        renderer, cfg_obj, renderer_owner = _precise_region_engine(project, cfg)
        rendered_crop, write_crop, src_crop, raw_diag = renderer(
            aligned_crop, target_crop, selection_crop, cfg_obj
        )
        raw_diag = dict(raw_diag or {})
        first_failure_diag = dict(raw_diag)
        first_reason = str(raw_diag.get("reason") or "精准蒙版区域处理失败")
        manual_threshold_retry = False
        manual_threshold_retry_succeeded = False
        # Automatic Precise has intentionally conservative minimum-ink gates.
        # A human Region is already authoritative, so retry *only* threshold-like
        # failures with a cloned config whose minimum evidence is one pixel.  This
        # never mutates cfg_obj and therefore cannot weaken whole-page automation.
        if rendered_crop is None and first_reason in {"insufficient_source_ink", "insufficient_target_ink"}:
            manual_threshold_retry = True
            relaxed_cfg = _manual_precise_relaxed_cfg(cfg_obj)
            rendered_crop, write_crop, src_crop, retry_diag = renderer(
                aligned_crop, target_crop, selection_crop, relaxed_cfg
            )
            retry_diag = dict(retry_diag or {})
            if rendered_crop is not None:
                manual_threshold_retry_succeeded = True
                raw_diag = retry_diag
            else:
                # If the Precise detector still cannot form a mask, fall back to
                # the detector-independent manual text extractor.  This is not a
                # Direct/full-rectangle paste: TARGET background stays authoritative.
                fb = _precise_manual_transparent_fallback(
                    current, target, source, project, row, cfg, page_dir=page_dir,
                    original_reason=str(retry_diag.get("reason") or first_reason),
                    original_diag={
                        "first": first_failure_diag,
                        "relaxed_retry": retry_diag,
                    },
                )
                if fb is not None:
                    return fb
                raw_diag = retry_diag or first_failure_diag
        if rendered_crop is None:
            reason = str((raw_diag or {}).get("reason") or first_reason or "精准蒙版区域处理失败")
            if reason in {"insufficient_source_ink", "insufficient_target_ink"}:
                reason = "manual_precise_no_text_authority"
            raise ValueError(reason)
        write_crop = cv2.bitwise_and(np.asarray(write_crop, np.uint8), selection_crop)
        src_crop = cv2.bitwise_and(np.asarray(src_crop, np.uint8), selection_crop)
        clear_crop = cv2.bitwise_and(
            np.asarray((raw_diag or {}).get("clear_mask", np.zeros_like(selection_crop)), np.uint8), selection_crop
        )
        # Do not rely on one private renderer always including clear pixels in
        # its write mask.  Region composition owns the linkage contract: every
        # selected TARGET-clear pixel and every SOURCE-write pixel is applied to
        # the *current reviewed result*, while everything else stays untouched.
        use = (write_crop > 0) | (clear_crop > 0)
        out_roi = out[ya:yb, xa:xb].copy(); out_roi[use] = rendered_crop[use]; out[ya:yb, xa:xb] = out_roi
        clear[ya:yb, xa:xb] = clear_crop
        layer_crop = _layer_from_rgb(aligned_crop, src_crop); layer[ya:yb, xa:xb] = layer_crop
        clean_diag = {k:v for k,v in dict(raw_diag or {}).items() if not isinstance(v, np.ndarray)}
        diag.update(clean_diag)
        diag.update({
            "manual_open_text_box": True, "identity_pixel_lock": bool(identity),
            "renderer_owner": renderer_owner,
            "write_pixels": int(cv2.countNonZero(write_crop)), "ocr_used": False,
            "roi_fast_path": True, "roi_bbox": [xa, ya, xb, yb], "roi_halo": int(halo),
            "roi_fraction": float(((xb-xa)*(yb-ya))/max(1,h*w)),
            "manual_threshold_retry": bool(manual_threshold_retry),
            "manual_threshold_retry_succeeded": bool(manual_threshold_retry_succeeded),
            "automatic_thresholds_unchanged": True,
        })
        if manual_threshold_retry:
            diag["automatic_failure"] = {
                k: v for k, v in first_failure_diag.items() if not isinstance(v, np.ndarray)
            }

    elif mode == "region_transparent":
        local_ops = _transparent_region_ops(project)
        request = deepcopy(row); request["mode"] = "effect_text"
        masks = local_ops.build_manual_effect_masks(source, target, project, request, cfg)
        src_mask = cv2.bitwise_and(np.asarray(masks.source_mask, np.uint8), selection)
        clear_base = cv2.bitwise_and(np.asarray(masks.target_clear_mask, np.uint8), selection)
        clear, clear_diag = _strengthen_region_transparent_clear_mask(
            target, masks.aligned_source, src_mask, clear_base, selection
        )
        # Re-assert the explicit manual authority boundary after every recovery
        # stage.  This remains a TARGET-background-preserving operation: unlike
        # brush reveal, we do not replace the whole corridor with SOURCE pixels.
        clear = cv2.bitwise_and(clear, selection)
        # Build a canonical TARGET-derived base for this region.  Never add the
        # SOURCE text delta to the *current* reviewed raster: doing so made a
        # repeated identical region action progressively darker/thicker.
        canonical_base = target.copy()
        if cv2.countNonZero(clear):
            cleaned, clean_diag = local_ops.clean_manual_target_text(target, clear, bbox=bbox)
            use = clear > 0
            out[use] = cleaned[use]
            canonical_base[use] = cleaned[use]
            diag["target_cleanup"] = clean_diag
        if cv2.countNonZero(src_mask):
            alpha_u8 = _feather(src_mask, int(row.get("feather_px", 0) or 0))
            write = alpha_u8 > 0
            paper_safe, paper_diag = _neutral_paper_selection(canonical_base, selection)
            if paper_safe:
                # Match the successful brush-reveal contract: on white/neutral
                # paper the SOURCE pixels themselves contain the correct glyph
                # antialiasing.  Do not dilate, threshold, outline, or re-render
                # them.  For the normal feather=0 case this is byte-for-byte
                # SOURCE fidelity inside the confirmed text authority mask.
                canonical = _alpha_composite(canonical_base, masks.aligned_source, alpha_u8)
                delta_diag = {
                    "mode": "exact_source_raster",
                    "source_exact_pixels": int(np.count_nonzero(write)),
                    "source_exact_byte_match": bool(
                        int(row.get("feather_px", 0) or 0) == 0
                        and np.array_equal(canonical[write], masks.aligned_source[write])
                    ),
                }
            else:
                # Colour/artwork backgrounds keep TARGET authoritative, but the
                # result is still computed from the fixed TARGET base.  Writing
                # this canonical raster makes replay idempotent instead of
                # accumulating the dark text delta on every repeated commit.
                bg = local_ops.estimate_source_background(masks.aligned_source, masks.source_mask)
                alpha = alpha_u8.astype(np.float32) / 255.0
                canonical, delta_diag = local_ops.composite_source_text_delta(
                    canonical_base, masks.aligned_source, src_mask,
                    source_background=bg, alpha=alpha,
                )
            out[write] = canonical[write]
            layer = _layer_from_rgb(canonical, alpha_u8)
            diag["delta_composite"] = delta_diag
            diag["source_fidelity"] = {
                **paper_diag,
                "idempotent_canonical_base": True,
                "write_pixels": int(np.count_nonzero(write)),
            }
        diag["transparent_clear_authority"] = clear_diag
        diag.update({
            "source_pixels": int(cv2.countNonZero(src_mask)),
            "target_clear_pixels": int(cv2.countNonZero(clear)),
            "target_clear_base_pixels": int(cv2.countNonZero(clear_base)),
            "ocr_used": False,
            "renderer_owner": "hybrid" if local_ops is hybrid_mask_ops else "mask_replace",
        })

    elif mode == "region_ocr":
        text = str(row.get("render_text") or row.get("ocr_text") or "").strip()
        if not text:
            raise ValueError("OCR 区域没有可排版的中文文本")
        clear = _refine_ocr_clear_mask(target, selection, _polygon_mask((h, w), list(row.get("target_ocr_polygons") or [])))
        if cv2.countNonZero(clear):
            cleaned, clean_diag = _clean_region_ocr_target(target, clear, selection, bbox)
            use = clear > 0; out[use] = cleaned[use]; diag["target_cleanup"] = clean_diag
        lcfg = cfg.lettering.model_copy(deep=True)
        lcfg.orientation = str(row.get("orientation") or "auto")
        font_path = str(row.get("font_path") or "").strip()
        if font_path:
            lcfg.font_path = font_path
        font_size = int(row.get("font_size") or 0)
        if font_size > 0:
            lcfg.min_font_size = font_size; lcfg.max_font_size = font_size; lcfg.preferred_font_size = font_size
        columns = int(row.get("columns") or 0)
        if columns > 0:
            lcfg.preferred_columns = columns
        if str(row.get("line_break_mode") or "smart") in {"smart", "balanced", "source"}:
            lcfg.line_break_mode = str(row.get("line_break_mode") or "smart")
        if str(row.get("layout_mode") or "smart_scaling") in {"strict", "smart_scaling", "balloon_fill"}:
            lcfg.layout_mode = str(row.get("layout_mode") or "smart_scaling")
        x0, y0, x1, y1 = bbox
        unit = TextUnit(
            id=str(row.get("id") or "region-ocr"), polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
            block_ids=[], text=text, confidence=float(row.get("confidence") or 1.0), kind="speech", reading_order=0,
            bubble_id=None, meta={"manual_region_ocr": True, "box_locked": True},
        )
        lr = fit_text(target.shape[:2], selection, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            raise ValueError(str(lr.reason or "OCR 区域排版失败"))
        before = out.copy(); out = composite_text(out, lr, lcfg)
        text_mask = cv2.bitwise_and(np.asarray(lr.text_mask, np.uint8), selection)
        # composite_text is constrained by fit mask, but enforce selection again
        # so a future renderer change cannot leak outside the explicit manual ROI.
        outside = selection == 0; out[outside] = before[outside]
        layer = np.zeros((h, w, 4), np.uint8); use = text_mask > 0; layer[use, :3] = out[use]; layer[use, 3] = text_mask[use]
        diag.update({"text": text, "text_pixels": int(cv2.countNonZero(text_mask)), "font_path": lr.font_path, "font_size": int(lr.font_size), "orientation": lr.orientation, "ocr_used": True})

    changed = np.any(out != current, axis=2)
    outside_changed = int(np.count_nonzero(changed & (selection == 0)))
    if outside_changed:
        raise RuntimeError(f"区域工具越界写入 {outside_changed} px")

    # Success is an execution/authority contract, not a pixel-delta contract.
    # A manual region can be perfectly valid yet idempotent: e.g. the selected
    # SOURCE pixels are already present in the current review base, or a reviewer
    # intentionally reapplies the same operation.  Older code used
    # ``bool(np.any(changed))`` which wrote success=false for such rows; the
    # transaction then threw *after* the visual operation had already completed.
    # Keep the two concepts separate:
    #   - authority_pixels > 0 => the region action genuinely had something to do
    #   - changed_pixels > 0   => this replay changed the current raster
    # Empty authority remains a real failure and is not silently accepted.
    layer_alpha = layer[:, :, 3] if layer.ndim == 3 and layer.shape[2] >= 4 else np.zeros_like(clear)
    authority_mask = np.maximum(np.asarray(clear, np.uint8), np.asarray(layer_alpha, np.uint8))
    authority_pixels = int(cv2.countNonZero(authority_mask))
    pixel_changed = bool(np.any(changed))
    audit = {
        "id": str(row.get("id") or ""), "success": bool(authority_pixels > 0), "mode": mode,
        "target_bbox": bbox, "selection_kind": str((row.get("selection_spec") or {}).get("kind") or "rect"),
        "changed_pixels": int(np.count_nonzero(changed)), "pixel_changed": pixel_changed,
        "idempotent_success": bool(authority_pixels > 0 and not pixel_changed),
        "authority_pixels": authority_pixels,
        "outside_selection_changed_pixels": outside_changed,
        "target_clear_pixels": int(cv2.countNonZero(clear)), "diagnostics": diag,
    }
    if authority_pixels <= 0:
        audit["reason"] = "region_action_empty_authority"
    return out, layer, clear, audit


__all__ = ["REGION_MODES", "is_region_mode", "apply_region_action"]
