from __future__ import annotations

import json
import re
import threading
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .config import BubbleConfig
from .model_downloads import discovered_model_path
from .geometry import bbox_polygon, mask_to_largest_polygon, polygon_bbox, polygon_centroid, rasterize_polygon, transform_to_homography, union_bbox
from .models import BubbleInstance, TextBlock, TextUnit
from .runtime import accelerator_lock, select_device

_YOLO_MODEL_CACHE: dict[tuple[str, int, int], object] = {}
_YOLO_MODEL_LOCK = threading.RLock()

_CJK_RE = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uf900-\ufaff]")


def _mangalens_model_cache_key(path: str | Path) -> tuple[str, int, int]:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"MangaLens model does not exist: {p}")
    st = p.stat()
    return (str(p), int(st.st_size), int(st.st_mtime_ns))


def _nearest_white_seed(mask: np.ndarray, x: int, y: int, radius: int) -> tuple[int, int] | None:
    h, w = mask.shape
    x = int(np.clip(x, 0, w - 1))
    y = int(np.clip(y, 0, h - 1))
    if mask[y, x] > 0:
        return x, y
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    ys, xs = np.where(mask[y0:y1, x0:x1] > 0)
    if len(xs) == 0:
        return None
    xs = xs + x0
    ys = ys + y0
    dist2 = (xs - x) ** 2 + (ys - y) ** 2
    idx = int(np.argmin(dist2))
    return int(xs[idx]), int(ys[idx])


def _safe_mask(component: np.ndarray, seed: tuple[int, int], margin: int) -> np.ndarray:
    binary = (component > 0).astype(np.uint8)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    safe = (dist >= max(1, margin)).astype(np.uint8) * 255
    if not np.any(safe):
        # Relax instead of returning an unusable region.
        safe = (dist >= max(1, margin // 2)).astype(np.uint8) * 255
    if not np.any(safe):
        return component.copy()

    count, labels, stats, _ = cv2.connectedComponentsWithStats((safe > 0).astype(np.uint8), 8)
    sx, sy = seed
    wanted = labels[int(np.clip(sy, 0, labels.shape[0] - 1)), int(np.clip(sx, 0, labels.shape[1] - 1))]
    if wanted == 0:
        if count <= 1:
            return safe
        wanted = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == wanted).astype(np.uint8) * 255


def detect_seeded_white_bubbles(
    image: np.ndarray, blocks: list[TextBlock], config: BubbleConfig | None = None
) -> list[BubbleInstance]:
    cfg = config or BubbleConfig()
    if not blocks:
        return []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # White-ish interior. A mild close fills tiny raster/compression gaps without deleting borders.
    white = (gray >= cfg.white_threshold).astype(np.uint8) * 255
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    h, w = gray.shape
    page_area = h * w
    label_to_bubble: dict[int, BubbleInstance] = {}

    for block in blocks:
        cx, cy = block.centroid
        seed = _nearest_white_seed(white, round(cx), round(cy), cfg.search_radius)
        if seed is None:
            continue
        sx, sy = seed
        label = int(labels[sy, sx])
        if label <= 0:
            continue
        area = int(stats[label, cv2.CC_STAT_AREA])
        ratio = area / max(1, page_area)
        if ratio < cfg.min_area_ratio or ratio > cfg.max_area_ratio:
            continue
        if label not in label_to_bubble:
            raw_component = (labels == label).astype(np.uint8) * 255
            # Text glyphs are dark holes inside an otherwise white bubble. For layout
            # geometry they must be considered usable interior after clearing, so fill
            # the external contour instead of preserving glyph-shaped holes.
            contours, _ = cv2.findContours(raw_component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            outer = max(contours, key=cv2.contourArea)
            component = np.zeros_like(raw_component)
            cv2.drawContours(component, [outer], -1, 255, thickness=cv2.FILLED)
            polygon = mask_to_largest_polygon(component)
            if len(polygon) < 3:
                continue
            x, y, bw, bh, _ = stats[label]
            margin = max(cfg.safe_margin_px, int(min(bw, bh) * cfg.safe_margin_ratio))
            safe = _safe_mask(component, seed, margin)
            contour_area = max(1.0, float(cv2.countNonZero(component)))
            rect_fill = contour_area / max(1.0, bw * bh)
            approx = cv2.approxPolyDP(
                max(cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea),
                0.02 * (bw + bh),
                True,
            )
            kind = "narration" if len(approx) <= 5 and rect_fill > 0.82 else "speech"
            label_to_bubble[label] = BubbleInstance(
                id=f"bubble-{len(label_to_bubble):04d}",
                polygon=polygon,
                confidence=float(np.clip(0.55 + 0.35 * min(1.0, rect_fill), 0.0, 0.95)),
                kind=kind,
                mask=component,
                safe_mask=safe,
                meta={"area_ratio": ratio, "rect_fill": rect_fill, "seed_label": label, "safe_margin": margin},
            )
        bubble = label_to_bubble[label]
        bubble.block_ids.append(block.id)
        block.bubble_id = bubble.id
        if block.kind == "unknown":
            block.kind = bubble.kind

    return list(label_to_bubble.values())



def detect_unseeded_white_containers(
    image: np.ndarray,
    config: BubbleConfig | None = None,
    *,
    prefix: str = "white",
) -> list[BubbleInstance]:
    """Detect enclosed white speech/narration containers without OCR seeds.

    This is intentionally a conservative completion detector, not a replacement
    for normal text detection. It is used only for strong same-layout B/W ->
    colour registration. Long panel rules and page margins are filtered before
    the regular rigid-container eligibility gate performs the final safety check.
    """
    cfg = config or BubbleConfig()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    if image.ndim == 3:
        sat = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1]
    else:
        sat = np.zeros_like(gray)
    # Keep this detector's thresholds on the config object dynamically so older
    # BubbleConfig instances remain compatible with v0.8.25.
    white_thr = int(getattr(cfg, "unseeded_white_threshold", 210))
    max_sat = int(getattr(cfg, "unseeded_max_saturation", 70))
    min_area = float(getattr(cfg, "unseeded_min_area_ratio", 0.0005))
    max_area = float(getattr(cfg, "unseeded_max_area_ratio", 0.12))
    min_white = float(getattr(cfg, "unseeded_min_white_ratio", 0.55))
    min_fill = float(getattr(cfg, "unseeded_min_fill_ratio", 0.30))
    min_dark = float(getattr(cfg, "unseeded_min_dark_ratio", 0.008))
    max_dark = float(getattr(cfg, "unseeded_max_dark_ratio", 0.28))
    max_aspect = float(getattr(cfg, "unseeded_max_aspect", 5.0))
    short_rect_enabled = bool(getattr(cfg, "unseeded_short_text_rectangle_enabled", True))
    short_rect_fill = float(getattr(cfg, "unseeded_short_text_min_rect_fill", 0.88))
    short_rect_white = float(getattr(cfg, "unseeded_short_text_min_white_ratio", 0.72))
    short_rect_dark = float(getattr(cfg, "unseeded_short_text_min_dark_ratio", 0.002))
    max_candidates = int(getattr(cfg, "unseeded_max_candidates", 96))

    white = ((gray >= white_thr) & (sat <= max_sat)).astype(np.uint8) * 255
    if cfg.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    h, w = gray.shape; page_area = max(1, h * w)
    rows: list[tuple[float, BubbleInstance]] = []
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        aratio = area / page_area
        if aratio < min_area or aratio > max_area:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
        if aspect > max_aspect:
            continue
        raw = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        outer = max(contours, key=cv2.contourArea)
        component = np.zeros_like(raw); cv2.drawContours(component, [outer], -1, 255, cv2.FILLED)
        bx, by, cw, ch = cv2.boundingRect(outer)
        fill_ratio = float(cv2.countNonZero(component) / max(1, cw * ch))
        if fill_ratio < min_fill:
            continue
        inner = cv2.erode(component, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        inner_sel = inner > 0
        vals = gray[inner_sel]
        if vals.size < 24:
            continue
        sat_vals = sat[inner_sel]
        dark_ratio = float(np.mean(vals < 190))
        white_ratio = float(np.mean(vals > 225))
        saturation_median = float(np.median(sat_vals)) if sat_vals.size else 0.0
        saturation_p75 = float(np.percentile(sat_vals, 75.0)) if sat_vals.size else 0.0
        if dark_ratio < min_dark or dark_ratio > max_dark or white_ratio < min_white:
            continue
        # Require at least a little compact character-like ink. This removes many
        # empty white architecture patches while keeping punctuation and short
        # speech. Large/long components are ignored as panel rules or artwork.
        dark = ((gray < 190) & (inner > 0)).astype(np.uint8)
        cc, _labs, ccstats, _ = cv2.connectedComponentsWithStats(dark, 8)
        compact = 0
        compact_area = 0
        for j in range(1, cc):
            _x, _y, ww, hh, aa = [int(v) for v in ccstats[j]]
            if aa < 2 or aa > max(600, int(0.03 * max(1, cv2.countNonZero(inner)))):
                continue
            if ww > 0.34 * cw or hh > 0.34 * ch:
                continue
            if max(ww / max(1.0, hh), hh / max(1.0, ww)) > 10.0:
                continue
            compact += 1; compact_area += aa
        compact_density = compact_area / max(1, cv2.countNonZero(inner))
        # A very short vertical narration may be only one glyph plus a long dash.
        # Old code required >=2 compact components and silently missed such boxes.
        # Rescue only strong neutral rectangular containers; later registered
        # SOURCE/TARGET ink-change evidence is still mandatory before transfer.
        short_rect = bool(
            short_rect_enabled
            and compact >= 1
            and fill_ratio >= short_rect_fill
            and white_ratio >= short_rect_white
            and dark_ratio >= short_rect_dark
            and saturation_p75 <= 24.0
        )
        if compact < 2 and compact_density < 0.004 and not short_rect:
            continue
        polygon = mask_to_largest_polygon(component)
        if len(polygon) < 3:
            continue
        margin = max(1, min(cfg.safe_margin_px, max(1, int(min(cw, ch) * 0.025))))
        # Seed with the centre; _safe_mask falls back to largest inner component.
        safe = _safe_mask(component, (bx + cw // 2, by + ch // 2), margin)
        approx = cv2.approxPolyDP(outer, 0.02 * (cw + ch), True)
        kind = "narration" if len(approx) <= 5 and fill_ratio > 0.82 else "speech"
        confidence = float(np.clip(0.55 + 0.25 * white_ratio + 0.12 * min(1.0, compact / 8.0), 0.0, 0.96))
        bubble = BubbleInstance(
            id=f"{prefix}-{len(rows):04d}", polygon=polygon, confidence=confidence,
            kind=kind, block_ids=[], mask=component, safe_mask=safe,
            meta={"backend": "unseeded_white", "area_ratio": aratio, "rect_fill": fill_ratio,
                  "white_ratio": white_ratio, "dark_ratio": dark_ratio, "compact_components": compact,
                  "saturation_median": saturation_median, "saturation_p75": saturation_p75},
        )
        # Prefer smaller, text-dense candidates if a safety cap is reached.
        priority = confidence + 0.15 * min(1.0, dark_ratio / 0.08) - 0.15 * aratio
        rows.append((priority, bubble))
    rows.sort(key=lambda x: x[0], reverse=True)
    out = [b for _, b in rows[:max_candidates]]
    for i, b in enumerate(out): b.id = f"{prefix}-{i:04d}"
    return out


def detect_target_colored_containers(image: np.ndarray, *, prefix: str = "color") -> list[BubbleInstance]:
    """Find compact coloured burst/text containers on the HD target page.

    These are common manga SFX/dialogue shapes (red/yellow starbursts).  They
    cannot pass the white-container detector, but their geometry is still a
    reliable target-side replacement boundary when the source edition is
    monochrome.
    """
    if image.ndim != 3:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    colour = ((hsv[..., 1] >= 85) & (hsv[..., 2] >= 55)).astype(np.uint8) * 255
    colour = cv2.morphologyEx(colour, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats((colour > 0).astype(np.uint8), 8)
    h, w = gray.shape; page_area = max(1, h * w); out = []
    for label in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[label]]
        ratio = area / page_area
        if ratio < 0.001 or ratio > 0.06 or x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        aspect = max(bw / max(1, bh), bh / max(1, bw))
        if aspect > 4.5:
            continue
        raw = (labels == label).astype(np.uint8) * 255
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        mask = np.zeros_like(raw); cv2.drawContours(mask, [contour], -1, 255, cv2.FILLED)
        inner = cv2.erode(mask, np.ones((5, 5), np.uint8))
        if cv2.countNonZero(inner) < 30:
            continue
        dark_ratio = float(np.mean(gray[inner > 0] < 175))
        sat_median = float(np.median(hsv[..., 1][inner > 0]))
        if sat_median < 70 or dark_ratio < 0.006:
            continue
        poly = mask_to_largest_polygon(mask)
        if len(poly) < 3:
            continue
        safe = _safe_mask(mask, (x + bw // 2, y + bh // 2), max(1, int(min(bw, bh) * 0.02)))
        out.append(BubbleInstance(
            id=f"{prefix}-{len(out):04d}", polygon=poly, confidence=0.82,
            kind="complex_text", block_ids=[], mask=mask, safe_mask=safe,
            meta={"backend": "target_colored_container", "target_colored_recovery": True,
                  "area_ratio": ratio, "dark_ratio": dark_ratio, "target_saturation": sat_median},
        ))
    return out


def _registered_ink_change_evidence(
    source_gray_warped: np.ndarray,
    target_gray: np.ndarray,
    target_mask: np.ndarray,
    mask_config,
) -> dict:
    """Measure whether a registered white region behaves like translated text.

    Publication safety requires more than a white connected component containing
    dark pixels. White clothes, windows and paving often satisfy that condition.
    Real translated text should show a meaningful SOURCE/TARGET ink-identity
    change after page registration, while same artwork should remain highly
    overlapping under a small tolerance.
    """
    m = (target_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(m) <= 0:
        return {"valid": False, "reason": "empty_mask"}
    inner = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    if cv2.countNonZero(inner) < 24:
        inner = m
    threshold = int(getattr(mask_config, "rigid_container_unseeded_ink_threshold", 190))
    s = ((source_gray_warped < threshold) & (inner > 0)).astype(np.uint8) * 255
    t = ((target_gray < threshold) & (inner > 0)).astype(np.uint8) * 255
    area = max(1, cv2.countNonZero(inner))
    sc = cv2.countNonZero(s); tc = cv2.countNonZero(t)
    sdens = float(sc / area); tdens = float(tc / area)
    tol = max(1, int(getattr(mask_config, "rigid_container_unseeded_ink_match_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    sd = cv2.dilate(s, k); td = cv2.dilate(t, k)
    smatch = float(np.count_nonzero((s > 0) & (td > 0)) / max(1, sc)) if sc else 0.0
    tmatch = float(np.count_nonzero((t > 0) & (sd > 0)) / max(1, tc)) if tc else 0.0
    identity = float(np.clip(0.5 * (smatch + tmatch), 0.0, 1.0))
    change = float(np.clip(1.0 - identity, 0.0, 1.0))
    min_sd = float(getattr(mask_config, "rigid_container_unseeded_min_source_ink_density", 0.025))
    min_td = float(getattr(mask_config, "rigid_container_unseeded_min_target_ink_density", 0.012))
    min_change = float(getattr(mask_config, "rigid_container_unseeded_min_ink_change_score", 0.08))
    max_ratio = float(getattr(mask_config, "rigid_container_unseeded_max_ink_density_ratio", 3.5))
    density_ratio = float(max(sdens, tdens) / max(1e-6, min(sdens, tdens))) if min(sdens, tdens) > 0 else 999.0
    passed = bool(sdens >= min_sd and tdens >= min_td and change >= min_change and density_ratio <= max_ratio)
    return {
        "valid": True,
        "passed": passed,
        "source_ink_density_registered": sdens,
        "target_ink_density_registered": tdens,
        "source_ink_match": smatch,
        "target_ink_match": tmatch,
        "ink_identity_overlap": identity,
        "ink_change_score": change,
        "ink_density_ratio": density_ratio,
        "thresholds": {"min_source_ink_density": min_sd, "min_target_ink_density": min_td, "min_ink_change_score": min_change, "max_ink_density_ratio": max_ratio},
    }


def pair_unseeded_white_containers(
    source: np.ndarray,
    target: np.ndarray,
    registration,
    mask_config,
    bubble_config: BubbleConfig | None = None,
    existing_target_bubbles: list[BubbleInstance] | None = None,
) -> tuple[list[BubbleInstance], list[BubbleInstance]]:
    """Pair OCR-free white containers through the existing page registration."""
    if not bool(getattr(mask_config, "rigid_container_unseeded_completion_enabled", True)):
        return [], []
    if float(getattr(registration, "confidence", 0.0)) < float(getattr(mask_config, "rigid_container_unseeded_min_registration_confidence", 0.72)):
        return [], []
    cfg = (bubble_config or BubbleConfig()).model_copy(deep=True) if hasattr((bubble_config or BubbleConfig()), "model_copy") else (bubble_config or BubbleConfig())
    # Project mask-replace tuning into this cheap detector without expanding the
    # public BubbleConfig surface for legacy callers.
    for name, default in (
        ("unseeded_white_threshold", 210), ("unseeded_max_saturation", 70),
        ("unseeded_min_area_ratio", 0.0005), ("unseeded_max_area_ratio", 0.12),
        ("unseeded_min_white_ratio", 0.55), ("unseeded_min_fill_ratio", 0.30), ("unseeded_min_dark_ratio", 0.008),
        ("unseeded_max_dark_ratio", 0.28), ("unseeded_max_aspect", 5.0),
        ("unseeded_short_text_rectangle_enabled", True),
        ("unseeded_short_text_min_rect_fill", 0.88),
        ("unseeded_short_text_min_white_ratio", 0.72),
        ("unseeded_short_text_min_dark_ratio", 0.002),
        ("unseeded_max_candidates", 96),
    ):
        object.__setattr__(cfg, name, getattr(mask_config, f"rigid_container_{name}", default))
    src = detect_unseeded_white_containers(source, cfg, prefix="auto-src")
    dst = detect_unseeded_white_containers(target, cfg, prefix="auto-dst")
    colored_dst = detect_target_colored_containers(target, prefix="auto-color-dst")
    dst.extend(colored_dst)
    if not dst:
        return [], []
    H = transform_to_homography(registration.matrix); th, tw = target.shape[:2]
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return [], []
    source_gray_page = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    target_gray_page = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    source_gray_warped = cv2.warpPerspective(source_gray_page, H, (tw, th), flags=cv2.INTER_LINEAR, borderValue=255)
    existing = list(existing_target_bubbles or [])
    existing_masks = [b.mask for b in existing if b.mask is not None and b.mask.shape == (th, tw)]
    min_cov = float(getattr(mask_config, "rigid_container_unseeded_min_pair_coverage", 0.82))
    min_iou = float(getattr(mask_config, "rigid_container_unseeded_min_pair_iou", 0.30))
    max_existing = float(getattr(mask_config, "rigid_container_unseeded_existing_overlap", 0.62))
    scored: list[tuple[float, float, float, int, int]] = []
    warped: list[np.ndarray] = []
    for i, sb in enumerate(src):
        wm = cv2.warpPerspective((sb.mask > 0).astype(np.uint8) * 255, H, (tw, th), flags=cv2.INTER_NEAREST)
        warped.append(wm); wa = max(1, cv2.countNonZero(wm))
        for j, tb in enumerate(dst):
            tm = (tb.mask > 0).astype(np.uint8) * 255; ta = max(1, cv2.countNonZero(tm))
            inter = cv2.countNonZero(cv2.bitwise_and(wm, tm))
            if inter <= 0: continue
            iou = inter / max(1, wa + ta - inter); coverage = inter / max(1, min(wa, ta))
            if iou < min_iou and coverage < min_cov:
                continue
            score = 0.58 * coverage + 0.42 * iou
            scored.append((score, iou, coverage, i, j))
    scored.sort(reverse=True)
    used_s: set[int] = set(); used_t: set[int] = set(); out_s: list[BubbleInstance] = []; out_t: list[BubbleInstance] = []
    for score, iou, coverage, i, j in scored:
        if i in used_s or j in used_t:
            continue
        tb = dst[j]; tm = (tb.mask > 0).astype(np.uint8) * 255; ta = max(1, cv2.countNonZero(tm))
        if bool(getattr(mask_config, "rigid_container_unseeded_ink_change_gate_enabled", True)):
            ink_ev = _registered_ink_change_evidence(source_gray_warped, target_gray_page, tm, mask_config)
            tb.meta["registered_ink_change"] = ink_ev
            if not bool(ink_ev.get("passed", False)):
                continue
        duplicate = False
        for em in existing_masks:
            inter = cv2.countNonZero(cv2.bitwise_and(tm, (em > 0).astype(np.uint8) * 255))
            if inter / max(1, min(ta, cv2.countNonZero(em))) >= max_existing:
                duplicate = True; break
        if duplicate:
            used_s.add(i); used_t.add(j); continue
        sb = src[i]
        tid = f"auto-dst-{len(out_t):04d}"; sid = f"auto-src-{len(out_s):04d}"
        sb.id = sid; tb.id = tid
        # Target container geometry is the truth for same-layout editions. A
        # translated B/W page can have a white region that leaks into the panel
        # background or an enlarged speech bubble. Inverse-warp the *target*
        # container back to the original source instead of trusting that leaky
        # source component. Final Chinese pixels are still sampled from the
        # untouched source page, so glyphs are never affine-warped.
        proxy = cv2.warpPerspective((tb.mask > 0).astype(np.uint8) * 255, H_inv, (source.shape[1], source.shape[0]), flags=cv2.INTER_NEAREST)
        if cv2.countNonZero(proxy) > 0:
            poly = mask_to_largest_polygon(proxy)
            if len(poly) >= 3:
                sb.mask = proxy; sb.polygon = poly
                x0, y0, x1, y1 = polygon_bbox(poly)
                margin = max(1, min(cfg.safe_margin_px, max(1, int(min(x1-x0, y1-y0) * 0.025))))
                sb.safe_mask = _safe_mask(proxy, (int((x0+x1)/2), int((y0+y1)/2)), margin)
        sb.meta.update({"paired_target_id": tid, "pair_iou": float(iou), "pair_coverage": float(coverage), "pair_score": float(score), "source_mask_mode": "inverse_target_container",
                       "target_driven_colored": bool(tb.meta.get("target_colored_recovery")),
                       "registered_ink_change": dict(tb.meta.get("registered_ink_change", {}) or {})})
        tb.meta.update({"paired_source_id": sid, "pair_iou": float(iou), "pair_coverage": float(coverage), "pair_score": float(score)})
        out_s.append(sb); out_t.append(tb); used_s.add(i); used_t.add(j)

    # Target-driven completion.  On a translated/scanned source page the
    # Chinese glyphs can split the white connected component so completely
    # that the source-side detector never produces a candidate.  The target
    # page is the authoritative geometry in same-layout pairs, so recover an
    # unpaired target container by inverse-warping its mask and sampling the
    # original source pixels from that area.  This is deliberately gated by
    # both pages' white/ink statistics and by duplicate overlap: it is a
    # completion path for missed speech containers, not a general paint-over.
    src_gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) if source.ndim == 3 else source
    src_sat = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)[..., 1] if source.ndim == 3 else np.zeros_like(src_gray)
    min_source_white = float(getattr(mask_config, "rigid_container_unseeded_min_white_ratio", 0.48))
    min_source_dark = float(getattr(mask_config, "rigid_container_unseeded_target_driven_min_dark_ratio", 0.004))
    for j, tb in enumerate(dst):
        if j in used_t:
            continue
        tm = (tb.mask > 0).astype(np.uint8) * 255
        ta = max(1, cv2.countNonZero(tm))
        duplicate = False
        for em in existing_masks + [b.mask for b in out_t if b.mask is not None]:
            if em is None or em.shape != (th, tw):
                continue
            inter = cv2.countNonZero(cv2.bitwise_and(tm, (em > 0).astype(np.uint8) * 255))
            if inter / max(1, min(ta, cv2.countNonZero(em))) >= max_existing:
                duplicate = True
                break
        if duplicate:
            continue
        if bool(getattr(mask_config, "rigid_container_unseeded_ink_change_gate_enabled", True)):
            ink_ev = _registered_ink_change_evidence(source_gray_warped, target_gray_page, tm, mask_config)
            tb.meta["registered_ink_change"] = ink_ev
            if not bool(ink_ev.get("passed", False)):
                continue
        proxy = cv2.warpPerspective(tm, H_inv, (source.shape[1], source.shape[0]), flags=cv2.INTER_NEAREST)
        if cv2.countNonZero(proxy) <= 0:
            continue
        inner = cv2.erode(proxy, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        if cv2.countNonZero(inner) < 24:
            continue
        vals = src_gray[inner > 0]
        white_ratio = float(np.mean(vals > 225))
        dark_ratio = float(np.mean(vals < 190))
        sat_ratio = float(np.mean(src_sat[inner > 0] > 90))
        # A white speech container may contain dense Chinese text, but it
        # should still be predominantly paper and have some source ink.
        if white_ratio < min_source_white or dark_ratio < min_source_dark or sat_ratio > 0.42:
            continue
        poly = mask_to_largest_polygon(proxy)
        if len(poly) < 3:
            continue
        sid = f"auto-src-target-{len(out_s):04d}"; tid = f"auto-dst-target-{len(out_t):04d}"
        x0, y0, x1, y1 = polygon_bbox(poly)
        margin = max(1, min(cfg.safe_margin_px, max(1, int(min(x1 - x0, y1 - y0) * 0.025))))
        sb = BubbleInstance(
            id=sid, polygon=poly, confidence=float(min(0.90, tb.confidence)),
            kind=tb.kind, block_ids=[], mask=proxy,
            safe_mask=_safe_mask(proxy, (int((x0 + x1) / 2), int((y0 + y1) / 2)), margin),
            meta={"backend": "unseeded_white_target_driven", "source_mask_mode": "inverse_target_container",
                  "target_driven_recovery": True, "target_driven_colored": bool(tb.meta.get("target_colored_recovery")),
                  "source_white_ratio": white_ratio,
                  "source_dark_ratio": dark_ratio, "source_saturation_ratio": sat_ratio,
                  "registered_ink_change": dict(tb.meta.get("registered_ink_change", {}) or {})},
        )
        tb.id = tid
        tb.meta.update({"paired_source_id": sid, "target_driven_recovery": True,
                        "source_mask_mode": "inverse_target_container"})
        out_s.append(sb); out_t.append(tb); used_t.add(j)
    return out_s, out_t


def detect_mangalens_bubbles(
    image: np.ndarray, blocks: list[TextBlock], config: BubbleConfig | None = None
) -> list[BubbleInstance]:
    """Run a local Ultralytics MangaLens-compatible instance segmentation model.

    Weights are never downloaded implicitly from this inference function. A
    user-triggered Model Center download is auto-discovered across restarts, or
    ``mangalens_model_path`` may point to another local YOLO checkpoint.
    """
    cfg = config or BubbleConfig()
    resolved = discovered_model_path("mangalens", cfg.mangalens_model_path)
    if resolved is None:
        raise ValueError("MangaLens 权重缺失；请到“识别与配准 → 模型下载与接入状态”主动下载 MangaLens，或选择本地 best.pt。")
    cfg.mangalens_model_path = str(resolved)
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError("MangaLens backend requires the optional 'ultralytics' package") from e

    model_key = _mangalens_model_cache_key(cfg.mangalens_model_path)
    model_path = model_key[0]
    with _YOLO_MODEL_LOCK:
        model = _YOLO_MODEL_CACHE.get(model_key)
        if model is None:
            model = YOLO(model_path)
            for old_key in [k for k in _YOLO_MODEL_CACHE if k[0] == model_path and k != model_key]:
                _YOLO_MODEL_CACHE.pop(old_key, None)
            _YOLO_MODEL_CACHE[model_key] = model
    device = select_device(cfg.device)
    # Ultralytics accepts device='mps' on Apple Silicon.  Keep inference
    # serialized on unified-memory GPUs to avoid batch-time memory spikes.
    try:
        with accelerator_lock():
            results = model.predict(
                source=image, conf=cfg.mangalens_confidence, imgsz=cfg.mangalens_imgsz,
                verbose=False, device=device,
            )
    except Exception:
        if device != "mps":
            raise
        # Some Ultralytics/custom ops can lag behind MPS support. Keep the batch
        # alive by falling back to CPU with the same local weights.
        with accelerator_lock():
            results = model.predict(
                source=image, conf=cfg.mangalens_confidence, imgsz=cfg.mangalens_imgsz,
                verbose=False, device="cpu",
            )
        device = "cpu-fallback"
    if not results:
        return []
    result = results[0]
    masks = getattr(result, "masks", None)
    if masks is None or getattr(masks, "xy", None) is None:
        return []
    confs = []
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "conf", None) is not None:
        try:
            confs = boxes.conf.detach().cpu().numpy().tolist()
        except Exception:
            confs = []

    h, w = image.shape[:2]
    bubbles: list[BubbleInstance] = []
    for idx, xy in enumerate(masks.xy):
        pts = np.asarray(xy, dtype=np.float32)
        if len(pts) < 3:
            continue
        polygon = [(float(x), float(y)) for x, y in pts]
        component = rasterize_polygon((h, w), polygon)
        if cv2.countNonZero(component) <= 0:
            continue
        x, y, bw, bh = polygon_bbox(polygon)
        margin = max(cfg.safe_margin_px, int(max(1.0, min(bw, bh)) * cfg.safe_margin_ratio))
        dist = cv2.distanceTransform((component > 0).astype(np.uint8), cv2.DIST_L2, 5)
        safe = (dist >= max(1, margin)).astype(np.uint8) * 255
        if not np.any(safe):
            safe = component.copy()
        confidence = float(confs[idx]) if idx < len(confs) else 0.8
        bubble = BubbleInstance(
            id=f"bubble-{idx:04d}",
            polygon=polygon,
            confidence=confidence,
            kind="speech",
            mask=component,
            safe_mask=safe,
            meta={"backend": "mangalens", "safe_margin": margin, "device": device, "model_cache": True},
        )
        bubbles.append(bubble)

    # Associate OCR blocks with the instance that actually contains their centre.
    # If several masks overlap, use the smallest containing instance.
    for block in blocks:
        cx, cy = block.centroid
        candidates = []
        for bubble in bubbles:
            if bubble.mask is None:
                continue
            ix = int(np.clip(round(cx), 0, w - 1)); iy = int(np.clip(round(cy), 0, h - 1))
            if bubble.mask[iy, ix] > 0:
                candidates.append((cv2.countNonZero(bubble.mask), bubble))
        if not candidates:
            continue
        _, chosen = min(candidates, key=lambda x: x[0])
        chosen.block_ids.append(block.id)
        block.bubble_id = chosen.id
        if block.kind == "unknown":
            block.kind = chosen.kind
    return bubbles

def load_bubble_sidecar(
    image: np.ndarray,
    image_path: str | Path,
    blocks: list[TextBlock],
    config: BubbleConfig | None = None,
) -> list[BubbleInstance]:
    cfg = config or BubbleConfig()
    p = Path(image_path)
    sidecar = p.with_suffix(cfg.sidecar_suffix) if cfg.sidecar_suffix.startswith(".") else p.parent / f"{p.stem}{cfg.sidecar_suffix}"
    if not sidecar.exists():
        raise FileNotFoundError(f"Bubble sidecar not found: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    rows = payload.get("bubbles", []) if isinstance(payload, dict) else payload
    h, w = image.shape[:2]
    out: list[BubbleInstance] = []
    for i, row in enumerate(rows):
        poly = row.get("polygon")
        mask = None
        mp = row.get("mask_path")
        if mp:
            path = Path(mp)
            if not path.is_absolute():
                path = sidecar.parent / path
            mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if mask is not None and mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            if mask is not None and not poly:
                poly = mask_to_largest_polygon(mask)
        if not poly:
            continue
        poly = [(float(x), float(y)) for x, y in poly]
        if mask is None:
            mask = rasterize_polygon(poly, (h, w))
        safe = None
        sp = row.get("safe_mask_path")
        if sp:
            path = Path(sp)
            if not path.is_absolute():
                path = sidecar.parent / path
            safe = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if safe is not None and safe.shape != (h, w):
                safe = cv2.resize(safe, (w, h), interpolation=cv2.INTER_NEAREST)
        if safe is None:
            x0, y0, x1, y1 = polygon_bbox(poly)
            margin = max(cfg.safe_margin_px, int(min(x1 - x0, y1 - y0) * cfg.safe_margin_ratio))
            # Use largest component after erosion as safe area; it naturally removes narrow tails.
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
            safe = cv2.erode((mask > 0).astype(np.uint8) * 255, k)
            if cv2.countNonZero(safe) == 0:
                safe = mask.copy()
        out.append(BubbleInstance(
            id=str(row.get("id") or f"bubble-{i:04d}"),
            polygon=poly,
            confidence=float(row.get("confidence", 1.0)),
            kind=str(row.get("kind", "speech")),
            block_ids=list(row.get("block_ids", [])),
            mask=(mask > 0).astype(np.uint8) * 255,
            safe_mask=(safe > 0).astype(np.uint8) * 255,
            meta=dict(row.get("meta", {})),
        ))
    assign_blocks_to_bubbles(blocks, out)
    return out


def assign_blocks_to_bubbles(blocks: list[TextBlock], bubbles: list[BubbleInstance]) -> None:
    lookup = {b.id: b for b in bubbles}
    for bubble in bubbles:
        bubble.block_ids = []
    for block in blocks:
        cx, cy = block.centroid
        candidates: list[tuple[float, BubbleInstance]] = []
        for bubble in bubbles:
            if bubble.mask is not None:
                y, x = round(cy), round(cx)
                if 0 <= y < bubble.mask.shape[0] and 0 <= x < bubble.mask.shape[1] and bubble.mask[y, x] > 0:
                    candidates.append((1.0, bubble))
            else:
                bx0, by0, bx1, by1 = bubble.bbox
                if bx0 <= cx <= bx1 and by0 <= cy <= by1:
                    candidates.append((0.5, bubble))
        if candidates:
            bubble = max(candidates, key=lambda x: x[0])[1]
            block.bubble_id = bubble.id
            if block.id not in bubble.block_ids:
                bubble.block_ids.append(block.id)
            if block.kind == "unknown":
                block.kind = bubble.kind
        elif block.bubble_id not in lookup:
            block.bubble_id = None
            if block.kind == "unknown":
                block.kind = "free_text"


def _join_fragments(fragments: list[str]) -> str:
    fragments = [f.strip() for f in fragments if f and f.strip()]
    if not fragments:
        return ""
    combined = "".join(fragments)
    cjk_ratio = len(_CJK_RE.findall(combined)) / max(1, len(combined))
    return "".join(fragments) if cjk_ratio >= 0.25 else " ".join(fragments)


def build_text_units(blocks: list[TextBlock], bubbles: list[BubbleInstance], prefix: str) -> list[TextUnit]:
    by_id = {b.id: b for b in blocks}
    units: list[TextUnit] = []
    used: set[str] = set()

    for bubble in bubbles:
        members = [by_id[i] for i in bubble.block_ids if i in by_id]
        if not members:
            continue
        members.sort(key=lambda x: x.reading_order)
        used.update(m.id for m in members)
        text = _join_fragments([m.text for m in members])
        conf = float(np.mean([m.confidence for m in members]))
        units.append(
            TextUnit(
                id=f"{prefix}-unit-{len(units):04d}",
                polygon=list(bubble.polygon),
                block_ids=[m.id for m in members],
                text=text,
                confidence=conf,
                kind=bubble.kind,
                reading_order=min(m.reading_order for m in members),
                bubble_id=bubble.id,
                meta={"geometry": "bubble"},
            )
        )

    for block in blocks:
        if block.id in used:
            continue
        units.append(
            TextUnit(
                id=f"{prefix}-unit-{len(units):04d}",
                polygon=list(block.polygon),
                block_ids=[block.id],
                text=block.text,
                confidence=block.confidence,
                kind=block.kind if block.kind != "unknown" else "free_text",
                reading_order=block.reading_order,
                bubble_id=None,
                meta={"geometry": "text_block"},
            )
        )

    units.sort(key=lambda u: u.reading_order)
    for i, unit in enumerate(units):
        unit.reading_order = i
    return units


def bubble_by_id(bubbles: list[BubbleInstance]) -> dict[str, BubbleInstance]:
    return {b.id: b for b in bubbles}
