from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .config import MaskReplaceConfig
from .geometry import mask_to_largest_polygon, transform_to_homography
from .models import BubbleInstance, RegistrationResult


@dataclass(slots=True)
class DiffBubbleRecord:
    source_id: str
    target_id: str
    change_density: float
    mask_iou: float
    confidence: float
    bbox_target: tuple[int, int, int, int]
    method: str = "raw_diff"
    source_ink_density: float = 0.0
    target_ink_density: float = 0.0
    # v0.8 compatibility diagnostics. New routes still populate these so older
    # tooling/tests can inspect a uniform record shape.
    region_kind: str = "bubble"
    changed_pixels: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(slots=True)
class PairedDiffResult:
    source_bubbles: list[BubbleInstance]
    target_bubbles: list[BubbleInstance]
    change_mask: np.ndarray
    records: list[DiffBubbleRecord]
    threshold: float
    noise_floor: float
    method: str = "raw_diff"
    safe_to_skip_ocr: bool = True
    diagnostics: dict = field(default_factory=dict)
    aligned_source: np.ndarray | None = None
    alignment_diagnostics: dict = field(default_factory=dict)
    # Optional non-overlapping v0.8 structural regions that complement the main
    # photographed-page detector. They are applied target-driven so they can
    # recover open burst bubbles/free text without weakening the safe closed-
    # container path or re-processing already matched bubbles.
    supplemental: PairedDiffResult | None = None


def _warp_source(source: np.ndarray, registration: RegistrationResult, target_shape: tuple[int, int]) -> np.ndarray:
    h, w = target_shape
    H = transform_to_homography(registration.matrix)
    return cv2.warpPerspective(
        source, H, (w, h), flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )


def _transform_xy(x: float, y: float, H: np.ndarray) -> tuple[float, float]:
    p = H @ np.array([x, y, 1.0], dtype=np.float64)
    if abs(p[2]) < 1e-9:
        return x, y
    return float(p[0] / p[2]), float(p[1] / p[2])


def _nearest_seed(binary: np.ndarray, x: float, y: float, radius: int) -> tuple[int, int] | None:
    h, w = binary.shape
    ix = int(np.clip(round(x), 0, w - 1)); iy = int(np.clip(round(y), 0, h - 1))
    if binary[iy, ix] > 0:
        return ix, iy
    # Expanding rings avoid a full distance transform for the small number of
    # candidate balloons on a page.
    for r in range(1, max(1, radius) + 1):
        x0, x1 = max(0, ix - r), min(w - 1, ix + r)
        y0, y1 = max(0, iy - r), min(h - 1, iy + r)
        candidates: list[tuple[int, int]] = []
        for xx in range(x0, x1 + 1):
            if binary[y0, xx]: candidates.append((xx, y0))
            if y1 != y0 and binary[y1, xx]: candidates.append((xx, y1))
        for yy in range(y0 + 1, y1):
            if binary[yy, x0]: candidates.append((x0, yy))
            if x1 != x0 and binary[yy, x1]: candidates.append((x1, yy))
        if candidates:
            return min(candidates, key=lambda p: (p[0] - x) ** 2 + (p[1] - y) ** 2)
    return None


def _component_from_seed(image: np.ndarray, x: float, y: float, threshold: int, search_radius: int) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    white = (gray >= threshold).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    seed = _nearest_seed(white, x, y, search_radius)
    if seed is None:
        return None
    n, labels, _, _ = cv2.connectedComponentsWithStats((white > 0).astype(np.uint8), 8)
    label = int(labels[seed[1], seed[0]])
    if label <= 0 or label >= n:
        return None
    raw = (labels == label).astype(np.uint8) * 255
    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    component = np.zeros_like(raw)
    cv2.drawContours(component, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return component


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _is_monochrome_to_color_pair(source: np.ndarray, target: np.ndarray) -> bool:
    """Cheap cross-rendition hint used to avoid an expensive structural supplement.

    The v0.8 structural supplement was designed for photographed colour/gray
    editions with similar local rendering.  On a black/white translated scan ->
    coloured Japanese master it can spend most of a page run exploring barrier
    components while adding no regions.  The v0.8.7 target-driven photo route is
    already the correct detector for that case.
    """
    def sat90(img: np.ndarray) -> float:
        if img.ndim != 3 or img.shape[2] < 3:
            return 0.0
        small = img
        h, w = img.shape[:2]
        if max(h, w) > 640:
            scale = 640.0 / max(h, w)
            small = cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        return float(np.percentile(hsv[..., 1], 90.0))
    return sat90(source) < 24.0 and sat90(target) >= 24.0


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa, bb = a > 0, b > 0
    inter = int(np.count_nonzero(aa & bb)); union = int(np.count_nonzero(aa | bb))
    return inter / union if union else 0.0


def _classify_kind(mask: np.ndarray) -> str:
    box = _bbox(mask)
    if box is None:
        return "speech"
    x0, y0, x1, y1 = box
    area = max(1, cv2.countNonZero(mask)); rect_area = max(1, (x1 - x0) * (y1 - y0))
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "speech"
    c = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(c, 0.02 * max(1, (x1 - x0) + (y1 - y0)), True)
    rect_fill = area / rect_area
    return "narration" if len(approx) <= 6 and rect_fill > 0.78 else "speech"


def _safe_mask(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    if box is None:
        return mask.copy()
    x0, y0, x1, y1 = box
    margin = max(3, int(0.025 * min(x1 - x0, y1 - y0)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
    safe = cv2.erode(mask, k)
    return safe if cv2.countNonZero(safe) else mask.copy()


def _warp_mask(mask: np.ndarray, H: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    return cv2.warpPerspective(mask, H, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _change_map(source_warped: np.ndarray, target: np.ndarray, cfg: MaskReplaceConfig) -> tuple[np.ndarray, float, float, np.ndarray]:
    sg = cv2.cvtColor(source_warped, cv2.COLOR_BGR2GRAY)
    tg = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    sb = cv2.GaussianBlur(sg, (3, 3), 0.55).astype(np.float32)
    tb = cv2.GaussianBlur(tg, (3, 3), 0.55).astype(np.float32)
    diff = np.abs(sb - tb)
    noise_floor = float(np.percentile(diff, 95.0))
    threshold = float(max(cfg.paired_diff_pixel_threshold, noise_floor + cfg.paired_diff_noise_margin))
    raw = (diff >= threshold).astype(np.uint8) * 255
    if cfg.paired_diff_close_px > 1:
        k = max(3, cfg.paired_diff_close_px | 1)
        raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    if cfg.paired_diff_dilate_px > 0:
        r = max(1, cfg.paired_diff_dilate_px)
        regions = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))
    else:
        regions = raw.copy()
    return raw, threshold, noise_floor, regions


def _build_result(
    rows: list[tuple[np.ndarray, np.ndarray, float, float, tuple[int, int, int, int], float, float]],
    registration: RegistrationResult,
    change_mask: np.ndarray,
    threshold: float,
    noise_floor: float,
    method: str,
    safe_to_skip_ocr: bool,
    diagnostics: dict | None = None,
) -> PairedDiffResult:
    source_bubbles: list[BubbleInstance] = []
    target_bubbles: list[BubbleInstance] = []
    records: list[DiffBubbleRecord] = []
    prefix = "diff" if method == "raw_diff" else "photo"
    for idx, (sm, tm, density, mask_iou, box, sdens, tdens) in enumerate(sorted(rows, key=lambda r: (r[4][1], r[4][0]))):
        sp = mask_to_largest_polygon(sm); tp = mask_to_largest_polygon(tm)
        if len(sp) < 3 or len(tp) < 3:
            continue
        kind = _classify_kind(tm)
        conf = float(np.clip(
            0.34 * registration.confidence + 0.38 * mask_iou
            + 0.18 * min(1.0, density / 0.35)
            + 0.10 * min(1.0, max(sdens, tdens) / 0.08),
            0.0, 0.995,
        ))
        sid, tid = f"{prefix}-src-{idx:03d}", f"{prefix}-dst-{idx:03d}"
        common_meta = {
            "backend": "paired_diff",
            "paired_diff_method": method,
            "mask_is_interior": True,
            "change_density": density,
            "paired_mask_iou": mask_iou,
            "source_ink_density": sdens,
            "target_ink_density": tdens,
            "diff_threshold": threshold,
            "noise_floor": noise_floor,
        }
        source_meta = dict(common_meta); source_meta["paired_target_id"] = tid
        target_meta = dict(common_meta); target_meta["paired_source_id"] = sid
        # A synthetic block id marks that this bubble contains translated text.
        # Actual OCR is still run for photo fallback because these ids are not OCR evidence.
        source_bubbles.append(BubbleInstance(sid, sp, conf, kind, [f"paired-{prefix}-{idx:03d}"], sm, _safe_mask(sm), source_meta))
        target_bubbles.append(BubbleInstance(tid, tp, conf, kind, [], tm, _safe_mask(tm), target_meta))
        changed_pixels = int(round(density * max(1, cv2.countNonZero(tm))))
        records.append(DiffBubbleRecord(
            sid, tid, density, mask_iou, conf, box, method, sdens, tdens,
            region_kind="bubble", changed_pixels=changed_pixels,
        ))
    return PairedDiffResult(
        source_bubbles, target_bubbles, change_mask, records, threshold, noise_floor,
        method=method, safe_to_skip_ocr=safe_to_skip_ocr, diagnostics=diagnostics or {},
    )


def _extract_raw(
    source: np.ndarray, target: np.ndarray, registration: RegistrationResult, cfg: MaskReplaceConfig,
) -> PairedDiffResult:
    target_shape = target.shape[:2]
    source_warped = _warp_source(source, registration, target_shape)
    raw_change, threshold, noise_floor, regions = _change_map(source_warped, target, cfg)
    n, labels, stats, cents = cv2.connectedComponentsWithStats((regions > 0).astype(np.uint8), 8)
    page_area = target_shape[0] * target_shape[1]
    H = transform_to_homography(registration.matrix); Hinv = np.linalg.inv(H)
    candidates: list[tuple[np.ndarray, np.ndarray, float, float, tuple[int, int, int, int], float, float]] = []
    for label in range(1, n):
        _, _, _, _, area = [int(v) for v in stats[label]]
        if area < cfg.paired_diff_min_region_area or area / max(1, page_area) > cfg.paired_diff_max_region_ratio:
            continue
        cx, cy = map(float, cents[label])
        tm = _component_from_seed(target, cx, cy, cfg.paired_diff_white_threshold, cfg.paired_diff_search_radius)
        if tm is None or cv2.countNonZero(tm) < cfg.paired_diff_min_component_area:
            continue
        sx, sy = _transform_xy(cx, cy, Hinv)
        sm = _component_from_seed(source, sx, sy, cfg.paired_diff_white_threshold, cfg.paired_diff_search_radius)
        if sm is None or cv2.countNonZero(sm) < cfg.paired_diff_min_component_area:
            continue
        warped_sm = _warp_mask(sm, H, target_shape)
        mask_iou = _iou(warped_sm, tm)
        if mask_iou < cfg.paired_diff_min_mask_iou:
            continue
        t_area = max(1, cv2.countNonZero(tm))
        change_density = float(np.count_nonzero((raw_change > 0) & (tm > 0)) / t_area)
        if change_density < cfg.paired_diff_min_change_density:
            continue
        box = _bbox(tm)
        if box is not None:
            candidates.append((sm, tm, change_density, mask_iou, box, 0.0, 0.0))
    candidates.sort(key=lambda r: (r[2], r[3]), reverse=True)
    kept = []
    for row in candidates:
        if any(_iou(row[1], k[1]) > 0.82 for k in kept):
            continue
        kept.append(row)
    # Raw-diff is safe to skip OCR only when photometric residuals are genuinely low.
    safe = bool(noise_floor < cfg.photo_pair_noise_floor_trigger and registration.confidence >= cfg.paired_diff_min_registration_confidence)
    result = _build_result(kept, registration, raw_change, threshold, noise_floor, "raw_diff", safe,
                           {"candidate_count": len(kept), "photometric_noise_floor": noise_floor})
    result.aligned_source = source_warped
    result.alignment_diagnostics = {"method": "global-only", "flow_used": False}
    return result


def _target_container_candidates(target: np.ndarray, cfg: MaskReplaceConfig) -> list[dict]:
    """Find closed white speech/narration interiors without OCR.

    The clean Japanese master is used as geometry truth. Dark line art becomes a
    barrier; connected bright regions enclosed by that barrier are candidate
    balloon/text-box interiors. This is intentionally conservative: missed regions
    are handled by OCR/hybrid fallback rather than copying arbitrary artwork.
    """
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    dark_thr = int(cfg.photo_pair_target_dark_threshold)
    dark = (gray < dark_thr).astype(np.uint8) * 255
    r = max(0, int(cfg.photo_pair_border_dilate_px))
    if r:
        dark = cv2.dilate(dark, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1)))
    free = (dark == 0).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(free, 8)
    h, w = gray.shape; page_area = h * w
    out: list[dict] = []
    for i in range(1, n):
        x, y, bw, bh, area = map(int, stats[i])
        area_ratio = area / max(1, page_area)
        fill = area / max(1, bw * bh)
        if not (cfg.photo_pair_min_region_ratio <= area_ratio <= cfg.photo_pair_max_region_ratio):
            continue
        if min(bw, bh) < cfg.photo_pair_min_side_px or fill < cfg.photo_pair_min_fill_ratio:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        raw = (labels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        tm = np.zeros_like(raw)
        cv2.drawContours(tm, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        ring = (cv2.dilate(tm, np.ones((5, 5), np.uint8)) > 0) & (tm == 0)
        ring_dark = float(np.mean(gray[ring] < dark_thr)) if np.any(ring) else 0.0
        inner = cv2.erode(tm, np.ones((5, 5), np.uint8)) > 0
        target_dark_density = float(np.mean(gray[inner] < 175)) if np.any(inner) else 0.0
        # A dark Japanese glyph can split a clean balloon's white interior into
        # several connected components. In that case ``inner`` may contain no
        # dark pixels even though the candidate visibly contains text. Admit only
        # the narrow, speech-like case: a strong boundary, mostly white box, and
        # enough dark pixels in the original bounding box to prove lettering.
        box_gray = gray[y:y + bh, x:x + bw]
        bbox_dark_density = float(np.mean(box_gray < 175)) if box_gray.size else 0.0
        split_text_container = (
            target_dark_density < cfg.photo_pair_min_target_dark_density
            and ring_dark >= max(0.50, float(cfg.photo_pair_min_boundary_dark))
            and fill >= 0.62
            and bbox_dark_density >= 0.012
            and area_ratio <= float(getattr(cfg, "photo_pair_large_textlike_max_area_ratio", 0.055))
        )
        if ring_dark < cfg.photo_pair_min_boundary_dark or (
            target_dark_density < cfg.photo_pair_min_target_dark_density and not split_text_container
        ):
            continue
        ar = max(bw / max(1, bh), bh / max(1, bw))
        if ar > cfg.photo_pair_max_aspect_ratio:
            continue
        out.append({
            "mask": tm, "bbox": (x, y, x + bw, y + bh), "area_ratio": area_ratio,
            "fill": fill, "ring_dark": ring_dark, "target_dark_density": target_dark_density,
            "centroid": (float(cents[i][0]), float(cents[i][1])),
        })
    # Some scan/color-master balloons have a tiny outline break or a tail that
    # connects their white interior to the page background. The dark-barrier
    # connected-components pass above then misses the whole balloon. Recover
    # those as a second, tightly bounded white-component route.
    white = (gray > 235).astype(np.uint8)
    wn, wlabels, wstats, _ = cv2.connectedComponentsWithStats(white, 8)
    for i in range(1, wn):
        x, y, bw, bh, area = map(int, wstats[i])
        area_ratio = area / max(1, page_area)
        fill = area / max(1, bw * bh)
        if not (cfg.photo_pair_min_region_ratio <= area_ratio <= float(getattr(cfg, "photo_pair_large_textlike_max_area_ratio", 0.055))):
            continue
        if min(bw, bh) < cfg.photo_pair_min_side_px or fill < 0.62:
            continue
        if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
            continue
        raw = (wlabels == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        tm = np.zeros_like(raw)
        cv2.drawContours(tm, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
        ring = (cv2.dilate(tm, np.ones((5, 5), np.uint8)) > 0) & (tm == 0)
        ring_dark = float(np.mean(gray[ring] < dark_thr)) if np.any(ring) else 0.0
        box_gray = gray[y:y + bh, x:x + bw]
        bbox_dark_density = float(np.mean(box_gray < 175)) if box_gray.size else 0.0
        if ring_dark < max(0.50, float(cfg.photo_pair_min_boundary_dark)) or bbox_dark_density < 0.012:
            continue
        if any(_iou(tm, prev["mask"]) > 0.80 for prev in out):
            continue
        out.append({
            "mask": tm, "bbox": (x, y, x + bw, y + bh), "area_ratio": area_ratio,
            "fill": fill, "ring_dark": ring_dark, "target_dark_density": 0.0,
            "centroid": (float(x + bw / 2), float(y + bh / 2)),
        })
    return out


def _candidate_stats(target: np.ndarray, mask: np.ndarray, dark_thr: int, cfg: MaskReplaceConfig) -> dict | None:
    """Return the same geometry/text evidence used by target-container routing."""
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    box = _bbox(mask)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    area = int(cv2.countNonZero(mask))
    if area <= 0:
        return None
    fill = area / max(1, (x1 - x0) * (y1 - y0))
    ring = (cv2.dilate(mask, np.ones((5, 5), np.uint8)) > 0) & (mask == 0)
    ring_dark = float(np.mean(gray[ring] < dark_thr)) if np.any(ring) else 0.0
    inner = cv2.erode(mask, np.ones((5, 5), np.uint8)) > 0
    inner_dark = float(np.mean(gray[inner] < 175)) if np.any(inner) else 0.0
    white_thr = int(getattr(cfg, "photo_pair_cross_rendition_white_threshold", 235))
    white_ratio = float(np.mean(gray[mask > 0] >= white_thr)) if np.any(mask > 0) else 0.0
    return {
        "mask": mask,
        "bbox": box,
        "area": area,
        "fill": float(fill),
        "ring_dark": ring_dark,
        "target_dark_density": inner_dark,
        "white_ratio": white_ratio,
        "centroid": (
            float(np.mean(np.where(mask > 0)[1])),
            float(np.mean(np.where(mask > 0)[0])),
        ),
    }


def _refine_cross_rendition_white_candidate(
    target: np.ndarray,
    cand: dict,
    cfg: MaskReplaceConfig,
) -> dict | None:
    """Tighten a leaky pale-region candidate to a real white text container.

    On monochrome->colour pages, the low dark-barrier threshold can connect a
    speech-balloon tail to pale train bodywork or signage.  A true balloon still
    contains a large near-white connected core.  Recover that core, fill only its
    outer contour, and re-evaluate speech-like fill/boundary/internal text ink.

    The filter is intentionally *not* used outside the cross-rendition route.
    """
    if not bool(getattr(cfg, "photo_pair_cross_rendition_white_guard_enabled", True)):
        return cand
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    tm = cand.get("mask")
    if tm is None or cv2.countNonZero(tm) == 0:
        return None
    dark_thr = int(cfg.photo_pair_target_dark_threshold)
    original = _candidate_stats(target, tm, dark_thr, cfg)
    if original is None:
        return None

    white_thr = int(getattr(cfg, "photo_pair_cross_rendition_white_threshold", 235))
    # If the original mask is already clean/white, keep it; otherwise search for
    # a high-white subcomponent that represents the actual balloon interior.
    chosen = original
    if original["white_ratio"] < float(getattr(cfg, "photo_pair_cross_rendition_min_white_ratio", 0.82)):
        binary = ((gray >= white_thr) & (tm > 0)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        original_area = max(1, int(original["area"]))
        min_seed = max(120, int(round(
            original_area * float(getattr(cfg, "photo_pair_cross_rendition_refine_min_area_fraction", 0.12))
        )))
        min_keep = max(150, int(round(
            original_area * float(getattr(cfg, "photo_pair_cross_rendition_refine_min_keep_fraction", 0.15))
        )))
        best: tuple[float, dict] | None = None
        for i in range(1, n):
            if int(stats[i, cv2.CC_STAT_AREA]) < min_seed:
                continue
            raw = (labels == i).astype(np.uint8) * 255
            contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            filled = np.zeros_like(raw)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            if cv2.countNonZero(filled) < min_keep:
                continue
            st = _candidate_stats(target, filled, dark_thr, cfg)
            if st is None:
                continue
            # Favour a dark enclosed outline + internal text + high-white paper.
            score = (
                2.2 * st["ring_dark"]
                + 1.4 * min(0.20, st["target_dark_density"]) / 0.20
                + st["white_ratio"]
                + 0.35 * min(1.0, st["fill"] / 0.70)
            )
            if best is None or score > best[0]:
                best = (score, st)
        if best is not None:
            chosen = best[1]

    if chosen["white_ratio"] < float(getattr(cfg, "photo_pair_cross_rendition_min_white_ratio", 0.82)):
        return None
    if chosen["fill"] < float(getattr(cfg, "photo_pair_cross_rendition_min_fill_ratio", 0.56)):
        return None
    if chosen["target_dark_density"] < float(getattr(cfg, "photo_pair_cross_rendition_min_inner_dark_density", 0.012)):
        return None
    if chosen["ring_dark"] < float(getattr(cfg, "photo_pair_cross_rendition_min_ring_dark", 0.45)):
        return None

    page_area = max(1, target.shape[0] * target.shape[1])
    result = dict(cand)
    result.update({
        "mask": chosen["mask"],
        "bbox": chosen["bbox"],
        "area_ratio": chosen["area"] / page_area,
        "fill": chosen["fill"],
        "ring_dark": chosen["ring_dark"],
        "target_dark_density": chosen["target_dark_density"],
        "centroid": chosen["centroid"],
        "cross_rendition_white_ratio": chosen["white_ratio"],
        "cross_rendition_refined": bool(chosen["mask"] is not tm),
    })
    return result


def _ink_map(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    bg = cv2.GaussianBlur(gray, (0, 0), 7.0)
    detail = cv2.subtract(bg, gray)
    work = (mask > 0).astype(np.uint8) * 255
    box = _bbox(work)
    if box is not None:
        x0, y0, x1, y1 = box
        margin = max(2, int(0.015 * min(x1 - x0, y1 - y0)))
        work = cv2.erode(work, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
    vals = detail[work > 0]
    if vals.size == 0:
        return np.zeros_like(gray, np.uint8), 0.0
    threshold = float(max(10, min(45, np.percentile(vals, 75))))
    ink = ((detail >= threshold) & (work > 0)).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats((ink > 0).astype(np.uint8), 8)
    clean = np.zeros_like(ink)
    mask_area = max(1, cv2.countNonZero(work))
    for i in range(1, n):
        _, _, bw, bh, area = map(int, stats[i])
        if area < 2 or area > 0.04 * mask_area:
            continue
        if max(bw, bh) > 0.75 * max(gray.shape):
            continue
        clean[labels == i] = 255
    return clean, cv2.countNonZero(clean) / mask_area


def _ink_change_score(source_warped: np.ndarray, target: np.ndarray, source_mask_warped: np.ndarray, target_mask: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    common = cv2.bitwise_and(source_mask_warped, target_mask)
    source_ink, sdens = _ink_map(source_warped, common)
    target_ink, tdens = _ink_map(target, common)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    source_d = cv2.dilate(source_ink, k); target_d = cv2.dilate(target_ink, k)
    sn, tn = cv2.countNonZero(source_ink), cv2.countNonZero(target_ink)
    if sn + tn == 0:
        return 0.0, sdens, tdens, np.zeros_like(common)
    matched = (
        np.count_nonzero((source_ink > 0) & (target_d > 0))
        + np.count_nonzero((target_ink > 0) & (source_d > 0))
    ) / max(1, sn + tn)
    residual = (((source_ink > 0) & (target_d == 0)) | ((target_ink > 0) & (source_d == 0))).astype(np.uint8) * 255
    return float(np.clip(1.0 - matched, 0.0, 1.0)), float(sdens), float(tdens), residual


def _photo_pair_fallback(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig,
    threshold: float,
    noise_floor: float,
) -> PairedDiffResult:
    target_shape = target.shape[:2]
    H = transform_to_homography(registration.matrix); Hinv = np.linalg.inv(H)
    source_warped = _warp_source(source, registration, target_shape)
    target_candidates = _target_container_candidates(target, cfg)
    cross_rendition_guard = bool(
        _is_monochrome_to_color_pair(source, target)
        and getattr(cfg, "photo_pair_cross_rendition_white_guard_enabled", True)
    )
    rejected_cross_rendition = 0
    refined_cross_rendition = 0
    if cross_rendition_guard:
        guarded: list[dict] = []
        for cand in target_candidates:
            refined = _refine_cross_rendition_white_candidate(target, cand, cfg)
            if refined is None:
                rejected_cross_rendition += 1
                continue
            if bool(refined.get("cross_rendition_refined", False)):
                refined_cross_rendition += 1
            guarded.append(refined)
        target_candidates = guarded
    # Convert target search radius to approximate source pixels. Photographs are
    # commonly 2-4x larger than the clean master.
    inv_scale = float(np.sqrt(max(1e-9, abs(np.linalg.det(Hinv[:2, :2])))))
    source_radius = int(np.clip(cfg.photo_pair_source_search_radius * inv_scale, 70, 480))
    rows: list[tuple[np.ndarray, np.ndarray, float, float, tuple[int, int, int, int], float, float]] = []
    residual_union = np.zeros(target_shape, np.uint8)
    thresholds = list(cfg.photo_pair_source_thresholds)
    for cand in target_candidates[: cfg.photo_pair_max_candidates]:
        tm = cand["mask"]; cx, cy = cand["centroid"]
        sx, sy = _transform_xy(cx, cy, Hinv)
        best = None
        for source_threshold in thresholds:
            sm = _component_from_seed(source, sx, sy, int(source_threshold), source_radius)
            if sm is None or cv2.countNonZero(sm) < cfg.paired_diff_min_component_area:
                continue
            warped_sm = _warp_mask(sm, H, target_shape)
            miou = _iou(warped_sm, tm)
            if best is None or miou > best[0]:
                best = (miou, sm, warped_sm)
        if best is None:
            continue
        miou, sm, warped_sm = best
        if miou < cfg.photo_pair_min_source_target_iou:
            continue
        change, sdens, tdens, residual = _ink_change_score(source_warped, target, warped_sm, tm)
        min_dens = min(sdens, tdens)
        small_high_change = (
            min_dens >= cfg.photo_pair_relaxed_min_ink_density
            and change >= cfg.photo_pair_relaxed_min_ink_change
            and cand["ring_dark"] >= cfg.photo_pair_relaxed_min_boundary_dark
            and cand["area_ratio"] <= cfg.photo_pair_relaxed_max_region_ratio
        )
        if min_dens < cfg.photo_pair_min_ink_density and not small_high_change:
            continue
        if change < cfg.photo_pair_min_ink_change:
            continue
        # Broad low-text-density white regions are usually two touching balloons
        # or a panel/background component. Directly transferring them can copy
        # unrelated artwork; leave them to OCR/hybrid fallback instead.
        if (cand["area_ratio"] > cfg.photo_pair_large_region_ratio
                and min_dens < cfg.photo_pair_large_min_ink_density):
            continue
        box = cand["bbox"]
        rows.append((sm, tm, change, miou, box, sdens, tdens))
        residual_union = np.maximum(residual_union, residual)

    # v0.8.7: cross-rendition pages (for example a monochrome Chinese scan and
    # a coloured Japanese master) can have the same text container with a visibly
    # different outline/scale. Requiring the source connected component to match
    # the target outline then rejects valid dialogue. Admit a tightly gated second
    # class of target-driven candidates by measuring registered ink directly inside
    # the clean target container. Their source mask is simply the target container
    # projected back into source coordinates; transfer later remains target-driven,
    # so this mask is identity metadata rather than a writable geometry claim.
    relaxed_added = 0
    if getattr(cfg, "photo_pair_relaxed_target_candidates", True):
        strict_target_masks = [row[1] for row in rows]
        sh, sw = source.shape[:2]
        for cand in target_candidates[: cfg.photo_pair_max_candidates]:
            tm = cand["mask"]
            if any(_iou(tm, existing) > 0.78 for existing in strict_target_masks):
                continue
            if cand["ring_dark"] < float(getattr(cfg, "photo_pair_relaxed_target_min_boundary_dark", 0.45)):
                continue
            if cand["target_dark_density"] > float(getattr(cfg, "photo_pair_relaxed_target_max_dark_density", 0.22)):
                continue
            change, sdens, tdens, residual = _ink_change_score(source_warped, target, tm, tm)
            # Large speech bubbles can have low raw pixel-difference when the
            # Japanese and Chinese vertical strokes occupy similar positions.
            # Keep a narrow text-like escape hatch for those cases: it requires
            # strong local ink on both editions, a clean dark boundary, low
            # target background density, and a non-zero local change. This does
            # not admit arbitrary large panels or artwork regions.
            large_textlike = (
                cand["area_ratio"] <= float(getattr(cfg, "photo_pair_large_textlike_max_area_ratio", 0.055))
                and cand["ring_dark"] >= float(getattr(cfg, "photo_pair_large_textlike_min_boundary_dark", 0.55))
                and cand["target_dark_density"] <= float(getattr(cfg, "photo_pair_large_textlike_max_dark_density", 0.12))
                and sdens >= float(getattr(cfg, "photo_pair_large_textlike_min_source_ink", 0.018))
                and tdens >= float(getattr(cfg, "photo_pair_large_textlike_min_target_ink", 0.012))
                and change >= float(getattr(cfg, "photo_pair_large_textlike_min_change", 0.05))
            )
            if sdens < float(getattr(cfg, "photo_pair_relaxed_target_min_source_ink", 0.018)):
                continue
            if tdens < float(getattr(cfg, "photo_pair_relaxed_target_min_target_ink", 0.008)):
                continue
            if change < float(getattr(cfg, "photo_pair_relaxed_target_min_change", 0.35)) and not large_textlike:
                continue
            if cand["area_ratio"] > float(getattr(cfg, "photo_pair_relaxed_target_max_area_ratio", 0.035)) and not large_textlike:
                continue
            sm = cv2.warpPerspective(
                tm, Hinv, (sw, sh), flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            if cv2.countNonZero(sm) < max(40, int(cfg.paired_diff_min_component_area * 0.35)):
                continue
            # Keep confidence intentionally below a perfect geometry match. The
            # clean target box is trusted for placement, but this route should be
            # surfaced as reviewable when OCR evidence is unavailable.
            synthetic_iou = float(min(
                getattr(cfg, "photo_pair_relaxed_target_confidence_cap", 0.84),
                0.62 + 0.18 * min(1.0, change) + 0.08 * min(1.0, sdens / 0.08),
            ))
            rows.append((sm, tm, change, synthetic_iou, cand["bbox"], sdens, tdens))
            strict_target_masks.append(tm)
            residual_union = np.maximum(residual_union, residual)
            relaxed_added += 1

    # Suppress broad parent regions when they contain multiple tighter candidates.
    def contains(parent, child) -> bool:
        px0, py0, px1, py1 = parent[4]; cx0, cy0, cx1, cy1 = child[4]
        ccx, ccy = (cx0 + cx1) / 2, (cy0 + cy1) / 2
        return px0 <= ccx <= px1 and py0 <= ccy <= py1 and cv2.countNonZero(parent[1]) > 1.7 * cv2.countNonZero(child[1])

    filtered = []
    for row in rows:
        children = [other for other in rows if other is not row and contains(row, other)]
        if len(children) >= 2:
            continue
        filtered.append(row)
    filtered.sort(key=lambda r: (r[2], r[3]), reverse=True)
    kept = []
    for row in filtered:
        if any(_iou(row[1], other[1]) > 0.80 for other in kept):
            continue
        kept.append(row)
    diagnostics = {
        "target_container_candidates": len(target_candidates),
        "accepted_photo_pairs": len(kept),
        "relaxed_target_candidates_added": relaxed_added,
        "cross_rendition_white_guard": cross_rendition_guard,
        "cross_rendition_rejected_candidates": rejected_cross_rendition,
        "cross_rendition_refined_candidates": refined_cross_rendition,
        "source_search_radius": source_radius,
        "registration_confidence": float(registration.confidence),
        "photometric_noise_floor": float(noise_floor),
        "incomplete_requires_ocr": True,
    }
    result = _build_result(kept, registration, residual_union, threshold, noise_floor,
                           "photo_pair", False, diagnostics)
    result.aligned_source = source_warped
    result.alignment_diagnostics = {"method": "global-photo-pair", "flow_used": False}
    return result



def _structural_v08_fallback(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig,
) -> PairedDiffResult | None:
    """Run the v0.8 structural detector as a non-destructive fallback.

    It lives in a separate compatibility module to avoid symbol collisions with
    v0.8.3's photo_pair implementation. Results are adapted to the v0.8.1 data
    model; OCR skipping remains disabled because this route may include free text.
    """
    if not getattr(cfg, "paired_diff_structural_fallback_enabled", True):
        return None
    if registration.confidence < float(getattr(cfg, "paired_diff_structural_min_registration_confidence", 0.62)):
        return None
    try:
        from .paired_diff_v08 import extract_paired_diff_bubbles as extract_v08
        legacy = extract_v08(source, target, registration, cfg)
    except Exception:
        return None
    if not legacy.source_bubbles or not legacy.target_bubbles:
        return None

    # Structural detection is permissive around SFX and panel edges. Require
    # actual source ink for open/complex candidates before exposing them to the
    # writer, otherwise line-art/halftone islands become rejected transfers.
    src_aligned = getattr(legacy, "aligned_source", None)
    if src_aligned is not None and src_aligned.shape[:2] == target.shape[:2]:
        src_gray = cv2.cvtColor(src_aligned, cv2.COLOR_BGR2GRAY)
        src_ink = cv2.adaptiveThreshold(
            src_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 31, 10,
        )
        keep = []
        for i, rec in enumerate(legacy.records):
            if str(getattr(rec, "region_kind", "bubble")) == "bubble":
                keep.append(i)
                continue
            # The legacy source mask is in source-image coordinates; the ink
            # map above is already aligned to the target page. Use the paired
            # target mask for this evidence test.
            tb = legacy.target_bubbles[i] if i < len(legacy.target_bubbles) else None
            if tb is not None and tb.mask is not None and tb.mask.shape == src_ink.shape:
                ink_px = int(cv2.countNonZero(cv2.bitwise_and(src_ink, tb.mask)))
                target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
                target_ink = cv2.adaptiveThreshold(
                    target_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 31, 10,
                )
                target_px = int(cv2.countNonZero(cv2.bitwise_and(target_ink, tb.mask)))
                if (ink_px >= int(getattr(cfg, "paired_diff_complex_min_source_ink_pixels", 16))
                        and target_px >= int(getattr(cfg, "paired_diff_complex_min_target_ink_pixels", 12))):
                    keep.append(i)
        if len(keep) != len(legacy.records):
            legacy.source_bubbles = [legacy.source_bubbles[i] for i in keep]
            legacy.target_bubbles = [legacy.target_bubbles[i] for i in keep]
            legacy.records = [legacy.records[i] for i in keep]
            legacy.alignment_diagnostics = dict(legacy.alignment_diagnostics or {})
            legacy.alignment_diagnostics["source_ink_filtered_regions"] = len(keep)

    target_by_suffix = {b.id.rsplit("-", 1)[-1]: b for b in legacy.target_bubbles}
    for sb in legacy.source_bubbles:
        suffix = sb.id.rsplit("-", 1)[-1]
        tb = target_by_suffix.get(suffix)
        sb.meta["paired_diff_method"] = "structural_v08"
        if tb is not None:
            sb.meta["paired_target_id"] = tb.id
            tb.meta["paired_diff_method"] = "structural_v08"
            tb.meta["paired_source_id"] = sb.id

    records = [
        DiffBubbleRecord(
            r.source_id, r.target_id, r.change_density, r.mask_iou, r.confidence,
            r.bbox_target, method="structural_v08", source_ink_density=0.0,
            target_ink_density=0.0, region_kind=getattr(r, "region_kind", "bubble"),
            changed_pixels=int(getattr(r, "changed_pixels", 0)),
        )
        for r in legacy.records
    ]
    diag = dict(getattr(legacy, "alignment_diagnostics", {}) or {})
    diag.update({
        "compatibility_route": "v0.8_structural",
        "bubble_regions": sum(1 for r in records if r.region_kind == "bubble"),
        "free_text_regions": sum(1 for r in records if r.region_kind == "free_text"),
        "complex_text_regions": sum(1 for r in records if r.region_kind == "complex_text"),
    })
    return PairedDiffResult(
        legacy.source_bubbles, legacy.target_bubbles, legacy.change_mask, records,
        legacy.threshold, legacy.noise_floor, method="structural_v08",
        safe_to_skip_ocr=False, diagnostics=diag,
        aligned_source=getattr(legacy, "aligned_source", None),
        alignment_diagnostics=dict(getattr(legacy, "alignment_diagnostics", {}) or {}),
    )

def _supplement_ink_identity_evidence(
    source_aligned: np.ndarray | None,
    target: np.ndarray,
    mask: np.ndarray,
    region_kind: str,
    cfg: MaskReplaceConfig,
) -> dict:
    if not bool(getattr(cfg, "paired_diff_supplement_ink_identity_gate_enabled", True)):
        return {"enabled": False, "passed": True}
    if source_aligned is None or source_aligned.shape[:2] != target.shape[:2]:
        return {"enabled": True, "passed": False, "reason": "missing_aligned_source"}
    ink_change, ink_sdens, ink_tdens, _ = _ink_change_score(source_aligned, target, mask, mask)
    min_source_density = 0.0
    if region_kind == "complex_text":
        min_change = float(getattr(cfg, "paired_diff_supplement_complex_min_ink_change_score", 0.55))
    elif region_kind == "free_text":
        min_change = float(getattr(cfg, "paired_diff_supplement_free_min_ink_change_score", 0.45))
        min_source_density = float(getattr(cfg, "paired_diff_supplement_free_min_source_ink_density", 0.025))
    else:
        min_change = float(getattr(cfg, "paired_diff_supplement_bubble_min_ink_change_score", 0.35))
    max_ratio = float(getattr(cfg, "paired_diff_supplement_max_ink_density_ratio", 3.5))
    density_ratio = float(max(ink_sdens, ink_tdens) / max(1e-6, min(ink_sdens, ink_tdens))) if min(ink_sdens, ink_tdens) > 0 else 999.0
    passed = bool(ink_change >= min_change and density_ratio <= max_ratio and ink_sdens >= min_source_density)
    return {
        "enabled": True,
        "passed": passed,
        "region_kind": str(region_kind),
        "ink_change_score": float(ink_change),
        "source_ink_density": float(ink_sdens),
        "target_ink_density": float(ink_tdens),
        "ink_density_ratio": density_ratio,
        "min_ink_change_score": min_change,
        "min_source_ink_density": min_source_density,
        "max_ink_density_ratio": max_ratio,
    }


def _structural_supplement(
    primary: PairedDiffResult,
    structural: PairedDiffResult | None,
    target: np.ndarray,
    cfg: MaskReplaceConfig,
) -> PairedDiffResult | None:
    """Keep high-value structural regions not already covered by the primary route.

    v0.8.21 uses this for both photographed and clean same-source pages. Closed
    balloon handling remains conservative, while open/complex text may live on a
    coloured background if the structural detector proved two-sided compact ink.
    Supplemental regions are always non-overlapping with the primary write masks.
    """
    if structural is None or not structural.records:
        return None
    h, w = target.shape[:2]
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    photo_boxes = []
    photo_masks = []
    for b in primary.target_bubbles:
        if b.mask is None or b.mask.shape != (h, w):
            continue
        box = _bbox(b.mask)
        if box is not None:
            photo_boxes.append(box)
            photo_masks.append(b.mask)

    src_by = {b.id: b for b in structural.source_bubbles}
    dst_by = {b.id: b for b in structural.target_bubbles}
    keep_records = []
    keep_src = []
    keep_dst = []
    for rec in structural.records:
        sb = src_by.get(rec.source_id); tb = dst_by.get(rec.target_id)
        if sb is None or tb is None or tb.mask is None or tb.mask.shape != (h, w):
            continue
        tm = tb.mask
        box = _bbox(tm)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        # Footer/page-number/copyright noise is common in structural diff.
        if y1 > int(h * 0.95):
            continue
        region_kind = str(getattr(rec, "region_kind", "bubble") or "bubble")
        changed_px = int(getattr(rec, "changed_pixels", 0))
        min_changed = (
            int(getattr(cfg, "paired_diff_complex_min_changed_pixels", 70))
            if region_kind == "complex_text" else 100
        )
        if changed_px < min_changed:
            continue
        vals = gray[tm > 0]
        if vals.size == 0:
            continue
        mean_gray = float(np.mean(vals))
        if region_kind == "bubble":
            # Closed-container supplement is still limited to bright/light interiors.
            if mean_gray < 212.0:
                continue
        elif region_kind == "free_text":
            # Original bright open-text rule.
            white_ratio = float(np.mean(vals > 225))
            dark_ratio = float(np.mean(vals < 175))
            if white_ratio < 0.60 or dark_ratio < 0.035:
                continue
        elif region_kind == "complex_text":
            # Coloured burst balloons / open captions: require the stronger metrics
            # produced by paired_diff_v08. Never admit a dark region merely because
            # the scan and master differ photometrically.
            lo = float(getattr(cfg, "paired_diff_complex_min_ink_density", 0.014))
            hi = float(getattr(cfg, "paired_diff_complex_max_ink_density", 0.42))
            sden = float(tb.meta.get("source_ink_density", 0.0))
            tden = float(tb.meta.get("target_ink_density", 0.0))
            scomp = int(tb.meta.get("source_compact_components", 0))
            tcomp = int(tb.meta.get("target_compact_components", 0))
            min_comp = int(getattr(cfg, "paired_diff_complex_min_compact_components", 3))
            if not (lo <= sden <= hi and lo <= tden <= hi and min(scomp, tcomp) >= min_comp):
                continue
            if int(getattr(rec, "changed_pixels", 0)) < int(getattr(cfg, "paired_diff_complex_min_changed_pixels", 70)):
                continue
            if cv2.countNonZero(tm) / max(1, h * w) > float(getattr(cfg, "paired_diff_complex_max_region_ratio", 0.065)):
                continue
        else:
            continue
        ink_ev = _supplement_ink_identity_evidence(structural.aligned_source, target, tm, region_kind, cfg)
        sb.meta["supplement_ink_identity_gate"] = ink_ev
        tb.meta["supplement_ink_identity_gate"] = ink_ev
        if not bool(ink_ev.get("passed", False)):
            continue
        # Reject anything overlapping or immediately adjacent to a region already
        # handled by the photo detector; this also protects the clipped 009 edge
        # bubble from being silently reintroduced through the legacy detector.
        overlap = any(_iou(tm, pm) > 0.04 for pm in photo_masks)
        if overlap:
            continue
        cx = (x0 + x1) * 0.5; cy = (y0 + y1) * 0.5
        near = False
        for px0, py0, px1, py1 in photo_boxes:
            pad = max(42, int(0.16 * max(px1 - px0, py1 - py0)))
            if px0 - pad <= cx <= px1 + pad and py0 - pad <= cy <= py1 + pad:
                near = True; break
        if near and region_kind != "complex_text":
            continue
        supplement_method = f"{primary.method}_structural_supplement"
        sb.meta["paired_diff_method"] = supplement_method
        tb.meta["paired_diff_method"] = supplement_method
        if primary.method == "photo_pair":
            sb.meta["photo_source"] = True
            tb.meta["photo_source"] = True
        keep_records.append(rec); keep_src.append(sb); keep_dst.append(tb)

    if not keep_records:
        return None
    diag = dict(structural.diagnostics)
    diag.update({
        "supplement_for": primary.method,
        "supplement_regions": len(keep_records),
        "supplement_free_text": sum(1 for r in keep_records if r.region_kind == "free_text"),
        "supplement_complex_text": sum(1 for r in keep_records if r.region_kind == "complex_text"),
        "supplement_bubbles": sum(1 for r in keep_records if r.region_kind == "bubble"),
    })
    return PairedDiffResult(
        keep_src, keep_dst, structural.change_mask, keep_records, structural.threshold,
        structural.noise_floor, method=f"{primary.method}_structural_supplement",
        safe_to_skip_ocr=False, diagnostics=diag,
        aligned_source=structural.aligned_source,
        alignment_diagnostics=dict(structural.alignment_diagnostics),
    )


def _structural_supplement_for_photo(
    photo: PairedDiffResult,
    structural: PairedDiffResult | None,
    target: np.ndarray,
    cfg: MaskReplaceConfig | None = None,
) -> PairedDiffResult | None:
    """Backward-compatible v0.8.20 helper name used by tests/plugins."""
    return _structural_supplement(photo, structural, target, cfg or MaskReplaceConfig())


def extract_paired_diff_bubbles(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: MaskReplaceConfig | None = None,
) -> PairedDiffResult:
    """Extract translated bubble/text-box regions from an aligned page pair.

    Two deterministic routes are used:
    * ``raw_diff`` for same-source scans/exports with small photometric residuals.
    * ``photo_pair`` for photographed old editions with glare, white-balance drift,
      perspective and blur. It trusts the clean Japanese page for geometry and
      compares illumination-invariant local ink structure rather than raw pixels.

    The photo route is deliberately conservative and never claims OCR completeness;
    hybrid mode should keep OCR enabled to recover regions that cannot be isolated
    as a closed white component.
    """
    cfg = cfg or MaskReplaceConfig()
    raw = _extract_raw(source, target, registration, cfg)
    photo_trigger = (
        cfg.photo_pair_fallback_enabled
        and registration.confidence >= cfg.photo_pair_min_registration_confidence
        and (raw.noise_floor >= cfg.photo_pair_noise_floor_trigger or not raw.source_bubbles)
    )
    if not photo_trigger:
        # A clean/same-source page can still contain open burst text that the
        # white-component raw detector cannot represent. Run the structural route
        # as a *supplement* rather than forcing OCR/relettering. If raw found no
        # region at all, structural recovery becomes the primary result.
        structural = _structural_v08_fallback(source, target, registration, cfg)
        if structural is not None:
            if not raw.source_bubbles:
                structural.diagnostics["raw_empty_structural_recovery"] = True
                return structural
            # A raw connected component can span almost the entire page when
            # rendition colour/halftone differences join several balloons into
            # one island.  Keeping that parent as the primary candidate makes
            # target-driven transfer reject the whole page and hides the much
            # better v0.8 structural children.  Prefer the children whenever
            # the raw result contains a page-scale parent and structural
            # recovery produced at least two usable regions.
            broad_raw = False
            for rb in raw.records:
                x0, y0, x1, y1 = rb.bbox_target
                area_ratio = max(0.0, (x1 - x0) * (y1 - y0) / max(1, target.shape[0] * target.shape[1]))
                if area_ratio >= 0.42 or (x1 - x0) >= target.shape[1] * 0.72 and (y1 - y0) >= target.shape[0] * 0.72:
                    broad_raw = True
                    break
            if broad_raw and len(structural.records) >= 2:
                structural.diagnostics["raw_broad_parent_replaced"] = True
                structural.diagnostics["raw_broad_parent_count"] = len(raw.records)
                return structural
            raw.supplemental = _structural_supplement(raw, structural, target, cfg)
            raw.diagnostics["structural_supplement_regions"] = (
                len(raw.supplemental.records) if raw.supplemental is not None else 0
            )
        return raw
    photo = _photo_pair_fallback(source, target, registration, cfg, raw.threshold, raw.noise_floor)
    # If the photo route found nothing, keep the raw result as evidence but do not
    # permit OCR skipping: high photometric noise means raw differences are unsafe.
    if not photo.source_bubbles:
        structural = _structural_v08_fallback(source, target, registration, cfg)
        if structural is not None:
            structural.diagnostics["photo_fallback_attempted"] = True
            return structural
        raw.safe_to_skip_ocr = False
        raw.method = "raw_diff_unreliable_photo"
        raw.diagnostics["photo_fallback_attempted"] = True
        raw.diagnostics["structural_v08_fallback_attempted"] = bool(getattr(cfg, "paired_diff_structural_fallback_enabled", True))
        return raw

    # v0.8.5: complement, do not replace, the safer photo detector. The legacy
    # structural route is filtered to non-overlapping bright/text-like regions so
    # open burst bubbles can be recovered while edge-clipped/closed bubbles keep
    # the v0.8.4 integrity guards.
    # Do not skip structural recovery merely because the pair is
    # monochrome-to-colour. The photo route handles enclosed white balloons,
    # while the structural supplement is what recovers coloured burst balloons,
    # open captions and large black/white text panels (notably pages 065/066).
    structural = _structural_v08_fallback(source, target, registration, cfg)
    photo.supplemental = _structural_supplement(photo, structural, target, cfg)
    photo.diagnostics["structural_supplement_regions"] = (
        len(photo.supplemental.records) if photo.supplemental is not None else 0
    )
    return photo
