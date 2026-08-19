from __future__ import annotations

"""Experimental page-aligned erase-to-reveal transfer.

This module is intentionally pixel-only.  It receives already-decoded SOURCE / TARGET
images plus an existing RegistrationResult and returns a reversible candidate image,
masks, per-region triage and diagnostics.  It never writes project state or final files.

Design constraints:
- TARGET remains the background/colour authority.
- SOURCE contributes only registered ink by default.
- Full SOURCE raster is permitted only inside a tightly bounded, proven near-white
  corridor and is never allowed on colourful TARGET pixels.
- Registration and area gates are stricter than the ordinary Direct path.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .config import AlignedOverlayRevealConfig
from .models import RegistrationResult
from .registration import warp_source_to_target


@dataclass(slots=True)
class AlignedOverlayRegion:
    id: str
    target_bbox: tuple[int, int, int, int]
    source_bbox: tuple[int, int, int, int]
    erase_mask: np.ndarray
    source_ink_mask: np.ndarray
    full_raster_mask: np.ndarray
    composite_mode: str
    triage: str
    reason: str
    white_ratio: float
    color_ratio: float
    erase_area_ratio: float
    source_ink_pixels: int
    target_ink_pixels: int
    border_guard_px: int
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_bbox": list(self.target_bbox),
            "source_bbox": list(self.source_bbox),
            "composite_mode": self.composite_mode,
            "triage": self.triage,
            "reason": self.reason,
            "white_ratio": float(self.white_ratio),
            "color_ratio": float(self.color_ratio),
            "erase_area_ratio": float(self.erase_area_ratio),
            "source_ink_pixels": int(self.source_ink_pixels),
            "target_ink_pixels": int(self.target_ink_pixels),
            "border_guard_px": int(self.border_guard_px),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(slots=True)
class AlignedOverlayPlan:
    accepted: bool
    reason: str
    aligned_source: np.ndarray
    valid_mask: np.ndarray
    erase_mask: np.ndarray
    source_ink_mask: np.ndarray
    full_raster_mask: np.ndarray
    regions: list[AlignedOverlayRegion]
    diagnostics: dict[str, Any]

    @property
    def applied_regions(self) -> list[AlignedOverlayRegion]:
        return [r for r in self.regions if r.triage != "REJECT"]

    @property
    def page_triage(self) -> str:
        states = {r.triage for r in self.regions}
        if "REJECT" in states:
            return "REJECT" if not self.applied_regions else "REVIEW"
        if "REVIEW" in states:
            return "REVIEW"
        if "SAFE" in states:
            return "SAFE"
        return "REJECT"


@dataclass(slots=True)
class AlignedOverlayResult:
    image: np.ndarray
    layer_rgba: np.ndarray
    erase_mask: np.ndarray
    source_ink_mask: np.ndarray
    regions_overlay: np.ndarray
    plan: AlignedOverlayPlan
    diagnostics: dict[str, Any]

    @property
    def applied_count(self) -> int:
        return len(self.plan.applied_regions)

    @property
    def accepted(self) -> bool:
        return bool(self.plan.accepted)

    @property
    def page_triage(self) -> str:
        return self.plan.page_triage


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()


def _kernel(radius: int) -> np.ndarray:
    radius = max(0, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8) * 255
    return cv2.dilate((mask > 0).astype(np.uint8) * 255, _kernel(radius))


def _erode(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return (mask > 0).astype(np.uint8) * 255
    return cv2.erode((mask > 0).astype(np.uint8) * 255, _kernel(radius))


def _registration_gate(registration: RegistrationResult, cfg: AlignedOverlayRevealConfig) -> tuple[bool, str, dict[str, Any]]:
    metrics = {
        "confidence": float(registration.confidence),
        "inlier_ratio": float(registration.inlier_ratio),
        "reprojection_error": float(registration.reprojection_error),
        "spatial_coverage": float(registration.spatial_coverage),
        "method": str(registration.method),
    }
    if registration.confidence < cfg.min_registration_confidence:
        return False, "registration_confidence", metrics
    if registration.reprojection_error > cfg.max_reprojection_error:
        return False, "registration_reprojection_error", metrics
    if registration.inlier_ratio < cfg.min_inlier_ratio:
        return False, "registration_inlier_ratio", metrics
    if registration.spatial_coverage < cfg.min_spatial_coverage:
        return False, "registration_spatial_coverage", metrics
    return True, "ok", metrics


def _warp_valid_mask(source_shape: tuple[int, int], registration: RegistrationResult) -> np.ndarray:
    sh, sw = source_shape
    tw, th = registration.target_size
    ones = np.full((sh, sw), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        ones,
        np.asarray(registration.matrix, dtype=np.float64),
        (tw, th),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _text_like_components(
    dark_mask: np.ndarray,
    exclusive_seed: np.ndarray,
    *,
    min_area: int,
    max_area_ratio: float,
    max_span_ratio: float,
) -> np.ndarray:
    """Keep complete dark connected components that contain changed-ink evidence.

    The seed is conservative (only source/target-exclusive pixels). Expanding from
    seed to its connected dark component recovers antialiased/full glyph strokes,
    while page frames and bubble outlines are rejected by their large span/bbox.
    """
    dark = (dark_mask > 0).astype(np.uint8)
    seed = exclusive_seed > 0
    h, w = dark.shape
    page_area = max(1, h * w)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    keep = np.zeros_like(dark, dtype=np.uint8)
    for label in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[label]]
        if area < max(1, int(min_area)):
            continue
        if area / page_area > float(max_area_ratio):
            continue
        if ww / max(1, w) > float(max_span_ratio) or hh / max(1, h) > float(max_span_ratio):
            continue
        comp = labels == label
        if not np.any(seed & comp):
            continue
        keep[comp] = 255
    return keep


def _structural_guard(target_gray: np.ndarray, source_gray: np.ndarray, cfg: AlignedOverlayRevealConfig) -> np.ndarray:
    """Protect long/large dark structures shared by SOURCE and TARGET.

    This deliberately does not try to classify every line.  It only protects
    components whose geometry looks more like a bubble/frame/art contour than a
    glyph.  Combined with exclusive-ink seeding, this is a second independent
    defence against writing across borders.
    """
    h, w = target_gray.shape
    target_dark = (target_gray <= int(cfg.target_ink_threshold)).astype(np.uint8)
    source_dark = (source_gray <= int(cfg.source_ink_threshold)).astype(np.uint8)
    common = target_dark & (_dilate(source_dark * 255, int(cfg.registration_tolerance_px)) > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(common.astype(np.uint8), 8)
    guard = np.zeros_like(target_gray, dtype=np.uint8)
    page_area = max(1, h * w)
    for label in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[label]]
        span = max(ww / max(1, w), hh / max(1, h))
        bbox_ratio = (ww * hh) / page_area
        if (
            area >= int(cfg.structural_component_min_area_px)
            or span >= float(cfg.structural_component_min_span_ratio)
            or bbox_ratio >= float(cfg.structural_component_min_bbox_ratio)
        ):
            guard[labels == label] = 255
    return _dilate(guard, int(cfg.border_protect_px))


def _white_container_envelope(
    target_gray: np.ndarray,
    text_seed: np.ndarray,
    structural_guard: np.ndarray,
    valid_mask: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> np.ndarray:
    """Recover a closed near-white TARGET container around a local text seed.

    v1.2.3 keeps this operation local.  The old implementation relabelled the
    entire page once per candidate region, which made the new ``hybrid`` default
    unnecessarily expensive on real 1600-2400 px pages and could make the GUI
    look frozen.  The safety semantics are unchanged: TARGET white pixels are
    the envelope authority and SOURCE RGB is never consulted here.
    """
    seed = text_seed > 0
    if not np.any(seed):
        return np.zeros_like(target_gray, dtype=np.uint8)

    seed_bbox = _bbox_from_mask(text_seed, pad=max(8, int(getattr(cfg, "white_container_search_pad_px", 96))))
    if seed_bbox is None:
        return np.zeros_like(target_gray, dtype=np.uint8)
    x0, y0, x1, y1 = seed_bbox
    tg = target_gray[y0:y1, x0:x1]
    sd = seed[y0:y1, x0:x1]
    gd = structural_guard[y0:y1, x0:x1] > 0
    vd = valid_mask[y0:y1, x0:x1] > 0

    white_floor = min(int(cfg.white_threshold), 232)
    passable = ((tg >= white_floor) | sd) & vd & ~gd
    n, labels, stats, _ = cv2.connectedComponentsWithStats(passable.astype(np.uint8), 8)
    best_label = 0
    best_overlap = 0
    page_area = max(1, target_gray.shape[0] * target_gray.shape[1])
    max_ratio = float(getattr(cfg, "white_container_max_area_ratio", 0.45))
    seed_touch = _dilate(sd.astype(np.uint8) * 255, 2).astype(bool)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0 or area / page_area > max_ratio:
            continue
        comp = labels == label
        overlap = int(np.count_nonzero(comp & seed_touch))
        if overlap > best_overlap:
            best_overlap = overlap
            best_label = label
    if best_label <= 0 or best_overlap <= 0:
        return np.zeros_like(target_gray, dtype=np.uint8)

    local = (labels == best_label).astype(np.uint8) * 255
    local[sd] = 255
    out = np.zeros_like(target_gray, dtype=np.uint8)
    out[y0:y1, x0:x1] = local
    return out


def _bbox_from_mask(mask: np.ndarray, pad: int = 0) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    h, w = mask.shape
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(w, int(xs.max()) + 1 + pad)
    y1 = min(h, int(ys.max()) + 1 + pad)
    return x0, y0, x1, y1


def _target_bbox_to_source(bbox: tuple[int, int, int, int], registration: RegistrationResult) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = bbox
    pts = np.array([[[x0, y0], [x1, y0], [x1, y1], [x0, y1]]], dtype=np.float32)
    try:
        inv = np.linalg.inv(np.asarray(registration.matrix, dtype=np.float64))
        mapped = cv2.perspectiveTransform(pts, inv)[0]
    except (np.linalg.LinAlgError, cv2.error):
        return bbox
    sw, sh = registration.source_size
    sx0 = max(0, min(sw, int(np.floor(mapped[:, 0].min()))))
    sy0 = max(0, min(sh, int(np.floor(mapped[:, 1].min()))))
    sx1 = max(sx0, min(sw, int(np.ceil(mapped[:, 0].max()))))
    sy1 = max(sy0, min(sh, int(np.ceil(mapped[:, 1].max()))))
    return sx0, sy0, sx1, sy1


def _region_background_metrics(target: np.ndarray, corridor: np.ndarray, cfg: AlignedOverlayRevealConfig) -> tuple[float, float]:
    sel = corridor > 0
    if not np.any(sel):
        return 0.0, 1.0
    gray = _gray(target)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    values = gray[sel]
    sats = hsv[:, :, 1][sel]
    white_ratio = float(np.count_nonzero(values >= int(cfg.white_threshold)) / max(1, values.size))
    color_ratio = float(np.count_nonzero(sats >= int(cfg.color_saturation_threshold)) / max(1, sats.size))
    return white_ratio, color_ratio


def _outer_dark_ratio(target_gray: np.ndarray, corridor: np.ndarray, radius: int = 2) -> float:
    outer = (_dilate(corridor, max(1, radius)) > 0) & ~(corridor > 0)
    if not np.any(outer):
        return 0.0
    return float(np.count_nonzero(target_gray[outer] <= 190) / max(1, np.count_nonzero(outer)))


def _page_tier(registration: RegistrationResult, regions: list[AlignedOverlayRegion], cfg: AlignedOverlayRevealConfig) -> str:
    applied = [r for r in regions if r.triage != "REJECT"]
    if not applied:
        return "REJECT"
    if any(r.triage == "REVIEW" for r in applied):
        return "REVIEW"
    if any(r.triage == "REJECT" for r in regions):
        return "REVIEW"
    return "SAFE"


def build_aligned_overlay_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayPlan:
    aligned = warp_source_to_target(source, registration)
    valid = _warp_valid_mask(source.shape[:2], registration)
    h, w = target.shape[:2]
    empty = np.zeros((h, w), dtype=np.uint8)
    erase_source = str(cfg.erase_source or "target_text_ink").strip().lower()
    if erase_source not in {"target_text_ink", "white_bubble_interior", "hybrid"}:
        return AlignedOverlayPlan(
            accepted=False, reason="unsupported_erase_source", aligned_source=aligned,
            valid_mask=valid, erase_mask=empty.copy(), source_ink_mask=empty.copy(),
            full_raster_mask=empty.copy(), regions=[],
            diagnostics={"registration_gate_passed": False, "erase_source": erase_source, "manual_effect_candidates": []},
        )

    gate_ok, gate_reason, reg_metrics = _registration_gate(registration, cfg)
    if not gate_ok:
        return AlignedOverlayPlan(
            accepted=False,
            reason=f"rejected_registration:{gate_reason}",
            aligned_source=aligned,
            valid_mask=valid,
            erase_mask=empty.copy(),
            source_ink_mask=empty.copy(),
            full_raster_mask=empty.copy(),
            regions=[],
            diagnostics={
                "registration_gate": reg_metrics,
                "registration_gate_passed": False,
                "reason": gate_reason,
                "thresholds": {
                    "min_registration_confidence": float(cfg.min_registration_confidence),
                    "max_reprojection_error": float(cfg.max_reprojection_error),
                    "min_inlier_ratio": float(cfg.min_inlier_ratio),
                    "min_spatial_coverage": float(cfg.min_spatial_coverage),
                },
                "manual_effect_candidates": [],
            },
        )

    sg = _gray(aligned)
    tg = _gray(target)
    valid_bool = valid > 0
    src_dark = (sg <= int(cfg.source_ink_threshold)) & valid_bool
    tgt_dark = (tg <= int(cfg.target_ink_threshold)) & valid_bool
    tol = max(0, int(cfg.registration_tolerance_px))
    src_near = _dilate(src_dark.astype(np.uint8) * 255, tol) > 0
    tgt_near = _dilate(tgt_dark.astype(np.uint8) * 255, tol) > 0

    # Exclusive pixels are the page-wide equivalent of a detector text seed:
    # shared artwork survives registration, while different Chinese/Japanese
    # glyph strokes become strong seeds.  A small tone delta further suppresses
    # antialias/noise differences on unchanged art.
    delta = max(0, int(cfg.ink_difference_delta))
    src_exclusive = src_dark & (~tgt_near | ((tg.astype(np.int16) - sg.astype(np.int16)) >= delta))
    tgt_exclusive = tgt_dark & (~src_near | ((sg.astype(np.int16) - tg.astype(np.int16)) >= delta))

    src_text = _text_like_components(
        src_dark.astype(np.uint8) * 255,
        src_exclusive.astype(np.uint8) * 255,
        min_area=int(cfg.min_component_area_px),
        max_area_ratio=float(cfg.max_component_area_ratio),
        max_span_ratio=float(cfg.max_component_span_ratio),
    )
    tgt_text = _text_like_components(
        tgt_dark.astype(np.uint8) * 255,
        tgt_exclusive.astype(np.uint8) * 255,
        min_area=int(cfg.min_component_area_px),
        max_area_ratio=float(cfg.max_component_area_ratio),
        max_span_ratio=float(cfg.max_component_span_ratio),
    )

    src_text_pixels = int(cv2.countNonZero(src_text))
    tgt_text_pixels = int(cv2.countNonZero(tgt_text))
    seed_diag = {
        "src_exclusive_approx": int(np.count_nonzero(src_exclusive)),
        "tgt_exclusive_approx": int(np.count_nonzero(tgt_exclusive)),
        "src_text_pixels": src_text_pixels,
        "tgt_text_pixels": tgt_text_pixels,
    }
    weak_text_seed = bool(
        erase_source in {"target_text_ink", "hybrid"}
        and tgt_text_pixels < max(30, int(cfg.min_region_ink_pixels) * 3)
    )

    guard_base = _structural_guard(tg, sg, cfg)
    group_seed = cv2.bitwise_or(src_text, tgt_text)
    # If conservative component recovery is weak, retain raw exclusive ink as
    # grouping evidence.  It is never written directly; the later guard,
    # corridor and TARGET-side container checks still control the erase/write.
    if weak_text_seed:
        raw_seed = cv2.bitwise_or(
            src_exclusive.astype(np.uint8) * 255,
            tgt_exclusive.astype(np.uint8) * 255,
        )
        group_seed = cv2.bitwise_or(group_seed, raw_seed)
    group_seed = _dilate(group_seed, int(cfg.region_group_radius_px))
    if int(cfg.region_close_radius_px) > 0:
        k = _kernel(int(cfg.region_close_radius_px))
        group_seed = cv2.morphologyEx(group_seed, cv2.MORPH_CLOSE, k)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((group_seed > 0).astype(np.uint8), 8)
    page_area = max(1, h * w)
    regions: list[AlignedOverlayRegion] = []
    page_erase = empty.copy()
    page_source = empty.copy()
    page_full = empty.copy()
    manual_candidates: list[dict[str, Any]] = []

    for label in range(1, n):
        x, y, ww, hh, area = [int(v) for v in stats[label]]
        if area <= 0:
            continue
        region_group = labels == label
        # ``max_single_region_area_ratio`` is an erase-mask safety cap, not a
        # bounding-box cap.  A normal speech bubble/text corridor can span well
        # over 4% of a page while the actual glyph erase pixels remain tiny.
        # Applying the cap to the grouped bbox would reject exactly the clean
        # white-bubble case this experimental route is intended to preview.

        local_src = np.where(region_group, src_text, 0).astype(np.uint8)
        local_tgt = np.where(region_group, tgt_text, 0).astype(np.uint8)
        source_pixels = int(cv2.countNonZero(local_src))
        target_pixels = int(cv2.countNonZero(local_tgt))
        if source_pixels < int(cfg.min_region_ink_pixels) or target_pixels < int(cfg.min_region_ink_pixels):
            continue

        corridor_seed = cv2.bitwise_or(local_src, local_tgt)
        text_corridor = _dilate(corridor_seed, int(cfg.text_corridor_radius_px)) if cfg.text_corridor_enabled else region_group.astype(np.uint8) * 255
        # Restrict to registered SOURCE support so no white warp border can become
        # an accidental full-raster authority.
        text_corridor[~valid_bool] = 0
        white_envelope = np.zeros_like(text_corridor)
        if erase_source in {"white_bubble_interior", "hybrid"}:
            white_envelope = _white_container_envelope(tg, corridor_seed, guard_base, valid, cfg)
        if erase_source == "white_bubble_interior":
            corridor = white_envelope
        elif erase_source == "hybrid" and weak_text_seed and cv2.countNonZero(white_envelope) > 0:
            # Weak text evidence: let the proven TARGET-side white container be
            # the primary safety envelope.  The actual erase/write below still
            # uses recovered TARGET/SOURCE ink, so this does not paste a whole
            # bubble or grant SOURCE background authority.
            corridor = white_envelope.copy()
            corridor = cv2.bitwise_or(corridor, text_corridor)
        elif erase_source == "hybrid" and cv2.countNonZero(white_envelope) > 0:
            corridor = cv2.bitwise_or(text_corridor, white_envelope)
        else:
            corridor = text_corridor

        outer_dark = _outer_dark_ratio(tg, corridor, radius=2)
        extra_guard = 0
        max_outer = max(1e-6, float(cfg.max_outer_dark_ratio))
        if outer_dark > max_outer:
            overflow = min(1.0, (outer_dark - max_outer) / max_outer)
            extra_guard = min(int(cfg.progressive_inset_steps), max(1, int(np.ceil(overflow * max(1, cfg.progressive_inset_steps)))))
        guard_px = int(cfg.border_protect_px) + extra_guard
        guard = _dilate(guard_base, max(0, extra_guard)) if guard_px > int(cfg.border_protect_px) else guard_base

        erase = cv2.bitwise_and(local_tgt, cv2.bitwise_not(guard))
        erase = _dilate(erase, int(cfg.erase_dilate_px))
        erase = cv2.bitwise_and(erase, corridor)
        erase = cv2.bitwise_and(erase, cv2.bitwise_not(guard))

        source_write = cv2.bitwise_and(local_src, cv2.bitwise_not(guard))
        # A one-pixel recovery around kept source glyph components restores their
        # antialias fringe without widening into unrelated page artwork.
        if int(cfg.source_ink_antialias_px) > 0:
            fringe = _dilate(source_write, int(cfg.source_ink_antialias_px))
            source_write = np.where((fringe > 0) & (sg <= int(cfg.source_antialias_threshold)), 255, source_write).astype(np.uint8)
        source_write = cv2.bitwise_and(source_write, corridor)
        source_write = cv2.bitwise_and(source_write, cv2.bitwise_not(guard))

        bbox = _bbox_from_mask(corridor, pad=int(cfg.region_bbox_pad_px)) or (x, y, x + ww, y + hh)
        src_bbox = _target_bbox_to_source(bbox, registration)
        white_ratio, color_ratio = _region_background_metrics(target, corridor, cfg)
        erase_ratio = float(cv2.countNonZero(erase) / page_area)
        full_mask = np.zeros_like(erase)
        composite_mode = "ink_only"
        source_bg_visible = False
        color_exposure = False

        if not bool(cfg.prefer_source_ink_only):
            if (
                bool(cfg.allow_full_source_raster_on_white)
                and white_ratio >= float(cfg.min_target_white_ratio_for_full_raster)
                and (not cfg.forbid_full_raster_on_color_target or color_ratio <= float(cfg.max_color_ratio_for_full_raster))
            ):
                # The full-raster fallback is intentionally corridor-limited and
                # eroded away from structural guards/borders. It is not a whole
                # page or whole bounding-box paste.
                full_mask = cv2.bitwise_and(corridor, cv2.bitwise_not(_dilate(guard, max(1, int(cfg.border_protect_px)))))
                if int(cfg.full_raster_inset_px) > 0:
                    full_mask = _erode(full_mask, int(cfg.full_raster_inset_px))
                full_ratio = float(cv2.countNonZero(full_mask) / page_area)
                # The plan's single-region cap is a hard write-area invariant. A
                # large white balloon may be used as a safe envelope for ink-only
                # work, but it may not turn into a giant SOURCE raster paste.
                if full_ratio > float(cfg.max_single_region_area_ratio):
                    full_mask[:] = 0
                    composite_mode = "ink_only"
                else:
                    composite_mode = "full_raster_white"
                sel = full_mask > 0
                if np.any(sel):
                    src_bg = sg[sel] >= int(cfg.white_threshold)
                    tgt_not_white = tg[sel] < int(cfg.white_threshold)
                    source_bg_visible = bool(np.any(src_bg & tgt_not_white))
                    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
                    color_exposure = bool(np.any((hsv[:, :, 1][sel] >= int(cfg.color_saturation_threshold)) & src_bg))
            else:
                composite_mode = "ink_only"

        reason = "clean_white_ink_only"
        triage = str(cfg.default_triage).upper()
        if triage not in {"SAFE", "REVIEW", "REJECT"}:
            triage = "REVIEW"

        if erase_source == "white_bubble_interior" and cv2.countNonZero(white_envelope) == 0:
            triage, reason = "REJECT", "no_closed_white_container"
        elif cv2.countNonZero(erase) == 0 or cv2.countNonZero(source_write) == 0:
            triage, reason = "REJECT", "empty_refined_mask"
        elif erase_ratio > float(cfg.max_single_region_area_ratio):
            # A large individual region is not automatically applied as SAFE,
            # but keeping it as REVIEW makes the existing manual Reveal workflow
            # usable instead of silently killing the whole candidate.
            triage, reason = "REVIEW", "erase_area_cap_review"
        elif color_ratio > float(cfg.reject_color_ratio):
            triage, reason = "REJECT", "color_or_open_text_risk"
        elif composite_mode == "full_raster_white" and cfg.forbid_full_raster_on_color_target and color_exposure:
            triage, reason = "REJECT", "full_raster_color_exposure"
        elif source_bg_visible and bool(cfg.force_review_if_source_bg_visible):
            triage, reason = "REVIEW", "source_background_visibility_risk"
        elif color_exposure and bool(cfg.force_review_if_any_color_exposure):
            triage, reason = "REVIEW", "color_exposure_risk"
        elif color_ratio > float(cfg.review_color_ratio):
            triage, reason = "REVIEW", "colored_background"
        elif outer_dark > float(cfg.max_outer_dark_ratio):
            triage, reason = "REVIEW", "border_density_high"
        elif (
            white_ratio >= float(cfg.safe_white_ratio)
            and color_ratio <= float(cfg.safe_color_ratio)
            and registration.confidence >= float(cfg.safe_registration_confidence)
            and erase_ratio <= float(cfg.safe_max_region_erase_ratio)
            and composite_mode == "ink_only"
        ):
            triage, reason = "SAFE", "clean_white_ink_only"

        region = AlignedOverlayRegion(
            id=f"aligned_{label:03d}",
            target_bbox=bbox,
            source_bbox=src_bbox,
            erase_mask=erase,
            source_ink_mask=source_write,
            full_raster_mask=full_mask,
            composite_mode=composite_mode,
            triage=triage,
            reason=reason,
            white_ratio=white_ratio,
            color_ratio=color_ratio,
            erase_area_ratio=erase_ratio,
            source_ink_pixels=int(cv2.countNonZero(source_write)),
            target_ink_pixels=int(cv2.countNonZero(erase)),
            border_guard_px=guard_px,
            diagnostics={
                "outer_dark_ratio": outer_dark,
                "progressive_extra_guard_px": extra_guard,
                "source_bg_visible": source_bg_visible,
                "color_exposure": color_exposure,
                "corridor_pixels": int(cv2.countNonZero(corridor)),
                "text_corridor_pixels": int(cv2.countNonZero(text_corridor)),
                "white_container_pixels": int(cv2.countNonZero(white_envelope)),
                "erase_source": erase_source,
            },
        )
        regions.append(region)
        if triage != "REJECT":
            page_erase = cv2.bitwise_or(page_erase, erase)
            page_source = cv2.bitwise_or(page_source, source_write)
            page_full = cv2.bitwise_or(page_full, full_mask)
        if triage in {"REVIEW", "REJECT"}:
            manual_candidates.append({
                "id": region.id,
                "workflow": "manual_effect",
                "reason": f"aligned_overlay:{reason}",
                "target_bbox": list(region.target_bbox),
                "source_bbox": list(region.source_bbox),
                "suggested_manual_mode": "reveal_text",
                "auto_actionable": True,
                "triage": triage,
            })

    total_erase_ratio = float(cv2.countNonZero(page_erase) / page_area)
    accepted = bool(regions) and bool([r for r in regions if r.triage != "REJECT"])
    reason = "ok" if accepted else "no_accepted_regions"
    if total_erase_ratio > float(cfg.max_erase_area_ratio_per_page):
        accepted = False
        reason = "page_erase_area_cap"
        page_erase[:] = 0
        page_source[:] = 0
        page_full[:] = 0
        for r in regions:
            if r.triage != "REJECT":
                r.triage = "REJECT"
                r.reason = "page_erase_area_cap"
        manual_candidates = [
            {
                "id": r.id, "workflow": "manual_effect", "reason": "aligned_overlay:page_erase_area_cap",
                "target_bbox": list(r.target_bbox), "source_bbox": list(r.source_bbox),
                "suggested_manual_mode": "reveal_text", "auto_actionable": True, "triage": "REJECT",
            }
            for r in regions
        ]

    diagnostics = {
        "registration_gate": reg_metrics,
        "registration_gate_passed": True,
        "erase_source": erase_source,
        "exclusive_source_pixels": int(np.count_nonzero(src_exclusive)),
        "exclusive_target_pixels": int(np.count_nonzero(tgt_exclusive)),
        "source_text_pixels": src_text_pixels,
        "target_text_pixels": tgt_text_pixels,
        "weak_text_seed": weak_text_seed,
        "seed_diagnostics": seed_diag,
        "erase_pixels": int(cv2.countNonZero(page_erase)),
        "source_ink_pixels": int(cv2.countNonZero(page_source)),
        "full_raster_pixels": int(cv2.countNonZero(page_full)),
        "erase_area_ratio": total_erase_ratio,
        "region_count": len(regions),
        "applied_region_count": len([r for r in regions if r.triage != "REJECT"]),
        "triage": _page_tier(registration, regions, cfg),
        "manual_effect_candidates": manual_candidates,
        "config_contract": "target_background_authority",
    }
    return AlignedOverlayPlan(
        accepted=accepted,
        reason=reason,
        aligned_source=aligned,
        valid_mask=valid,
        erase_mask=page_erase,
        source_ink_mask=page_source,
        full_raster_mask=page_full,
        regions=regions,
        diagnostics=diagnostics,
    )


def _regions_overlay(target: np.ndarray, regions: list[AlignedOverlayRegion]) -> np.ndarray:
    out = target.copy()
    # Colours are diagnostic only (OpenCV BGR): SAFE green, REVIEW amber,
    # REJECT red.  These pixels are never used for final compositing.
    colors = {"SAFE": (60, 180, 60), "REVIEW": (0, 170, 255), "REJECT": (40, 40, 220)}
    for r in regions:
        x0, y0, x1, y1 = r.target_bbox
        color = colors.get(r.triage, (255, 255, 0))
        cv2.rectangle(out, (x0, y0), (max(x0, x1 - 1), max(y0, y1 - 1)), color, 2)
        cv2.putText(out, f"{r.id}:{r.triage}", (x0, max(12, y0 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    return out


def execute_aligned_overlay(
    plan: AlignedOverlayPlan,
    source: np.ndarray,
    target: np.ndarray,
    cfg: AlignedOverlayRevealConfig,
) -> AlignedOverlayResult:
    del source  # SOURCE pixels are already aligned/frozen inside the plan.
    rendered = target.copy()
    if plan.accepted and cv2.countNonZero(plan.erase_mask) > 0:
        # Telea is deterministic, local and keeps TARGET colour/texture authority.
        # White-bubble text normally becomes a near-white fill; coloured regions
        # remain gated to REVIEW/REJECT and never receive SOURCE background RGB.
        rendered = cv2.inpaint(
            rendered,
            (plan.erase_mask > 0).astype(np.uint8) * 255,
            max(0.5, float(cfg.inpaint_radius)),
            cv2.INPAINT_TELEA,
        )

        for region in plan.applied_regions:
            if region.composite_mode == "full_raster_white" and cv2.countNonZero(region.full_raster_mask) > 0:
                sel = region.full_raster_mask > 0
                rendered[sel] = plan.aligned_source[sel]
            else:
                sel = region.source_ink_mask > 0
                rendered[sel] = plan.aligned_source[sel]

    changed = np.any(rendered != target, axis=2)
    layer = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
    # export.write_rgba expects RGBA channel order.
    rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
    layer[changed, :3] = rgb[changed]
    layer[changed, 3] = 255

    diagnostics = dict(plan.diagnostics)
    changed_pixels = int(np.count_nonzero(changed))
    nearly_unchanged = bool(plan.accepted and (len(plan.applied_regions) == 0 or changed_pixels < max(16, int(target.shape[0] * target.shape[1] * 0.00001))))
    diagnostics.update({
        "accepted": bool(plan.accepted),
        "reason": plan.reason,
        "applied_count": len(plan.applied_regions),
        "changed_pixels": changed_pixels,
        "nearly_unchanged": nearly_unchanged,
        "result_hint": "accepted_but_almost_no_visible_change" if nearly_unchanged else "ok",
        "source_background_authority": False,
        "page_triage": "REVIEW" if nearly_unchanged else plan.page_triage,
    })
    return AlignedOverlayResult(
        image=rendered,
        layer_rgba=layer,
        erase_mask=plan.erase_mask.copy(),
        source_ink_mask=plan.source_ink_mask.copy(),
        regions_overlay=_regions_overlay(target, plan.regions),
        plan=plan,
        diagnostics=diagnostics,
    )
