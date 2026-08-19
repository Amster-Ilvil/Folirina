from __future__ import annotations

"""Whole-page registered transparent-bubble reveal.

This route intentionally does *not* reuse Direct Patch, Mask Transfer, inpaint,
or lettering. SOURCE is warped once into TARGET coordinates and used only as the
opaque lower layer. Bubble/container detection runs only on TARGET. The TARGET
RGBA layer then receives transparent holes so the aligned Chinese pixels show
through when composited underneath.
"""

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from .bubbles import detect_mangalens_bubbles, detect_seeded_white_bubbles, detect_unseeded_white_containers
from .config import BubbleConfig, MaskReplaceConfig, TransparentBubbleRevealConfig
from .layout_evidence import (
    collect_koharu_layout_evidence_cached, classify_layout_authority,
    filter_candidates_by_layout_authority,
)
from .detector_policy import (
    primary_detector, detector_strategy, transparent_auxiliary_backends,
    should_run_auxiliaries, policy_uses_koharu, koharu_is_primary, STRATEGY_ALWAYS,
)
from .geometry import mask_to_largest_polygon, polygon_bbox, rasterize_polygon


def collect_koharu_layout_evidence(image, bubble_cfg=None, role="page", allow_missing=True, **kwargs):
    return collect_koharu_layout_evidence_cached(
        image, bubble_cfg, role=role, allow_missing=allow_missing, **kwargs
    )
from .models import BubbleInstance, RegistrationResult, TextBlock
from .pipeline_bubble_service import primary_bubbles_cached
from .registration import warp_source_to_target
from .semantic import analyze_semantic_layout, decide_candidate, constrain_text_only_mask

from .text_only_transfer import (
    changed_text_masks,
    source_text_alpha, source_text_render,
    target_container_border_mask,
    target_white_container_text_mask,
    transfer_text_only,
    white_container_paper_mask,
)


@dataclass(slots=True)
class TransparentBubbleRegion:
    id: str
    target_bbox: tuple[int, int, int, int]
    polygon: list[tuple[float, float]]
    clear_mask: np.ndarray
    confidence: float
    backend: str
    triage: str
    reason: str
    applied: bool = True
    clear_mode: str = "full_bubble"
    text_bbox: tuple[int, int, int, int] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_bbox": [int(v) for v in self.target_bbox],
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "confidence": float(self.confidence),
            "backend": self.backend,
            "triage": self.triage,
            "reason": self.reason,
            "applied": bool(self.applied),
            "clear_mode": self.clear_mode,
            "clear_pixels": int(cv2.countNonZero(self.clear_mask)),
            **({"text_bbox": [int(v) for v in self.text_bbox]} if self.text_bbox is not None else {}),
        }


@dataclass(slots=True)
class TransparentBubblePlan:
    accepted: bool
    reason: str
    aligned_source: np.ndarray
    valid_mask: np.ndarray
    clear_mask: np.ndarray
    regions: list[TransparentBubbleRegion] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    page_triage: str = "REJECT"

    @property
    def applied_regions(self) -> list[TransparentBubbleRegion]:
        return [r for r in self.regions if r.applied and cv2.countNonZero(r.clear_mask) > 0]


@dataclass(slots=True)
class TransparentBubbleResult:
    image_rgb: np.ndarray
    # The transparent top layer. This is deliberately *not* flattened against CN;
    # saving it as final_rgba.png preserves alpha=0 in the bubble openings.
    image_rgba: np.ndarray
    jp_layer_rgba: np.ndarray
    cn_layer_rgb: np.ndarray
    clear_mask: np.ndarray
    plan: TransparentBubblePlan
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def applied_count(self) -> int:
        return len(self.plan.applied_regions)


def alpha_over(top_rgba: np.ndarray, bottom_rgba: np.ndarray) -> np.ndarray:
    """Porter-Duff source-over for uint8 RGBA arrays.

    Solid (alpha=255) and clear (alpha=0) pixels are copied byte-for-byte so
    Japanese skin/artwork outside transparent holes never shifts colour from
    float round-trips. Only partial alpha uses blending.
    """
    if top_rgba.shape != bottom_rgba.shape or top_rgba.ndim != 3 or top_rgba.shape[2] != 4:
        raise ValueError("alpha_over expects equal HxWx4 RGBA arrays")
    top = top_rgba
    bottom = bottom_rgba
    at = top[..., 3]
    out = np.empty_like(top)
    solid = at == 255
    clear = at == 0
    partial = ~solid & ~clear
    out[solid] = top[solid]
    out[clear, :3] = bottom[clear, :3]
    out[clear, 3] = bottom[clear, 3]
    if np.any(partial):
        tf = top[partial].astype(np.float32) / 255.0
        bf = bottom[partial].astype(np.float32) / 255.0
        atp = tf[..., 3:4]
        abp = bf[..., 3:4]
        ao = atp + abp * (1.0 - atp)
        rgb_premul = tf[..., :3] * atp + bf[..., :3] * abp * (1.0 - atp)
        rgb = np.divide(rgb_premul, np.maximum(ao, 1e-8), out=np.zeros_like(rgb_premul), where=ao > 1e-8)
        blended = np.concatenate([rgb, ao], axis=1)
        out[partial] = np.clip(np.rint(blended * 255.0), 0, 255).astype(np.uint8)
    return out


def _registration_gate(registration: RegistrationResult, cfg: TransparentBubbleRevealConfig) -> tuple[bool, str]:
    if float(registration.confidence) < float(cfg.min_registration_confidence):
        return False, "registration_confidence"
    if not np.isfinite(float(registration.reprojection_error)) or float(registration.reprojection_error) > float(cfg.max_reprojection_error):
        return False, "registration_reprojection_error"
    if float(registration.inlier_ratio) < float(cfg.min_inlier_ratio):
        return False, "registration_inlier_ratio"
    return True, "ok"


def _valid_warp_mask(source: np.ndarray, registration: RegistrationResult) -> np.ndarray:
    h, w = source.shape[:2]
    tw, th = registration.target_size
    src = np.full((h, w), 255, dtype=np.uint8)
    return cv2.warpPerspective(
        src, registration.matrix, (int(tw), int(th)), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )


def _pseudo_target_text_blocks(target: np.ndarray) -> list[TextBlock]:
    """Create OCR-free printed-text seeds in TARGET coordinates.

    These seeds contain no recognized text. They exist only to let the mature
    seeded-white geometry detector find a white container. No SOURCE pixel is
    consulted here.
    """
    from .source_detectors import _cluster_text_components, _compact_character_components

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    comps = _compact_character_components(gray)
    groups = _cluster_text_components(gray, comps, 2)
    blocks: list[TextBlock] = []
    for i, group in enumerate(groups[:96]):
        x0, y0, x1, y1 = [float(v) for v in group["bbox"]]
        blocks.append(TextBlock(
            id=f"target-seed-{i:04d}",
            polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            text="", confidence=0.70, kind="unknown",
            meta={"backend": "target_non_ocr_text_seed", "target_only": True},
        ))
    return blocks


def _clone_as_target_only(rows: list[BubbleInstance], backend: str) -> list[BubbleInstance]:
    out: list[BubbleInstance] = []
    for i, b in enumerate(rows):
        meta = dict(b.meta or {})
        meta.pop("source_only", None)
        meta["target_only"] = True
        meta["backend"] = backend
        out.append(BubbleInstance(
            id=f"tbr-{backend}-{i:04d}", polygon=list(b.polygon), confidence=float(b.confidence),
            kind=b.kind, block_ids=[],
            mask=None if b.mask is None else b.mask.copy(),
            safe_mask=None if b.safe_mask is None else b.safe_mask.copy(),
            meta=meta,
        ))
    return out


def _target_text_seed_blocks(
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
    bubble_cfg: BubbleConfig | None = None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> tuple[list[TextBlock], list[dict[str, Any]]]:
    """Locate TARGET text geometry without violating the Detector Policy.

    Koharu text/SFX seeds are used when Koharu is the selected primary or an
    eligible auxiliary that is being consulted after the primary.  The mode
    contract remains strictly 0-OCR; legacy ``paddle`` values downgrade to the
    cheap heuristic instead of instantiating OCR.
    """
    requested = str(getattr(cfg, "target_text_seed_backend", "auto") or "auto").strip().lower()
    audit: list[dict[str, Any]] = []
    blocks: list[TextBlock] = []

    def add_unique(rows: list[TextBlock], backend: str) -> None:
        for row in rows:
            x0, y0, x1, y1 = row.bbox
            if x1 <= x0 or y1 <= y0:
                continue
            duplicate = False
            for old in blocks:
                ox0, oy0, ox1, oy1 = old.bbox
                ix = max(0.0, min(x1, ox1) - max(x0, ox0))
                iy = max(0.0, min(y1, oy1) - max(y0, oy0))
                inter = ix * iy
                area = min((x1-x0)*(y1-y0), (ox1-ox0)*(oy1-oy0))
                if area > 0 and inter / area >= 0.72:
                    duplicate = True
                    break
            if duplicate:
                continue
            meta = dict(row.meta or {})
            meta.update({"backend": backend, "target_only": True, "text_seed": True})
            blocks.append(TextBlock(
                id=f"target-text-{len(blocks):04d}", polygon=list(row.polygon),
                text=str(row.text or ""), confidence=float(row.confidence), kind=row.kind,
                reading_order=len(blocks), meta=meta,
            ))

    evidence = None
    koharu_enabled = bubble_cfg is None or policy_uses_koharu(bubble_cfg)
    if koharu_enabled:
        try:
            evidence = collect_koharu_layout_evidence(
                target, bubble_cfg, role="transparent_target_text_seed", image_path=target_path,
                cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=True,
            )
            if evidence.available:
                rows = evidence.text_blocks(include_sfx=True, backend_name="koharu_layout", target_only=True)
                add_unique(rows, "koharu_layout")
                audit.append({"backend": "koharu_layout", "status": "ok", "detected": len(rows), "phase": "primary"})
                if blocks and bubble_cfg is not None and koharu_is_primary(bubble_cfg):
                    fallback_name = "heuristic" if requested in {"auto", "heuristic"} else requested
                    if fallback_name != "koharu_layout":
                        audit.append({"backend": fallback_name, "status": "skipped_koharu_primary_sufficient", "detected": 0, "phase": "fallback"})
                    return blocks, audit
            else:
                audit.append({"backend": "koharu_layout", "status": "unavailable", "phase": "primary", **{k: v for k, v in evidence.diagnostics.items() if k != "shape"}})
        except Exception as exc:
            audit.append({"backend": "koharu_layout", "status": "unavailable", "phase": "primary", "error": str(exc)})
    else:
        audit.append({"backend": "koharu_layout", "status": "skipped_detector_policy", "phase": "primary"})

    if requested == "koharu_layout" and koharu_enabled:
        return blocks, audit

    fallback = "heuristic" if requested in {"auto", "heuristic"} else requested
    if fallback == "heuristic":
        try:
            heuristic = _pseudo_target_text_blocks(target)
            heuristic = heuristic[: max(1, int(getattr(cfg, "target_text_seed_max_candidates", 96)))]
            add_unique(heuristic, "heuristic_text_detector")
            audit.append({"backend": "heuristic", "status": "ok", "detected": len(heuristic), "phase": "fallback"})
        except Exception as exc:
            audit.append({"backend": "heuristic", "status": "unavailable", "phase": "fallback", "error": str(exc)})
    elif fallback == "paddle":
        # Transparent Reveal has a strict 0-OCR mode contract.  Older configs
        # could name Paddle here because this field pre-dated the mode contract;
        # do not let that legacy value silently instantiate an OCR engine.
        # Preserve recovery quality with the cheap geometry/text-contour seed
        # detector instead, and make the downgrade explicit in diagnostics.
        audit.append({
            "backend": "paddle",
            "status": "skipped_mode_contract_0_ocr",
            "detected": 0,
            "phase": "fallback",
        })
        try:
            heuristic = _pseudo_target_text_blocks(target)
            heuristic = heuristic[: max(1, int(getattr(cfg, "target_text_seed_max_candidates", 96)))]
            add_unique(heuristic, "heuristic_text_detector")
            audit.append({
                "backend": "heuristic",
                "status": "ok",
                "detected": len(heuristic),
                "phase": "fallback_after_ocr_contract",
            })
        except Exception as exc:
            audit.append({
                "backend": "heuristic",
                "status": "unavailable",
                "phase": "fallback_after_ocr_contract",
                "error": str(exc),
            })
    return blocks, audit

def _detect_target_text_contour_bubbles(
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
    bubble_cfg: BubbleConfig | None = None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> tuple[list[BubbleInstance], list[dict[str, Any]]]:
    """Recover speech-bubble interiors from TARGET text seeds and dark outlines.

    The old v2.0 path asked a white-region/bubble model to discover the container
    before it knew where the text was.  On a white wall, the wall and bubble can
    become the same bright component.  Here text is the anchor: search only a
    bounded ROI around each detected text group, then accept the smallest clean
    closed contour that encloses that group.  Burst/open containers continue to
    fall through to MangaLens/barrier/white backends.
    """
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    h, w = gray.shape
    seeds, seed_audit = _target_text_seed_blocks(
        target, cfg, bubble_cfg, stage_cache=stage_cache, cache_stats=cache_stats,
        target_path=target_path, cache_enabled=cache_enabled,
    )
    out: list[BubbleInstance] = []
    thresholds = [int(v) for v in list(getattr(cfg, "target_text_contour_thresholds", [205, 215, 225, 235]))]
    min_white = float(getattr(cfg, "target_text_contour_min_white_ratio", 0.68))
    max_dark = float(getattr(cfg, "target_text_contour_max_dark_ratio", 0.22))
    min_ratio = float(getattr(cfg, "target_text_contour_min_area_to_text_ratio", 1.20))
    max_ratio = float(getattr(cfg, "target_text_contour_max_area_to_text_ratio", 18.0))
    padding_ratio = float(getattr(cfg, "target_text_contour_padding_ratio", 1.20))
    max_seeds = max(1, int(getattr(cfg, "target_text_seed_max_candidates", 96)))
    accepted_seed_ids: list[str] = []

    for seed in seeds[:max_seeds]:
        tx0, ty0, tx1, ty1 = [int(round(v)) for v in seed.bbox]
        tx0 = max(0, min(w-1, tx0)); tx1 = max(tx0+1, min(w, tx1))
        ty0 = max(0, min(h-1, ty0)); ty1 = max(ty0+1, min(h, ty1))
        tw, th = max(1, tx1-tx0), max(1, ty1-ty0)
        text_area = max(1, tw * th)
        pad_x = max(64, int(round(max(tw * padding_ratio, min(240.0, th * 0.80)))))
        pad_y = max(56, int(round(max(th * 0.65, min(180.0, tw * 0.90)))))
        rx0, ry0 = max(0, tx0-pad_x), max(0, ty0-pad_y)
        rx1, ry1 = min(w, tx1+pad_x), min(h, ty1+pad_y)
        roi = gray[ry0:ry1, rx0:rx1]
        if roi.size == 0:
            continue
        cx = 0.5 * (tx0 + tx1) - rx0
        cy = 0.5 * (ty0 + ty1) - ry0
        best: tuple[float, np.ndarray, tuple[int,int,int,int], float, float, float, int, int] | None = None
        for thr in thresholds:
            dark = (roi < thr).astype(np.uint8) * 255
            for close_px in (1, 3, 5):
                work = dark
                if close_px > 1:
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
                    work = cv2.morphologyEx(work, cv2.MORPH_CLOSE, kernel)
                contours, _ = cv2.findContours(work, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    area = float(abs(cv2.contourArea(contour)))
                    ratio = area / text_area
                    if ratio < min_ratio or ratio > max_ratio:
                        continue
                    if cv2.pointPolygonTest(contour, (float(cx), float(cy)), False) < 0:
                        continue
                    bx, by, bw, bh = cv2.boundingRect(contour)
                    gx0, gy0, gx1, gy1 = bx+rx0, by+ry0, bx+rx0+bw, by+ry0+bh
                    # A contour clipped by the search ROI is usually wall/panel art,
                    # not the local speech bubble we are looking for.
                    if bx <= 1 or by <= 1 or bx+bw >= roi.shape[1]-1 or by+bh >= roi.shape[0]-1:
                        continue
                    if not (gx0 <= tx0+4 and gy0 <= ty0+4 and gx1 >= tx1-4 and gy1 >= ty1-4):
                        continue
                    local = np.zeros_like(roi, np.uint8)
                    cv2.drawContours(local, [contour], -1, 255, cv2.FILLED)
                    vals = roi[local > 0]
                    if vals.size < 64:
                        continue
                    white = float(np.mean(vals >= 220))
                    dark_ratio = float(np.mean(vals < 170))
                    if white < min_white or dark_ratio > max_dark:
                        continue
                    # Prefer a tight, bright enclosure; large white rooms/walls lose.
                    score = ratio + 5.0 * (1.0-white) + 2.0 * dark_ratio + close_px * 0.01
                    if best is None or score < best[0]:
                        full = np.zeros((h, w), np.uint8)
                        shifted = contour.copy()
                        shifted[:, 0, 0] += rx0; shifted[:, 0, 1] += ry0
                        cv2.drawContours(full, [shifted], -1, 255, cv2.FILLED)
                        best = (score, full, (gx0, gy0, gx1, gy1), white, dark_ratio, ratio, thr, close_px)
        if best is None:
            continue
        _score, mask, bb, white, dark_ratio, ratio, thr, close_px = best
        # Printed glyphs can physically touch/overlap a thin bubble outline. In a
        # threshold contour that connection appears as a notch into the interior
        # (the exact reason the sample burst left ``ねば`` behind). Regularise a
        # locally-selected speech container with its convex hull, but only when
        # the hull is a small inflation and remains bright/clean.
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        raw_area = max(1, int(cv2.countNonZero(mask)))
        hull = cv2.convexHull(contour)
        hull_mask = np.zeros_like(mask)
        cv2.drawContours(hull_mask, [hull], -1, 255, cv2.FILLED)
        hull_area = int(cv2.countNonZero(hull_mask))
        hull_vals = gray[hull_mask > 0]
        hull_white = float(np.mean(hull_vals >= 220)) if hull_vals.size else 0.0
        hull_dark = float(np.mean(hull_vals < 170)) if hull_vals.size else 1.0
        hull_applied = False
        if (
            hull_area <= int(raw_area * 1.24)
            and hull_white >= max(0.60, min_white - 0.06)
            and hull_dark <= max_dark + 0.04
        ):
            mask = hull_mask
            contour = hull
            white, dark_ratio = hull_white, hull_dark
            hull_applied = True
        if any(_mask_iou(mask, row.mask) >= 0.68 for row in out if row.mask is not None):
            continue
        eps = max(0.5, cv2.arcLength(contour, True) * 0.002)
        poly = [(float(x), float(y)) for x, y in cv2.approxPolyDP(contour, eps, True).reshape(-1, 2)]
        if len(poly) < 3:
            continue
        confidence = float(np.clip(0.78 + 0.18*white - 0.12*dark_ratio - 0.02*max(0.0, ratio-4.0), 0.58, 0.96))
        out.append(BubbleInstance(
            id=f"tbr-target-text-contour-{len(out):04d}", polygon=poly,
            confidence=confidence, kind="speech", block_ids=[seed.id], mask=mask,
            safe_mask=mask.copy(), meta={
                "backend": "target_text_contour", "target_only": True,
                "text_seed_backend": str((seed.meta or {}).get("backend", "unknown")),
                "text_bbox": [tx0, ty0, tx1, ty1], "container_bbox": list(bb),
                "white_ratio": white, "dark_ratio": dark_ratio,
                "area_to_text_ratio": ratio, "threshold": thr, "close_px": close_px,
                "convex_regularized": bool(hull_applied),
            },
        ))
        accepted_seed_ids.append(seed.id)

    audit = [*seed_audit, {
        "backend": "target_text_contour", "status": "ok", "text_seeds": len(seeds),
        "detected": len(out), "accepted_seed_ids": accepted_seed_ids,
    }]
    return out, audit

def _white_candidate_has_text_support(
    target: np.ndarray,
    row: BubbleInstance,
    seeds: list[TextBlock],
    cfg: TransparentBubbleRevealConfig,
) -> bool:
    """Reject white-room/artwork components that are not a local text container."""
    mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
    if mask is None or cv2.countNonZero(mask) <= 0:
        return False
    area = int(cv2.countNonZero(mask))
    page_area = max(1, int(mask.shape[0] * mask.shape[1]))
    # A fallback white component spanning a large room/page is exactly the v2.0
    # failure mode.  Keep the broad explicit backend available, but auto never
    # treats such a component as a local speech bubble.
    if area / page_area > 0.14:
        return False
    x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
    if bw / max(1, mask.shape[1]) > 0.58 or bh / max(1, mask.shape[0]) > 0.48:
        return False
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    vals = gray[mask > 0]
    if vals.size < 64:
        return False
    white = float(np.mean(vals >= 220))
    dark = float(np.mean(vals < 170))
    if white < float(getattr(cfg, "target_text_contour_min_white_ratio", 0.68)):
        return False
    if dark > float(getattr(cfg, "target_text_contour_max_dark_ratio", 0.22)):
        return False
    max_ratio = max(20.0, float(getattr(cfg, "target_text_contour_max_area_to_text_ratio", 18.0)) * 1.45)
    h, w = mask.shape
    for seed in seeds:
        x0, y0, x1, y1 = seed.bbox
        cx = int(np.clip(round((x0+x1)*0.5), 0, w-1))
        cy = int(np.clip(round((y0+y1)*0.5), 0, h-1))
        if mask[cy, cx] <= 0:
            continue
        text_area = max(1.0, (x1-x0)*(y1-y0))
        if area / text_area <= max_ratio:
            return True
    return False

def _candidate_target_text_presence(
    target: np.ndarray,
    row: BubbleInstance,
    seeds: list[TextBlock],
    cfg: TransparentBubbleRevealConfig,
) -> tuple[bool, dict[str, Any]]:
    """Verify that a TARGET candidate really contains text-like structure.

    Heuristic seeds are only *proposals* and must not count as independent proof.
    The verifier therefore checks compact glyph-like components for density,
    alignment and size consistency. Koharu/text-model metadata remains stronger
    evidence than the cheap heuristic fallback.
    """
    mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
    if mask is None or cv2.countNonZero(mask) <= 0:
        return False, {"reason": "empty_candidate_mask", "components": 0, "seed_support": 0}
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(mask_u8)
    if w <= 0 or h <= 0:
        return False, {"reason": "empty_candidate_bbox", "components": 0, "seed_support": 0}
    if _is_page_furniture_bbox((x, y, x + w, y + h), target.shape[:2], cfg):
        return False, {
            "reason": "page_furniture", "accepted": False, "components": 0,
            "seed_support": 0, "candidate_bbox": [int(x), int(y), int(x + w), int(y + h)],
        }
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    crop = gray[y:y+h, x:x+w]
    local_mask = mask_u8[y:y+h, x:x+w] > 0
    try:
        from .source_detectors import _compact_character_components
        comps = _compact_character_components(crop)
    except Exception:
        comps = []
    inside = []
    for cx0, cy0, cw, ch, c_area, (ccx, ccy) in comps:
        xi = int(np.clip(round(ccx), 0, max(0, local_mask.shape[1] - 1)))
        yi = int(np.clip(round(ccy), 0, max(0, local_mask.shape[0] - 1)))
        if local_mask[yi, xi]:
            inside.append((cx0, cy0, cw, ch, c_area, (ccx, ccy)))

    meta = dict(row.meta or {})
    text_bbox = meta.get("text_bbox")
    has_text_bbox = isinstance(text_bbox, (list, tuple)) and len(text_bbox) == 4
    seed_backend = str(meta.get("text_seed_backend", "") or "").strip().lower()
    backend = str(meta.get("backend", "") or "").strip().lower()
    strong_seed_backend = seed_backend in {"koharu_layout", "ocr", "paddle", "apple", "apple_live_text", "manga_ocr", "baberu_ocr", "ocr48px"}
    strong_text_bbox = bool(has_text_bbox and (strong_seed_backend or backend == "koharu_layout"))

    seed_support_ids: list[str] = []
    strong_seed_support_ids: list[str] = []
    for seed in seeds:
        sx0, sy0, sx1, sy1 = [int(round(v)) for v in seed.bbox]
        sx0 = max(0, min(mask.shape[1], sx0)); sx1 = max(0, min(mask.shape[1], sx1))
        sy0 = max(0, min(mask.shape[0], sy0)); sy1 = max(0, min(mask.shape[0], sy1))
        if sx1 <= sx0 or sy1 <= sy0:
            continue
        scx = int(np.clip(round((sx0 + sx1) * 0.5), 0, mask.shape[1] - 1))
        scy = int(np.clip(round((sy0 + sy1) * 0.5), 0, mask.shape[0] - 1))
        overlap = int(np.count_nonzero((mask_u8[sy0:sy1, sx0:sx1] > 0)))
        seed_area = max(1, (sx1 - sx0) * (sy1 - sy0))
        if mask_u8[scy, scx] > 0 or overlap / seed_area >= 0.18:
            seed_support_ids.append(str(seed.id))
            sb = str((seed.meta or {}).get("backend", "") or "").strip().lower()
            if sb in {"koharu_layout", "ocr", "paddle", "apple", "apple_live_text", "manga_ocr", "baberu_ocr", "ocr48px"}:
                strong_seed_support_ids.append(str(seed.id))

    comp_count = len(inside)
    widths = np.asarray([c[2] for c in inside], dtype=np.float32)
    heights = np.asarray([c[3] for c in inside], dtype=np.float32)
    centers = np.asarray([[c[5][0], c[5][1]] for c in inside], dtype=np.float32) if inside else np.empty((0, 2), np.float32)
    if comp_count >= 2:
        sx = float(np.std(centers[:, 0])); sy = float(np.std(centers[:, 1]))
        line_ratio = float(min(sx, sy) / max(1e-6, max(sx, sy)))
        width_cv = float(np.std(widths) / max(1.0, float(np.mean(widths))))
        height_cv = float(np.std(heights) / max(1.0, float(np.mean(heights))))
    else:
        line_ratio = 1.0
        width_cv = height_cv = 1.0

    area = int(cv2.countNonZero(mask_u8))
    min_components = max(1, int(getattr(cfg, "target_text_presence_min_components", 2)))
    # Cheap heuristic seeds are not independent proof. Real text typically forms
    # either many compact components, a clearly aligned row/column, or a small
    # multi-glyph group with consistent glyph height.
    dense = comp_count >= max(8, min_components + 5)
    aligned = comp_count >= max(3, min_components) and line_ratio <= 0.50 and height_cv <= 0.60
    compact_grid = comp_count >= max(5, min_components + 2) and height_cv <= 0.30 and width_cv <= 0.58
    # Stylized SFX may merge into one large outlined component. Admit that shape
    # without lowering the general threshold for face/skin/hair false positives.
    single_effect = False
    if comp_count == 1 and has_text_bbox and inside:
        _cx, _cy, cw, ch, ca, _cc = inside[0]
        single_effect = bool(
            ca >= 180 and min(cw, ch) >= 18 and max(cw / max(1, ch), ch / max(1, cw)) <= 2.8
            and backend in {"text_seed_white_container", "text_seed_fallback", "target_text_contour"}
        )
    strong_semantic = bool(strong_text_bbox or strong_seed_support_ids)
    accepted = bool(strong_semantic and comp_count >= 1) or dense or aligned or compact_grid or single_effect
    reason = "insufficient_target_text_presence"
    if strong_semantic and comp_count >= 1:
        reason = "strong_semantic_text_support"
    elif dense:
        reason = "dense_text_components"
    elif aligned:
        reason = "aligned_text_components"
    elif compact_grid:
        reason = "compact_multi_glyph_text"
    elif single_effect:
        reason = "single_stylized_effect_component"
    return accepted, {
        "reason": reason,
        "accepted": accepted,
        "components": comp_count,
        "minimum_components": min_components,
        "seed_support": len(seed_support_ids),
        "strong_seed_support": len(strong_seed_support_ids),
        "seed_ids": seed_support_ids[:12],
        "has_text_bbox": bool(has_text_bbox),
        "strong_text_bbox": bool(strong_text_bbox),
        "line_ratio": float(line_ratio),
        "width_cv": float(width_cv),
        "height_cv": float(height_cv),
        "dense": bool(dense),
        "aligned": bool(aligned),
        "compact_grid": bool(compact_grid),
        "single_effect": bool(single_effect),
        "candidate_area": area,
        "candidate_bbox": [int(x), int(y), int(x + w), int(y + h)],
    }


def _text_has_semantic_character(value: str) -> bool:
    for ch in str(value or ""):
        if ch.isalnum():
            return True
        code = ord(ch)
        if 0x3040 <= code <= 0x30FF or 0x3400 <= code <= 0x9FFF or 0xFF66 <= code <= 0xFF9D:
            return True
    return False


def _candidate_ocr_text_presence(
    target: np.ndarray,
    row: BubbleInstance,
    cfg: TransparentBubbleRevealConfig,
    ocr_backend: Any | None,
    *,
    target_path: str | None = None,
) -> tuple[bool | None, dict[str, Any]]:
    """Optional OCR-only presence check; text content is never used downstream."""
    if not bool(getattr(cfg, "target_text_presence_ocr_enabled", False)):
        return None, {"status": "disabled", "accepted": None}
    if ocr_backend is None:
        return None, {"status": "unavailable", "accepted": None, "reason": "no_ocr_backend"}
    tb = (row.meta or {}).get("text_bbox")
    if isinstance(tb, (list, tuple)) and len(tb) == 4:
        x0, y0, x1, y1 = [int(round(float(v))) for v in tb]
    else:
        x0, y0, x1, y1 = _bbox_int(list(row.polygon), target.shape[:2])
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    padx = max(4, min(18, int(round(bw * 0.10))))
    pady = max(4, min(18, int(round(bh * 0.10))))
    x0 = max(0, x0 - padx); x1 = min(target.shape[1], x1 + padx)
    y0 = max(0, y0 - pady); y1 = min(target.shape[0], y1 + pady)
    if x1 <= x0 or y1 <= y0:
        return False, {"status": "ok", "accepted": False, "reason": "empty_ocr_bbox"}
    try:
        blocks = ocr_backend.recognize_region(target, (x0, y0, x1, y1), image_path=target_path)
    except Exception as exc:
        return None, {"status": "unavailable", "accepted": None, "reason": "ocr_exception", "error": str(exc)}
    min_conf = float(getattr(cfg, "target_text_presence_ocr_min_confidence", 0.35))
    hits = []
    for block in blocks or []:
        text = str(getattr(block, "text", "") or "").strip()
        conf = float(getattr(block, "confidence", 0.0) or 0.0)
        if conf >= min_conf and _text_has_semantic_character(text):
            hits.append({"text": text[:32], "confidence": conf})
    accepted = bool(hits)
    return accepted, {
        "status": "ok",
        "accepted": accepted,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "min_confidence": min_conf,
        "recognized_blocks": len(blocks or []),
        "semantic_hits": hits[:6],
    }



def _source_translation_evidence_for_candidate(
    bubble: BubbleInstance,
    aligned_source: np.ndarray,
    target: np.ndarray,
    base_mask: np.ndarray,
    *,
    prefer_white_container: bool,
) -> tuple[bool, dict[str, Any]]:
    if prefer_white_container:
        accepted, diag = _white_container_source_translation_evidence(aligned_source, target, base_mask)
        return bool(accepted), {"reason": "white_container_translation_evidence", **diag}
    return _bubble_translation_evidence(bubble, aligned_source, target)


def _detect_backend(
    target: np.ndarray,
    backend: str,
    bubble_cfg: BubbleConfig,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> list[BubbleInstance]:
    backend = str(backend or "auto").strip().lower()
    if backend == "koharu_layout":
        evidence = collect_koharu_layout_evidence(
            target, bubble_cfg, role="transparent_target_bubbles", image_path=target_path,
            cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=False,
        )
        return _clone_as_target_only(evidence.bubble_instances(backend_name="koharu_layout", target_only=True), "koharu_layout")
    if backend == "mangalens":
        return _clone_as_target_only(detect_mangalens_bubbles(target, [], bubble_cfg), "mangalens")
    if backend == "seeded_white":
        blocks = _pseudo_target_text_blocks(target)
        return _clone_as_target_only(detect_seeded_white_bubbles(target, blocks, bubble_cfg), "seeded_white")
    if backend in {"unseeded_white", "unseeded"}:
        return _clone_as_target_only(detect_unseeded_white_containers(target, bubble_cfg, prefix="tbr-white"), "unseeded_white")
    if backend == "rtdetr_v2":
        from .source_detectors import detect_source_rtdetr_v2
        return _clone_as_target_only(detect_source_rtdetr_v2(target, MaskReplaceConfig(), bubble_cfg, existing=[]), "rtdetr_v2")
    if backend == "sam2":
        from .source_detectors import detect_source_sam2
        return _clone_as_target_only(detect_source_sam2(target, MaskReplaceConfig(), bubble_cfg, existing=[]), "sam2")
    raise ValueError(f"Unsupported transparent bubble backend: {backend}")


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0; bb = b > 0
    union = int(np.count_nonzero(aa | bb))
    return float(np.count_nonzero(aa & bb)) / max(1, union)


def _detect_target_bubbles(
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
    bubble_cfg: BubbleConfig | None = None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> tuple[list[BubbleInstance], list[dict[str, Any]]]:
    bcfg = bubble_cfg or BubbleConfig()
    primary = primary_detector(bcfg)
    strategy = detector_strategy(bcfg)
    accepted: list[BubbleInstance] = []
    audit: list[dict[str, Any]] = []

    authority = None
    if koharu_is_primary(bcfg):
        try:
            authority = collect_koharu_layout_evidence(
                target, bcfg, role="transparent_target_bubbles", image_path=target_path,
                cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=True,
            )
        except Exception as exc:
            audit.append({"backend":"koharu_layout","status":"unavailable","phase":"authority","error":str(exc)})

    # Single selected primary gets the first decision.  Explicitly selecting an
    # expensive model is permission to run it; auxiliaries remain policy-gated.
    try:
        if primary == "koharu_layout":
            if authority is not None and authority.available:
                rows = _clone_as_target_only(
                    authority.bubble_instances(backend_name="koharu_layout", target_only=True),
                    "koharu_layout",
                )
            else:
                rows = []
        else:
            if stage_cache is not None and target_path:
                rows = primary_bubbles_cached(
                    "target", target, target_path, bubble_config=bcfg, cache=stage_cache,
                    cache_enabled=bool(cache_enabled), stats=cache_stats if cache_stats is not None else {},
                )
                rows = _clone_as_target_only(rows, primary)
            else:
                rows = _detect_backend(
                    target, primary, bcfg, stage_cache=stage_cache, cache_stats=cache_stats,
                    target_path=target_path, cache_enabled=cache_enabled,
                )
        before = len(accepted)
        for row in sorted(rows, key=lambda b: float(b.confidence), reverse=True):
            if float(row.confidence) < float(cfg.min_bubble_confidence):
                continue
            mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
            row.mask = mask
            if cv2.countNonZero(mask) <= 0 or not _auto_candidate_safe(target, row, primary):
                continue
            if any(_mask_iou(mask, ex.mask if ex.mask is not None else rasterize_polygon(ex.polygon, target.shape[:2])) >= 0.62 for ex in accepted):
                continue
            accepted.append(row)
        audit.append({
            "backend": primary, "status": "ok", "detected": len(rows),
            "accepted_new": len(accepted)-before, "phase": "primary", "strategy": strategy,
        })
    except Exception as exc:
        audit.append({"backend": primary, "status": "unavailable", "phase": "primary", "error": str(exc)})

    primary_sufficient = bool(accepted)
    if not should_run_auxiliaries(bcfg, primary_sufficient=primary_sufficient):
        status = "skipped_koharu_primary_sufficient" if primary == "koharu_layout" and primary_sufficient else "skipped_by_detector_policy"
        audit.append({"backend":"auxiliary_chain","status":status,"phase":"fallback","strategy":strategy})
        return accepted, audit

    chain = transparent_auxiliary_backends(bcfg)
    support_seeds: list[TextBlock] = []
    try:
        support_seeds, _ = _target_text_seed_blocks(
            target, cfg, bcfg, stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=target_path, cache_enabled=cache_enabled,
        )
    except Exception:
        support_seeds = []

    for backend in chain:
        if backend == primary:
            continue
        try:
            extra_audit: list[dict[str, Any]] = []
            if backend == "target_text_contour":
                rows, extra_audit = _detect_target_text_contour_bubbles(
                    target, cfg, bcfg, stage_cache=stage_cache, cache_stats=cache_stats,
                    target_path=target_path, cache_enabled=cache_enabled,
                )
            elif backend == "koharu_layout":
                # Koharu auxiliary is loaded lazily only after the selected primary
                # has completed and the policy actually requests fallback.
                aux_authority = authority
                if aux_authority is None:
                    aux_authority = collect_koharu_layout_evidence(
                        target, bcfg, role="transparent_target_bubbles_aux", image_path=target_path,
                        cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=True,
                    )
                rows = _clone_as_target_only(
                    aux_authority.bubble_instances(backend_name="koharu_layout", target_only=True) if aux_authority.available else [],
                    "koharu_layout",
                )
            else:
                rows = _detect_backend(
                    target, backend, bcfg, stage_cache=stage_cache, cache_stats=cache_stats,
                    target_path=target_path, cache_enabled=cache_enabled,
                )
            before = len(accepted)
            authority_audit: list[dict[str, Any]] = []
            # Primary priority is strict.  Only Koharu-as-primary has a semantic
            # panel/artwork PROTECT map capable of vetoing auxiliary candidates.
            if koharu_is_primary(bcfg) and authority is not None:
                rows, authority_audit = filter_candidates_by_layout_authority(
                    rows, authority, target.shape[:2], region_kind="bubble", cfg=None,
                    allow_unknown=True, meta_key="koharu_layout_authority",
                )
            for row in sorted(rows, key=lambda b: float(b.confidence), reverse=True):
                if backend in {"seeded_white", "unseeded_white", "unseeded"} and not _white_candidate_has_text_support(target, row, support_seeds, cfg):
                    continue
                if float(row.confidence) < float(cfg.min_bubble_confidence):
                    continue
                mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
                row.mask = mask
                if cv2.countNonZero(mask) <= 0 or not _auto_candidate_safe(target, row, backend):
                    continue
                if any(_mask_iou(mask, ex.mask if ex.mask is not None else rasterize_polygon(ex.polygon, target.shape[:2])) >= 0.62 for ex in accepted):
                    continue
                accepted.append(row)
            audit.extend(extra_audit)
            audit.append({
                "backend": backend, "status": "ok", "detected": len(rows),
                "accepted_new": len(accepted)-before, "phase": "fallback",
                "authority_rejected": sum(1 for row in authority_audit if not row.get("accepted")),
            })
            if strategy != STRATEGY_ALWAYS and len(accepted) > before:
                break
        except Exception as exc:
            audit.append({"backend": backend, "status": "unavailable", "phase": "fallback", "error": str(exc)})
    return accepted, audit

def _mask_pixel_count(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(cv2.countNonZero((mask > 0).astype(np.uint8)))


def _mask_pair_overlap_ratio(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
    if mask_a is None or mask_b is None:
        return 0.0
    a = (mask_a > 0)
    b = (mask_b > 0)
    inter = int(np.count_nonzero(a & b))
    if inter <= 0:
        return 0.0
    denom = max(1, min(int(np.count_nonzero(a)), int(np.count_nonzero(b))))
    return float(inter / denom)


def _collect_protected_white_container_masks(
    bubbles: list[BubbleInstance],
    target: np.ndarray,
) -> list[np.ndarray]:
    protected: list[np.ndarray] = []
    for row in bubbles:
        meta = dict(row.meta or {})
        backend = str(meta.get("backend", "")).strip().lower()
        mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
        if mask is None or _mask_pixel_count(mask) <= 0:
            continue
        if backend in {"target_text_contour", "koharu_layout", "seeded_white", "unseeded_white", "text_seed_white_container", "text_seed_white_rect"} or _bubble_is_white(target, mask):
            protected.append((mask > 0).astype(np.uint8) * 255)
    return protected


def _seed_fallback_hits_protected_container(
    row: BubbleInstance,
    protected_masks: list[np.ndarray],
) -> tuple[bool, dict[str, Any]]:
    mask = row.mask
    if mask is None or _mask_pixel_count(mask) <= 0 or not protected_masks:
        return False, {"reason": "no_overlap", "overlap_ratio": 0.0}
    best = 0.0
    for ex in protected_masks:
        best = max(best, _mask_pair_overlap_ratio(mask, ex))
    hit = bool(best >= 0.18)
    return hit, {"reason": "protected_white_container_overlap" if hit else "no_overlap", "overlap_ratio": float(best)}


def _seeded_container_quality_ok(
    target: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    seed_bbox: tuple[int, int, int, int],
    cfg: TransparentBubbleRevealConfig | None = None,
) -> tuple[bool, dict[str, Any]]:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    sx0, sy0, sx1, sy1 = [int(v) for v in seed_bbox]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    seed_area = max(1, (sx1 - sx0) * (sy1 - sy0))
    bbox_area = max(1, bw * bh)
    aspect = float(max(bw, bh) / max(1, min(bw, bh)))
    expand_ratio = float(bbox_area / seed_area)
    profile = _neutral_paper_profile(target, mask)
    min_fill = float(getattr(cfg, "seeded_container_quality_min_fill_ratio", 0.56)) if cfg is not None else 0.56
    min_compact = float(getattr(cfg, "seeded_container_quality_min_compactness", 0.31)) if cfg is not None else 0.31
    ok = bool(
        profile["bright_ratio"] >= 0.72
        and profile["mean_bgr"] >= 192.0
        and profile["low_sat_p75"] <= 42.0
        and profile["bbox_fill_ratio"] >= min_fill
        and profile["compactness"] >= min_compact
        and aspect <= 4.6
        and expand_ratio <= 7.2
    )
    return ok, {
        "accepted": ok,
        "bbox": [int(x0), int(y0), int(x1), int(y1)],
        "seed_bbox": [int(sx0), int(sy0), int(sx1), int(sy1)],
        "aspect_ratio": float(aspect),
        "bbox_expand_ratio": float(expand_ratio),
        "profile": {k: float(v) for k, v in profile.items()},
    }


def _seed_evidence_region(seed: TextBlock, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    x0, y0, x1, y1 = [int(round(v)) for v in seed.bbox]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad = max(10, min(28, int(round(max(bw, bh) * 0.20))))
    x0 = max(0, x0 - pad); x1 = min(w, x1 + pad)
    y0 = max(0, y0 - pad); y1 = min(h, y1 + pad)
    region = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        region[y0:y1, x0:x1] = 255
    return region


def _translation_evidence_for_seed(
    aligned_source: np.ndarray,
    target: np.ndarray,
    seed: TextBlock,
) -> tuple[bool, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Verify that an uncovered TARGET text seed is actually translated text.

    Blind TARGET-only seed fallback caused v2.0.15 to classify headers, SFX,
    facial features, sweat drops and line art as text.  Cross-edition evidence
    solves that ambiguity: a real translated region has substantial SOURCE-only
    *and* TARGET-only text ink, while unchanged SFX/art is mostly common structure.
    """
    region = _seed_evidence_region(seed, target.shape[:2])
    if cv2.countNonZero(region) <= 0:
        return False, {"reason": "empty_seed_region"}, region, np.zeros_like(region), np.zeros_like(region)
    source_mask, target_mask, diff_diag = changed_text_masks(
        aligned_source, target, region,
        tolerance_px=2,
        min_unique_ratio=0.045,
        max_component_fraction=0.09,
    )
    region_pixels = max(1, int(cv2.countNonZero(region)))
    source_pixels = int(cv2.countNonZero(source_mask))
    target_pixels = int(cv2.countNonZero(target_mask))
    source_ratio = float(source_pixels / region_pixels)
    target_ratio = float(target_pixels / region_pixels)

    # Require the TARGET-specific ink to overlap the seed itself; this prevents
    # a nearby registration mismatch from legitimising an unrelated art seed.
    seed_mask = np.zeros_like(region)
    x0, y0, x1, y1 = [int(round(v)) for v in seed.bbox]
    x0 = max(0, min(region.shape[1], x0)); x1 = max(0, min(region.shape[1], x1))
    y0 = max(0, min(region.shape[0], y0)); y1 = max(0, min(region.shape[0], y1))
    if x1 > x0 and y1 > y0:
        seed_mask[y0:y1, x0:x1] = 255
    seed_support = cv2.dilate(seed_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    target_near_seed = int(np.count_nonzero((target_mask > 0) & seed_support))
    target_seed_support_ratio = float(target_near_seed / max(1, target_pixels))

    # Calibrated on the supplied 049 pair: all genuine translated columns are
    # comfortably above these ratios, while ground texture, wall detail, SFX and
    # the eye/face false positives fall below at least one side of the gate.
    accepted = bool(
        source_pixels >= 35
        and target_pixels >= 25
        and source_ratio >= 0.025
        and target_ratio >= 0.015
        and target_near_seed >= 18
        and target_seed_support_ratio >= 0.28
    )
    diag = {
        "accepted": accepted,
        "source_unique_pixels": source_pixels,
        "target_unique_pixels": target_pixels,
        "region_pixels": region_pixels,
        "source_unique_ratio": source_ratio,
        "target_unique_ratio": target_ratio,
        "target_near_seed_pixels": target_near_seed,
        "target_seed_support_ratio": target_seed_support_ratio,
        "diff": diff_diag,
    }
    return accepted, diag, region, source_mask, target_mask


def _attach_text_seed_bbox_to_containers(
    bubbles: list[BubbleInstance],
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
    bubble_cfg: BubbleConfig | None = None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> tuple[list[BubbleInstance], list[dict[str, Any]]]:
    """Attach tight TARGET text geometry to geometry-only bubble candidates."""
    try:
        seeds, seed_audit = _target_text_seed_blocks(
            target, cfg, bubble_cfg, stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=target_path, cache_enabled=cache_enabled,
        )
    except Exception as exc:
        return bubbles, [{"backend": "container_text_seed_attach", "status": "unavailable", "error": str(exc)}]
    h, w = target.shape[:2]
    attached = 0
    rows: list[BubbleInstance] = []
    for bubble in bubbles:
        meta = dict(bubble.meta or {})
        mask = bubble.mask if bubble.mask is not None else rasterize_polygon(bubble.polygon, target.shape[:2])
        if mask is None or cv2.countNonZero(mask) <= 0:
            rows.append(bubble)
            continue
        bx, by, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
        bubble_bbox_area = max(1, bw * bh)
        existing_tb = meta.get("text_bbox")
        has_existing = isinstance(existing_tb, (list, tuple)) and len(existing_tb) == 4
        if has_existing:
            ex0, ey0, ex1, ey1 = [float(v) for v in existing_tb]
            # Existing contour text_bbox is often only one vertical column. Search
            # nearby inside the same container for sibling columns before fixing the
            # destructive text corridor. This is the real-pair failure that left
            # 聞き出さねば behind while only clearing ……!!.
            near_x = max(20.0, float(bw) * 0.30)
            near_y = max(18.0, float(bh) * 0.14)
        selected: list[TextBlock] = []
        for seed in seeds:
            x0, y0, x1, y1 = seed.bbox
            cx = int(np.clip(round((x0 + x1) * 0.5), 0, w - 1))
            cy = int(np.clip(round((y0 + y1) * 0.5), 0, h - 1))
            center_inside = bool(mask[cy, cx] > 0)
            seed_area = max(1.0, (x1 - x0) * (y1 - y0))
            if seed_area / bubble_bbox_area > 0.48:
                continue
            if has_existing:
                # Keep sibling text columns that share the same vertical band or
                # sit immediately next to the original seed. Reject distant text/
                # artwork even when the bright contour mask overshoots.
                x_gap = max(0.0, max(ex0 - x1, x0 - ex1))
                y_gap = max(0.0, max(ey0 - y1, y0 - ey1))
                y_overlap = max(0.0, min(ey1, y1) - max(ey0, y0))
                min_h = max(1.0, min(ey1 - ey0, y1 - y0))
                same_band = (y_overlap / min_h) >= 0.22
                if not ((x_gap <= near_x and y_gap <= near_y) or same_band):
                    continue
                if not center_inside:
                    # A bright contour mask can have deep notches/holes exactly where
                    # the text sits. Do not discard a nearby sibling column only
                    # because its centre lands in that hole; require only that the
                    # seed still overlaps the accepted container bbox materially.
                    ix0 = max(float(bx), float(x0)); iy0 = max(float(by), float(y0))
                    ix1 = min(float(bx + bw), float(x1)); iy1 = min(float(by + bh), float(y1))
                    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
                    if (inter / seed_area) < 0.35:
                        continue
            elif not center_inside:
                continue
            selected.append(seed)
        if selected:
            x0 = int(max(0, min(s.bbox[0] for s in selected)))
            y0 = int(max(0, min(s.bbox[1] for s in selected)))
            x1 = int(min(w, max(s.bbox[2] for s in selected)))
            y1 = int(min(h, max(s.bbox[3] for s in selected)))
            if has_existing:
                # Never destructively shrink a contour-provided text corridor just
                # because the seed-attach pass only saw one of several sibling text
                # blocks. This keeps the whole-page transparent route independent
                # from seed fallback quirks.
                x0 = int(max(0, min(x0, ex0)))
                y0 = int(max(0, min(y0, ey0)))
                x1 = int(min(w, max(x1, ex1)))
                y1 = int(min(h, max(y1, ey1)))
            if x1 > x0 and y1 > y0:
                meta["text_bbox"] = [x0, y0, x1, y1]
                meta["text_seed_ids"] = [s.id for s in selected]
                meta["text_seed_attached"] = True
                attached += 1
        bubble.meta = meta
        rows.append(bubble)
    return rows, [*seed_audit, {
        "backend": "container_text_seed_attach", "status": "ok",
        "bubble_count": len(bubbles), "attached": attached,
    }]


def _add_verified_seed_fallbacks(
    bubbles: list[BubbleInstance],
    aligned_source: np.ndarray,
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
    bubble_cfg: BubbleConfig | None = None,
    *,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
) -> tuple[list[BubbleInstance], list[dict[str, Any]]]:
    if str(cfg.bubble_backend or "auto").strip().lower() != "auto":
        return bubbles, []
    try:
        seeds, seed_audit = _target_text_seed_blocks(
            target, cfg, bubble_cfg, stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=target_path, cache_enabled=cache_enabled,
        )
    except Exception as exc:
        return bubbles, [{"backend": "text_seed_fallback", "status": "unavailable", "error": str(exc)}]
    out = list(bubbles)
    audit: list[dict[str, Any]] = [*seed_audit]
    authority = None
    if bubble_cfg is None or koharu_is_primary(bubble_cfg):
        try:
            authority = collect_koharu_layout_evidence(
                target, bubble_cfg, role="transparent_seed_fallback_authority", image_path=target_path,
                cache=stage_cache, cache_enabled=cache_enabled, stats=cache_stats, allow_missing=True,
            )
        except Exception:
            authority = None
    added = 0
    protected_masks = _collect_protected_white_container_masks(out, target) if bool(getattr(cfg, "seed_fallback_protect_detected_white_bubbles", True)) else []
    for seed in seeds:
        if _accepted_covers_seed(out, seed, target.shape[:2]):
            continue
        if not _seed_fallback_candidate_safe(target, seed, cfg):
            audit.append({
                "backend": "text_seed_fallback", "status": "seed_rejected_geometry",
                "seed_id": str(seed.id), "bbox": [int(round(v)) for v in seed.bbox],
            })
            continue
        ok, evidence, evidence_region, source_unique, target_unique = _translation_evidence_for_seed(aligned_source, target, seed)
        row = _seed_fallback_bubble(seed, target, cfg)
        protected_hit, protected_diag = _seed_fallback_hits_protected_container(row, protected_masks)
        if protected_hit:
            audit.append({
                "backend":"text_seed_fallback", "status":"seed_rejected_protected_container_overlap",
                "seed_id":str(seed.id), "bbox":[int(round(v)) for v in seed.bbox], **protected_diag,
            })
            continue
        authority_decision = classify_layout_authority(
            authority, row, target.shape[:2], region_kind="bubble", cfg=None,
        )
        if authority_decision.state == "PROTECT":
            audit.append({
                "backend":"text_seed_fallback", "status":"seed_rejected_koharu_protect",
                "seed_id":str(seed.id), "bbox":[int(round(v)) for v in seed.bbox],
                **authority_decision.to_dict(),
            })
            continue
        meta = dict(row.meta or {})
        meta.update({
            "translation_evidence": evidence,
            "translation_evidence_optional": not bool(getattr(cfg, "require_source_translation_evidence", False)),
            "verified_target_text_mask": target_unique.copy(),
            "verified_source_text_mask": source_unique.copy(),
            "koharu_layout_authority": authority_decision.to_dict(),
        })
        row.meta = meta
        out.append(row)
        added += 1
    audit.append({
        "backend": "text_seed_fallback", "status": "ok",
        "seed_count": len(seeds), "accepted_new": added,
        "policy": "text_seed_geometry_source_gate" if bool(getattr(cfg, "require_source_translation_evidence", False)) else "text_seed_geometry_no_source_gate",
    })
    return out, audit


def _tight_text_bbox_support(bubble: BubbleInstance, shape: tuple[int, int], pad: int = 4) -> np.ndarray:
    meta = dict(bubble.meta or {})
    tb = meta.get("text_bbox")
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    if not isinstance(tb, (list, tuple)) or len(tb) != 4:
        return out
    x0, y0, x1, y1 = [int(round(float(v))) for v in tb]
    p = max(0, int(pad))
    x0 = max(0, min(w, x0 - p)); x1 = max(0, min(w, x1 + p))
    y0 = max(0, min(h, y0 - p)); y1 = max(0, min(h, y1 + p))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def _region_text_bbox_support(region: TransparentBubbleRegion, shape: tuple[int, int], pad: int = 8) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), np.uint8)
    tb = region.text_bbox
    if not isinstance(tb, (list, tuple)) or len(tb) != 4:
        return out
    x0, y0, x1, y1 = [int(round(float(v))) for v in tb]
    p = max(0, int(pad))
    x0 = max(0, min(w, x0 - p)); x1 = max(0, min(w, x1 + p))
    y0 = max(0, min(h, y0 - p)); y1 = max(0, min(h, y1 + p))
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def _target_clear_gate_for_region(
    region: TransparentBubbleRegion,
    envelope: np.ndarray,
    forced_mask: np.ndarray,
) -> np.ndarray:
    """Allow extra TARGET cleanup only in a tight text corridor.

    This is wider than the verified preclear mask so residual JP antialias and
    slightly shifted strokes can still be removed, but it is far narrower than
    the whole coloured/effect container and therefore cannot wash out the
    container background.
    """
    gate = np.zeros_like(envelope)
    if region.text_bbox is not None:
        support = _region_text_bbox_support(region, envelope.shape, pad=10)
        gate = cv2.bitwise_or(gate, support)
    if cv2.countNonZero(forced_mask) > 0:
        gate = cv2.bitwise_or(gate, forced_mask)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        gate = cv2.dilate(gate, k)
    gate = cv2.bitwise_and(gate, envelope)
    return gate


def _tight_verified_text_ink_mask(target: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Complete compact printed ink inside a *verified tight text bbox*.

    The generic text-only selector intentionally drops large/odd components to
    protect artwork, but that can omit a full kanji such as 来.  Once the bbox is
    independently verified as text, use a much more permissive component rule
    and reject only obvious long container/panel rules.
    """
    use = support > 0
    out = np.zeros_like(support)
    if not np.any(use):
        return out
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    vals = gray[use]
    bg = float(np.percentile(vals, 76.0))
    spread = float(np.percentile(vals, 88.0) - np.percentile(vals, 20.0))
    margin = max(20.0, min(58.0, 22.0 + spread * 0.16))
    thr = int(np.clip(bg - margin, 70, 215))
    dark = ((gray <= thr) & use).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    ys, xs = np.where(use)
    rw = max(1, int(xs.max() - xs.min() + 1)); rh = max(1, int(ys.max() - ys.min() + 1))
    region_area = max(1, int(np.count_nonzero(use)))
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 2:
            continue
        fill = float(area / max(1, bw * bh))
        span_x = float(bw / rw); span_y = float(bh / rh)
        horizontal_rule = span_x >= 0.72 and bh <= max(5, int(round(rh * 0.10))) and fill <= 0.45
        vertical_rule = span_y >= 0.72 and bw <= max(5, int(round(rw * 0.10))) and fill <= 0.45
        huge_sparse = area / region_area >= 0.26 and fill <= 0.12
        if horizontal_rule or vertical_rule or huge_sparse:
            continue
        out[labels == lab] = 255
    return out


def _candidate_text_only_clear_mask(
    bubble: BubbleInstance,
    aligned_source: np.ndarray,
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> np.ndarray:
    """Build a detail-safe TARGET clear mask for a real text container.

    For text-seeded containers, the detector already knows a tight text bbox.
    Clear all compact lettering *inside that bbox* (including glyphs shared by JP
    and CN, such as 来), then add edition-specific TARGET ink.  This avoids the
    v2.0.16 first-pass problem where border decorations and burst rays were erased
    together with the text while also eliminating residual identical glyphs.
    """
    backend = str((bubble.meta or {}).get("backend", "")).strip().lower()
    base = _text_only_base_with_seed_support(bubble, target.shape[:2])
    if cv2.countNonZero(base) <= 0:
        return np.zeros(target.shape[:2], np.uint8)

    source_unique, target_unique, _diag = changed_text_masks(
        aligned_source, target, base,
        tolerance_px=2,
        min_unique_ratio=0.045,
        max_component_fraction=0.09,
    )

    if backend in {"target_text_contour", "text_seed_fallback"}:
        support = _tight_text_bbox_support(bubble, target.shape[:2], pad=5)
        if cv2.countNonZero(support) > 0:
            heuristic = _tight_verified_text_ink_mask(target, support)
            # Add edition-specific TARGET ink only within a small halo of the
            # known text bbox; this catches antialias/stray punctuation but never
            # reaches the container border or character artwork.
            halo = cv2.dilate(support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
            extra = ((target_unique > 0) & halo).astype(np.uint8) * 255
            clear = cv2.bitwise_or(heuristic, extra)
        else:
            clear = target_unique
    else:
        # For geometry-only containers, do not trust all dark pixels. Keep only
        # edition-specific TARGET ink and compact target components touching it.
        clear = target_unique.copy()
        if cv2.countNonZero(clear) > 0:
            heuristic = _text_only_clear_mask(target, base, cfg)
            support = cv2.dilate((clear > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))) > 0
            hn, hlabels, hstats, _ = cv2.connectedComponentsWithStats((heuristic > 0).astype(np.uint8), 8)
            completed = np.zeros_like(heuristic)
            for hlab in range(1, hn):
                comp = hlabels == hlab
                if np.any(comp & support):
                    completed[comp] = 255
            clear = cv2.bitwise_or(clear, completed)

    if cv2.countNonZero(clear) > 0:
        clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        clear[base == 0] = 0
    return clear


def _verified_seed_clear_mask(
    bubble: BubbleInstance,
    aligned_source: np.ndarray,
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> np.ndarray:
    meta = dict(bubble.meta or {})
    cached = meta.get("verified_target_text_mask")
    if isinstance(cached, np.ndarray) and cached.shape == target.shape[:2]:
        clear = ((cached > 0).astype(np.uint8) * 255)
    else:
        tb = meta.get("text_bbox")
        if not isinstance(tb, (list, tuple)) or len(tb) != 4:
            return np.zeros(target.shape[:2], np.uint8)
        seed = TextBlock(
            id=str(bubble.id),
            polygon=[(float(tb[0]), float(tb[1])), (float(tb[2]), float(tb[1])), (float(tb[2]), float(tb[3])), (float(tb[0]), float(tb[3]))],
            text="", confidence=float(bubble.confidence), kind="unknown", meta={"backend": "text_seed_fallback"},
        )
        ok, _diag, _region, _source_mask, target_mask = _translation_evidence_for_seed(aligned_source, target, seed)
        if not ok:
            return np.zeros(target.shape[:2], np.uint8)
        clear = target_mask
    # Only a one-pixel antialias expansion. v2.0.15 expanded a whole seed bbox and
    # could eat hair/skin linework; verified edition-specific ink does not need it.
    if cv2.countNonZero(clear) > 0:
        clear = cv2.dilate(clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return clear


def _flat_white_full_reveal_safe(target: np.ndarray, mask: np.ndarray) -> bool:
    """Full SOURCE reveal is reserved for truly flat neutral paper.

    Any textured/radiating/illustrated white container falls back to text-only,
    keeping TARGET halftone, gradients, burst rays and line details intact.
    """
    use = mask > 0
    if not np.any(use) or not _bubble_is_white(target, mask):
        return False
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # Ignore printed dark text and the container boundary itself when judging
    # whether the paper is flat. Boundary contrast otherwise makes even a pure
    # white 40x40 box look "textured" to the Laplacian metric.
    core = cv2.erode(use.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))) > 0
    if np.count_nonzero(core) < 32:
        core = use
    paper = core & (gray >= 205) & (hsv[..., 1] <= 55)
    if np.count_nonzero(paper) < max(32, int(np.count_nonzero(core) * 0.42)):
        return False
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    edge_ratio = float(np.mean(np.abs(lap[paper]) >= 18.0)) if np.any(paper) else 1.0
    paper_std = float(np.std(gray[paper])) if np.any(paper) else 999.0
    return bool(edge_ratio <= 0.065 and paper_std <= 16.0)


def _neutral_white_full_reveal_safe(target: np.ndarray, mask: np.ndarray) -> bool:
    """Allow full SOURCE reveal for closed neutral paper even with scan halftone.

    The older flatness gate relied on Laplacian edge density. Scanned manga paper
    can have dense halftone/JPEG texture while still being a perfectly safe white
    narration/speech container. For full reveal we only need a dominant, neutral,
    bright TARGET paper interior; `_resolve_full_bubble_reveal_mask` later clips
    the actual hole back to that TARGET paper, so dark borders/rays stay on top.
    """
    use = mask > 0
    if not np.any(use) or not _bubble_is_white(target, mask):
        return False
    paper = white_container_paper_mask(target, (use.astype(np.uint8) * 255), None)
    paper_b = paper > 0
    region_pixels = int(np.count_nonzero(use))
    paper_pixels = int(np.count_nonzero(paper_b))
    if paper_pixels < max(48, int(round(region_pixels * 0.50))):
        return False
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    vals = gray[paper_b]
    sats = hsv[..., 1][paper_b]
    if vals.size < 32:
        return False
    bright_ratio = float(np.mean(vals >= 195))
    low_sat_ratio = float(np.mean(sats <= 65))
    # Reject coloured/gradient effect fields. Neutral scan texture is fine.
    return bool(bright_ratio >= 0.80 and low_sat_ratio >= 0.92 and float(np.percentile(sats, 95.0)) <= 72.0)


def _is_page_furniture_bbox(
    seed_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    cfg: TransparentBubbleRevealConfig | None = None,
) -> bool:
    """Conservatively reject running headers/footers and page furniture.

    This is deliberately independent of OCR and SOURCE translation evidence:
    a running header is real text, so OCR will confirm it, and it may even exist
    in both editions.  Position/layout is therefore the correct signal.
    """
    if cfg is not None and not bool(getattr(cfg, "suppress_page_furniture", True)):
        return False
    h, w = shape
    x0, y0, x1, y1 = [int(v) for v in seed_bbox]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    top_ratio = float(getattr(cfg, "page_furniture_top_ratio", 0.14)) if cfg is not None else 0.14
    bottom_ratio = float(getattr(cfg, "page_furniture_bottom_ratio", 0.94)) if cfg is not None else 0.94
    # Typical manga running headers are shallow horizontal text bands near a page
    # edge and outside the main panel field. Keep the condition intentionally
    # narrow so a real speech balloon near the top edge is not rejected merely by
    # vertical position.
    top_band = y1 <= int(round(h * top_ratio))
    shallow = bh <= int(round(h * 0.060))
    horizontalish = bw >= max(28, int(round(bh * 1.65)))
    edgeish = x0 <= int(round(w * 0.34)) or x1 >= int(round(w * 0.66))
    if top_band and shallow and horizontalish and edgeish:
        return True
    if y0 >= int(round(h * bottom_ratio)) and bh <= int(round(h * 0.035)) and horizontalish:
        return True
    return False


def _recover_rectangular_white_container_from_seed(
    target: np.ndarray, seed_bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, dict[str, Any]]:
    """Recover a closed white narration/text rectangle around a fallback text seed.

    A seed-only fallback clips translated text whenever CN uses more rows/columns
    than JP. We therefore look for four TARGET border lines enclosing the seed.
    The candidate is accepted only when its interior is overwhelmingly bright,
    neutral paper. This keeps the recovery local and prevents white clothing/page
    margins from becoming destructive reveal regions.
    """
    h, w = target.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in seed_bbox]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = max(70, min(180, int(round(sw * 1.60))))
    pad_top = max(120, min(280, int(round(sh * 2.20))))
    pad_bottom = max(140, min(320, int(round(sh * 2.40))))
    wx0, wx1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    wy0, wy1 = max(0, y0 - pad_top), min(h, y1 + pad_bottom)
    roi = target[wy0:wy1, wx0:wx1]
    if roi.size == 0:
        return None, None, {"status": "empty_search_window"}
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 45, 140)
    min_len = max(28, int(round(min(sw, sh) * 0.42)))
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=max(20, int(round(min_len * 0.55))),
        minLineLength=min_len, maxLineGap=12,
    )
    if lines is None:
        return None, None, {"status": "no_lines"}
    try:
        line_rows = np.asarray(lines).reshape(-1, 4)
    except Exception:
        return None, None, {"status": "bad_lines_shape", "shape": list(np.asarray(lines).shape)}

    vertical: list[tuple[float, float, int, int]] = []
    horizontal: list[tuple[float, float, int, int]] = []
    for xa, ya, xb, yb in line_rows:
        xa, xb = int(xa) + wx0, int(xb) + wx0
        ya, yb = int(ya) + wy0, int(yb) + wy0
        dx, dy = xb - xa, yb - ya
        length = float(np.hypot(dx, dy))
        if abs(dx) <= max(4.0, abs(dy) * 0.12):
            xx = (xa + xb) * 0.5; lo, hi = min(ya, yb), max(ya, yb)
            overlap = max(0, min(hi, y1) - max(lo, y0))
            if length >= min_len and (overlap >= sh * 0.35 or lo <= (y0 + y1) * 0.5 <= hi):
                vertical.append((xx, length, lo, hi))
        elif abs(dy) <= max(4.0, abs(dx) * 0.12):
            yy = (ya + yb) * 0.5; lo, hi = min(xa, xb), max(xa, xb)
            overlap = max(0, min(hi, x1) - max(lo, x0))
            if length >= min_len and (overlap >= sw * 0.35 or lo <= (x0 + x1) * 0.5 <= hi):
                horizontal.append((yy, length, lo, hi))

    left = sorted((v for v in vertical if v[0] < x0 - 4), key=lambda v: x0 - v[0])[:6]
    right = sorted((v for v in vertical if v[0] > x1 + 4), key=lambda v: v[0] - x1)[:6]
    top = sorted((q for q in horizontal if q[0] < y0 - 4), key=lambda q: y0 - q[0])[:6]
    bottom = sorted((q for q in horizontal if q[0] > y1 + 4), key=lambda q: q[0] - y1)[:6]
    if not left or not right:
        return None, None, {"status": "missing_vertical_sides"}

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    page_area = max(1, h * w)
    best: tuple[float, tuple[int, int, int, int], float, float, float] | None = None

    def consider_rect(rx0: int, ry0: int, rx1: int, ry1: int, *, bonus: float = 0.0) -> None:
        nonlocal best
        rw, rh = rx1 - rx0, ry1 - ry0
        if rw < sw * 1.15 or rh < sh * 1.15:
            return
        if not (rx0 < x0 and rx1 > x1 and ry0 < y0 and ry1 > y1):
            return
        area = rw * rh
        if area <= 0 or area / page_area > 0.08:
            return
        ix0, ix1, iy0, iy1 = rx0 + 3, rx1 - 3, ry0 + 3, ry1 - 3
        if ix1 <= ix0 or iy1 <= iy0:
            return
        g = gray[iy0:iy1, ix0:ix1]; sat = hsv[iy0:iy1, ix0:ix1, 1]
        neutral = float(np.mean((g >= 195) & (sat <= 65)))
        dark = float(np.mean(g < 160)); high_sat = float(np.mean(sat > 100))
        if neutral < 0.55 or dark > 0.25 or high_sat > 0.12:
            return
        gap = float((x0 - rx0) + (rx1 - x1) + (y0 - ry0) + (ry1 - y1))
        score = neutral * 3.0 - dark * 2.0 - high_sat * 2.0 - gap / max(1.0, max(sw, sh) * 8.0) + bonus
        if best is None or score > best[0]:
            best = (score, (rx0, ry0, rx1, ry1), neutral, dark, high_sat)

    # Primary path: four detected TARGET border lines.
    if top and bottom:
        for l in left:
            for r in right:
                for t in top:
                    for b in bottom:
                        consider_rect(int(round(l[0])), int(round(t[0])), int(round(r[0])), int(round(b[0])), bonus=0.08)

    # Some scanned boxes have a very faint top/bottom rule that Hough misses.
    # The neutral paper interior still has a strong row profile separated from
    # surrounding artwork by the border. Infer those two horizontal bounds while
    # keeping Hough-detected vertical sides.
    seed_cy = int(round((y0 + y1) * 0.5))
    for l in left:
        for r in right:
            rx0, rx1 = int(round(l[0])), int(round(r[0]))
            if rx1 - rx0 < sw * 1.45:
                continue
            ix0, ix1 = rx0 + 3, rx1 - 3
            if ix1 <= ix0:
                continue
            sy0, sy1 = max(0, y0 - pad_top), min(h, y1 + pad_bottom)
            neutral_rows = (gray[sy0:sy1, ix0:ix1] >= 190) & (hsv[sy0:sy1, ix0:ix1, 1] <= 75)
            if neutral_rows.size == 0:
                continue
            row_score = np.mean(neutral_rows, axis=1)
            good = row_score >= 0.58
            start: int | None = None
            for idx in range(len(good) + 1):
                on = bool(good[idx]) if idx < len(good) else False
                if on and start is None:
                    start = idx
                elif (not on) and start is not None:
                    end = idx
                    gy0, gy1 = sy0 + start, sy0 + end
                    if end - start >= max(8, int(round(sh * 0.65))) and gy0 <= seed_cy < gy1:
                        # Good rows are the paper interior; step one pixel outward
                        # to approximate the TARGET border coordinate.
                        consider_rect(rx0, max(0, gy0 - 1), rx1, min(h, gy1), bonus=0.03)
                    start = None
    if best is None:
        return None, None, {"status": "no_safe_rectangle"}
    _, rect, neutral, dark, high_sat = best
    rx0, ry0, rx1, ry1 = rect
    mask = np.zeros((h, w), np.uint8)
    # Keep the TARGET border itself opaque; the reveal hole starts just inside it.
    inset = 2
    if rx1 - rx0 <= inset * 2 + 2 or ry1 - ry0 <= inset * 2 + 2:
        return None, None, {"status": "rectangle_too_small"}
    mask[ry0 + inset:ry1 - inset, rx0 + inset:rx1 - inset] = 255
    return mask, rect, {
        "status": "ok", "neutral_ratio": neutral, "dark_ratio": dark,
        "high_saturation_ratio": high_sat, "rect": [rx0, ry0, rx1, ry1],
    }


def _recover_neutral_white_container_from_seed(
    target: np.ndarray, seed_bbox: tuple[int, int, int, int], cfg: TransparentBubbleRevealConfig | None = None,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, dict[str, Any]]:
    """Recover a local neutral-white closed container around a text seed.

    The old seed fallback only knew how to grow a rectangular narration box.
    Small oval speech balloons then fell through to a plain text bbox, which is
    safe but often degrades to text-only clearing.  This helper keeps the search
    local and TARGET-only: look for a neutral/bright connected component around
    the verified text seed and use it when it is compact, page-local and clearly
    whiter than its surroundings.
    """
    rect_mask, rect_bbox, rect_diag = _recover_rectangular_white_container_from_seed(target, seed_bbox)
    if rect_mask is not None and rect_bbox is not None and cv2.countNonZero(rect_mask) > 0:
        return rect_mask, rect_bbox, {"status": "rectangular", **rect_diag}

    h, w = target.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in seed_bbox]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    if _is_page_furniture_bbox((x0, y0, x1, y1), (h, w), cfg):
        return None, None, {"status": "page_furniture"}
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = max(36, min(160, int(round(sw * 1.35))))
    pad_y = max(36, min(160, int(round(sh * 1.55))))
    rx0, rx1 = max(0, x0 - pad_x), min(w, x1 + pad_x)
    ry0, ry1 = max(0, y0 - pad_y), min(h, y1 + pad_y)
    roi = target[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None, None, {"status": "empty_roi", "rect_diag": rect_diag}

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    neutral = ((gray >= 192) & (hsv[..., 1] <= 92)).astype(np.uint8) * 255
    neutral = cv2.morphologyEx(neutral, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    seed_local = np.zeros_like(neutral)
    sx0, sy0, sx1, sy1 = x0 - rx0, y0 - ry0, x1 - rx0, y1 - ry0
    seed_local[sy0:sy1, sx0:sx1] = 255
    # The seed bbox is mostly dark ink, so the recovered white component usually
    # surrounds it rather than overlapping it.  Admit candidates touching a
    # modest dilation of the seed box as well; this is what lets a true white
    # speech bubble be recovered instead of falling through to plain text_only.
    support_pad = max(12, min(34, int(round(max(sw, sh) * 0.18))))
    support_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support_pad * 2 + 1, support_pad * 2 + 1))
    seed_support = cv2.dilate(seed_local, support_kernel, iterations=1)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((neutral > 0).astype(np.uint8), 8)
    page_area = max(1, h * w)
    seed_area = max(1, sw * sh)
    best = None
    for lab in range(1, n):
        bx, by, bw, bh, area = [int(v) for v in stats[lab]]
        if area < max(80, int(round(seed_area * 1.15))):
            continue
        if area > min(int(round(seed_area * 42.0)), int(round(page_area * 0.08))):
            continue
        comp = labels == lab
        overlap = int(np.count_nonzero(comp & (seed_local > 0)))
        support_overlap = int(np.count_nonzero(comp & (seed_support > 0)))
        cx = int(round((sx0 + sx1) * 0.5)); cy = int(round((sy0 + sy1) * 0.5))
        contains_center = 0 <= cy < labels.shape[0] and 0 <= cx < labels.shape[1] and int(labels[cy, cx]) == lab
        encloses_seed = bool(bx <= sx0 and by <= sy0 and (bx + bw) >= sx1 and (by + bh) >= sy1)
        if support_overlap <= 0 and not contains_center and not encloses_seed:
            continue
        full = np.zeros((h, w), np.uint8)
        full[ry0:ry1, rx0:rx1][comp] = 255
        if not _fallback_white_candidate_safe(target, full):
            continue
        gap = float(abs((bx + bw * 0.5) - (sx0 + sx1) * 0.5) + abs((by + bh * 0.5) - (sy0 + sy1) * 0.5))
        score = float(overlap) * 2.0 + float(support_overlap) * 0.75 + (24.0 if contains_center else 0.0) + (18.0 if encloses_seed else 0.0) - gap * 0.08 - area * 0.0006
        if best is None or score > best[0]:
            best = (score, full, (bx + rx0, by + ry0, bx + rx0 + bw, by + ry0 + bh), overlap, support_overlap, area)
    if best is None:
        flood_mask, flood_bbox, flood_diag = _recover_floodfill_white_container_from_seed(
            target, (rx0, ry0, rx1, ry1), (x0, y0, x1, y1), seed_support
        )
        if flood_mask is not None and flood_bbox is not None and cv2.countNonZero(flood_mask) > 0:
            return flood_mask, flood_bbox, {"status": "floodfill_component", **flood_diag, "rect_diag": rect_diag}
        return None, None, {"status": "no_safe_neutral_component", "rect_diag": rect_diag, "flood_diag": flood_diag}
    _, full_mask, bbox, overlap, support_overlap, area = best
    inset = cv2.erode(full_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    if cv2.countNonZero(inset) > 0:
        full_mask = inset
    return full_mask, tuple(int(v) for v in bbox), {
        "status": "neutral_component",
        "seed_overlap_pixels": int(overlap),
        "seed_support_overlap_pixels": int(support_overlap),
        "component_pixels": int(area),
        "bbox": [int(v) for v in bbox],
        "rect_diag": rect_diag,
    }


def _recover_floodfill_white_container_from_seed(
    target: np.ndarray,
    roi_bounds: tuple[int, int, int, int],
    seed_bbox: tuple[int, int, int, int],
    seed_support: np.ndarray,
) -> tuple[np.ndarray | None, tuple[int, int, int, int] | None, dict[str, Any]]:
    """Recover a local white speech container with an interior flood fill.

    Some speech balloons connect to the page gutter in a simple white-threshold
    map, so connected-components over "neutral" pixels return one giant region.
    A fixed-range flood fill from a bright point *inside* the seed support is much
    closer to how the actual white interior behaves: the black bubble outline
    blocks the fill, but the text itself does not matter because the seed point is
    chosen from the local distance-transform maxima on white pixels.
    """
    h, w = target.shape[:2]
    rx0, ry0, rx1, ry1 = [int(v) for v in roi_bounds]
    roi = target[ry0:ry1, rx0:rx1]
    if roi.size == 0:
        return None, None, {"status": "empty_roi"}
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    white_like = ((gray >= 206) & (hsv[..., 1] <= 72)).astype(np.uint8)
    if np.count_nonzero(white_like) <= 0:
        return None, None, {"status": "no_white_like_pixels"}
    support = (seed_support > 0).astype(np.uint8)
    dist = cv2.distanceTransform(white_like, cv2.DIST_L2, 5)
    scored = np.where((support > 0) & (dist >= max(2.0, float(np.percentile(dist[dist > 0], 55.0)) if np.count_nonzero(dist > 0) else 2.0)))
    if scored[0].size == 0:
        scored = np.where((support > 0) & (white_like > 0))
    if scored[0].size == 0:
        return None, None, {"status": "no_seeded_white_anchor"}
    pts = sorted([(float(dist[y, x]), int(x), int(y)) for y, x in zip(scored[0], scored[1])], reverse=True)[:8]
    best = None
    for score0, sx, sy in pts:
        mask = np.zeros((gray.shape[0] + 2, gray.shape[1] + 2), np.uint8)
        work = gray.copy()
        flags = 8 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
        try:
            cv2.floodFill(work, mask, (sx, sy), 0, 12, 12, flags)
        except cv2.error:
            continue
        comp = (mask[1:-1, 1:-1] > 0).astype(np.uint8) * 255
        if cv2.countNonZero(comp) <= 0:
            continue
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        full = np.zeros((h, w), np.uint8)
        full[ry0:ry1, rx0:rx1] = comp
        area = int(cv2.countNonZero(full))
        if area <= 0:
            continue
        x, y, bw, bh = cv2.boundingRect((full > 0).astype(np.uint8))
        area_ratio = float(area / max(1, h * w))
        seed_box_area = max(1, (seed_bbox[2] - seed_bbox[0]) * (seed_bbox[3] - seed_bbox[1]))
        if area_ratio > 0.06 or area < max(240, int(round(seed_box_area * 0.45))):
            continue
        cover_x0 = max(x, int(seed_bbox[0])); cover_y0 = max(y, int(seed_bbox[1]))
        cover_x1 = min(x + bw, int(seed_bbox[2])); cover_y1 = min(y + bh, int(seed_bbox[3]))
        seed_bbox_covered = max(0, cover_x1 - cover_x0) * max(0, cover_y1 - cover_y0)
        coverage_ratio = float(seed_bbox_covered / max(1, seed_box_area))
        # A recovered speech-bubble interior must substantially cover the text
        # seed itself; otherwise white page margin / neighbour gutters can win.
        if coverage_ratio < 0.78:
            continue
        overlap = int(np.count_nonzero((full > 0) & (np.pad(support, ((ry0, h - ry1), (rx0, w - rx1)), mode='constant') > 0)))
        profile = _neutral_paper_profile(target, full)
        if not (profile['bright_ratio'] >= 0.62 and profile['low_sat_p75'] <= 46.0 and profile['mean_bgr'] >= 188.0 and profile['bbox_fill_ratio'] >= 0.48):
            continue
        score = float(score0) * 10.0 + float(overlap) * 0.05 + coverage_ratio * 120.0 + float(profile['compactness']) * 80.0 - area * 0.0004
        if best is None or score > best[0]:
            best = (score, full, (x, y, x + bw, y + bh), overlap, coverage_ratio, area, profile)
    if best is None:
        return None, None, {"status": "no_safe_floodfill_component"}
    _, full, bbox, overlap, coverage_ratio, area, profile = best
    inset = cv2.erode(full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    if cv2.countNonZero(inset) > 0:
        full = inset
    return full, tuple(int(v) for v in bbox), {
        "status": "floodfill_component",
        "seed_support_overlap_pixels": int(overlap),
        "seed_bbox_coverage_ratio": float(coverage_ratio),
        "component_pixels": int(area),
        "profile": {k: float(v) for k, v in profile.items()},
        "bbox": [int(v) for v in bbox],
    }


def _interior_base_mask(bubble: BubbleInstance, shape: tuple[int, int]) -> np.ndarray:
    base = bubble.mask.copy() if bubble.mask is not None and bubble.mask.shape == shape else rasterize_polygon(bubble.polygon, shape)
    return ((base > 0).astype(np.uint8) * 255)


def _full_bubble_clear_mask(base: np.ndarray, cfg: TransparentBubbleRevealConfig) -> np.ndarray:
    out = base.copy()
    if int(cfg.expand_px) > 0:
        r = int(cfg.expand_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = cv2.dilate(out, k)
    if int(cfg.inset_px) > 0:
        r = int(cfg.inset_px)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        out = cv2.erode(out, k)
    if bool(cfg.protect_border):
        # Clamp to the original container interior and leave a deterministic edge
        # band fully opaque. This is what prevents the Japanese bubble outline from
        # becoming a transparent seam even when expand_px > 0.
        dist = cv2.distanceTransform((base > 0).astype(np.uint8), cv2.DIST_L2, 5)
        guard = max(1.0, float(getattr(cfg, "border_protect_px", 2)))
        out[(dist <= guard)] = 0
        out[base == 0] = 0
    return ((out > 0).astype(np.uint8) * 255)


def _text_only_clear_mask(target: np.ndarray, base: np.ndarray, cfg: TransparentBubbleRevealConfig) -> np.ndarray:
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    interior = base.copy()
    guard = max(1, int(getattr(cfg, "border_protect_px", 2))) if bool(cfg.protect_border) else 0
    if guard > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard + 1, 2 * guard + 1))
        interior = cv2.erode(interior, k)
    use = interior > 0
    vals = gray[use]
    configured_thr = int(getattr(cfg, "text_ink_threshold", 190))
    if vals.size:
        # Absolute gray thresholds fail on coloured bubbles because the fill itself
        # can be darker than 190.  Estimate the local fill level first and require
        # lettering to be substantially darker than that fill.
        bg = float(np.percentile(vals, 72.0))
        spread = float(np.percentile(vals, 85.0) - np.percentile(vals, 25.0))
        margin = max(22.0, min(60.0, 24.0 + spread * 0.18))
        adaptive_thr = int(np.clip(bg - margin, 55, configured_thr))
    else:
        adaptive_thr = configured_thr
    dark = ((gray <= adaptive_thr) & use).astype(np.uint8) * 255
    n, labels, stats, _ = cv2.connectedComponentsWithStats((dark > 0).astype(np.uint8), 8)
    clean = np.zeros_like(dark)
    region_area = max(1, cv2.countNonZero(interior))
    ys, xs = np.where(interior > 0)
    bw = int(xs.max() - xs.min() + 1) if len(xs) else 1
    bh = int(ys.max() - ys.min() + 1) if len(ys) else 1
    for i in range(1, n):
        x, y, w, h, area = [int(v) for v in stats[i]]
        if area < 2 or area > max(1200, int(region_area * 0.035)):
            continue
        if w > max(18, int(bw * 0.62)) or h > max(18, int(bh * 0.62)):
            continue
        if max(w / max(1.0, h), h / max(1.0, w)) > 12.0:
            continue
        clean[labels == i] = 255
    r = max(0, int(cfg.expand_px))
    if r > 0 and cv2.countNonZero(clean):
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        clean = cv2.dilate(clean, k)
    clean[interior == 0] = 0
    return clean


def _mask_bbox_fill_ratio(mask: np.ndarray) -> float:
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return 0.0
    bw = int(xs.max() - xs.min() + 1)
    bh = int(ys.max() - ys.min() + 1)
    return float(np.count_nonzero(mask) / max(1, bw * bh))


def _mask_compactness(mask: np.ndarray) -> float:
    binary = ((mask > 0).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(cnt))
    peri = float(cv2.arcLength(cnt, True))
    if area <= 0.0 or peri <= 0.0:
        return 0.0
    return float((4.0 * np.pi * area) / max(1e-6, peri * peri))


def _neutral_paper_profile(target: np.ndarray, base: np.ndarray) -> dict[str, float]:
    sel = base > 0
    if not np.any(sel):
        return {
            "bright_ratio": 0.0, "low_sat_p75": 255.0, "mean_bgr": 0.0,
            "bbox_fill_ratio": 0.0, "compactness": 0.0,
        }
    gray_all = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    bgr = target[sel]
    gray = gray_all[sel]
    sat = hsv[..., 1][sel]
    return {
        "bright_ratio": float(np.mean(gray >= 220)),
        "low_sat_p75": float(np.percentile(sat, 75.0)),
        "mean_bgr": float(np.mean(bgr)),
        "bbox_fill_ratio": _mask_bbox_fill_ratio(base),
        "compactness": _mask_compactness(base),
    }


def _bubble_is_white(target: np.ndarray, base: np.ndarray) -> bool:
    profile = _neutral_paper_profile(target, base)
    return bool(
        profile["bright_ratio"] >= 0.70
        and profile["low_sat_p75"] <= 35.0
        and profile["mean_bgr"] >= 190.0
        and profile["bbox_fill_ratio"] >= 0.60
        and profile["compactness"] >= 0.22
    )



def _candidate_area_ratio(mask: np.ndarray) -> float:
    return float(cv2.countNonZero((mask > 0).astype(np.uint8)) / max(1, mask.shape[0] * mask.shape[1]))


def _fallback_white_candidate_safe(target: np.ndarray, mask: np.ndarray) -> bool:
    """Strict gate for heuristic white-region backends.

    These backends can otherwise mistake skin, hair gaps, walls or an entire
    manga panel for a bubble.  A fallback candidate must look like neutral paper,
    occupy a compact local shape, and remain small relative to the page.
    """
    profile = _neutral_paper_profile(target, mask)
    area_ratio = _candidate_area_ratio(mask)
    x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
    return bool(
        0.0008 <= area_ratio <= 0.08
        and bw / max(1, mask.shape[1]) <= 0.48
        and bh / max(1, mask.shape[0]) <= 0.40
        and profile["bright_ratio"] >= 0.70
        and profile["low_sat_p75"] <= 35.0
        and profile["mean_bgr"] >= 195.0
        and profile["bbox_fill_ratio"] >= 0.58
        and profile["compactness"] >= 0.20
    )


def _effect_container_candidate_safe(target: np.ndarray, mask: np.ndarray) -> bool:
    """Allow real coloured/burst narration containers but reject tiny SFX/text art."""
    profile = _neutral_paper_profile(target, mask)
    area_ratio = _candidate_area_ratio(mask)
    x, y, bw, bh = cv2.boundingRect((mask > 0).astype(np.uint8))
    return bool(
        0.003 <= area_ratio <= 0.08
        and bw / max(1, mask.shape[1]) <= 0.48
        and bh / max(1, mask.shape[0]) <= 0.40
        and profile["bright_ratio"] >= 0.42
        and profile["mean_bgr"] >= 175.0
        and profile["bbox_fill_ratio"] >= 0.60
        and profile["compactness"] >= 0.26
    )


def _auto_candidate_safe(target: np.ndarray, row: BubbleInstance, backend: str) -> bool:
    """Final destructive gate for automatically discovered containers."""
    mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
    if mask is None or cv2.countNonZero(mask) <= 0:
        return False
    backend = str(backend or "").strip().lower()
    if backend == "target_text_contour":
        return bool(_bubble_is_white(target, mask) or _effect_container_candidate_safe(target, mask))
    if backend in {"seeded_white", "unseeded_white", "unseeded"}:
        return _fallback_white_candidate_safe(target, mask)
    # Model-based bubble detectors are still constrained before destructive use.
    if backend in {"mangalens", "rtdetr_v2", "sam2"}:
        return bool(_fallback_white_candidate_safe(target, mask) or _effect_container_candidate_safe(target, mask))
    return False


def _seed_fallback_candidate_safe(target: np.ndarray, seed: TextBlock, cfg: TransparentBubbleRevealConfig) -> bool:
    x0, y0, x1, y1 = [int(round(v)) for v in seed.bbox]
    h, w = target.shape[:2]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    area_ratio = float((bw * bh) / max(1, h * w))
    # Local text fallback only: never synthesize a giant region.
    if area_ratio > 0.035 or bw / max(1, w) > 0.22 or bh / max(1, h) > 0.28:
        return False
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    roi = gray[y0:y1, x0:x1]
    if roi.size < 20:
        return False
    dark_ratio = float(np.mean(roi < 190))
    mid_ratio = float(np.mean(roi < 220))
    # True text seeds need some ink, but not a mostly-dark illustration crop.
    return bool(0.015 <= dark_ratio <= 0.55 and mid_ratio <= 0.72)


def _accepted_covers_seed(accepted: list[BubbleInstance], seed: TextBlock, shape: tuple[int, int]) -> bool:
    x0, y0, x1, y1 = seed.bbox
    h, w = shape
    cx = int(np.clip(round((x0 + x1) * 0.5), 0, w - 1))
    cy = int(np.clip(round((y0 + y1) * 0.5), 0, h - 1))
    for row in accepted:
        mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, shape)
        if mask is not None and mask[cy, cx] > 0:
            return True
    return False


def _refined_text_seed_hypotheses(
    target: np.ndarray,
    seed_bbox: tuple[int, int, int, int],
) -> list[tuple[tuple[int, int, int, int], dict[str, Any]]]:
    """Split an oversized heuristic text seed into denser glyph sub-regions.

    Heuristic text clustering can bridge a real speech-bubble text column to
    nearby anatomy/line art.  The resulting huge seed then prevents white-bubble
    recovery and is forced down the open/text-only route.  Work from compact dark
    components *inside* the seed, cluster their centres with modest directional
    kernels, and return only dense sub-regions that are materially tighter than
    the original seed.

    The routine is deliberately TARGET-only and conservative: it never creates a
    new candidate by itself; it only gives the existing seed a few tighter
    hypotheses for closed-container recovery.
    """
    h, w = target.shape[:2]
    x0, y0, x1, y1 = [int(v) for v in seed_bbox]
    x0 = max(0, min(w - 1, x0)); x1 = max(x0 + 1, min(w, x1))
    y0 = max(0, min(h - 1, y0)); y1 = max(y0 + 1, min(h, y1))
    sw, sh = max(1, x1 - x0), max(1, y1 - y0)
    if sw < 48 and sh < 72:
        return []
    roi = target[y0:y1, x0:x1]
    if roi.size == 0:
        return []
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    try:
        from .source_detectors import _compact_character_components
        comps = _compact_character_components(gray)
    except Exception:
        return []
    if len(comps) < 4:
        return []

    widths = np.asarray([c[2] for c in comps], np.float32)
    heights = np.asarray([c[3] for c in comps], np.float32)
    mw = float(np.median(widths)) if widths.size else 12.0
    mh = float(np.median(heights)) if heights.size else 14.0
    max_dim = max(36, int(round(max(mw, mh) * 2.5)))
    filtered = []
    for comp in comps:
        cx0, cy0, cw, ch, area, center = comp
        fill = float(area / max(1, cw * ch))
        if max(cw, ch) > max_dim:
            continue
        if fill < 0.11:
            continue
        filtered.append(comp)
    if len(filtered) < 4:
        return []

    # Centre-only morphology prevents a long line-art stroke from becoming the
    # bridge.  Try a vertical-CJK-biased and a horizontal-caption-biased kernel.
    impulse = np.zeros(gray.shape, np.uint8)
    for comp in filtered:
        cx, cy = comp[5]
        ix = int(np.clip(round(cx), 0, impulse.shape[1] - 1))
        iy = int(np.clip(round(cy), 0, impulse.shape[0] - 1))
        impulse[iy, ix] = 255
    kernels = [
        (max(21, int(round(mw * 2.1))), max(35, int(round(mh * 2.8))), "vertical"),
        (max(35, int(round(mw * 2.8))), max(21, int(round(mh * 2.1))), "horizontal"),
    ]
    original_area = max(1, sw * sh)
    candidates: list[tuple[float, tuple[int, int, int, int], dict[str, Any]]] = []
    for kx, ky, orientation in kernels:
        merged = cv2.dilate(impulse, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)), iterations=1)
        n, labels, _stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
        for lab in range(1, n):
            members = []
            for comp in filtered:
                cx, cy = comp[5]
                ix = int(np.clip(round(cx), 0, labels.shape[1] - 1))
                iy = int(np.clip(round(cy), 0, labels.shape[0] - 1))
                if int(labels[iy, ix]) == lab:
                    members.append(comp)
            if len(members) < 4:
                continue
            lx0 = min(c[0] for c in members); ly0 = min(c[1] for c in members)
            lx1 = max(c[0] + c[2] for c in members); ly1 = max(c[1] + c[3] for c in members)
            bw, bh = max(1, lx1 - lx0), max(1, ly1 - ly0)
            area = bw * bh
            area_ratio = float(area / original_area)
            if area_ratio >= 0.82:
                continue
            # Reject tiny accidental clusters unless the original seed itself is
            # small. A real Japanese speech block normally contributes several
            # compact pieces and a meaningful span.
            if area < 240 or max(bw, bh) < 26:
                continue
            density = float(len(members) / max(1.0, area / max(36.0, mw * mh)))
            orient_bonus = 0.12 if ((orientation == "vertical" and bh >= bw) or (orientation == "horizontal" and bw >= bh)) else 0.0
            score = float(len(members)) * 2.0 + density * 4.0 + orient_bonus * 10.0 - area_ratio * 5.0
            pad_x = max(3, min(8, int(round(mw * 0.35))))
            pad_y = max(3, min(8, int(round(mh * 0.35))))
            bx0 = max(x0, x0 + lx0 - pad_x); by0 = max(y0, y0 + ly0 - pad_y)
            bx1 = min(x1, x0 + lx1 + pad_x); by1 = min(y1, y0 + ly1 + pad_y)
            bbox = (int(bx0), int(by0), int(bx1), int(by1))
            candidates.append((score, bbox, {
                "orientation": orientation,
                "component_count": int(len(members)),
                "area_ratio": float(area_ratio),
                "density": float(density),
            }))

    # Density-window pass: the morphology above can still bridge a long seed
    # when anatomy components happen to form a chain into the true text block.
    # Search for a dense vertical/horizontal sub-window and use robust 10/90%
    # trimming on the orthogonal axis. This is especially effective for vertical
    # Japanese text inside a partially clipped speech bubble.
    centers_y = sorted({int(round(c[5][1])) for c in filtered})
    min_y_span = max(44, int(round(mh * 3.0)))
    for ya in centers_y:
        for yb in centers_y:
            if yb <= ya or yb - ya < min_y_span:
                continue
            members = [c for c in filtered if ya - mh * 0.5 <= c[5][1] <= yb + mh * 0.5]
            if len(members) < 4:
                continue
            cxs = np.asarray([c[5][0] for c in members], np.float32)
            qlo, qhi = np.percentile(cxs, [10.0, 90.0])
            core = [c for c in members if qlo - mw <= c[5][0] <= qhi + mw]
            if len(core) < 4:
                continue
            lx0 = min(c[0] for c in core); ly0 = min(c[1] for c in core)
            lx1 = max(c[0] + c[2] for c in core); ly1 = max(c[1] + c[3] for c in core)
            bw, bh = max(1, lx1 - lx0), max(1, ly1 - ly0)
            area = bw * bh
            area_ratio = float(area / original_area)
            if area_ratio >= 0.82 or area < 240:
                continue
            density = float(len(core) / max(1.0, area / max(36.0, mw * mh)))
            score = density * 25.0 + len(core) * 0.90 - area_ratio * 25.0 + (1.4 if bh >= bw else 0.0)
            pad_x = max(3, min(8, int(round(mw * 0.35))))
            pad_y = max(3, min(8, int(round(mh * 0.35))))
            bbox = (
                int(max(x0, x0 + lx0 - pad_x)), int(max(y0, y0 + ly0 - pad_y)),
                int(min(x1, x0 + lx1 + pad_x)), int(min(y1, y0 + ly1 + pad_y)),
            )
            candidates.append((float(score), bbox, {
                "orientation": "vertical_density_window",
                "component_count": int(len(core)),
                "area_ratio": float(area_ratio),
                "density": float(density),
            }))

    centers_x = sorted({int(round(c[5][0])) for c in filtered})
    min_x_span = max(44, int(round(mw * 3.0)))
    for xa in centers_x:
        for xb in centers_x:
            if xb <= xa or xb - xa < min_x_span:
                continue
            members = [c for c in filtered if xa - mw * 0.5 <= c[5][0] <= xb + mw * 0.5]
            if len(members) < 4:
                continue
            cys = np.asarray([c[5][1] for c in members], np.float32)
            qlo, qhi = np.percentile(cys, [10.0, 90.0])
            core = [c for c in members if qlo - mh <= c[5][1] <= qhi + mh]
            if len(core) < 4:
                continue
            lx0 = min(c[0] for c in core); ly0 = min(c[1] for c in core)
            lx1 = max(c[0] + c[2] for c in core); ly1 = max(c[1] + c[3] for c in core)
            bw, bh = max(1, lx1 - lx0), max(1, ly1 - ly0)
            area = bw * bh
            area_ratio = float(area / original_area)
            if area_ratio >= 0.82 or area < 240:
                continue
            density = float(len(core) / max(1.0, area / max(36.0, mw * mh)))
            score = density * 25.0 + len(core) * 0.90 - area_ratio * 25.0 + (1.4 if bw >= bh else 0.0)
            pad_x = max(3, min(8, int(round(mw * 0.35))))
            pad_y = max(3, min(8, int(round(mh * 0.35))))
            bbox = (
                int(max(x0, x0 + lx0 - pad_x)), int(max(y0, y0 + ly0 - pad_y)),
                int(min(x1, x0 + lx1 + pad_x)), int(min(y1, y0 + ly1 + pad_y)),
            )
            candidates.append((float(score), bbox, {
                "orientation": "horizontal_density_window",
                "component_count": int(len(core)),
                "area_ratio": float(area_ratio),
                "density": float(density),
            }))

    candidates.sort(key=lambda row: row[0], reverse=True)
    out: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = []
    for score, bbox, diag in candidates:
        bx0, by0, bx1, by1 = bbox
        duplicate = False
        for old, _ in out:
            ox0, oy0, ox1, oy1 = old
            ix = max(0, min(bx1, ox1) - max(bx0, ox0))
            iy = max(0, min(by1, oy1) - max(by0, oy0))
            inter = ix * iy
            amin = min((bx1 - bx0) * (by1 - by0), (ox1 - ox0) * (oy1 - oy0))
            if amin > 0 and inter / amin >= 0.82:
                duplicate = True
                break
        if duplicate:
            continue
        out.append((bbox, {**diag, "score": float(score)}))
        if len(out) >= 5:
            break
    return out


def _seed_fallback_bubble(seed: TextBlock, target: np.ndarray, cfg: TransparentBubbleRevealConfig | None = None) -> BubbleInstance:
    h, w = target.shape[:2]
    original_seed_bbox = tuple(int(round(v)) for v in seed.bbox)
    hypotheses = _refined_text_seed_hypotheses(target, original_seed_bbox)
    # Try denser sub-seeds before the original oversized heuristic bbox.  This
    # gives closed white bubbles priority over a bridged open-text/line-art seed.
    attempts: list[tuple[tuple[int, int, int, int], dict[str, Any]]] = list(hypotheses)
    attempts.append((original_seed_bbox, {"orientation": "original", "component_count": 0, "area_ratio": 1.0, "score": 0.0}))
    valid_recoveries: list[tuple[float, np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int], dict[str, Any], dict[str, Any] | None, dict[str, Any]]] = []
    attempt_diags: list[dict[str, Any]] = []
    for candidate_seed_bbox, refine_diag in attempts:
        recovered_mask, recovered_rect, recovery_diag = _recover_neutral_white_container_from_seed(target, candidate_seed_bbox, cfg)
        row_diag = {
            "seed_bbox": [int(v) for v in candidate_seed_bbox],
            "refine": dict(refine_diag),
            "recovery_status": str(recovery_diag.get("status", "unknown")),
        }
        if recovered_mask is None or recovered_rect is None or cv2.countNonZero(recovered_mask) <= 0:
            attempt_diags.append(row_diag)
            continue
        quality_ok = True
        quality_diag: dict[str, Any] | None = None
        if str(recovery_diag.get('status')) != 'rectangular':
            quality_ok, quality_diag = _seeded_container_quality_ok(target, recovered_mask, recovered_rect, candidate_seed_bbox, cfg)
            if quality_ok:
                # Flood/neutral recovery follows white pixels around dark JP ink,
                # so the raw mask can contain letter-shaped notches.  A true
                # closed speech bubble should clear as one solid interior.  Fill
                # the external convex envelope only when the inflation is small
                # and the strict white-container quality gate still passes.
                contours, _ = cv2.findContours((recovered_mask > 0).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    contour_area = max(1.0, float(cv2.contourArea(contour)))
                    hull = cv2.convexHull(contour)
                    hull_area = float(cv2.contourArea(hull))
                    if hull_area <= contour_area * 1.18:
                        solid = np.zeros_like(recovered_mask)
                        cv2.drawContours(solid, [hull], -1, 255, cv2.FILLED)
                        solid_ok, solid_diag = _seeded_container_quality_ok(target, solid, recovered_rect, candidate_seed_bbox, cfg)
                        if solid_ok:
                            recovered_mask = solid
                            quality_diag = solid_diag
                            recovery_diag = {**dict(recovery_diag), 
                                "solidified_external_hull": True,
                                "solid_hull_inflation_ratio": float(hull_area / contour_area),
                            }
        row_diag["quality"] = quality_diag
        row_diag["quality_ok"] = bool(quality_ok)
        attempt_diags.append(row_diag)
        if not quality_ok:
            continue
        profile = _neutral_paper_profile(target, recovered_mask)
        rx0, ry0, rx1, ry1 = recovered_rect
        bbox_area = max(1, (rx1 - rx0) * (ry1 - ry0))
        # Prefer compact, bright, high-fill closed containers and a refined seed
        # that has substantial internal glyph support.
        score = (
            float(profile.get("bright_ratio", 0.0)) * 4.0
            + float(profile.get("bbox_fill_ratio", 0.0)) * 3.0
            + float(profile.get("compactness", 0.0)) * 2.0
            + min(3.0, float(refine_diag.get("component_count", 0)) * 0.08)
            - bbox_area / max(1.0, float(h * w)) * 12.0
        )
        valid_recoveries.append((score, recovered_mask, recovered_rect, candidate_seed_bbox, recovery_diag, quality_diag, refine_diag))

    if valid_recoveries:
        valid_recoveries.sort(key=lambda row: row[0], reverse=True)
        _score, recovered_mask, recovered_rect, chosen_seed_bbox, recovery_diag, quality_diag, refine_diag = valid_recoveries[0]
        rx0, ry0, rx1, ry1 = recovered_rect
        poly = [(float(rx0), float(ry0)), (float(rx1), float(ry0)), (float(rx1), float(ry1)), (float(rx0), float(ry1))]
        backend = 'text_seed_white_rect' if str(recovery_diag.get('status')) == 'rectangular' else 'text_seed_white_container'
        kind = 'narration' if backend == 'text_seed_white_rect' else 'speech'
        meta = {
            'backend': backend, 'target_only': True,
            'recovered_from_text_seed_fallback': True, 'white_rect_recovery': recovery_diag,
            'text_seed_backend': str((seed.meta or {}).get('backend', 'unknown')),
            'text_bbox': [int(v) for v in chosen_seed_bbox],
            'original_text_bbox': [int(v) for v in original_seed_bbox],
            'text_seed_refined': bool(tuple(chosen_seed_bbox) != tuple(original_seed_bbox)),
            'text_seed_refine': dict(refine_diag),
            'text_seed_recovery_attempts': attempt_diags,
        }
        if quality_diag is not None:
            meta['seeded_container_quality'] = quality_diag
        return BubbleInstance(
            id=f"tbr-{backend}-{seed.id}", polygon=poly, confidence=max(0.84, float(seed.confidence)),
            kind=kind, block_ids=[seed.id], mask=recovered_mask, safe_mask=recovered_mask.copy(), meta=meta,
        )

    # If no closed container is recoverable, keep the open/text-only fallback but
    # use the best dense sub-seed when available instead of the oversized original
    # cluster.  This makes open-text candidates tighter and keeps them from
    # swallowing adjacent real bubbles.
    fallback_seed_bbox = original_seed_bbox
    fallback_refine: dict[str, Any] | None = None
    if hypotheses:
        fallback_seed_bbox, fallback_refine = hypotheses[0]
    x0, y0, x1, y1 = fallback_seed_bbox
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = max(6, min(18, int(round(bw * 0.18))))
    pad_y = max(6, min(18, int(round(bh * 0.14))))
    x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    poly = [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]
    return BubbleInstance(
        id=f"tbr-text-seed-fallback-{seed.id}", polygon=poly, confidence=max(0.74, float(seed.confidence)),
        kind='speech', block_ids=[seed.id], mask=mask, safe_mask=mask.copy(),
        meta={
            'backend': 'text_seed_fallback', 'target_only': True,
            'white_rect_recovery': {"status": "no_valid_recovery", "attempts": attempt_diags},
            'text_seed_backend': str((seed.meta or {}).get('backend', 'unknown')),
            'text_bbox': [int(v) for v in fallback_seed_bbox],
            'original_text_bbox': [int(v) for v in original_seed_bbox],
            'text_seed_refined': bool(tuple(fallback_seed_bbox) != tuple(original_seed_bbox)),
            'text_seed_refine': dict(fallback_refine or {}),
        },
    )



def _text_only_base_with_seed_support(
    bubble: BubbleInstance,
    shape: tuple[int, int],
) -> np.ndarray:
    """Recover text-seed notches in a contour-derived effect container.

    target_text_contour is built from bright-region contours. Dark JP glyphs can
    cut notches/holes into that bright component, so the raw bubble mask may
    literally exclude the first characters we need to erase. The detector
    already owns a tightly localized TARGET text bbox; for text-only processing
    we may safely admit that bbox as *search support*. This never makes the whole
    bbox transparent: _text_only_clear_mask still selects only compact dark ink.
    """
    base = _interior_base_mask(bubble, shape)
    meta = dict(bubble.meta or {})
    tb = meta.get("text_bbox")
    if str(meta.get("backend", "")).strip().lower() != "target_text_contour":
        return base
    if not isinstance(tb, (list, tuple)) or len(tb) != 4:
        return base
    h, w = shape
    x0, y0, x1, y1 = [int(round(float(v))) for v in tb]
    pad = 4
    x0 = max(0, min(w, x0 - pad)); x1 = max(0, min(w, x1 + pad))
    y0 = max(0, min(h, y0 - pad)); y1 = max(0, min(h, y1 + pad))
    if x1 <= x0 or y1 <= y0:
        return base
    # The TARGET text bbox is independent evidence and may extend a few pixels
    # beyond a threshold-derived bright contour (for example trailing ellipsis).
    # Do not clamp it back to the damaged contour; it is already tightly local.
    if x1 > x0 and y1 > y0:
        base[y0:y1, x0:x1] = 255
    return base


def _bubble_translation_evidence(
    bubble: BubbleInstance,
    aligned_source: np.ndarray,
    target: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Reject TARGET text-contour false positives before any destructive write.

    Facial features, sweat drops, clothing seams and unchanged SFX can be found by
    TARGET-only text heuristics on coloured pages.  A real translation candidate
    must have edition-specific ink on both SOURCE and TARGET.  This reuses the
    mature seed evidence gate that already rejects eye/face false positives.
    """
    meta = bubble.meta or {}
    backend = str(meta.get("backend", "") or "").strip().lower()
    if backend != "target_text_contour":
        return True, {"reason": "not_target_text_contour"}
    tb = meta.get("text_bbox")
    if not isinstance(tb, (list, tuple)) or len(tb) != 4:
        return False, {"reason": "missing_text_bbox"}
    x0, y0, x1, y1 = [int(round(float(v))) for v in tb]
    if x1 <= x0 or y1 <= y0:
        return False, {"reason": "invalid_text_bbox"}
    seed = TextBlock(
        id=f"evidence-{bubble.id}",
        polygon=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)],
        text="", confidence=float(bubble.confidence), kind="unknown",
        meta={"backend": backend},
    )
    accepted, diag, _region, _source_mask, _target_mask = _translation_evidence_for_seed(
        aligned_source, target, seed
    )
    return bool(accepted), {"reason": "translation_evidence", **diag}


def _white_container_source_translation_evidence(
    aligned_source: np.ndarray,
    target: np.ndarray,
    base_mask: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Broad SOURCE-side evidence for white containers with shifted layouts.

    A translated CN block may sit far from the JP seed, so the tight seed gate can
    legitimately miss SOURCE-exclusive ink.  For a verified neutral white
    container, it is sufficient to prove that meaningful SOURCE-exclusive text
    exists somewhere inside the whole container.  Unchanged punctuation/bubbles
    (e.g. identical ``!!``) produce no such SOURCE-exclusive ink and are skipped.
    """
    s_mask, t_mask, diag = changed_text_masks(
        aligned_source, target, base_mask,
        tolerance_px=2, min_unique_ratio=0.035, max_component_fraction=0.10,
    )
    area = max(1, int(cv2.countNonZero(base_mask)))
    sp = int(cv2.countNonZero(s_mask)); tp = int(cv2.countNonZero(t_mask))
    sr = float(sp / area); tr = float(tp / area)
    accepted = bool(sp >= 24 and sr >= 0.006)
    return accepted, {
        "accepted": accepted, "source_unique_pixels": sp, "target_unique_pixels": tp,
        "source_unique_ratio": sr, "target_unique_ratio": tr, "diff": diag,
    }


def _full_reveal_has_text_anchor(region: TransparentBubbleRegion) -> bool:
    backend = str(region.backend or "").strip().lower()
    # Internal synthetic tests use backend=test; production paths require a
    # TARGET text anchor or a detector that is intrinsically text-seeded.
    if backend == "test":
        return True
    if region.text_bbox is not None:
        return True
    return backend in {"seeded_white", "text_seed_white_rect"}


def _bbox_int(poly: list[tuple[float, float]], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = polygon_bbox(poly)
    h, w = shape
    return (
        max(0, min(w, int(np.floor(x0)))), max(0, min(h, int(np.floor(y0)))),
        max(0, min(w, int(np.ceil(x1)))), max(0, min(h, int(np.ceil(y1)))),
    )


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    if mask is None or cv2.countNonZero(mask) <= 0:
        return None
    x, y, w, h = cv2.boundingRect((mask > 0).astype(np.uint8))
    return int(x), int(y), int(x + w), int(y + h)


def _candidate_mask_stats(target: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return {
            "white_ratio": 0.0, "dark_ratio": 0.0, "compact_components": 0,
            "compact_density": 0.0, "saturation_median": 0.0, "saturation_p75": 0.0,
            "fill_ratio": 0.0, "area": 0,
        }
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    sat = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[..., 1] if target.ndim == 3 else np.zeros_like(gray)
    x0, y0, x1, y1 = bbox
    area = int(cv2.countNonZero(mask))
    fill_ratio = float(area / max(1, (x1 - x0) * (y1 - y0)))
    inner = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
    if cv2.countNonZero(inner) < 24:
        inner = mask
    inner_sel = inner > 0
    vals = gray[inner_sel]
    if vals.size <= 0:
        return {
            "white_ratio": 0.0, "dark_ratio": 0.0, "compact_components": 0,
            "compact_density": 0.0, "saturation_median": 0.0, "saturation_p75": 0.0,
            "fill_ratio": fill_ratio, "area": area,
        }
    sat_vals = sat[inner_sel]
    dark_ratio = float(np.mean(vals < 190))
    white_ratio = float(np.mean(vals > 225))
    saturation_median = float(np.median(sat_vals)) if sat_vals.size else 0.0
    saturation_p75 = float(np.percentile(sat_vals, 75.0)) if sat_vals.size else 0.0
    dark = ((gray < 190) & inner_sel).astype(np.uint8)
    cc, _labs, ccstats, _ = cv2.connectedComponentsWithStats(dark, 8)
    compact = 0
    compact_area = 0
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    inner_area = max(1, int(np.count_nonzero(inner_sel)))
    for j in range(1, cc):
        _x, _y, ww, hh, aa = [int(v) for v in ccstats[j]]
        if aa < 2 or aa > max(600, int(0.03 * inner_area)):
            continue
        if ww > 0.34 * bw or hh > 0.34 * bh:
            continue
        if max(ww / max(1.0, hh), hh / max(1.0, ww)) > 10.0:
            continue
        compact += 1
        compact_area += aa
    return {
        "white_ratio": white_ratio,
        "dark_ratio": dark_ratio,
        "compact_components": int(compact),
        "compact_density": float(compact_area / inner_area),
        "saturation_median": saturation_median,
        "saturation_p75": saturation_p75,
        "fill_ratio": fill_ratio,
        "area": area,
    }


def _mask_overlap_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int] | None) -> float:
    if bbox is None or mask is None or cv2.countNonZero(mask) <= 0:
        return 0.0
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x1 = max(x0 + 1, x1)
    y1 = max(y0 + 1, y1)
    roi = (mask[y0:y1, x0:x1] > 0)
    if roi.size <= 0:
        return 0.0
    return float(np.count_nonzero(roi) / max(1, (x1 - x0) * (y1 - y0)))


def _plain_text_seed_candidate(
    seed_bbox: tuple[int, int, int, int],
    target: np.ndarray,
    *,
    candidate_id: str,
    confidence: float,
    meta: dict[str, Any] | None = None,
) -> BubbleInstance:
    h, w = target.shape[:2]
    x0, y0, x1, y1 = [int(round(v)) for v in seed_bbox]
    bw, bh = max(1, x1 - x0), max(1, y1 - y0)
    pad_x = max(6, min(18, int(round(bw * 0.18))))
    pad_y = max(6, min(18, int(round(bh * 0.14))))
    x0 = max(0, x0 - pad_x); x1 = min(w, x1 + pad_x)
    y0 = max(0, y0 - pad_y); y1 = min(h, y1 + pad_y)
    mask = np.zeros((h, w), np.uint8)
    mask[y0:y1, x0:x1] = 255
    poly = [(float(x0), float(y0)), (float(x1), float(y0)), (float(x1), float(y1)), (float(x0), float(y1))]
    payload = dict(meta or {})
    payload.setdefault("backend", "text_seed_fallback")
    payload["text_bbox"] = [int(x0 + pad_x), int(y0 + pad_y), int(x1 - pad_x), int(y1 - pad_y)]
    payload["split_open_text_fallback"] = True
    return BubbleInstance(
        id=candidate_id, polygon=poly, confidence=float(confidence), kind="complex_text",
        block_ids=[], mask=mask, safe_mask=mask.copy(), meta=payload,
    )


def _refine_suspicious_white_candidates(
    bubbles: list[BubbleInstance],
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> tuple[list[BubbleInstance], list[dict[str, Any]]]:
    """Split mixed white/open candidates into true white bubbles and local text fallbacks.

    TARGET-only seed recovery can occasionally flood through a nearby coloured SFX
    patch and produce one oversized 'white container' that swallows a real speech
    bubble plus adjacent open text.  Before destructive planning, look for strict
    white subcomponents with real ink support.  If found, replace the oversized
    row by those true white subcomponents and optionally reintroduce a tiny local
    text fallback for the original seed when it sits outside the refined bubble.
    """
    if not bubbles:
        return bubbles, []
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target
    sat = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[..., 1] if target.ndim == 3 else np.zeros_like(gray)
    out: list[BubbleInstance] = []
    audit: list[dict[str, Any]] = []
    for row in bubbles:
        meta = dict(row.meta or {})
        backend = str(meta.get("backend", "") or "").strip().lower()
        mask = row.mask if row.mask is not None else rasterize_polygon(row.polygon, target.shape[:2])
        row.mask = mask
        if mask is None or cv2.countNonZero(mask) <= 0:
            out.append(row)
            continue
        if backend not in {"unseeded_white", "text_seed_white_container", "text_seed_white_rect"}:
            out.append(row)
            continue
        stats = _candidate_mask_stats(target, mask)
        suspicious = bool(
            backend == "text_seed_white_container"
            or stats["fill_ratio"] < 0.68
            or stats["saturation_p75"] > 28.0
            or stats["compact_components"] <= 4
        )
        if not suspicious:
            out.append(row)
            continue
        strict = (((gray >= 222) & (sat <= 26) & (mask > 0))).astype(np.uint8) * 255
        strict = cv2.morphologyEx(strict, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        strict = cv2.morphologyEx(strict, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        n, labels, ccstats, _ = cv2.connectedComponentsWithStats((strict > 0).astype(np.uint8), 8)
        if n <= 1:
            out.append(row)
            continue
        orig_area = max(1, int(cv2.countNonZero(mask)))
        text_bbox_raw = meta.get("text_bbox")
        text_bbox = tuple(int(round(float(v))) for v in text_bbox_raw) if isinstance(text_bbox_raw, (list, tuple)) and len(text_bbox_raw) == 4 else None
        refined_rows: list[tuple[float, BubbleInstance, dict[str, Any]]] = []
        for lab in range(1, n):
            x, y, w0, h0, area = [int(v) for v in ccstats[lab]]
            if area < max(120, int(round(orig_area * 0.05))):
                continue
            comp = (labels == lab).astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            filled = np.zeros_like(comp)
            cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
            bbox = _mask_bbox(filled)
            if bbox is None:
                continue
            if _is_page_furniture_bbox(bbox, target.shape[:2], cfg):
                continue
            sub_stats = _candidate_mask_stats(target, filled)
            if sub_stats["white_ratio"] < 0.74 or sub_stats["dark_ratio"] < 0.002 or sub_stats["dark_ratio"] > 0.28:
                continue
            if sub_stats["saturation_p75"] > 18.0:
                continue
            if sub_stats["compact_components"] < 1 and sub_stats["compact_density"] < 0.0035:
                continue
            overlap = _mask_overlap_ratio(filled, text_bbox)
            score = (
                sub_stats["white_ratio"] * 2.2 + min(1.0, sub_stats["compact_components"] / 10.0)
                + min(0.45, sub_stats["compact_density"] * 28.0) + min(0.35, area / orig_area)
                + min(0.18, overlap)
            )
            poly = mask_to_largest_polygon(filled)
            if len(poly) < 3:
                continue
            new_meta = dict(meta)
            new_meta["white_candidate_refined"] = True
            new_meta["refined_from_backend"] = backend
            new_meta["refined_component_stats"] = sub_stats
            if text_bbox is None or overlap >= 0.08:
                new_meta["text_bbox"] = [int(v) for v in text_bbox] if text_bbox is not None else new_meta.get("text_bbox")
            else:
                new_meta.pop("text_bbox", None)
            refined_rows.append((score, BubbleInstance(
                id=f"{row.id}-split-{len(refined_rows):02d}", polygon=poly, confidence=max(float(row.confidence), 0.80),
                kind="speech", block_ids=list(row.block_ids or []), mask=filled, safe_mask=filled.copy(), meta=new_meta,
            ), {"bbox": [int(v) for v in bbox], "overlap_with_seed_bbox": overlap, **sub_stats}))
        if not refined_rows:
            out.append(row)
            continue
        refined_rows.sort(key=lambda item: item[0], reverse=True)
        chosen: list[BubbleInstance] = []
        kept_diags: list[dict[str, Any]] = []
        for _score, cand, diag in refined_rows:
            if any(_mask_iou(cand.mask, ex.mask) >= 0.25 for ex in chosen):
                continue
            chosen.append(cand)
            kept_diags.append(diag)
        if not chosen:
            out.append(row)
            continue
        for cand in chosen:
            out.append(cand)
        fallback_added = False
        if text_bbox is not None and not any(_mask_overlap_ratio(cand.mask, text_bbox) >= 0.08 for cand in chosen):
            seed = TextBlock(
                id=f"{row.id}-split-seed", polygon=[(text_bbox[0], text_bbox[1]), (text_bbox[2], text_bbox[1]), (text_bbox[2], text_bbox[3]), (text_bbox[0], text_bbox[3])],
                text="", confidence=float(row.confidence), kind="unknown", meta={"backend": backend},
            )
            if _seed_fallback_candidate_safe(target, seed, cfg):
                fallback_meta = dict(meta)
                fallback_meta["refined_from_mixed_white_candidate"] = True
                out.append(_plain_text_seed_candidate(text_bbox, target, candidate_id=f"{row.id}-split-open-text", confidence=max(0.72, float(row.confidence)), meta=fallback_meta))
                fallback_added = True
        audit.append({
            "backend": backend,
            "status": "candidate_split_white_refine",
            "candidate_id": str(row.id),
            "original_bbox": [int(v) for v in (_mask_bbox(mask) or (0, 0, 0, 0))],
            "refined_count": len(chosen),
            "open_text_fallback_added": fallback_added,
            "refined": kept_diags,
        })
    return out, audit


def build_transparent_bubble_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    cfg: TransparentBubbleRevealConfig,
    *,
    bubble_config: BubbleConfig | None = None,
    stage_cache=None,
    cache_stats: dict[str, Any] | None = None,
    source_path: str | None = None,
    target_path: str | None = None,
    cache_enabled: bool = True,
    target_text_ocr: Any | None = None,
    semantic_config: Any | None = None,
) -> TransparentBubblePlan:
    aligned = warp_source_to_target(source, registration)
    valid = _valid_warp_mask(source, registration)
    clear = np.zeros(target.shape[:2], dtype=np.uint8)
    gate_ok, gate_reason = _registration_gate(registration, cfg)
    diagnostics: dict[str, Any] = {
        "registration_confidence": float(registration.confidence),
        "reprojection_error": float(registration.reprojection_error),
        "inlier_ratio": float(registration.inlier_ratio),
        "registration_gate": gate_reason,
        "source_detector_used": False,
        "target_only_detection": True,
    }
    if not gate_ok:
        return TransparentBubblePlan(False, f"rejected_registration:{gate_reason}", aligned, valid, clear, [], diagnostics, "REJECT")

    semantic_layout = analyze_semantic_layout(target, semantic_config, role="target")
    semantic_enabled = bool(semantic_config is not None and getattr(semantic_config, "enabled", False) and getattr(semantic_config, "apply_to_reveal", True))
    diagnostics["semantic_layout"] = semantic_layout.to_dict()
    diagnostics["semantic_enabled_for_reveal"] = bool(semantic_enabled and semantic_layout.available)

    bubbles, audit = _detect_target_bubbles(
        target, cfg, bubble_config, stage_cache=stage_cache, cache_stats=cache_stats,
        target_path=target_path, cache_enabled=cache_enabled,
    )
    bubbles, attach_audit = _attach_text_seed_bbox_to_containers(
        bubbles, target, cfg, bubble_config, stage_cache=stage_cache, cache_stats=cache_stats,
        target_path=target_path, cache_enabled=cache_enabled,
    )
    audit.extend(attach_audit)
    bubbles, fallback_audit = _add_verified_seed_fallbacks(
        bubbles, aligned, target, cfg, bubble_config, stage_cache=stage_cache,
        cache_stats=cache_stats, target_path=target_path, cache_enabled=cache_enabled,
    )
    audit.extend(fallback_audit)
    bubbles, refine_audit = _refine_suspicious_white_candidates(bubbles, target, cfg)
    audit.extend(refine_audit)
    diagnostics["detector_audit"] = audit
    diagnostics["detected_bubbles"] = len(bubbles)
    diagnostics["verified_seed_fallbacks"] = sum(1 for b in bubbles if str((b.meta or {}).get("backend", "")) == "text_seed_fallback")
    try:
        support_seeds, support_seed_audit = _target_text_seed_blocks(
            target, cfg, bubble_config, stage_cache=stage_cache, cache_stats=cache_stats,
            target_path=target_path, cache_enabled=cache_enabled,
        )
    except Exception as exc:
        support_seeds, support_seed_audit = [], [{"backend": "target_text_presence", "status": "unavailable", "error": str(exc)}]
    diagnostics["target_text_seed_audit"] = support_seed_audit
    diagnostics["support_seed_count"] = len(support_seeds)
    if not bubbles:
        return TransparentBubblePlan(False, "no_target_bubbles", aligned, valid, clear, [], diagnostics, "REJECT")

    regions: list[TransparentBubbleRegion] = []
    page_area = max(1, int(target.shape[0] * target.shape[1]))
    requested_mode = str(cfg.clear_mode or "full_bubble").strip().lower()
    if requested_mode not in {"full_bubble", "text_only", "hybrid"}:
        return TransparentBubblePlan(False, "unsupported_clear_mode", aligned, valid, clear, [], diagnostics, "REJECT")

    candidates: list[TransparentBubbleRegion] = []
    skipped_page_furniture = 0
    skipped_target_text = 0
    skipped_ocr_text = 0
    ocr_unavailable = 0
    skipped_source_evidence = 0
    skipped_semantic = 0
    reviewed_semantic = 0
    for i, bubble in enumerate(bubbles):
        candidate_bbox = _bbox_int(list(bubble.polygon), target.shape[:2])
        if bool(getattr(cfg, "suppress_page_furniture", True)) and _is_page_furniture_bbox(candidate_bbox, target.shape[:2], cfg):
            skipped_page_furniture += 1
            audit.append({
                "backend": str((bubble.meta or {}).get("backend", cfg.bubble_backend or "")),
                "status": "candidate_rejected_page_furniture",
                "candidate_id": str(bubble.id),
                "bbox": [int(v) for v in candidate_bbox],
            })
            continue
        base = _interior_base_mask(bubble, target.shape[:2])
        semantic_decision = decide_candidate(
            base, semantic_layout, strategy=str(getattr(semantic_config, "strategy", "auto"))
        ) if semantic_enabled and semantic_layout.available else None
        if semantic_decision is not None and semantic_decision.action == "DROP":
            skipped_semantic += 1
            audit.append({
                "backend": str((bubble.meta or {}).get("backend", cfg.bubble_backend or "")),
                "status": "candidate_rejected_semantic_layout",
                "candidate_id": str(bubble.id),
                "semantic": semantic_decision.to_dict(),
            })
            continue
        if semantic_decision is not None and semantic_decision.action == "REVIEW":
            reviewed_semantic += 1
        mode = requested_mode
        backend_name = str((bubble.meta or {}).get("backend", cfg.bubble_backend or ""))
        # Whole-page alignment mode is container-authoritative: once a bubble or
        # text box has been detected safely on TARGET, it may be processed even if
        # SOURCE-exclusive translation evidence is weak or absent.  The SOURCE
        # evidence can optionally be restored as a strict gate.
        base_is_neutral_white = bool(
            _bubble_is_white(target, base)
            or _neutral_white_full_reveal_safe(target, base)
        )
        target_text_ok, target_text_diag = _candidate_target_text_presence(target, bubble, support_seeds, cfg)
        ocr_text_ok, ocr_text_diag = _candidate_ocr_text_presence(
            target, bubble, cfg, target_text_ocr, target_path=target_path,
        )
        evidence_ok, evidence_diag = _source_translation_evidence_for_candidate(
            bubble, aligned, target, base, prefer_white_container=base_is_neutral_white,
        )
        bubble.meta = dict(bubble.meta or {})
        bubble.meta.update({
            "target_text_presence": target_text_diag,
            "target_ocr_text_presence": ocr_text_diag,
            "semantic_layout_decision": semantic_decision.to_dict() if semantic_decision is not None else None,
            "translation_evidence": evidence_diag,
            "translation_evidence_optional": not bool(getattr(cfg, "require_source_translation_evidence", False)),
        })
        verify_target = bool(getattr(cfg, "verify_target_text_presence", True))
        use_ocr_presence = bool(getattr(cfg, "target_text_presence_ocr_enabled", False))
        if use_ocr_presence:
            if ocr_text_ok is True:
                # OCR positive may rescue stylized text that the cheap component
                # verifier cannot confidently classify.
                combined_target_text_ok = True
            elif ocr_text_ok is False:
                combined_target_text_ok = False
            else:
                ocr_unavailable += 1
                combined_target_text_ok = bool(target_text_ok) if bool(getattr(cfg, "target_text_presence_ocr_fail_open", True)) else False
        else:
            combined_target_text_ok = bool(target_text_ok)
        if verify_target and not combined_target_text_ok:
            if use_ocr_presence and ocr_text_ok is False:
                skipped_ocr_text += 1
            else:
                skipped_target_text += 1
            continue
        if bool(getattr(cfg, "require_source_translation_evidence", False)) and not evidence_ok:
            skipped_source_evidence += 1
            continue
        if backend_name == "text_seed_fallback":
            mode = "full_bubble" if base_is_neutral_white else "text_only"
        elif mode == "hybrid" and backend_name in {"text_seed_white_container", "text_seed_white_rect"}:
            # These backends only exist after the strict seeded-container quality
            # gate has recovered a closed neutral-white container. Treat that
            # recovery as authoritative and clear the whole bubble interior rather
            # than degrading it back to open/text_only.
            mode = "full_bubble"
        elif mode == "hybrid":
            mode = "full_bubble" if (
                _flat_white_full_reveal_safe(target, base)
                or _neutral_white_full_reveal_safe(target, base)
            ) else "text_only"
        if mode == "full_bubble":
            local = _full_bubble_clear_mask(base, cfg)
        else:
            local = _candidate_text_only_clear_mask(bubble, aligned, target, cfg)
            if semantic_enabled and semantic_layout.available:
                local = constrain_text_only_mask(
                    local, semantic_layout, pad=int(getattr(semantic_config, "semantic_roi_pad_px", 10))
                )
        # Never reveal synthetic white warp border when SOURCE does not cover that
        # TARGET location.
        local[valid == 0] = 0
        pixels = int(cv2.countNonZero(local))
        ratio = pixels / page_area
        triage = "SAFE"
        reason = "ok"
        if float(bubble.confidence) < max(float(cfg.min_bubble_confidence), 0.50):
            triage, reason = "REVIEW", "low_bubble_confidence"
        if ratio > float(cfg.max_clear_area_ratio):
            triage, reason = "REVIEW", "single_region_over_page_cap"
        if pixels <= 0:
            triage, reason = "REJECT", "empty_clear_region"
        elif semantic_decision is not None and semantic_decision.action == "REVIEW" and triage == "SAFE":
            triage, reason = "REVIEW", "semantic_review_region"
        _tb = (bubble.meta or {}).get("text_bbox")
        text_bbox = tuple(int(round(float(v))) for v in _tb) if isinstance(_tb, (list, tuple)) and len(_tb) == 4 else None
        candidates.append(TransparentBubbleRegion(
            id=f"transparent-{i:04d}", target_bbox=_bbox_int(list(bubble.polygon), target.shape[:2]),
            polygon=list(bubble.polygon), clear_mask=local, confidence=float(bubble.confidence),
            backend=str((bubble.meta or {}).get("backend", cfg.bubble_backend)), triage=triage,
            reason=reason, applied=pixels > 0, clear_mode=mode, text_bbox=text_bbox,
        ))

    # Enforce the per-page destructive cap without cutting a bubble in half. High
    # confidence/small regions win; any omitted region is surfaced as REVIEW.
    cap_pixels = max(0, int(round(float(cfg.max_clear_area_ratio) * page_area)))
    running = 0
    cap_triggered = False
    order = sorted(range(len(candidates)), key=lambda i: (-candidates[i].confidence, cv2.countNonZero(candidates[i].clear_mask)))
    chosen: set[int] = set()
    for idx in order:
        region = candidates[idx]
        count = int(cv2.countNonZero(region.clear_mask))
        if count <= 0:
            region.applied = False
            continue
        candidate_union = cv2.bitwise_or(clear, region.clear_mask)
        union_count = int(cv2.countNonZero(candidate_union))
        if cap_pixels > 0 and union_count > cap_pixels:
            region.applied = False
            region.triage = "REVIEW"
            region.reason = "page_clear_area_cap"
            cap_triggered = True
            continue
        clear = candidate_union
        running = union_count
        chosen.add(idx)

    regions = candidates
    applied = [r for r in regions if r.applied]
    if not applied:
        diagnostics.update({"clear_pixels": 0, "clear_area_ratio": 0.0, "page_cap_triggered": cap_triggered})
        # Area-gate rejection is reviewable rather than a structural failure: the
        # page stays unchanged, but the operator can deliberately raise the cap or
        # switch to text_only. Registration/no-bubble failures remain REJECT.
        if cap_triggered:
            return TransparentBubblePlan(False, "page_clear_area_cap", aligned, valid, clear, regions, diagnostics, "REVIEW")
        return TransparentBubblePlan(False, "no_accepted_regions", aligned, valid, clear, regions, diagnostics, "REJECT")

    page_triage = "REVIEW" if cap_triggered or any(r.triage == "REVIEW" for r in applied) or any((not r.applied and r.triage == "REVIEW") for r in regions) else "SAFE"
    if cap_triggered and bool(cfg.force_review_if_over_ratio):
        page_triage = "REVIEW"
    diagnostics.update({
        "clear_pixels": int(running),
        "clear_area_ratio": float(running / page_area),
        "page_cap_pixels": int(cap_pixels),
        "page_cap_triggered": bool(cap_triggered),
        "applied_region_count": len(applied),
        "review_region_count": sum(1 for r in regions if r.triage == "REVIEW"),
        "skipped_by_page_furniture": int(skipped_page_furniture),
        "skipped_by_target_text_presence": int(skipped_target_text),
        "skipped_by_ocr_text_presence": int(skipped_ocr_text),
        "ocr_text_presence_unavailable": int(ocr_unavailable),
        "skipped_by_source_translation_evidence": int(skipped_source_evidence),
        "skipped_by_semantic_layout": int(skipped_semantic),
        "reviewed_by_semantic_layout": int(reviewed_semantic),
        "semantic_provider": semantic_layout.provider,
        "semantic_block_count": len(semantic_layout.blocks),
        "suppress_page_furniture": bool(getattr(cfg, "suppress_page_furniture", True)),
        "verify_target_text_presence": bool(getattr(cfg, "verify_target_text_presence", True)),
        "target_text_presence_ocr_enabled": bool(getattr(cfg, "target_text_presence_ocr_enabled", False)),
        "require_source_translation_evidence": bool(getattr(cfg, "require_source_translation_evidence", False)),
        "backend": str(cfg.bubble_backend),
        "clear_mode": requested_mode,
    })
    reason = "page_clear_area_cap" if cap_triggered else "ok"
    return TransparentBubblePlan(True, reason, aligned, valid, clear, regions, diagnostics, page_triage)


def reject_transparent_bubble_plan(
    source: np.ndarray,
    target: np.ndarray,
    registration: RegistrationResult,
    reason: str,
) -> TransparentBubblePlan:
    aligned = warp_source_to_target(source, registration)
    valid = _valid_warp_mask(source, registration)
    return TransparentBubblePlan(
        False, str(reason), aligned, valid, np.zeros(target.shape[:2], dtype=np.uint8), [],
        {"target_only_detection": True, "source_detector_used": False, "forced_reject": True}, "REJECT",
    )


def _alpha_from_clear_mask(clear_mask: np.ndarray, feather_px: int) -> np.ndarray:
    clear = (clear_mask > 0).astype(np.uint8) * 255
    feather = max(0, int(feather_px))
    if feather <= 0:
        return (255 - clear).astype(np.uint8)
    k = max(3, feather * 2 + 1)
    if k % 2 == 0:
        k += 1
    soft = cv2.GaussianBlur(clear, (k, k), sigmaX=max(0.5, feather * 0.55), sigmaY=max(0.5, feather * 0.55))
    return (255 - soft).astype(np.uint8)


def _estimate_region_paper_color(
    target: np.ndarray,
    gray_t: np.ndarray,
    mask: np.ndarray,
    paper_thr: int,
) -> np.ndarray:
    paper_sample = mask & (gray_t >= int(paper_thr))
    if np.count_nonzero(paper_sample) >= 20:
        return np.median(target[paper_sample], axis=0).astype(np.uint8)
    ring = cv2.dilate(mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1) > 0
    ring &= ~mask
    if np.count_nonzero(ring) >= 20:
        return np.median(target[ring], axis=0).astype(np.uint8)
    return np.array([250, 250, 250], dtype=np.uint8)


def _region_source_text_mask(gray_c: np.ndarray, region_mask: np.ndarray, cfg: TransparentBubbleRevealConfig) -> np.ndarray:
    thr = int(max(getattr(cfg, "composite_ink_threshold", 110), min(225, getattr(cfg, "text_ink_threshold", 190))))
    mask = ((gray_c <= thr) & region_mask).astype(np.uint8) * 255
    dilate_px = max(0, int(getattr(cfg, "ink_dilate_px", 1)))
    if dilate_px > 0 and cv2.countNonZero(mask) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        mask = cv2.dilate(mask, k)
        mask[~region_mask] = 0
    return mask


def _full_bubble_target_text_completion_mask(
    target: np.ndarray,
    region: TransparentBubbleRegion,
    region_mask: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> np.ndarray:
    """Recover JP glyph pixels accidentally excluded from a full-bubble reveal.

    The closed-bubble-priority recovery added in v2.1.06 correctly upgrades some
    large text seeds into a true white bubble, but ``_resolve_full_bubble_reveal_mask``
    intentionally keeps only verified paper and can omit tiny dark punctuation at
    the edge of that paper mask.  Those omitted TARGET pixels then survive on top
    of the revealed Chinese page (the two stray dots seen on p-044).

    Complete only compact printed ink inside the already verified text bbox and
    inside the trusted bubble region.  Long border/panel rules are rejected by
    ``_tight_verified_text_ink_mask`` and the bubble perimeter is protected again
    below, so this cannot turn into a broad artwork erase.
    """
    base = ((region_mask > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(base) <= 0 or region.text_bbox is None:
        return np.zeros(base.shape, np.uint8)
    # Do NOT clip the support to the recovered bubble mask.  The regression we
    # are fixing is precisely a few JP glyph pixels that sit just outside the
    # recovered white-container polygon even though they are still inside the
    # detector's verified text bbox.
    support = _region_text_bbox_support(region, base.shape, pad=4)
    if cv2.countNonZero(support) <= 0:
        return np.zeros(base.shape, np.uint8)
    ink = _tight_verified_text_ink_mask(target, support)
    if cv2.countNonZero(ink) <= 0:
        return ink
    # Only admit text lying in a tiny halo around the trusted white container;
    # this clears clipped punctuation without re-opening the oversized original
    # seed or touching anatomy elsewhere in the text bbox.
    halo = cv2.dilate(base, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    ink = cv2.bitwise_and(ink, halo)
    if cv2.countNonZero(ink) > 0:
        ink = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        ink = cv2.bitwise_and(ink, halo)
    # _tight_verified_text_ink_mask already rejects long horizontal/vertical
    # rules, so do not apply a broad border dilation here; that was exactly what
    # could protect the tiny punctuation we need to remove near the panel edge.
    return ink


def _resolve_full_bubble_reveal_mask(
    target: np.ndarray,
    gray_c: np.ndarray,
    region_mask: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> np.ndarray:
    """Return only a verified neutral-paper *interior* for full reveal.

    Never fall back to the raw detector mask.  Raw fallback was the path that
    allowed skin/face or neighbouring artwork to become transparent when a
    TARGET text heuristic produced a false/oversized region.  The hole is now:
      1. recovered TARGET neutral paper,
      2. clipped to low-saturation paper/ink pixels,
      3. inset from the candidate perimeter, and
      4. stripped of TARGET border/rule geometry.
    If any of those checks fail, return an empty mask and let the caller demote
    the region instead of exposing SOURCE artwork.
    """
    region_u8 = (region_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(region_u8) == 0:
        return np.zeros(region_mask.shape, dtype=bool)
    source_text = _region_source_text_mask(gray_c, region_mask > 0, cfg)
    paper = white_container_paper_mask(target, region_u8, source_text)
    if cv2.countNonZero(paper) == 0:
        return np.zeros(region_mask.shape, dtype=bool)

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    # Neutral paper and black/gray printed ink are low saturation.  Coloured skin,
    # hair highlights, clothing and effect fields are excluded even if contour
    # filling would otherwise bridge across them.
    neutral = hsv[..., 1] <= 78
    safe = ((paper > 0) & neutral & (region_u8 > 0)).astype(np.uint8) * 255
    region_pixels = int(cv2.countNonZero(region_u8))
    safe_pixels = int(cv2.countNonZero(safe))
    if safe_pixels < max(32, int(round(region_pixels * 0.08))):
        return np.zeros(region_mask.shape, dtype=bool)

    if bool(getattr(cfg, "protect_border", True)):
        guard = max(1, int(getattr(cfg, "border_protect_px", 2)))
        # Immutable perimeter band: the target outline/tail must never be sourced
        # from the lower CN page, even when the detector polygon is slightly off.
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * guard + 1, 2 * guard + 1))
        interior = cv2.erode(region_u8, k, iterations=1)
        safe = cv2.bitwise_and(safe, interior)
        border = target_container_border_mask(target, region_u8, band_px=max(4, guard + 2))
        if cv2.countNonZero(border) > 0:
            protected = cv2.dilate(border, k, iterations=1)
            safe[protected > 0] = 0

    # Do not permit isolated coloured/texture holes created by morphology.
    safe[(hsv[..., 1] > 78)] = 0
    return safe > 0

def _text_only_region_envelope(region: TransparentBubbleRegion, shape: tuple[int, int], cfg: TransparentBubbleRevealConfig) -> np.ndarray:
    """Build a local search envelope around proven TARGET text seeds.

    For open/effect text the raw bright-contour region can be much larger than
    the actual text corridor.  Use the detector's text bbox as the primary cap
    when it exists, then only allow a modest expansion around it.  This keeps
    coloured burst fields from acquiring oversized cleanup windows while still
    allowing trailing punctuation/ellipsis near the seed to be found.
    """
    h, w = shape
    seed = ((region.clear_mask > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(seed) == 0:
        return np.zeros((h, w), np.uint8)

    bbox = region.text_bbox if region.text_bbox is not None else region.target_bbox
    x0, y0, x1, y1 = [int(v) for v in bbox]
    bw = max(1, x1 - x0)
    bh = max(1, y1 - y0)
    span = max(1, int(max(bw, bh)))
    effect_like = str(region.backend or '').strip().lower() == 'target_text_contour'
    radius = max(8, min(34 if effect_like else 52, int(round(span * (0.12 if effect_like else 0.18)))))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    env = cv2.dilate(seed, k)

    pad_x = max(6, min(18 if effect_like else 28, int(round(bw * (0.16 if effect_like else 0.24)))))
    pad_y = max(8, min(26 if effect_like else 34, int(round(bh * (0.18 if effect_like else 0.28)))))
    cx0 = max(0, x0 - pad_x); cy0 = max(0, y0 - pad_y)
    cx1 = min(w, x1 + pad_x); cy1 = min(h, y1 + pad_y)
    cap = np.zeros((h, w), np.uint8)
    if cx1 > cx0 and cy1 > cy0:
        cap[cy0:cy1, cx0:cx1] = 255
        env = cv2.bitwise_and(env, cap)
    return env



def _local_paper_reconstruct_under_text(
    target: np.ndarray,
    text_mask: np.ndarray,
    envelope: np.ndarray,
    *,
    dilate_px: int = 2,
    ring_px: int = 8,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fill proven text strokes from nearby bright TARGET paper samples.

    This is intentionally component-local: it cannot recreate the old Japanese
    glyph from neighbouring strokes (a Telea/Gaussian failure seen on 来), and it
    does not smear black burst rays into a white text area.
    """
    forced = ((text_mask > 0) & (envelope > 0)).astype(np.uint8) * 255
    d = max(0, int(dilate_px))
    if d > 0 and cv2.countNonZero(forced) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        forced = cv2.dilate(forced, k)
        forced[envelope == 0] = 0
    out = target.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((forced > 0).astype(np.uint8), 8)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    filled = 0
    fallback = 0
    rp = max(4, int(ring_px))
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rp + 1, 2 * rp + 1))
    for lab in range(1, n):
        comp = labels == lab
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        ring = cv2.dilate(comp.astype(np.uint8), ring_kernel) > 0
        ring &= (~comp) & (envelope > 0)
        if np.count_nonzero(ring) < 8:
            fallback += area
            continue
        rg = gray[ring]
        rs = hsv[..., 1][ring]
        # Prefer the brighter half of the local ring to reject text, burst rays,
        # outlines and hair while retaining the actual paper colour/gradient.
        q = float(np.percentile(rg, 55.0))
        good = ring & (gray >= max(150.0, q)) & (hsv[..., 1] <= max(90.0, float(np.percentile(rs, 82.0))))
        pix = target[good]
        if len(pix) < 8:
            pix = target[ring & (gray >= q)]
        if len(pix) < 4:
            fallback += area
            continue
        color = np.median(pix, axis=0).astype(np.uint8)
        out[comp] = color
        filled += area
    # Rare tiny components without enough ring support use normalized TARGET
    # reconstruction, still restricted to the same proven text mask.
    if fallback:
        missing = (forced > 0) & np.all(out == target, axis=2)
        if np.any(missing):
            tmp, _ = _smooth_reconstruct_under_text(target, missing.astype(np.uint8) * 255, envelope, dilate_px=0, sigma=4.0)
            out[missing] = tmp[missing]
    return out, forced, {
        "local_paper_filled_pixels": int(filled),
        "local_paper_fallback_pixels": int(fallback),
    }


def _preclear_background_is_paper_like(target: np.ndarray, text_mask: np.ndarray, envelope: np.ndarray) -> bool:
    use = (text_mask > 0) & (envelope > 0)
    if not np.any(use):
        return False
    ring = cv2.dilate(use.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))) > 0
    ring &= (~use) & (envelope > 0)
    if np.count_nonzero(ring) < 32:
        return False
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    vals = gray[ring]
    sats = hsv[..., 1][ring]
    return bool(float(np.mean(vals >= 190)) >= 0.54 and float(np.percentile(sats, 80.0)) <= 85.0)


def _smooth_reconstruct_under_text(
    target: np.ndarray,
    text_mask: np.ndarray,
    envelope: np.ndarray,
    *,
    dilate_px: int = 3,
    sigma: float = 7.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct smooth coloured paper under lettering without SOURCE pixels.

    Gaussian normalized convolution ignores the masked lettering while
    interpolating the surrounding TARGET colour/gradient. Unlike Telea on a
    tall text column, it does not create visible blocky/white patches in smooth
    purple/pink/yellow effect frames.
    """
    forced = ((text_mask > 0) & (envelope > 0)).astype(np.uint8) * 255
    d = max(0, int(dilate_px))
    if d > 0 and cv2.countNonZero(forced) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        forced = cv2.dilate(forced, k)
        forced[envelope == 0] = 0
    use = forced > 0
    if not np.any(use):
        return target.copy(), forced
    valid = (~use).astype(np.float32)
    s = max(2.0, float(sigma))
    den = cv2.GaussianBlur(valid, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
    out = target.astype(np.float32).copy()
    for c in range(3):
        src = target[..., c].astype(np.float32)
        num = cv2.GaussianBlur(src * valid, (0, 0), sigmaX=s, sigmaY=s, borderType=cv2.BORDER_REFLECT)
        bg = num / np.maximum(den, 1e-4)
        out[..., c][use] = bg[use]
    return np.clip(out, 0, 255).astype(np.uint8), forced



def _split_boundary_geometry_from_text_mask(
    text_mask: np.ndarray,
    surface_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Separate true container/tail geometry accidentally classified as text.

    ``target_white_container_text_mask`` intentionally searches compact dark ink
    across the neutral paper surface.  On speech balloons, a triangular tail or
    a short oval-outline segment can lie *inside* that paper surface and therefore
    look like a glyph component.  If text is always allowed to beat structure
    protection, those outline pixels get erased together with JP lettering.

    Boundary geometry has a stronger invariant than text: it is thin/line-like,
    touches the paper boundary band materially, and spans farther than ordinary
    punctuation.  Reclassify only those conservative components.  The returned
    geometry mask is slightly AA-expanded and becomes immutable TARGET structure.
    """
    if text_mask.shape != surface_mask.shape:
        raise ValueError("boundary/text split inputs must share shape")
    text = ((text_mask > 0) & (surface_mask > 0)).astype(np.uint8) * 255
    geometry = np.zeros_like(text)
    if cv2.countNonZero(text) == 0:
        return text, geometry, {"boundary_geometry_components": 0, "boundary_geometry_pixels": 0}

    use = (surface_mask > 0).astype(np.uint8)
    ys, xs = np.where(use > 0)
    if xs.size == 0:
        return text, geometry, {"boundary_geometry_components": 0, "boundary_geometry_pixels": 0}
    rw = max(1, int(xs.max() - xs.min() + 1))
    rh = max(1, int(ys.max() - ys.min() + 1))
    band_px = max(5, min(14, int(round(min(rw, rh) * 0.055))))
    er = cv2.erode(
        use,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)),
        iterations=1,
    )
    boundary_band = (use > 0) & (er == 0)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((text > 0).astype(np.uint8), 8)
    kept = 0
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area < 12:
            continue
        comp = labels == lab
        boundary_pixels = int(np.count_nonzero(comp & boundary_band))
        if boundary_pixels < max(4, int(round(area * 0.18))):
            continue
        max_dim = max(bw, bh)
        min_dim = max(1, min(bw, bh))
        aspect = float(max_dim / min_dim)
        fill = float(area / max(1, bw * bh))
        span = max(float(bw / rw), float(bh / rh))
        # A real tail/outline component is a comparatively long sparse stroke.
        # Kanji/kana blobs are denser; tiny punctuation is rejected by area/span.
        line_like = bool(
            max_dim >= max(14, int(round(min(rw, rh) * 0.075)))
            and span >= 0.075
            and (aspect >= 1.75 or fill <= 0.30)
            and fill <= 0.38
        )
        if not line_like:
            continue
        geometry[comp] = 255
        text[comp] = 0
        kept += 1

    if cv2.countNonZero(geometry) > 0:
        # Preserve antialias around the structural stroke, but do not grow deep
        # into the paper where nearby JP text may sit.
        geometry = cv2.dilate(
            geometry,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        near_boundary = cv2.dilate(
            boundary_band.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )
        geometry[near_boundary == 0] = 0
        text[geometry > 0] = 0
    return text, geometry, {
        "boundary_geometry_components": int(kept),
        "boundary_geometry_pixels": int(cv2.countNonZero(geometry)),
    }


def _target_linear_structure_guard(
    target: np.ndarray,
    region_mask: np.ndarray,
    *,
    min_line_px: int | None = None,
    guard_px: int = 2,
) -> np.ndarray:
    """Protect TARGET note/bubble borders and tape-like long line art.

    The whole-page transparent route must never erase structural lines merely
    because they intersect a text seed or SOURCE ink candidate.  This guard is
    TARGET-only and intentionally conservative: it keeps the mature container
    boundary detector plus long Hough line segments, while ordinary glyph
    strokes are too short to qualify.
    """
    if target.shape[:2] != region_mask.shape:
        raise ValueError("linear structure guard inputs must share shape")
    use = (region_mask > 0).astype(np.uint8) * 255
    if cv2.countNonZero(use) == 0:
        return np.zeros_like(region_mask)

    border = target_container_border_mask(target, use, band_px=5)
    ys, xs = np.where(use > 0)
    if xs.size == 0:
        return border
    rw = max(1, int(xs.max() - xs.min() + 1))
    rh = max(1, int(ys.max() - ys.min() + 1))
    line_floor = int(min_line_px) if min_line_px is not None else max(20, min(72, int(round(max(rw, rh) * 0.12))))

    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) if target.ndim == 3 else target.astype(np.uint8)
    # Canny is used only to locate long geometry; all accepted pixels are still
    # clipped to the trusted region and later restored from TARGET byte-for-byte.
    edges = cv2.Canny(gray, 70, 170, apertureSize=3, L2gradient=True)
    edges[use == 0] = 0
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(12, int(round(line_floor * 0.45))),
        minLineLength=max(12, line_floor),
        maxLineGap=5,
    )
    linear = np.zeros_like(region_mask)
    # Only structural lines tied to the paper/container boundary are immutable.
    # Central text columns can also look like long Hough segments, so never guard
    # a line solely because it is long.
    band_px = max(8, min(28, int(round(min(rw, rh) * 0.08))))
    er = cv2.erode(
        (use > 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (band_px * 2 + 1, band_px * 2 + 1)),
        iterations=1,
    )
    boundary_band = (use > 0) & (er == 0)
    if lines is not None:
        # OpenCV builds do not agree on the exact HoughLinesP container shape.
        # Typical outputs are (N, 1, 4), but some macOS/OpenCV wheels return
        # (N, 4), and a single detected segment can even arrive as (4,).  Always
        # normalize to a flat list of [x0, y0, x1, y1] rows before indexing.
        raw_lines = np.asarray(lines)
        if raw_lines.size >= 4 and raw_lines.size % 4 == 0:
            line_rows = raw_lines.reshape(-1, 4)
        else:
            line_rows = np.empty((0, 4), dtype=np.int32)
        thickness = max(1, int(guard_px) * 2 + 1)
        for row in line_rows:
            x0, y0, x1, y1 = [int(v) for v in row]
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length < float(line_floor):
                continue
            cand = np.zeros_like(region_mask)
            cv2.line(cand, (x0, y0), (x1, y1), 255, thickness=thickness, lineType=cv2.LINE_AA)
            cand[use == 0] = 0
            cp = int(cv2.countNonZero(cand))
            if cp <= 0:
                continue
            boundary_support = int(np.count_nonzero((cand > 0) & boundary_band))
            endpoint_support = bool(
                (0 <= y0 < boundary_band.shape[0] and 0 <= x0 < boundary_band.shape[1] and boundary_band[y0, x0])
                or (0 <= y1 < boundary_band.shape[0] and 0 <= x1 < boundary_band.shape[1] and boundary_band[y1, x1])
            )
            if not endpoint_support and (boundary_support / max(1, cp)) < 0.12:
                continue
            linear[cand > 0] = 255
    linear[use == 0] = 0
    guard = cv2.bitwise_or(border, linear)
    if cv2.countNonZero(guard) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        guard = cv2.dilate(guard, k, iterations=1)
        guard[use == 0] = 0
    return guard



def _effect_text_bbox_mask(region: TransparentBubbleRegion, shape: tuple[int, int], *, pad_x: int = 8, pad_y: int = 12) -> np.ndarray:
    """Return a tight text corridor for open/effect text.

    Inspired by BallonsTranslator's text-block-bounded mask generation: the
    destructive mask is anchored to the detector text box rather than the full
    effect/bubble contour.  That means burst rays, panel art and colour fields
    outside the text box are never candidates for erasure.
    """
    h, w = shape
    bbox = region.text_bbox if region.text_bbox is not None else region.target_bbox
    x0, y0, x1, y1 = [int(round(v)) for v in bbox]
    x0 = max(0, x0 - max(0, int(pad_x))); x1 = min(w, x1 + max(0, int(pad_x)))
    y0 = max(0, y0 - max(0, int(pad_y))); y1 = min(h, y1 + max(0, int(pad_y)))
    out = np.zeros((h, w), np.uint8)
    if x1 > x0 and y1 > y0:
        out[y0:y1, x0:x1] = 255
    return out


def _refine_effect_target_text_mask(
    target: np.ndarray,
    region: TransparentBubbleRegion,
    envelope: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Extract TARGET JP glyphs inside an open/effect text box.

    This intentionally follows the two useful ideas shared by MTU and
    BallonsTranslator:
      * fit connected components to a known text line/box instead of erasing the
        whole detected region;
      * dilate only the accepted glyph components so antialias/stroke fringes are
        removed while surrounding artwork remains byte-stable.

    No OCR text content is used here; this remains a visual-only route.
    """
    if target.shape[:2] != envelope.shape:
        raise ValueError("effect text refinement inputs must share shape")
    corridor = _effect_text_bbox_mask(region, envelope.shape, pad_x=8, pad_y=12)
    corridor = cv2.bitwise_and(corridor, (envelope > 0).astype(np.uint8) * 255)
    ys, xs = np.where(corridor > 0)
    if xs.size == 0:
        return np.zeros_like(envelope), {"reason": "empty_text_corridor"}

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    roi = target[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.astype(np.uint8)

    # Local-background contrast is robust on coloured gradients: it responds to
    # printed strokes but not to the slowly varying purple/pink/yellow field.
    bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=5.0, sigmaY=5.0, borderType=cv2.BORDER_REFLECT)
    signed = bg.astype(np.int16) - gray.astype(np.int16)
    dark_candidate = ((signed >= 18) & (gray <= 205)).astype(np.uint8)
    light_candidate = ((signed <= -20) & (gray >= 50)).astype(np.uint8)

    def select_components(candidate: np.ndarray) -> tuple[np.ndarray, int, int]:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
        local = np.zeros_like(candidate, np.uint8)
        roi_area = max(1, int(candidate.shape[0] * candidate.shape[1]))
        kept = 0
        rejected = 0
        for lab in range(1, n):
            bx, by, bw, bh, area = [int(v) for v in stats[lab]]
            if area < 2 or area > min(700, max(120, int(round(roi_area * 0.10)))):
                rejected += 1
                continue
            if (bw > 52 and bh < 4) or (bh > 90 and bw < 3):
                rejected += 1
                continue
            if max(bw, bh) > 58 and min(bw, bh) < 5:
                rejected += 1
                continue
            local[labels == lab] = 255
            kept += 1
        return local, kept, rejected

    dark_local, dark_kept, dark_rejected = select_components(dark_candidate)
    light_local, light_kept, light_rejected = select_components(light_candidate)

    # Choose the polarity that best agrees with the detector's prior text mask.
    # The prior mask may be oversized, so use it only as a soft overlap score.
    old_seed_full = cv2.bitwise_and((region.clear_mask > 0).astype(np.uint8) * 255, corridor)
    old_seed = old_seed_full[y0:y1, x0:x1]
    old_support = cv2.dilate(old_seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1) > 0
    dark_overlap = int(np.count_nonzero((dark_local > 0) & old_support))
    light_overlap = int(np.count_nonzero((light_local > 0) & old_support))
    dark_score = dark_overlap * 4.0 + cv2.countNonZero(dark_local) * 0.25
    light_score = light_overlap * 4.0 + cv2.countNonZero(light_local) * 0.25
    if light_score > dark_score:
        local, kept, rejected, polarity = light_local, light_kept, light_rejected, "light"
    else:
        local, kept, rejected, polarity = dark_local, dark_kept, dark_rejected, "dark"

    # The detector's original verified JP mask may contain antialias fragments that
    # the contrast detector misses.  Admit only the part inside this tight text
    # corridor; never bring back the oversized effect contour.
    old = cv2.bitwise_and((region.clear_mask > 0).astype(np.uint8) * 255, corridor)
    local_full = np.zeros_like(envelope)
    local_full[y0:y1, x0:x1] = local
    # The old mask is evidence, not authority.  Only keep connected components of
    # the old mask that substantially touch newly detected glyph support; a single
    # tangential contact may not resurrect an entire rectangular/burst fragment.
    near = cv2.dilate(local_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    old_extra = np.zeros_like(old)
    old_n, old_labels, old_stats, _ = cv2.connectedComponentsWithStats((old > 0).astype(np.uint8), 8)
    old_kept = 0
    old_rejected = 0
    for lab in range(1, old_n):
        bx, by, bw, bh, area = [int(v) for v in old_stats[lab]]
        comp = (old_labels == lab)
        touch = int(np.count_nonzero(comp & (near > 0)))
        if touch <= 0:
            old_rejected += 1
            continue
        touch_ratio = float(touch / max(1, area))
        # Reject obvious wide/short blocks or huge attached fragments unless they
        # are strongly supported by freshly detected glyph pixels.
        too_wide_short = bool(bw >= max(28, int(round((x1 - x0) * 0.26))) and bh <= max(18, int(round((y1 - y0) * 0.11))))
        weak_touch = bool(touch_ratio < 0.18 and touch < max(18, int(round(area * 0.10))))
        oversized = bool(area > max(950, int(round((x1 - x0) * (y1 - y0) * 0.09))))
        if (too_wide_short or oversized) and weak_touch:
            old_rejected += 1
            continue
        old_extra[comp] = 255
        old_kept += 1
    refined = cv2.bitwise_or(local_full, old_extra)
    if cv2.countNonZero(refined) > 0:
        refined = cv2.dilate(refined, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
        refined[corridor == 0] = 0
        # Final cleanup: drop components that are too detached from the original
        # text lane support inside this region.
        seed_support = cv2.dilate(old_seed_full, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        fin = np.zeros_like(refined)
        n2, labels2, stats2, _ = cv2.connectedComponentsWithStats((refined > 0).astype(np.uint8), 8)
        final_kept = 0
        final_rejected = 0
        for lab in range(1, n2):
            bx, by, bw, bh, area = [int(v) for v in stats2[lab]]
            comp = (labels2 == lab)
            support = int(np.count_nonzero(comp & (seed_support > 0)))
            local_touch = int(np.count_nonzero(comp & (local_full > 0)))
            too_wide_short = bool(bw >= max(32, int(round((x1 - x0) * 0.28))) and bh <= max(18, int(round((y1 - y0) * 0.12))))
            if support <= 0 and local_touch <= 0:
                final_rejected += 1
                continue
            if too_wide_short and local_touch < max(20, int(round(area * 0.10))):
                final_rejected += 1
                continue
            fin[comp] = 255
            final_kept += 1
        refined = fin
    return refined, {
        "reason": "ok" if cv2.countNonZero(refined) else "no_effect_text_components",
        "kept_components": int(kept),
        "rejected_components": int(rejected),
        "old_components_kept": int(old_kept),
        "old_components_rejected": int(old_rejected),
        "final_components_kept": int(final_kept if 'final_kept' in locals() else 0),
        "final_components_rejected": int(final_rejected if 'final_rejected' in locals() else 0),
        "polarity": polarity,
        "dark_overlap": int(dark_overlap),
        "light_overlap": int(light_overlap),
        "corridor_pixels": int(cv2.countNonZero(corridor)),
        "refined_pixels": int(cv2.countNonZero(refined)),
        "bbox": [x0, y0, x1, y1],
    }


def _inpaint_effect_text_by_block(
    target: np.ndarray,
    erase_mask: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Locally inpaint tight text components while keeping TARGET elsewhere exact.

    BallonsTranslator performs inpainting per text block/window.  We use the same
    locality principle with OpenCV here so this route remains lightweight: each
    connected text group gets a context crop, inpainting changes only masked
    pixels, and every non-mask TARGET pixel is copied back byte-for-byte.
    """
    mask = ((erase_mask > 0).astype(np.uint8) * 255)
    if cv2.countNonZero(mask) == 0:
        return target.copy(), {"blocks": 0, "inpaint_pixels": 0}
    out = target.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    blocks = 0
    inpaint_pixels = 0
    h, w = mask.shape
    for lab in range(1, n):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area <= 0:
            continue
        # Merge enough context around each component to reconstruct gradients and
        # screentone, but never write outside the actual mask.
        margin = max(12, min(42, int(round(max(bw, bh) * 0.55))))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(w, x + bw + margin), min(h, y + bh + margin)
        crop = out[y0:y1, x0:x1].copy()
        cmask = np.zeros((y1 - y0, x1 - x0), np.uint8)
        comp = labels[y0:y1, x0:x1] == lab
        cmask[comp] = 255
        # Include neighbouring accepted glyph fragments in the same crop.  This
        # avoids seams between kanji strokes while preserving mask locality.
        cmask = cv2.bitwise_or(cmask, mask[y0:y1, x0:x1])
        if cv2.countNonZero(cmask) == 0:
            continue
        repaired = cv2.inpaint(crop, cmask, 3.0, cv2.INPAINT_TELEA)
        use = cmask > 0
        crop[use] = repaired[use]
        out[y0:y1, x0:x1] = crop
        blocks += 1
        inpaint_pixels += int(cv2.countNonZero(cmask))
    # Hard guarantee: non-mask pixels are bit-identical to TARGET.
    out[mask == 0] = target[mask == 0]
    return out, {"blocks": int(blocks), "inpaint_pixels": int(inpaint_pixels)}


def _compose_effect_text_on_clean_target(
    clean_target: np.ndarray,
    aligned_source: np.ndarray,
    envelope: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Extract SOURCE CN glyphs against the cleaned TARGET and draw glyph-only ink."""
    source_mask, _target_mask, diag = changed_text_masks(
        aligned_source, clean_target, envelope,
        tolerance_px=2, min_unique_ratio=0.025, max_component_fraction=0.08,
    )
    out = clean_target.copy()
    alpha, source_gray = source_text_render(aligned_source, source_mask)
    if np.any(alpha > 0):
        a3 = alpha[..., None]
        ink = np.repeat(source_gray[..., None], 3, axis=2).astype(np.float32)
        out = np.clip(out.astype(np.float32) * (1.0 - a3) + ink * a3, 0, 255).astype(np.uint8)
    return out, source_mask, {
        "source_text_pixels": int(cv2.countNonZero(source_mask)),
        "source_alpha_pixels": int(np.count_nonzero(alpha > 0)),
        "diff": diag,
    }


def _apply_text_only_region(
    final: np.ndarray,
    target: np.ndarray,
    aligned: np.ndarray,
    alpha: np.ndarray,
    region: TransparentBubbleRegion,
    cfg: TransparentBubbleRevealConfig,
) -> tuple[dict[str, Any], np.ndarray, int, int]:
    """Pre-clear only proven TARGET text, then discover/paint SOURCE text.

    Artwork preservation is strict: the pre-clear modifies only the verified JP
    mask (plus a tiny antialias halo), and automatic TARGET difference cleanup is
    gated to the same mask. A wider envelope is used only to *find SOURCE text*.
    """
    envelope = _text_only_region_envelope(region, target.shape[:2], cfg)
    if cv2.countNonZero(envelope) == 0:
        return {
            "region_id": region.id,
            "envelope_pixels": 0,
            "effective_write_pixels": 0,
            "source_text_pixels": 0,
            "cleared_target_pixels": 0,
            "forced_preclear_pixels": 0,
            "residual_specks_removed": 0,
            "reason": "empty_envelope",
        }, np.zeros(target.shape[:2], np.uint8), 0, 0

    poly_mask = rasterize_polygon(region.polygon, target.shape[:2])
    forced = cv2.bitwise_and(
        ((region.clear_mask > 0).astype(np.uint8) * 255),
        ((envelope > 0).astype(np.uint8) * 255),
    )
    if cv2.countNonZero(forced) == 0:
        return {
            "region_id": region.id,
            "envelope_pixels": int(cv2.countNonZero(envelope)),
            "effective_write_pixels": 0,
            "source_text_pixels": 0,
            "cleared_target_pixels": 0,
            "forced_preclear_pixels": 0,
            "residual_specks_removed": 0,
            "reason": "empty_verified_text_mask",
        }, np.zeros(target.shape[:2], np.uint8), 0, 0

    base_envelope = envelope.copy()
    base_forced = forced.copy()
    target_is_white = _bubble_is_white(target, poly_mask)
    source_anchor = cv2.bitwise_or(forced, (region.clear_mask > 0).astype(np.uint8) * 255)
    paper_env = white_container_paper_mask(target, (poly_mask > 0).astype(np.uint8) * 255, source_anchor)
    use_paper_envelope = False
    paper_env_neutral = False
    if cv2.countNonZero(paper_env) > 0:
        paper_use = paper_env > 0
        gray_p = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)[paper_use]
        sat_p = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[..., 1][paper_use]
        if gray_p.size:
            paper_env_neutral = bool(
                float(np.percentile(gray_p, 50.0)) >= 242.0
                and float(np.percentile(gray_p, 25.0)) >= 236.0
                and float(np.percentile(sat_p, 80.0)) <= 36.0
            )
        base_pixels = int(cv2.countNonZero(base_envelope))
        paper_pixels = int(cv2.countNonZero(paper_env))
        # Seed-only dilation can be far too tight for narration notes / pasted text
        # boxes where CN uses a taller or wider layout than JP.  Only a genuinely
        # neutral/white TARGET paper field may broaden SOURCE discovery; pale
        # coloured effect frames stay on the effect-safe route.
        if paper_env_neutral and paper_pixels >= max(900, int(round(base_pixels * 0.34))):
            envelope = paper_env
            forced = cv2.bitwise_and((region.clear_mask > 0).astype(np.uint8) * 255, envelope)
            use_paper_envelope = True
    # For neutral white speech/narration containers, clear the complete TARGET
    # lettering rather than only edition-difference seeds.  This is critical when
    # JP and CN layouts do not overlap (e.g. a lone Japanese 知 or trailing よ。 near
    # the bubble edge): difference-only cleanup has no SOURCE neighbour to anchor
    # those glyphs, so they would otherwise survive forever.
    complete_target_text = np.zeros(target.shape[:2], np.uint8)
    boundary_geometry_guard = np.zeros(target.shape[:2], np.uint8)
    boundary_geometry_diag: dict[str, int] = {
        "boundary_geometry_components": 0,
        "boundary_geometry_pixels": 0,
    }
    if use_paper_envelope or target_is_white:
        text_surface = paper_env if cv2.countNonZero(paper_env) > 0 else envelope
        complete_target_text = target_white_container_text_mask(target, text_surface)
        complete_target_text, boundary_geometry_guard, boundary_geometry_diag = _split_boundary_geometry_from_text_mask(
            complete_target_text, text_surface
        )
        if cv2.countNonZero(complete_target_text) > 0:
            forced = cv2.bitwise_or(forced, complete_target_text)
            forced[envelope == 0] = 0

    structure_guard = np.zeros(target.shape[:2], np.uint8)
    if use_paper_envelope or target_is_white:
        structure_guard = _target_linear_structure_guard(target, envelope, guard_px=2)
        if cv2.countNonZero(structure_guard) > 0 and cv2.countNonZero(complete_target_text) > 0:
            # Verified JP lettering beats *generic* Hough geometry.  True boundary
            # geometry/tails were split above and are re-added afterwards at the
            # highest priority, so a speech-bubble tail can never be erased just
            # because the text detector saw it as a glyph-like stroke.
            text_exclusion = cv2.dilate(
                complete_target_text,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                iterations=1,
            )
            structure_guard[text_exclusion > 0] = 0
        if cv2.countNonZero(boundary_geometry_guard) > 0:
            structure_guard = cv2.bitwise_or(structure_guard, boundary_geometry_guard)
        if cv2.countNonZero(structure_guard) > 0:
            # Structural geometry is immutable in whole-page alignment mode.
            # Do not preclear it even if a contour/text seed accidentally touched it.
            forced[structure_guard > 0] = 0

    smooth_effect = bool(
        str(region.backend or "").strip().lower() == "target_text_contour"
        and not target_is_white
        and not use_paper_envelope
        and _effect_container_candidate_safe(target, poly_mask)
    )
    paper_like = _preclear_background_is_paper_like(target, forced, envelope)
    paper_diag: dict[str, Any] = {}
    effect_refine_diag: dict[str, Any] = {}
    effect_inpaint_diag: dict[str, Any] = {}
    effect_compose_diag: dict[str, Any] = {}
    if smooth_effect:
        # MTU/BallonsTranslator-inspired route: refine a glyph-only JP mask inside
        # the detected text box, inpaint that mask locally, then paint SOURCE CN
        # glyphs on the cleaned TARGET.  This fixes the regression where the
        # preview kept Japanese text to avoid changing the purple background.
        effective_forced, effect_refine_diag = _refine_effect_target_text_mask(
            target, region, envelope,
        )
        if cv2.countNonZero(structure_guard) > 0:
            effective_forced[structure_guard > 0] = 0
        target_clear_gate = effective_forced.copy()
        preclean, effect_inpaint_diag = _inpaint_effect_text_by_block(
            target, effective_forced,
        )
        if cv2.countNonZero(structure_guard) > 0:
            preclean[structure_guard > 0] = target[structure_guard > 0]
        local_out, source_mask, effect_compose_diag = _compose_effect_text_on_clean_target(
            preclean, aligned, envelope,
        )
        if cv2.countNonZero(structure_guard) > 0:
            source_mask = source_mask.copy()
            source_mask[structure_guard > 0] = 0
            local_out[structure_guard > 0] = target[structure_guard > 0]
        local_diag = {
            "changed_target_text_pixels": int(cv2.countNonZero(effective_forced)),
            "residual_specks_removed": 0,
            "target_clear_region_restricted": True,
        }
        preclear_strategy = "mtu_bt_component_mask_local_inpaint"
    elif paper_like:
        preclean, effective_forced, paper_diag = _local_paper_reconstruct_under_text(
            target, forced, envelope, dilate_px=2, ring_px=8,
        )
        target_clear_gate = _target_clear_gate_for_region(region, envelope, effective_forced)
        preclear_strategy = "target_local_paper"
        if cv2.countNonZero(structure_guard) > 0:
            target_clear_gate[structure_guard > 0] = 0
            effective_forced[structure_guard > 0] = 0
            preclean[structure_guard > 0] = target[structure_guard > 0]
        local_out, _transfer_write, source_mask, local_diag = transfer_text_only(
            preclean, aligned, envelope, tolerance_px=2, clear_dilate_px=0,
            inpaint_radius=1.5, white_container=False, localized_white_text=True,
            target_clear_region_mask=target_clear_gate, forced_target_clear_mask=effective_forced,
        )
    else:
        preclean, effective_forced = _smooth_reconstruct_under_text(
            target, forced, envelope, dilate_px=2, sigma=4.5,
        )
        target_clear_gate = _target_clear_gate_for_region(region, envelope, effective_forced)
        preclear_strategy = "target_only_normalized_gaussian"
        if cv2.countNonZero(structure_guard) > 0:
            target_clear_gate[structure_guard > 0] = 0
            effective_forced[structure_guard > 0] = 0
            preclean[structure_guard > 0] = target[structure_guard > 0]
        local_out, _transfer_write, source_mask, local_diag = transfer_text_only(
            preclean, aligned, envelope, tolerance_px=2, clear_dilate_px=0,
            inpaint_radius=1.5, white_container=False, localized_white_text=False,
            target_clear_region_mask=target_clear_gate, forced_target_clear_mask=effective_forced,
        )

    if not smooth_effect and cv2.countNonZero(structure_guard) > 0:
        source_mask = source_mask.copy()
        source_mask[structure_guard > 0] = 0
        local_out[structure_guard > 0] = target[structure_guard > 0]

    effective = np.any(local_out != target, axis=2) & (envelope > 0) & (structure_guard == 0)
    write_mask = effective.astype(np.uint8) * 255
    if np.any(effective):
        final[effective] = local_out[effective]
        alpha[effective] = 0

    cleared = int(cv2.countNonZero(effective_forced))
    cn_ink = int(cv2.countNonZero(source_mask))
    diag = {
        "region_id": region.id,
        "envelope_pixels": int(cv2.countNonZero(envelope)),
        "effective_write_pixels": int(cv2.countNonZero(write_mask)),
        "source_text_pixels": cn_ink,
        "changed_target_text_pixels": int(local_diag.get("changed_target_text_pixels", 0)),
        "preview_composite_mode": "clean_target_plus_source_glyphs" if smooth_effect else "local_out_difference",
        "effect_mask_refinement": effect_refine_diag,
        "effect_local_inpaint": effect_inpaint_diag,
        "effect_source_compose": effect_compose_diag,
        "cleared_target_pixels": cleared,
        "forced_preclear_pixels": cleared,
        "residual_specks_removed": int(local_diag.get("residual_specks_removed", 0)),
        "preclear_before_source_paint": True,
        "preclear_strategy": preclear_strategy,
        "target_clear_region_restricted": bool(local_diag.get("target_clear_region_restricted", False)),
        "target_clear_gate_pixels": int(cv2.countNonZero(target_clear_gate)),
        "smooth_effect": smooth_effect,
        "paper_like": paper_like,
        "paper_envelope_used": bool(use_paper_envelope),
        "paper_envelope_neutral": bool(paper_env_neutral),
        "paper_envelope_pixels": int(cv2.countNonZero(paper_env)) if 'paper_env' in locals() else 0,
        "base_envelope_pixels": int(cv2.countNonZero(base_envelope)) if 'base_envelope' in locals() else int(cv2.countNonZero(envelope)),
        "complete_target_text_pixels": int(cv2.countNonZero(complete_target_text)),
        "boundary_geometry_components": int(boundary_geometry_diag.get("boundary_geometry_components", 0)),
        "boundary_geometry_pixels": int(boundary_geometry_diag.get("boundary_geometry_pixels", 0)),
        "structure_guard_pixels": int(cv2.countNonZero(structure_guard)),
        "structure_guard_text_overlap_pixels": int(np.count_nonzero((structure_guard > 0) & (complete_target_text > 0))),
        "structure_guard_byte_exact": bool(
            cv2.countNonZero(structure_guard) == 0
            or np.array_equal(local_out[structure_guard > 0], target[structure_guard > 0])
        ),
        "paper_reconstruct": paper_diag,
    }
    return diag, write_mask, cleared, cn_ink


def execute_transparent_bubble(
    plan: TransparentBubblePlan,
    source: np.ndarray,
    target: np.ndarray,
    cfg: TransparentBubbleRevealConfig,
) -> TransparentBubbleResult:
    """Reveal aligned CN inside selected TARGET bubble regions.

    Contract:
    - Outside selected reveal holes, TARGET (Japanese) pixels remain byte-stable.
    - ``full_bubble`` truly clears/reveals the full bubble interior again.
    - ``text_only`` keeps the safer text-mask behaviour for coloured/open art.
    """
    del source
    if not plan.accepted:
        jp_rgba = cv2.cvtColor(target, cv2.COLOR_BGR2RGBA)
        return TransparentBubbleResult(
            image_rgb=target.copy(), image_rgba=jp_rgba.copy(), jp_layer_rgba=jp_rgba,
            cn_layer_rgb=plan.aligned_source.copy(), clear_mask=plan.clear_mask.copy(), plan=plan,
            diagnostics={**dict(plan.diagnostics), "applied_region_count": 0, "flattened": False},
        )

    aligned = plan.aligned_source
    if aligned.shape[:2] != target.shape[:2]:
        aligned = cv2.resize(aligned, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

    final = target.copy()
    gray_t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    gray_c = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    paper_thr = int(getattr(cfg, "paper_white_threshold", 215))
    alpha = np.full(target.shape[:2], 255, np.uint8)

    total_reveal_pixels = 0
    total_jp_clear_pixels = 0
    total_cn_ink_pixels = 0
    total_full_bubble_text_completion_pixels = 0
    full_region_count = 0
    text_region_count = 0
    text_region_diags: list[dict[str, Any]] = []
    page_clear = np.zeros(target.shape[:2], np.uint8)
    requested_clear = np.zeros(target.shape[:2], np.uint8)

    for region in plan.applied_regions:
        region_mask = region.clear_mask > 0
        if not np.any(region_mask):
            continue
        requested_clear[region_mask] = 255
        mode = str(region.clear_mode or "full_bubble").strip().lower()
        candidate_reveal = _resolve_full_bubble_reveal_mask(target, gray_c, region_mask, cfg) if mode == "full_bubble" else None
        if mode == "full_bubble" and candidate_reveal is not None and np.any(candidate_reveal):
            completion = _full_bubble_target_text_completion_mask(target, region, region_mask, cfg)
            if cv2.countNonZero(completion) > 0:
                added_completion = (completion > 0) & (~candidate_reveal)
                total_full_bubble_text_completion_pixels += int(np.count_nonzero(added_completion))
                candidate_reveal = candidate_reveal | (completion > 0)
        region_pixels = int(np.count_nonzero(region_mask))
        candidate_pixels = int(np.count_nonzero(candidate_reveal)) if candidate_reveal is not None else 0
        candidate_coverage = float(candidate_pixels / max(1, region_pixels))
        allow_full = False
        if mode == "full_bubble" and candidate_reveal is not None and np.any(candidate_reveal):
            explicit_full = str(getattr(cfg, "clear_mode", "") or "").strip().lower() == "full_bubble"
            base_mask_u8 = region_mask.astype(np.uint8) * 255
            reveal_mask_u8 = candidate_reveal.astype(np.uint8) * 255
            if explicit_full:
                # Preserve the explicit legacy contract: a user-forced full-bubble
                # mode may reveal a verified white paper interior even when printed
                # text makes the local texture metric non-flat. Non-white/effect
                # frames still fail this gate and are demoted to text-only.
                base_white = _bubble_is_white(target, base_mask_u8)
                recovered_white = _bubble_is_white(target, reveal_mask_u8)
            else:
                base_white = bool(
                    _flat_white_full_reveal_safe(target, base_mask_u8)
                    or _neutral_white_full_reveal_safe(target, base_mask_u8)
                )
                recovered_white = bool(
                    _flat_white_full_reveal_safe(target, reveal_mask_u8)
                    or _neutral_white_full_reveal_safe(target, reveal_mask_u8)
                )
            # Full reveal is no longer a force-through switch.  It means “reveal
            # the complete verified neutral-paper interior of a real text
            # container”.  A raw detector mask can never be used as a fallback.
            # This keeps skin/art byte-exact while still exposing all CN text in
            # the recovered white interior.
            anchor_ok = _full_reveal_has_text_anchor(region)
            min_coverage = 0.10 if explicit_full else 0.18
            effect_like = bool((not base_white) and _effect_container_candidate_safe(target, base_mask_u8))
            recovered_rect_like = bool(_mask_bbox_fill_ratio(reveal_mask_u8) >= 0.86)
            recovered_from_rect_seed = str(region.backend or "").strip().lower() == "text_seed_white_rect"
            # If the raw region is not itself white, only a strongly rectangular
            # recovered paper island is allowed to full-reveal. This preserves
            # oversize narration-box recovery while preventing a white oval in a
            # coloured/starburst effect frame from becoming a full SOURCE window.
            raw_or_recovered_container_ok = bool(base_white or recovered_rect_like or recovered_from_rect_seed)
            allow_full = bool(
                anchor_ok and recovered_white and candidate_coverage >= min_coverage
                and raw_or_recovered_container_ok and not effect_like
            )
        if allow_full:
            full_region_count += 1
            reveal_mask = candidate_reveal
            if not np.any(reveal_mask):
                continue
            page_clear[reveal_mask] = 255
            final[reveal_mask] = aligned[reveal_mask]
            alpha[reveal_mask] = 0
            total_reveal_pixels += int(np.count_nonzero(reveal_mask))
            continue

        # Any non-white / effect / burst container is forced through the safe
        # text-only renderer, even if a previous stage or manual setting marked
        # the region as full_bubble. This protects coloured starbursts and other
        # effect frames from being replaced wholesale by the CN page.
        text_region_count += 1
        local_region = region
        fallback_reason = None
        if mode == "full_bubble" and not allow_full:
            fallback_reason = "full_bubble_demoted_to_text_only_nonwhite_or_effect"
            local_region = TransparentBubbleRegion(
                id=region.id, target_bbox=region.target_bbox, polygon=region.polygon, clear_mask=region.clear_mask,
                confidence=region.confidence, backend=region.backend, triage=region.triage, reason=region.reason,
                applied=region.applied, clear_mode="text_only",
            )
        diag, write_mask, cleared_pixels, cn_ink_pixels = _apply_text_only_region(
            final, target, aligned, alpha, local_region, cfg
        )
        if fallback_reason is not None:
            diag["fallback_reason"] = fallback_reason
        if cv2.countNonZero(write_mask) > 0:
            page_clear[write_mask > 0] = 255
        total_jp_clear_pixels += int(cleared_pixels)
        total_cn_ink_pixels += int(cn_ink_pixels)
        text_region_diags.append(diag)

    jp_rgba = cv2.cvtColor(target, cv2.COLOR_BGR2RGBA)
    jp_rgba[..., 3] = alpha

    diagnostics = {
        **dict(plan.diagnostics),
        "applied_region_count": len(plan.applied_regions),
        "clear_pixels": int(np.count_nonzero(page_clear)),
        "requested_clear_pixels": int(np.count_nonzero(requested_clear)),
        "effective_clear_pixels": int(np.count_nonzero(page_clear)),
        "full_reveal_pixels": int(total_reveal_pixels),
        "jp_clear_pixels": int(total_jp_clear_pixels),
        "cn_ink_pixels": int(total_cn_ink_pixels),
        "full_bubble_text_completion_pixels": int(total_full_bubble_text_completion_pixels),
        "full_bubble_region_count": int(full_region_count),
        "text_only_region_count": int(text_region_count),
        "text_only_regions": text_region_diags,
        "flattened": False,
        "pixel_contract": "target_authority_plus_true_full_bubble_reveal",
        "color_preserve_outside_holes": True,
        "engine": "v2.0.24_mtu_bt_effect_inpaint",
    }
    return TransparentBubbleResult(
        image_rgb=final,
        image_rgba=jp_rgba.copy(),
        jp_layer_rgba=jp_rgba,
        cn_layer_rgb=aligned.copy(),
        clear_mask=page_clear,
        plan=plan,
        diagnostics=diagnostics,
    )


__all__ = [
    "TransparentBubbleRegion", "TransparentBubblePlan", "TransparentBubbleResult",
    "alpha_over", "build_transparent_bubble_plan", "reject_transparent_bubble_plan",
    "execute_transparent_bubble",
]
