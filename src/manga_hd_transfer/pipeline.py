from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .bubbles import build_text_units, detect_mangalens_bubbles, detect_seeded_white_bubbles, load_bubble_sidecar, pair_unseeded_white_containers
from .aligned_overlay_reveal import build_aligned_overlay_plan, execute_aligned_overlay, AlignedOverlayResult
from .config import PipelineConfig
from .cache import (
    PageStageCache, blocks_signature, image_stage_signature, load_completed_page,
    page_job_fingerprint, registration_stage_signature,
)
from .debug import mask_overlay, matching_overlay, registration_overlay, structure_overlay
from .direct_containers import build_source_direct_container_plan
from .dual_source import build_direct_source_evidence, select_direct_source_candidate
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import InpaintResult, inpaint_image
from .geometry import transform_points, transform_to_homography
from .io_utils import read_image, save_json, stem_id, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask, find_default_font
from .masking import MaskBuildResult, build_clear_mask
from .matching import match_units
from .mask_transfer import (
    finalize_transfer_records,
    transfer_bubble_patches, transfer_ocr_guided_text_units,
    transfer_paired_diff_regions, transfer_photo_color_sfx,
    transfer_rigid_container_rasters,
)
from .models import BookProject, BubbleInstance, PagePair, PageProject, QAItem, RegistrationResult, TextBlock, TextUnit, UnitMatch
from .ocr import OCRBackend, NullOCRBackend, RetryingOCRBackend, build_backend
from .pairing import pair_directories
from .page_management import PageMark, resolve_mark
from .result_state import invalidate_manual_review_state, commit_automatic_result
from .page_pairing import PagePairingCheck, verify_registered_page_pair
from .paired_diff import extract_paired_diff_bubbles
from .qa import qa_summary, run_direct_patch_qa, run_mask_replace_qa, run_page_qa
from .registration import register_images
from .runtime import configure_runtime, empty_accelerator_cache, runtime_summary
from .source_detectors import run_source_detector_chain
from .plugins import REGISTRY as PROVIDER_REGISTRY
from .transfer_planner import choose_transfer_strategy

logger = logging.getLogger(__name__)


class PipelineCancelled(RuntimeError):
    """Cooperative cancellation raised only at safe stage boundaries."""


def _check_cancel(cancel_cb, stage: str = "") -> None:
    if cancel_cb is not None and cancel_cb():
        raise PipelineCancelled(stage or "cancelled")


def _merge_mask_transfer(base, extra):
    """Merge a sequential transfer result whose input image was ``base.image``."""
    if extra is None:
        return base
    if base is None:
        return extra
    base.image = extra.image
    use = extra.layer_rgba[..., 3] > 0
    base.layer_rgba[use, :3] = extra.layer_rgba[use, :3]
    base.layer_rgba[..., 3] = np.maximum(base.layer_rgba[..., 3], extra.layer_rgba[..., 3])
    base.composite_mask = np.maximum(base.composite_mask, extra.composite_mask)
    if extra.clear_mask is not None:
        if base.clear_mask is None:
            base.clear_mask = np.zeros_like(base.composite_mask)
        base.clear_mask = np.maximum(base.clear_mask, extra.clear_mask)
    base.matches.extend(extra.matches)
    base.records.extend(extra.records)
    return base


def _sidecar_with_suffix(path: str | Path, suffix: str) -> Path:
    p = Path(path)
    return p.with_suffix(suffix) if suffix.startswith(".") else p.parent / f"{p.stem}{suffix}"


def _load_additional_source_specs(source_path: str | Path, cfg) -> list[dict]:
    if not bool(getattr(cfg, "additional_source_enabled", True)):
        return []
    manifest = _sidecar_with_suffix(source_path, str(getattr(cfg, "additional_source_manifest_suffix", ".replace_sources.json")))
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("sources", payload) if isinstance(payload, dict) else payload
    out: list[dict] = []
    if not isinstance(rows, list):
        return out
    for row in rows[: max(0, int(getattr(cfg, "additional_source_max_candidates", 2)))]:
        if isinstance(row, str):
            item = {"path": row, "kind": "alternate"}
        elif isinstance(row, dict):
            item = {"path": row.get("path", ""), "kind": row.get("kind", "alternate")}
        else:
            continue
        rp = Path(item["path"])
        if not rp.is_absolute():
            rp = manifest.parent / rp
        if rp.exists():
            item["path"] = str(rp)
            out.append(item)
    return out


def _resolve_secondary_source_spec(primary_source_path: str | Path, target_path: str | Path, cfg) -> dict | None:
    if not bool(getattr(cfg, "enabled", False)):
        return None
    root_value = getattr(cfg, "secondary_source_dir", None)
    if not root_value:
        return None
    root = Path(root_value).expanduser()
    if root.is_file():
        return {"path": str(root), "kind": "secondary_dir", "origin": "dual_source"}
    if not root.is_dir():
        return None
    primary = Path(primary_source_path)
    target = Path(target_path)
    candidates = [root / primary.name, root / target.name]
    extensions = [primary.suffix, target.suffix, ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"]
    for stem in (primary.stem, target.stem):
        for ext in extensions:
            if ext:
                candidates.append(root / f"{stem}{ext}")
    if bool(getattr(cfg, "recursive_lookup", False)):
        for stem in (primary.stem, target.stem):
            candidates.extend(root.rglob(f"{stem}.*"))
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return {"path": str(candidate), "kind": "secondary_dir", "origin": "dual_source"}
    return None


def _reason_metric(match: UnitMatch, key: str, default: float = 0.0) -> float:
    prefix = key + "="
    for reason in list(getattr(match, "reasons", []) or []):
        if str(reason).startswith(prefix):
            try:
                return float(str(reason).split("=", 1)[1])
            except Exception:
                return default
    return default


def _replace_translation_regions(source_units, target_units, matches, overlap_threshold: float = 0.30) -> list[dict]:
    target_by_id = {u.id: u for u in target_units}
    match_by_source: dict[str, UnitMatch] = {}
    for match in matches:
        prior = match_by_source.get(match.source_unit_id)
        if prior is None or float(match.confidence) > float(prior.confidence):
            match_by_source[match.source_unit_id] = match
    rows: list[dict] = []
    for source in source_units:
        match = match_by_source.get(source.id)
        target = target_by_id.get(match.target_unit_id) if match is not None else None
        overlap = _reason_metric(match, "overlap", 0.0) if match is not None else 0.0
        rows.append({
            "source_unit_id": source.id,
            "target_unit_id": target.id if target is not None else None,
            "translated_text": str(getattr(source, "text", "") or ""),
            "source_bbox": [float(x) for x in source.bbox],
            "target_bbox": [float(x) for x in target.bbox] if target is not None else None,
            "overlap": float(overlap),
            "matched": bool(match is not None and target is not None and overlap >= float(overlap_threshold)),
            "relation": str(match.relation) if match is not None else "unmatched",
            "confidence": float(match.confidence) if match is not None else 0.0,
            "cost": float(match.cost) if match is not None else 1.0,
        })
    return rows


def _write_replace_translation_bundle(page_root: Path, cfg, source_blocks, target_blocks, matches, summary: dict) -> dict[str, str]:
    if not bool(getattr(cfg, "export_enabled", True)):
        return {}
    root = page_root / str(getattr(cfg, "export_dirname", "replace_translation"))
    root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    if bool(getattr(cfg, "export_ocr_json", True)):
        src_path = root / "source_ocr.json"
        tgt_path = root / "target_ocr.json"
        save_json(src_path, {"blocks": [x.to_dict() for x in source_blocks]})
        save_json(tgt_path, {"blocks": [x.to_dict() for x in target_blocks]})
        artifacts["source_ocr"] = str(src_path)
        artifacts["target_ocr"] = str(tgt_path)
    if bool(getattr(cfg, "export_matches_json", True)):
        match_path = root / "unit_matches.json"
        save_json(match_path, {"matches": [x.to_dict() for x in matches]})
        artifacts["matches"] = str(match_path)
    if bool(getattr(cfg, "export_summary_json", True)):
        summary_path = root / "summary.json"
        save_json(summary_path, summary)
        artifacts["summary"] = str(summary_path)
    return artifacts


def _remaining_paired_bubbles(source_bubbles, target_bubbles, handled_target_ids: set[str]):
    """Preserve explicit paired identity while excluding already-rendered regions."""
    target_by_id = {b.id: b for b in target_bubbles}
    out_s, out_t = [], []
    for sb in source_bubbles:
        tid = str(sb.meta.get("paired_target_id") or "")
        tb = target_by_id.get(tid) if tid else None
        if tb is not None and tb.id not in handled_target_ids:
            out_s.append(sb); out_t.append(tb)
    if not out_s and not handled_target_ids and len(source_bubbles) == len(target_bubbles):
        return list(source_bubbles), list(target_bubbles)
    return out_s, out_t


def _mask_transfer_completion_needed(mask_transfer) -> bool:
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    if not records:
        return True
    return any((not bool(getattr(r, "applied", False))) or bool(getattr(r, "review_required", False)) for r in records)


def _completion_existing_target_bubbles(mask_transfer, *candidate_groups):
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    handled_ids = {
        str(getattr(r, "target_bubble_id", ""))
        for r in records
        if bool(getattr(r, "applied", False))
        and not bool(getattr(r, "review_required", False))
        and bool(getattr(r, "content_complete", True))
        and str(getattr(r, "target_bubble_id", ""))
    }
    if not handled_ids:
        return []
    existing = []
    seen = set()
    for group in candidate_groups:
        for bubble in list(group or []):
            bid = str(getattr(bubble, "id", ""))
            if not bid or bid not in handled_ids or bid in seen:
                continue
            existing.append(bubble)
            seen.add(bid)
    return existing


def _completion_review_regions(mask_transfer):
    records = list(getattr(mask_transfer, "records", []) or []) if mask_transfer is not None else []
    boxes: list[tuple[int, int, int, int]] = []
    for rec in records:
        box = getattr(rec, "target_bbox", None)
        if not box or len(box) != 4:
            continue
        if bool(getattr(rec, "review_required", False)) or not bool(getattr(rec, "content_complete", True)) or not bool(getattr(rec, "applied", False)):
            boxes.append(tuple(int(v) for v in box))
    return boxes


def _bbox_tuple_from_bubble(bubble):
    mask = getattr(bubble, "mask", None)
    if mask is not None and getattr(mask, "size", 0):
        ys, xs = np.where(mask > 0)
        if len(xs) > 0:
            return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    poly = getattr(bubble, "polygon", None) or []
    if poly:
        xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
        return (min(xs), min(ys), max(xs), max(ys))
    return None


def _bbox_intersection_over_smaller(a, b) -> float:
    ax0, ay0, ax1, ay1 = map(int, a); bx0, by0, bx1, by1 = map(int, b)
    ix0=max(ax0,bx0); iy0=max(ay0,by0); ix1=min(ax1,bx1); iy1=min(ay1,by1)
    inter=max(0,ix1-ix0)*max(0,iy1-iy0)
    aa=max(1,(ax1-ax0)*(ay1-ay0)); ba=max(1,(bx1-bx0)*(by1-by0))
    return float(inter / max(1, min(aa, ba)))


def _filter_uncovered_white_completion_pairs(source_bubbles, target_bubbles, existing_boxes, config=None, *, overlap_threshold: float = 0.28):
    """Keep ordinary unseeded white containers that are not already represented.

    v1.0.6 removes the old publication-only 0.84/0.16 paper thresholds.  Those
    values silently discarded perfectly valid small speech balloons with dense
    CJK glyphs (the real page-45 round balloon measures about 0.81 white ratio).
    Detection/pairing already proved white-container geometry and registered ink
    change, so this stage now uses the detector's own permissive limits and only
    keeps the duplicate/coloured-region exclusions.
    """
    kept_s, kept_t = [], []
    existing = [tuple(map(int, b)) for b in list(existing_boxes or []) if b and len(b) == 4]
    min_white = float(getattr(config, "rigid_container_unseeded_min_white_ratio", 0.55)) if config is not None else 0.55
    max_dark = float(getattr(config, "rigid_container_unseeded_max_dark_ratio", 0.28)) if config is not None else 0.28
    # Dense translated CJK can legitimately lower paper ratio; never make this
    # completion gate stricter than the detector that produced the candidate.
    min_white = min(min_white, 0.55)
    max_dark = max(max_dark, 0.28)
    for sb, tb in zip(list(source_bubbles or []), list(target_bubbles or [])):
        meta = dict(getattr(tb, "meta", {}) or {})
        if str(meta.get("backend", "")) != "unseeded_white":
            continue
        if bool(meta.get("target_colored_recovery", False)) or bool(meta.get("target_driven_colored", False)):
            continue
        white = float(meta.get("white_ratio", 1.0) or 0.0)
        dark = float(meta.get("dark_ratio", 0.0) or 0.0)
        sat_median = float(meta.get("saturation_median", 0.0) or 0.0)
        sat_p75 = float(meta.get("saturation_p75", 0.0) or 0.0)
        if white < min_white or dark > max_dark:
            continue
        # This is route classification, not publication gating: a light purple/
        # pink burst may contain a large high-value component that the white
        # detector sees as paper.  Rigid white completion would paste a white
        # island over the coloured TARGET.  Keep only genuinely neutral paper;
        # coloured regions stay on the target-aware component/Reveal route.
        if sat_median > 12.0 or sat_p75 > 24.0:
            continue
        box = _bbox_tuple_from_bubble(tb)
        if box is None:
            continue
        if any(_bbox_intersection_over_smaller(box, old) >= float(overlap_threshold) for old in existing):
            continue
        kept_s.append(sb); kept_t.append(tb)
    return kept_s, kept_t


def _completion_filter_pairs_to_review_regions(source_bubbles, target_bubbles, review_boxes):
    if not source_bubbles or not target_bubbles:
        return source_bubbles, target_bubbles
    kept_src = []
    kept_dst = []
    max_aspect = 2.1
    for sb, tb in zip(source_bubbles, target_bubbles):
        tbox = None
        tmask = getattr(tb, "mask", None)
        if tmask is not None and getattr(tmask, 'size', 0):
            ys, xs = np.where(tmask > 0)
            if len(xs) > 0:
                tbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        if tbox is None:
            poly = getattr(tb, "polygon", None) or []
            if poly:
                xs = [int(p[0]) for p in poly]; ys = [int(p[1]) for p in poly]
                tbox = (min(xs), min(ys), max(xs), max(ys))
        if tbox is None:
            continue
        tx0, ty0, tx1, ty1 = tbox
        tw = max(1, tx1 - tx0); th = max(1, ty1 - ty0)
        aspect = max(tw / max(1.0, th), th / max(1.0, tw))
        keep = aspect <= max_aspect
        if review_boxes:
            for rx0, ry0, rx1, ry1 in review_boxes:
                ix0 = max(tx0, rx0); iy0 = max(ty0, ry0); ix1 = min(tx1, rx1); iy1 = min(ty1, ry1)
                inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
                if inter > 0:
                    keep = True
                    break
                cx = (rx0 + rx1) // 2; cy = (ry0 + ry1) // 2
                if tx0 <= cx <= tx1 and ty0 <= cy <= ty1:
                    keep = True
                    break
        if keep:
            kept_src.append(sb)
            kept_dst.append(tb)
    return kept_src, kept_dst


def _blocking_direct_invariant_issues(issues: list[str]) -> list[str]:
    """Return only structural Direct invariants that can still block execution.

    v1.0.6 removes publication/content-completeness blocking.  Border writes, OCR
    use and target-bubble rematching are still architectural contract violations;
    ``content_incomplete:*`` remains diagnostic and must not force Auto back to
    Mask after Direct already produced valid regions.
    """
    return [str(x) for x in list(issues or []) if not str(x).startswith("content_incomplete:")]


def _has_transferable_source_text(
    blocks: list[TextBlock], bubbles: list[BubbleInstance], enabled_kinds: list[str],
    candidate_bubbles: list[BubbleInstance] | None = None,
) -> bool:
    """Return True only for OCR-backed source speech/narration containers.

    Paired-difference geometry can create synthetic block ids without proving that
    the old Chinese page actually contains Chinese text.  We therefore require a
    *real* non-empty OCR block, either explicitly associated with a detected source
    bubble or geometrically located inside a paired-difference candidate bubble.
    The latter keeps photographed/edge-clipped pages valid even when the ordinary
    seeded bubble detector cannot reconstruct the whole container.
    """
    enabled = {str(x) for x in enabled_kinds}
    real_blocks = [b for b in blocks if str(getattr(b, "text", "")).strip()]
    by_id = {str(b.id): b for b in real_blocks}
    for bubble in bubbles:
        if str(getattr(bubble, "kind", "")) not in enabled:
            continue
        if any(str(block_id) in by_id for block_id in list(getattr(bubble, "block_ids", []) or [])):
            return True
    for bubble in list(candidate_bubbles or []):
        if str(getattr(bubble, "kind", "")) not in enabled:
            continue
        x0, y0, x1, y1 = bubble.bbox
        for block in real_blocks:
            cx, cy = block.centroid
            if x0 <= cx <= x1 and y0 <= cy <= y1:
                return True
    return False


def _saturation_p90(image: np.ndarray) -> float:
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 1], 90.0))


def _cross_rendition_monochrome_source(source: np.ndarray, target: np.ndarray) -> bool:
    """Detect BW/grayscale translated scan -> coloured master pairs."""
    return _saturation_p90(source) < 24.0 and _saturation_p90(target) >= 24.0


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff"
        for ch in str(text or "")
    )


def _infer_region_orientation(crop: np.ndarray, text: str, kind: str = "speech") -> tuple[str, dict]:
    """Infer vertical/horizontal layout from the *source glyph image*.

    VisionKit Live Text/Shortcuts return transcript only, so their synthetic
    TextBlock polygon is the whole balloon and cannot describe text direction.
    Use dark connected components in the already-isolated source balloon instead.
    Strong image evidence wins; ambiguous CJK speech defaults to vertical, which
    matches the dominant manga dialogue convention while still allowing obvious
    horizontal captions/dialogue to remain horizontal.
    """
    if crop.size == 0:
        return "vertical" if _contains_cjk(text) and kind in {"speech", "narration", "unknown"} else "horizontal", {"reason": "empty_crop"}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    # Keep printed ink while ignoring pale paper/antialias noise.
    ink = (gray < 185).astype(np.uint8)
    h, w = ink.shape[:2]
    # Remove a tiny border where balloon/panel edges can survive the safe mask.
    border = max(1, int(round(min(h, w) * 0.025)))
    if h > border * 2 and w > border * 2:
        ink[:border, :] = 0; ink[-border:, :] = 0
        ink[:, :border] = 0; ink[:, -border:] = 0
    ys, xs = np.where(ink > 0)
    if len(xs) < 8:
        fallback = "vertical" if _contains_cjk(text) and kind in {"speech", "narration", "unknown"} else "horizontal"
        return fallback, {"reason": "too_little_ink", "ink_pixels": int(len(xs))}

    ink_w = max(1, int(xs.max() - xs.min() + 1)); ink_h = max(1, int(ys.max() - ys.min() + 1))
    ink_aspect = float(ink_h / ink_w)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    page_area = float(max(1, h * w))
    for idx in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[idx]]
        # Reject dust and large artwork/borders. Individual CJK glyph components
        # are small relative to an isolated balloon crop.
        if area < max(3, int(page_area * 0.00008)):
            continue
        if area > page_area * 0.09 or cw > w * 0.55 or ch > h * 0.55:
            continue
        comps.append((float(centroids[idx][0]), float(centroids[idx][1]), cw, ch, area))

    vertical_neighbors = 0; horizontal_neighbors = 0
    if len(comps) >= 2:
        for i, (cx, cy, cw, ch, _area) in enumerate(comps):
            best = None
            for j, (dx0, dy0, dw, dh, _a2) in enumerate(comps):
                if i == j:
                    continue
                # Normalize displacement by local glyph scale so punctuation and
                # mixed-size CJK characters do not dominate the vote.
                sx = max(2.0, (cw + dw) * 0.5); sy = max(2.0, (ch + dh) * 0.5)
                dx = abs(dx0 - cx) / sx; dy = abs(dy0 - cy) / sy
                dist = dx * dx + dy * dy
                if best is None or dist < best[0]:
                    best = (dist, dx, dy)
            if best is None:
                continue
            _, dx, dy = best
            if dy > dx * 1.20:
                vertical_neighbors += 1
            elif dx > dy * 1.30:
                horizontal_neighbors += 1

    # Strong overall shape is the most stable clue. The neighbor vote resolves
    # roughly square multi-column bubbles (common in manga).
    if ink_aspect >= 1.18:
        orientation, reason = "vertical", "ink_bbox_tall"
    elif ink_aspect <= 0.66:
        orientation, reason = "horizontal", "ink_bbox_wide"
    elif vertical_neighbors >= max(2, int(horizontal_neighbors * 1.25 + 0.5)):
        orientation, reason = "vertical", "component_flow_vertical"
    elif horizontal_neighbors >= max(2, int(vertical_neighbors * 1.40 + 0.5)):
        orientation, reason = "horizontal", "component_flow_horizontal"
    elif _contains_cjk(text) and kind in {"speech", "narration", "unknown"}:
        orientation, reason = "vertical", "ambiguous_cjk_manga_default"
    else:
        orientation, reason = ("vertical", "bubble_crop_tall") if h / max(1.0, w) >= 1.35 else ("horizontal", "bubble_crop_wide")
    return orientation, {
        "reason": reason,
        "ink_bbox_aspect": round(ink_aspect, 4),
        "component_count": len(comps),
        "vertical_neighbor_votes": vertical_neighbors,
        "horizontal_neighbor_votes": horizontal_neighbors,
    }




def _source_layout_profile(crop: np.ndarray, text: str, orientation: str) -> dict:
    """Estimate source typography from isolated translated ink.

    The key invariant is that OCR contributes Unicode only.  Column count and glyph
    pitch come from the source image.  We intentionally avoid connected-component
    counts because CJK glyphs split into many stroke components; instead solve the
    approximate grid from ink-bbox aspect ratio and character count.
    """
    if crop.size == 0 or not str(text or "").strip():
        return {}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()
    h, w = gray.shape[:2]
    raw = (gray < 188).astype(np.uint8) * 255
    # Remove dust while retaining punctuation and thin antialiased strokes.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    kept = np.zeros_like(raw)
    area_total = float(max(1, h*w))
    for i in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[i]]
        if area < max(2, int(area_total * 0.000015)):
            continue
        if area > area_total * 0.12 or cw > w * 0.72 or ch > h * 0.72:
            continue
        kept[labels == i] = 255
    ys, xs = np.where(kept > 0)
    if len(xs) < 10:
        return {}
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ink_w, ink_h = max(1, x1-x0), max(1, y1-y0)
    chars = [ch for ch in str(text) if not ch.isspace()]
    n = max(1, len(chars))
    common = {
        "ink_bbox": [x0, y0, x1, y1],
        "ink_bbox_size": [ink_w, ink_h],
        "container_size": [int(w), int(h)],
        "fill_ratio": round(float((ink_w * ink_h) / max(1.0, w * h)), 4),
    }

    if orientation == "vertical":
        # For a roughly square glyph grid: ink_w/ink_h ≈ columns/rows and
        # rows ≈ n/columns, hence columns ≈ sqrt(n * aspect).
        aspect = float(ink_w / max(1.0, ink_h))
        columns = int(np.clip(round(np.sqrt(max(0.05, aspect) * n)), 1, 8))
        rows = max(1, int(np.ceil(n / columns)))
        pitch_h = ink_h / rows
        pitch_w = ink_w / columns
        pitch = float(min(pitch_h, pitch_w) * 0.96)
        return {
            "orientation": "vertical", "columns": columns, "rows": rows,
            "glyph_pitch_px": round(max(4.0, pitch), 3),
            **common,
        }

    aspect = float(ink_h / max(1.0, ink_w))
    rows = int(np.clip(round(np.sqrt(max(0.05, aspect) * n)), 1, 8))
    cols = max(1, int(np.ceil(n / rows)))
    pitch_h = ink_h / rows
    pitch_w = ink_w / cols
    pitch = float(min(pitch_h, pitch_w) * 0.96)
    return {
        "orientation": "horizontal", "rows": rows, "columns": cols,
        "glyph_pitch_px": round(max(4.0, pitch), 3),
        **common,
    }


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Local non-zero bbox helper for pipeline masks.

    Do not depend on lettering.py's private _safe_bbox; v0.8.13 accidentally
    referenced that private helper without importing it, causing a NameError on
    the new layout-integrity path on macOS.
    """
    if mask is None or getattr(mask, "size", 0) == 0:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _masked_layout_profile(image: np.ndarray, mask: np.ndarray, text: str, orientation: str) -> dict:
    if image.size == 0 or mask.size == 0:
        return {}
    box = _mask_bbox(mask)
    if box is None:
        return {}
    x0, y0, x1, y1 = box
    crop = image[y0:y1, x0:x1].copy()
    local = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return {}
    if crop.ndim == 3:
        crop[local == 0] = 255
    else:
        crop[local == 0] = 255
    return _source_layout_profile(crop, text, orientation)


def _preserved_layout_looks_complete(source_profile: dict, target_profile: dict) -> bool:
    """Reject apparently clipped sharp transfers before skipping OCR re-lettering.

    v0.8.11 proved that a sharp-but-partial transferred patch can still pass the
    generic mask/coverage gates, causing missing leading columns like “小丽差不多”.
    Compare source-vs-target layout recovery in normalized bubble space: if a crisp
    preserved transfer collapses columns/rows or loses too much ink extent, force
    OCR re-letter fallback instead of publishing incomplete text.
    """
    if not source_profile or not target_profile:
        return False
    orientation = str(source_profile.get("orientation") or target_profile.get("orientation") or "")
    sw, sh = [max(1.0, float(v)) for v in (source_profile.get("container_size") or [1, 1])]
    tw, th = [max(1.0, float(v)) for v in (target_profile.get("container_size") or [1, 1])]
    siw, sih = [max(1.0, float(v)) for v in (source_profile.get("ink_bbox_size") or [1, 1])]
    tiw, tih = [max(1.0, float(v)) for v in (target_profile.get("ink_bbox_size") or [1, 1])]
    src_fill_x, src_fill_y = siw / sw, sih / sh
    tgt_fill_x, tgt_fill_y = tiw / tw, tih / th
    src_cols = max(1, int(source_profile.get("columns") or 1))
    tgt_cols = max(1, int(target_profile.get("columns") or 1))
    src_rows = max(1, int(source_profile.get("rows") or 1))
    tgt_rows = max(1, int(target_profile.get("rows") or 1))
    src_pitch = float(source_profile.get("glyph_pitch_px") or 0.0)
    tgt_pitch = float(target_profile.get("glyph_pitch_px") or 0.0)

    if orientation == "vertical":
        if src_cols >= 2 and tgt_cols < src_cols:
            return False
        if src_rows >= 4 and tgt_rows + 1 < src_rows:
            return False
        if src_cols >= 2 and tgt_fill_x < src_fill_x * 0.72:
            return False
        if tgt_pitch > 0 and src_pitch > 0 and tgt_pitch < src_pitch * 0.58:
            return False
        return True

    if src_rows >= 2 and tgt_rows < src_rows:
        return False
    if src_cols >= 4 and tgt_cols + 1 < src_cols:
        return False
    if src_rows >= 2 and tgt_fill_y < src_fill_y * 0.72:
        return False
    if tgt_pitch > 0 and src_pitch > 0 and tgt_pitch < src_pitch * 0.58:
        return False
    return True

def _reletter_orientation(base_orientation: str, unit: TextUnit, blocks_by_id: dict) -> str:
    """Infer manga text direction, honoring source-image hints first."""
    if base_orientation in {"horizontal", "vertical"}:
        return base_orientation
    ratios = []
    for bid in unit.block_ids:
        b = blocks_by_id.get(bid)
        if b is None:
            continue
        hinted = str(b.meta.get("orientation_hint") or "").lower()
        if hinted in {"horizontal", "vertical"}:
            return hinted
        x0, y0, x1, y1 = b.bbox
        bw = max(1.0, x1 - x0); bh = max(1.0, y1 - y0)
        ratios.append(bh / bw)
    if ratios:
        med = float(np.median(ratios))
        if med >= 1.30:
            return "vertical"
        if med <= 0.78:
            return "horizontal"
    x0, y0, x1, y1 = unit.bbox
    return "vertical" if (y1 - y0) / max(1.0, x1 - x0) >= 1.45 else "horizontal"




def _should_preserve_transferred_layout(record, mask_cfg) -> bool:
    """True when OCR should remain evidence-only for an already sharp transfer."""
    if record is None or not bool(getattr(mask_cfg, "photo_pair_preserve_sharp_source_layout", True)):
        return False
    if not bool(getattr(record, "applied", False)) or bool(getattr(record, "review_required", False)):
        return False
    min_rel = float(getattr(mask_cfg, "photo_pair_preserve_layout_min_relative_sharpness", 1.15))
    modes = set(getattr(mask_cfg, "photo_pair_preserve_layout_clarity_modes", []) or [])
    return (
        float(getattr(record, "relative_sharpness", 0.0)) >= min_rel
        and str(getattr(record, "clarity_mode", "")) in modes
    )

def _review_candidate_overlay(image: np.ndarray, queue: list[dict]) -> np.ndarray:
    """Create a review-only preview; never burns warnings into clean final output."""
    if not queue:
        return image.copy()
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font_path = find_default_font(None)
        font = ImageFont.truetype(font_path, max(13, round(image.shape[1] * 0.015)))
    except Exception:
        font = ImageFont.load_default()
    # Deliberately use a strong magenta review color; clean final.png remains clean.
    color = (255, 80, 150)
    for row in queue:
        box = row.get("target_bbox") or []
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = map(int, box)
        pad = 4
        draw.rectangle((max(0, x0-pad), max(0, y0-pad), min(image.shape[1]-1, x1+pad), min(image.shape[0]-1, y1+pad)), outline=color, width=3)
        reason = str(row.get("reason", ""))
        workflow = str(row.get("workflow", ""))
        if workflow == "manual_effect" or "manual_reveal" in reason or "open_text" in reason or "colored_text" in reason or "spiky_text" in reason:
            label = "开放式文字候选 · 建议框选补漏 / 擦除显字"
        elif reason == "photographed_text_without_ocr_reletter":
            label = "摄影中文字形 · 可能模糊/扭曲 · 可编辑/还原"
        else:
            label = "中文版候选 · 可能不完整/不准确 · 可编辑/还原"
        tb = draw.textbbox((0, 0), label, font=font)
        tw, th = tb[2]-tb[0], tb[3]-tb[1]
        lx = max(0, min(image.shape[1]-tw-6, x0-pad))
        ly = max(0, y0-pad-th-7)
        draw.rounded_rectangle((lx, ly, lx+tw+6, ly+th+5), radius=3, fill=(255,255,255), outline=color, width=2)
        draw.text((lx+3, ly+1), label, font=font, fill=color)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


class TransferPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        source_ocr: OCRBackend | None = None,
        target_ocr: OCRBackend | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self._source_ocr = source_ocr
        self._target_ocr = target_ocr
        self._ocr_soft_failures: list[str] = []
        configure_runtime(self.config.runtime)
        # Keep all optional PyTorch backends on the same explicit device policy.
        if self.config.registration.device == "auto":
            self.config.registration.device = self.config.runtime.device
        if self.config.bubbles.device == "auto":
            self.config.bubbles.device = self.config.runtime.device
        if self.config.mask_replace.sr_device == "auto":
            self.config.mask_replace.sr_device = self.config.runtime.device

    def _build_ocr_backend_soft(self, lang: str, backend_name: str) -> OCRBackend:
        try:
            return build_backend(self.config.ocr, lang, backend_name)
        except (RuntimeError, ImportError, ModuleNotFoundError) as exc:
            if not bool(getattr(self.config.ocr, "soft_fail_missing_backend", True)):
                raise
            message = f"OCR backend {backend_name!r} unavailable: {exc}"
            logger.warning("%s; continuing without OCR evidence", message)
            self._ocr_soft_failures.append(message)
            return NullOCRBackend()

    @property
    def source_ocr(self) -> OCRBackend:
        if self._source_ocr is None:
            backend_name = self.config.ocr.source_backend or self.config.ocr.backend
            backend = self._build_ocr_backend_soft(self.config.ocr.source_lang, backend_name)
            if self.config.ocr.retry_low_confidence and backend_name == "paddle" and not isinstance(backend, NullOCRBackend):
                backend = RetryingOCRBackend(backend, self.config.ocr.retry_confidence, self.config.ocr.retry_scale)
            self._source_ocr = backend
        return self._source_ocr

    @property
    def target_ocr(self) -> OCRBackend:
        if self._target_ocr is None:
            backend_name = self.config.ocr.target_backend or self.config.ocr.backend
            self._target_ocr = self._build_ocr_backend_soft(self.config.ocr.target_lang, backend_name)
        return self._target_ocr

    def _passthrough_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None,
        mark: PageMark,
        *,
        source: np.ndarray | None = None,
        target: np.ndarray | None = None,
        registration: RegistrationResult | None = None,
        passthrough_reason: str = "page_manager_exclusion",
        extra_meta: dict | None = None,
        qa: list[QAItem] | None = None,
    ) -> PageProject:
        """Emit an unchanged HD target page for page-manager exclusions.

        The page remains a first-class PageProject so navigation, resume metadata
        and book-level output stay synchronized even though no OCR or transfer was
        executed.
        """
        page_root = Path(page_root)
        page_root.mkdir(parents=True, exist_ok=True)
        if target is None:
            target = read_image(pair.target_path)
        if source is None:
            # Manual exclusions do not need source pixels. Use target dimensions as
            # a harmless identity-registration placeholder and avoid extra I/O.
            source_size = (int(target.shape[1]), int(target.shape[0]))
        else:
            source_size = (int(source.shape[1]), int(source.shape[0]))
        target_size = (int(target.shape[1]), int(target.shape[0]))
        if registration is None:
            registration = RegistrationResult(
                matrix=np.eye(3, dtype=np.float64),
                method="page_manager_passthrough",
                confidence=1.0, inlier_ratio=1.0, reprojection_error=0.0,
                spatial_coverage=1.0, num_matches=0,
                source_size=source_size, target_size=target_size,
                diagnostics={"page_management": mark.to_dict()},
            )
        final = Path(final_path) if final_path is not None else page_root / "final.png"
        final.parent.mkdir(parents=True, exist_ok=True)
        write_image(final, target)
        try:
            job_fingerprint = page_job_fingerprint(pair, self.config)
        except Exception:
            # A manually excluded target page can still be preserved even if its
            # optional source counterpart was moved after pairing. Such a page is
            # simply not eligible for resume fingerprint reuse.
            job_fingerprint = ""
        project = PageProject(
            page_id=stem_id(pair.target_path), pair=pair, registration=registration,
            source_blocks=[], target_blocks=[], source_bubbles=[], target_bubbles=[],
            source_units=[], target_units=[], matches=[], lettering=[], qa=list(qa or []),
            artifacts={"final": str(final), "book_final": str(final)},
            meta={
                "page_management": mark.to_dict(),
                "passthrough": True,
                "passthrough_reason": str(passthrough_reason or "page_manager_exclusion"),
                "job_fingerprint": job_fingerprint,
                "registration_route": registration.diagnostics.get("route", registration.method),
                "qa_summary": qa_summary(list(qa or [])),
                **dict(extra_meta or {}),
            },
        )
        save_json(page_root / "page_management.json", mark.to_dict())
        if self.config.export.save_project_json:
            save_json(page_root / "project.json", project.to_dict())
        return project


    def _emit_aligned_overlay_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None,
        mark: PageMark,
        *,
        source: np.ndarray,
        target: np.ndarray,
        registration: RegistrationResult,
        pair_check: PagePairingCheck,
        result: AlignedOverlayResult,
        requested_mode: str,
        planner_decision,
        cache_stats: dict[str, str],
    ) -> PageProject:
        """Persist one aligned-overlay candidate without invoking OCR/Mask.

        The pixel module owns no files.  This pipeline boundary and result_state
        own automatic artifacts/final output so manual Reveal can safely layer on
        top later.
        """
        page_root = Path(page_root)
        page_root.mkdir(parents=True, exist_ok=True)
        local_final, book_final = commit_automatic_result(page_root, result.image, final_path)

        source_original = page_root / "source_original.png"
        target_original = page_root / "target_original.png"
        layer_path = page_root / "aligned_overlay_reveal_layer.png"
        mask_path = page_root / "aligned_overlay_reveal_mask.png"
        source_mask_path = page_root / "aligned_overlay_reveal_source_ink.png"
        regions_path = page_root / "aligned_overlay_reveal_regions.png"
        meta_path = page_root / "aligned_overlay_reveal.json"
        review_preview = page_root / "review_preview.png"
        target_clear_mask = page_root / "target_clear_mask.png"

        write_image(source_original, source)
        write_image(target_original, target)
        write_rgba(layer_path, result.layer_rgba)
        write_image(mask_path, result.erase_mask)
        write_image(source_mask_path, result.source_ink_mask)
        write_image(regions_path, result.regions_overlay)
        write_image(target_clear_mask, result.erase_mask)

        regions = [r.to_dict() for r in result.plan.regions]
        manual_candidates = list(result.diagnostics.get("manual_effect_candidates", []) or [])
        review_regions = [
            {
                "id": r.get("id", ""),
                "target_bbox": list(r.get("target_bbox", [])),
                "source_bbox": list(r.get("source_bbox", [])),
                "reason": r.get("reason", ""),
                "triage": r.get("triage", "REVIEW"),
                "review_level": "required" if r.get("triage") == "REJECT" else "recommended",
                "restorable": True,
                "editable": True,
            }
            for r in regions if r.get("triage") in {"REVIEW", "REJECT"}
        ]
        write_image(review_preview, result.regions_overlay if review_regions else result.image)

        qa: list[QAItem] = []
        if not result.plan.accepted:
            qa.append(QAItem(
                "aligned_overlay_reveal_rejected", "warning",
                "Experimental aligned erase-to-reveal did not pass its strict gates; TARGET was kept unchanged.",
                meta={"reason": result.plan.reason, **dict(result.diagnostics)},
            ))
        elif result.plan.page_triage == "REVIEW":
            qa.append(QAItem(
                "aligned_overlay_reveal_review", "warning",
                "Experimental aligned erase-to-reveal produced reviewable regions; inspect the diagnostic overlay or finish with manual Reveal.",
                meta={"applied_count": result.applied_count},
            ))
        elif result.plan.page_triage == "SAFE":
            qa.append(QAItem(
                "aligned_overlay_reveal_safe", "info",
                "Experimental aligned erase-to-reveal passed strict white-region and registration gates.",
                meta={"applied_count": result.applied_count},
            ))

        payload = {
            "schema": "manga_hd_translation_transfer.aligned_overlay_reveal.v1",
            "requested_mode": requested_mode,
            "used": bool(result.applied_count > 0),
            "accepted": bool(result.plan.accepted),
            "reason": result.plan.reason,
            "page_triage": result.plan.page_triage,
            "contract": "target_background_authority",
            "diagnostics": dict(result.diagnostics),
            "regions": regions,
            "review_regions": review_regions,
            "manual_effect_candidates": manual_candidates,
            "planner": planner_decision.to_dict(),
            "page_pairing_check": pair_check.to_dict(),
        }
        save_json(meta_path, payload)

        artifacts = {
            "source_original": str(source_original),
            "target_original": str(target_original),
            "target_clear_mask": str(target_clear_mask),
            "aligned_overlay_reveal_layer": str(layer_path),
            "aligned_overlay_reveal_mask": str(mask_path),
            "aligned_overlay_reveal_source_ink": str(source_mask_path),
            "aligned_overlay_reveal_regions": str(regions_path),
            "aligned_overlay_reveal_json": str(meta_path),
            "review_preview": str(review_preview),
            "final": str(local_final),
            "book_final": str(book_final) if book_final is not None else str(local_final),
        }
        project = PageProject(
            page_id=stem_id(pair.target_path), pair=pair, registration=registration,
            source_blocks=[], target_blocks=[], source_bubbles=[], target_bubbles=[],
            source_units=[], target_units=[], matches=[], lettering=[], qa=qa,
            artifacts=artifacts,
            meta={
                "page_management": mark.to_dict(),
                "transfer_mode": requested_mode,
                "transfer_planner": planner_decision.to_dict(),
                "page_pairing_check": pair_check.to_dict(),
                "aligned_overlay_reveal": payload,
                "direct_patch": {"used": False, "manual_effect_candidates": []},
                "mask_replace": {"used": False, "review_regions": []},
                "auto_applied_count": int(result.applied_count),
                "job_fingerprint": page_job_fingerprint(pair, self.config),
                "cache": dict(cache_stats),
                "runtime": runtime_summary(self.config.runtime.device),
                "registration_route": registration.diagnostics.get("route", registration.method),
                "qa_summary": qa_summary(qa),
            },
        )
        save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
        if self.config.export.save_project_json:
            save_json(page_root / "project.json", project.to_dict())
        return project

    def _bubbles(self, image: np.ndarray, blocks, image_path: str | Path) -> list[BubbleInstance]:
        backend = self.config.bubbles.backend.lower()
        if backend == "none":
            return []
        if backend == "seeded_white":
            return detect_seeded_white_bubbles(image, blocks, self.config.bubbles)
        if backend == "mangalens":
            return detect_mangalens_bubbles(image, blocks, self.config.bubbles)
        if backend == "sidecar":
            return load_bubble_sidecar(image, image_path, blocks, self.config.bubbles)
        raise ValueError(f"Unsupported bubble backend: {self.config.bubbles.backend}")

    def _recognize_cached(self, role: str, backend: OCRBackend, image: np.ndarray, image_path: str | Path, cache: PageStageCache, stats: dict) -> list:
        sig = image_stage_signature(
            image_path, self.config.ocr,
            {"role": role, "backend": type(backend).__name__, "lang": self.config.ocr.source_lang if role == "source" else self.config.ocr.target_lang},
        )
        if self.config.cache.ocr:
            hit = cache.load_blocks(role, sig)
            if hit is not None:
                stats[f"ocr_{role}"] = "hit"
                return hit
        blocks = backend.recognize(image, image_path=image_path)
        if self.config.cache.ocr:
            cache.save_blocks(role, sig, blocks)
        stats[f"ocr_{role}"] = "miss"
        return blocks

    def _recognize_source_rectified_cached(
        self, backend: OCRBackend, source: np.ndarray, source_path: str | Path,
        target_shape: tuple[int, int], registration, cache: PageStageCache, stats: dict,
    ) -> list:
        H = transform_to_homography(registration.matrix)
        th, tw = target_shape
        rect_scale = 1.0
        if self.config.ocr.rectify_preserve_source_resolution:
            sh, sw = source.shape[:2]
            density_scale = min(sw / max(1, tw), sh / max(1, th))
            rect_scale = float(np.clip(density_scale, 1.0, self.config.ocr.rectify_max_scale))
            long_side = max(tw * rect_scale, th * rect_scale)
            if long_side > self.config.ocr.rectify_max_long_side:
                rect_scale *= self.config.ocr.rectify_max_long_side / long_side
                rect_scale = max(1.0, rect_scale)
        S = np.array([[rect_scale, 0.0, 0.0], [0.0, rect_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        H_ocr = S @ H
        sig = image_stage_signature(
            source_path, self.config.ocr,
            {
                "role": "source", "backend": type(backend).__name__,
                "lang": self.config.ocr.source_lang, "rectified": True,
                "target_shape": list(target_shape), "rectified_scale": round(rect_scale, 4),
                "registration": np.round(H, 6).tolist(),
            },
        )
        if self.config.cache.ocr:
            hit = cache.load_blocks("source", sig)
            if hit is not None:
                stats["ocr_source"] = "hit_rectified"
                return hit
        rectified = cv2.warpPerspective(
            source, H_ocr, (max(1, round(tw * rect_scale)), max(1, round(th * rect_scale))),
            flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
        )
        blocks = backend.recognize(rectified, image_path=None)
        try:
            inv = np.linalg.inv(H_ocr)
        except np.linalg.LinAlgError:
            inv = np.eye(3, dtype=np.float64)
        for block in blocks:
            block.polygon = transform_points(block.polygon, inv)
            block.meta["ocr_rectified_source"] = True
            block.meta["registration_confidence"] = float(registration.confidence)
            block.meta["ocr_rectified_scale"] = float(rect_scale)
        if self.config.cache.ocr:
            cache.save_blocks("source", sig, blocks)
        stats["ocr_source"] = "miss_rectified"
        return blocks

    def _recognize_paired_regions_text_only(
        self, backend: OCRBackend, source: np.ndarray, source_path: str | Path,
        source_bubbles: list[BubbleInstance], target_bubbles: list[BubbleInstance],
        cache: PageStageCache, stats: dict,
    ) -> tuple[list[TextBlock], list[TextBlock], list[BubbleInstance], list[BubbleInstance]]:
        """OCR already-detected translated bubbles with a transcript-only backend.

        VisionKit Live Text and the macOS ExtractText shortcut intentionally do
        not supply trustworthy per-line boxes.  For manga mask replacement that
        is a feature, not a limitation: paired-diff already knows the source and
        target bubble geometry.  OCR each source region independently and attach
        the returned Unicode text to that known geometry.  Target blocks are
        geometry-only placeholders so normal matching/lettering can reuse the
        existing pipeline without asking Apple OCR to rediscover positions.
        """
        source_to_target = {}
        target_to_source = {}
        for idx, sb in enumerate(source_bubbles):
            tid = str(sb.meta.get("paired_target_id") or "")
            if not tid and idx < len(target_bubbles):
                tid = target_bubbles[idx].id
            if tid:
                source_to_target[sb.id] = tid
                target_to_source[tid] = sb.id
        for tb in target_bubbles:
            sid = str(tb.meta.get("paired_source_id") or "")
            if sid:
                target_to_source[tb.id] = sid
                source_to_target[sid] = tb.id

        region_rows = [
            {
                "id": b.id, "bbox": [round(v, 2) for v in b.bbox],
                "paired": source_to_target.get(b.id, ""),
            }
            for b in source_bubbles
        ]
        sig = image_stage_signature(
            source_path, self.config.ocr,
            {
                "role": "source_paired_regions", "backend": type(backend).__name__,
                "lang": self.config.ocr.source_lang, "regions": region_rows,
                "orientation_policy": "source_ink_v1",
            },
        )
        if self.config.cache.ocr:
            cached = cache.load_blocks("source_paired_regions", sig)
            if cached is not None:
                by_bubble = {str(b.meta.get("paired_region_source_id")): b.id for b in cached}
                for bubble in source_bubbles:
                    bid = by_bubble.get(bubble.id)
                    bubble.block_ids = [bid] if bid else []
                target_blocks: list[TextBlock] = []
                for i, bubble in enumerate(target_bubbles):
                    source_id = target_to_source.get(bubble.id, str(bubble.meta.get("paired_source_id") or ""))
                    if source_id not in by_bubble:
                        bubble.block_ids = []
                        continue
                    block_id = f"apple-target-geometry-{i:04d}"
                    bubble.block_ids = [block_id]
                    target_blocks.append(TextBlock(
                        id=block_id, polygon=list(bubble.polygon), text="□",
                        confidence=1.0, kind=bubble.kind, reading_order=i, bubble_id=bubble.id,
                        meta={"backend": "paired_geometry", "synthetic_geometry_only": True},
                    ))
                stats["ocr_source"] = "hit_paired_regions"
                stats["ocr_target"] = "geometry_only"
                return cached, target_blocks, source_bubbles, target_bubbles

        h, w = source.shape[:2]
        pad_ratio = float(getattr(self.config.ocr, "apple_live_text_region_padding_ratio", 0.08))
        min_side = int(getattr(self.config.ocr, "apple_live_text_region_min_side_px", 28))
        whiten = bool(getattr(self.config.ocr, "apple_live_text_region_whiten_outside_mask", True))
        blocks: list[TextBlock] = []
        recognized_source_ids: set[str] = set()
        for i, bubble in enumerate(source_bubbles):
            x0, y0, x1, y1 = bubble.bbox
            bw, bh = max(1.0, x1-x0), max(1.0, y1-y0)
            pad = max(3, int(round(max(bw, bh) * pad_ratio)))
            ix0 = max(0, int(np.floor(x0)) - pad); iy0 = max(0, int(np.floor(y0)) - pad)
            ix1 = min(w, int(np.ceil(x1)) + pad); iy1 = min(h, int(np.ceil(y1)) + pad)
            if ix1-ix0 < min_side or iy1-iy0 < min_side:
                bubble.block_ids = []
                continue
            crop = source[iy0:iy1, ix0:ix1].copy()
            layout_crop = crop.copy()
            if whiten and bubble.mask is not None and bubble.mask.shape[:2] == source.shape[:2]:
                raw_local_mask = bubble.mask[iy0:iy1, ix0:ix1]
                # OCR gets a slightly expanded interior so antialiased edge glyphs
                # survive. Layout analysis gets the opposite: an eroded interior
                # that excludes balloon/panel outlines, otherwise border ink would
                # make short text look artificially huge.
                k = max(3, int(round(min(bw, bh) * 0.025)) | 1)
                ocr_mask = cv2.dilate(raw_local_mask, np.ones((k, k), np.uint8), iterations=1)
                crop[ocr_mask == 0] = 255
                inner_k = max(3, min(11, k | 1))
                inner = cv2.erode(raw_local_mask, np.ones((inner_k, inner_k), np.uint8), iterations=1)
                layout_crop[inner == 0] = 255
            try:
                raw = backend.recognize(crop, image_path=None)
            except Exception:
                raise
            text = "\n".join(str(b.text).strip() for b in raw if str(b.text).strip()).strip()
            if not text:
                bubble.block_ids = []
                continue
            orientation_hint, orientation_meta = _infer_region_orientation(crop, text, bubble.kind)
            layout_profile = _source_layout_profile(layout_crop, text, orientation_hint)
            confs = [float(b.confidence) for b in raw if str(b.text).strip()]
            conf = float(np.mean(confs)) if confs else float(getattr(self.config.ocr, "apple_live_text_assumed_confidence", 0.88))
            block_id = f"apple-region-{i:04d}"
            meta = {
                "backend": str(raw[0].meta.get("backend") if raw else "apple_live_text"),
                "paired_region_ocr": True,
                "paired_region_source_id": bubble.id,
                "paired_target_id": source_to_target.get(bubble.id, str(bubble.meta.get("paired_target_id") or "")),
                "ocr_region_bbox": [ix0, iy0, ix1, iy1],
                "text_only_geometry": "paired_bubble",
                "orientation_hint": orientation_hint,
                "orientation_evidence": orientation_meta,
                "source_layout_profile": layout_profile,
            }
            if raw:
                for key in ("apple_auto_route", "apple_live_text_fallback_reason", "languages"):
                    if key in raw[0].meta:
                        meta[key] = raw[0].meta[key]
            block = TextBlock(
                id=block_id, polygon=list(bubble.polygon), text=text, confidence=conf,
                kind=bubble.kind, reading_order=i, bubble_id=bubble.id, meta=meta,
            )
            blocks.append(block)
            bubble.block_ids = [block_id]
            recognized_source_ids.add(bubble.id)

        target_blocks: list[TextBlock] = []
        for i, bubble in enumerate(target_bubbles):
            source_id = target_to_source.get(bubble.id, str(bubble.meta.get("paired_source_id") or ""))
            if source_id not in recognized_source_ids:
                bubble.block_ids = []
                continue
            block_id = f"apple-target-geometry-{i:04d}"
            bubble.block_ids = [block_id]
            target_blocks.append(TextBlock(
                id=block_id, polygon=list(bubble.polygon), text="□", confidence=1.0,
                kind=bubble.kind, reading_order=i, bubble_id=bubble.id,
                meta={"backend": "paired_geometry", "synthetic_geometry_only": True},
            ))

        if self.config.cache.ocr:
            cache.save_blocks("source_paired_regions", sig, blocks)
        stats["ocr_source"] = "miss_paired_regions"
        stats["ocr_target"] = "geometry_only"
        return blocks, target_blocks, source_bubbles, target_bubbles

    def _bubbles_cached(self, role: str, image: np.ndarray, blocks, image_path: str | Path, cache: PageStageCache, stats: dict) -> list[BubbleInstance]:
        sig = image_stage_signature(
            image_path, self.config.bubbles,
            {"role": role, "blocks": blocks_signature(blocks)},
        )
        if self.config.cache.bubbles:
            hit = cache.load_bubbles(role, sig)
            if hit is not None:
                # Restore block -> bubble relations omitted from the compact cache.
                by_id = {b.id: b for b in hit}
                for b in hit:
                    for bid in b.block_ids:
                        for block in blocks:
                            if block.id == bid:
                                block.bubble_id = b.id
                                if block.kind == "unknown": block.kind = b.kind
                stats[f"bubbles_{role}"] = "hit"
                return hit
        bubbles = self._bubbles(image, blocks, image_path)
        if self.config.cache.bubbles:
            cache.save_bubbles(role, sig, bubbles)
        stats[f"bubbles_{role}"] = "miss"
        return bubbles

    def _accepted_matches(
        self,
        pair: PagePair,
        registration_confidence: float,
        source_units: list[TextUnit],
        target_units: list[TextUnit],
        matches: list[UnitMatch],
    ) -> list[UnitMatch]:
        if pair.confidence < self.config.pairing.confidence_floor:
            return []
        if registration_confidence < self.config.qa.registration_min_confidence:
            return []
        su = {u.id: u for u in source_units}
        tu = {u.id: u for u in target_units}
        accepted = []
        for match in matches:
            s, t = su.get(match.source_unit_id), tu.get(match.target_unit_id)
            if s is None or t is None:
                continue
            if match.relation != "one_to_one":
                continue
            if match.confidence < self.config.matching.review_confidence:
                continue
            if s.confidence < self.config.ocr.min_confidence or t.confidence < self.config.ocr.min_confidence:
                continue
            if s.kind not in self.config.matching.auto_apply_kinds or t.kind not in self.config.matching.auto_apply_kinds:
                continue
            if not s.text.strip():
                continue
            accepted.append(match)
        return accepted

    def process_page(
        self,
        pair: PagePair,
        page_root: str | Path,
        final_path: str | Path | None = None,
        *,
        page_mark: PageMark | dict | None = None,
        cancel_cb=None,
    ) -> PageProject:
        page_root = Path(page_root)
        page_root.mkdir(parents=True, exist_ok=True)
        # A fresh automatic process invalidates the previous manual-edit baseline.
        # Reviewed output may have been mirrored into final.png, so keeping an old
        # final_auto/manual_effect_base would make the next omission edit rebuild
        # against stale pixels from an earlier pipeline run.
        invalidate_manual_review_state(page_root)
        mark = page_mark if isinstance(page_mark, PageMark) else PageMark.from_dict(page_mark) if page_mark else PageMark(
            page_type="content", origin="default",
            source_name=Path(pair.source_path).name, target_name=Path(pair.target_path).name,
        )
        _check_cancel(cancel_cb, "before_page")
        if not mark.should_process:
            return self._passthrough_page(pair, page_root, final_path, mark)
        authority_source_path = pair.source_path
        source_path_local = authority_source_path
        target_path_local = pair.target_path
        authority_source = read_image(authority_source_path)
        source = authority_source
        target = read_image(target_path_local)
        replace_source_specs = _load_additional_source_specs(authority_source_path, self.config.replace_translation)
        secondary_source_spec = _resolve_secondary_source_spec(authority_source_path, target_path_local, self.config.dual_source)
        if secondary_source_spec is not None and all(str(x.get("path")) != str(secondary_source_spec.get("path")) for x in replace_source_specs):
            replace_source_specs.append(secondary_source_spec)
        secondary_source_available = secondary_source_spec is not None
        selected_source_kind = "primary"
        selected_secondary_source = False
        dual_source_arbitration: list[dict] = []
        _check_cancel(cancel_cb, "after_decode")
        stage_cache = PageStageCache(page_root, enabled=self.config.cache.enabled)
        cache_stats: dict[str, str] = {}

        reg_sig = registration_stage_signature(pair, self.config.registration)
        registration = stage_cache.load_registration(reg_sig) if self.config.cache.registration else None
        if registration is None:
            registration = register_images(source, target, self.config.registration)
            if self.config.cache.registration:
                stage_cache.save_registration(reg_sig, registration)
            cache_stats["registration"] = "miss"
        else:
            cache_stats["registration"] = "hit"
        _check_cancel(cancel_cb, "after_registration")
        mode = self.config.transfer.mode.lower().strip()
        if mode not in {"auto", "direct_patch", "aligned_overlay_reveal", "reletter", "mask_replace", "hybrid"}:
            raise ValueError(f"Unsupported transfer.mode: {self.config.transfer.mode}")

        # Phase-1 destructive safety: verify that the registered pair still depicts
        # the same page. This check is OCR-free and heavily suppresses glyph detail.
        if bool(getattr(self.config.pairing, "same_page_precheck_enabled", True)):
            pair_check = verify_registered_page_pair(
                source, target, registration,
                max_side=int(getattr(self.config.pairing, "same_page_max_side", 720)),
                min_confidence=float(getattr(self.config.pairing, "same_page_min_confidence", 0.72)),
                min_valid_ratio=float(getattr(self.config.pairing, "same_page_min_valid_ratio", 0.45)),
            )
        else:
            pair_check = PagePairingCheck(
                True, float(registration.confidence), float(registration.confidence),
                float(registration.confidence), float(registration.confidence),
                {"disabled": True, "ocr_used": False},
            )

        # Experimental page-aligned erase-to-reveal is a first-class explicit
        # route. It never silently falls through to Direct/Mask, and its pixel
        # module cannot write final/project state.
        if mode == "aligned_overlay_reveal":
            a_cfg = self.config.aligned_overlay_reveal
            if not bool(a_cfg.enabled):
                disabled_decision = choose_transfer_strategy(
                    mode, same_page=bool(pair_check.same_page),
                    same_page_confidence=float(pair_check.confidence),
                    direct_plan_available=False, direct_plan_safe=False,
                    aligned_plan_available=False, aligned_plan_safe=False,
                )
                return self._passthrough_page(
                    pair, page_root, final_path, mark, source=source, target=target,
                    registration=registration, passthrough_reason="aligned_overlay_reveal_disabled",
                    extra_meta={
                        "transfer_mode": mode,
                        "transfer_planner": disabled_decision.to_dict(),
                        "page_pairing_check": pair_check.to_dict(),
                        "aligned_overlay_reveal": {
                            "used": False, "attempted": False, "reason": "feature_flag_disabled",
                            "review_required": True, "manual_effect_candidates": [],
                        },
                    },
                    qa=[QAItem(
                        "aligned_overlay_reveal_disabled", "warning",
                        "Experimental aligned erase-to-reveal is disabled by configuration; TARGET was kept unchanged.",
                    )],
                )
            aligned_plan = build_aligned_overlay_plan(source, target, registration, a_cfg)
            if not bool(pair_check.same_page):
                aligned_plan.accepted = False
                aligned_plan.reason = "rejected_page_pair_verification"
                aligned_plan.erase_mask[:] = 0
                aligned_plan.source_ink_mask[:] = 0
                aligned_plan.full_raster_mask[:] = 0
                aligned_plan.diagnostics["page_pairing_rejected"] = True
                aligned_plan.diagnostics["page_pairing_check"] = pair_check.to_dict()
                for region in aligned_plan.regions:
                    region.triage = "REJECT"
                    region.reason = "page_pair_verification_failed"
            aligned_result = execute_aligned_overlay(aligned_plan, source, target, a_cfg)
            aligned_available = bool(aligned_result.plan.accepted and aligned_result.applied_count > 0)
            aligned_safe = bool(aligned_available and aligned_result.plan.page_triage == "SAFE")
            aligned_decision = choose_transfer_strategy(
                mode, same_page=bool(pair_check.same_page),
                same_page_confidence=float(pair_check.confidence),
                direct_plan_available=False, direct_plan_safe=False,
                aligned_plan_available=aligned_available, aligned_plan_safe=aligned_safe,
            )
            return self._emit_aligned_overlay_page(
                pair, page_root, final_path, mark, source=source, target=target,
                registration=registration, pair_check=pair_check, result=aligned_result,
                requested_mode=mode, planner_decision=aligned_decision, cache_stats=cache_stats,
            )

        direct_container_plan = None
        direct_container_fast = False
        direct_requested = mode in {"direct_patch", "auto"}
        if direct_requested and (pair_check.same_page or not bool(getattr(self.config.direct_patch, "require_same_page_precheck", True))):
            try:
                # Direct Patch has its own configuration namespace and content
                # contract. It is no longer a hidden optimization inside Mask.
                source_hints, provider_audit = run_source_detector_chain(
                    source, self.config.direct_patch, self.config.bubbles,
                    existing=[], source_path=source_path_local, allow_expensive=False,
                )
                direct_container_plan = build_source_direct_container_plan(
                    source, target, registration, self.config.direct_patch,
                    source_hint_bubbles=source_hints,
                )
                direct_plan_safe = bool(
                    direct_container_plan is not None
                    and direct_container_plan.safe_to_skip_other_paths
                    and direct_container_plan.result.applied_count > 0
                )
                if (
                    not direct_plan_safe
                    and bool(getattr(self.config.direct_patch, "source_direct_detector_fallback_enabled", False))
                ):
                    expensive_hints, expensive_audit = run_source_detector_chain(
                        source, self.config.direct_patch, self.config.bubbles,
                        existing=(direct_container_plan.source_bubbles if direct_container_plan is not None else source_hints),
                        source_path=source_path_local, allow_expensive=True, only_expensive=True,
                    )
                    provider_audit.extend(expensive_audit)
                    if expensive_hints:
                        direct_container_plan = build_source_direct_container_plan(
                            source, target, registration, self.config.direct_patch,
                            source_hint_bubbles=source_hints + expensive_hints,
                        )
                        direct_plan_safe = bool(
                            direct_container_plan is not None
                            and direct_container_plan.safe_to_skip_other_paths
                            and direct_container_plan.result.applied_count > 0
                        )
                if direct_container_plan is not None:
                    direct_container_plan.diagnostics["source_detector_provider_audit"] = provider_audit
                    qa_provider = PROVIDER_REGISTRY.get("qa_check", "source_direct_invariants")
                    invariant_issues = qa_provider(direct_container_plan) if qa_provider is not None else []
                    direct_container_plan.diagnostics["plugin_invariant_issues"] = invariant_issues
                    blocking_invariants = _blocking_direct_invariant_issues(invariant_issues)
                    direct_container_plan.diagnostics["plugin_blocking_invariant_issues"] = blocking_invariants
                    if blocking_invariants:
                        direct_container_plan.safe_to_skip_other_paths = False
                        direct_plan_safe = False
                    direct_container_plan.diagnostics["provider_registry"] = PROVIDER_REGISTRY.snapshot()
                    direct_container_plan.diagnostics["source_completion_plan_strategy"] = "cheap_source_hints_then_single_plan"
                # Explicit Direct mode uses only accepted Direct regions and never
                # silently switches to mask/OCR. Auto requires a fully safe plan.
                direct_container_fast = bool(
                    direct_container_plan is not None
                    and direct_container_plan.result.applied_count > 0
                    and (mode == "direct_patch" or direct_plan_safe)
                )
            except Exception as exc:
                logger.warning("Direct Patch plan failed: %s", exc)
                direct_container_plan = None
                direct_container_fast = False
        dual_prefer = bool(getattr(self.config.dual_source, "enabled", False) and getattr(self.config.dual_source, "prefer_secondary_for_direct", True))
        allow_replace_retry = bool(getattr(self.config.replace_translation, "additional_source_retry_direct", True))
        allow_secondary_retry = bool(getattr(self.config.dual_source, "enabled", False) and getattr(self.config.dual_source, "accept_secondary_direct", True))
        arbitration_enabled = bool(getattr(self.config.dual_source, "arbitration_enabled", True))
        arbitration_candidates = []
        if direct_requested and direct_container_plan is not None:
            primary_ev = build_direct_source_evidence(
                path=authority_source_path, kind="primary", is_secondary=False,
                source=authority_source, registration=registration, pair_check=pair_check,
                plan=direct_container_plan, config=self.config.dual_source,
            )
            dual_source_arbitration.append(primary_ev.to_dict())
            arbitration_candidates.append((primary_ev, {
                "spec": {"path": authority_source_path, "kind": "primary", "origin": "primary"},
                "source": authority_source, "registration": registration, "pair_check": pair_check,
                "plan": direct_container_plan, "is_secondary": False,
            }))
        should_try_alternates = bool(
            direct_requested and replace_source_specs and (
                not direct_container_fast
                or (allow_secondary_retry and secondary_source_available)
                or (not arbitration_enabled and dual_prefer and secondary_source_available)
            )
        )
        legacy_best_alt = None
        if should_try_alternates:
            for spec in replace_source_specs:
                is_secondary = str(spec.get("origin", "")) == "dual_source" or str(spec.get("kind", "")) == "secondary_dir"
                if is_secondary and not allow_secondary_retry:
                    continue
                if (not is_secondary) and not allow_replace_retry:
                    continue
                try:
                    alt_source = read_image(spec["path"])
                    alt_registration = register_images(alt_source, target, self.config.registration)
                    alt_check = verify_registered_page_pair(
                        alt_source, target, alt_registration,
                        max_side=int(getattr(self.config.pairing, "same_page_max_side", 720)),
                        min_confidence=float(getattr(self.config.pairing, "same_page_min_confidence", 0.72)),
                        min_valid_ratio=float(getattr(self.config.pairing, "same_page_min_valid_ratio", 0.45)),
                    )
                    if not alt_check.same_page and bool(getattr(self.config.direct_patch, "require_same_page_precheck", True)):
                        # Keep rejected evidence for diagnostics when possible.
                        alt_plan = None
                    else:
                        alt_hints, alt_audit = run_source_detector_chain(
                            alt_source, self.config.direct_patch, self.config.bubbles,
                            existing=[], source_path=spec["path"], allow_expensive=False,
                        )
                        alt_plan = build_source_direct_container_plan(
                            alt_source, target, alt_registration, self.config.direct_patch,
                            source_hint_bubbles=alt_hints,
                        )
                        if alt_plan is not None:
                            alt_plan.diagnostics["source_detector_provider_audit"] = alt_audit
                            alt_plan.diagnostics["replace_translation_additional_source"] = dict(spec)
                            qa_provider = PROVIDER_REGISTRY.get("qa_check", "source_direct_invariants")
                            invariant_issues = qa_provider(alt_plan) if qa_provider is not None else []
                            alt_plan.diagnostics["plugin_invariant_issues"] = invariant_issues
                            blocking_invariants = _blocking_direct_invariant_issues(invariant_issues)
                            alt_plan.diagnostics["plugin_blocking_invariant_issues"] = blocking_invariants
                            if blocking_invariants:
                                alt_plan.safe_to_skip_other_paths = False
                    alt_ev = build_direct_source_evidence(
                        path=spec["path"], kind=str(spec.get("kind", "alternate")), is_secondary=is_secondary,
                        source=alt_source, registration=alt_registration, pair_check=alt_check,
                        plan=alt_plan, config=self.config.dual_source,
                    )
                    dual_source_arbitration.append(alt_ev.to_dict())
                    payload = {
                        "spec": spec, "source": alt_source, "registration": alt_registration,
                        "pair_check": alt_check, "plan": alt_plan, "is_secondary": is_secondary,
                    }
                    arbitration_candidates.append((alt_ev, payload))
                    legacy_score = (
                        1 if alt_ev.safe else 0,
                        2 if (is_secondary and dual_prefer) else 1,
                        alt_ev.applied_count,
                        int(alt_source.shape[0] * alt_source.shape[1]),
                        alt_ev.registration_confidence,
                    )
                    if legacy_best_alt is None or legacy_score > legacy_best_alt[0]:
                        legacy_best_alt = (legacy_score, alt_ev, payload)
                except Exception as exc:
                    logger.warning("Alternate replace_translation source failed (%s): %s", spec.get("path"), exc)
        selected_candidate = select_direct_source_candidate(arbitration_candidates) if arbitration_enabled else None
        if selected_candidate is not None:
            selected_ev, payload = selected_candidate
            # An explicit Direct primary may be a reviewable partial plan. Arbitration
            # replaces it only with a fully publication-safe candidate.
            if selected_ev.path != str(authority_source_path):
                source = payload["source"]
                registration = payload["registration"]
                pair_check = payload["pair_check"]
                direct_container_plan = payload["plan"]
                source_path_local = str(payload["spec"]["path"])
                selected_source_kind = str(payload["spec"].get("kind", "alternate"))
                selected_secondary_source = bool(payload["is_secondary"])
                direct_container_fast = True
            elif direct_container_plan is not None:
                source = authority_source
                source_path_local = authority_source_path
                selected_source_kind = "primary"
                selected_secondary_source = False
                # Auto requires a safe primary. Explicit Direct preserves its prior
                # reviewable-partial semantics if no safer candidate wins.
                if mode == "auto":
                    direct_container_fast = bool(direct_container_plan.safe_to_skip_other_paths and direct_container_plan.result.applied_count > 0)
        elif (not arbitration_enabled) and legacy_best_alt is not None:
            _legacy_score, ev, payload = legacy_best_alt
            if ev.safe and (not direct_container_fast or (payload["is_secondary"] and dual_prefer)):
                source = payload["source"]
                registration = payload["registration"]
                pair_check = payload["pair_check"]
                direct_container_plan = payload["plan"]
                source_path_local = str(payload["spec"]["path"])
                selected_source_kind = str(payload["spec"].get("kind", "alternate"))
                selected_secondary_source = bool(payload["is_secondary"])
                direct_container_fast = True
        selected_arbitration_evidence = next(
            (row for row in dual_source_arbitration if str(row.get("path")) == str(source_path_local)),
            None,
        )
        # Experimental aligned erase-to-reveal is intentionally evaluated only
        # after Direct arbitration.  Auto may use it only when the dedicated
        # feature flag is on, auto is explicitly allowed, and the strict
        # require-explicit guard has been relaxed by the user.
        aligned_auto_result = None
        aligned_auto_allowed = bool(
            mode == "auto"
            and self.config.aligned_overlay_reveal.enabled
            and self.config.aligned_overlay_reveal.allow_in_auto
            and not self.config.aligned_overlay_reveal.require_explicit_mode
            and bool(pair_check.same_page)
            and not direct_container_fast
        )
        if aligned_auto_allowed:
            try:
                aligned_auto_plan = build_aligned_overlay_plan(
                    source, target, registration, self.config.aligned_overlay_reveal
                )
                aligned_auto_result = execute_aligned_overlay(
                    aligned_auto_plan, source, target, self.config.aligned_overlay_reveal
                )
            except Exception as exc:
                logger.warning("Aligned overlay reveal auto candidate failed: %s", exc)
                aligned_auto_result = None

        aligned_auto_available = bool(
            aligned_auto_result is not None
            and aligned_auto_result.accepted
            and aligned_auto_result.applied_count > 0
        )
        aligned_auto_safe = bool(
            aligned_auto_available
            and str(aligned_auto_result.page_triage).upper() == "SAFE"
        )
        decision = choose_transfer_strategy(
            mode,
            same_page=bool(pair_check.same_page),
            same_page_confidence=float(pair_check.confidence),
            direct_plan_available=bool(direct_container_plan is not None and direct_container_plan.result.applied_count > 0),
            direct_plan_safe=bool(direct_container_plan is not None and direct_container_plan.safe_to_skip_other_paths),
            secondary_source_available=bool(secondary_source_available),
            secondary_source_selected=bool(selected_secondary_source),
            aligned_plan_available=aligned_auto_available,
            aligned_plan_safe=aligned_auto_safe,
            aligned_auto_allowed=aligned_auto_allowed,
        )

        if decision.strategy == "aligned_overlay_reveal" and aligned_auto_result is not None:
            return self._emit_aligned_overlay_page(
                pair, page_root, final_path, mark, source=source, target=target,
                registration=registration, pair_check=pair_check,
                result=aligned_auto_result, requested_mode=mode,
                planner_decision=decision, cache_stats=cache_stats,
            )

        # Direct Patch is strict: a wrong page or a page with no safe whole-raster
        # container is preserved unchanged and marked for review. It does not become
        # Mask Replace behind the user's back.
        if mode == "direct_patch" and not direct_container_fast:
            reject_mark = PageMark(
                page_type="content", origin=mark.origin, confidence=float(pair_check.confidence),
                reason=f"direct_patch_rejected:{decision.reason}",
                bubble_regions=0, free_text_regions=0,
                registration_confidence=float(registration.confidence),
                source_name=Path(pair.source_path).name, target_name=Path(pair.target_path).name,
            )
            return self._passthrough_page(
                pair, page_root, final_path, reject_mark, source=source, target=target,
                registration=registration, passthrough_reason="direct_patch_rejected",
                extra_meta={
                    "transfer_mode": mode,
                    "transfer_planner": decision.to_dict(),
                    "page_pairing_check": pair_check.to_dict(),
                    "direct_patch": {
                        "used": False,
                        "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                        "manual_effect_candidates": list(((direct_container_plan.diagnostics if direct_container_plan is not None else {}) or {}).get("manual_effect_candidates", []) or []),
                        "review_required": True,
                    },
                },
                qa=[QAItem(
                    "direct_patch_rejected", "warning",
                    "Direct Patch could not prove a safe same-layout whole-container transfer; target page was kept unchanged. Use Mask Transfer/Auto for non-identical layouts.",
                    meta={"reason": decision.reason, "pairing": pair_check.to_dict()},
                )],
            )

        paired_diff = None
        use_paired_diff = False
        # Keep structural recovery reachable at its own lower confidence floor.
        # The photo threshold is stricter and must not gate structural pages.
        paired_gate = min(
            self.config.mask_replace.paired_diff_min_registration_confidence,
            self.config.mask_replace.photo_pair_min_registration_confidence,
            self.config.mask_replace.paired_diff_structural_min_registration_confidence,
        )
        if (not direct_container_fast and mode in {"auto", "mask_replace", "hybrid"} and self.config.mask_replace.paired_diff_enabled
                and registration.confidence >= paired_gate):
            try:
                paired_diff = extract_paired_diff_bubbles(source, target, registration, self.config.mask_replace)
                use_paired_diff = bool(paired_diff.source_bubbles and paired_diff.target_bubbles)
            except Exception as exc:
                logger.warning("Paired-difference bubble extraction failed; falling back to OCR/bubble pipeline: %s", exc)
        _check_cancel(cancel_cb, "after_paired_diff")

        source_backend: OCRBackend | None = None
        if direct_container_fast and direct_container_plan is not None:
            source_blocks = []
            target_blocks = []
            source_bubbles = direct_container_plan.source_bubbles
            target_bubbles = direct_container_plan.target_bubbles
            cache_stats["ocr_source"] = "skipped_source_direct_container"
            cache_stats["ocr_target"] = "skipped_source_direct_container"
            cache_stats["bubbles_source"] = "source_direct_container"
            cache_stats["bubbles_target"] = "source_direct_container"
        elif use_paired_diff and paired_diff.safe_to_skip_ocr and self.config.mask_replace.paired_diff_skip_ocr:
            source_blocks = []
            target_blocks = []
            source_bubbles = paired_diff.source_bubbles
            target_bubbles = paired_diff.target_bubbles
            cache_stats["ocr_source"] = "skipped_paired_diff"
            cache_stats["ocr_target"] = "skipped_paired_diff"
            cache_stats["bubbles_source"] = "paired_diff"
            cache_stats["bubbles_target"] = "paired_diff"
        else:
            source_backend = self.source_ocr
            # Apple VisionKit/Shortcuts are transcript-only routes.  When the
            # paired-difference detector already supplied reliable bubble geometry,
            # OCR each translated source bubble independently and bind its text to
            # that geometry.  Do not ask Apple's OCR to rediscover bounding boxes.
            if (use_paired_diff and getattr(source_backend, "region_text_only", False)
                    and paired_diff is not None and paired_diff.source_bubbles and paired_diff.target_bubbles):
                source_blocks, target_blocks, source_bubbles, target_bubbles = self._recognize_paired_regions_text_only(
                    source_backend, source, source_path_local,
                    paired_diff.source_bubbles, paired_diff.target_bubbles,
                    stage_cache, cache_stats,
                )
            else:
                can_rectify_ocr = type(source_backend).__name__ != "InjectedOCRBackend"
                if (self.config.ocr.rectify_source_with_registration and can_rectify_ocr
                        and registration.confidence >= self.config.ocr.rectify_min_registration_confidence):
                    source_blocks = self._recognize_source_rectified_cached(
                        source_backend, source, source_path_local, target.shape[:2], registration, stage_cache, cache_stats
                    )
                else:
                    source_blocks = self._recognize_cached("source", source_backend, source, source_path_local, stage_cache, cache_stats)
                target_blocks = self._recognize_cached("target", self.target_ocr, target, target_path_local, stage_cache, cache_stats)
                source_bubbles = self._bubbles_cached("source", source, source_blocks, source_path_local, stage_cache, cache_stats)
                target_bubbles = self._bubbles_cached("target", target, target_blocks, target_path_local, stage_cache, cache_stats)
                # Keep OCR-seeded bubbles for text identity/fallback. Photo-pair masks
                # are a separate high-precision transfer geometry layer; replacing the
                # OCR bubbles here would discard real OCR block associations.

        # Content is now the default page type, but it must not mean "force a
        # replacement".  If source OCR actually ran and no OCR-backed Chinese
        # speech/narration container exists, preserve the HD Japanese page exactly.
        # If OCR was intentionally skipped/unavailable, fail open and keep the
        # established precise-mask path instead of risking a false negative.
        ocr_route = str(cache_stats.get("ocr_source", ""))
        configured_source_ocr = str(self.config.ocr.source_backend or self.config.ocr.backend or "").strip().lower()
        source_evidence_available = (
            source_backend is not None
            and configured_source_ocr not in {"", "none", "null"}
            and not ocr_route.startswith("skipped")
        )
        real_source_blocks = [b for b in source_blocks if str(getattr(b, "text", "")).strip()]
        region_text_only = bool(source_backend is not None and getattr(source_backend, "region_text_only", False))
        no_transferable_source_text = False
        if source_evidence_available:
            if not real_source_blocks:
                # Strong negative evidence: OCR actually ran and found no source
                # text at all.
                no_transferable_source_text = True
            elif region_text_only:
                # Transcript-only Apple routes were invoked *inside already known
                # paired bubble/text-box geometry*, so non-empty OCR outside those
                # candidates cannot confuse the decision.
                no_transferable_source_text = not _has_transferable_source_text(
                    source_blocks, source_bubbles, self.config.mask_replace.enabled_kinds,
                    paired_diff.source_bubbles if paired_diff is not None else None,
                )
            # Full-page OCR with text but no reconstructed bubble is ambiguous.
            # Fail open rather than silently skipping a real translation because
            # the bubble detector itself missed an unusual/open container.
        if (bool(self.config.page_management.skip_transfer_when_source_has_no_text_boxes)
                and no_transferable_source_text):
            no_text_mark = PageMark(
                page_type="content", origin=mark.origin if mark.origin in {"default", "manual"} else "default",
                confidence=1.0, reason="source_no_transferable_chinese_text_box;keep_target_unchanged",
                bubble_regions=0, free_text_regions=0,
                registration_confidence=float(registration.confidence),
                source_name=Path(pair.source_path).name, target_name=Path(pair.target_path).name,
            )
            return self._passthrough_page(
                pair, page_root, final_path, no_text_mark, source=source, target=target,
                registration=registration, passthrough_reason="source_no_transferable_text",
            )

        _check_cancel(cancel_cb, "after_ocr_and_bubbles")
        mask_source_bubbles = paired_diff.source_bubbles if use_paired_diff else source_bubbles
        mask_target_bubbles = paired_diff.target_bubbles if use_paired_diff else target_bubbles

        source_units = build_text_units(source_blocks, source_bubbles, "src")
        target_units = build_text_units(target_blocks, target_bubbles, "dst")
        match_result = match_units(source_units, target_units, registration, self.config.matching)
        matches = match_result.matches
        accepted = self._accepted_matches(pair, registration.confidence, source_units, target_units, matches)

        mask_transfer = None
        unseeded_white_pair_count = 0
        transfer_rgba = np.zeros((target.shape[0], target.shape[1], 4), dtype=np.uint8)
        fallback_matches = accepted

        _check_cancel(cancel_cb, "before_transfer")
        if mode in {"auto", "direct_patch", "mask_replace", "hybrid"}:
            if direct_container_fast and direct_container_plan is not None:
                # Direct remains the primary renderer, but a publication-safe
                # OCR-free completion pass is now allowed for *isolated ordinary
                # white balloons* that Direct did not discover.  This fixes the
                # silent-missing-bubble failure where all Direct records were SAFE
                # yet one or two plain speech balloons were absent from the plan.
                mask_transfer = direct_container_plan.result
                if (bool(getattr(self.config.direct_patch, "rigid_container_unseeded_completion_enabled", True))
                        and _cross_rendition_monochrome_source(source, target)
                        and registration.confidence >= float(getattr(self.config.direct_patch, "rigid_container_unseeded_min_registration_confidence", 0.72))):
                    extra_src, extra_dst = pair_unseeded_white_containers(
                        source, target, registration, self.config.direct_patch, self.config.bubbles,
                        existing_target_bubbles=direct_container_plan.target_bubbles,
                    )
                    existing_boxes = [tuple(map(int, r.target_bbox)) for r in list(mask_transfer.records or [])
                                      if bool(getattr(r, "applied", False)) and getattr(r, "target_bbox", None)]
                    extra_src, extra_dst = _filter_uncovered_white_completion_pairs(
                        extra_src, extra_dst, existing_boxes, self.config.direct_patch,
                    )
                    unseeded_white_pair_count = len(extra_src)
                    if extra_src and extra_dst:
                        recovered = transfer_rigid_container_rasters(
                            source, target, mask_transfer.image, extra_src, extra_dst, self.config.direct_patch,
                        )
                        if recovered.records:
                            mask_transfer = _merge_mask_transfer(mask_transfer, recovered)
            elif (
                paired_diff is not None
                and paired_diff.aligned_source is not None
                and self.config.mask_replace.paired_diff_target_driven_transfer
                and (
                    paired_diff.method == "structural_v08"
                    or (paired_diff.method == "photo_pair"
                        and self.config.mask_replace.photo_pair_target_driven_enabled
                        and _cross_rendition_monochrome_source(source, target))
                )
            ):
                # v0.8.23: same-layout white containers are rendered from the
                # ORIGINAL source page with one uniform local scale.  The affine
                # page registration remains useful for pairing/detection, but its
                # anisotropic X/Y correction is never inherited by final CJK
                # raster glyphs.  Anything that fails this strict white-container
                # gate continues through the established component/saturated path.
                rigid = transfer_rigid_container_rasters(
                    source, target, target, mask_source_bubbles, mask_target_bubbles,
                    self.config.mask_replace,
                )
                handled = {r.target_bubble_id for r in rigid.records if r.applied}
                rem_s, rem_t = _remaining_paired_bubbles(mask_source_bubbles, mask_target_bubbles, handled)
                mask_transfer = rigid
                if rem_s and rem_t:
                    legacy = transfer_paired_diff_regions(
                        paired_diff.aligned_source, mask_transfer.image, rem_s, rem_t,
                        self.config.mask_replace,
                    )
                    mask_transfer = _merge_mask_transfer(mask_transfer, legacy)
            else:
                mask_transfer = transfer_bubble_patches(
                    source,
                    target,
                    mask_source_bubbles,
                    mask_target_bubbles,
                    registration,
                    self.config.mask_replace,
                )

            # v0.8.5: photographed pages can carry a tightly filtered v0.8
            # structural supplement for open burst bubbles/free text that the
            # conservative closed-container route intentionally misses. Apply it
            # only after the main photo transfer, using target-driven masks and the
            # locally aligned source, then merge the editable layers/QA records.
            supplement = getattr(paired_diff, "supplemental", None) if paired_diff is not None else None
            if (not direct_container_fast and supplement is not None and supplement.aligned_source is not None
                    and supplement.source_bubbles and supplement.target_bubbles):
                # Structural local/dense flow is excellent for *detecting* changed
                # islands but it can bend CJK strokes on B/W -> colour editions.
                # Keep that geometry for the masks, while sourcing final Chinese
                # pixels from the primary/global registration whenever possible.
                supplement_transfer_source = supplement.aligned_source
                if (paired_diff is not None and paired_diff.aligned_source is not None
                        and paired_diff.aligned_source.shape[:2] == target.shape[:2]):
                    supplement_transfer_source = paired_diff.aligned_source
                # The structural detector is also allowed to discover ordinary
                # white bubbles that the closed-container pass missed.  Try the
                # same locked whole-raster route first; only coloured/open regions
                # fall through to component/saturated reconstruction.
                rigid_extra = transfer_rigid_container_rasters(
                    source, target, mask_transfer.image,
                    supplement.source_bubbles, supplement.target_bubbles,
                    self.config.mask_replace,
                )
                rigid_handled = {r.target_bubble_id for r in rigid_extra.records if r.applied}
                if rigid_extra.records:
                    mask_transfer = _merge_mask_transfer(mask_transfer, rigid_extra)
                rem_s, rem_t = _remaining_paired_bubbles(
                    supplement.source_bubbles, supplement.target_bubbles, rigid_handled
                )
                if rem_s and rem_t:
                    extra = transfer_paired_diff_regions(
                        supplement_transfer_source, mask_transfer.image,
                        rem_s, rem_t, self.config.mask_replace,
                    )
                    if extra.records:
                        mask_transfer = _merge_mask_transfer(mask_transfer, extra)
            # v0.8.25 completion pass: OCR-free white-container pairing. The
            # source and target editions often have identical dialogue balloons
            # even when paired-diff misses one entirely (for example a lower-right
            # speech box whose text changed very little). Detect full enclosed
            # white containers on both pages, pair them through page registration,
            # then send *only* those pairs through the rigid uniform-raster gate.
            # Failed eligibility is ignored rather than falling into component
            # transfer, so architecture/panel false positives cannot be published.
            completion_needed = _mask_transfer_completion_needed(mask_transfer)
            if (not direct_container_fast
                    and bool(getattr(self.config.mask_replace, "rigid_container_unseeded_completion_enabled", True))
                    and _cross_rendition_monochrome_source(source, target)
                    and registration.confidence >= float(getattr(self.config.mask_replace, "rigid_container_unseeded_min_registration_confidence", 0.72))):
                existing_completion_targets = _completion_existing_target_bubbles(
                    mask_transfer,
                    mask_target_bubbles,
                    supplement.target_bubbles if supplement is not None else [],
                )
                extra_src, extra_dst = pair_unseeded_white_containers(
                    source, target, registration, self.config.mask_replace, self.config.bubbles,
                    # Only already-applied target containers block the OCR-free
                    # completion retry. Rejected proposals remain eligible, but
                    # successfully transferred bubbles must not be rediscovered
                    # and written a second time.
                    existing_target_bubbles=existing_completion_targets,
                )
                review_boxes = _completion_review_regions(mask_transfer)
                extra_src, extra_dst = _completion_filter_pairs_to_review_regions(extra_src, extra_dst, review_boxes)
                existing_boxes = [tuple(map(int, r.target_bbox)) for r in list(mask_transfer.records or [])
                                  if bool(getattr(r, "applied", False)) and getattr(r, "target_bbox", None)]
                extra_src, extra_dst = _filter_uncovered_white_completion_pairs(extra_src, extra_dst, existing_boxes, self.config.mask_replace)
                unseeded_white_pair_count = len(extra_src)
                if extra_src and extra_dst:
                    recovered = transfer_rigid_container_rasters(
                        source, target, mask_transfer.image, extra_src, extra_dst, self.config.mask_replace,
                    )
                    if recovered.records:
                        mask_transfer = _merge_mask_transfer(mask_transfer, recovered)

            # v0.8.21 completeness fallback: OCR may confirm a source/target text
            # correspondence that paired-diff/container geometry missed.  In
            # Precise Mask mode OCR is geometry/evidence only: clear concrete
            # Japanese glyph components and copy registered source raster ink; it
            # never re-typesets or substitutes OCR text.  Low-confidence matches
            # become reversible review candidates instead of disappearing.
            if (not direct_container_fast
                    and bool(getattr(self.config.mask_replace, "ocr_guided_component_transfer_enabled", True))
                    and registration.confidence >= float(getattr(self.config.mask_replace, "ocr_guided_min_registration_confidence", 0.62))
                    and source_units and target_units and matches):
                aligned_for_ocr = None
                if (paired_diff is not None and paired_diff.aligned_source is not None
                        and paired_diff.aligned_source.shape[:2] == target.shape[:2]):
                    aligned_for_ocr = paired_diff.aligned_source
                if aligned_for_ocr is None:
                    H = transform_to_homography(registration.matrix)
                    aligned_for_ocr = cv2.warpPerspective(
                        source, H, (target.shape[1], target.shape[0]),
                        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(255, 255, 255),
                    )
                ocr_extra = transfer_ocr_guided_text_units(
                    aligned_for_ocr, mask_transfer.image, source_units, target_units,
                    matches, registration, self.config.mask_replace,
                    exclude_mask=mask_transfer.composite_mask,
                )
                if ocr_extra.records:
                    mask_transfer.image = ocr_extra.image
                    use = ocr_extra.layer_rgba[..., 3] > 0
                    mask_transfer.layer_rgba[use, :3] = ocr_extra.layer_rgba[use, :3]
                    mask_transfer.layer_rgba[..., 3] = np.maximum(
                        mask_transfer.layer_rgba[..., 3], ocr_extra.layer_rgba[..., 3]
                    )
                    mask_transfer.composite_mask = np.maximum(
                        mask_transfer.composite_mask, ocr_extra.composite_mask
                    )
                    if ocr_extra.clear_mask is not None:
                        if mask_transfer.clear_mask is None:
                            mask_transfer.clear_mask = np.zeros_like(mask_transfer.composite_mask)
                        mask_transfer.clear_mask = np.maximum(mask_transfer.clear_mask, ocr_extra.clear_mask)
                    mask_transfer.matches.extend(ocr_extra.matches)
                    mask_transfer.records.extend(ocr_extra.records)

            # Large vivid-red translated SFX are not speech bubbles and can be
            # missed by both container and OCR geometry. Rebuild only red groups
            # whose registered shapes differ strongly between source and target.
            color_source = (getattr(getattr(paired_diff, "supplemental", None), "aligned_source", None)
                            if paired_diff is not None else None)
            if color_source is None and paired_diff is not None:
                color_source = paired_diff.aligned_source
            if (not direct_container_fast and color_source is not None and paired_diff is not None and paired_diff.method == "photo_pair"
                    and self.config.mask_replace.photo_pair_color_sfx_enabled):
                color_extra = transfer_photo_color_sfx(color_source, mask_transfer.image, self.config.mask_replace)
                if color_extra.records:
                    mask_transfer.image = color_extra.image
                    use = color_extra.layer_rgba[..., 3] > 0
                    mask_transfer.layer_rgba[use, :3] = color_extra.layer_rgba[use, :3]
                    mask_transfer.layer_rgba[..., 3] = np.maximum(mask_transfer.layer_rgba[..., 3], color_extra.layer_rgba[..., 3])
                    mask_transfer.composite_mask = np.maximum(mask_transfer.composite_mask, color_extra.composite_mask)
                    if color_extra.clear_mask is not None:
                        if mask_transfer.clear_mask is None:
                            mask_transfer.clear_mask = np.zeros_like(mask_transfer.composite_mask)
                        mask_transfer.clear_mask = np.maximum(mask_transfer.clear_mask, color_extra.clear_mask)
                    mask_transfer.matches.extend(color_extra.matches)
                    mask_transfer.records.extend(color_extra.records)
            transfer_rgba = mask_transfer.layer_rgba
            applied_source_bubbles = {r.source_bubble_id for r in mask_transfer.records if r.applied}
            applied_target_bubbles = {r.target_bubble_id for r in mask_transfer.records if r.applied}
            source_unit_by_id = {u.id: u for u in source_units}
            target_unit_by_id = {u.id: u for u in target_units}
            mask_source_by_id = {b.id: b for b in mask_source_bubbles}
            mask_target_by_id = {b.id: b for b in mask_target_bubbles}

            def _center_in_boxes(unit, boxes):
                if unit is None:
                    return False
                cx, cy = unit.centroid
                return any(x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in boxes)

            applied_source_boxes = [mask_source_by_id[i].bbox for i in applied_source_bubbles if i in mask_source_by_id]
            applied_target_boxes = [mask_target_by_id[i].bbox for i in applied_target_bubbles if i in mask_target_by_id]

            # Source-integrity rejection is stronger than an OCR/reletter fallback.
            # A camera-edge-clipped photographed bubble may still yield *some* OCR
            # text, but those characters are necessarily only a prefix/suffix of
            # the translation that existed outside the frame. Relettering that
            # partial OCR string into the complete HD target would recreate the
            # exact v0.8.2 failure through a different code path. Keep these
            # regions blocked until a complete source or a manual full translation
            # is supplied.
            integrity_block_reasons = {"source_text_region_clipped_at_page_edge"}
            def _integrity_blocked_record(r):
                return (
                    r.reason in integrity_block_reasons
                    or (bool(getattr(r, "review_required", False)) and getattr(r, "review_reason", "") in integrity_block_reasons)
                )
            integrity_blocked_src = {
                r.source_bubble_id for r in mask_transfer.records if _integrity_blocked_record(r)
            }
            integrity_blocked_dst = {
                r.target_bubble_id for r in mask_transfer.records if _integrity_blocked_record(r)
            }
            integrity_blocked_source_boxes = [
                mask_source_by_id[i].bbox for i in integrity_blocked_src if i in mask_source_by_id
            ]
            integrity_blocked_target_boxes = [
                mask_target_by_id[i].bbox for i in integrity_blocked_dst if i in mask_target_by_id
            ]

            if mode == "hybrid":
                # Paired-diff bubbles use separate ids from OCR bubbles. Exclude OCR
                # matches geometrically covered by an already-applied transfer, or
                # hybrid mode would render the same translation a second time.
                fallback_matches = [
                    m for m in accepted
                    if source_unit_by_id.get(m.source_unit_id) is not None
                    and target_unit_by_id.get(m.target_unit_id) is not None
                    and not _center_in_boxes(source_unit_by_id[m.source_unit_id], applied_source_boxes)
                    and not _center_in_boxes(target_unit_by_id[m.target_unit_id], applied_target_boxes)
                ]
            elif (mode == "mask_replace" and paired_diff is not None
                    and paired_diff.method == "photo_pair"
                    and self.config.mask_replace.photo_pair_fallback_reletter_missing):
                # A photographed pair is deliberately conservative: direct mask
                # transfer handles only geometrically safe containers. Use OCR
                # re-lettering for every accepted text match not already covered,
                # including missed/open/clipped burst bubbles and blurry patches.
                fallback_matches = [
                    m for m in accepted
                    if source_unit_by_id.get(m.source_unit_id) is not None
                    and target_unit_by_id.get(m.target_unit_id) is not None
                    and not _center_in_boxes(source_unit_by_id[m.source_unit_id], applied_source_boxes)
                    and not _center_in_boxes(target_unit_by_id[m.target_unit_id], applied_target_boxes)
                ]
            elif mode == "mask_replace" and self.config.mask_replace.fallback_reletter_on_blur:
                blur_rejected_src = {
                    r.source_bubble_id for r in mask_transfer.records
                    if (not r.applied and r.reason in {
                        "source_text_too_blurry_for_pixel_transfer",
                        "source_text_fidelity_rejected",
                    })
                }
                blur_rejected_dst = {
                    r.target_bubble_id for r in mask_transfer.records
                    if (not r.applied and r.reason in {
                        "source_text_too_blurry_for_pixel_transfer",
                        "source_text_fidelity_rejected",
                    })
                }
                rejected_source_boxes = [mask_source_by_id[i].bbox for i in blur_rejected_src if i in mask_source_by_id]
                rejected_target_boxes = [mask_target_by_id[i].bbox for i in blur_rejected_dst if i in mask_target_by_id]
                fallback_matches = [
                    m for m in accepted
                    if source_unit_by_id.get(m.source_unit_id) is not None
                    and target_unit_by_id.get(m.target_unit_id) is not None
                    and _center_in_boxes(source_unit_by_id[m.source_unit_id], rejected_source_boxes)
                    and _center_in_boxes(target_unit_by_id[m.target_unit_id], rejected_target_boxes)
                ]
            else:
                fallback_matches = []

            # Apply the source-integrity block *after* every fallback branch so
            # hybrid mode, photo-pair OCR fallback and future fallback routes all
            # inherit the same publication guarantee.
            if integrity_blocked_source_boxes or integrity_blocked_target_boxes:
                fallback_matches = [
                    m for m in fallback_matches
                    if source_unit_by_id.get(m.source_unit_id) is not None
                    and target_unit_by_id.get(m.target_unit_id) is not None
                    and not _center_in_boxes(source_unit_by_id[m.source_unit_id], integrity_blocked_source_boxes)
                    and not _center_in_boxes(target_unit_by_id[m.target_unit_id], integrity_blocked_target_boxes)
                ]

            # v0.8.15 strict mode contract. "精准蒙版替换" means exactly that:
            # preserve the translated source glyph pixels/ink, including punctuation
            # and stylized symbols. OCR can be used as evidence for detection/review,
            # but must never rewrite final text in this mode. This intentionally
            # overrides legacy saved configs whose old fallback flags may still be true.
            strict_mask_only = bool(
                (mode == "mask_replace" and getattr(self.config.mask_replace, "strict_mask_replace_no_ocr_reletter", True))
                or (mode == "direct_patch")
                or (mode == "auto" and direct_container_fast)
            )
            if strict_mask_only:
                fallback_matches = []

            # Publication-quality photo pages should prefer HD re-lettering when OCR
            # evidence is available. Mask transfer still establishes the region and
            # safely covers OCR-missing areas, but accepted OCR dialogue matches are
            # re-rendered sharply instead of publishing photographed glyph pixels.
            if (not strict_mask_only and mode == "mask_replace" and paired_diff is not None
                    and paired_diff.method == "photo_pair"
                    and self.config.mask_replace.photo_pair_prefer_reletter_with_ocr
                    and source_blocks and target_blocks and applied_target_boxes):
                min_ocr = float(self.config.mask_replace.photo_pair_prefer_reletter_min_confidence)
                min_match = float(min(self.config.matching.review_confidence, max(0.0, min_ocr - 0.08)))
                preferred = []
                transfer_record_by_target = {
                    str(r.target_bubble_id): r for r in (mask_transfer.records if mask_transfer is not None else [])
                }
                source_block_by_id = {b.id: b for b in source_blocks}
                target_bubble_by_id = {b.id: b for b in target_bubbles}
                for m in accepted:
                    su = source_unit_by_id.get(m.source_unit_id)
                    tu = target_unit_by_id.get(m.target_unit_id)
                    if su is None or tu is None:
                        continue
                    if not str(su.text).strip():
                        continue
                    if su.confidence < min_ocr or tu.confidence < min_ocr or m.confidence < min_match:
                        continue
                    if _center_in_boxes(su, integrity_blocked_source_boxes) or _center_in_boxes(tu, integrity_blocked_target_boxes):
                        continue
                    if not _center_in_boxes(tu, applied_target_boxes):
                        continue
                    # v0.8.11: when a clean translated scan already produced a
                    # sharp, geometry-preserving glyph transfer, OCR should remain
                    # evidence/review only. Re-typesetting a short transcript from
                    # scratch is what created oversized words and lost source columns
                    # on real Macs. Only low-quality/blurred transfers are promoted
                    # to OCR re-lettering automatically.
                    rec = transfer_record_by_target.get(str(tu.bubble_id or ""))
                    if _should_preserve_transferred_layout(rec, self.config.mask_replace):
                        profiles = [
                            source_block_by_id[bid].meta.get("source_layout_profile")
                            for bid in su.block_ids if bid in source_block_by_id
                            and source_block_by_id[bid].meta.get("source_layout_profile")
                        ]
                        profile = profiles[0] if profiles else {}
                        safe = None
                        if tu.bubble_id and tu.bubble_id in target_bubble_by_id:
                            safe = target_bubble_by_id[tu.bubble_id].safe_mask
                        if safe is None or cv2.countNonZero(safe) == 0:
                            safe = polygon_safe_mask(tu, target.shape[:2], margin=max(2, self.config.bubbles.safe_margin_px // 2))
                        transferred_profile = _masked_layout_profile(mask_transfer.image, safe, su.text, _reletter_orientation("auto", su, source_block_by_id)) if mask_transfer is not None else {}
                        if profile and _preserved_layout_looks_complete(profile, transferred_profile):
                            continue
                    preferred.append(m)
                if preferred:
                    seen = {(m.source_unit_id, m.target_unit_id) for m in fallback_matches}
                    for m in preferred:
                        key = (m.source_unit_id, m.target_unit_id)
                        if key not in seen:
                            fallback_matches.append(m)
                            seen.add(key)

        if mask_transfer is not None:
            triage_cfg = self.config.direct_patch if direct_container_fast else self.config.mask_replace
            finalize_transfer_records(mask_transfer.records, triage_cfg)
            # v0.8.34.4: Auto must preserve a usable pixel/mask result. OCR may
            # still provide evidence, but SAFE/REVIEW regions are not re-typeset
            # over the source raster merely because a text match exists. Only
            # REJECT or genuinely uncovered regions continue to heavy fallback.
            if (mode == "auto"
                    and bool(getattr(self.config.mask_replace, "auto_preserve_safe_and_review_pixel_results", True))
                    and fallback_matches):
                protected_target_bubbles = {
                    str(getattr(r, "target_bubble_id", ""))
                    for r in mask_transfer.records
                    if bool(getattr(r, "applied", False))
                    and str(getattr(r, "triage_state", "")) in {"SAFE", "REVIEW"}
                }
                if protected_target_bubbles:
                    kept_fallback = []
                    for m in fallback_matches:
                        tu = target_unit_by_id.get(m.target_unit_id)
                        bubble_id = str(getattr(tu, "bubble_id", "") or "") if tu is not None else ""
                        if bubble_id and bubble_id in protected_target_bubbles:
                            continue
                        kept_fallback.append(m)
                    fallback_matches = kept_fallback

        if mode in {"auto", "direct_patch", "mask_replace"} and not fallback_matches:
            assert mask_transfer is not None
            rendered = mask_transfer.image.copy()
            operation_mask = mask_transfer.composite_mask
            mask_result = MaskBuildResult(
                mask=operation_mask,
                per_unit={},
                clipped_pixels=0,
                source_pixels=int(cv2.countNonZero(operation_mask)),
            )
            # No Japanese inpainting is needed: the aligned Chinese bubble interior
            # itself covers the Japanese text. Keep the untouched HD page as base layer.
            inpaint_method = "direct-patch-no-inpaint" if direct_container_fast else "mask-replace-no-inpaint"
            inpaint_result = InpaintResult(target.copy(), inpaint_method, {"applied": mask_transfer.applied_count})
            lettering = []
            lettering_masks: list[np.ndarray] = []
            if direct_container_fast:
                qa = run_direct_patch_qa(
                    pair, registration, source_units, mask_source_bubbles, mask_transfer.records,
                    self.config.qa, self.config.direct_patch,
                )
            else:
                qa = run_mask_replace_qa(
                    pair, registration, source_units, mask_source_bubbles, mask_transfer.records,
                    self.config.qa, self.config.mask_replace,
                )
            if mode == "direct_patch" and direct_container_plan is not None and not direct_container_plan.safe_to_skip_other_paths:
                qa.append(QAItem(
                    "direct_patch_partial_review", "warning",
                    "Direct Patch applied only regions that passed the whole-raster safety gate, but one or more container-like regions were skipped for review.",
                    meta={"diagnostics": dict(direct_container_plan.diagnostics)},
                ))
        else:
            base = mask_transfer.image.copy() if mask_transfer is not None else target
            mask_result = build_clear_mask(
                target.shape[:2],
                target_blocks,
                target_units,
                target_bubbles,
                fallback_matches,
                self.config.masking,
                min_match_confidence=self.config.matching.review_confidence,
                allow_relations={"one_to_one"},
                target_image=target, current_image=base,
            )
            inpaint_result = inpaint_image(base, mask_result.mask, self.config.inpainting)
            rendered = inpaint_result.image.copy()

            source_by_id = {u.id: u for u in source_units}
            target_by_id = {u.id: u for u in target_units}
            bubbles_by_id = {b.id: b for b in target_bubbles}
            lettering = []
            lettering_masks = []
            for match in fallback_matches:
                src = source_by_id[match.source_unit_id]
                dst = target_by_id[match.target_unit_id]
                safe = None
                if dst.bubble_id and dst.bubble_id in bubbles_by_id:
                    safe = bubbles_by_id[dst.bubble_id].safe_mask
                if safe is None or cv2.countNonZero(safe) == 0:
                    safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, self.config.bubbles.safe_margin_px // 2))
                lcfg = self.config.lettering.model_copy(deep=True)
                source_block_by_id = {b.id: b for b in source_blocks}
                lcfg.orientation = _reletter_orientation(lcfg.orientation, src, source_block_by_id)
                # Recover typography from the translated source scan. OCR identifies
                # Unicode only; it must not invent a new font scale/column count.
                profiles = [
                    source_block_by_id[bid].meta.get("source_layout_profile")
                    for bid in src.block_ids if bid in source_block_by_id
                    and source_block_by_id[bid].meta.get("source_layout_profile")
                ]
                if profiles:
                    profile = profiles[0]
                    if lcfg.orientation == "vertical" and int(profile.get("columns") or 0) > 0:
                        lcfg.preferred_columns = int(profile["columns"])
                    pitch = float(profile.get("glyph_pitch_px") or 0.0)
                    if pitch > 0:
                        sx0, sy0, sx1, sy1 = src.bbox
                        sbw, sbh = max(1.0, sx1-sx0), max(1.0, sy1-sy0)
                        safe_box = cv2.boundingRect((safe > 0).astype(np.uint8))
                        _x, _y, sw, sh = safe_box
                        scale = min(max(0.25, sw / sbw), max(0.25, sh / sbh))
                        predicted = int(round(pitch * scale))
                        lcfg.preferred_font_size = int(np.clip(predicted, lcfg.min_font_size, lcfg.max_font_size))
                result = fit_text(target.shape[:2], safe, dst, src.text, lcfg)
                lettering.append(result)
                if result.success and result.text_mask is not None:
                    rendered = composite_text(rendered, result, self.config.lettering)
                    lettering_masks.append(result.text_mask)
                elif mask_transfer is not None:
                    # Never turn a valid Chinese transfer into an empty balloon just
                    # because transcript re-layout could not fit. Restore the exact
                    # pre-reletter candidate under the clear mask and keep QA/review
                    # visible. This preserves a usable Chinese page for manual edit.
                    restore_mask = mask_result.per_unit.get(dst.id)
                    if restore_mask is not None and cv2.countNonZero(restore_mask) > 0:
                        use = restore_mask > 0
                        rendered[use] = mask_transfer.image[use]

            if mode == "reletter":
                qa = run_page_qa(
                    pair, registration, source_units, target_units, matches, lettering,
                    mask_result, inpaint_result.image, self.config.qa,
                )
            else:
                assert mask_transfer is not None
                mask_qa = run_mask_replace_qa(
                    pair, registration, source_units, source_bubbles, mask_transfer.records,
                    self.config.qa, self.config.mask_replace,
                )
                # A rejected mask patch is expected to fall back to OCR re-lettering in
                # hybrid mode. Keep it visible, but do not make that fallback itself a
                # publication-blocking error.
                for item in mask_qa:
                    if item.code in {"mask_replace_rejected", "source_bubble_unmatched"}:
                        item.severity = "warning"
                fsids = {m.source_unit_id for m in fallback_matches}
                ftids = {m.target_unit_id for m in fallback_matches}
                fallback_qa = run_page_qa(
                    pair, registration,
                    [u for u in source_units if u.id in fsids],
                    [u for u in target_units if u.id in ftids],
                    fallback_matches, lettering, mask_result, inpaint_result.image, self.config.qa,
                )
                merged = []
                seen = set()
                for item in mask_qa + fallback_qa:
                    key = (item.code, item.unit_id, item.message)
                    if key not in seen:
                        seen.add(key); merged.append(item)
                qa = merged

        if (paired_diff is not None and paired_diff.method == "photo_pair"
                and self.config.mask_replace.photo_pair_require_ocr_evidence
                and not source_blocks and not target_blocks):
            photo_records = list(mask_transfer.records) if mask_transfer is not None else []
            # Judge the conservative photo route on its own records. Structural
            # supplements may legitimately add/reject open-text candidates and
            # should not turn a successfully transferred photographed pair into
            # a publication-blocking "OCR missing" error.
            photo_only = [
                r for r in photo_records
                if getattr(r, "geometry_mode", "") in {"photo_pair", "rigid_uniform_container"}
                or getattr(r, "sr_backend", "") == "rigid-container-raster"
            ]
            content_verified = bool(
                photo_records
                and all(
                    (not bool(getattr(r, "applied", False)))
                    or (str(getattr(r, "content_check", "")).startswith("checked")
                        and bool(getattr(r, "content_complete", False)))
                    for r in photo_records
                )
            )
            # OCR absence is a review warning in the user's preferred policy:
            # once a raster candidate was detected and published, incomplete
            # content must not turn the page back into Japanese.
            fully_applied_photo = bool(
                photo_only
                and all(r.applied for r in photo_only)
                and registration.confidence >= 0.78
            )
            qa.append(QAItem(
                "photo_pair_ocr_evidence_missing",
                "warning" if fully_applied_photo else "error",
                (
                    "All detected photographed-page regions passed the independent raster-content check under strong registration. OCR is unavailable, so this verifies detected regions only; review the page for any entirely undiscovered open/SFX text."
                    if fully_applied_photo else
                    "Photographed-edition extraction is conservative and OCR evidence is unavailable; at least one detected region is not independently content-verified, or page registration is weak. Precise Mask mode will not OCR-rewrite final glyphs."
                ),
                meta={
                    "detected_regions": len(photo_records),
                    "content_verified_detected_regions": content_verified,
                    "verification_scope": "detected_regions_only",
                    "registration_confidence": registration.confidence,
                },
            ))

        page_id = stem_id(pair.target_path)
        pd_records = list(paired_diff.records) if paired_diff is not None else []
        pd_supp = getattr(paired_diff, "supplemental", None) if paired_diff is not None else None
        supp_records = list(getattr(pd_supp, "records", []) or [])
        transfer_records = list(mask_transfer.records) if mask_transfer is not None else []
        clear_pixels = int(cv2.countNonZero(mask_transfer.clear_mask)) if (mask_transfer is not None and mask_transfer.clear_mask is not None) else 0
        write_pixels = int(cv2.countNonZero(mask_transfer.composite_mask)) if mask_transfer is not None else 0
        reason_counts: dict[str, int] = {}
        for rec in transfer_records:
            reason = str(getattr(rec, "reason", "") or "unknown")
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        kind_counts = {"bubble": 0, "free_text": 0, "complex_text": 0}
        for rec in pd_records + supp_records:
            kind = str(getattr(rec, "region_kind", "bubble") or "bubble")
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
        transfer_audit = {
            "schema": "manga_hd_translation_transfer.transfer_audit.v2",
            "page_id": page_id,
            "registration": {
                "method": registration.method,
                "confidence": float(registration.confidence),
                "accepted_for_structural": bool(registration.confidence >= float(getattr(self.config.mask_replace, "paired_diff_structural_min_registration_confidence", 0.62))),
                "route": registration.diagnostics.get("route", registration.method),
            },
            "page_pairing_check": pair_check.to_dict(),
            "planner": decision.to_dict(),
            "candidate_detection": {
                "direct_patch_used": bool(direct_container_fast),
                "source_direct_container_used": bool(direct_container_fast),
                "mask_route_used": bool(not direct_container_fast and mode in {"auto", "mask_replace", "hybrid"}),
                "source_direct_container_diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                "paired_diff_used": bool(paired_diff is not None),
                "paired_diff_method": paired_diff.method if paired_diff is not None else None,
                "primary_regions": len(pd_records),
                "supplemental_regions": len(supp_records),
                "regions_by_kind": kind_counts,
                "unseeded_white_container_pairs": int(unseeded_white_pair_count),
                "raw_diagnostics": dict(paired_diff.diagnostics) if paired_diff is not None else {},
            },
            "ocr_evidence": {
                "source_route": str(cache_stats.get("ocr_source", "")),
                "target_route": str(cache_stats.get("ocr_target", "")),
                "source_blocks": len(source_blocks),
                "target_blocks": len(target_blocks),
                "source_nonempty_blocks": len([b for b in source_blocks if str(getattr(b, "text", "")).strip()]),
                "source_units": len(source_units),
                "target_units": len(target_units),
                "unit_matches": len(matches),
                "accepted_unit_matches": len(accepted),
                "ambiguous_source_units": list(match_result.ambiguous_source),
            },
            "transfer": {
                "mode": mode,
                "records": len(transfer_records),
                # Geometry write success and semantic/raster content success are
                # deliberately separate.  Do not equate ``applied`` with a
                # publication-complete translation.
                "applied": sum(1 for r in transfer_records if bool(getattr(r, "applied", False))),
                "geometry_applied": sum(1 for r in transfer_records if bool(getattr(r, "applied", False))),
                "rejected": sum(1 for r in transfer_records if not bool(getattr(r, "applied", False))),
                "content_checked": sum(1 for r in transfer_records if str(getattr(r, "content_check", "")).startswith("checked")),
                "content_complete": sum(1 for r in transfer_records if bool(getattr(r, "content_complete", False))),
                "content_incomplete": sum(1 for r in transfer_records if str(getattr(r, "content_check", "")).startswith("checked") and not bool(getattr(r, "content_complete", False))),
                "content_unverified": sum(1 for r in transfer_records if bool(getattr(r, "applied", False)) and not str(getattr(r, "content_check", "")).startswith("checked")),
                "auto_repair_attempted": sum(1 for r in transfer_records if bool(getattr(r, "repair_attempted", False))),
                "auto_repair_succeeded": sum(1 for r in transfer_records if bool(getattr(r, "repair_succeeded", False))),
                "triage_safe": sum(1 for r in transfer_records if str(getattr(r, "triage_state", "")) == "SAFE"),
                "triage_review": sum(1 for r in transfer_records if str(getattr(r, "triage_state", "")) == "REVIEW"),
                "triage_reject": sum(1 for r in transfer_records if str(getattr(r, "triage_state", "")) == "REJECT"),
                "min_source_ink_coverage": min([float(getattr(r, "source_ink_coverage", 0.0)) for r in transfer_records if str(getattr(r, "content_check", "")).startswith("checked")] or [0.0]),
                "max_target_residual_ratio": max([float(getattr(r, "target_residual_ratio", 0.0)) for r in transfer_records if str(getattr(r, "content_check", "")).startswith("checked")] or [0.0]),
                "verification_scope": "detected_regions_plus_ocr" if (source_blocks and target_blocks) else "detected_regions_only_no_ocr",
                "review_required": sum(1 for r in transfer_records if bool(getattr(r, "review_required", False))),
                "low_confidence_candidates": sum(1 for r in transfer_records if bool(getattr(r, "candidate", False))),
                "ocr_guided_records": sum(1 for r in transfer_records if str(getattr(r, "geometry_mode", "")) == "ocr_guided_components"),
                "complex_text_records": sum(1 for r in transfer_records if str(getattr(r, "geometry_mode", "")) == "complex_text"),
                "clear_pixels": clear_pixels,
                "write_pixels": write_pixels,
                "reason_counts": reason_counts,
            },
            "qa": {"summary": qa_summary(qa), "issue_codes": [x.code for x in qa]},
        }
        active_records = list(mask_transfer.records) if mask_transfer is not None else []
        active_matches = list(mask_transfer.matches) if mask_transfer is not None else []
        direct_manual_effect_candidates = list(((direct_container_plan.diagnostics if direct_container_plan is not None else {}) or {}).get("manual_effect_candidates", []) or [])
        active_review_regions = [
            {
                "source_bubble_id": r.source_bubble_id,
                "target_bubble_id": r.target_bubble_id,
                "source_bbox": list(r.source_bbox),
                "target_bbox": list(r.target_bbox),
                "source_edge_sides": r.source_edge_sides,
                "reason": (getattr(r, "review_reason", "") or r.reason) if bool(getattr(r, "review_required", False)) else "photographed_text_without_ocr_reletter",
                "candidate_applied": bool(getattr(r, "candidate", False)),
                "clarity_mode": getattr(r, "clarity_mode", "pixels"),
                "restorable": True,
                "editable": True,
                "review_level": "required" if bool(getattr(r, "review_required", False)) else "recommended",
            }
            for r in active_records
            if (
                bool(getattr(r, "review_required", False))
                or (not direct_container_fast and not source_blocks and bool(getattr(r, "applied", False))
                    and str(getattr(r, "clarity_mode", "")).startswith("photo-"))
            )
        ]
        mask_manual_reletter = [
            {
                "source_bubble_id": r.source_bubble_id,
                "target_bubble_id": r.target_bubble_id,
                "source_bbox": list(r.source_bbox),
                "target_bbox": list(r.target_bbox),
                "source_edge_sides": r.source_edge_sides,
                "reason": getattr(r, "review_reason", "") or r.reason,
                "candidate_applied": bool(getattr(r, "candidate", False)),
                "clarity_mode": getattr(r, "clarity_mode", "pixels"),
                "restorable": bool(getattr(r, "restorable", False)),
                "editable": bool(getattr(r, "editable", False)),
            }
            for r in active_records
            if (not direct_container_fast and bool(getattr(r, "review_required", False)))
        ]

        project = PageProject(
            page_id=page_id,
            pair=pair,
            registration=registration,
            source_blocks=source_blocks,
            target_blocks=target_blocks,
            source_bubbles=source_bubbles,
            target_bubbles=target_bubbles,
            source_units=source_units,
            target_units=target_units,
            matches=matches,
            lettering=lettering,
            qa=qa,
            meta={
                "page_management": mark.to_dict(),
                "transfer_audit": transfer_audit,
                "auto_applied_match_ids": [f"{m.source_unit_id}->{m.target_unit_id}" for m in fallback_matches],
                "auto_applied_count": len(fallback_matches) + (mask_transfer.applied_count if mask_transfer is not None else 0),
                "reletter_applied_count": len(fallback_matches),
                "inpainting": {"method": inpaint_result.method, **inpaint_result.diagnostics},
                "mask_clipped_ratio": mask_result.clipped_ratio,
                "qa_summary": qa_summary(qa),
                "unmatched_source_units": match_result.unmatched_source,
                "unmatched_target_units": match_result.unmatched_target,
                "ambiguous_source_units": match_result.ambiguous_source,
                "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
                "transfer_mode": mode,
                "transfer_planner": decision.to_dict(),
                "page_pairing_check": pair_check.to_dict(),
                "job_fingerprint": page_job_fingerprint(pair, self.config),
                "cache": cache_stats,
                "runtime": runtime_summary(self.config.runtime.device),
                "registration_route": registration.diagnostics.get("route", registration.method),
                "direct_patch": {
                    "used": bool(direct_container_fast),
                    "requested": bool(mode == "direct_patch"),
                    "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                    "contract": "text_only_target_background",
                    "applied_count": mask_transfer.applied_count if (direct_container_fast and mask_transfer is not None) else 0,
                    "records": [r.to_dict() for r in active_records] if direct_container_fast else [],
                    "bubble_matches": [m.to_dict() for m in active_matches] if direct_container_fast else [],
                    "review_regions": active_review_regions if direct_container_fast else [],
                    # Keep Direct safety-review candidates even when Auto falls
                    # through to Mask.  A page may have four safe white speech
                    # bubbles handled by Mask while purple/pink open-effect text
                    # is rejected by Direct; those rejected colored regions are
                    # still actionable omission-repair candidates for the GUI.
                    "manual_effect_candidates": direct_manual_effect_candidates,
                },
                # Backward-compatible alias for v0.8.33 project readers.
                "source_direct_container": {
                    "used": bool(direct_container_fast),
                    "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                },
                "paired_diff": {
                    "used": bool(use_paired_diff),
                    "method": paired_diff.method if paired_diff is not None else None,
                    "safe_to_skip_ocr": bool(paired_diff.safe_to_skip_ocr) if paired_diff is not None else False,
                    "threshold": float(paired_diff.threshold) if paired_diff is not None else None,
                    "noise_floor": float(paired_diff.noise_floor) if paired_diff is not None else None,
                    "diagnostics": dict(paired_diff.diagnostics) if paired_diff is not None else {},
                    "records": [r.to_dict() for r in paired_diff.records] if paired_diff is not None else [],
                    "supplemental": {
                        "used": bool(getattr(paired_diff, "supplemental", None)),
                        "method": getattr(getattr(paired_diff, "supplemental", None), "method", None),
                        "records": [r.to_dict() for r in getattr(getattr(paired_diff, "supplemental", None), "records", [])],
                        "diagnostics": dict(getattr(getattr(paired_diff, "supplemental", None), "diagnostics", {}) or {}),
                    } if paired_diff is not None else {},
                },
                "replace_translation": {
                    "schema": "manga-hd-transfer/replace_translation/v1",
                    "compatible_with": "manga-translator-ui/replace_translation",
                    "authority_source_path": str(authority_source_path),
                    "selected_source_path": str(source_path_local),
                    "selected_source_kind": str(selected_source_kind),
                    "secondary_source_available": bool(secondary_source_available),
                    "secondary_source_selected": bool(selected_secondary_source),
                    "arbitration": list(dual_source_arbitration),
                    "selected_arbitration_evidence": dict(selected_arbitration_evidence or {}),
                    "source_candidates": [{"path": pair.source_path, "kind": "primary"}] + [dict(x) for x in replace_source_specs],
                    "regions": _replace_translation_regions(
                        source_units, target_units, matches,
                        overlap_threshold=float(getattr(self.config.matching, "replace_translation_overlap_gate", 0.30)),
                    ),
                    "unmatched_source": list(match_result.unmatched_source),
                    "unmatched_target": list(match_result.unmatched_target),
                    "ambiguous_source": list(match_result.ambiguous_source),
                    "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
                    "force_actions": list(getattr(match_result, "diagnostics", {}).get("force_actions", [])),
                },
                "mask_replace": {
                    "used": bool(mask_transfer is not None and not direct_container_fast and mode in {"auto", "mask_replace", "hybrid"}),
                    "strict_no_ocr_reletter": bool(mode == "mask_replace" and getattr(self.config.mask_replace, "strict_mask_replace_no_ocr_reletter", True)),
                    "applied_count": mask_transfer.applied_count if (mask_transfer is not None and not direct_container_fast) else 0,
                    "records": [r.to_dict() for r in active_records] if not direct_container_fast else [],
                    "bubble_matches": [m.to_dict() for m in active_matches] if not direct_container_fast else [],
                    "ocr_reletter_preferred_count": len([m for m in fallback_matches if mode == "mask_replace" and paired_diff is not None and paired_diff.method == "photo_pair"]),
                    "manual_reletter_required": mask_manual_reletter,
                    "review_regions": active_review_regions if not direct_container_fast else [],
                },
            },
        )

        _check_cancel(cancel_cb, "before_export")
        # Artifacts are intentionally explicit and lossless.
        source_original_path = page_root / "source_original.png"
        authority_source_path_artifact = page_root / "source_authority_original.png"
        original_path = page_root / "target_original.png"
        final_local = page_root / "final.png"
        inpainted_path = page_root / "inpainted.png"
        clear_mask_path = page_root / "clear_mask.png"
        target_clear_mask_path = page_root / "target_clear_mask.png"
        text_layer_path = page_root / "text_layer.png"
        transfer_layer_path = page_root / "mask_transfer_layer.png"
        direct_layer_path = page_root / "direct_patch_layer.png"
        chinese_layer_path = page_root / "chinese_transfer_layer.png"
        transfer_mask_path = page_root / "mask_transfer_mask.png"
        direct_region_path = page_root / "direct_patch_regions.png"
        transfer_audit_path = page_root / "transfer_audit.json"
        write_image(source_original_path, source)
        write_image(authority_source_path_artifact, authority_source)
        write_image(original_path, target)
        write_image(final_local, rendered)
        active_review_meta = project.meta.get("direct_patch", {}) if direct_container_fast else project.meta.get("mask_replace", {})
        # Manual-effect candidates originate from Direct safety analysis even
        # when Auto falls through to Mask for the actual white-bubble transfer.
        # Always include them in the review preview/queue so purple/pink open
        # effects do not silently disappear after a successful Mask run.
        direct_review_meta = project.meta.get("direct_patch", {}) if isinstance(project.meta.get("direct_patch", {}), dict) else {}
        review_queue_for_preview = (
            list(active_review_meta.get("review_regions", []) or [])
            + list(direct_review_meta.get("manual_effect_candidates", []) or [])
        )
        review_preview_path = page_root / "review_preview.png"
        write_image(
            review_preview_path,
            _review_candidate_overlay(rendered, review_queue_for_preview) if review_queue_for_preview else rendered,
        )
        exact_clear = (
            mask_transfer.clear_mask
            if mask_transfer is not None and mask_transfer.clear_mask is not None
            else mask_result.mask
        )
        write_image(target_clear_mask_path, exact_clear)
        save_json(transfer_audit_path, transfer_audit)
        if self.config.export.save_inpainted:
            write_image(inpainted_path, inpaint_result.image)
        if self.config.export.save_masks:
            write_image(clear_mask_path, mask_result.mask)
            for unit_id, mask in mask_result.per_unit.items():
                write_image(page_root / "masks" / f"{unit_id}.png", mask)
            for bubble in target_bubbles:
                if bubble.mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}.png", bubble.mask)
                if bubble.safe_mask is not None:
                    write_image(page_root / "bubbles" / f"{bubble.id}_safe.png", bubble.safe_mask)

        text_rgba = make_text_layer_rgba(target.shape[:2], lettering_masks, color=self.config.lettering.fill)
        write_rgba(text_layer_path, text_rgba)
        chinese_rgba = np.zeros_like(text_rgba)
        if mask_transfer is not None:
            if direct_container_fast:
                # Direct has its own artifacts. Do not also write mask_transfer_*
                # aliases: those aliases made two intentionally different
                # algorithms look identical to users and downstream tooling.
                write_rgba(direct_layer_path, transfer_rgba)
                write_image(direct_region_path, mask_transfer.composite_mask)
            else:
                write_rgba(transfer_layer_path, transfer_rgba)
                write_image(transfer_mask_path, mask_transfer.composite_mask)
            chinese_rgba = transfer_rgba.copy()
        text_use = text_rgba[..., 3] > 0
        chinese_rgba[text_use, :3] = text_rgba[text_use, :3]
        chinese_rgba[..., 3] = np.maximum(chinese_rgba[..., 3], text_rgba[..., 3])
        write_rgba(chinese_layer_path, chinese_rgba)
        if self.config.export.layer_bundle:
            ora_path = page_root / "editable.ora"
            export_openraster(ora_path, target, inpaint_result.image, text_rgba, transfer_rgba if mask_transfer is not None else None)
            psd_path = page_root / "editable.psd"
            active_transfer_layer_path = direct_layer_path if direct_container_fast else transfer_layer_path
            psd_ok = export_psd_imagemagick(
                psd_path, original_path, inpainted_path, text_layer_path,
                active_transfer_layer_path if mask_transfer is not None and active_transfer_layer_path.exists() else None,
            ) if inpainted_path.exists() else False
            project.meta["psd_exported"] = psd_ok

        if self.config.export.save_debug:
            write_image(page_root / "debug_registration.png", registration_overlay(source, target, registration))
            write_image(page_root / "debug_structure.png", structure_overlay(target, target_units, target_bubbles))
            write_image(page_root / "debug_matching.png", matching_overlay(target, source_units, target_units, matches, registration))
            write_image(page_root / "debug_clear_mask.png", mask_overlay(target, mask_result.mask))
            if mask_transfer is not None:
                debug_name = "debug_direct_patch.png" if direct_container_fast else "debug_mask_replace.png"
                write_image(page_root / debug_name, mask_overlay(target, mask_transfer.composite_mask))
            if paired_diff is not None:
                write_image(page_root / "debug_paired_diff.png", mask_overlay(target, paired_diff.change_mask))
                if paired_diff.aligned_source is not None and paired_diff.method == "structural_v08":
                    write_image(page_root / "debug_paired_aligned_source.png", paired_diff.aligned_source)

        if mask_transfer is not None and not direct_container_fast and self.config.mask_replace.save_patch_artifacts:
            save_json(
                page_root / "mask_transfer.json",
                {
                    "mode": mode,
                    "applied_count": mask_transfer.applied_count,
                    "matches": [m.to_dict() for m in mask_transfer.matches],
                    "records": [r.to_dict() for r in mask_transfer.records],
                    "manual_reletter_required": [
                        {
                            "source_bubble_id": r.source_bubble_id,
                            "target_bubble_id": r.target_bubble_id,
                            "source_bbox": list(r.source_bbox),
                            "target_bbox": list(r.target_bbox),
                            "source_edge_sides": r.source_edge_sides,
                            "reason": getattr(r, "review_reason", "") or r.reason,
                            "candidate_applied": bool(getattr(r, "candidate", False)),
                            "clarity_mode": getattr(r, "clarity_mode", "pixels"),
                            "restorable": bool(getattr(r, "restorable", False)),
                            "editable": bool(getattr(r, "editable", False)),
                        }
                        for r in mask_transfer.records
                        if bool(getattr(r, "review_required", False))
                    ],
                },
            )
        if direct_container_fast and mask_transfer is not None:
            save_json(
                page_root / "direct_patch.json",
                {
                    "schema": "manga_hd_translation_transfer.direct_patch.v1",
                    "mode": mode,
                    "contract": "text_only_target_background",
                    "planner": decision.to_dict(),
                    "page_pairing_check": pair_check.to_dict(),
                    "diagnostics": dict(direct_container_plan.diagnostics) if direct_container_plan is not None else {},
                    "applied_count": mask_transfer.applied_count,
                    "records": [r.to_dict() for r in mask_transfer.records],
                },
            )
        if final_path is not None:
            write_image(final_path, rendered)
            project.artifacts["book_final"] = str(Path(final_path))
        project.artifacts.update(
            {
                "source_original": str(source_original_path),
                "source_authority_original": str(authority_source_path_artifact),
                "target_original": str(original_path),
                "target_clear_mask": str(target_clear_mask_path),
                "chinese_transfer_layer": str(chinese_layer_path),
                "transfer_audit": str(transfer_audit_path),
                "final": str(final_local),
                "review_preview": str(review_preview_path),
                "inpainted": str(inpainted_path) if inpainted_path.exists() else "",
                "clear_mask": str(clear_mask_path) if clear_mask_path.exists() else "",
                "text_layer": str(text_layer_path),
                "mask_transfer_layer": str(transfer_layer_path) if transfer_layer_path.exists() else "",
                "mask_transfer_mask": str(transfer_mask_path) if transfer_mask_path.exists() else "",
                "mask_transfer_json": str(page_root / "mask_transfer.json") if (page_root / "mask_transfer.json").exists() else "",
                "direct_patch_layer": str(direct_layer_path) if direct_layer_path.exists() else "",
                "direct_patch_regions": str(direct_region_path) if direct_region_path.exists() else "",
                "direct_patch_json": str(page_root / "direct_patch.json") if (page_root / "direct_patch.json").exists() else "",
                "openraster": str(page_root / "editable.ora") if (page_root / "editable.ora").exists() else "",
                "psd": str(page_root / "editable.psd") if (page_root / "editable.psd").exists() else "",
            }
        )
        manual_queue = project.meta.get("mask_replace", {}).get("manual_reletter_required", []) if (not direct_container_fast and isinstance(project.meta.get("mask_replace"), dict)) else []
        if manual_queue:
            template = {
                "status": "needs_manual_reletter",
                "notes": "Fill text for clipped or otherwise incomplete source bubbles, then run review apply.",
                "manual_reletter": [
                    {
                        "target_bubble_id": row.get("target_bubble_id", ""),
                        "target_bbox": list(row.get("target_bbox", [])),
                        "text": "",
                        "orientation": "auto",
                        "reason": row.get("reason", ""),
                        "source_edge_sides": row.get("source_edge_sides", ""),
                    }
                    for row in manual_queue
                ],
            }
            template_path = page_root / "review_overrides.template.json"
            save_json(template_path, template)
            project.artifacts["review_template"] = str(template_path)
        rt_summary = {
            "schema": "manga-hd-transfer/replace_translation/v1",
            "compatible_with": "manga-translator-ui/replace_translation",
            "authority_source_path": str(authority_source_path),
            "selected_source_path": str(source_path_local),
            "selected_source_kind": str(selected_source_kind),
            "secondary_source_available": bool(secondary_source_available),
            "secondary_source_selected": bool(selected_secondary_source),
            "arbitration": list(dual_source_arbitration),
            "selected_arbitration_evidence": dict(selected_arbitration_evidence or {}),
            "source_path": str(pair.source_path),
            "target_path": str(target_path_local),
            "regions": _replace_translation_regions(
                source_units, target_units, matches,
                overlap_threshold=float(getattr(self.config.matching, "replace_translation_overlap_gate", 0.30)),
            ),
            "unmatched_source": list(match_result.unmatched_source),
            "unmatched_target": list(match_result.unmatched_target),
            "ambiguous_source": list(match_result.ambiguous_source),
            "matching_diagnostics": dict(getattr(match_result, "diagnostics", {}) or {}),
            "force_actions": list(getattr(match_result, "diagnostics", {}).get("force_actions", [])),
            "match_stats": {
                "total": len(matches),
                "one_to_one": len([m for m in matches if m.relation == "one_to_one"]),
                "many_to_one": len([m for m in matches if m.relation == "many_to_one"]),
                "one_to_many": len([m for m in matches if m.relation == "one_to_many"]),
            },
        }
        rt_artifacts = _write_replace_translation_bundle(page_root, self.config.replace_translation, source_blocks, target_blocks, matches, rt_summary)
        if rt_artifacts:
            project.artifacts.update({f"replace_translation_{k}": v for k, v in rt_artifacts.items()})
            if isinstance(project.meta.get("replace_translation"), dict):
                project.meta["replace_translation"]["artifacts"] = rt_artifacts
        save_json(page_root / "qa.json", {"summary": qa_summary(qa), "issues": [x.to_dict() for x in qa]})
        save_json(page_root / "project.json", project.to_dict())
        return project

    def run_book(
        self,
        source_dir: str | Path,
        target_dir: str | Path,
        output_dir: str | Path,
        *,
        progress_cb=None,
        cancel_cb=None,
        resume: bool | None = None,
        pairs_override: list[PagePair] | None = None,
        page_marks: dict | None = None,
    ) -> BookProject:
        """Process a full book with resume, per-stage caches and cheap-first routing."""
        output = Path(output_dir)
        pages_root = output / "pages"
        final_root = output / "final"
        pages_root.mkdir(parents=True, exist_ok=True)
        final_root.mkdir(parents=True, exist_ok=True)
        resume = self.config.batch.resume if resume is None else bool(resume)

        if pairs_override is None:
            pairs, unmatched_source, unmatched_target = pair_directories(source_dir, target_dir, self.config.pairing)
        else:
            pairs = list(pairs_override)
            unmatched_source, unmatched_target = [], []
        pages: list[PageProject] = []
        failures: list[dict] = []
        resumed = 0
        started = time.perf_counter()
        cancelled = False

        planned = []
        used_final_names: set[str] = set()
        for idx, pair in enumerate(pairs, start=1):
            stem = Path(pair.target_path).stem
            target_name = stem + ".png"
            # Different source files can legally share a stem (for example
            # page.jpg + page.png). Never let one final overwrite another: page
            # count/order must remain one output per paired target page.
            if target_name.casefold() in used_final_names:
                target_name = f"{stem}__{idx:04d}.png"
                salt = 2
                while target_name.casefold() in used_final_names:
                    target_name = f"{stem}__{idx:04d}_{salt}.png"; salt += 1
            used_final_names.add(target_name.casefold())
            planned.append((idx, pair, pages_root / stem_id(pair.target_path), final_root / target_name))
        resume_hits: dict[int, PageProject] = {}
        if resume and self.config.batch.skip_completed and planned:
            workers = max(1, min(int(self.config.batch.prefetch_workers), 8, len(planned)))
            if workers > 1:
                # File hashing/JSON parsing are independent I/O work. Preflighting
                # them concurrently keeps the main page loop focused on CV/MPS.
                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mhd-resume") as ex:
                    futures = {ex.submit(load_completed_page, page_dir, pair, self.config, final_path): idx for idx,pair,page_dir,final_path in planned}
                    for fut, idx in [(f, i) for f, i in futures.items()]:
                        try:
                            hit = fut.result()
                        except Exception:
                            hit = None
                        if hit is not None:
                            resume_hits[idx] = hit
            else:
                for idx,pair,page_dir,final_path in planned:
                    hit = load_completed_page(page_dir, pair, self.config, final_path)
                    if hit is not None: resume_hits[idx] = hit

        def emit(done: int, total: int, pair: PagePair | None, status: str, cache_hit: bool = False, message: str = ""):
            if progress_cb is None:
                return
            try:
                progress_cb(done, total, pair, status, cache_hit, message)
            except TypeError:
                progress_cb(done, total, status)

        for idx, pair, page_dir, final_path in planned:
            if cancel_cb is not None and cancel_cb():
                cancelled = True
                emit(idx - 1, len(pairs), pair, "cancelled", False, "用户取消")
                break
            logger.info("Processing page %d/%d: %s", idx, len(pairs), Path(pair.target_path).name)
            emit(idx, len(pairs), pair, "running", False, "正在处理")

            requested_mark = resolve_mark(page_marks, pair) if page_marks is not None else None
            if resume and self.config.batch.skip_completed:
                cached = resume_hits.get(idx)
                if cached is not None and requested_mark is not None and requested_mark.origin != "default":
                    cached_pm = (cached.meta or {}).get("page_management")
                    cached_passthrough = bool((cached.meta or {}).get("passthrough"))
                    # Page Manager admission is authoritative over resume. A page
                    # manually changed to cover/skip must not resurrect an older
                    # translated result; a page changed back to content must not
                    # resurrect an older passthrough result. Auto scan decisions get
                    # the same protection. Legacy processed pages without page
                    # metadata remain compatible only with processable marks.
                    if requested_mark.should_process:
                        cached_reason = str((cached.meta or {}).get("passthrough_reason") or "")
                        # Runtime no-op pages are valid content results: the source
                        # was conclusively found to contain no transferable Chinese
                        # speech/narration box. Job fingerprinting already protects
                        # this cache if either image/config changes.
                        if cached_passthrough and cached_reason != "source_no_transferable_text":
                            cached = None
                    else:
                        if not cached_passthrough:
                            cached = None
                        elif requested_mark.origin == "manual":
                            cached_type = PageMark.from_dict(cached_pm).page_type if cached_pm else ""
                            if cached_type != requested_mark.page_type:
                                cached = None
                if cached is not None:
                    resumed += 1
                    pages.append(cached)
                    emit(idx, len(pairs), pair, "completed", True, "断点命中")
                    continue

            try:
                mark = requested_mark
                page = self.process_page(
                    pair, page_dir, final_path,
                    page_mark=mark, cancel_cb=cancel_cb,
                )
                pages.append(page)
                if bool((page.meta or {}).get("passthrough")):
                    pm = (page.meta or {}).get("page_management", {}) or {}
                    reason = str((page.meta or {}).get("passthrough_reason") or "")
                    if reason == "source_no_transferable_text":
                        emit(idx, len(pairs), pair, "skipped", False, "无需替换 · 中文页无气泡/文本框")
                    else:
                        label = str(pm.get("label") or pm.get("page_type") or "跳过")
                        emit(idx, len(pairs), pair, "skipped", False, f"跳过 · {label}")
                else:
                    emit(idx, len(pairs), pair, "completed", False, page.registration.method)
            except PipelineCancelled:
                cancelled = True
                emit(idx - 1, len(pairs), pair, "cancelled", False, "用户停止")
                break
            except Exception as exc:
                row = {"index": idx, "source": pair.source_path, "target": pair.target_path, "error": f"{type(exc).__name__}: {exc}"}
                failures.append(row)
                emit(idx, len(pairs), pair, "failed", False, row["error"])
                if self.config.batch.stop_on_error:
                    raise

            if self.config.runtime.release_cache_every > 0 and idx % self.config.runtime.release_cache_every == 0:
                empty_accelerator_cache(self.config.runtime.device)
            if self.config.batch.save_manifest_every > 0 and idx % self.config.batch.save_manifest_every == 0:
                save_json(output / "batch_manifest.json", {
                    "processed": idx, "total": len(pairs), "resumed": resumed,
                    "failed": failures, "cancelled": cancelled,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                })

        elapsed = time.perf_counter() - started
        route_counts: dict[str, int] = {}
        stage_cache_hits = 0
        skipped_pages = [p for p in pages if bool((p.meta or {}).get("passthrough"))]
        for page in pages:
            route = str((page.meta or {}).get("registration_route") or page.registration.method)
            route_counts[route] = route_counts.get(route, 0) + 1
            stage_cache_hits += sum(1 for v in ((page.meta or {}).get("cache") or {}).values() if v == "hit")
        book = BookProject(
            source_dir=str(source_dir), target_dir=str(target_dir), output_dir=str(output), pages=pages,
            unmatched_source=unmatched_source, unmatched_target=unmatched_target,
            meta={
                "page_count": len(pages), "paired_count": len(pairs), "resumed_count": resumed,
                "skipped_count": len(skipped_pages),
                "skipped_pages": [
                    {
                        "page_id": p.page_id,
                        "target": p.pair.target_path,
                        "page_management": (p.meta or {}).get("page_management", {}),
                    } for p in skipped_pages
                ],
                "failed_count": len(failures), "failures": failures, "cancelled": cancelled,
                "elapsed_seconds": round(elapsed, 3), "runtime": runtime_summary(self.config.runtime.device),
                "registration_routes": route_counts, "stage_cache_hits": stage_cache_hits,
                "qa_errors": sum(1 for p in pages for q in p.qa if q.severity == "error"),
                "qa_warnings": sum(1 for p in pages for q in p.qa if q.severity == "warning"),
            },
        )
        save_json(output / "book_project.json", book.to_dict())
        save_json(output / "batch_manifest.json", {
            "processed": len(pages) + len(failures), "total": len(pairs), "resumed": resumed,
            "failed": failures, "cancelled": cancelled, "elapsed_seconds": round(elapsed, 3),
        })
        save_json(output / "qa_summary.json", {
            "pages": [{"page_id": p.page_id, "summary": qa_summary(p.qa), "project": p.artifacts.get("final", "")} for p in pages],
            "unmatched_source": unmatched_source, "unmatched_target": unmatched_target,
        })
        return book
