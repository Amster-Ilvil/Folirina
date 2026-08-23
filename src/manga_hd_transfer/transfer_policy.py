from __future__ import annotations

"""Transfer metadata/review policies extracted from the page orchestrator.

These helpers summarize matching, export replace-translation evidence and build
review-only overlays. They do not execute OCR, registration or raster transfer.
"""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .io_utils import save_json
from .lettering import find_default_font
from .models import BubbleInstance, TextBlock, UnitMatch

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
        paired_identity = bool(
            match is not None
            and any(str(reason or "").startswith("route=paired_id_binding") for reason in (getattr(match, "reasons", None) or []))
        )
        rows.append({
            "source_unit_id": source.id,
            "target_unit_id": target.id if target is not None else None,
            "translated_text": str(getattr(source, "text", "") or ""),
            "source_bbox": [float(x) for x in source.bbox],
            "target_bbox": [float(x) for x in target.bbox] if target is not None else None,
            "overlap": float(overlap),
            # Deterministic paired-ID binding is already an identity proof; it
            # must not be mislabeled unmatched merely because the synthetic
            # target-driven OCR region reports geometric overlap=0.
            "matched": bool(match is not None and target is not None and (paired_identity or overlap >= float(overlap_threshold))),
            "match_route": "paired_id_binding" if paired_identity else "geometric",
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


def _review_candidate_overlay(image: np.ndarray, queue: list[dict], *, effective_mask: np.ndarray | None = None) -> np.ndarray:
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
        x0=max(0,min(image.shape[1]-1,x0)); y0=max(0,min(image.shape[0]-1,y0)); x1=max(x0+1,min(image.shape[1],x1)); y1=max(y0+1,min(image.shape[0],y1))
        # Review records often store a whole bubble/container bbox although the
        # actual automatic clear mask is only a few glyphs.  For the visible
        # magenta annotation use the real effective mask whenever it overlaps
        # this candidate, so the editor does not suggest an unsafe giant region.
        if effective_mask is not None and getattr(effective_mask, "shape", None) == image.shape[:2]:
            local = effective_mask[y0:y1, x0:x1] > 0
            ys, xs = np.where(local)
            if xs.size >= 4:
                tx0, tx1 = x0 + int(xs.min()), x0 + int(xs.max() + 1)
                ty0, ty1 = y0 + int(ys.min()), y0 + int(ys.max() + 1)
                original_area = max(1, (x1-x0)*(y1-y0))
                tight_area = max(1, (tx1-tx0)*(ty1-ty0))
                if tight_area <= original_area * 0.88:
                    x0,y0,x1,y1 = tx0,ty0,tx1,ty1
        pad = max(3, min(10, int(round(max(x1-x0,y1-y0)*0.06))))
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


__all__ = ['_reason_metric', '_replace_translation_regions', '_write_replace_translation_bundle', '_blocking_direct_invariant_issues', '_has_transferable_source_text', '_should_preserve_transferred_layout', '_review_candidate_overlay']
