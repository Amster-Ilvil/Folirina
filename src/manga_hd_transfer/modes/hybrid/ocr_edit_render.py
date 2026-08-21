from __future__ import annotations

"""Local renderer for manually edited OCR blocks.

Only hybrid (product name: 精准蒙版+OCR) and reletter (OCR重排) may call this
module. It never imports or dispatches Direct / pure Mask / Reveal renderers.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ...io_utils import read_image, write_image, save_json, load_json
from .lettering_ops import fit_text, composite_text
from .manual_effect_ops import clean_manual_target_text
from .text_transfer import clear_broad_neutral_paper_components
from ...models import TextUnit
from .ocr_edit_blocks import is_ocr_edit_mode, load_ocr_blocks, ocr_edit_dir
from ...result_state import ensure_manual_baseline
from ...schema_compat import as_dict


def _rect_mask(shape: tuple[int, int], bbox: list[int], inset: int = 1) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    if len(bbox) != 4:
        return mask
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(w, x0 + inset)); x1 = max(0, min(w, x1 - inset))
    y0 = max(0, min(h, y0 + inset)); y1 = max(0, min(h, y1 - inset))
    if x1 > x0 and y1 > y0:
        mask[y0:y1, x0:x1] = 255
    return mask


def _polygon_clear_mask(shape: tuple[int, int], polygons: list) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), np.uint8)
    for poly in polygons or []:
        pts = np.asarray(poly, np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, w - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, h - 1))
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    if cv2.countNonZero(mask):
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    return mask


def _fallback_text_mask(target: np.ndarray, bbox: list[int]) -> np.ndarray:
    """Conservative dark-component fallback, strictly inside the selected box."""
    h, w = target.shape[:2]
    out = np.zeros((h, w), np.uint8)
    if len(bbox) != 4:
        return out
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0=max(0,x0); y0=max(0,y0); x1=min(w,x1); y1=min(h,y1)
    if x1-x0 < 4 or y1-y0 < 4:
        return out
    crop = target[y0:y1, x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # A local threshold catches glyphs without treating light halftone as text.
    thresh = min(175, int(np.percentile(gray, 35)) + 24)
    cand = (gray < thresh).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    ch, cw = cand.shape
    area_total = max(1, ch*cw)
    keep = np.zeros_like(cand)
    border = max(1, int(round(min(ch,cw)*0.015)))
    for lab in range(1, count):
        x,y,bw,bh,area = [int(v) for v in stats[lab]]
        if area < 2 or area > area_total*0.08:
            continue
        # Bubble borders / panel lines usually touch the manual rectangle edge;
        # fail closed rather than clearing them.
        if x <= border or y <= border or x+bw >= cw-border or y+bh >= ch-border:
            continue
        if bw > cw*0.72 or bh > ch*0.72:
            continue
        keep[labels == lab] = 1
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    out[y0:y1, x0:x1] = keep*255
    return out


def _rebuild_reletter_base_from_target(page_dir: Path, target: np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Reconstruct an old OCR-Reletter auto result without inherited inpaint shadows.

    v2.3.14 could already have ``final.png``/``final_auto.png`` containing grey
    triangles.  The mode-owned editor base must not freeze those pixels forever.
    OCR-Reletter has an explicit TARGET clear mask plus a transparent Chinese
    text layer, so it can be deterministically rebuilt from pristine TARGET.
    """
    clear_path = page_dir / "target_clear_mask.png"
    text_path = page_dir / "text_layer.png"
    if not clear_path.exists() or not text_path.exists():
        return None, {"rebuilt": False, "reason": "missing_target_clear_or_text_layer"}
    clear = cv2.imread(str(clear_path), cv2.IMREAD_GRAYSCALE)
    layer = cv2.imread(str(text_path), cv2.IMREAD_UNCHANGED)
    if clear is None or clear.shape != target.shape[:2] or layer is None or layer.shape[:2] != target.shape[:2] or layer.ndim != 3 or layer.shape[2] != 4:
        return None, {"rebuilt": False, "reason": "invalid_target_clear_or_text_layer"}
    paper_base, handled, _changed, diag = clear_broad_neutral_paper_components(target, target, clear)
    # Non-paper regions still need the already published auto result; refusing to
    # rebuild the whole page is safer than silently dropping coloured cleanup.
    remaining = clear.copy()
    remaining[handled > 0] = 0
    if cv2.countNonZero(remaining) > 0:
        return None, {"rebuilt": False, "reason": "nonpaper_clear_regions_present", **diag}
    alpha = layer[..., 3:4].astype(np.float32) / 255.0
    rgb = layer[..., :3].astype(np.float32)
    out = np.clip(paper_base.astype(np.float32) * (1.0 - alpha) + rgb * alpha, 0, 255).astype(np.uint8)
    return out, {"rebuilt": True, "reason": "pristine_target_plus_text_layer", **diag}


def apply_ocr_edit_blocks(page_dir: str | Path, project: dict[str, Any], cfg) -> Path:
    page_dir = Path(page_dir)
    mode = str(as_dict(project.get("meta")).get("transfer_mode") or cfg.transfer.mode or "").strip().lower()
    if not is_ocr_edit_mode(mode):
        raise ValueError("人工 OCR 文本块只属于 精准蒙版+OCR / OCR重排。")
    blocks = [row for row in load_ocr_blocks(page_dir, mode) if str(row.get("render_text") or row.get("ocr_text") or "").strip()]
    if not blocks:
        base = page_dir / "final_reviewed.png"
        return base if base.exists() else page_dir / "final.png"

    target = read_image(page_dir / "target_original.png")
    scope_dir = ocr_edit_dir(page_dir, mode)
    scope_dir.mkdir(parents=True, exist_ok=True)
    base_path = scope_dir / "base.png"
    base_state_path = scope_dir / "base_state.json"
    base_state = {}
    if base_state_path.exists():
        try:
            base_state = load_json(base_state_path)
        except Exception:
            base_state = {}
    # v2 invalidates the v2.3.14 frozen base because that base may itself contain
    # the grey inpaint shadow. Fresh automatic runs also delete this state.
    if (not base_path.exists()) or str(base_state.get("schema") or "") != "folirina.ocr_edit.base.v2":
        base_img = None
        base_diag: dict[str, Any] = {}
        if mode == "reletter":
            base_img, base_diag = _rebuild_reletter_base_from_target(page_dir, target)
        if base_img is None:
            candidate = ensure_manual_baseline(page_dir, preferred_source=page_dir / "final.png")
            base_img = read_image(candidate)
            base_diag = {"rebuilt": False, "reason": "stable_automatic_base", "source": str(candidate)}
        write_image(base_path, base_img)
        save_json(base_state_path, {
            "schema":"folirina.ocr_edit.base.v2", "scope":scope_dir.name, "mode":mode, **base_diag,
        })
    out = read_image(base_path)
    if out.shape != target.shape:
        raise ValueError("OCR 编辑基线与 TARGET 尺寸不一致。")

    applied: list[dict[str, Any]] = []
    text_masks: list[np.ndarray] = []
    for row in blocks:
        bbox = [int(v) for v in list(row.get("target_bbox") or [])]
        safe = _rect_mask(target.shape[:2], bbox, inset=1)
        if cv2.countNonZero(safe) == 0:
            applied.append({"id":row.get("id"),"success":False,"reason":"empty_bbox"}); continue
        clear = _polygon_clear_mask(target.shape[:2], list(row.get("target_ocr_polygons") or []))
        clear = cv2.bitwise_and(clear, safe)
        if cv2.countNonZero(clear) == 0:
            clear = cv2.bitwise_and(_fallback_text_mask(target, bbox), safe)
        # White OCR boxes must never be sent through interpolating inpaint.
        # First prove broad neutral TARGET paper and clear only the original JP
        # glyph ink on that paper. Any unproven coloured/artwork remainder keeps
        # the existing conservative local-component cleanup path.
        cleaned, paper_handled, paper_changed, paper_diag = clear_broad_neutral_paper_components(
            target, target, clear,
        )
        remaining_clear = clear.copy()
        if cv2.countNonZero(paper_handled) > 0:
            remaining_clear[paper_handled > 0] = 0
        fallback_diag = {"mode":"none","pixels":0}
        if cv2.countNonZero(remaining_clear) > 0:
            fallback_cleaned, fallback_diag = clean_manual_target_text(target, remaining_clear, bbox=bbox)
            ruse = remaining_clear > 0
            cleaned[ruse] = fallback_cleaned[ruse]
        clean_diag = {
            "ocr_paper_first": True,
            **paper_diag,
            "paper_changed_pixels": int(cv2.countNonZero(paper_changed)),
            "remaining_fallback_pixels": int(cv2.countNonZero(remaining_clear)),
            "fallback": fallback_diag,
        }

        # Reset the selected region to pristine TARGET before drawing. This
        # removes previous automatic Chinese glyphs while preserving original art.
        use = safe > 0
        out[use] = target[use]
        cuse = clear > 0
        out[cuse] = cleaned[cuse]

        text = str(row.get("render_text") or row.get("ocr_text") or "").strip()
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
        lb = str(row.get("line_break_mode") or "smart")
        if lb in {"smart","balanced","source"}: lcfg.line_break_mode = lb
        lm = str(row.get("layout_mode") or "smart_scaling")
        if lm in {"strict","smart_scaling","balloon_fill"}: lcfg.layout_mode = lm
        spacing = row.get("line_spacing_ratio")
        if spacing is not None:
            try: lcfg.line_spacing_ratio = float(np.clip(float(spacing),0.0,0.6))
            except Exception: pass
        letter_spacing = row.get("letter_spacing_ratio")
        if letter_spacing is not None and hasattr(lcfg, "letter_spacing_ratio"):
            try: lcfg.letter_spacing_ratio = float(np.clip(float(letter_spacing),-0.2,0.5))
            except Exception: pass

        x0,y0,x1,y1 = bbox
        poly=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        unit=TextUnit(
            id=str(row.get("id") or "manual-ocr"), polygon=poly, block_ids=[], text=text,
            confidence=float(row.get("confidence") or 1.0), kind="speech", reading_order=0,
            bubble_id=None, meta={"manual_ocr_block":True,"box_locked":True},
        )
        lr=fit_text(target.shape[:2], safe, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            applied.append({"id":row.get("id"),"success":False,"reason":str(lr.reason or "layout_failed"),"clear":clean_diag}); continue
        out=composite_text(out,lr,lcfg)
        text_masks.append(lr.text_mask)
        applied.append({
            "id":row.get("id"),"success":True,"bbox":bbox,"text":text,
            "orientation":lr.orientation,"font_path":lr.font_path,"font_size":int(lr.font_size),
            "lines":list(lr.lines),"clear":clean_diag,
        })

    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, out)
    # Also mirror mode-owned artifacts for diagnostics without touching Direct/Mask files.
    write_image(scope_dir / "final.png", out)
    save_json(scope_dir / "render_state.json", {
        "schema":"folirina.ocr_edit.render_state.v1","scope":scope_dir.name,"mode":mode,
        "block_count":len(blocks),"applied":applied,
        "base":str(base_path),"final":str(final_path),
    })
    return final_path


def reset_ocr_edit_blocks(page_dir: str | Path, project: dict[str, Any], cfg) -> Path:
    page_dir=Path(page_dir)
    mode=str(as_dict(project.get("meta")).get("transfer_mode") or cfg.transfer.mode or "").strip().lower()
    if not is_ocr_edit_mode(mode):
        raise ValueError("当前模式没有 OCR 编辑层。")
    scope=ocr_edit_dir(page_dir,mode)
    base=scope/"base.png"
    if base.exists():
        img=read_image(base); final=page_dir/"final_reviewed.png"; write_image(final,img); return final
    return page_dir/"final.png"


__all__=["apply_ocr_edit_blocks","reset_ocr_edit_blocks"]
