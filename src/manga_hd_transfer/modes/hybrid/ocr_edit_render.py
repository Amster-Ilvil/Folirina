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
from .ocr_layout_region import recover_layout_region, infer_text_orientation
from .manual_effect_ops import clean_manual_target_text
from .text_transfer import clear_broad_neutral_paper_components
from ...models import TextUnit
from .ocr_edit_blocks import is_ocr_edit_mode, load_ocr_blocks, ocr_edit_dir
from ...result_state import ensure_manual_baseline
from ...schema_compat import as_dict
from ...ocr_block_conflicts import resolve_render_container_conflicts


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
        # The OCR rectangle is only a locator. A glyph may legitimately be
        # clipped by that locator edge (common for the last Japanese character
        # in a vertical balloon), so "touches selection edge" alone is not a
        # reason to discard it. Reject only edge-touching components that look
        # like long/thin structural geometry. This keeps small clipped glyphs
        # while still excluding balloon/panel outlines.
        touches_edge = bool(x <= border or y <= border or x+bw >= cw-border or y+bh >= ch-border)
        fill = float(area / max(1, bw * bh))
        aspect = float(max(bw, bh) / max(1.0, min(bw, bh)))
        if touches_edge and (
            bw >= cw * 0.12 or bh >= ch * 0.12 or aspect >= 4.5 or fill <= 0.22
        ):
            continue
        if bw > cw*0.72 or bh > ch*0.72:
            continue
        keep[labels == lab] = 1
    keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)
    out[y0:y1, x0:x1] = keep*255
    return out



def _polygon_is_region_locator(polygons: list, bbox: list[int]) -> bool:
    """Return True when OCR returned the search rectangle, not glyph geometry.

    Some region OCR backends (notably Apple's region API) report the requested
    crop rectangle as the only polygon.  Treating that polygon as an erase mask
    makes the UI locator rectangle physically erase balloon borders.  A locator
    is metadata only; actual erase pixels must be re-derived from image ink.
    """
    if len(bbox) != 4 or not polygons:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox]
    bw, bh = max(1.0, x1-x0), max(1.0, y1-y0)
    box_area = bw * bh
    valid = 0
    for poly in polygons:
        pts = np.asarray(poly, np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        valid += 1
        px0, py0 = np.min(pts, axis=0); px1, py1 = np.max(pts, axis=0)
        pw, ph = max(0.0, float(px1-px0)), max(0.0, float(py1-py0))
        area = abs(float(cv2.contourArea(pts.astype(np.float32))))
        if pw >= bw * 0.82 and ph >= bh * 0.82 and area >= box_area * 0.52:
            return True
    return valid == 1 and len(polygons) == 1 and False


def _existing_text_delta_mask(base: np.ndarray, target: np.ndarray, selection: np.ndarray, guard: np.ndarray) -> np.ndarray:
    """Find text-like pixels already rendered in the selected review region.

    This deliberately does *not* use the selection rectangle as a paint mask.
    It only removes dark, text-sized pixels that differ from pristine TARGET,
    while rejecting panel/bubble lines and broad background patches.
    """
    h, w = target.shape[:2]
    out = np.zeros((h, w), np.uint8)
    if base.shape != target.shape:
        return out
    allowed = (selection > 0) & (guard > 0)
    if not np.any(allowed):
        return out
    diff = np.max(np.abs(base.astype(np.int16) - target.astype(np.int16)), axis=2)
    base_gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cand = ((diff >= 14) & (np.minimum(base_gray, target_gray) <= 232) & allowed).astype(np.uint8)
    if not np.any(cand):
        return out
    count, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 8)
    ys, xs = np.where(selection > 0)
    if len(xs) == 0:
        return out
    sw, sh = max(1, int(xs.max()-xs.min()+1)), max(1, int(ys.max()-ys.min()+1))
    sel_area = max(1, int(np.count_nonzero(selection)))
    keep = np.zeros_like(cand)
    for lab in range(1, count):
        x, y, cw, ch, area = [int(v) for v in stats[lab]]
        if area < 1 or area > sel_area * 0.18:
            continue
        # Long straight structures are much more likely balloon/panel borders.
        if cw >= sw * 0.78 and ch <= max(3, sh * 0.12):
            continue
        if ch >= sh * 0.78 and cw <= max(3, sw * 0.12):
            continue
        keep[labels == lab] = 255
    if cv2.countNonZero(keep):
        keep = cv2.dilate(keep, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)), iterations=1)
        keep = cv2.bitwise_and(keep, selection)
        keep = cv2.bitwise_and(keep, guard)
    return keep


def _mode_text_layer_mask(page_dir: Path, mode: str, shape: tuple[int, int]) -> np.ndarray:
    """Return this mode's automatic OCR text alpha.

    The editor module is mode-private; it must never probe a sibling mode's
    artifacts merely because a caller passed a different mode string.
    """
    names: tuple[str, ...] = ("hybrid_text_layer.png", "text_layer.png") if str(mode or "").strip().lower() == "hybrid" else ()
    for name in names:
        layer = cv2.imread(str(page_dir / name), cv2.IMREAD_UNCHANGED)
        if layer is None or layer.shape[:2] != shape:
            continue
        if layer.ndim == 3 and layer.shape[2] == 4:
            return np.where(layer[...,3] > 0, 255, 0).astype(np.uint8)
    return np.zeros(shape, np.uint8)

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


def apply_ocr_edit_blocks(
    page_dir: str | Path, project: dict[str, Any], cfg, *,
    blocks_override: list[dict[str, Any]] | None = None, preview: bool = False,
):
    page_dir = Path(page_dir)
    mode = str(as_dict(project.get("meta")).get("transfer_mode") or cfg.transfer.mode or "").strip().lower()
    if not is_ocr_edit_mode(mode):
        raise ValueError("人工 OCR 文本块只属于 精准蒙版+OCR / OCR重排。")
    source_rows = blocks_override if blocks_override is not None else load_ocr_blocks(page_dir, mode)
    blocks = [dict(row) for row in source_rows if str(row.get("render_text") or row.get("ocr_text") or "").strip()]
    if not blocks:
        base = page_dir / "final_reviewed.png"
        if preview:
            if not base.exists(): base = page_dir / "final.png"
            image = read_image(base) if base.exists() else read_image(page_dir / "target_original.png")
            return {"image":image,"state":{"block_count":0,"applied":[],"preview":True}}
        return base if base.exists() else page_dir / "final.png"

    target = read_image(page_dir / "target_original.png")
    # Storage already collapses near-identical legacy ROIs.  This renderer-side
    # guard is intentionally independent: two smaller locators can still recover
    # to the same closed balloon/textbox. Only one OCR block may own that whole
    # layout container, otherwise full text layouts would stack on each other.
    blocks, suppressed_render_conflicts = resolve_render_container_conflicts(
        blocks, target, recover_layout_region
    )
    scope_dir = ocr_edit_dir(page_dir, mode)
    if not preview:
        scope_dir.mkdir(parents=True, exist_ok=True)
    base_path = scope_dir / "base.png"
    base_state_path = scope_dir / "base_state.json"
    base_state = {}
    if base_state_path.exists():
        try:
            base_state = load_json(base_state_path)
        except Exception:
            base_state = {}
    # Live preview is strictly read-only. Prefer the exact frozen no-OCR base;
    # on the first unsaved block, the current visible result is necessarily the
    # pre-OCR page and is safe to preview from without creating artifacts.
    if preview:
        if base_path.exists() and str(base_state.get("schema") or "") == "folirina.ocr_edit.base.v2":
            out = read_image(base_path)
        else:
            candidate = page_dir / "final_reviewed.png"
            if not candidate.exists(): candidate = page_dir / "final.png"
            if not candidate.exists(): candidate = page_dir / "target_original.png"
            out = read_image(candidate)
    else:
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
        # The OCR selection tells us which original text to replace; it is not
        # automatically the translated-text container. Recover a closed TARGET
        # balloon/caption around it. Open lettering keeps the selection itself.
        selection = _rect_mask(target.shape[:2], bbox, inset=1)
        if cv2.countNonZero(selection) == 0:
            applied.append({"id":row.get("id"),"success":False,"reason":"empty_bbox"}); continue
        recovered_safe, layout_diag = recover_layout_region(target, bbox, open_inset=2)
        if cv2.countNonZero(recovered_safe) == 0:
            recovered_safe = selection.copy()
            layout_diag = {"layout_kind":"open","layout_source":"selection_fallback","safe_pixels":int(cv2.countNonZero(recovered_safe))}
        polygons = list(row.get("target_ocr_polygons") or [])
        coarse_locator = _polygon_is_region_locator(polygons, bbox)
        # IMPORTANT: layout safety and TARGET-clear safety are different.
        # ``recovered_safe`` is deliberately eroded so newly rendered Chinese
        # never touches a balloon border. Original Japanese, however, may sit
        # much closer to that border. A coarse OCR rectangle therefore falls
        # back to compact *text components inside the locator*, not to the
        # eroded layout mask. The component filter rejects long/thin balloon and
        # panel geometry, so border protection does not require sacrificing
        # border-adjacent Japanese glyphs.
        text_component_clear = bool(coarse_locator)
        clear = np.zeros(target.shape[:2], np.uint8) if coarse_locator else _polygon_clear_mask(target.shape[:2], polygons)
        clear = cv2.bitwise_and(clear, selection)
        if cv2.countNonZero(clear) == 0:
            clear = cv2.bitwise_and(_fallback_text_mask(target, bbox), selection)
            text_component_clear = True
        closed_layout = str(layout_diag.get("layout_kind") or "") in {"bubble", "textbox"}
        if closed_layout and not text_component_clear:
            # Fine OCR polygons are trusted only inside the eroded container.
            # Coarse/derived masks already contain text components only and may
            # reach the border-adjacent interior band where old JP glyphs live.
            clear = cv2.bitwise_and(clear, recovered_safe)
        else:
            clear = cv2.bitwise_and(clear, selection)
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

        # The OCR rectangle is a locator, never a drawable/erasable object.
        # Remove only previously rendered text-like pixels inside the structural
        # guard, then remove TARGET Japanese glyphs.  No rectangular background
        # reset is permitted, so balloon outlines and artwork remain pixel-owned
        # by the pre-OCR base.
        prior_text = _mode_text_layer_mask(page_dir, mode, target.shape[:2])
        # Previous OCR text can itself be an old overflow bug, so remove it by
        # text authority inside the locator rather than by the new layout-safe
        # mask. This never resets the rectangle background.
        prior_guard = selection
        if cv2.countNonZero(prior_text):
            prior_text = cv2.bitwise_and(prior_text, selection)
        else:
            prior_text = _existing_text_delta_mask(out, target, selection, prior_guard)
        ruse = prior_text > 0
        out[ruse] = target[ruse]
        cuse = clear > 0
        out[cuse] = cleaned[cuse]
        clean_diag["coarse_locator_polygon_ignored"] = bool(coarse_locator)
        clean_diag["layout_guard"] = "recovered_safe" if closed_layout else "selection"
        clean_diag["target_clear_guard"] = (
            "component_filtered_selection" if text_component_clear
            else ("recovered_safe" if closed_layout else "selection")
        )
        # Backward-compatible diagnostic name; it now describes the renderer
        # layout authority, not the TARGET glyph-clear authority.
        clean_diag["structural_guard"] = "recovered_safe" if closed_layout else "selection"
        clean_diag["prior_text_reset_pixels"] = int(cv2.countNonZero(prior_text))

        text = str(row.get("render_text") or row.get("ocr_text") or "").strip()
        lcfg = cfg.lettering.model_copy(deep=True)
        requested_orientation = str(row.get("orientation") or "auto")
        lcfg.orientation = requested_orientation
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
        # Manual OCR selection is the original text/search rectangle. Strict mode
        # deliberately keeps it as the hard layout fence; smart/balloon modes may
        # use a proven enclosing bubble/textbox but never an unproven larger area.
        safe = selection.copy() if lm == "strict" else recovered_safe.copy()
        if cv2.countNonZero(safe) == 0:
            safe = selection.copy()
        orientation_diag = {"source":"user", "confidence":1.0}
        if requested_orientation == "auto" and str(layout_diag.get("layout_kind") or "") in {"bubble", "textbox"}:
            inferred, orientation_diag = infer_text_orientation(target, bbox)
            lcfg.orientation = inferred
        elif requested_orientation == "auto":
            orientation_diag = {"source":"layout_aspect", "confidence":0.0}
        spacing = row.get("line_spacing_ratio")
        if spacing is not None:
            try: lcfg.line_spacing_ratio = float(np.clip(float(spacing),0.0,0.6))
            except Exception: pass
        letter_spacing = row.get("letter_spacing_ratio")
        if letter_spacing is not None and hasattr(lcfg, "letter_spacing_ratio"):
            try: lcfg.letter_spacing_ratio = float(np.clip(float(letter_spacing),-0.25,0.80))
            except Exception: pass
        column_spacing = row.get("column_spacing_ratio")
        if column_spacing is not None:
            try: lcfg.column_spacing_ratio = float(np.clip(float(column_spacing),-0.25,0.80))
            except Exception: pass
        alignment=str(row.get("text_alignment") or "center").strip().lower()
        if alignment in {"auto","left","center","right"}:
            lcfg.text_alignment=alignment
        fill=row.get("fill_color")
        if isinstance(fill,(list,tuple)) and len(fill)>=3:
            try: lcfg.fill=tuple(int(np.clip(int(v),0,255)) for v in fill[:3])
            except Exception: pass
        elif isinstance(fill,str) and fill.strip().startswith("#") and len(fill.strip()) in {4,7}:
            try:
                raw=fill.strip().lstrip("#")
                if len(raw)==3: raw="".join(ch*2 for ch in raw)
                lcfg.fill=tuple(int(raw[i:i+2],16) for i in (0,2,4))
            except Exception: pass

        # The OCR locator decides what old text is cleared. ``layout_bbox`` is a
        # separate, user-editable typesetting frame and never gains erase authority.
        manual_layout=list(row.get("layout_bbox") or [])
        manual_mode=str(row.get("layout_box_mode") or "auto").lower()=="manual"
        if manual_mode and len(manual_layout)==4:
            manual_layout=[int(v) for v in manual_layout]
            manual_mask=_rect_mask(target.shape[:2],manual_layout,inset=0)
            if cv2.countNonZero(manual_mask)>0:
                safe=cv2.bitwise_and(recovered_safe,manual_mask) if closed_layout else manual_mask
                if cv2.countNonZero(safe)==0:
                    safe=manual_mask
                old_container=list(layout_diag.get("container_bbox") or [])
                layout_diag=dict(layout_diag)
                if old_container: layout_diag["recovered_container_bbox"]=old_container
                layout_diag["container_bbox"]=manual_layout
                layout_diag["manual_layout_bbox"]=manual_layout
                layout_diag["layout_source"]="manual_layout_frame"

        x0,y0,x1,y1 = bbox
        poly=[(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        unit=TextUnit(
            id=str(row.get("id") or "manual-ocr"), polygon=poly, block_ids=[], text=text,
            confidence=float(row.get("confidence") or 1.0), kind="speech", reading_order=0,
            bubble_id=None, meta={"manual_ocr_block":True,"box_locked":True},
        )
        lr=fit_text(target.shape[:2], safe, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            applied.append({
                "id":row.get("id"),"success":False,"reason":str(lr.reason or "layout_failed"),
                "clear":clean_diag,"layout":layout_diag,"orientation_inference":orientation_diag,
            }); continue
        angle=float(row.get("rotation_degrees") or 0.0)
        if abs(angle) > 1e-3 and lr.text_mask is not None:
            center_box=manual_layout if manual_mode and len(manual_layout)==4 else list(lr.bbox)
            if len(center_box)==4:
                cc=((float(center_box[0])+float(center_box[2]))/2.0,(float(center_box[1])+float(center_box[3]))/2.0)
                mat=cv2.getRotationMatrix2D(cc,angle,1.0)
                rotated=cv2.warpAffine(lr.text_mask,mat,(target.shape[1],target.shape[0]),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_CONSTANT,borderValue=0)
                lr.text_mask=np.where(rotated>4,rotated,0).astype(np.uint8)
                ys,xs=np.where(lr.text_mask>0)
                if len(xs): lr.bbox=(int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1))
        outside_safe = int(np.count_nonzero((lr.text_mask > 0) & (safe <= 0)))
        if outside_safe != 0:
            applied.append({
                "id":row.get("id"),"success":False,"reason":"glyph_outside_safe_mask",
                "clear":clean_diag,"layout":layout_diag,"orientation_inference":orientation_diag,
                "outside_safe_pixels":outside_safe,
            }); continue
        out=composite_text(out,lr,lcfg)
        text_masks.append(lr.text_mask)
        applied.append({
            "id":row.get("id"),"success":True,"bbox":bbox,"text":text,
            "orientation":lr.orientation,"font_path":lr.font_path,"font_size":int(lr.font_size),
            "lines":list(lr.lines),"clear":clean_diag,"layout":layout_diag,
            "orientation_inference":orientation_diag,
            "coverage_inside_safe":float(lr.coverage_inside_safe),
            "outside_safe_pixels":outside_safe,
            "text_alignment":str(getattr(lcfg,"text_alignment","center")),
            "letter_spacing_ratio":float(getattr(lcfg,"letter_spacing_ratio",0.0)),
            "column_spacing_ratio":float(getattr(lcfg,"column_spacing_ratio",0.06)),
            "rotation_degrees":float(row.get("rotation_degrees") or 0.0),
            "layout_bbox":list(manual_layout) if manual_mode and len(manual_layout)==4 else [],
        })

    state={
        "schema":"folirina.ocr_edit.render_state.v1","scope":scope_dir.name,"mode":mode,
        "block_count":len(blocks),"applied":applied,
        "conflict_policy":"single_layout_container_authority_v1",
        "suppressed_render_conflicts":suppressed_render_conflicts,
        "base":str(base_path),"preview":bool(preview),
    }
    if preview:
        return {"image":out,"state":state}
    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, out)
    # Also mirror mode-owned artifacts for diagnostics without touching Direct/Mask files.
    write_image(scope_dir / "final.png", out)
    state["final"]=str(final_path)
    save_json(scope_dir / "render_state.json", state)
    return final_path


def preview_ocr_edit_blocks(page_dir: str | Path, project: dict[str, Any], cfg, blocks: list[dict[str,Any]]):
    """Render draft OCR blocks in memory without touching page artifacts."""
    return apply_ocr_edit_blocks(page_dir,project,cfg,blocks_override=[dict(x) for x in blocks],preview=True)


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


__all__=["apply_ocr_edit_blocks","preview_ocr_edit_blocks","reset_ocr_edit_blocks"]
