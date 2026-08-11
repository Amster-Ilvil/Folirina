from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import cv2

from .config import PipelineConfig
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .io_utils import load_json, read_image, save_json, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask
from .masking import build_clear_mask
from .models import BubbleInstance, TextBlock, TextUnit, UnitMatch


def _text_block(row: dict) -> TextBlock:
    return TextBlock(**row)


def _text_unit(row: dict) -> TextUnit:
    return TextUnit(**row)


def _load_target_bubbles(page_dir: Path, rows: list[dict]) -> list[BubbleInstance]:
    out = []
    for row in rows:
        b = BubbleInstance(
            id=row["id"],
            polygon=row["polygon"],
            confidence=row.get("confidence", 1.0),
            kind=row.get("kind", "speech"),
            block_ids=list(row.get("block_ids", [])),
            meta=dict(row.get("meta", {})),
        )
        mp = page_dir / "bubbles" / f"{b.id}.png"
        sp = page_dir / "bubbles" / f"{b.id}_safe.png"
        if mp.exists():
            b.mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if sp.exists():
            b.safe_mask = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        out.append(b)
    return out


def apply_review_page(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    project = load_json(page_dir / "project.json")
    override_path = page_dir / "review_overrides.json"
    overrides = load_json(override_path) if override_path.exists() else {}

    source_units = [_text_unit(x) for x in project.get("source_units", [])]
    target_units = [_text_unit(x) for x in project.get("target_units", [])]
    target_blocks = [_text_block(x) for x in project.get("target_blocks", [])]
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    target = read_image(page_dir / "target_original.png")

    source_by_id = {u.id: u for u in source_units}
    target_by_id = {u.id: u for u in target_units}
    bubbles_by_id = {b.id: b for b in target_bubbles}

    for source_id, text in dict(overrides.get("text_overrides", {})).items():
        if source_id in source_by_id:
            source_by_id[source_id].text = str(text)

    existing = {}
    for row in project.get("matches", []):
        if row.get("relation") == "one_to_one":
            existing[row["source_unit_id"]] = row["target_unit_id"]
    existing.update({str(k): str(v) for k, v in dict(overrides.get("match_overrides", {})).items()})

    if "accepted_source_units" in overrides:
        accepted_ids = set(map(str, overrides.get("accepted_source_units", [])))
    else:
        accepted_ids = {
            x.split("->", 1)[0]
            for x in project.get("meta", {}).get("auto_applied_match_ids", [])
            if "->" in x
        }
    matches: list[UnitMatch] = []
    for source_id in accepted_ids:
        target_id = existing.get(source_id)
        if source_id in source_by_id and target_id in target_by_id:
            matches.append(UnitMatch(source_id, target_id, 1.0, 0.0, "one_to_one", ["review_accepted"]))

    manual_mask = page_dir / "manual_clear_mask.png"
    if manual_mask.exists():
        mask = cv2.imread(str(manual_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != target.shape[:2]:
            raise ValueError("manual_clear_mask.png has invalid dimensions")
        from .masking import MaskBuildResult
        mask_result = MaskBuildResult((mask > 0).astype("uint8") * 255, {}, 0, int(cv2.countNonZero(mask)))
    else:
        mask_result = build_clear_mask(
            target.shape[:2], target_blocks, target_units, target_bubbles, matches, cfg.masking,
            min_match_confidence=0.0, allow_relations={"one_to_one"}
        )

    inpainted = inpaint_image(target, mask_result.mask, cfg.inpainting).image
    rendered = inpainted.copy()
    lettering = []
    masks = []
    for match in matches:
        src = source_by_id[match.source_unit_id]
        dst = target_by_id[match.target_unit_id]
        safe = bubbles_by_id.get(dst.bubble_id).safe_mask if dst.bubble_id and dst.bubble_id in bubbles_by_id else None
        if safe is None or cv2.countNonZero(safe) == 0:
            safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, cfg.bubbles.safe_margin_px // 2))
        lr = fit_text(target.shape[:2], safe, dst, src.text, cfg.lettering)
        lettering.append(lr)
        if lr.success and lr.text_mask is not None:
            rendered = composite_text(rendered, lr, cfg.lettering)
            masks.append(lr.text_mask)

    final_path = page_dir / "final_reviewed.png"
    inpainted_path = page_dir / "inpainted_reviewed.png"
    text_path = page_dir / "text_layer_reviewed.png"
    write_image(final_path, rendered)
    write_image(inpainted_path, inpainted)
    text_rgba = make_text_layer_rgba(target.shape[:2], masks, color=cfg.lettering.fill)
    write_rgba(text_path, text_rgba)
    export_openraster(page_dir / "editable_reviewed.ora", target, inpainted, text_rgba)
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", inpainted_path, text_path)
    save_json(
        page_dir / "review_applied.json",
        {
            "status": overrides.get("status", "reviewed"),
            "notes": overrides.get("notes", ""),
            "accepted_source_units": sorted(accepted_ids),
            "matches": [m.to_dict() for m in matches],
            "lettering": [x.to_dict() for x in lettering],
            "manual_mask": manual_mask.exists(),
            "psd_exported": psd_ok,
            "final": str(final_path),
        },
    )
    return final_path
