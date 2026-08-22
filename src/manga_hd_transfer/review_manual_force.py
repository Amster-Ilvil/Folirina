from __future__ import annotations

"""Manual-force review service.

Owns the reviewer-activated force-transfer mask path.  It is independent from
Qt widgets and from the main review dispatcher.
"""

from pathlib import Path
import cv2
import numpy as np

from .config import PipelineConfig
from .inpainting import inpaint_image
from .io_utils import load_json, read_image, save_json, write_image
from .manual_effect import (
    build_manual_effect_masks, estimate_source_background, composite_source_text_delta,
    strip_border_ring_components, clean_manual_target_text,
)
from .mode_contracts import review_owner_compatible
from .result_state import ensure_manual_baseline, manual_baseline_path, commit_reviewed_result
from .schema_compat import normalize_project
from .text_only_transfer import target_text_mask_in_container
from .workspace_guard import guarded_page_write
from .review_target_layer import _apply_target_layer_erase_to_rendered, _apply_target_layer_restore_to_rendered
from .review_common import _dict_or_empty, _source_for_review, _project_text_ink_mask, _write_bgra

def manual_force_auto_evidence_masks(
    page_dir: str | Path,
    project: dict | None = None,
    target: np.ndarray | None = None,
    source: np.ndarray | None = None,
    *,
    include_override: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Collect OCR/automatic text evidence for the manual-force brush.

    Returns TARGET cleanup evidence and SOURCE-Chinese evidence in TARGET
    coordinates.  Existing automatic masks are preferred, while OCR/detector
    polygons in project.json provide a second route.  This is *supporting*
    evidence: the reviewer brush remains the activation authority.
    """
    page_dir = Path(page_dir)
    if target is None:
        target = read_image(page_dir / "target_original.png")
    if project is None:
        project = normalize_project(load_json(page_dir / "project.json"))
    shape = target.shape[:2]
    target_auto = np.zeros(shape, np.uint8)
    artifact_sources: list[str] = []
    artifact_raw_pixels = 0
    artifact_compact_pixels = 0
    for name in ("target_clear_mask.png", "clear_mask.png"):
        path = page_dir / name
        if not path.exists():
            continue
        probe = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if probe is None or probe.shape != shape:
            continue
        raw = (probe > 0).astype(np.uint8) * 255
        artifact_raw_pixels += int(cv2.countNonZero(raw))
        # Automatic clear artifacts can be whole bubbles/large review envelopes.
        # Convert them back to compact lettering evidence before showing or
        # reusing them in the manual-force editor. A small dilation gives a thin
        # glyph mask enough interior for the text selector without expanding the
        # final evidence itself.
        roi = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)), iterations=1)
        compact = target_text_mask_in_container(target, roi)
        if cv2.countNonZero(compact) > 0:
            # Never admit distant dark artwork that is outside the original
            # auto area plus a narrow antialias halo.
            allowed = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            compact = cv2.bitwise_and(compact, allowed)
        if cv2.countNonZero(compact) <= 0:
            # If the auto artifact was already a compact text mask, retain it.
            # Broad artifacts are intentionally not used verbatim.
            raw_ratio = float(cv2.countNonZero(raw)) / max(1.0, float(shape[0] * shape[1]))
            compact = raw if raw_ratio <= 0.035 else np.zeros_like(raw)
        target_auto = np.maximum(target_auto, compact)
        artifact_compact_pixels += int(cv2.countNonZero(compact))
        artifact_sources.append(name)

    target_ocr = np.zeros(shape, np.uint8)
    target_ocr_regions = 0
    for target_rows in (project.get("target_units", []), project.get("target_blocks", [])):
        probe, used = _project_text_ink_mask(target, target_rows)
        target_ocr = np.maximum(target_ocr, probe)
        target_ocr_regions += int(used)
    target_auto = np.maximum(target_auto, target_ocr)
    target_override_used = False
    target_override_pixels = 0
    if include_override:
        override_path = page_dir / "manual_force_auto_target_override.png"
        if override_path.exists():
            override = cv2.imread(str(override_path), cv2.IMREAD_GRAYSCALE)
            if override is not None and override.shape == shape:
                target_auto = (override > 0).astype(np.uint8) * 255
                target_override_used = True
                target_override_pixels = int(cv2.countNonZero(target_auto))

    source_auto = np.zeros(shape, np.uint8)
    source_ocr_regions = 0
    registration_ok = False
    try:
        if source is None:
            source = _source_for_review(page_dir, project)
        source_local = np.zeros(source.shape[:2], np.uint8)
        for source_rows in (project.get("source_units", []), project.get("source_blocks", [])):
            probe, used = _project_text_ink_mask(source, source_rows)
            source_local = np.maximum(source_local, probe)
            source_ocr_regions += int(used)
        reg = _dict_or_empty(project.get("registration"))
        matrix = np.asarray(reg.get("matrix"), dtype=np.float64)
        if matrix.shape == (3, 3) and cv2.countNonZero(source_local) > 0:
            source_auto = cv2.warpPerspective(
                source_local, matrix, (shape[1], shape[0]),
                flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            source_auto = (source_auto > 0).astype(np.uint8) * 255
            registration_ok = True
    except Exception:
        # Automatic evidence is optional.  Never make the human fallback depend
        # on OCR/project metadata being perfectly intact.
        source_auto = np.zeros(shape, np.uint8)

    return target_auto, source_auto, {
        "artifact_sources": artifact_sources,
        "artifact_raw_pixels": int(artifact_raw_pixels),
        "artifact_compact_pixels": int(artifact_compact_pixels),
        "target_ocr_regions": int(target_ocr_regions),
        "source_ocr_regions": int(source_ocr_regions),
        "target_pixels": int(cv2.countNonZero(target_auto)),
        "source_pixels": int(cv2.countNonZero(source_auto)),
        "source_registration_ok": bool(registration_ok),
        "target_override_used": bool(target_override_used),
        "target_override_pixels": int(target_override_pixels),
    }

def _auto_evidence_touching(evidence: np.ndarray, seed: np.ndarray, *, join_radius: int = 8) -> np.ndarray:
    """Return complete nearby auto-text groups touched by a human seed.

    OCR masks are often fragmented glyph-by-glyph.  Group on a dilated copy, but
    return only the original compact ink pixels.  A tiny brush stroke can thus
    activate the whole OCR-recognized word/line without turning its bbox into an
    unsafe rectangle.
    """
    ev = (np.asarray(evidence, dtype=np.uint8) > 0).astype(np.uint8) * 255
    sd = (np.asarray(seed, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(ev) == 0 or cv2.countNonZero(sd) == 0:
        return np.zeros_like(ev)
    r = max(2, min(24, int(join_radius)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
    grouped = cv2.dilate(ev, k, iterations=1)
    count, labels = cv2.connectedComponents((grouped > 0).astype(np.uint8), 8)
    if count <= 1:
        return np.zeros_like(ev)
    seed_near = cv2.dilate(sd, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)), iterations=1) > 0
    touched = np.unique(labels[seed_near & (labels > 0)])
    if touched.size == 0:
        return np.zeros_like(ev)
    selected_group = np.isin(labels, touched)
    return np.where((ev > 0) & selected_group, 255, 0).astype(np.uint8)


def _keep_components_touching(mask: np.ndarray, support: np.ndarray, *, halo_px: int = 3) -> np.ndarray:
    """Keep only mask components that touch supporting text evidence.

    The manual-force brush may span both text and neighbouring face/artwork.
    ``mask`` is a candidate compact dark-ink map; ``support`` is paired/source/
    OCR evidence that actually proves text.  Components with no nearby support
    are discarded so eye/face/illustration strokes cannot enter the cleanup mask.
    """
    probe = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    anchor = (np.asarray(support, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(probe) == 0 or cv2.countNonZero(anchor) == 0:
        return np.zeros_like(probe)
    h = max(0, min(10, int(halo_px)))
    if h > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (h * 2 + 1, h * 2 + 1))
        anchor = cv2.dilate(anchor, k, iterations=1)
    count, labels = cv2.connectedComponents((probe > 0).astype(np.uint8), 8)
    if count <= 1:
        return probe if np.any((probe > 0) & (anchor > 0)) else np.zeros_like(probe)
    keep = np.zeros_like(probe)
    for lab in range(1, count):
        comp = labels == lab
        if np.any(comp & (anchor > 0)):
            keep[comp] = 255
    return keep

def _manual_force_settings(page_dir: Path) -> dict:
    path = page_dir / "manual_force_settings.json"
    data = load_json(path) if path.exists() else {}
    return {
        "use_auto_evidence": bool(data.get("use_auto_evidence", True)),
        "owner_transfer_mode": str(data.get("owner_transfer_mode", "") or "").strip().lower(),
        "schema": str(data.get("schema", "") or ""),
    }

def _manual_force_mask_regions(mask: np.ndarray) -> list[tuple[np.ndarray, list[int]]]:
    """Cluster a reviewer brush mask into text-sized local rerun regions.

    The saved mask is the authority for TARGET cleanup.  A lightly dilated copy
    is used only to group nearby brush strokes (for example, separate Japanese
    glyphs in the same line) so SOURCE Chinese can be recovered from the whole
    corresponding text block without OCR.
    """
    raw = (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(raw) == 0:
        return []
    h, w = raw.shape
    # Merge nearby strokes, but never turn the actual cleanup mask into a broad
    # rectangle.  The connected region is only an analysis envelope.
    merge = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)), iterations=1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((merge > 0).astype(np.uint8), 8)
    rows: list[tuple[np.ndarray, list[int]]] = []
    for lab in range(1, count):
        x, y, bw, bh, area = [int(v) for v in stats[lab]]
        if area <= 0:
            continue
        # Grab the undilated reviewer paint belonging to this cluster.
        cluster = np.zeros_like(raw)
        cluster[(labels == lab) & (raw > 0)] = 255
        if cv2.countNonZero(cluster) == 0:
            # The raw stroke may sit just outside a dilated label edge due to
            # integer morphology. Fall back to the label footprint intersected
            # with raw so the editor can never silently discard a stroke.
            cluster[(labels == lab)] = raw[(labels == lab)]
        ys, xs = np.where(cluster > 0)
        if xs.size == 0:
            continue
        x0, x1 = int(xs.min()), int(xs.max() + 1)
        y0, y1 = int(ys.min()), int(ys.max() + 1)
        span = max(x1 - x0, y1 - y0)
        pad = max(8, min(28, int(round(span * 0.12))))
        bbox = [max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)]
        rows.append((cluster, bbox))
    return rows

def _apply_manual_force_transfer_mask(
    page_dir: Path,
    rendered: np.ndarray,
    target: np.ndarray,
    project: dict,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Apply editable manual + OCR/automatic mask corrections.

    The red reviewer layer remains a local activation mask.  The blue automatic
    layer can now also be edited directly: additions become local rerun seeds,
    while removals restore exact TARGET pixels so an oversized/wrong automatic
    region can be corrected without repainting it into the red layer first.
    """
    mask_path = page_dir / "manual_force_transfer_mask.png"
    override_path = page_dir / "manual_force_auto_target_override.png"
    empty_layer = np.zeros((*target.shape[:2], 4), np.uint8)
    empty_mask = np.zeros(target.shape[:2], np.uint8)

    if mask_path.exists():
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != target.shape[:2]:
            raise ValueError("manual_force_transfer_mask.png has invalid dimensions")
        raw = (raw > 0).astype(np.uint8) * 255
    else:
        raw = np.zeros(target.shape[:2], np.uint8)

    auto_added = np.zeros_like(raw)
    auto_removed = np.zeros_like(raw)
    edited_auto = None
    if override_path.exists():
        edited_auto = cv2.imread(str(override_path), cv2.IMREAD_GRAYSCALE)
        if edited_auto is None or edited_auto.shape != target.shape[:2]:
            raise ValueError("manual_force_auto_target_override.png has invalid dimensions")
        edited_auto = (edited_auto > 0).astype(np.uint8) * 255
        original_auto, _ignored_source, _ignored_diag = manual_force_auto_evidence_masks(
            page_dir, project=project, target=target, include_override=False
        )
        auto_added[(edited_auto > 0) & (original_auto == 0)] = 255
        auto_removed[(original_auto > 0) & (edited_auto == 0)] = 255

    activation = np.maximum(raw, auto_added)
    if cv2.countNonZero(activation) == 0 and cv2.countNonZero(auto_removed) == 0:
        if not mask_path.exists() and not override_path.exists():
            return rendered, empty_layer, empty_mask, {"enabled": False, "reason": "mask_missing"}
        return rendered, empty_layer, empty_mask, {
            "enabled": True, "reason": "empty_mask", "manual_pixels": 0,
            "auto_added_pixels": 0, "auto_removed_pixels": 0,
        }

    settings = _manual_force_settings(page_dir)
    project_mode = str(_dict_or_empty(project.get("meta")).get("transfer_mode", "") or "").strip().lower()
    owner_mode = str(settings.get("owner_transfer_mode", "") or "").strip().lower()
    if owner_mode and not review_owner_compatible(owner_mode, project_mode):
        return rendered, empty_layer, empty_mask, {
            "enabled": False, "reason": "manual_force_owner_mode_mismatch",
            "owner_transfer_mode": owner_mode, "project_transfer_mode": project_mode,
        }
    use_auto_evidence = bool(settings.get("use_auto_evidence", True))
    source = None
    if cv2.countNonZero(activation) > 0:
        source = _source_for_review(page_dir, project)
    auto_target = np.zeros(target.shape[:2], np.uint8)
    auto_source = np.zeros(target.shape[:2], np.uint8)
    auto_diag = {"enabled": False, "target_pixels": 0, "source_pixels": 0}
    if use_auto_evidence:
        auto_target, auto_source, collected = manual_force_auto_evidence_masks(
            page_dir, project=project, target=target, source=source, include_override=True
        )
        auto_diag = {"enabled": True, **collected}
        write_image(page_dir / "manual_force_auto_target_evidence.png", auto_target)
        write_image(page_dir / "manual_force_auto_source_evidence.png", auto_source)

    out = rendered.copy()
    all_source = np.zeros(target.shape[:2], np.uint8)
    all_clear = np.zeros(target.shape[:2], np.uint8)
    cleaned_target = target.copy()
    aligned_source = None
    rows_diag: list[dict] = []

    # A blue-layer erase means "this automatic text region was wrong". Restore
    # exact TARGET pixels immediately; this also removes a stale Chinese candidate
    # that may already be present in the automatic baseline.
    removed_sel = auto_removed > 0
    if np.any(removed_sel):
        out[removed_sel] = target[removed_sel]
        write_image(page_dir / "manual_force_auto_removed_mask.png", auto_removed)
    else:
        try: (page_dir / "manual_force_auto_removed_mask.png").unlink(missing_ok=True)
        except OSError: pass

    # Blue additions and red strokes share the same safe local rerun path.
    for index, (cluster, bbox) in enumerate(_manual_force_mask_regions(activation)):
        auto_target_local = _auto_evidence_touching(auto_target, cluster) if use_auto_evidence else np.zeros_like(cluster)
        auto_source_local = _auto_evidence_touching(auto_source, cluster) if use_auto_evidence else np.zeros_like(cluster)
        support = np.maximum(cluster, np.maximum(auto_target_local, auto_source_local))
        ys, xs = np.where(support > 0)
        if xs.size:
            x0, x1 = int(xs.min()), int(xs.max() + 1)
            y0, y1 = int(ys.min()), int(ys.max() + 1)
            span = max(x1 - x0, y1 - y0)
            pad = max(8, min(32, int(round(span * 0.10))))
            bbox = [max(0, x0 - pad), max(0, y0 - pad), min(target.shape[1], x1 + pad), min(target.shape[0], y1 + pad)]
        row = {
            "id": f"manual-force-{index:03d}",
            "mode": "reveal_text",
            "target_bbox": bbox,
            # A reviewer has already localized the missed text, so the paired
            # extractor can be intentionally more permissive than auto mode.
            "diff_threshold": 16,
            "edge_threshold": 36.0,
            "expand_px": 3,
            "auto_clear_target": True,
        }
        masks = build_manual_effect_masks(source, target, project, row, cfg)
        aligned_source = masks.aligned_source if aligned_source is None else aligned_source
        source_mask = (masks.source_mask > 0).astype(np.uint8) * 255
        detected_clear = (masks.target_clear_mask > 0).astype(np.uint8) * 255
        if use_auto_evidence:
            # OCR/automatic evidence is compact ink, never a broad bbox.  The
            # reviewer stroke must touch the recognized group before it is used.
            source_mask = np.maximum(source_mask, auto_source_local)
            detected_clear = np.maximum(detected_clear, auto_target_local)

        # Detector-free fallback: inside a human-confirmed local text envelope,
        # recover compact SOURCE ink that differs from TARGET. This handles cases
        # where the automatic paired-difference gate emitted no source glyphs at
        # all (the exact failure this tool is designed to repair).
        if cv2.countNonZero(source_mask) < 8:
            x0, y0, x1, y1 = map(int, bbox)
            region = np.zeros(target.shape[:2], np.uint8)
            region[y0:y1, x0:x1] = 255
            compact = target_text_mask_in_container(masks.aligned_source, region)
            delta = np.max(cv2.absdiff(masks.aligned_source, target), axis=2)
            changed = (delta >= 10).astype(np.uint8) * 255
            changed = cv2.dilate(changed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            fallback = cv2.bitwise_and(compact, changed)
            if cv2.countNonZero(fallback) >= 4:
                source_mask = np.maximum(source_mask, fallback)

        # The red force brush is a *localisation/reveal command*, not a command
        # to erase every painted background pixel.  Previous builds used the raw
        # filled brush as the clear mask, so painting broadly across a gradient
        # bubble could inpaint the whole centre and create radial smears.  Reduce
        # the reviewer envelope to compact TARGET ink first, then combine it with
        # the paired/colour detector evidence.  This lets a user paint over the
        # whole sentence/bubble without damaging the TARGET artwork.
        brush_region = cv2.dilate(cluster, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        brush_text = target_text_mask_in_container(target, brush_region)
        if cv2.countNonZero(brush_text):
            brush_text, brush_text_diag = strip_border_ring_components(brush_text, brush_region)
            if cv2.countNonZero(brush_text):
                brush_text = cv2.dilate(brush_text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                brush_text[brush_region == 0] = 0
                # v2.3.61: broad reviewer envelopes may cover a nearby eye/face.
                # Keep only compact dark-ink components that are supported by
                # actual text evidence from SOURCE/paired-diff/OCR.
                delta = np.max(cv2.absdiff(masks.aligned_source, target), axis=2)
                delta_support = (delta >= 10).astype(np.uint8) * 255
                delta_support = cv2.dilate(delta_support, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
                support = np.maximum(source_mask, np.maximum(detected_clear, np.maximum(auto_target_local, delta_support)))
                supported_brush_text = _keep_components_touching(brush_text, support, halo_px=4)
                if cv2.countNonZero(supported_brush_text) > 0:
                    brush_text = supported_brush_text
                else:
                    brush_text = np.zeros_like(brush_text)
        else:
            brush_text_diag = {"removed_components": 0, "removed_pixels": 0, "kept_components": 0, "kept_pixels": 0}
        local_clear = np.maximum(detected_clear, brush_text)
        # Extremely small reviewer strokes can intentionally target one missed
        # fringe that no component detector sees. Honour only such sparse masks;
        # never fall back to erasing a broad filled rectangle.
        bbox_area=max(1,(bbox[2]-bbox[0])*(bbox[3]-bbox[1]))
        brush_fraction=float(cv2.countNonZero(cluster)/bbox_area)
        raw_brush_fallback=False
        if cv2.countNonZero(local_clear)==0 and brush_fraction<=0.10:
            local_clear=brush_region.copy()
            raw_brush_fallback=True
        local_clean, clean_diag = clean_manual_target_text(
            target,
            local_clear,
            bbox=bbox,
            # The red force brush is an explicit reviewer command.  A coloured
            # gradient detector is allowed to improve the cleanup, but it must
            # never clip away the part the reviewer actually painted.
            honor_mask_outside_colored_safe=True,
        )
        changed_clean = np.any(local_clean != target, axis=2)
        if np.any(changed_clean):
            cleaned_target[changed_clean] = local_clean[changed_clean]
            all_clear[changed_clean] = 255
        else:
            # Defensive fallback: if a backend returns byte-identical pixels,
            # still honor the exact reviewer brush through configured inpainting.
            fallback_clean = inpaint_image(target, local_clear, cfg.inpainting).image
            sel = local_clear > 0
            cleaned_target[sel] = fallback_clean[sel]
            all_clear[sel] = 255

        all_source = np.maximum(all_source, source_mask)
        rows_diag.append({
            "bbox": bbox,
            "brush_pixels": int(cv2.countNonZero(cluster)),
            "brush_text_pixels": int(cv2.countNonZero(brush_text)),
            "brush_fraction": float(brush_fraction),
            "raw_brush_clear_fallback": bool(raw_brush_fallback),
            "brush_text_border_filter": brush_text_diag,
            "source_pixels": int(cv2.countNonZero(source_mask)),
            "detected_target_pixels": int(cv2.countNonZero(detected_clear)),
            "auto_target_pixels": int(cv2.countNonZero(auto_target_local)),
            "auto_source_pixels": int(cv2.countNonZero(auto_source_local)),
            "effective_clear_pixels": int(np.count_nonzero(changed_clean)) if np.any(changed_clean) else int(cv2.countNonZero(local_clear)),
            "extractor": masks.diagnostics,
            "cleanup": clean_diag,
        })

    # Replace only TARGET-layer pixels under the explicit cleanup footprint.
    clear_sel = all_clear > 0
    out[clear_sel] = cleaned_target[clear_sel]

    transfer_diag: dict = {"delta_pixels": 0}
    if aligned_source is not None and cv2.countNonZero(all_source) > 0:
        source_background = estimate_source_background(aligned_source, all_source)
        before = out.copy()
        out, transfer_diag = composite_source_text_delta(
            out, aligned_source, all_source, source_background=source_background
        )
        diff = np.max(cv2.absdiff(out, before), axis=2)
        alpha = np.where(diff > 0, 255, 0).astype(np.uint8)
    else:
        alpha = np.zeros(target.shape[:2], np.uint8)

    layer = np.zeros((*target.shape[:2], 4), np.uint8)
    layer[:, :, :3] = out
    layer[:, :, 3] = alpha
    write_image(page_dir / "manual_force_source_mask.png", all_source)
    write_image(page_dir / "manual_force_target_clear_mask.png", all_clear)
    _write_bgra(page_dir / "manual_force_transfer_layer.png", layer)
    save_json(page_dir / "manual_force_transfer.json", {
        "schema": "manga_hd_translation_transfer.manual_force_transfer.v1",
        "manual_mask": str(mask_path),
        "manual_pixels": int(cv2.countNonZero(raw)),
        "activation_pixels": int(cv2.countNonZero(activation)),
        "auto_added_pixels": int(cv2.countNonZero(auto_added)),
        "auto_removed_pixels": int(cv2.countNonZero(auto_removed)),
        "auto_override_present": bool(override_path.exists()),
        "region_count": len(rows_diag),
        "source_pixels": int(cv2.countNonZero(all_source)),
        "target_clear_pixels": int(cv2.countNonZero(all_clear)),
        "written_pixels": int(cv2.countNonZero(alpha)),
        "background_policy": "TARGET_only_SOURCE_text_delta",
        "ocr_required": False,
        "auto_evidence_enabled": bool(use_auto_evidence),
        "auto_evidence": auto_diag,
        "regions": rows_diag,
        "transfer": transfer_diag,
    })
    return out, layer, all_clear, {
        "enabled": True,
        "manual_pixels": int(cv2.countNonZero(raw)),
        "region_count": len(rows_diag),
        "source_pixels": int(cv2.countNonZero(all_source)),
        "target_clear_pixels": int(cv2.countNonZero(all_clear)),
        "written_pixels": int(cv2.countNonZero(alpha)),
        "auto_evidence_enabled": bool(use_auto_evidence),
        "auto_evidence": auto_diag,
        "regions": rows_diag,
        "transfer": transfer_diag,
    }

def apply_manual_force_transfer_review(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    """Fast local rerun driven by ``manual_force_transfer_mask.png``.

    The first invocation freezes the current good result as ``final_auto.png``.
    Every subsequent brush edit starts from that immutable baseline, so repainting
    or erasing the manual mask cannot accumulate blur or duplicated Chinese.
    """
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    target = read_image(page_dir / "target_original.png")
    project = normalize_project(load_json(page_dir / "project.json"))
    baseline_path = ensure_manual_baseline(page_dir)
    baseline = read_image(baseline_path)
    if baseline.shape != target.shape:
        raise ValueError("manual force baseline size mismatch")
    rendered, layer, clear_mask, diag = _apply_manual_force_transfer_mask(page_dir, baseline, target, project, cfg)
    # Existing TARGET-only finishing brushes remain downstream of the new Chinese
    # layer, and its alpha is explicitly protected from erasing.
    extra = [layer[:, :, 3]] if layer.ndim == 3 and layer.shape[2] >= 4 else []
    rendered, erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=extra
    )
    rendered, restore_diag = _apply_target_layer_restore_to_rendered(
        page_dir, rendered, target, refresh_base=True
    )
    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, rendered)
    save_json(page_dir / "manual_force_apply.json", {
        "schema": "manga_hd_translation_transfer.manual_force_apply.v1",
        "baseline": str(baseline_path),
        "manual_force": diag,
        "target_layer_erase": erase_diag,
        "target_layer_restore": restore_diag,
        "final": str(final_path),
    })
    return commit_reviewed_result(page_dir, final_path)

def reset_manual_force_transfer_review(page_dir: str | Path, config: PipelineConfig | None = None) -> Path | None:
    """Remove the manual force mask and return to the immutable pre-tool result."""
    page_dir = Path(page_dir)
    for name in (
        "manual_force_transfer_mask.png", "manual_force_source_mask.png",
        "manual_force_target_clear_mask.png", "manual_force_transfer_layer.png",
        "manual_force_auto_target_evidence.png", "manual_force_auto_source_evidence.png",
        "manual_force_auto_target_override.png", "manual_force_auto_removed_mask.png",
        "manual_force_settings.json", "manual_force_transfer.json", "manual_force_apply.json",
    ):
        try:
            (page_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    baseline = manual_baseline_path(page_dir)
    if not baseline.exists():
        return None
    target = read_image(page_dir / "target_original.png") if (page_dir / "target_original.png").exists() else None
    rendered = read_image(baseline)
    if target is not None and rendered.shape == target.shape:
        cfg = config or PipelineConfig()
        rendered, _ = _apply_target_layer_erase_to_rendered(page_dir, rendered, target, cfg, refresh_base=True)
        rendered, _ = _apply_target_layer_restore_to_rendered(page_dir, rendered, target, refresh_base=True)
    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, rendered)
    return commit_reviewed_result(page_dir, final_path)

__all__ = ['manual_force_auto_evidence_masks', '_auto_evidence_touching', '_manual_force_settings', '_manual_force_mask_regions', '_apply_manual_force_transfer_mask', 'apply_manual_force_transfer_review', 'reset_manual_force_transfer_review']
