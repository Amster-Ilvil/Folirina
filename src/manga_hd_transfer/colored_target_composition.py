"""Compose pixel-mode translations over an independently cleaned colour master."""
from dataclasses import dataclass

import cv2
import numpy as np

from .target_text_removal import coloured_container_removal, inpaint_target
from .text_ink_equivalence import equivalent_registered_ink


@dataclass
class TargetFirstComposition:
    image: np.ndarray
    clean_base: np.ndarray
    clear_mask: np.ndarray
    authority: np.ndarray
    source_layer: np.ndarray
    diagnostics: dict


def compose_coloured_containers(target, proposed, aligned_source, source_ink, records, callback):
    if proposed.shape != target.shape or aligned_source.shape != target.shape or source_ink.shape != target.shape[:2]:
        raise ValueError("TARGET-first composition dimensions differ")
    shape = target.shape[:2]
    clear = np.zeros(shape, np.uint8)
    owned = np.zeros(shape, np.uint8)
    restore = np.zeros(shape, np.uint8)
    source_support = np.zeros(shape, np.uint8)
    white_clear = np.zeros(shape, np.uint8)
    source_gray = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2GRAY)
    source_sat = cv2.cvtColor(aligned_source, cv2.COLOR_BGR2HSV)[..., 1]
    rows, plans = [], []
    flat_fills = []
    for record in records or ():
        if not bool(getattr(record, "applied", False)):
            continue
        box = getattr(record, "target_bbox", None)
        if not box or len(box) != 4:
            continue
        x0, y0, x1, y1 = map(int, box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(shape[1], x1), min(shape[0], y1)
        if x1 <= x0 or y1 <= y0:
            continue
        seed = np.zeros(shape, np.uint8)
        seed[y0:y1, x0:x1] = 255
        plan = coloured_container_removal(target, seed)
        if not np.any(plan.domain):
            if plan.diagnostics.get("reason") not in {"not_coloured_container", "empty_seed"}:
                rows.append({"id": str(getattr(record, "target_bubble_id", "")),
                             "status": "deferred_target_container_unproven", **plan.diagnostics})
            continue
        white_plan = plan.diagnostics.get("reason") == "target_white_container_holes"
        if white_plan:
            from .white_bubble_composite import white_bubble_interior
            contours, _ = cv2.findContours(plan.domain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            polygon = max(contours, key=cv2.contourArea).reshape(-1, 2).tolist() if contours else []
            inner, _ = white_bubble_interior(target, {"polygon": polygon, "kind": "speech", "confidence": 1.0})
            if inner is None:
                continue
            plan.mask = inner
        domain = plan.domain > 0
        if np.count_nonzero(domain & (owned > 0)) / max(1, np.count_nonzero(domain)) > 0.5:
            continue
        row = {"id": str(getattr(record, "target_bubble_id", "")), **plan.diagnostics}
        paper = (source_gray[domain] >= 215) & (source_sat[domain] <= 35)
        if float(np.mean(paper)) < 0.65:
            rows.append({**row, "status": "deferred_source_is_not_monochrome_paper"})
            continue
        # The target mask has already been built without reading SOURCE. SOURCE
        # is consulted only now to admit and render translated lettering.
        interior = cv2.erode(plan.domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) > 0
        ink = (source_ink > 0) & interior & (source_gray < 235)
        if np.count_nonzero(ink) < 16:
            rows.append({**row, "status": "deferred_missing_source_lettering"})
            continue
        compare_scope = np.maximum(plan.mask, ink.astype(np.uint8)*255)
        if equivalent_registered_ink(aligned_source, target, compare_scope):
            rows.append({**row, "status": "preserved_shared_lettering"})
            continue
        # Prove flat TARGET colour independently of SOURCE. Model inpainting of
        # neon paper can leave pale outlines of the old glyphs, invisible to a
        # dark-pixel residual audit. Exact paper reconstruction avoids that.
        colour_inner = cv2.erode(plan.domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        samples = target[(colour_inner > 0) & (plan.mask == 0)]
        flat_colour = None
        if not white_plan and len(samples) >= max(64, int(np.count_nonzero(colour_inner) * .5)):
            paper_colour = np.median(samples, axis=0)
            deviations = np.max(np.abs(samples.astype(np.float32) - paper_colour), axis=1)
            paper_hsv = cv2.cvtColor(np.uint8([[paper_colour]]), cv2.COLOR_BGR2HSV)[0, 0]
            if paper_hsv[1] >= 50 and paper_hsv[2] >= 150 and np.percentile(deviations, 95) <= 10:
                flat_colour = np.rint(paper_colour).astype(np.uint8)
        if callback is None and not white_plan and flat_colour is None:
            rows.append({**row, "status": "deferred_inpainting_unavailable"})
            continue
        owned[domain] = 255
        clear = np.maximum(clear, plan.mask)
        if white_plan:
            white_clear = np.maximum(white_clear, plan.mask)
        if flat_colour is not None:
            plan.mask = colour_inner
            clear = np.maximum(clear, colour_inner)
            flat_fills.append((colour_inner, flat_colour))
        source_support[ink] = 255
        # Restore the original contour and a small external AA fringe; no
        # SOURCE paper or previous fill operation can become the new master.
        restore = np.maximum(restore, cv2.dilate(plan.domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))))
        row["status"] = "applied_white_blank_first" if white_plan else "applied_target_first_lama"
        if flat_colour is not None:
            row.update(status="applied_flat_colour_blank_first", paper_bgr=flat_colour.tolist())
        rows.append(row)
        plans.append((plan, row))
    if not plans:
        return TargetFirstComposition(proposed, target.copy(), clear, restore,
                                      np.zeros((*shape, 4), np.uint8),
                                      {"applied": 0, "rows": rows})
    model_mask = clear.copy(); model_mask[white_clear > 0] = 0
    for flat_mask, _ in flat_fills:
        model_mask[flat_mask > 0] = 0
    if np.any(model_mask):
        clean, model_diagnostics = inpaint_target(target, model_mask, callback)
    else:
        clean, model_diagnostics = target.copy(), {"clear_pixels": 0, "model_invoked": False}
    clean[white_clear > 0] = 255
    for flat_mask, paper_colour in flat_fills:
        clean[flat_mask > 0] = paper_colour
    # Inpainting can itself leave or hallucinate a small fragment. Inspect the
    # clean base before adding Chinese, and retry once within the same container.
    retry = np.zeros(shape, np.uint8)
    audit_scopes = []
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    clean_gray = cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)
    for plan, row in plans:
        interior = cv2.erode(plan.domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        if not np.any(interior):
            interior = plan.mask > 0
        dark_limit = min(100, float(np.percentile(target_gray[interior], 75))-45)
        residual = interior & (plan.mask > 0) & (clean_gray < dark_limit)
        audit_scopes.append((interior, dark_limit))
        if np.count_nonzero(residual) > 16:
            expanded = cv2.dilate(residual.astype(np.uint8)*255,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
            expanded[~interior] = 0
            retry = np.maximum(retry, expanded)
    if np.any(retry):
        clean, retry_diagnostics = inpaint_target(clean, retry, callback)
        clear = np.maximum(clear, retry)
        model_diagnostics["residual_retry"] = retry_diagnostics
    # Koharu-style deterministic cleanup is safe for an actually uniform white
    # balloon. LaMa can leave faint phantom glyphs even after its second pass.
    # Colour or gradient balloons never enter this exact-paper branch.
    for plan, row in plans:
        if row.get("reason") != "target_white_container_holes":
            continue
        interior = cv2.erode(plan.domain, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))) > 0
        clean[plan.mask > 0] = 255
        row["flat_white_cleanup"] = True
    alpha = np.where(source_support > 0, (255.0-source_gray.astype(np.float32))/255.0, 0.0)
    source_layer = np.zeros((*shape, 4), np.uint8)
    source_layer[..., 3] = np.rint(alpha*255).astype(np.uint8)
    out = proposed.copy()
    use = restore > 0
    out[use] = clean[use]
    translated = np.clip(clean.astype(np.float32)*(1.0-alpha[..., None]), 0, 255).astype(np.uint8)
    out[source_support > 0] = translated[source_support > 0]
    for (plan, row), (interior, dark_limit) in zip(plans, audit_scopes):
        # Audit the whole interior, not just the mask that we generated: a
        # mask-based audit cannot discover lettering that the mask missed.
        original_dark = interior & (target_gray < dark_limit)
        retained = original_dark & np.all(out == target, axis=2) & (alpha < 0.05)
        row["unchanged_target_dark_pixels_outside_source_ink"] = int(np.count_nonzero(retained))
        row["clean_base_dark_pixels"] = int(np.count_nonzero(interior &
            (cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY) < dark_limit)))
        row["source_ink_pixels"] = int(np.count_nonzero((source_support > 0) & (plan.domain > 0)))
    return TargetFirstComposition(out, clean, clear, restore, source_layer, {
        "schema": "folirina.target_first_colour.v1", "applied": len(plans),
        "policy": "target_only_mask_then_lama_then_source_alpha",
        "source_background_composited": False, "model": model_diagnostics, "rows": rows,
    })
