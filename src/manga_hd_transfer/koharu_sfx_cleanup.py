from __future__ import annotations

"""Conservative flat-background cleanup for Koharu foreground text masks.

Koharu supplies semantic SFX instances and their disconnected foreground masks,
    but it does not identify the script/language. In this workflow TARGET is the
    Japanese edition. This stage therefore does not infer text or invent a
    background: it only clears high-confidence text/SFX foreground when the
    visible local ring is demonstrably flat, and leaves textured/color artwork
    for a real inpainting model or human review.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable
import subprocess

import cv2
import numpy as np

from .koharu_mask_codec import decode_mask
from .text_ink_equivalence import equivalent_registered_ink


@dataclass(slots=True)
class KoharuSfxCleanupResult:
    rendered: np.ndarray
    editable_base: np.ndarray
    clear_mask: np.ndarray
    diagnostics: dict[str, Any] = field(default_factory=dict)


def clean_koharu_text_on_flat_backgrounds(
    target: np.ndarray,
    rendered: np.ndarray,
    editable_base: np.ndarray,
    sfx_blocks: Iterable[Any],
    *,
    confidence_threshold: float = 0.80,
    dilate_px: int = 3,
    ring_radius: int = 8,
    spread_limit: float = 10.0,
    inlier_tolerance: float = 18.0,
    min_inlier_ratio: float = 0.90,
    min_ring_samples: int = 24,
    unchanged_tolerance: int = 2,
    protected_mask: np.ndarray | None = None,
    aligned_source: np.ndarray | None = None,
    inpaint_callback: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, dict[str, Any]]] | None = None,
) -> KoharuSfxCleanupResult:
    """Clear eligible Koharu text/SFX pixels without touching translated pixels.

    Only metadata emitted by Koharu's TARGET layout evidence is accepted.  The
    candidate is expanded by a few pixels to remove antialiased remnants, then
    intersected with pixels that are still unchanged from TARGET.  Textured
    rings are explicitly deferred rather than sent through Telea/NS inpainting.
    """
    if target.ndim != 3 or rendered.shape != target.shape or editable_base.shape != target.shape:
        raise ValueError("Koharu SFX cleanup expects equal HxWx3 TARGET/render/base arrays")
    h, w = target.shape[:2]
    empty = np.zeros((h, w), np.uint8)
    rows: list[dict[str, Any]] = []
    candidates = 0
    applied = 0
    needs_model_inpainting = 0
    unresolved_foreground_pixels = 0
    model_mask = empty.copy()
    model_rows: list[dict[str, Any]] = []
    model_error = ""
    model_diagnostics: dict[str, Any] = {}
    out = rendered
    base = editable_base
    combined = empty
    unchanged_pixels = np.max(
        np.abs(rendered.astype(np.int16) - target.astype(np.int16)), axis=2
    ) <= max(0, int(unchanged_tolerance))
    if protected_mask is not None:
        if protected_mask.shape != (h, w):
            raise ValueError("Protected SOURCE ink mask must match TARGET dimensions")
        unchanged_pixels &= protected_mask == 0

    for block in sfx_blocks or ():
        meta = dict(getattr(block, "meta", {}) or {})
        label = str(meta.get("layout_label") or "").strip().lower()
        backend = str(meta.get("backend") or "").strip().lower()
        if backend != "koharu_layout" or label not in {"text", "sfx"} or not bool(meta.get("layout_evidence")):
            continue
        candidates += 1
        row: dict[str, Any] = {
            "id": str(getattr(block, "id", "")),
            "layout_label": label,
            "confidence": round(float(getattr(block, "confidence", 0.0) or 0.0), 4),
        }
        if row["confidence"] < float(confidence_threshold):
            row["status"] = "skipped_low_confidence"
            rows.append(row)
            continue
        payload = meta.get("foreground_mask")
        if not isinstance(payload, dict):
            row["status"] = "skipped_missing_foreground_mask"
            rows.append(row)
            continue
        try:
            native = decode_mask(payload, (h, w))
        except (KeyError, TypeError, ValueError) as exc:
            row["status"] = "skipped_invalid_foreground_mask"
            row["detail"] = str(exc)[:240]
            rows.append(row)
            continue
        if cv2.countNonZero(native) <= 0:
            row["status"] = "skipped_empty_foreground_mask"
            rows.append(row)
            continue
        if equivalent_registered_ink(aligned_source, target, native):
            row["status"] = "preserved_equivalent_source_lettering"
            rows.append(row)
            continue
        source_area = int(cv2.countNonZero(native))
        if source_area > int(h * w * 0.05):
            row["status"] = "skipped_oversized_foreground_mask"
            row["foreground_pixels"] = source_area
            rows.append(row)
            continue
        radius = max(0, int(dilate_px))
        if radius:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            expanded = cv2.dilate((native > 0).astype(np.uint8), kernel)
        else:
            expanded = (native > 0).astype(np.uint8)

        # The ring is measured from the current rendered image, but only where
        # no transfer has changed TARGET. This preserves the current local color
        # while preventing a nearby Chinese glyph from becoming the fill sample.
        ring_radius_px = max(1, int(ring_radius))
        ring_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * ring_radius_px + 1, 2 * ring_radius_px + 1)
        )
        ring = (cv2.dilate(expanded, ring_kernel) > 0) & (expanded == 0) & unchanged_pixels
        ring_pixels = rendered[ring]
        if int(ring_pixels.shape[0]) < int(min_ring_samples):
            row.update({"status": "skipped_background_ring_too_small", "ring_samples": int(ring_pixels.shape[0])})
            rows.append(row)
            continue
        p10, p90 = np.percentile(ring_pixels.astype(np.float32), [10.0, 90.0], axis=0)
        spread = float(np.max(p90 - p10))
        fill = np.median(ring_pixels, axis=0).astype(np.uint8)
        inliers = np.max(np.abs(ring_pixels.astype(np.int16) - fill.astype(np.int16)), axis=1) <= float(inlier_tolerance)
        inlier_ratio = float(np.count_nonzero(inliers) / max(1, ring_pixels.shape[0]))
        row.update({
            "ring_samples": int(ring_pixels.shape[0]),
            "ring_p10_p90_spread": round(spread, 3),
            "ring_inlier_ratio": round(inlier_ratio, 4),
            "fill_bgr": fill.tolist(),
        })
        if spread > float(spread_limit) or inlier_ratio < float(min_inlier_ratio):
            row["status"] = "skipped_non_uniform_background"
            unresolved = int(np.count_nonzero((expanded > 0) & unchanged_pixels))
            row["unresolved_foreground_pixels"] = unresolved
            row["recommended_action"] = "model_inpainting_or_manual_review"
            needs_model_inpainting += 1
            unresolved_foreground_pixels += unresolved
            eligible_model_mask = (expanded > 0) & unchanged_pixels
            model_mask[eligible_model_mask] = 255
            model_rows.append(row)
            rows.append(row)
            continue

        eligible = (expanded > 0) & unchanged_pixels
        # A neutral paper fill never owns coloured background pixels in a
        # speech/text corridor. Detector masks may include the blue fringe of
        # a burst balloon; filling that fringe white damages the gradient.
        if label == "text" and int(fill.max()) - int(fill.min()) <= 8:
            eligible &= cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[..., 1] <= 8
        count = int(np.count_nonzero(eligible))
        if count <= 0:
            row["status"] = "skipped_no_unmodified_target_pixels"
            rows.append(row)
            continue
        if out is rendered:
            out = rendered.copy()
            base = editable_base.copy()
            combined = empty.copy()
        out[eligible] = fill
        base[eligible] = fill
        combined[eligible] = 255
        applied += 1
        row.update({
            "status": "applied_flat_background_fill",
            "cleared_pixels": count,
            "protected_rendered_pixels": int(np.count_nonzero(expanded > 0)) - count,
        })
        rows.append(row)

    flat_applied = applied
    model_applied = 0
    if inpaint_callback is not None and cv2.countNonZero(model_mask) > 0:
        try:
            model_image, model_diagnostics = inpaint_callback(out, model_mask)
            if not isinstance(model_image, np.ndarray) or model_image.shape != target.shape or model_image.dtype != np.uint8:
                raise ValueError("LaMa Manga runner must return a uint8 image matching TARGET dimensions")
            if out is rendered:
                out = rendered.copy()
                base = editable_base.copy()
                combined = empty.copy()
            allowed = model_mask > 0
            out[allowed] = model_image[allowed]
            base[allowed] = model_image[allowed]
            combined[allowed] = 255
            model_applied = int(np.count_nonzero(allowed))
            applied += len(model_rows)
            for row in model_rows:
                row["status"] = "applied_lama_manga"
                row["cleared_pixels"] = int(row.get("unresolved_foreground_pixels", 0) or 0)
                row["inpainting_backend"] = "lama_manga"
            needs_model_inpainting = 0
            unresolved_foreground_pixels = 0
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            model_error = str(exc)[:800]
            for row in model_rows:
                row["inpainting_error"] = model_error

    return KoharuSfxCleanupResult(
        rendered=out,
        editable_base=base,
        clear_mask=combined,
        diagnostics={
            "schema": "folirina.koharu_text_cleanup.v1",
            "policy": "high_confidence_foreground_mask_flat_ring_only",
            "confidence_threshold": float(confidence_threshold),
            "candidates": candidates,
            "applied": applied,
            "flat_applied": flat_applied,
            "model_inpainting_candidates": len(model_rows),
            "model_inpainting_applied_pixels": model_applied,
            "model_inpainting_diagnostics": model_diagnostics,
            "model_inpainting_error": model_error,
            "deferred": int(sum(1 for row in rows if str(row.get("status", "")).startswith("skipped_"))),
            "needs_model_inpainting": needs_model_inpainting,
            "unresolved_foreground_pixels": unresolved_foreground_pixels,
            "rows": rows,
        },
    )


__all__ = ["KoharuSfxCleanupResult", "clean_koharu_text_on_flat_backgrounds"]
