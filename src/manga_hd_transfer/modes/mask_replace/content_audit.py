from __future__ import annotations

"""Renderer-independent content completeness audit and bounded repair.

This module owns the decision layer that verifies whether expected SOURCE ink
survived and TARGET-only ink was removed. It deliberately does not import the
Mask renderer or page pipeline, so the audit can be exercised independently.
"""

from typing import Any

import cv2
import numpy as np

from ...config import MaskReplaceConfig
from .raster_primitives import _expand_safe_write_mask
from .selection_policy import _publication_safety_enabled


def _evaluate_content_completeness(
    rec: Any,
    source_ink: np.ndarray | None,
    target_ink: np.ndarray | None,
    final_image: np.ndarray,
    cfg: MaskReplaceConfig,
    *,
    tolerance_px: int | None = None,
    min_source_coverage: float | None = None,
    max_target_residual: float | None = None,
) -> None:
    """Verify content independently from the fact that a raster write occurred.

    The check is deliberately language/OCR independent.  It asks two local
    questions: (1) did the expected registered source ink survive in the final
    pixels, and (2) did target-only ink disappear?  Overlapping source/target
    strokes are excluded from the residual test so legitimate Chinese ink is not
    mistaken for leftover Japanese.
    """
    if not bool(getattr(cfg, "content_completeness_enabled", True)):
        rec.content_check = "disabled"
        return
    if source_ink is None or source_ink.shape != final_image.shape[:2]:
        rec.content_check = "insufficient_source_ink_evidence"
        return
    src = (source_ink > 0).astype(np.uint8) * 255
    src_count = int(cv2.countNonZero(src))
    min_ink = int(getattr(cfg, "content_completeness_min_ink_pixels", 18))
    if src_count < min_ink:
        rec.content_check = "insufficient_source_ink_evidence"
        return
    tol = max(1, int(tolerance_px if tolerance_px is not None else getattr(cfg, "content_completeness_tolerance_px", 2)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))
    fgray = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
    final_ink = (fgray <= 220).astype(np.uint8) * 255
    final_near = cv2.dilate(final_ink, k)
    coverage = float(np.count_nonzero((src > 0) & (final_near > 0)) / max(1, src_count))

    residual = 0.0
    tgt_count = 0
    if target_ink is not None and target_ink.shape == src.shape:
        tgt = (target_ink > 0).astype(np.uint8) * 255
        tgt_count = int(cv2.countNonZero(tgt))
        if tgt_count >= min_ink:
            src_near = cv2.dilate(src, k)
            target_only = (tgt > 0) & (src_near == 0)
            target_only_count = int(np.count_nonzero(target_only))
            if target_only_count >= max(4, min_ink // 3):
                # A slightly stricter darkness threshold catches real leftover
                # glyph cores while ignoring mild inpaint/halftone texture.
                residual = float(np.count_nonzero(target_only & (fgray <= 205)) / target_only_count)
            else:
                residual = 0.0

    rec.source_ink_coverage = coverage
    rec.target_residual_ratio = residual
    rec.content_check = "checked" if tgt_count >= min_ink else "checked_source_only"
    min_cov = float(min_source_coverage if min_source_coverage is not None else getattr(cfg, "content_completeness_min_source_coverage", 0.90))
    max_res = float(max_target_residual if max_target_residual is not None else getattr(cfg, "content_completeness_max_target_residual", 0.10))
    rec.content_complete = bool(coverage >= min_cov and residual <= max_res)


def _repair_content_region(
    rec: Any,
    rendered: np.ndarray,
    source_image: np.ndarray,
    target_original: np.ndarray,
    current_write_mask: np.ndarray,
    safe_envelope: np.ndarray,
    source_ink: np.ndarray | None,
    target_ink: np.ndarray | None,
    cfg: MaskReplaceConfig,
    *,
    tolerance_px: int | None = None,
    min_source_coverage: float | None = None,
    max_target_residual: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Attempt one bounded OCR-free completeness repair inside a trusted region.

    The repair is intentionally conservative: it can only grow inside the
    existing safe target envelope, it refuses strong target edges, and it clears
    only compact target-only ink that is not near expected source ink. The result
    is re-audited before it can be considered successful.
    """
    if not bool(getattr(cfg, "content_auto_repair_enabled", True)):
        return rendered, current_write_mask, {"enabled": False}
    rec.repair_attempted = True
    before_cov = float(getattr(rec, "source_ink_coverage", 0.0))
    before_res = float(getattr(rec, "target_residual_ratio", 0.0))
    before_check = str(getattr(rec, "content_check", "not_checked"))
    before_complete = bool(getattr(rec, "content_complete", False))
    out = rendered.copy()
    repair_mask, growth = _expand_safe_write_mask(
        current_write_mask, safe_envelope, source_image, target_original, cfg,
        max_px=int(getattr(cfg, "content_auto_repair_max_growth_px", 5)),
    )
    new_pixels = (repair_mask > 0) & (current_write_mask == 0)
    if np.any(new_pixels):
        out[new_pixels] = source_image[new_pixels]

    residual_clear = np.zeros_like(repair_mask)
    if source_ink is not None and target_ink is not None and source_ink.shape == repair_mask.shape and target_ink.shape == repair_mask.shape:
        tol = max(1, int(tolerance_px if tolerance_px is not None else getattr(cfg, "content_completeness_tolerance_px", 2)))
        src_near = cv2.dilate((source_ink > 0).astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tol * 2 + 1, tol * 2 + 1))) > 0
        target_only = (target_ink > 0) & (~src_near) & (safe_envelope > 0)
        if np.any(target_only):
            # ``target_ink`` is already compact-component filtered and the safe
            # envelope excludes the bubble border. Do not reject it for being a
            # strong edge: Japanese glyph cores are strong edges by definition.
            residual_clear[target_only] = 255
            grow = max(0, int(getattr(cfg, "content_auto_repair_residual_dilate_px", 1)))
            if grow > 0 and cv2.countNonZero(residual_clear) > 0:
                residual_clear = cv2.dilate(residual_clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1, grow * 2 + 1)))
                residual_clear[safe_envelope == 0] = 0
            if cv2.countNonZero(residual_clear) > 0:
                out = cv2.inpaint(out, residual_clear, float(getattr(cfg, "content_auto_repair_inpaint_radius", 2.5)), cv2.INPAINT_TELEA)
                # Restore expected source content wherever the expanded write mask allows it.
                repaint = (repair_mask > 0) & (source_ink > 0)
                if np.any(repaint):
                    out[repaint] = source_image[repaint]

    _evaluate_content_completeness(
        rec, source_ink, target_ink, out, cfg,
        tolerance_px=tolerance_px,
        min_source_coverage=min_source_coverage,
        max_target_residual=max_target_residual,
    )
    if cv2.countNonZero(residual_clear) > 0:
        repair_mask = np.maximum(repair_mask, residual_clear)
    after_cov = float(getattr(rec, "source_ink_coverage", 0.0))
    after_res = float(getattr(rec, "target_residual_ratio", 0.0))
    gain = (after_cov - before_cov) + (before_res - after_res)
    min_gain = float(getattr(cfg, "content_auto_repair_min_gain", 0.01))
    improved = bool(rec.content_complete or gain >= min_gain)
    rec.repair_succeeded = bool(improved)
    if not improved:
        rec.source_ink_coverage = before_cov
        rec.target_residual_ratio = before_res
        rec.content_check = before_check
        rec.content_complete = before_complete
    return out if improved else rendered, repair_mask if improved else current_write_mask, {
        "enabled": True,
        "growth": growth,
        "residual_clear_pixels": int(cv2.countNonZero(residual_clear)),
        "before_source_coverage": before_cov,
        "after_source_coverage": after_cov,
        "before_target_residual": before_res,
        "after_target_residual": after_res,
        "gain": float(gain),
        "improved": bool(improved),
        "content_complete": bool(rec.content_complete),
    }


def finalize_transfer_records(records: list[Any], cfg: MaskReplaceConfig) -> None:
    """Assign one auditable SAFE/REVIEW/REJECT state to every transfer record."""
    safe_conf = float(getattr(cfg, "triage_safe_confidence", 0.82))
    reject_conf = float(getattr(cfg, "triage_reject_confidence", 0.55))
    for rec in records:
        if not bool(getattr(rec, "applied", False)):
            rec.triage_state = "REJECT"
            continue
        if not _publication_safety_enabled(cfg):
            # Diagnostics remain on the record, but they no longer block or
            # downgrade a successfully written region.
            rec.triage_state = "SAFE"
            continue
        if float(getattr(rec, "confidence", 0.0)) < reject_conf:
            rec.triage_state = "REJECT"
            continue
        check = str(getattr(rec, "content_check", "not_checked") or "not_checked")
        verified = check.startswith("checked") and bool(getattr(rec, "content_complete", False))
        if verified and float(getattr(rec, "confidence", 0.0)) >= safe_conf and not bool(getattr(rec, "review_required", False)):
            rec.triage_state = "SAFE"
        else:
            rec.triage_state = "REVIEW"



__all__ = ["_evaluate_content_completeness", "_repair_content_region", "finalize_transfer_records"]
