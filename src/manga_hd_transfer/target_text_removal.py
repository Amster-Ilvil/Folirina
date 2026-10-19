"""TARGET-only removal masks and hard-clipped model inpainting.

The mask does not depend on SOURCE glyphs, edition differences, or whether a
translation was already drawn. Colour-connected container geometry bounds its
dilation; only the lettering holes inside that container are reconstructed.
"""
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np


@dataclass
class TargetRemovalPlan:
    domain: np.ndarray
    mask: np.ndarray
    diagnostics: dict


def coloured_container_removal(target: np.ndarray, seed: np.ndarray) -> TargetRemovalPlan:
    # A short translated line can sit inside a much taller Japanese balloon.
    # Grow only the search window, never the mask or the accepted container.
    last = None
    for family in ("colour", "white"):
        for scale in (0.65, 1.3, 2.6):
            plan = _container_in_window(target, seed, scale, family)
            last = plan
            if plan.diagnostics.get("reason") != "container_reaches_search_boundary":
                break
        if np.any(plan.domain):
            return plan
        if plan.diagnostics.get("reason") not in {"not_coloured_container", "no_connected_container"}:
            return plan
    return last


def _container_in_window(target: np.ndarray, seed: np.ndarray, scale: float, family: str) -> TargetRemovalPlan:
    shape = target.shape[:2]
    empty = np.zeros(shape, np.uint8)
    if seed.shape != shape:
        raise ValueError("TARGET removal seed dimensions differ")
    points = cv2.findNonZero((seed > 0).astype(np.uint8))
    if points is None:
        return TargetRemovalPlan(empty, empty.copy(), {"reason": "empty_seed"})
    x, y, w, h = cv2.boundingRect(points)
    pad = max(32, int(max(w, h) * scale))
    x0, y0 = max(0, x-pad), max(0, y-pad)
    x1, y1 = min(shape[1], x+w+pad), min(shape[0], y+h+pad)
    roi = target[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    local_seed = seed[y0:y1, x0:x1] > 0
    bright_colour = ((hsv[..., 1] >= 60) & (hsv[..., 2] >= 145)) if family == "colour" else (
        (hsv[..., 1] <= 35) & (hsv[..., 2] >= 215))
    sample = bright_colour & local_seed
    if np.count_nonzero(sample) < max(100, np.count_nonzero(local_seed)*0.40):
        return TargetRemovalPlan(empty, empty.copy(), {"reason": "not_coloured_container"})
    if family == "colour":
        hist = np.bincount((hsv[..., 0][sample] // 6).astype(int), minlength=30)
        hue = int(hist.argmax()) * 6 + 3
        distance = np.abs(hsv[..., 0].astype(np.int16)-hue)
        distance = np.minimum(distance, 180-distance)
        colour = (bright_colour & (distance <= 10)).astype(np.uint8)
    else:
        colour = bright_colour.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(colour, 8)
    best = None
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        overlap = int(np.count_nonzero((labels == i) & local_seed))
        if area >= 180 and overlap >= 100 and (best is None or overlap > best[0]):
            best = (overlap, i)
    if best is None:
        return TargetRemovalPlan(empty, empty.copy(), {"reason": "no_connected_container"})
    component = (labels == best[1]).astype(np.uint8)*255
    # Do not turn a truncated crop of a coloured panel into an invented balloon.
    cut = ((x0 > 0 and np.any(component[:, 0])) or
           (x1 < shape[1] and np.any(component[:, -1])) or
           (y0 > 0 and np.any(component[0])) or
           (y1 < shape[0] and np.any(component[-1])))
    if cut:
        return TargetRemovalPlan(empty, empty.copy(), {"reason": "container_reaches_search_boundary"})
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(component)
    cv2.drawContours(filled, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    area = int(cv2.countNonZero(filled))
    if area > shape[0]*shape[1]*0.20:
        # A page-edge starburst can legitimately occupy a third of a page.
        # Admit it only when its connected paper proves a flat saturated fill;
        # size alone must not prevent clearing before translated ink is drawn.
        paper = roi[component > 0]
        median = np.median(paper, axis=0)
        deviation = np.max(np.abs(paper.astype(np.float32) - median), axis=1)
        flat_large = (family == "colour" and area <= shape[0]*shape[1]*0.50
                      and np.count_nonzero(component) >= area*0.90
                      and np.percentile(deviation, 95) <= 15)
        if not flat_large:
            return TargetRemovalPlan(empty, empty.copy(), {"reason": "oversized_container"})
    # Preserve the original contour, rays and nearby line art. Letter holes are
    # recovered without filtering away bold/connected Japanese characters.
    interior = cv2.erode(filled, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    holes = ((filled > 0) & (colour == 0) & (interior > 0)).astype(np.uint8)*255
    n, labs, st, _ = cv2.connectedComponentsWithStats((holes > 0).astype(np.uint8), 8)
    glyphs = np.zeros_like(holes)
    for i in range(1, n):
        if int(st[i, cv2.CC_STAT_AREA]) >= 2:
            glyphs[labs == i] = 255
    if cv2.countNonZero(glyphs) > max(1, area)*0.40:
        return TargetRemovalPlan(empty, empty.copy(), {"reason": "non_textured_paper_not_proven"})
    glyphs = cv2.dilate(glyphs, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    glyphs[interior == 0] = 0
    domain, removal = empty.copy(), empty.copy()
    domain[y0:y1, x0:x1] = filled
    removal[y0:y1, x0:x1] = glyphs
    return TargetRemovalPlan(domain, removal, {
        "reason": "target_colour_container_holes" if family == "colour" else "target_white_container_holes", "domain_pixels": area,
        "clear_pixels": int(cv2.countNonZero(removal)),
        "source_used_for_removal": False, "dilation_px": 2,
    })


def inpaint_target(target: np.ndarray, mask: np.ndarray, callback: Callable) -> tuple[np.ndarray, dict]:
    """Return a clean TARGET base; a model cannot write outside its mask."""
    if mask.shape != target.shape[:2]:
        raise ValueError("TARGET removal mask dimensions differ")
    if not np.any(mask):
        return target.copy(), {"clear_pixels": 0}
    repaired, diagnostics = callback(target, mask)
    if repaired.shape != target.shape or repaired.dtype != np.uint8:
        raise ValueError("Invalid TARGET inpainting result")
    clean = target.copy()
    clean[mask > 0] = repaired[mask > 0]
    return clean, {**diagnostics, "clear_pixels": int(cv2.countNonZero(mask)),
                   "source_used_for_removal": False, "outside_mask_changed_pixels": 0}
