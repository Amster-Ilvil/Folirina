from __future__ import annotations

import cv2
import numpy as np

from .geometry import transform_points
from .models import BubbleInstance, RegistrationResult, TextUnit, UnitMatch


def _poly_int(poly):
    return np.asarray(poly, dtype=np.float32).round().astype(np.int32).reshape(-1, 1, 2)


def registration_overlay(source: np.ndarray, target: np.ndarray, reg: RegistrationResult) -> np.ndarray:
    tw, th = reg.target_size
    warped = cv2.warpPerspective(source, reg.matrix, (tw, th), flags=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(target, 0.60, warped, 0.40, 0)
    cv2.putText(overlay, f"{reg.method} conf={reg.confidence:.3f} err={reg.reprojection_error:.2f}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)
    return overlay


def structure_overlay(image: np.ndarray, units: list[TextUnit], bubbles: list[BubbleInstance]) -> np.ndarray:
    out = image.copy()
    for bubble in bubbles:
        if len(bubble.polygon) >= 3:
            cv2.polylines(out, [_poly_int(bubble.polygon)], True, (255, 180, 0), 2, cv2.LINE_AA)
        if bubble.safe_mask is not None:
            contours, _ = cv2.findContours((bubble.safe_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (0, 180, 0), 1)
    for unit in units:
        cv2.polylines(out, [_poly_int(unit.polygon)], True, (0, 255, 255), 1, cv2.LINE_AA)
        x, y = map(int, unit.centroid)
        cv2.putText(out, unit.id.split("-")[-1], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
    return out


def matching_overlay(
    target: np.ndarray,
    source_units: list[TextUnit],
    target_units: list[TextUnit],
    matches: list[UnitMatch],
    reg: RegistrationResult,
) -> np.ndarray:
    out = target.copy()
    su = {u.id: u for u in source_units}
    tu = {u.id: u for u in target_units}
    for t in target_units:
        cv2.polylines(out, [_poly_int(t.polygon)], True, (0, 200, 0), 2, cv2.LINE_AA)
    for match in matches:
        s = su.get(match.source_unit_id)
        t = tu.get(match.target_unit_id)
        if s is None or t is None:
            continue
        projected = transform_points(s.polygon, reg.matrix)
        cv2.polylines(out, [_poly_int(projected)], True, (255, 0, 0), 1, cv2.LINE_AA)
        sc = np.mean(np.asarray(projected), axis=0).round().astype(int)
        tc = np.mean(np.asarray(t.polygon), axis=0).round().astype(int)
        color = (0, 200, 0) if match.confidence >= 0.60 and match.relation == "one_to_one" else (0, 0, 255)
        cv2.line(out, tuple(sc), tuple(tc), color, 2, cv2.LINE_AA)
        cv2.putText(out, f"{match.confidence:.2f}/{match.relation}", tuple(tc + np.array([4, -4])), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def mask_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = image.copy()
    layer = image.copy()
    layer[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(out, 1.0 - alpha, layer, alpha, 0)
