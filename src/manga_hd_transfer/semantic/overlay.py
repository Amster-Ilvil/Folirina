from __future__ import annotations

import cv2
import numpy as np

from .schema import SemanticLayoutResult


def render_semantic_overlay(image: np.ndarray, layout: SemanticLayoutResult) -> np.ndarray:
    out = image.copy()
    for idx, b in enumerate(layout.blocks):
        x0, y0, x1, y1 = b.bbox
        if b.action == "PROCESS": color = (40, 180, 40)
        elif b.action == "IGNORE": color = (40, 40, 230)
        else: color = (0, 180, 230)
        cv2.rectangle(out, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        label = f"{idx}:{b.action} {b.raw_label} {b.confidence:.2f}"
        ty = max(16, y0 - 5)
        cv2.putText(out, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (255,255,255), 3, cv2.LINE_AA)
        cv2.putText(out, label, (x0, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    title = f"Semantic Layout: {layout.provider} | blocks={len(layout.blocks)}"
    cv2.putText(out, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255,255,255), 3, cv2.LINE_AA)
    cv2.putText(out, title, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (20,20,20), 1, cv2.LINE_AA)
    return out
