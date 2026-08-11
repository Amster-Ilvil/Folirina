from __future__ import annotations

import cv2
import numpy as np


def make_art_page(width=1000, height=1400):
    img = np.full((height, width, 3), 255, np.uint8)
    # Panels and non-text art create stable registration anchors.
    cv2.rectangle(img, (35, 35), (width - 35, height - 35), (0, 0, 0), 5)
    cv2.line(img, (40, 480), (width - 40, 520), (0, 0, 0), 4)
    cv2.line(img, (500, 40), (540, 500), (0, 0, 0), 3)
    cv2.circle(img, (180, 250), 85, (20, 20, 20), 4)
    cv2.circle(img, (780, 880), 120, (20, 20, 20), 5)
    cv2.rectangle(img, (90, 1010), (410, 1280), (40, 40, 40), 4)
    # Bubble 1 and 2.
    cv2.ellipse(img, (300, 700), (170, 110), 0, 0, 360, (0, 0, 0), 5)
    cv2.ellipse(img, (720, 300), (140, 90), -10, 0, 360, (0, 0, 0), 5)
    # A face-like anchor.
    cv2.circle(img, (650, 1050), 100, (0, 0, 0), 4)
    cv2.circle(img, (620, 1020), 8, (0, 0, 0), -1)
    cv2.circle(img, (680, 1020), 8, (0, 0, 0), -1)
    cv2.ellipse(img, (650, 1060), (45, 30), 0, 20, 160, (0, 0, 0), 3)
    return img


def draw_fake_text(img, bbox, lines=3):
    x0, y0, x1, y1 = map(int, bbox)
    gap = max(4, (y1 - y0) // (lines * 2 + 1))
    y = y0 + gap
    for _ in range(lines):
        cv2.rectangle(img, (x0 + 5, y), (x1 - 5, min(y + gap, y1 - 2)), (0, 0, 0), -1)
        y += gap * 2
