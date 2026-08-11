from __future__ import annotations

import cv2
import numpy as np

from manga_hd_transfer.bubbles import build_text_units, detect_seeded_white_bubbles
from manga_hd_transfer.config import BubbleConfig, LetteringConfig
from manga_hd_transfer.lettering import fit_text
from manga_hd_transfer.models import TextBlock


def test_bubble_safe_area_and_chinese_lettering():
    img = np.full((500, 500, 3), 255, np.uint8)
    cv2.ellipse(img, (250,250),(150,100),0,0,360,(0,0,0),5)
    block = TextBlock("b0", [(190,220),(310,220),(310,280),(190,280)], "这是用于排版的中文测试。", .99, reading_order=0)
    cv2.rectangle(img,(200,235),(300,250),(0,0,0),-1)
    bubbles = detect_seeded_white_bubbles(img,[block],BubbleConfig(white_threshold=200,safe_margin_px=10))
    assert len(bubbles)==1
    assert bubbles[0].safe_mask is not None
    units = build_text_units([block],bubbles,"src")
    cfg = LetteringConfig(max_font_size=42,min_font_size=12,orientation="horizontal")
    result = fit_text(img.shape[:2],bubbles[0].safe_mask,units[0],units[0].text,cfg)
    assert result.success
    assert result.coverage_inside_safe >= cfg.min_safe_coverage
    assert result.font_size >= 12
