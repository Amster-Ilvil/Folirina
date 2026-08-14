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


def test_supersampled_lettering_keeps_soft_antialiased_edges():
    from manga_hd_transfer.config import LetteringConfig
    from manga_hd_transfer.lettering import fit_text
    from manga_hd_transfer.models import TextUnit
    shape=(180,220)
    safe=np.zeros(shape,np.uint8); cv2.rectangle(safe,(40,30),(180,150),255,-1)
    unit=TextUnit('u',[(40,30),(180,30),(180,150),(40,150)],[],'高清中文',.99,'speech',0,None,{})
    cfg=LetteringConfig(orientation='horizontal', min_font_size=12, max_font_size=36, supersample_factor=4)
    out=fit_text(shape,safe,unit,'高清中文',cfg)
    assert out.success and out.text_mask is not None
    vals=np.unique(out.text_mask)
    # Lanczos-downsampled alpha should retain subpixel edge values, not binary 0/255 only.
    assert np.any((vals>0) & (vals<255))


def test_fit_text_uses_local_canvas_for_large_page(monkeypatch):
    import manga_hd_transfer.lettering as lettering_mod
    from manga_hd_transfer.models import LetteringResult, TextUnit
    from manga_hd_transfer.config import LetteringConfig

    shape = (1600, 1200)
    safe = np.zeros(shape, np.uint8)
    safe[300:520, 700:880] = 255
    unit = TextUnit(
        id="u-local", polygon=[(700,300),(880,300),(880,520),(700,520)],
        block_ids=["b"], text="测试", confidence=.99, kind="speech"
    )
    captured = {}

    def fake_core(image_shape, safe_mask, unit, text, config):
        captured["shape"] = image_shape
        h, w = image_shape
        mask = np.zeros((h, w), np.uint8)
        mask[10:min(h,40), 10:min(w,60)] = 255
        return LetteringResult(unit.id, text, "font", 18, "horizontal", [text], (10,10,min(w,60),min(h,40)), 1.0, True, text_mask=mask)

    monkeypatch.setattr(lettering_mod, "_fit_text_uncropped", fake_core)
    result = lettering_mod.fit_text(shape, safe, unit, "测试", LetteringConfig(supersample_factor=4))
    assert captured["shape"][0] < shape[0] // 2
    assert captured["shape"][1] < shape[1] // 2
    assert result.text_mask is not None and result.text_mask.shape == shape
    assert result.bbox[0] >= 690 and result.bbox[1] >= 290


def test_source_typography_hint_prevents_short_text_from_ballooning():
    from manga_hd_transfer.models import TextUnit
    shape = (260, 220)
    safe = np.zeros(shape, np.uint8)
    cv2.rectangle(safe, (35, 25), (185, 235), 255, -1)
    unit = TextUnit('u-hint', [(35,25),(185,25),(185,235),(35,235)], [], '无计可施了', .99, 'narration')
    cfg = LetteringConfig(
        orientation='vertical', min_font_size=10, max_font_size=72,
        preferred_font_size=24, preferred_columns=2,
        preferred_font_tolerance_ratio=.15, supersample_factor=1,
    )
    out = fit_text(shape, safe, unit, unit.text, cfg)
    assert out.success
    assert 20 <= out.font_size <= 28
    assert len(out.lines) == 2
