import numpy as np
import cv2
from manga_hd_transfer.config import PipelineConfig
from manga_hd_transfer.direct_containers import _colored_text_component_refiner


def test_colored_component_refiner_keeps_text_and_rejects_large_blob():
    cfg = PipelineConfig().mask_replace
    use = np.ones((100, 120), np.uint8) * 255
    raw = np.zeros_like(use)
    # glyph-like components
    cv2.rectangle(raw, (15, 20), (20, 40), 255, -1)
    cv2.rectangle(raw, (30, 22), (36, 42), 255, -1)
    # large decorative blob > 12% of container area
    cv2.rectangle(raw, (60, 10), (115, 60), 255, -1)
    refined, diag = _colored_text_component_refiner(raw, use, cfg)
    assert refined[30, 17] == 255
    assert refined[30, 33] == 255
    assert refined[30, 80] == 0
    assert diag["components_kept"] >= 2
    assert diag["components_rejected"] >= 1
