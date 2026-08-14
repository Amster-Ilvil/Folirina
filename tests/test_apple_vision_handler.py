"""Regression tombstone for the removed PyObjC Vision legacy handler.

v0.8.9+ deliberately routes Apple OCR through VisionKit Live Text / Shortcuts.
The old VNImageRequestHandler backend must stay absent so a stale import cannot
silently reintroduce the macOS NSDictionary/NSInvalidArgumentException crash.
"""

import manga_hd_transfer.ocr as ocr
from manga_hd_transfer.config import OCRConfig


def test_legacy_vision_handler_stays_removed():
    assert not hasattr(ocr, "AppleVisionOCRBackend")


def test_apple_name_resolves_to_live_text_auto_family():
    backend = ocr.build_backend(OCRConfig(backend="apple"), "ch", "apple")
    assert backend.__class__.__name__ == "AppleAutoLiveTextBackend"
