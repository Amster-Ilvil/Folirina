from __future__ import annotations

import pytest

from manga_hd_transfer.config import OCRConfig
from manga_hd_transfer.ocr import build_backend


def test_removed_apple_vision_legacy_backend_names_are_rejected():
    cfg = OCRConfig(backend="apple")
    for name in ("apple_legacy", "apple_vision_legacy", "pyobjc_vision"):
        with pytest.raises(ValueError, match="Unknown OCR backend"):
            build_backend(cfg, "ch", name)


def test_apple_backend_is_live_text_auto_family():
    cfg = OCRConfig(backend="apple")
    backend = build_backend(cfg, "ch", "apple")
    assert backend.__class__.__name__ == "AppleAutoLiveTextBackend"
