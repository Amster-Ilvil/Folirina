from __future__ import annotations

"""Minimal OCR backend protocol shared by OCR implementations.

Keeping the abstract backend outside ``ocr.py`` lets crop recognizers depend on
the protocol without creating a backend-factory import cycle.
"""

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from .models import TextBlock


class OCRBackend(ABC):
    supports_crop_recognition: bool = True
    supports_region_query: bool = False
    supports_rectified_input: bool = True
    retry_crops: bool = False

    @abstractmethod
    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        raise NotImplementedError

    def recognize_region(
        self,
        page_image: np.ndarray,
        bbox: tuple[int, int, int, int],
        *,
        image_path: str | Path | None = None,
    ) -> list[TextBlock]:
        if not self.supports_crop_recognition:
            raise RuntimeError(f"{type(self).__name__} 不支持局部 OCR。")
        x0, y0, x1, y1 = [int(v) for v in bbox]
        crop = page_image[max(0, y0):max(0, y1), max(0, x0):max(0, x1)]
        return self.recognize(crop, image_path=None)


__all__ = ["OCRBackend"]
