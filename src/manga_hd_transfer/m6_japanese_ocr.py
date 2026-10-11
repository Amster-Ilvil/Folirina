from __future__ import annotations

"""Safe automatic router for Japanese vertical manga OCR on Apple Silicon.

The router does not replace the project's default Paddle route.  It is an
explicit opt-in backend designed around the three model families used by the
project: NDLOCR-Lite for whole-page text geometry, Hayai OCR for crop-aware
Japanese OCR, and the native 48px AR OCR as a conservative crop fallback.
Every model remains individually selectable and failures fall through without
changing the source/target transfer semantics.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .m6_runtime_plan import m6_ocr_backend_order
from .models import TextBlock
from .ocr_base import OCRBackend


@dataclass(slots=True)
class _BackendAttempt:
    name: str
    backend: OCRBackend


class M6JapaneseAutoOCRBackend(OCRBackend):
    supports_crop_recognition = True
    supports_rectified_input = True
    retry_crops = False

    def __init__(self, lang: str, config: Any | None = None) -> None:
        self.config = config
        self.lang = str(lang or "ja")
        self.attempts: list[_BackendAttempt] = []
        self._build_attempts()

    def _build_attempts(self) -> None:
        order = m6_ocr_backend_order(config=self.config, japanese_vertical=True)
        failures: list[str] = []
        from .koharu_ocr import KoharuCropOCRBackend

        for name in order:
            try:
                # NDL is a whole-page engine and is only attempted in recognize().
                if name == "ndlocr_lite":
                    from .ndlocr_lite_runtime import NDLOCRLiteBackend
                    backend = NDLOCRLiteBackend(self.lang, self.config)
                elif name in {"hayai_ocr", "ocr48px"}:
                    backend = KoharuCropOCRBackend(name, self.lang, self.config)
                else:
                    continue
                self.attempts.append(_BackendAttempt(name, backend))
            except Exception as exc:
                failures.append(f"{name}: {exc}")
        self._build_failures = tuple(failures)

    @property
    def available_backends(self) -> tuple[str, ...]:
        return tuple(row.name for row in self.attempts)

    def _record_meta(self, blocks: list[TextBlock], *, selected: str, route: str) -> list[TextBlock]:
        for block in blocks:
            block.meta = dict(block.meta or {})
            block.meta.update({
                "m6_japanese_auto": True,
                "m6_route": route,
                "m6_selected_backend": selected,
                "m6_fallback_candidates": list(self.available_backends),
            })
        return blocks

    def recognize(self, image: np.ndarray, *, image_path=None) -> list[TextBlock]:
        # Whole-page NDL route first because it provides its own geometry.
        for row in self.attempts:
            if row.name != "ndlocr_lite":
                continue
            try:
                blocks = row.backend.recognize(image, image_path=image_path)
                if blocks:
                    return self._record_meta(blocks, selected=row.name, route="whole_page")
            except Exception:
                continue

        # Hayai / 48px need Koharu geometry for a whole page.  They are tried in
        # the same order, and a failed model never mutates the caller image.
        for name in ("hayai_ocr", "ocr48px"):
            for row in self.attempts:
                if row.name != name:
                    continue
                try:
                    blocks = row.backend.recognize(image, image_path=image_path)
                    if blocks:
                        return self._record_meta(blocks, selected=row.name, route="layout_crop")
                except Exception:
                    continue
        return []

    def recognize_region(self, page_image: np.ndarray, bbox: tuple[int, int, int, int], *, image_path=None) -> list[TextBlock]:
        # NDL is intentionally excluded here because its contract is whole-page.
        for name in ("hayai_ocr", "ocr48px"):
            for row in self.attempts:
                if row.name != name or not getattr(row.backend, "supports_crop_recognition", True):
                    continue
                try:
                    blocks = row.backend.recognize_region(page_image, bbox, image_path=image_path)
                    if blocks:
                        return self._record_meta(blocks, selected=row.name, route="region_crop")
                except Exception:
                    continue
        return []


__all__ = ["M6JapaneseAutoOCRBackend"]
