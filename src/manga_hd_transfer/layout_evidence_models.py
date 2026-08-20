from __future__ import annotations

"""Renderer-neutral layout-evidence domain models.

This module intentionally owns no cache/runtime dependencies.  Both the cache
layer and the Koharu-layout adapter import these dataclasses, which keeps the
domain model reusable without a cache <-> layout adapter import cycle.
"""

from dataclasses import dataclass, field
from typing import Any, Iterable

import cv2
import numpy as np

from .models import BubbleInstance, TextBlock

_LABEL_ALIASES = {
    "onomatopoeia": "sfx",
    "sound_effect": "sfx",
    "sound-effect": "sfx",
    "speech_bubble": "bubble",
}


def normalize_layout_label(label: str) -> str:
    key = str(label or "").strip().lower().replace("-", "_")
    return _LABEL_ALIASES.get(key, key)


@dataclass(slots=True)
class LayoutEvidenceItem:
    label: str
    confidence: float
    polygon: list[tuple[float, float]]
    mask: np.ndarray
    box: tuple[int, int, int, int]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayoutAuthorityDecision:
    state: str  # ALLOW | PROTECT | UNKNOWN
    reason: str
    bubble_overlap: float = 0.0
    text_overlap: float = 0.0
    sfx_overlap: float = 0.0
    panel_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "bubble_overlap": round(float(self.bubble_overlap), 4),
            "text_overlap": round(float(self.text_overlap), 4),
            "sfx_overlap": round(float(self.sfx_overlap), 4),
            "panel_overlap": round(float(self.panel_overlap), 4),
        }


@dataclass(slots=True)
class LayoutEvidence:
    available: bool
    backend: str
    items: list[LayoutEvidenceItem] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_label(self, *labels: str) -> list[LayoutEvidenceItem]:
        wanted = {normalize_layout_label(label) for label in labels}
        return [row for row in self.items if row.label in wanted]

    def combined_mask(self, labels: Iterable[str], *, dilate_px: int = 0) -> np.ndarray:
        shape = self.diagnostics.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("LayoutEvidence diagnostics missing shape")
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros((h, w), np.uint8)
        wanted = {normalize_layout_label(label) for label in labels}
        for row in self.items:
            if row.label not in wanted:
                continue
            if row.mask.shape != out.shape:
                mask = cv2.resize(row.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = row.mask
            out = np.maximum(out, mask)
        radius = max(0, int(dilate_px))
        if radius > 0 and cv2.countNonZero(out) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            out = cv2.dilate(out, kernel)
        return out

    def text_blocks(
        self, *, include_sfx: bool = True, backend_name: str = "koharu_layout",
        source_only: bool = False, target_only: bool = False,
    ) -> list[TextBlock]:
        out: list[TextBlock] = []
        for idx, row in enumerate(self.items):
            if row.label == "text":
                kind = "speech"
            elif include_sfx and row.label == "sfx":
                kind = "sfx"
            else:
                continue
            meta = dict(row.meta)
            meta.update({
                "backend": backend_name,
                "layout_label": row.label,
                "layout_evidence": True,
                "text_seed": True,
            })
            if source_only:
                meta["source_only"] = True
                meta.pop("target_only", None)
            if target_only:
                meta["target_only"] = True
                meta.pop("source_only", None)
            out.append(TextBlock(
                id=f"koharu-text-{idx:04d}",
                polygon=list(row.polygon),
                text="",
                confidence=float(row.confidence),
                kind=kind,
                reading_order=idx,
                meta=meta,
            ))
        return out

    def bubble_instances(self, *, backend_name: str = "koharu_layout", target_only: bool = False, source_only: bool = False) -> list[BubbleInstance]:
        out: list[BubbleInstance] = []
        for idx, row in enumerate(self.items):
            if row.label != "bubble":
                continue
            mask = row.mask.copy()
            safe = mask.copy()
            meta = dict(row.meta)
            meta.update({
                "backend": backend_name,
                "layout_label": row.label,
                "layout_evidence": True,
            })
            if target_only:
                meta["target_only"] = True
                meta.pop("source_only", None)
            if source_only:
                meta["source_only"] = True
                meta.pop("target_only", None)
            out.append(BubbleInstance(
                id=f"koharu-bubble-{idx:04d}",
                polygon=list(row.polygon),
                confidence=float(row.confidence),
                kind="speech",
                block_ids=[],
                mask=mask,
                safe_mask=safe,
                meta=meta,
            ))
        return out

    def authority_map(self, *, allow_dilate_px: int = 0) -> np.ndarray:
        """Return the page semantic authority map (0 UNKNOWN, 1 ALLOW, 2 PROTECT).

        ALLOW has priority over panel protection.  The map is intentionally a
        first-layer semantic prior, not a replacement for candidate-specific
        geometry checks: fallback detectors may operate in UNKNOWN but must not
        destructively override PROTECT.
        """
        shape = self.diagnostics.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("LayoutEvidence diagnostics missing shape")
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros((h, w), np.uint8)
        if not self.available:
            return out
        allow = self.combined_mask(("bubble", "text", "sfx"), dilate_px=allow_dilate_px)
        panel = self.combined_mask(("panel",), dilate_px=0)
        out[panel > 0] = 2
        out[allow > 0] = 1
        return out



__all__ = ["LayoutEvidenceItem", "LayoutAuthorityDecision", "LayoutEvidence", "normalize_layout_label"]
