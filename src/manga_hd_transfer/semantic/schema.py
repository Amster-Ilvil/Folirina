from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class SemanticBlock:
    id: str
    source: str
    raw_label: str
    semantic_type: str
    confidence: float
    bbox: tuple[int, int, int, int]
    polygon: list[tuple[float, float]]
    reading_order: int | None = None
    text: str | None = None
    action: str = "REVIEW"  # PROCESS|IGNORE|REVIEW
    processable: bool = False
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def area(self) -> int:
        x0, y0, x1, y1 = self.bbox
        return max(0, x1 - x0) * max(0, y1 - y0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "raw_label": self.raw_label,
            "semantic_type": self.semantic_type,
            "confidence": float(self.confidence),
            "bbox": [int(v) for v in self.bbox],
            "polygon": [[float(x), float(y)] for x, y in self.polygon],
            "reading_order": self.reading_order,
            "text": self.text,
            "action": self.action,
            "processable": bool(self.processable),
            "meta": dict(self.meta),
        }


@dataclass(slots=True)
class SemanticLayoutResult:
    available: bool
    provider: str
    blocks: list[SemanticBlock] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def process_blocks(self) -> list[SemanticBlock]:
        return [b for b in self.blocks if b.action == "PROCESS"]

    @property
    def ignore_blocks(self) -> list[SemanticBlock]:
        return [b for b in self.blocks if b.action == "IGNORE"]

    @property
    def review_blocks(self) -> list[SemanticBlock]:
        return [b for b in self.blocks if b.action == "REVIEW"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "folirina.semantic_layout.v1",
            "available": bool(self.available),
            "provider": self.provider,
            "blocks": [b.to_dict() for b in self.blocks],
            "diagnostics": dict(self.diagnostics),
        }
