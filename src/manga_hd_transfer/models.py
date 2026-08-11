from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .geometry import Polygon, polygon_bbox, polygon_centroid


@dataclass(slots=True)
class PageFingerprint:
    path: str
    index: int
    width: int
    height: int
    dhash: int
    edge_hist: list[float]

    @property
    def aspect(self) -> float:
        return self.width / max(1, self.height)


@dataclass(slots=True)
class PagePair:
    source_path: str
    target_path: str
    source_index: int
    target_index: int
    confidence: float
    score: float
    reasons: list[str] = field(default_factory=list)


@dataclass(slots=True)
class TextBlock:
    id: str
    polygon: Polygon
    text: str = ""
    confidence: float = 1.0
    kind: str = "unknown"
    reading_order: int = 0
    bubble_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return polygon_bbox(self.polygon)

    @property
    def centroid(self) -> tuple[float, float]:
        return polygon_centroid(self.polygon)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BubbleInstance:
    id: str
    polygon: Polygon
    confidence: float = 1.0
    kind: str = "speech"
    block_ids: list[str] = field(default_factory=list)
    mask: np.ndarray | None = field(default=None, repr=False)
    safe_mask: np.ndarray | None = field(default=None, repr=False)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return polygon_bbox(self.polygon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "polygon": self.polygon,
            "confidence": self.confidence,
            "kind": self.kind,
            "block_ids": self.block_ids,
            "meta": self.meta,
        }


@dataclass(slots=True)
class TextUnit:
    id: str
    polygon: Polygon
    block_ids: list[str]
    text: str = ""
    confidence: float = 1.0
    kind: str = "unknown"
    reading_order: int = 0
    bubble_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return polygon_bbox(self.polygon)

    @property
    def centroid(self) -> tuple[float, float]:
        return polygon_centroid(self.polygon)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RegistrationResult:
    matrix: np.ndarray
    method: str
    confidence: float
    inlier_ratio: float
    reprojection_error: float
    spatial_coverage: float
    num_matches: int
    source_size: tuple[int, int]
    target_size: tuple[int, int]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix": np.asarray(self.matrix, dtype=float).tolist(),
            "method": self.method,
            "confidence": self.confidence,
            "inlier_ratio": self.inlier_ratio,
            "reprojection_error": self.reprojection_error,
            "spatial_coverage": self.spatial_coverage,
            "num_matches": self.num_matches,
            "source_size": list(self.source_size),
            "target_size": list(self.target_size),
            "diagnostics": self.diagnostics,
        }


@dataclass(slots=True)
class UnitMatch:
    source_unit_id: str
    target_unit_id: str
    confidence: float
    cost: float
    relation: str = "one_to_one"
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LetteringResult:
    unit_id: str
    text: str
    font_path: str
    font_size: int
    orientation: str
    lines: list[str]
    bbox: tuple[int, int, int, int]
    coverage_inside_safe: float
    success: bool
    reason: str = ""
    text_mask: np.ndarray | None = field(default=None, repr=False)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("text_mask", None)
        return d


@dataclass(slots=True)
class QAItem:
    code: str
    severity: str
    message: str
    unit_id: str | None = None
    value: float | str | None = None
    threshold: float | str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PageProject:
    page_id: str
    pair: PagePair
    registration: RegistrationResult
    source_blocks: list[TextBlock]
    target_blocks: list[TextBlock]
    source_bubbles: list[BubbleInstance]
    target_bubbles: list[BubbleInstance]
    source_units: list[TextUnit]
    target_units: list[TextUnit]
    matches: list[UnitMatch]
    lettering: list[LetteringResult] = field(default_factory=list)
    qa: list[QAItem] = field(default_factory=list)
    artifacts: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "manga_hd_translation_transfer.page.v1",
            "page_id": self.page_id,
            "pair": asdict(self.pair),
            "registration": self.registration.to_dict(),
            "source_blocks": [x.to_dict() for x in self.source_blocks],
            "target_blocks": [x.to_dict() for x in self.target_blocks],
            "source_bubbles": [x.to_dict() for x in self.source_bubbles],
            "target_bubbles": [x.to_dict() for x in self.target_bubbles],
            "source_units": [x.to_dict() for x in self.source_units],
            "target_units": [x.to_dict() for x in self.target_units],
            "matches": [x.to_dict() for x in self.matches],
            "lettering": [x.to_dict() for x in self.lettering],
            "qa": [x.to_dict() for x in self.qa],
            "artifacts": self.artifacts,
            "meta": self.meta,
        }


@dataclass(slots=True)
class BookProject:
    source_dir: str
    target_dir: str
    output_dir: str
    pages: list[PageProject]
    unmatched_source: list[str] = field(default_factory=list)
    unmatched_target: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "manga_hd_translation_transfer.book.v1",
            "source_dir": self.source_dir,
            "target_dir": self.target_dir,
            "output_dir": self.output_dir,
            "pages": [p.to_dict() for p in self.pages],
            "unmatched_source": self.unmatched_source,
            "unmatched_target": self.unmatched_target,
            "meta": self.meta,
        }
