from __future__ import annotations

"""Disk-backed PageProject loading for long-running book jobs.

The authoritative per-page artifact is ``pages/<id>/project.json``.  Long books
should not retain every deserialized PageProject in RAM merely to keep the public
``BookProject.pages`` iteration API working.  This module provides a lazy
Sequence that reconstructs at most one requested page at a time.
"""

import json
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any, overload

import numpy as np

from .models import (
    BubbleInstance,
    LetteringResult,
    PagePair,
    PageProject,
    QAItem,
    RegistrationResult,
    TextBlock,
    TextUnit,
    UnitMatch,
)
from .schema_compat import as_dict, as_dict_rows


def page_project_from_dict(
    payload: dict[str, Any], *, pair_override: PagePair | None = None,
    resume_hit: bool = False,
) -> PageProject:
    """Reconstruct a PageProject from its persisted JSON representation."""
    obj = as_dict(payload)
    regp = as_dict(obj.get("registration"))
    reg = RegistrationResult(
        matrix=np.asarray(regp["matrix"], np.float64),
        method=str(regp["method"]),
        confidence=float(regp["confidence"]),
        inlier_ratio=float(regp["inlier_ratio"]),
        reprojection_error=float(regp["reprojection_error"]),
        spatial_coverage=float(regp["spatial_coverage"]),
        num_matches=int(regp["num_matches"]),
        source_size=tuple(regp["source_size"]),
        target_size=tuple(regp["target_size"]),
        diagnostics=dict(regp.get("diagnostics") or {}),
    )
    if pair_override is None:
        pairp = as_dict(obj.get("pair"))
        pair = PagePair(
            source_path=str(pairp.get("source_path") or ""),
            target_path=str(pairp.get("target_path") or ""),
            source_index=int(pairp.get("source_index") or 0),
            target_index=int(pairp.get("target_index") or 0),
            confidence=float(pairp.get("confidence") or 0.0),
            score=float(pairp.get("score") or 0.0),
            reasons=list(pairp.get("reasons") or []),
        )
    else:
        pair = pair_override

    source_blocks = [TextBlock(**x) for x in as_dict_rows(obj.get("source_blocks"))]
    target_blocks = [TextBlock(**x) for x in as_dict_rows(obj.get("target_blocks"))]
    source_bubbles = [
        BubbleInstance(**{**x, "polygon": [tuple(p) for p in x.get("polygon", [])]})
        for x in as_dict_rows(obj.get("source_bubbles"))
    ]
    target_bubbles = [
        BubbleInstance(**{**x, "polygon": [tuple(p) for p in x.get("polygon", [])]})
        for x in as_dict_rows(obj.get("target_bubbles"))
    ]
    source_units = [TextUnit(**x) for x in as_dict_rows(obj.get("source_units"))]
    target_units = [TextUnit(**x) for x in as_dict_rows(obj.get("target_units"))]
    matches = [UnitMatch(**x) for x in as_dict_rows(obj.get("matches"))]
    lettering = [LetteringResult(**x) for x in as_dict_rows(obj.get("lettering"))]
    qa = [QAItem(**x) for x in as_dict_rows(obj.get("qa"))]
    meta = as_dict(obj.get("meta"))
    if resume_hit:
        meta["batch_resume_hit"] = True
    return PageProject(
        page_id=str(obj["page_id"]), pair=pair, registration=reg,
        source_blocks=source_blocks, target_blocks=target_blocks,
        source_bubbles=source_bubbles, target_bubbles=target_bubbles,
        source_units=source_units, target_units=target_units,
        matches=matches, lettering=lettering, qa=qa,
        artifacts=as_dict(obj.get("artifacts")), meta=meta,
    )


def load_page_project(
    project_path: str | Path, *, pair_override: PagePair | None = None,
    resume_hit: bool = False,
) -> PageProject:
    path = Path(project_path)
    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return page_project_from_dict(payload, pair_override=pair_override, resume_hit=resume_hit)


class DiskBackedPageList(Sequence[PageProject]):
    """Lazy, list-like view over authoritative per-page project JSON files.

    Only paths/pairs are retained in memory.  Indexing or iteration loads one
    PageProject on demand and does not cache it, so a caller that simply walks the
    sequence keeps memory bounded by the largest page project.
    """

    def __init__(self, project_paths: Sequence[str | Path], pairs: Sequence[PagePair] | None = None):
        self._paths = tuple(Path(p) for p in project_paths)
        if pairs is not None and len(pairs) != len(self._paths):
            raise ValueError("pairs length must match project_paths length")
        self._pairs = tuple(pairs) if pairs is not None else None

    @property
    def project_paths(self) -> tuple[Path, ...]:
        return self._paths

    def __len__(self) -> int:
        return len(self._paths)

    @overload
    def __getitem__(self, index: int) -> PageProject: ...

    @overload
    def __getitem__(self, index: slice) -> list[PageProject]: ...

    def __getitem__(self, index: int | slice) -> PageProject | list[PageProject]:
        if isinstance(index, slice):
            indices = range(*index.indices(len(self)))
            return [self[i] for i in indices]
        if index < 0:
            index += len(self._paths)
        if index < 0 or index >= len(self._paths):
            raise IndexError(index)
        pair = self._pairs[index] if self._pairs is not None else None
        return load_page_project(self._paths[index], pair_override=pair)

    def __iter__(self) -> Iterator[PageProject]:
        for idx in range(len(self._paths)):
            yield self[idx]

    def raw_json(self, index: int) -> str:
        return self._paths[index].read_text(encoding="utf-8")


__all__ = ["page_project_from_dict", "load_page_project", "DiskBackedPageList"]
