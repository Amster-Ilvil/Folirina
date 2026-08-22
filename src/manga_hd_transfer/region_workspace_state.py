from __future__ import annotations

"""Qt-free linkage state for the Region Composite workbench.

The workbench has two visual button families (selection-region tools and brush
reveal tools), but only one action may be active at a time.  Selection is an
orthogonal authority and survives tool switches.  OCR recognition is explicitly
bound to the selection that produced it so a new selection can never reuse stale
text/polygons silently.
"""

from dataclasses import dataclass
import hashlib
import json
from typing import Any


def normalized_selection_spec(spec: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(spec or {})
    kind = str(data.get("kind") or "rect").strip().lower()
    box = [int(v) for v in list(data.get("bbox") or [])[:4]]
    points = []
    for p in list(data.get("points") or []):
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            points.append([int(round(float(p[0]))), int(round(float(p[1])))])
    return {
        "schema": "folirina.region_selection.v1",
        "kind": kind,
        "bbox": box,
        "points": points,
        "snapped": bool(data.get("snapped", False)),
    }


def selection_signature(spec: dict[str, Any] | None) -> str:
    data = normalized_selection_spec(spec)
    if len(data["bbox"]) != 4:
        return ""
    raw = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.blake2s(raw, digest_size=12).hexdigest()


@dataclass
class RegionWorkspaceLinkState:
    active_family: str = "region"  # region | brush
    last_region_tool: str = "region_precise_mask"
    brush_mode: str = "transparent"
    selection_sig: str = ""
    ocr_selection_sig: str = ""
    applying: bool = False

    def activate_region(self, tool: str | None) -> str:
        key = str(tool or self.last_region_tool or "region_precise_mask").strip().lower()
        if not key.startswith("region_") or key == "region_brush_reveal":
            key = "region_precise_mask"
        self.active_family = "region"
        self.last_region_tool = key
        return key

    def activate_brush(self, mode: str | None) -> str:
        key = str(mode or self.brush_mode or "transparent").strip().lower()
        if key not in {"transparent", "hole", "restore"}:
            key = "transparent"
        self.active_family = "brush"
        self.brush_mode = key
        return key

    def bind_selection(self, spec: dict[str, Any] | None) -> bool:
        """Bind current selection and report whether OCR authority became stale."""
        new_sig = selection_signature(spec)
        changed = new_sig != self.selection_sig
        self.selection_sig = new_sig
        return bool(changed and self.ocr_selection_sig and self.ocr_selection_sig != new_sig)

    def bind_ocr(self, spec: dict[str, Any] | None) -> str:
        self.ocr_selection_sig = selection_signature(spec)
        return self.ocr_selection_sig

    def clear_ocr(self) -> None:
        self.ocr_selection_sig = ""

    def ocr_matches_current_selection(self) -> bool:
        return bool(self.selection_sig and self.ocr_selection_sig and self.selection_sig == self.ocr_selection_sig)

    def can_apply(self, *, selection_valid: bool, pending_brush_pixels: int) -> bool:
        if self.applying:
            return False
        if self.active_family == "brush":
            return int(pending_brush_pixels) > 0
        return bool(selection_valid)


__all__ = ["RegionWorkspaceLinkState", "selection_signature", "normalized_selection_spec"]
