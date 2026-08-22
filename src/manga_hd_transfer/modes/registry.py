from __future__ import annotations

import re

from .direct_patch import SPEC as DIRECT_PATCH
from .mask_replace import SPEC as MASK_REPLACE
from .aligned_overlay_reveal import SPEC as ALIGNED_OVERLAY_REVEAL
from .hybrid import SPEC as HYBRID
from .reletter import SPEC as RELETTER
from .transparent_bubble_reveal import SPEC as TRANSPARENT_BUBBLE_REVEAL
from .legacy_auto import SPEC as LEGACY_AUTO

ACTIVE_MODE_ORDER = ("direct_patch", "mask_replace", "aligned_overlay_reveal", "transparent_bubble_reveal", "hybrid", "reletter")
LEGACY_MODE_ORDER = ("auto",)
SUPPORTED_MODE_ORDER = ACTIVE_MODE_ORDER + LEGACY_MODE_ORDER
SUPPORTED_MODES = set(SUPPORTED_MODE_ORDER)
_SPECS = {s.key: s for s in (DIRECT_PATCH, MASK_REPLACE, ALIGNED_OVERLAY_REVEAL, HYBRID, RELETTER, TRANSPARENT_BUBBLE_REVEAL, LEGACY_AUTO)}


def compact_mode_ui_label(label: str) -> str:
    """Return route-selector copy without parenthetical implementation notes."""
    value = re.sub(r"\s*[（(][^）)]*[）)]\s*", " ", str(label or ""))
    return " ".join(value.split()).strip()

def get_mode_spec(mode: str):
    key = str(mode or "").strip().lower()
    if key not in _SPECS:
        raise ValueError(f"Unsupported transfer.mode: {mode}")
    return _SPECS[key]

_ACTIVE_UI_LABELS = {
    "direct_patch": "直接贴图 · 无边框内层贴图",
    "mask_replace": "精准蒙版 · 原字保真 / 拍照边缘保护",
    "aligned_overlay_reveal": "整页对齐挖孔显中文",
    "transparent_bubble_reveal": "整页对齐透明显中文",
    "hybrid": "精准蒙版+OCR",
    "reletter": "OCR重排",
}

def active_mode_ui_items():
    # Renderer/spec labels are immutable diagnostic contracts. The selector uses
    # deliberately short product copy and never exposes implementation notes such
    # as “SOURCE 在上 / TARGET 在下” or “0 OCR” in parentheses.
    return [(_ACTIVE_UI_LABELS.get(key, compact_mode_ui_label(get_mode_spec(key).label)), key) for key in ACTIVE_MODE_ORDER]

def is_active_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in ACTIVE_MODE_ORDER

def is_legacy_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in LEGACY_MODE_ORDER

def all_mode_manifests() -> dict:
    return {key: get_mode_spec(key).to_manifest() for key in SUPPORTED_MODE_ORDER}
