from __future__ import annotations

from .direct_patch import SPEC as DIRECT_PATCH
from .mask_replace import SPEC as MASK_REPLACE
from .aligned_overlay_reveal import SPEC as ALIGNED_OVERLAY_REVEAL
from .hybrid import SPEC as HYBRID
from .reletter import SPEC as RELETTER
from .transparent_bubble_reveal import SPEC as TRANSPARENT_BUBBLE_REVEAL
from .legacy_auto import SPEC as LEGACY_AUTO

ACTIVE_MODE_ORDER = ("direct_patch", "mask_replace", "aligned_overlay_reveal", "hybrid", "reletter")
LEGACY_MODE_ORDER = ("auto", "transparent_bubble_reveal")
SUPPORTED_MODE_ORDER = ACTIVE_MODE_ORDER + LEGACY_MODE_ORDER
SUPPORTED_MODES = set(SUPPORTED_MODE_ORDER)
_SPECS = {s.key: s for s in (DIRECT_PATCH, MASK_REPLACE, ALIGNED_OVERLAY_REVEAL, HYBRID, RELETTER, TRANSPARENT_BUBBLE_REVEAL, LEGACY_AUTO)}

def get_mode_spec(mode: str):
    key = str(mode or "").strip().lower()
    if key not in _SPECS:
        raise ValueError(f"Unsupported transfer.mode: {mode}")
    return _SPECS[key]

def active_mode_ui_items():
    return [(get_mode_spec(key).label, key) for key in ACTIVE_MODE_ORDER]

def is_active_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in ACTIVE_MODE_ORDER

def is_legacy_mode(mode: str) -> bool:
    return str(mode or "").strip().lower() in LEGACY_MODE_ORDER

def all_mode_manifests() -> dict:
    return {key: get_mode_spec(key).to_manifest() for key in SUPPORTED_MODE_ORDER}
