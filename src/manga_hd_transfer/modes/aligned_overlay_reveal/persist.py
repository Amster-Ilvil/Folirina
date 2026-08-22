from __future__ import annotations

from ...pipeline_reveal_persistence import emit_aligned_overlay_page as _emit


def persist_aligned_hole_page(*args, **kwargs):
    return _emit(*args, **kwargs)


__all__ = ["persist_aligned_hole_page"]
