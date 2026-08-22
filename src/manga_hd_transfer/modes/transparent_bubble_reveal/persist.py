from __future__ import annotations

from ...pipeline_reveal_persistence import emit_transparent_bubble_page as _emit


def persist_transparent_page(*args, **kwargs):
    return _emit(*args, **kwargs)


__all__ = ["persist_transparent_page"]
