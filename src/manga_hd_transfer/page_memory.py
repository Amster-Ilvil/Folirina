from __future__ import annotations

"""Release page-local heavy arrays once a page has been durably published."""

import numpy as np


def release_page_heavy_arrays(project) -> int:
    """Drop non-serialized full-resolution arrays from a completed PageProject.

    Bubble masks and lettering masks are persisted through artifacts/cache when
    required, and PageProject.to_dict intentionally omits them.  Keeping them in
    BookProject.pages makes batch memory scale with page count for no benefit.
    Returns the approximate number of ndarray bytes released.
    """
    released = 0
    seen: set[int] = set()

    def drop(obj, attr: str) -> None:
        nonlocal released
        arr = getattr(obj, attr, None)
        if isinstance(arr, np.ndarray):
            ident = id(arr)
            if ident not in seen:
                released += int(arr.nbytes)
                seen.add(ident)
            setattr(obj, attr, None)

    for bubble in list(getattr(project, "source_bubbles", []) or []) + list(getattr(project, "target_bubbles", []) or []):
        drop(bubble, "mask")
        drop(bubble, "safe_mask")
    for lettering in list(getattr(project, "lettering", []) or []):
        drop(lettering, "text_mask")
    return released
