from __future__ import annotations

"""Pure geometry helpers for the desktop whole-page responsive shell.

The business UI must never be laid out in a canvas smaller than the geometry
known to keep its controls readable and non-overlapping.  Instead, the logical
canvas may *grow* to match the host window aspect ratio and the outer Qt view
applies one uniform transform to fit it into the real viewport.

Keeping the math Qt-free makes this contract testable in core-only CI where
PySide6 is intentionally not installed.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class CanvasFit:
    logical_width: int
    logical_height: int
    scale: float



def fit_scale_to_canvas(
    viewport_width: int,
    viewport_height: int,
    logical_width: int,
    logical_height: int,
    *,
    maximum_scale: float | None = None,
) -> float:
    """Fit an already-laid-out logical canvas into the current viewport.

    This helper deliberately does *not* change the logical canvas geometry.
    The desktop view uses it while the user is actively dragging/resizing the
    native window so the expensive QWidget layout tree is not recomputed on
    every resize event.  Once the resize settles, :func:`fit_logical_canvas`
    is used once to commit the final aspect-matched logical geometry.
    """

    vw = max(1, int(viewport_width))
    vh = max(1, int(viewport_height))
    lw = max(1, int(logical_width))
    lh = max(1, int(logical_height))
    scale = min(vw / lw, vh / lh)
    if maximum_scale is not None:
        scale = min(scale, max(0.001, float(maximum_scale)))
    return max(0.001, float(scale))

def fit_logical_canvas(
    viewport_width: int,
    viewport_height: int,
    *,
    minimum_width: int = 1480,
    minimum_height: int = 960,
    maximum_scale: float | None = None,
) -> CanvasFit:
    """Return an aspect-matched logical canvas plus its uniform display scale.

    The logical canvas is never smaller than ``minimum_width x minimum_height``.
    To avoid letterboxing it grows only along the dimension required to match
    the viewport aspect ratio.  Consequently Qt layouts may receive *more*
    logical room on unusual screens but are never squeezed below the safe
    design geometry.  The entire logical canvas can then be rendered using one
    uniform transform.
    """

    vw = max(1, int(viewport_width))
    vh = max(1, int(viewport_height))
    mw = max(1, int(minimum_width))
    mh = max(1, int(minimum_height))

    viewport_aspect = vw / vh
    minimum_aspect = mw / mh
    if viewport_aspect >= minimum_aspect:
        logical_h = mh
        logical_w = max(mw, int(math.ceil(mh * viewport_aspect)))
    else:
        logical_w = mw
        logical_h = max(mh, int(math.ceil(mw / viewport_aspect)))

    scale = min(vw / logical_w, vh / logical_h)
    if maximum_scale is not None:
        scale = min(scale, max(0.001, float(maximum_scale)))
    return CanvasFit(logical_w, logical_h, max(0.001, float(scale)))
