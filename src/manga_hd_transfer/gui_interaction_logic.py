from __future__ import annotations

"""Headless interaction helpers: toast timing/placement and window-geometry sanity."""


def toast_duration_ms(text: str, kind: str = "success") -> int:
    ms = 1800 + 55 * len(str(text or ""))
    if kind == "warning":
        ms += 1200
    return min(7000, max(2200, ms))


def toast_origin(parent_w: int, parent_h: int, w: int, h: int, bottom: int = 52) -> tuple[int, int]:
    return max(0, (int(parent_w) - int(w)) // 2), max(0, int(parent_h) - int(h) - int(bottom))


def geometry_fits(win_w: int, win_h: int, avail_w: int, avail_h: int, min_visible: int = 160) -> bool:
    """A saved window rectangle is only restored when it is not absurdly larger than the screen."""
    if avail_w <= 0 or avail_h <= 0:
        return True
    return win_w >= min_visible and win_h >= min_visible and win_w <= avail_w * 1.05 and win_h <= avail_h * 1.05


__all__ = ["toast_duration_ms", "toast_origin", "geometry_fits"]
