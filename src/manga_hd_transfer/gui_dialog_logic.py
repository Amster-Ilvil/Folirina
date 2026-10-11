from __future__ import annotations

"""Headless rules for polishing every dialog window (screen fit + safe default button)."""

import re
from typing import Sequence


def clamp_dialog_min(min_w: int, min_h: int, avail_w: int, avail_h: int, ratio: float = 0.92) -> tuple[int, int]:
    """Never let a dialog's minimum size exceed ``ratio`` of the usable screen (small laptops, scaled displays)."""
    limit_w, limit_h = int(avail_w * ratio), int(avail_h * ratio)
    return (min(int(min_w), limit_w) if limit_w > 0 else int(min_w),
            min(int(min_h), limit_h) if limit_h > 0 else int(min_h))


_PRIMARY_PREFIXES = ("保存", "确定", "确认", "应用", "完成", "关闭", "好的")
_LEADING_JUNK = re.compile(r"^[^\w\u3040-\u30ff\u3400-\u9fff]+")


def pick_default_button(texts: Sequence[str], has_text_input: bool, max_buttons: int = 6) -> int | None:
    """Enter should confirm simple dialogs, but must never commit inside editors.

    No default when the dialog has text inputs, or has many buttons (canvas editors with brush/region tools),
    so a stray Enter cannot apply a tool or overwrite a mask.
    """
    if has_text_input or len(texts) > max_buttons:
        return None
    for index, raw in enumerate(texts):
        text = _LEADING_JUNK.sub("", str(raw or "")).strip()
        if text.startswith(_PRIMARY_PREFIXES):
            return index
    return None


__all__ = ["clamp_dialog_min", "pick_default_button"]
