from __future__ import annotations

"""Small side-effect-free desktop platform helpers."""

import platform
import sys


def platform_family(platform_name: str | None = None) -> str:
    value = str(platform_name or platform.system()).strip().lower()
    if value in {"darwin", "mac", "macos"}:
        return "macos"
    if value.startswith("win"):
        return "windows"
    if value == "linux":
        return "linux"
    return value or "unknown"


def desktop_platform_badge(platform_name: str | None = None) -> str:
    family = platform_family(platform_name)
    if family == "macos":
        return "MAC · MPS/CPU"
    if family == "windows":
        return "WINDOWS · CUDA/CPU"
    if family == "linux":
        return "LINUX · CUDA/CPU"
    return f"{(platform_name or sys.platform).upper()} · CPU"


def desktop_platform_summary(platform_name: str | None = None) -> str:
    family = platform_family(platform_name)
    if family == "macos":
        return "macOS：Apple Live Text / MPS（可选）/ CPU"
    if family == "windows":
        return "Windows：PaddleOCR / CUDA（可选）/ CPU"
    if family == "linux":
        return "Linux：PaddleOCR / CUDA（可选）/ CPU"
    return f"{platform_name or platform.system()}：CPU / 可用外部 OCR"


__all__ = ["platform_family", "desktop_platform_badge", "desktop_platform_summary"]
