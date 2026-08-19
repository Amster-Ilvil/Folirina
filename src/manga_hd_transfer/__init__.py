"""Folirina manga translation-transfer core package.

The package-level version is intentionally re-exported from ``version.py`` so
lightweight helpers (including the updater) can import the version without
pulling in Qt or any optional ML runtime.
"""
from __future__ import annotations

from .version import __version__

__all__ = ["__version__"]
