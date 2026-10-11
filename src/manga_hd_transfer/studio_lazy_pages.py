from __future__ import annotations

"""Deterministic lazy page registry for Folirina Studio.

The registry owns widget construction only.  Durable project/runtime state stays
in StudioWindow and its controllers, mirroring Novel Formatter's Phase 18
lazy-workspace boundary without creating a second business-state model.
"""

from dataclasses import dataclass
from typing import Callable

from PySide6.QtWidgets import QStackedWidget, QWidget


@dataclass(slots=True)
class _Entry:
    index: int
    factory: Callable[[], QWidget]
    placeholder: QWidget
    instance: QWidget | None = None
    constructing: bool = False


class LazyStudioPageRegistry:
    """Create heavy Studio pages once, on first real navigation."""

    def __init__(self, stack: QStackedWidget):
        self._stack = stack
        self._entries: dict[str, _Entry] = {}

    def register(self, key: str, index: int, factory: Callable[[], QWidget]) -> QWidget:
        name = str(key or "").strip()
        if not name:
            raise ValueError("lazy page key cannot be empty")
        if name in self._entries:
            raise ValueError(f"lazy page already registered: {name}")
        placeholder = QWidget()
        placeholder.setObjectName(f"lazyPagePlaceholder_{name}")
        inserted = self._stack.insertWidget(int(index), placeholder)
        if inserted != int(index):
            raise RuntimeError(f"failed to reserve lazy page slot {name}: {inserted} != {index}")
        self._entries[name] = _Entry(index=int(index), factory=factory, placeholder=placeholder)
        return placeholder

    def get(self, key: str) -> QWidget:
        name = str(key or "").strip()
        try:
            entry = self._entries[name]
        except KeyError as exc:
            raise KeyError(f"unknown lazy page: {name}") from exc
        if entry.instance is not None:
            return entry.instance
        if entry.constructing:
            raise RuntimeError(f"recursive lazy page construction: {name}")
        entry.constructing = True
        try:
            widget = entry.factory()
            if not isinstance(widget, QWidget):
                raise TypeError(f"lazy page {name!r} did not create QWidget")
            current = self._stack.currentIndex()
            placeholder = entry.placeholder
            actual_index = self._stack.indexOf(placeholder)
            if actual_index < 0:
                raise RuntimeError(f"lazy page placeholder disappeared: {name}")
            self._stack.removeWidget(placeholder)
            placeholder.deleteLater()
            inserted = self._stack.insertWidget(actual_index, widget)
            if inserted != actual_index:
                raise RuntimeError(f"failed to install lazy page {name}: {inserted} != {actual_index}")
            entry.instance = widget
            if current == actual_index:
                self._stack.setCurrentWidget(widget)
            return widget
        finally:
            entry.constructing = False

    def loaded(self, key: str) -> QWidget | None:
        entry = self._entries.get(str(key or "").strip())
        return None if entry is None else entry.instance

    def is_loaded(self, key: str) -> bool:
        return self.loaded(key) is not None

    def loaded_items(self) -> tuple[tuple[str, QWidget], ...]:
        return tuple((key, entry.instance) for key, entry in self._entries.items() if entry.instance is not None)


__all__ = ["LazyStudioPageRegistry"]
