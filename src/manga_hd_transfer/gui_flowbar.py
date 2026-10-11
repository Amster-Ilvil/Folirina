from __future__ import annotations

"""Slim workflow bar: clickable stage chips + a one-click “下一步”."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from .studio_flow import FLOW_KEYS, FLOW_LABELS, FLOW_SHORT, flow_position, next_route, step_state
from .studio_navigation import route_index


class FlowBar(QFrame):
    navigate = Signal(int)          # route index; connected to StudioWindow.show_page

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("flowBar")
        self.setFixedHeight(46)
        row = QHBoxLayout(self); row.setContentsMargins(18, 5, 18, 5); row.setSpacing(2)
        self._chips: dict[str, QPushButton] = {}
        for pos, key in enumerate(FLOW_KEYS):
            if pos:
                sep = QLabel("›"); sep.setObjectName("flowSep"); row.addWidget(sep)
            chip = QPushButton(FLOW_SHORT[key]); chip.setObjectName("flowChip")
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(f"跳到「{FLOW_LABELS[key]}」")
            chip.clicked.connect(lambda _=False, k=key: self.navigate.emit(route_index(k)))
            self._chips[key] = chip; row.addWidget(chip)
        row.addStretch(1)
        self._next = QPushButton("下一步 →"); self._next.setObjectName("flowNext")
        self._next.setCursor(Qt.CursorShape.PointingHandCursor)
        self._next.clicked.connect(self._go_next)
        row.addWidget(self._next)
        self._route = ""
        self.hide()

    def _go_next(self) -> None:
        target = next_route(self._route)
        if target is not None:
            self.navigate.emit(route_index(target))

    def set_route(self, key: str) -> None:
        self._route = str(key)
        in_flow = flow_position(key) is not None
        self.setVisible(in_flow)
        if not in_flow:
            return
        for step_key, chip in self._chips.items():
            state = step_state(step_key, key)
            chip.setProperty("state", state)
            chip.setText(("✓ " if state == "done" else "") + FLOW_SHORT[step_key])
            chip.style().unpolish(chip); chip.style().polish(chip)
        target = next_route(key)
        self._next.setVisible(target is not None)
        if target is not None:
            self._next.setText(f"下一步：{FLOW_LABELS[target]} →")


__all__ = ["FlowBar"]
