from __future__ import annotations

"""Qt binding for the shortcut registry: one QShortcut per key, page-aware dispatch, tooltip hints, F1 help."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication, QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .gui_shortcuts_logic import (
    SHORTCUTS, ShortcutSpec, focus_blocks, grouped_for_help, hint_suffix, pick_spec, specs_by_key,
)


def _native(keys) -> list[str]:
    return [QKeySequence(k).toString(QKeySequence.SequenceFormat.NativeText) or k for k in keys]


def _current_scope(window) -> str:
    try:
        index = window.stack.currentIndex()
        if index == window.page_index("workbench"):
            return "workbench"
        if index == window.page_index("project"):
            return "project"
    except Exception:
        pass
    return "global"


def _click(widget) -> None:
    if widget is not None and widget.isEnabled() and widget.isVisible():
        widget.click()


def _workbench(window):
    return window.loaded_page("workbench")


def dispatch(window, spec: ShortcutSpec) -> None:
    """Run the action behind a shortcut; an unknown widget name must never raise inside a key handler."""
    try:
        _dispatch(window, spec)
    except AttributeError:
        pass


def _dispatch(window, spec: ShortcutSpec) -> None:
    sid = spec.id
    if sid.startswith("goto_"):
        window.show_page(int(sid.split("_")[1]) - 1)
    elif sid == "run_current_project":
        _click(getattr(window.loaded_page("project"), "run_page", None))
    elif sid == "run_current_workbench":
        _click(getattr(_workbench(window), "run", None))
    elif sid == "stop_task":
        stop = getattr(window.loaded_page("project"), "cancel", None) or getattr(window, "stop_button", None)
        if stop is not None and stop.isEnabled():
            (stop.click if hasattr(stop, "click") else window.cancel_worker)()
    elif sid in ("prev_page", "next_page", "toggle_rail", "toggle_inspector", "inspector_params", "inspector_review") or sid.startswith("view_"):
        page = _workbench(window)
        if page is None:
            return
        if sid == "prev_page":
            _click(page.prev_page)
        elif sid == "next_page":
            _click(page.next_page)
        elif sid == "toggle_rail":
            _click(page.page_rail_toggle)
        elif sid == "toggle_inspector":
            _click(page.inspector_toggle)
        elif sid == "inspector_params":
            _click(page.inspector_settings_tab)
        elif sid == "inspector_review":
            _click(page.inspector_review_tab)
        else:
            buttons = getattr(page, "view_buttons", [])
            index = int(sid.split("_")[1])
            if 0 <= index < len(buttons):
                _click(buttons[index][0])
    elif sid == "page_log":
        _click(getattr(window, "log_button", None))
    elif sid == "toggle_theme":
        _click(getattr(window, "theme_button", None))
    elif sid == "help":
        ShortcutHelpDialog(window).exec()


def _focus_chain() -> list[str]:
    widget = QApplication.focusWidget()
    return [cls.__name__ for cls in type(widget).__mro__] if widget is not None else []


def _activate(window, key: str, candidates: list[ShortcutSpec]) -> None:
    if focus_blocks(key, _focus_chain()):
        return
    spec = pick_spec(candidates, _current_scope(window))
    if spec is not None:
        dispatch(window, spec)


class ShortcutHelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("快捷键速查")
        self.setObjectName("confirmDialog")
        self.setModal(True)
        layout = QVBoxLayout(self); layout.setContentsMargins(24, 20, 24, 18); layout.setSpacing(12)
        title = QLabel("快捷键速查"); title.setObjectName("confirmTitle"); layout.addWidget(title)
        note = QLabel("数字、PgUp / PgDn 这类单键在输入框里不会触发，不影响打字。Ctrl 在 macOS 上对应 ⌘。")
        note.setObjectName("quiet"); note.setWordWrap(True); layout.addWidget(note)
        grid = QGridLayout(); grid.setHorizontalSpacing(18); grid.setVerticalSpacing(6)
        row = 0
        for group, items in grouped_for_help():
            head = QLabel(group); head.setObjectName("railSection"); grid.addWidget(head, row, 0, 1, 2); row += 1
            for label, keys in items:
                grid.addWidget(QLabel(label), row, 0)
                holder = QWidget(); box = QHBoxLayout(holder); box.setContentsMargins(0, 0, 0, 0); box.setSpacing(5)
                for part in keys.split(" / "):
                    chip = QLabel(part); chip.setObjectName("shortcutKey"); box.addWidget(chip)
                box.addStretch(1)
                grid.addWidget(holder, row, 1); row += 1
        layout.addLayout(grid)
        close = QPushButton("关闭"); close.setObjectName("primary"); close.setDefault(True); close.clicked.connect(self.accept)
        foot = QHBoxLayout(); foot.addStretch(1); foot.addWidget(close); layout.addLayout(foot)
        self.setMinimumWidth(520)


def _hint(widget, spec_ids: tuple[str, ...]) -> None:
    if widget is None or widget.property("nfShortcutHint"):
        return
    keys: list[str] = []
    for spec in SHORTCUTS:
        if spec.id in spec_ids:
            keys.extend(_native(spec.keys[:1]))
    if not keys:
        return
    base = widget.toolTip().strip()
    widget.setToolTip((base + "\n" if base else "") + hint_suffix(keys).strip("（）"))
    widget.setProperty("nfShortcutHint", True)


def apply_hints(window) -> None:
    """Add shortcut hints to tooltips (workbench is lazy-loaded, so this runs after every page change)."""
    try:
        for index, button in enumerate(window.nav_group.buttons()):       # 左侧导航按钮：Ctrl+1…6
            _hint(button, (f"goto_{index + 1}",))
        _hint(getattr(window, "log_button", None), ("page_log",))
        _hint(getattr(window, "theme_button", None), ("toggle_theme",))
        project = window.loaded_page("project")
        _hint(getattr(project, "run_page", None), ("run_current_project",))
        page = _workbench(window)
        if page is not None:
            _hint(page.prev_page, ("prev_page",)); _hint(page.next_page, ("next_page",))
            _hint(page.page_rail_toggle, ("toggle_rail",)); _hint(page.inspector_toggle, ("toggle_inspector",))
            _hint(getattr(page, "run", None), ("run_current_workbench",))
            _hint(page.inspector_settings_tab, ("inspector_params",)); _hint(page.inspector_review_tab, ("inspector_review",))
            for i, (button, _key) in enumerate(getattr(page, "view_buttons", [])):
                _hint(button, (f"view_{i}",))
    except Exception:
        pass            # 提示文字永远不能影响页面本身


def install_shortcuts(window) -> None:
    window._nf_shortcuts = []
    for key, candidates in specs_by_key().items():
        shortcut = QShortcut(QKeySequence(key), window)
        shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        shortcut.activated.connect(lambda k=key, c=candidates: _activate(window, k, c))
        window._nf_shortcuts.append(shortcut)
    apply_hints(window)
    window.stack.currentChanged.connect(lambda _i: apply_hints(window))


__all__ = ["install_shortcuts", "apply_hints", "dispatch", "ShortcutHelpDialog"]
