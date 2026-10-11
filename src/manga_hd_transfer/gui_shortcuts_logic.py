from __future__ import annotations

"""Keyboard shortcut registry for the Studio shell (headless: no Qt import, unit-testable).

Design rules
- One QShortcut per key sequence; the active page decides which spec handles it, so the same key can mean
  “处理当前页” on both 页面管理 and 替换工作台 without Qt's ambiguous-shortcut dead zone.
- Keys without any modifier (digits, PgUp/PgDn, F-keys) never fire while a text field has focus.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence

SCOPES = ("global", "project", "workbench")


@dataclass(frozen=True)
class ShortcutSpec:
    id: str
    keys: tuple[str, ...]
    label: str
    group: str
    scope: str = "global"


VIEW_LABELS = ("日文原图", "旧中文版", "最终结果", "复核标注", "迁移蒙版", "清除蒙版", "中文迁移层", "只清日文", "TARGET 擦除预览", "TARGET 恢复预览")
_VIEW_KEYS = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")

SHORTCUTS: tuple[ShortcutSpec, ...] = (
    *(ShortcutSpec(f"goto_{i}", (f"Ctrl+{i}",), f"跳到「{name}」", "页面切换")
      for i, name in enumerate(("工作区", "页面管理", "识别与配准", "替换工作台", "出版输出", "设置"), start=1)),
    ShortcutSpec("run_current_project", ("Ctrl+Return", "Ctrl+Enter"), "处理当前页", "处理", "project"),
    ShortcutSpec("run_current_workbench", ("Ctrl+Return", "Ctrl+Enter"), "处理当前页", "处理", "workbench"),
    ShortcutSpec("stop_task", ("Ctrl+.",), "停止当前任务", "处理"),
    ShortcutSpec("prev_page", ("PgUp", "Alt+Left"), "上一页", "工作台翻页", "workbench"),
    ShortcutSpec("next_page", ("PgDown", "Alt+Right"), "下一页", "工作台翻页", "workbench"),
    *(ShortcutSpec(f"view_{i}", (k,), f"切换到「{label}」", "工作台视图", "workbench")
      for i, (k, label) in enumerate(zip(_VIEW_KEYS, VIEW_LABELS))),
    ShortcutSpec("toggle_rail", ("Ctrl+B",), "收起 / 展开页面栏", "工作台布局", "workbench"),
    ShortcutSpec("toggle_inspector", ("Ctrl+Shift+B",), "隐藏 / 显示参数栏", "工作台布局", "workbench"),
    ShortcutSpec("inspector_params", ("Ctrl+[",), "切换到「参数」", "工作台布局", "workbench"),
    ShortcutSpec("inspector_review", ("Ctrl+]",), "切换到「复核 / QA」", "工作台布局", "workbench"),
    ShortcutSpec("page_log", ("Ctrl+Shift+L",), "打开页面日志", "工具"),
    ShortcutSpec("toggle_theme", ("Ctrl+Shift+T",), "切换浅色 / 深色", "工具"),
    ShortcutSpec("help", ("F1",), "快捷键速查", "工具"),
)

_MODIFIERS = ("Ctrl", "Alt", "Shift", "Meta", "Cmd")


def has_modifier(key: str) -> bool:
    return any(part in _MODIFIERS for part in str(key).split("+")[:-1])


def find_conflicts(specs: Sequence[ShortcutSpec] = SHORTCUTS) -> list[tuple[str, str, str]]:
    """Return (key, id_a, id_b) when two specs would fight: same scope, or a global spec vs any scoped spec."""
    seen: dict[str, list[ShortcutSpec]] = {}
    out: list[tuple[str, str, str]] = []
    for spec in specs:
        for key in spec.keys:
            for other in seen.get(key, []):
                if other.scope == spec.scope or "global" in (other.scope, spec.scope):
                    out.append((key, other.id, spec.id))
            seen.setdefault(key, []).append(spec)
    return out


def specs_by_key(specs: Sequence[ShortcutSpec] = SHORTCUTS) -> dict[str, list[ShortcutSpec]]:
    grouped: dict[str, list[ShortcutSpec]] = {}
    for spec in specs:
        for key in spec.keys:
            grouped.setdefault(key, []).append(spec)
    return grouped


def pick_spec(candidates: Iterable[ShortcutSpec], current_scope: str) -> ShortcutSpec | None:
    """Scoped specs win over global ones; a spec whose scope is not the current page is ignored."""
    scoped = [s for s in candidates if s.scope == current_scope and s.scope != "global"]
    if scoped:
        return scoped[0]
    return next((s for s in candidates if s.scope == "global"), None)


_TEXT_INPUT_NAMES = frozenset({"QLineEdit", "QTextEdit", "QPlainTextEdit", "QAbstractSpinBox", "QSpinBox", "QDoubleSpinBox", "QComboBox"})


def focus_blocks(key: str, focus_class_chain: Iterable[str]) -> bool:
    """Plain keys must not steal typing from fields; modified shortcuts always work."""
    if has_modifier(key):
        return False
    return any(name in _TEXT_INPUT_NAMES for name in focus_class_chain)


def hint_suffix(keys: Sequence[str]) -> str:
    return "（快捷键：" + " / ".join(keys) + "）" if keys else ""


def grouped_for_help(specs: Sequence[ShortcutSpec] = SHORTCUTS) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(group, [(label, 'key1 / key2'), …]), …] with 0-9 view keys folded into one readable row."""
    order: list[str] = []
    rows: dict[str, list[tuple[str, str]]] = {}
    views_added = False
    for spec in specs:
        if spec.id.startswith("view_"):
            if views_added:
                continue
            views_added = True
            label, keys = "切换工作台视图（顺序：日文原图 … TARGET 恢复预览）", "1 … 9 / 0"
        elif spec.id.startswith("goto_") and spec.id != "goto_1":
            continue
        elif spec.id == "goto_1":
            label, keys = "跳到对应页面（工作区 … 设置）", "Ctrl+1 … Ctrl+6"
        elif spec.id == "run_current_workbench":
            continue                      # 与页面管理里的“处理当前页”同一个快捷键，只列一行
        else:
            label, keys = spec.label, " / ".join(spec.keys)
        rows.setdefault(spec.group, []).append((label, keys))
        if spec.group not in order:
            order.append(spec.group)
    return [(g, rows[g]) for g in order]


__all__ = ["ShortcutSpec", "SHORTCUTS", "VIEW_LABELS", "has_modifier", "find_conflicts", "specs_by_key", "pick_spec",
           "focus_blocks", "hint_suffix", "grouped_for_help", "SCOPES"]
