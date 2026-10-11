from __future__ import annotations

"""Workflow order for the Studio shell (no Qt dependency; unit-testable headless).

页面管理 → 替换工作台 → 出版输出。「识别与配准」是模型/检测/硬件的配置页，不是流程中的一步，
保持在侧边栏里随时可进，但不进入流程条。
"""

from .studio_navigation import ROUTES, route_index

FLOW_KEYS: tuple[str, ...] = ("project", "workbench", "export")
FLOW_LABELS: dict[str, str] = {r.key: r.label for r in ROUTES}
FLOW_SHORT: dict[str, str] = {"project": "页面管理", "workbench": "替换工作台", "export": "出版输出"}


def flow_position(key: str):
    try:
        return FLOW_KEYS.index(str(key))
    except ValueError:
        return None


def next_route(key: str):
    pos = flow_position(key)
    if pos is None or pos + 1 >= len(FLOW_KEYS):
        return None
    return FLOW_KEYS[pos + 1]


def step_state(step_key: str, current_key: str) -> str:
    cur, step = flow_position(current_key), flow_position(step_key)
    if cur is None or step is None:
        return "todo"
    return "current" if step == cur else ("done" if step < cur else "todo")


__all__ = ["FLOW_KEYS", "FLOW_LABELS", "FLOW_SHORT", "flow_position", "next_route", "step_state", "route_index"]
