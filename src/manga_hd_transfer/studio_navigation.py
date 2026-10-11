from __future__ import annotations

"""Stable workflow routing shared by the Studio shell and split pages."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StudioRoute:
    key: str
    label: str
    english: str


ROUTES: tuple[StudioRoute, ...] = (
    StudioRoute("workspace", "工作区", "WORKSPACE"),
    StudioRoute("project", "页面管理", "PAGE MANAGER"),
    StudioRoute("models", "识别与配准", "DETECT & ALIGN"),
    StudioRoute("workbench", "替换工作台", "TRANSFER DESK"),
    StudioRoute("export", "出版输出", "EXPORT"),
    StudioRoute("settings", "设置", "SETTINGS"),
)

ROUTE_INDEX = {route.key: index for index, route in enumerate(ROUTES)}


def route_index(key: str) -> int:
    try:
        return ROUTE_INDEX[str(key)]
    except KeyError as exc:
        raise KeyError(f"unknown studio route: {key}") from exc


__all__ = ["StudioRoute", "ROUTES", "ROUTE_INDEX", "route_index"]
