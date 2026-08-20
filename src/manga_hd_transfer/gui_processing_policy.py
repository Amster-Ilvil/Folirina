from __future__ import annotations

"""Headless processing policy used by the Qt controller.

The GUI owns widgets and signals; this module owns pure decisions about busy
state, worker snapshots, progress classification and completion messaging.  It
has no Qt imports, so the behavior is testable on every CI runner.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BusyState:
    busy: bool
    cancellable: bool


def compute_busy_state(
    *, pipeline_running: bool = False, prepare_running: bool = False,
    page_action_running: bool = False, settings_updating: bool = False,
) -> BusyState:
    pipeline_running = bool(pipeline_running)
    prepare_running = bool(prepare_running)
    page_action_running = bool(page_action_running)
    settings_updating = bool(settings_updating)
    return BusyState(
        busy=pipeline_running or prepare_running or page_action_running or settings_updating,
        cancellable=pipeline_running or prepare_running,
    )


def classify_progress_state(message: str, *, cache_hit: bool = False) -> str:
    text = str(message or "")
    if cache_hit:
        return "缓存"
    if "跳过" in text:
        return "跳过"
    if "正在" in text:
        return "处理中"
    if "Error" in text or "失败" in text:
        return "失败"
    if "取消" in text or "停止" in text:
        return "已停止"
    return "完成"


def worker_config_snapshot(config: Any, *, resume: bool | None = None):
    """Return an isolated worker config without mutating the GUI's live config."""
    snapshot = config.model_copy(deep=True)
    if resume is not None:
        snapshot.batch.resume = bool(resume)
        snapshot.batch.skip_completed = bool(resume)
    return snapshot


def completion_message(project: Any) -> str:
    """Generate the whole-book completion message from immutable result metadata."""
    meta = dict(getattr(project, "meta", {}) or {}) if hasattr(project, "meta") else {}
    cancelled = bool(meta.get("cancelled"))
    skipped = int(meta.get("skipped_count", 0) or 0)
    resumed = int(meta.get("resumed_count", 0) or 0)
    if cancelled:
        return "已停止，已完成页面已保留；下次可点“继续处理整本”"
    if resumed:
        return f"继续处理完成 · 跳过已完成 {resumed} 页" + (f" · 另跳过 {skipped} 页" if skipped else "")
    if skipped:
        return f"从头处理完成 · 自动/手动跳过 {skipped} 页"
    return "从头处理完成"


__all__ = [
    "BusyState",
    "compute_busy_state",
    "classify_progress_state",
    "worker_config_snapshot",
    "completion_message",
]
