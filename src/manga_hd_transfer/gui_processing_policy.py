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
    model_write_running: bool = False,
) -> BusyState:
    pipeline_running = bool(pipeline_running)
    prepare_running = bool(prepare_running)
    page_action_running = bool(page_action_running)
    settings_updating = bool(settings_updating)
    model_write_running = bool(model_write_running)
    return BusyState(
        busy=(pipeline_running or prepare_running or page_action_running or settings_updating or model_write_running),
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
        # The button label is an explicit user contract. “继续处理整本” must
        # preserve already-published successful pages and continue from the first
        # unfinished page; “从头处理整本” remains the way to apply current settings
        # to every page again.
        snapshot.batch.resume_policy = "continue" if bool(resume) else "strict"
    return snapshot


def page_completion_message(project: Any, *, reprocessed: bool = False) -> str:
    """Explain what a single-page run actually published.

    A mode switch may be a valid no-op when the selected renderer finds no safe
    transferable regions or when two modes happen to produce identical pixels.
    Report that explicitly instead of the ambiguous generic "processing done".
    """
    meta = dict(getattr(project, "meta", {}) or {}) if hasattr(project, "meta") else {}
    effect = dict(meta.get("publish_effect") or {})
    current = str(effect.get("current_mode") or meta.get("transfer_mode") or "").strip()
    previous = str(effect.get("previous_mode") or "").strip()
    applied = int(effect.get("applied_regions", 0) or 0)
    compared = bool(effect.get("pixel_comparison_available"))
    identical = bool(effect.get("pixel_identical")) if compared else False
    changed = effect.get("changed_pixels")
    run_status = str(meta.get("run_status") or "").strip().lower()
    if bool(effect.get("mode_changed")):
        if compared and identical:
            return f"模式切换已执行 · {previous} → {current} · 新结果与上一模式逐像素相同（不是旧结果缓存复用）"
        if compared:
            return f"模式切换完成 · {previous} → {current} · 输出已更新 {int(changed or 0)} px · 应用 {applied} 个区域"
        return f"模式切换完成 · {previous} → {current} · 应用 {applied} 个区域"
    prefix = "重新处理完成" if reprocessed else "当前页处理完成"
    if run_status == "partial":
        return f"{prefix} · 部分区域成功，仍有区域失败或需复核 · 已应用 {applied} 个区域"
    if run_status == "skipped":
        return f"{prefix} · 当前页按页面规则跳过，已保留稳定输出"
    if run_status == "integrity_failed":
        return f"{prefix} · 输出完整性检查未通过，请查看日志后再导出"
    if run_status == "noop" and not compared:
        return f"{prefix} · 当前模式已执行，但没有安全可发布的迁移区域"
    if compared and identical:
        return f"{prefix} · 已重新执行当前模式，但输出与处理前逐像素相同 · 应用 {applied} 个区域"
    if compared:
        return f"{prefix} · 输出已更新 {int(changed or 0)} px · 应用 {applied} 个区域"
    if applied == 0:
        return f"{prefix} · 当前模式未生成可发布迁移区域"
    return f"{prefix} · 应用 {applied} 个区域"


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
    "page_completion_message",
]
