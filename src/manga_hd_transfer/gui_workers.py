from __future__ import annotations

"""Qt worker threads used by the GUI shell.

The widgets/pages remain in ``gui_qt.py`` for now, but long-running page-local,
model probe/download and dependency preparation jobs live here so the UI module
does not own installation/runtime behavior.  No worker in this module imports
``gui_qt`` or ``pipeline``.
"""

from typing import Callable
import logging

from PySide6.QtCore import QThread, Signal

from .config import PipelineConfig
from .dependency_install import (
    install_all_model_dependencies,
    install_model_dependencies,
    missing_dependency_modules,
)
from .model_downloads import download_builtin_model, diagnose_download_network
from .runtime_catalog import probe_components
from .runtime_preflight import plan_runtime_requirements, model_artifact_ready

logger = logging.getLogger(__name__)


class PageActionWorker(QThread):
    """Run one page-local review/editor action outside the Qt GUI thread."""

    done = Signal(object)
    failed = Signal(str)

    def __init__(self, label: str, action: Callable[[], object], parent=None):
        # A QObject parent keeps the Python wrapper alive until Qt has delivered
        # finished/deleteLater.  This matters for short page-local jobs where the
        # dialog may otherwise be the only strong reference to the QThread.
        super().__init__(parent)
        self.label = str(label)
        self.action = action

    def run(self):
        try:
            self.done.emit(self.action())
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ComponentProbeWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config

    def run(self):
        try:
            self.done.emit(probe_components(self.config))
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelDownloadWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(self, key: str, config: PipelineConfig):
        super().__init__()
        self.key = str(key)
        self.config = config

    def run(self):
        try:
            def report(done: int, total: int | None, message: str):
                if total and total > 0:
                    percent = max(0, min(100, int(round(done * 100.0 / total))))
                else:
                    percent = -1
                self.progress.emit(percent, str(message))

            result = download_builtin_model(self.key, self.config, report)
            self.done.emit(result)
        except Exception as exc:
            import traceback
            logger.exception("model download worker failed key=%s", self.key)
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class DependencyInstallWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, key: str):
        super().__init__()
        self.key = str(key)

    def run(self):
        try:
            report = lambda msg: self.progress.emit(str(msg))
            result = (
                install_all_model_dependencies(report)
                if self.key == "all"
                else install_model_dependencies(self.key, report)
            )
            self.done.emit(result)
        except Exception as exc:
            import traceback
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class AutoPrepareModelsWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(str)
    cancelled = Signal()

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self._cancel_requested = False

    def request_cancel(self):
        # Cooperative cancellation only. Never terminate while pip/download code
        # may be mutating a runtime environment.
        self._cancel_requested = True
        self.requestInterruption()

    def _cancelled(self) -> bool:
        return bool(self._cancel_requested or self.isInterruptionRequested())

    def _stop_if_cancelled(self) -> bool:
        if not self._cancelled():
            return False
        self.cancelled.emit()
        return True

    def run(self):
        try:
            if self._stop_if_cancelled():
                return
            plan = plan_runtime_requirements(self.config)
            if plan.errors:
                raise RuntimeError("\n".join(plan.errors))
            if not plan.requirements:
                self.done.emit({"prepared": [], "message": "当前任务没有需要准备的可下载模型。"})
                return

            prepared: list[str] = []
            count = len(plan.requirements)
            for index, req in enumerate(plan.requirements, 1):
                if self._stop_if_cancelled():
                    return
                local_cfg = self.config.model_copy(deep=True)
                if req.key == "paddle" and req.profile:
                    local_cfg.ocr.paddle_model_profile = req.profile
                runtime_key = str(req.runtime_key or req.key)
                missing = tuple(missing_dependency_modules(runtime_key))
                runtime_repaired = bool(missing)
                if missing:
                    self.progress.emit(
                        f"[{index}/{count}] {req.label} · 正在安装/修复独立运行依赖…"
                    )
                    install_model_dependencies(
                        runtime_key,
                        lambda msg, _i=index, _n=count, _label=req.label: self.progress.emit(
                            f"[{_i}/{_n}] {_label} · {msg}"
                        ),
                    )
                    if self._stop_if_cancelled():
                        return

                if model_artifact_ready(local_cfg, req) and not (
                    req.key == "paddle" and runtime_repaired
                ):
                    self.progress.emit(f"[{index}/{count}] {req.label} 已就绪，跳过下载。")
                    prepared.append(req.label)
                    continue

                self.progress.emit(f"[{index}/{count}] 自动下载并校验 {req.label}…")

                def report(
                    done: int,
                    total: int | None,
                    message: str,
                    *,
                    _index=index,
                    _count=count,
                    _label=req.label,
                ):
                    prefix = f"[{_index}/{_count}] {_label}"
                    if total and total > 0:
                        percent = max(
                            0.0, min(100.0, float(done) * 100.0 / float(total))
                        )
                        self.progress.emit(f"{prefix} · {percent:.0f}% · {message}")
                    else:
                        self.progress.emit(f"{prefix} · {message}")

                download_builtin_model(req.key, local_cfg, report)
                if self._stop_if_cancelled():
                    return
                if not model_artifact_ready(local_cfg, req):
                    raise RuntimeError(
                        f"{req.label} 下载流程结束，但本地就绪检查仍未通过。"
                    )
                prepared.append(req.label)

            if self._stop_if_cancelled():
                return
            msg = "自动准备完成：" + (" / ".join(prepared) if prepared else "无")
            self.done.emit({"prepared": prepared, "message": msg})
        except Exception as exc:
            import traceback
            logger.exception("automatic model preparation failed")
            self.failed.emit(f"{exc}\n\n{traceback.format_exc()}")


class ModelNetworkProbeWorker(QThread):
    done = Signal(object)
    failed = Signal(str)

    def run(self):
        try:
            self.done.emit(diagnose_download_network())
        except Exception as exc:
            self.failed.emit(str(exc))


__all__ = [
    "PageActionWorker",
    "ComponentProbeWorker",
    "ModelDownloadWorker",
    "DependencyInstallWorker",
    "AutoPrepareModelsWorker",
    "ModelNetworkProbeWorker",
]
