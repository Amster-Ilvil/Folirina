from __future__ import annotations

"""Application-wide rotating logs and crash capture.

Page processing already owns per-page ``run.log`` / ``run_trace.jsonl`` files.
This module covers the layer those files cannot see: GUI startup, optional-runtime
probing, dependency installation, Qt/Python crashes, and platform integration.
It deliberately uses only the Python standard library so the launcher can start
logging *before* PySide6 or heavy image/model modules are imported.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import traceback
from typing import Mapping

from .version import __version__

_LOGGER_NAME = "folirina"
_FILE_HANDLER_MARKER = "_mhd_app_file_handler"
_CONSOLE_HANDLER_MARKER = "_mhd_app_console_handler"
_HOOKS_INSTALLED = False
_SESSION_ID = datetime.now().strftime("%Y%m%d-%H%M%S-%f")


@dataclass(frozen=True, slots=True)
class ApplicationLogPaths:
    directory: Path
    log_file: Path
    crash_file: Path
    runtime_info_file: Path


def default_log_dir(
    *,
    platform_name: str | None = None,
    env: Mapping[str, str] | None = None,
    home: str | Path | None = None,
) -> Path:
    """Return the native per-user log folder without creating it."""
    system = str(platform_name or platform.system())
    environ = os.environ if env is None else env
    home_path = Path(home).expanduser() if home is not None else Path.home()
    if system == "Darwin":
        return home_path / "Library" / "Logs" / "Folirina"
    if system == "Windows":
        root = Path(environ.get("LOCALAPPDATA", str(home_path / "AppData" / "Local")))
        return root / "Folirina" / "logs"
    state_home = str(environ.get("XDG_STATE_HOME", "") or "").strip()
    if state_home:
        return Path(state_home).expanduser() / "folirina" / "logs"
    return home_path / ".local" / "state" / "folirina" / "logs"


def runtime_log_dir() -> Path:
    """Create and return the application log directory.

    A temporary-folder fallback keeps logging from becoming a startup blocker on
    locked-down machines or unusual portable installations.
    """
    override = str(os.environ.get("FOLIRINA_LOG_DIR", "") or os.environ.get("MHD_LOG_DIR", "") or "").strip()
    preferred = Path(override).expanduser() if override else default_log_dir()
    for candidate in (preferred, Path(tempfile.gettempdir()) / "Folirina" / "logs"):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        except OSError:
            continue
    return Path(tempfile.gettempdir())


def application_log_paths() -> ApplicationLogPaths:
    root = runtime_log_dir()
    return ApplicationLogPaths(
        directory=root,
        log_file=root / "folirina.log",
        crash_file=root / "crash.log",
        runtime_info_file=root / "runtime_info.json",
    )


def _level(value: int | str) -> int:
    if isinstance(value, int):
        return value
    return int(getattr(logging, str(value).upper(), logging.DEBUG))


def configure_application_logging(
    *,
    component: str = "app",
    level: int | str = logging.DEBUG,
    console: bool = False,
) -> ApplicationLogPaths:
    """Attach an idempotent rotating application log to the root logger."""
    paths = application_log_paths()
    root = logging.getLogger()
    root.setLevel(min(root.level if root.level else logging.WARNING, _level(level)))

    file_handler = next((h for h in root.handlers if getattr(h, _FILE_HANDLER_MARKER, False)), None)
    if file_handler is None:
        file_handler = RotatingFileHandler(
            paths.log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
            delay=True,
        )
        setattr(file_handler, _FILE_HANDLER_MARKER, True)
        file_handler.setLevel(_level(level))
        file_handler.setFormatter(logging.Formatter(
            fmt=f"%(asctime)s.%(msecs)03d %(levelname)s [{_SESSION_ID}] [%(threadName)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root.addHandler(file_handler)
    else:
        file_handler.setLevel(_level(level))

    if console and not any(getattr(h, _CONSOLE_HANDLER_MARKER, False) for h in root.handlers):
        stream = logging.StreamHandler()
        setattr(stream, _CONSOLE_HANDLER_MARKER, True)
        stream.setLevel(_level(level))
        stream.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
        root.addHandler(stream)

    logging.captureWarnings(True)
    logger = logging.getLogger(_LOGGER_NAME)
    logger.info(
        "application logging ready component=%s version=%s platform=%s python=%s log_dir=%s",
        component,
        __version__,
        platform.platform(),
        platform.python_version(),
        paths.directory,
    )
    _write_runtime_snapshot(paths, component=component)
    return paths


def _write_runtime_snapshot(paths: ApplicationLogPaths, *, component: str) -> None:
    payload = {
        "schema": "folirina.runtime_info.v1",
        "session_id": _SESSION_ID,
        "started_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "component": str(component),
        "version": __version__,
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "executable": sys.executable,
        "frozen": bool(getattr(sys, "frozen", False)),
        "log_file": str(paths.log_file),
        "crash_file": str(paths.crash_file),
    }
    try:
        tmp = paths.runtime_info_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, paths.runtime_info_file)
    except OSError:
        logging.getLogger(_LOGGER_NAME).debug("runtime_info.json write failed", exc_info=True)


def _append_crash(exc_type, exc_value, exc_tb, *, thread_name: str = "") -> None:
    paths = application_log_paths()
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    header = f"\n[{stamp}] session={_SESSION_ID} thread={thread_name or threading.current_thread().name}\n"
    try:
        with paths.crash_file.open("a", encoding="utf-8") as fh:
            fh.write(header)
            fh.write(text[-50000:])
            if not text.endswith("\n"):
                fh.write("\n")
    except OSError:
        pass


def install_exception_hooks() -> None:
    """Capture uncaught main-thread and worker-thread Python exceptions."""
    global _HOOKS_INSTALLED
    if _HOOKS_INSTALLED:
        return
    _HOOKS_INSTALLED = True
    previous_sys_hook = sys.excepthook

    def sys_hook(exc_type, exc_value, exc_tb):
        logging.getLogger(_LOGGER_NAME).critical(
            "uncaught exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        _append_crash(exc_type, exc_value, exc_tb)
        if previous_sys_hook not in {None, sys_hook}:
            try:
                previous_sys_hook(exc_type, exc_value, exc_tb)
            except Exception:
                pass

    sys.excepthook = sys_hook

    if hasattr(threading, "excepthook"):
        previous_thread_hook = threading.excepthook

        def thread_hook(args):
            logging.getLogger(_LOGGER_NAME).critical(
                "uncaught worker exception thread=%s",
                getattr(args.thread, "name", ""),
                exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
            )
            _append_crash(
                args.exc_type,
                args.exc_value,
                args.exc_traceback,
                thread_name=getattr(args.thread, "name", ""),
            )
            if previous_thread_hook not in {None, thread_hook}:
                try:
                    previous_thread_hook(args)
                except Exception:
                    pass

        threading.excepthook = thread_hook


def log_directory_text() -> str:
    return str(runtime_log_dir())


__all__ = [
    "ApplicationLogPaths",
    "application_log_paths",
    "configure_application_logging",
    "default_log_dir",
    "install_exception_hooks",
    "log_directory_text",
    "runtime_log_dir",
]
