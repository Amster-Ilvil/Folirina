from __future__ import annotations

"""Per-page processing guard and crash-residue cleanup.

A page workspace is a shared mutable directory: automatic processing, GUI review,
and batch resume can all touch it.  Running two automatic writers against the same
page can interleave otherwise-atomic artifact replacements and leave a logically
mixed result.  The guard makes page processing single-writer while allowing
independent pages to run concurrently.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import time
import uuid
import threading
from functools import wraps


_LOCK_NAME = ".page_processing.lock"
_DEFAULT_STALE_SECONDS = 6 * 60 * 60
_LOCAL_GUARDS: dict[str, dict] = {}
_LOCAL_GUARDS_LOCK = threading.RLock()


class PageRunBusyError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # A process owned by another user is still alive from our perspective.
        return True
    except OSError:
        return False


def _load_lock(path: Path) -> dict:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _is_stale(path: Path, obj: dict, stale_seconds: int) -> bool:
    try:
        age = max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return True
    if age >= max(60, int(stale_seconds)):
        return True
    host = str(obj.get("host") or "")
    pid = int(obj.get("pid") or 0)
    if host and host == socket.gethostname() and pid > 0 and not _pid_alive(pid):
        return True
    return False


@dataclass(slots=True)
class PageRunGuard:
    page_root: Path
    mode: str
    stale_seconds: int = _DEFAULT_STALE_SECONDS
    _token: str = field(init=False, repr=False, default="")
    _lock_path: Path = field(init=False, repr=False, default=Path("."))
    _acquired: bool = field(init=False, repr=False, default=False)
    _key: str = field(init=False, repr=False, default="")
    _reentrant: bool = field(init=False, repr=False, default=False)

    def __init__(self, page_root: str | Path, mode: str, stale_seconds: int = _DEFAULT_STALE_SECONDS) -> None:
        self.page_root = Path(page_root)
        self.mode = str(mode or "")
        self.stale_seconds = int(stale_seconds)
        self._token = uuid.uuid4().hex
        self._lock_path = self.page_root / _LOCK_NAME
        try:
            self._key = str(self.page_root.resolve())
        except Exception:
            self._key = str(self.page_root)
        self._acquired = False
        self._reentrant = False

    def acquire(self) -> dict:
        self.page_root.mkdir(parents=True, exist_ok=True)
        thread_id = int(threading.get_ident())
        # Nested review helpers in the same logical call may write the same page.
        # Re-enter only for the same Python thread; a concurrent GUI/batch thread
        # in the same process is still blocked.
        with _LOCAL_GUARDS_LOCK:
            local = _LOCAL_GUARDS.get(self._key)
            if local and int(local.get("thread_id") or -1) == thread_id:
                self._token = str(local.get("token") or self._token)
                local["count"] = int(local.get("count") or 1) + 1
                self._acquired = True
                self._reentrant = True
                return {
                    "acquired": True, "reentrant": True, "recovered_stale": False,
                    "token": self._token, "pid": os.getpid(), "thread_id": thread_id,
                    "host": socket.gethostname(), "mode": self.mode,
                }
        recovered_stale = False
        for _ in range(2):
            payload = {
                "schema": "manga_hd_translation_transfer.page_lock.v1",
                "token": self._token,
                "pid": os.getpid(),
                "thread_id": thread_id,
                "host": socket.gethostname(),
                "mode": self.mode,
                "started_at": _utc_now(),
            }
            try:
                fd = os.open(str(self._lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
                try:
                    raw = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
                    os.write(fd, raw)
                    try:
                        os.fsync(fd)
                    except OSError:
                        pass
                finally:
                    os.close(fd)
                self._acquired = True
                with _LOCAL_GUARDS_LOCK:
                    _LOCAL_GUARDS[self._key] = {"token": self._token, "thread_id": thread_id, "count": 1}
                return {"acquired": True, "reentrant": False, "recovered_stale": recovered_stale, **payload}
            except FileExistsError:
                owner = _load_lock(self._lock_path)
                if _is_stale(self._lock_path, owner, self.stale_seconds):
                    try:
                        self._lock_path.unlink()
                        recovered_stale = True
                        continue
                    except OSError:
                        pass
                detail = ""
                if owner:
                    detail = f" pid={owner.get('pid','?')} mode={owner.get('mode','?')} started={owner.get('started_at','?')}"
                raise PageRunBusyError(
                    "当前页正在被另一个处理任务写入，已阻止并发覆盖。请等待该任务结束后重试。" + detail
                )
        raise PageRunBusyError("当前页处理锁无法安全取得，请稍后重试。")

    def release(self) -> None:
        if not self._acquired:
            return
        remove_file = True
        with _LOCAL_GUARDS_LOCK:
            local = _LOCAL_GUARDS.get(self._key)
            if local and str(local.get("token") or "") == self._token:
                count = int(local.get("count") or 1) - 1
                if count > 0:
                    local["count"] = count
                    remove_file = False
                else:
                    _LOCAL_GUARDS.pop(self._key, None)
        if remove_file:
            try:
                owner = _load_lock(self._lock_path)
                if str(owner.get("token") or "") == self._token:
                    self._lock_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._acquired = False

    def __enter__(self) -> "PageRunGuard":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.release()
        return False


def cleanup_orphan_temp_files(page_root: str | Path) -> dict:
    """Remove only our sibling atomic-write temp files.

    User files are never touched.  ``write_image``/``save_json`` use the hidden
    ``.<destination>.*.tmp`` convention, so interrupted writes can be identified
    without broad wildcard deletion.
    """
    root = Path(page_root)
    removed: list[str] = []
    if not root.exists():
        return {"removed": removed, "count": 0}
    try:
        # io_utils uses ``.*.tmp`` while result-state atomic mirrors use
        # ``.*.tmp-sync``. Both are crash-only staging artifacts and neither
        # should accumulate in a page workspace after an interrupted publish.
        candidates = list(root.glob(".*.tmp")) + list(root.glob(".*.tmp-sync"))
    except OSError:
        candidates = []
    for p in candidates:
        if not p.is_file() or p.name == _LOCK_NAME:
            continue
        try:
            p.unlink()
            removed.append(p.name)
        except OSError:
            pass
    return {"removed": sorted(removed), "count": len(removed)}


def guarded_page_write(label: str):
    """Decorator for public review/editor functions that mutate one page.

    The same-thread reentrant guard allows composed review operations to call
    other guarded helpers, while automatic processing or another GUI worker on
    the same page remains excluded.
    """
    def deco(fn):
        @wraps(fn)
        def wrapped(page_dir, *args, **kwargs):
            with PageRunGuard(page_dir, f"review:{label}"):
                return fn(page_dir, *args, **kwargs)
        return wrapped
    return deco
