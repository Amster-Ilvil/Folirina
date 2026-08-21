from __future__ import annotations

"""Persistent, mode-safe undo/redo history for review_overrides.json.

The image editor already freezes automatic output, but textual/manual review state
used to be overwritten in place. A mistaken font/layout/text edit therefore had
no structured undo path. This module stores only lightweight JSON snapshots; no
page images or automatic pipeline artifacts are duplicated.
"""

from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from .io_utils import save_json

_HISTORY_NAME = "review_history.json"
_OVERRIDES_NAME = "review_overrides.json"
_MAX_HISTORY = 50


def _load_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return deepcopy(default)


def _save_json(path: Path, obj: Any) -> None:
    # Use the shared unique-temp atomic writer. A fixed ``.tmp`` filename can be
    # trampled by rapid GUI/CLI review actions on the same page.
    save_json(path, obj)


def _capture_pair(root: Path) -> dict[str, tuple[bool, bytes]]:
    out: dict[str, tuple[bool, bytes]] = {}
    for name in (_OVERRIDES_NAME, _HISTORY_NAME):
        path = root / name
        if path.exists() and path.is_file():
            out[name] = (True, path.read_bytes())
        else:
            out[name] = (False, b"")
    return out


def _restore_raw(path: Path, existed: bool, payload: bytes) -> None:
    if not existed:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".history-rollback", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _restore_pair(root: Path, snapshot: dict[str, tuple[bool, bytes]]) -> None:
    for name, (existed, payload) in snapshot.items():
        _restore_raw(root / name, existed, payload)


def _entry(state: dict, reason: str) -> dict:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "reason": str(reason or "edit"),
        "state": deepcopy(state),
    }


def load_review_history(page_root: str | Path) -> dict:
    root = Path(page_root)
    obj = _load_json(root / _HISTORY_NAME, {"schema": "mhd.review.history.v1", "undo": [], "redo": []})
    if not isinstance(obj, dict):
        obj = {"schema": "mhd.review.history.v1", "undo": [], "redo": []}
    obj.setdefault("schema", "mhd.review.history.v1")
    obj["undo"] = list(obj.get("undo") or [])[-_MAX_HISTORY:]
    obj["redo"] = list(obj.get("redo") or [])[-_MAX_HISTORY:]
    return obj


def record_review_state(page_root: str | Path, current_state: dict, reason: str) -> dict:
    root = Path(page_root)
    hist = load_review_history(root)
    undo = list(hist.get("undo") or [])
    # Avoid stacking byte-identical snapshots on repeated Apply clicks.
    if not undo or undo[-1].get("state") != current_state:
        undo.append(_entry(current_state, reason))
    hist["undo"] = undo[-_MAX_HISTORY:]
    hist["redo"] = []
    _save_json(root / _HISTORY_NAME, hist)
    return {"undo": len(hist["undo"]), "redo": 0}


def _current_overrides(root: Path) -> dict:
    obj = _load_json(root / _OVERRIDES_NAME, {})
    return obj if isinstance(obj, dict) else {}


def undo_review_state(page_root: str | Path) -> dict | None:
    root = Path(page_root)
    hist = load_review_history(root)
    undo = list(hist.get("undo") or [])
    if not undo:
        return None
    snapshot = _capture_pair(root)
    current = _current_overrides(root)
    entry = undo.pop()
    redo = list(hist.get("redo") or [])
    redo.append(_entry(current, f"redo:{entry.get('reason','edit')}"))
    restored = deepcopy(entry.get("state") or {})
    try:
        _save_json(root / _OVERRIDES_NAME, restored)
        hist["undo"] = undo
        hist["redo"] = redo[-_MAX_HISTORY:]
        _save_json(root / _HISTORY_NAME, hist)
    except Exception:
        _restore_pair(root, snapshot)
        raise
    return restored


def redo_review_state(page_root: str | Path) -> dict | None:
    root = Path(page_root)
    hist = load_review_history(root)
    redo = list(hist.get("redo") or [])
    if not redo:
        return None
    snapshot = _capture_pair(root)
    current = _current_overrides(root)
    entry = redo.pop()
    undo = list(hist.get("undo") or [])
    undo.append(_entry(current, f"undo:{entry.get('reason','edit')}"))
    restored = deepcopy(entry.get("state") or {})
    try:
        _save_json(root / _OVERRIDES_NAME, restored)
        hist["undo"] = undo[-_MAX_HISTORY:]
        hist["redo"] = redo
        _save_json(root / _HISTORY_NAME, hist)
    except Exception:
        _restore_pair(root, snapshot)
        raise
    return restored


def review_history_counts(page_root: str | Path) -> tuple[int, int]:
    hist = load_review_history(page_root)
    return len(hist.get("undo") or []), len(hist.get("redo") or [])
