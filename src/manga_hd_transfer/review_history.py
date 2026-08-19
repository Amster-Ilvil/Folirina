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
from pathlib import Path
from typing import Any

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
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


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
    current = _current_overrides(root)
    entry = undo.pop()
    redo = list(hist.get("redo") or [])
    redo.append(_entry(current, f"redo:{entry.get('reason','edit')}"))
    restored = deepcopy(entry.get("state") or {})
    _save_json(root / _OVERRIDES_NAME, restored)
    hist["undo"] = undo
    hist["redo"] = redo[-_MAX_HISTORY:]
    _save_json(root / _HISTORY_NAME, hist)
    return restored


def redo_review_state(page_root: str | Path) -> dict | None:
    root = Path(page_root)
    hist = load_review_history(root)
    redo = list(hist.get("redo") or [])
    if not redo:
        return None
    current = _current_overrides(root)
    entry = redo.pop()
    undo = list(hist.get("undo") or [])
    undo.append(_entry(current, f"undo:{entry.get('reason','edit')}"))
    restored = deepcopy(entry.get("state") or {})
    _save_json(root / _OVERRIDES_NAME, restored)
    hist["undo"] = undo[-_MAX_HISTORY:]
    hist["redo"] = redo
    _save_json(root / _HISTORY_NAME, hist)
    return restored


def review_history_counts(page_root: str | Path) -> tuple[int, int]:
    hist = load_review_history(page_root)
    return len(hist.get("undo") or []), len(hist.get("redo") or [])
