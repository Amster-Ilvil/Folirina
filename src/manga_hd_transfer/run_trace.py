from __future__ import annotations

import json
import os
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _json_safe(value: Any):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.size <= 64:
                return value.tolist()
            return {"type": "ndarray", "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    return str(value)


class PageRunTrace:
    """Per-page append-only diagnostic log.

    JSONL is intended for precise debugging; ``run.log`` is a compact human
    readable mirror that can be opened directly from the GUI.  Logging failures
    are deliberately swallowed so diagnostics can never break image processing.
    """

    def __init__(self, page_root: str | Path, *, mode: str = "", run_id: str | None = None):
        self.root = Path(page_root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.mode = str(mode or "")
        self.jsonl_path = self.root / "run_trace.jsonl"
        self.text_path = self.root / "run.log"
        self._lock = threading.Lock()
        self._started_perf = time.perf_counter()
        self._sequence = 0

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="milliseconds")

    def event(self, stage: str, /, **payload: Any) -> None:
        safe_payload = {str(k): _json_safe(v) for k, v in payload.items()}
        try:
            message = payload.get("message") or payload.get("reason") or ""
            detail = f" · {message}" if message else ""
            # Sequence and monotonic elapsed time make the stage order explicit
            # even when wall-clock time changes (NTP/time-zone adjustment). Keep
            # both values inside the writer lock so concurrent diagnostics can
            # never publish duplicate/out-of-order sequence numbers.
            with self._lock:
                self._sequence += 1
                row = {
                    "timestamp": self._now(),
                    "run_id": self.run_id,
                    "mode": self.mode,
                    "sequence": int(self._sequence),
                    "elapsed_ms": round((time.perf_counter() - self._started_perf) * 1000.0, 3),
                    "stage": str(stage),
                    **safe_payload,
                }
                encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                human = f"[{row['timestamp']}] [{self.run_id}] [{stage}]{detail}\n"
                with self.jsonl_path.open("a", encoding="utf-8") as fh:
                    fh.write(encoded + "\n")
                    fh.flush()
                with self.text_path.open("a", encoding="utf-8") as fh:
                    fh.write(human)
                    # A few high-value fields make the text log useful without
                    # duplicating the complete structured row.
                    for key in (
                        "selected_mode", "selected_strategy", "ocr_source", "ocr_target",
                        "registration_confidence", "paired_method", "recognized_regions",
                        "target_bubble_id", "source_bubble_id", "region_id",
                        "source_crop_route", "success", "text", "clear_pixels",
                        "max_missing_target_text_ratio", "successful_regions",
                        "failed_regions", "error_type",
                    ):
                        if key in payload:
                            fh.write(f"    {key}: {_json_safe(payload[key])}\n")
                    fh.flush()
        except Exception:
            pass

    def exception(self, stage: str, exc: BaseException, **payload: Any) -> None:
        self.event(
            stage,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback="".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-20000:],
            **payload,
        )


def latest_trace_summary(page_root: str | Path) -> dict[str, Any]:
    root = Path(page_root)
    state = {}
    try:
        p = root / "last_run_state.json"
        if p.exists():
            state = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        state = {}
    return {
        "last_run": state,
        "run_log": str(root / "run.log") if (root / "run.log").exists() else "",
        "run_trace": str(root / "run_trace.jsonl") if (root / "run_trace.jsonl").exists() else "",
    }
