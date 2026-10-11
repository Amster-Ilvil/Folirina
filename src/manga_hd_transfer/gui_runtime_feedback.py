from __future__ import annotations

"""Small, side-effect-free GUI preflight cache and stage elapsed-time bookkeeping.

The cache only records a model-preparation job that *actually completed*. A new
config fingerprint, a failed/cancelled preparation, or timeout forces a fresh
preflight. It never claims that the image inference/transfer is cached.
"""

from dataclasses import dataclass
import hashlib
import time


@dataclass
class RuntimePreparationCache:
    ttl_seconds: float = 600.0
    signature: str = ""
    verified_at: float = 0.0

    @staticmethod
    def key(config) -> str:
        return hashlib.sha256(config.model_dump_json().encode("utf-8")).hexdigest()

    def is_ready(self, signature: str, *, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return bool(self.signature and self.signature == signature and now >= self.verified_at
                    and now - self.verified_at < self.ttl_seconds)

    def record_success(self, signature: str, *, now: float | None = None) -> None:
        self.signature = str(signature)
        self.verified_at = time.monotonic() if now is None else float(now)

    def invalidate(self) -> None:
        self.signature = ""
        self.verified_at = 0.0


@dataclass
class ProcessingStageClock:
    message: str = ""
    started_at: float = 0.0

    def change(self, message: str, *, now: float | None = None) -> None:
        text = str(message)
        if text != self.message:
            self.message = text
            self.started_at = time.monotonic() if now is None else float(now)

    def duration(self, *, now: float | None = None) -> int:
        current = time.monotonic() if now is None else float(now)
        return max(0, int(current - self.started_at)) if self.message else 0

    def reset(self) -> None:
        self.message = ""
        self.started_at = 0.0


def format_processing_heartbeat(*, stage: str, elapsed: int, current: int, total: int,
                                preparing: bool = False) -> str:
    """Format truthful feedback for page, book and model-preparation workers.

    A book progress bar uses a page count (e.g. 1/195), not a percentage.
    Calling it '正在准备' after the real worker has started is misleading.
    """
    if preparing or total <= 0:
        prefix = "正在准备"
    elif total == 100:
        prefix = f"{current}%"
    else:
        prefix = f"{current}/{total}"
    return f"{prefix} · {stage} · 本阶段 {max(0, int(elapsed))}s"
