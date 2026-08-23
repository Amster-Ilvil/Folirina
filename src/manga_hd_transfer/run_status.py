from __future__ import annotations

"""Mode-neutral page run outcome classification.

Renderers report mode-specific counters in ``project.meta``.  The lifecycle is
responsible for translating those counters plus publication/integrity evidence
into one stable status vocabulary.  This keeps GUI, resume, logs and future
automation from interpreting ``0 applied`` as an unconditional success.
"""

from enum import StrEnum
from typing import Any, Mapping


class PageRunStatus(StrEnum):
    SUCCESS = "success"
    PARTIAL = "partial"
    NOOP = "noop"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTEGRITY_FAILED = "integrity_failed"


def classify_page_run_status(
    *,
    integrity_pass: bool,
    applied_regions: int = 0,
    failed_regions: int = 0,
    changed_pixels: int | None = None,
    passthrough_reason: str = "",
    cancelled: bool = False,
) -> PageRunStatus:
    """Return the canonical terminal status for a completed lifecycle.

    ``changed_pixels`` is deliberately optional: first-time runs have no prior
    snapshot to compare.  In that case a renderer-reported applied count is the
    effect authority.  Passthrough pages are explicit ``skipped`` outcomes, not
    failed/no-op renders.
    """
    if cancelled:
        return PageRunStatus.CANCELLED
    if not bool(integrity_pass):
        return PageRunStatus.INTEGRITY_FAILED
    if str(passthrough_reason or "").strip():
        return PageRunStatus.SKIPPED

    applied = max(0, int(applied_regions or 0))
    failed = max(0, int(failed_regions or 0))
    changed = None if changed_pixels is None else max(0, int(changed_pixels or 0))

    if failed > 0 and (applied > 0 or (changed is not None and changed > 0)):
        return PageRunStatus.PARTIAL
    if applied > 0:
        return PageRunStatus.SUCCESS
    if changed is not None and changed > 0:
        # Some page-level reveal/passthrough-compatible paths publish a changed
        # page without region counters.  Pixel evidence is enough to call it a
        # successful publication, but never when integrity failed above.
        return PageRunStatus.SUCCESS
    return PageRunStatus.NOOP


def status_from_project(project: Any, integrity: Mapping[str, Any], publish_effect: Mapping[str, Any]) -> PageRunStatus:
    meta = dict(getattr(project, "meta", {}) or {})
    return classify_page_run_status(
        integrity_pass=bool(integrity.get("pass")),
        applied_regions=int(publish_effect.get("applied_regions") or 0),
        failed_regions=int(publish_effect.get("failed_regions") or 0),
        changed_pixels=(
            int(publish_effect.get("changed_pixels") or 0)
            if publish_effect.get("pixel_comparison_available")
            else None
        ),
        passthrough_reason=str(meta.get("passthrough_reason") or ""),
        cancelled=bool(meta.get("cancelled")),
    )


__all__ = ["PageRunStatus", "classify_page_run_status", "status_from_project"]
