from __future__ import annotations

"""Book/batch orchestration independent from the page pipeline implementation.

The service receives a ``process_page`` callable rather than importing
``TransferPipeline``.  Resume admission, crash checkpoints, progress emission and
book manifests can therefore be tested without constructing the image pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
import time
from typing import Callable, Type

from .cache import load_completed_page
from .io_utils import save_json, stem_id
from .models import BookProject, PagePair, PageProject
from .page_management import PageMark, resolve_mark
from .pairing import pair_directories
from .qa import qa_summary
from .runtime import empty_accelerator_cache, runtime_summary

logger = logging.getLogger(__name__)


def _planned_book_pages(pairs: list[PagePair], output: Path) -> list[tuple[int, PagePair, Path, Path]]:
    pages_root = output / "pages"
    final_root = output / "final"
    pages_root.mkdir(parents=True, exist_ok=True)
    final_root.mkdir(parents=True, exist_ok=True)
    planned: list[tuple[int, PagePair, Path, Path]] = []
    used_final_names: set[str] = set()
    for idx, pair in enumerate(pairs, start=1):
        stem = Path(pair.target_path).stem
        target_name = stem + ".png"
        if target_name.casefold() in used_final_names:
            target_name = f"{stem}__{idx:04d}.png"
            salt = 2
            while target_name.casefold() in used_final_names:
                target_name = f"{stem}__{idx:04d}_{salt}.png"
                salt += 1
        used_final_names.add(target_name.casefold())
        planned.append((idx, pair, pages_root / stem_id(pair.target_path), final_root / target_name))
    return planned


def _resume_mark_allows_cached(cached: PageProject, requested_mark: PageMark | None) -> bool:
    if requested_mark is None or requested_mark.origin == "default":
        return True
    cached_pm = (cached.meta or {}).get("page_management")
    cached_passthrough = bool((cached.meta or {}).get("passthrough"))
    if requested_mark.should_process:
        cached_reason = str((cached.meta or {}).get("passthrough_reason") or "")
        return not cached_passthrough or cached_reason == "source_no_transferable_text"
    if not cached_passthrough:
        return False
    if requested_mark.origin == "manual":
        cached_type = PageMark.from_dict(cached_pm).page_type if cached_pm else ""
        return cached_type == requested_mark.page_type
    return True


def _prefetch_resume_hits(planned, *, config, enabled: bool) -> dict[int, PageProject]:
    hits: dict[int, PageProject] = {}
    if not enabled or not planned:
        return hits
    workers = max(1, min(int(config.batch.prefetch_workers), 8, len(planned)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mhd-resume") as ex:
            futures = {
                ex.submit(load_completed_page, page_dir, pair, config, final_path): idx
                for idx, pair, page_dir, final_path in planned
            }
            for fut, idx in [(f, i) for f, i in futures.items()]:
                try:
                    hit = fut.result()
                except Exception:
                    hit = None
                if hit is not None:
                    hits[idx] = hit
    else:
        for idx, pair, page_dir, final_path in planned:
            hit = load_completed_page(page_dir, pair, config, final_path)
            if hit is not None:
                hits[idx] = hit
    return hits


def run_book_orchestration(
    *,
    config,
    process_page: Callable,
    cancelled_exception: Type[BaseException],
    source_dir: str | Path,
    target_dir: str | Path,
    output_dir: str | Path,
    progress_cb=None,
    cancel_cb=None,
    resume: bool | None = None,
    pairs_override: list[PagePair] | None = None,
    page_marks: dict | None = None,
) -> BookProject:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    resume = config.batch.resume if resume is None else bool(resume)

    if pairs_override is None:
        pairs, unmatched_source, unmatched_target = pair_directories(source_dir, target_dir, config.pairing)
    else:
        pairs = list(pairs_override)
        unmatched_source, unmatched_target = [], []

    planned = _planned_book_pages(pairs, output)
    resume_hits = _prefetch_resume_hits(
        planned, config=config, enabled=bool(resume and config.batch.skip_completed)
    )
    pages: list[PageProject] = []
    failures: list[dict] = []
    resumed = 0
    started = time.perf_counter()
    cancelled = False

    def emit(done: int, total: int, pair: PagePair | None, status: str, cache_hit: bool = False, message: str = ""):
        if progress_cb is None:
            return
        try:
            progress_cb(done, total, pair, status, cache_hit, message)
        except TypeError:
            progress_cb(done, total, status)

    def save_checkpoint(last_index: int, *, current_pair: PagePair | None = None, status: str = "running"):
        save_json(output / "batch_checkpoint.json", {
            "last_index": int(last_index), "total": len(pairs),
            "completed_pages": len(pages), "resumed": int(resumed),
            "failed": list(failures), "status": str(status),
            "current_target": str(current_pair.target_path) if current_pair is not None else "",
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })

    for idx, pair, page_dir, final_path in planned:
        if cancel_cb is not None and cancel_cb():
            cancelled = True
            emit(idx - 1, len(pairs), pair, "cancelled", False, "用户取消")
            save_checkpoint(idx - 1, current_pair=pair, status="cancelled")
            break
        logger.info("Processing page %d/%d: %s", idx, len(pairs), Path(pair.target_path).name)
        emit(idx, len(pairs), pair, "running", False, "正在处理")

        requested_mark = resolve_mark(page_marks, pair) if page_marks is not None else None
        if resume and config.batch.skip_completed:
            cached = resume_hits.get(idx)
            if cached is not None and not _resume_mark_allows_cached(cached, requested_mark):
                cached = None
            if cached is not None:
                resumed += 1
                pages.append(cached)
                emit(idx, len(pairs), pair, "completed", True, "继续：已完成页面，跳过")
                save_checkpoint(idx, current_pair=pair, status="resumed")
                continue

        try:
            page = process_page(
                pair, page_dir, final_path,
                page_mark=requested_mark, cancel_cb=cancel_cb,
            )
            pages.append(page)
            if bool((page.meta or {}).get("passthrough")):
                pm = (page.meta or {}).get("page_management", {}) or {}
                reason = str((page.meta or {}).get("passthrough_reason") or "")
                if reason == "source_no_transferable_text":
                    emit(idx, len(pairs), pair, "skipped", False, "无需替换 · 中文页无气泡/文本框")
                else:
                    label = str(pm.get("label") or pm.get("page_type") or "跳过")
                    emit(idx, len(pairs), pair, "skipped", False, f"跳过 · {label}")
            else:
                emit(idx, len(pairs), pair, "completed", False, page.registration.method)
        except cancelled_exception:
            cancelled = True
            emit(idx - 1, len(pairs), pair, "cancelled", False, "用户停止")
            save_checkpoint(idx - 1, current_pair=pair, status="cancelled")
            break
        except Exception as exc:
            row = {
                "index": idx, "source": pair.source_path, "target": pair.target_path,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(row)
            emit(idx, len(pairs), pair, "failed", False, row["error"])
            if config.batch.stop_on_error:
                save_checkpoint(idx, current_pair=pair, status="failed")
                raise

        save_checkpoint(
            idx, current_pair=pair,
            status="failed" if failures and failures[-1].get("index") == idx else "completed",
        )
        if config.runtime.release_cache_every > 0 and idx % config.runtime.release_cache_every == 0:
            empty_accelerator_cache(config.runtime.device)
        if config.batch.save_manifest_every > 0 and idx % config.batch.save_manifest_every == 0:
            save_json(output / "batch_manifest.json", {
                "processed": idx, "total": len(pairs), "resumed": resumed,
                "failed": failures, "cancelled": cancelled,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            })

    elapsed = time.perf_counter() - started
    route_counts: dict[str, int] = {}
    stage_cache_hits = 0
    skipped_pages = [p for p in pages if bool((p.meta or {}).get("passthrough"))]
    for page in pages:
        route = str((page.meta or {}).get("registration_route") or page.registration.method)
        route_counts[route] = route_counts.get(route, 0) + 1
        stage_cache_hits += sum(1 for v in ((page.meta or {}).get("cache") or {}).values() if v == "hit")
    book = BookProject(
        source_dir=str(source_dir), target_dir=str(target_dir), output_dir=str(output), pages=pages,
        unmatched_source=unmatched_source, unmatched_target=unmatched_target,
        meta={
            "page_count": len(pages), "paired_count": len(pairs), "resumed_count": resumed,
            "skipped_count": len(skipped_pages),
            "skipped_pages": [
                {
                    "page_id": p.page_id,
                    "target": p.pair.target_path,
                    "page_management": (p.meta or {}).get("page_management", {}),
                } for p in skipped_pages
            ],
            "failed_count": len(failures), "failures": failures, "cancelled": cancelled,
            "elapsed_seconds": round(elapsed, 3), "runtime": runtime_summary(config.runtime.device),
            "registration_routes": route_counts, "stage_cache_hits": stage_cache_hits,
            "qa_errors": sum(1 for p in pages for q in p.qa if q.severity == "error"),
            "qa_warnings": sum(1 for p in pages for q in p.qa if q.severity == "warning"),
        },
    )
    save_json(output / "book_project.json", book.to_dict())
    save_json(output / "batch_manifest.json", {
        "processed": len(pages) + len(failures), "total": len(pairs), "resumed": resumed,
        "failed": failures, "cancelled": cancelled, "elapsed_seconds": round(elapsed, 3),
    })
    save_json(output / "qa_summary.json", {
        "pages": [{"page_id": p.page_id, "summary": qa_summary(p.qa), "project": p.artifacts.get("final", "")} for p in pages],
        "unmatched_source": unmatched_source, "unmatched_target": unmatched_target,
    })
    save_checkpoint(
        len(pairs) if not cancelled else min(len(pairs), len(pages) + len(failures)),
        status="cancelled" if cancelled else "completed",
    )
    return book


__all__ = [
    "_planned_book_pages", "_resume_mark_allows_cached", "_prefetch_resume_hits",
    "run_book_orchestration",
]
