from __future__ import annotations

"""Book/batch orchestration independent from the page pipeline implementation.

The service receives a ``process_page`` callable rather than importing
``TransferPipeline``.  Resume admission, crash checkpoints, progress emission and
book manifests can therefore be tested without constructing the image pipeline.
"""

from concurrent.futures import ThreadPoolExecutor
import json
import inspect
import logging
import os
import tempfile
from pathlib import Path
import time
from typing import Callable, Type

from .cache import load_completed_page
from .mode_contracts import mode_scoped_config_payload
from .io_utils import save_json, stem_id
from .models import BookProject, PagePair, PageProject
from .project_store import DiskBackedPageList
from .page_memory import release_page_heavy_arrays
from .workspace_cleanup import prune_stage_cache
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


def _load_resume_candidate(page_dir, pair, config, final_path, *, scoped_config_payload=None, allow_compatible_identity: bool = False):
    # Preserve the historical loader call contract for strict resume so tests,
    # plugins and monkeypatches that implement the old keyword-only signature do
    # not break merely because GUI continuation gained an extra policy option.
    if allow_compatible_identity:
        return load_completed_page(
            page_dir, pair, config, final_path,
            scoped_config_payload=scoped_config_payload,
            allow_compatible_identity=True,
        )
    return load_completed_page(
        page_dir, pair, config, final_path,
        scoped_config_payload=scoped_config_payload,
    )


def _prefetch_resume_hits(planned, *, config, enabled: bool, scoped_config_payload=None, allow_compatible_identity: bool = False) -> dict[int, PageProject]:
    hits: dict[int, PageProject] = {}
    if not enabled or not planned:
        return hits
    workers = max(1, min(int(config.batch.prefetch_workers), 8, len(planned)))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mhd-resume") as ex:
            futures = {
                ex.submit(_load_resume_candidate, page_dir, pair, config, final_path, scoped_config_payload=scoped_config_payload, allow_compatible_identity=allow_compatible_identity): idx
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
            hit = _load_resume_candidate(page_dir, pair, config, final_path, scoped_config_payload=scoped_config_payload, allow_compatible_identity=allow_compatible_identity)
            if hit is not None:
                hits[idx] = hit
    return hits



def _resume_window_is_worthwhile(planned, min_project_bytes: int) -> bool:
    """Sample existing project files to avoid threading overhead on tiny resumes."""
    if min_project_bytes <= 0:
        return True
    if not planned:
        return False
    # Evenly sample the book so a partially completed run whose first pages are
    # missing still gets a representative decision.
    sample_count = min(16, len(planned))
    positions = sorted({int(round(i * (len(planned) - 1) / max(1, sample_count - 1))) for i in range(sample_count)})
    sizes: list[int] = []
    for pos in positions:
        project_path = planned[pos][2] / "project.json"
        try:
            sizes.append(int(project_path.stat().st_size))
        except OSError:
            continue
    if not sizes:
        return False
    sizes.sort()
    median = sizes[len(sizes) // 2]
    return median >= int(min_project_bytes)


class _ResumeSlidingWindow:
    """Bounded asynchronous resume admission for long books."""

    def __init__(self, planned, *, config, enabled: bool, window: int, scoped_config_payload=None, allow_compatible_identity: bool = False):
        self._planned = planned
        self._config = config
        self._payload = scoped_config_payload
        self._allow_compatible_identity = bool(allow_compatible_identity)
        self._enabled = bool(enabled and planned and window > 0)
        self._window = max(1, int(window))
        self._executor = None
        self._futures: dict[int, object] = {}
        self._next_submit = 1
        if self._enabled:
            workers = max(1, min(int(config.batch.prefetch_workers), 8, self._window, len(planned)))
            self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="mhd-resume-window")
            self._fill(1)

    def _submit(self, one_based_index: int) -> None:
        if self._executor is None or one_based_index < 1 or one_based_index > len(self._planned):
            return
        idx, pair, page_dir, final_path = self._planned[one_based_index - 1]
        self._futures[idx] = self._executor.submit(
            _load_resume_candidate, page_dir, pair, self._config, final_path,
            scoped_config_payload=self._payload, allow_compatible_identity=self._allow_compatible_identity,
        )

    def _fill(self, current_index: int) -> None:
        if not self._enabled:
            return
        wanted_end = min(len(self._planned), int(current_index) + self._window - 1)
        self._next_submit = max(self._next_submit, int(current_index))
        while self._next_submit <= wanted_end:
            if self._next_submit not in self._futures:
                self._submit(self._next_submit)
            self._next_submit += 1

    def get(self, one_based_index: int) -> PageProject | None:
        if not self._enabled:
            return None
        self._fill(one_based_index)
        fut = self._futures.pop(one_based_index, None)
        hit = None
        if fut is not None:
            try:
                hit = fut.result()
            except Exception:
                hit = None
        self._fill(one_based_index + 1)
        return hit

    def close(self) -> None:
        ex = self._executor
        self._executor = None
        self._futures.clear()
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)


class _DecodeLookahead:
    """One-page CPU/I/O decode lane; never overlaps accelerator/model execution."""

    def __init__(self, planned, *, decode_fn: Callable | None, enabled: bool):
        self._planned = planned
        self._decode_fn = decode_fn
        self._enabled = bool(enabled and decode_fn is not None and len(planned) > 1)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mhd-decode") if self._enabled else None
        self._future = None
        self._index = 0

    def schedule(self, one_based_index: int) -> None:
        if not self._enabled or self._executor is None:
            return
        if one_based_index < 1 or one_based_index > len(self._planned):
            return
        # Exactly one future/result may be resident. Do not queue multiple 4K
        # pages behind a slow renderer; bounded memory is more important than
        # speculative depth.
        if self._future is not None:
            return
        _, pair, _, _ = self._planned[one_based_index - 1]
        self._index = one_based_index
        self._future = self._executor.submit(self._decode_fn, pair)

    def take(self, one_based_index: int):
        if self._future is None or self._index != one_based_index:
            return None
        fut = self._future
        self._future = None
        self._index = 0
        try:
            return fut.result()
        except Exception:
            # Decode prefetch is only an acceleration layer. The normal page path
            # will decode synchronously and surface the canonical error if needed.
            return None

    def close(self) -> None:
        ex = self._executor
        self._executor = None
        fut = self._future
        self._future = None
        if fut is not None:
            fut.cancel()
        if ex is not None:
            ex.shutdown(wait=True, cancel_futures=True)


def _ensure_page_project_file(page_dir: Path, page: PageProject) -> Path:
    """Return the authoritative page project path, creating it only if absent.

    Production page pipelines already commit project.json transactionally.  The
    fallback exists for orchestration tests/custom process_page callables and does
    not rewrite a normal completed page.
    """
    path = Path(page_dir) / "project.json"
    if not path.is_file():
        save_json(path, page.to_dict())
    return path



def _save_book_project_streaming(path: str | Path, book: BookProject) -> None:
    """Atomically serialize a book with at most one page payload in memory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))

    def _encode(value) -> str:
        return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))

    fields_before_pages = [
        ("schema", "manga_hd_translation_transfer.book.v1"),
        ("source_dir", book.source_dir),
        ("target_dir", book.target_dir),
        ("output_dir", book.output_dir),
    ]
    fields_after_pages = [
        ("unmatched_source", book.unmatched_source),
        ("unmatched_target", book.unmatched_target),
        ("meta", book.meta),
    ]
    disk_pages = book.pages if isinstance(book.pages, DiskBackedPageList) else None
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("{")
            first = True
            for key, value in fields_before_pages:
                if not first:
                    fh.write(",")
                first = False
                fh.write(json.dumps(key, ensure_ascii=False)); fh.write(":")
                fh.write(_encode(value))
            fh.write(',"pages":[')
            if disk_pages is not None:
                for idx in range(len(disk_pages)):
                    if idx:
                        fh.write(",")
                    # The page project is already valid authoritative JSON. Copy
                    # it directly instead of deserialize -> to_dict -> re-encode.
                    fh.write(disk_pages.raw_json(idx))
            else:
                for idx, page in enumerate(book.pages):
                    if idx:
                        fh.write(",")
                    fh.write(_encode(page.to_dict()))
            fh.write("]")
            for key, value in fields_after_pages:
                fh.write(","); fh.write(json.dumps(key, ensure_ascii=False)); fh.write(":")
                fh.write(_encode(value))
            fh.write("}")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def _save_qa_summary_streaming(
    path: str | Path, pages, unmatched_source: list[str], unmatched_target: list[str],
) -> None:
    """Write QA rows without building a whole-book Python list for long books."""
    if not isinstance(pages, DiskBackedPageList):
        # Keep the established small-book write path and formatting/observability.
        save_json(path, {
            "pages": [{"page_id": p.page_id, "summary": qa_summary(p.qa), "project": p.artifacts.get("final", "")} for p in pages],
            "unmatched_source": unmatched_source, "unmatched_target": unmatched_target,
        })
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write('{"pages":[')
            for idx in range(len(pages)):
                if idx:
                    fh.write(",")
                # For disk-backed books parse the persisted dict directly rather
                # than reconstructing hundreds of TextBlock/Bubble/PageProject
                # dataclasses merely to count QA severities.
                obj = json.loads(pages.raw_json(idx))
                counts = {"error": 0, "warning": 0, "info": 0}
                for issue in obj.get("qa", []) or []:
                    sev = str((issue or {}).get("severity") or "")
                    if sev in counts:
                        counts[sev] += 1
                row = {
                    "page_id": str(obj.get("page_id") or ""),
                    "summary": {
                        "errors": counts["error"], "warnings": counts["warning"],
                        "info": counts["info"], "pass": counts["error"] == 0,
                    },
                    "project": str((obj.get("artifacts") or {}).get("final") or ""),
                }
                fh.write(json.dumps(row, ensure_ascii=False, default=str, separators=(",", ":")))
            fh.write('],"unmatched_source":')
            fh.write(json.dumps(unmatched_source, ensure_ascii=False, separators=(",", ":")))
            fh.write(',"unmatched_target":')
            fh.write(json.dumps(unmatched_target, ensure_ascii=False, separators=(",", ":")))
            fh.write("}")
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


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
    prefetch_page_images: Callable | None = None,
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
    resume_enabled = bool(resume and config.batch.skip_completed)
    resume_policy = str(getattr(config.batch, "resume_policy", "strict") or "strict").strip().lower()
    allow_compatible_resume = bool(resume_enabled and resume_policy == "continue")
    resume_prefetch_limit = max(0, int(getattr(config.batch, "resume_prefetch_page_limit", 48) or 0))
    use_resume_prefetch = bool(resume_enabled and resume_prefetch_limit > 0 and len(planned) <= resume_prefetch_limit)
    resume_window_size = max(0, int(getattr(config.batch, "resume_prefetch_window", 16) or 0))
    resume_window_min_bytes = max(0, int(getattr(config.batch, "resume_prefetch_min_project_bytes", 16384) or 0))
    resume_window_worthwhile = _resume_window_is_worthwhile(planned, resume_window_min_bytes)
    use_resume_window = bool(
        resume_enabled and not use_resume_prefetch and resume_window_size > 0 and resume_window_worthwhile
    )
    resume_config_payload = mode_scoped_config_payload(config) if resume_enabled else None
    resume_hits = _prefetch_resume_hits(
        planned, config=config, enabled=use_resume_prefetch,
        scoped_config_payload=resume_config_payload,
        allow_compatible_identity=allow_compatible_resume,
    )
    resume_window = _ResumeSlidingWindow(
        planned, config=config, enabled=use_resume_window, window=resume_window_size,
        scoped_config_payload=resume_config_payload,
        allow_compatible_identity=allow_compatible_resume,
    )
    decode_prefetch_depth = max(0, min(1, int(getattr(config.batch, "decode_prefetch_pages", 1) or 0)))
    decode_prefetch_requested = (
        bool(getattr(config.batch, "decode_prefetch_enabled", False))
        and bool(getattr(config.batch, "decode_prefetch_experimental_opt_in", False))
        and decode_prefetch_depth > 0
    )
    # Resume admission may skip arbitrary future pages; avoid decoding full 4K
    # images speculatively on continue-runs. From-scratch runs get one-page
    # look-ahead while the current page performs CPU/model/composition work.
    use_decode_prefetch = bool(decode_prefetch_requested and not resume_enabled and prefetch_page_images is not None)
    decode_lookahead = _DecodeLookahead(planned, decode_fn=prefetch_page_images, enabled=use_decode_prefetch)
    try:
        proc_params = inspect.signature(process_page).parameters.values()
        proc_params = list(proc_params)
        process_accepts_prefetched = any(p.name == "prefetched_images" for p in proc_params) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in proc_params
        )
        process_accepts_progress = any(p.name == "progress_cb" for p in proc_params) or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in proc_params
        )
    except (TypeError, ValueError):
        process_accepts_prefetched = True
        process_accepts_progress = True

    stream_threshold = max(0, int(getattr(config.batch, "stream_book_page_threshold", 96) or 0))
    use_disk_backed_pages = bool(stream_threshold > 0 and len(planned) >= stream_threshold)
    pages: list[PageProject] = []
    completed_project_paths: list[Path] = []
    completed_pairs: list[PagePair] = []
    failures: list[dict] = []
    resumed = 0
    compatible_resumed = 0
    completed_count = 0
    started = time.perf_counter()
    cancelled = False
    released_array_bytes = 0
    pruned_cache_bytes = 0
    pruned_cache_pages = 0
    checkpoint_every = max(1, int(getattr(config.batch, "checkpoint_every", 4) or 1))
    checkpoint_max_interval = max(0.0, float(getattr(config.batch, "checkpoint_max_interval_seconds", 15.0) or 0.0))
    checkpoint_writes = 0
    manifest_writes = 0
    last_checkpoint_index = 0
    last_checkpoint_at = started

    route_counts: dict[str, int] = {}
    stage_cache_hits = 0
    skipped_count = 0
    skipped_pages: list[dict] = []
    page_management_updates: list[dict] = []
    qa_errors = 0
    qa_warnings = 0

    def _retire_old_stage_cache(current_index: int) -> None:
        nonlocal pruned_cache_bytes, pruned_cache_pages
        retain = max(0, int(getattr(config.cache, "retain_recent_page_caches", 0) or 0))
        if retain <= 0 or current_index <= retain:
            return
        old_index = current_index - retain - 1
        if not (0 <= old_index < len(planned)):
            return
        old_page_dir = planned[old_index][2]
        result = prune_stage_cache(old_page_dir, preserve_review_pages=True)
        if bool(result.get("pruned")):
            pruned_cache_pages += 1
            pruned_cache_bytes += int(result.get("bytes_freed") or 0)

    def emit(done: int, total: int, pair: PagePair | None, status: str, cache_hit: bool = False, message: str = ""):
        if progress_cb is None:
            return
        try:
            progress_cb(done, total, pair, status, cache_hit, message)
        except TypeError:
            progress_cb(done, total, status)

    def save_checkpoint(
        last_index: int, *, current_pair: PagePair | None = None,
        status: str = "running", force: bool = False,
    ) -> bool:
        nonlocal checkpoint_writes, last_checkpoint_index, last_checkpoint_at
        now = time.perf_counter()
        pages_due = int(last_index) - int(last_checkpoint_index) >= checkpoint_every
        time_due = checkpoint_max_interval > 0.0 and (now - last_checkpoint_at) >= checkpoint_max_interval
        if not force and not pages_due and not time_due:
            return False
        save_json(output / "batch_checkpoint.json", {
            "last_index": int(last_index), "total": len(pairs),
            "completed_pages": int(completed_count), "resumed": int(resumed),
            "failed": list(failures), "status": str(status),
            "current_target": str(current_pair.target_path) if current_pair is not None else "",
            "elapsed_seconds": round(now - started, 3),
        })
        checkpoint_writes += 1
        last_checkpoint_index = max(last_checkpoint_index, int(last_index))
        last_checkpoint_at = now
        return True

    def save_manifest(processed: int) -> None:
        nonlocal manifest_writes
        save_json(output / "batch_manifest.json", {
            "processed": int(processed), "total": len(pairs), "resumed": resumed,
            "failed": failures, "cancelled": cancelled,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        })
        manifest_writes += 1

    def record_page(page: PageProject, page_dir: Path, pair: PagePair) -> None:
        nonlocal completed_count, stage_cache_hits, skipped_count, qa_errors, qa_warnings
        project_path = _ensure_page_project_file(page_dir, page)
        completed_project_paths.append(project_path)
        completed_pairs.append(pair)
        completed_count += 1
        if not use_disk_backed_pages:
            pages.append(page)

        route = str((page.meta or {}).get("registration_route") or page.registration.method)
        route_counts[route] = route_counts.get(route, 0) + 1
        stage_cache_hits += sum(1 for v in ((page.meta or {}).get("cache") or {}).values() if v == "hit")
        qa_errors += sum(1 for q in page.qa if q.severity == "error")
        qa_warnings += sum(1 for q in page.qa if q.severity == "warning")
        pm = (page.meta or {}).get("page_management")
        if pm:
            page_management_updates.append({"page_id": page.page_id, "page_management": pm})
        if bool((page.meta or {}).get("passthrough")):
            skipped_count += 1
            skipped_pages.append({
                "page_id": page.page_id,
                "target": page.pair.target_path,
                "page_management": pm or {},
            })

    try:
        for idx, pair, page_dir, final_path in planned:
            if cancel_cb is not None and cancel_cb():
                cancelled = True
                emit(idx - 1, len(pairs), pair, "cancelled", False, "用户取消")
                save_checkpoint(idx - 1, current_pair=pair, status="cancelled", force=True)
                break
            prefetched_images = decode_lookahead.take(idx) if use_decode_prefetch else None

            requested_mark = resolve_mark(page_marks, pair) if page_marks is not None else None
            if resume_enabled:
                # Do not tell the GUI/log that an already-completed page is being
                # processed before resume admission has even been checked.
                emit(idx, len(pairs), pair, "checking", False, "检查已有结果")
                if use_resume_prefetch:
                    cached = resume_hits.pop(idx, None)
                elif use_resume_window:
                    cached = resume_window.get(idx)
                else:
                    cached = _load_resume_candidate(
                        page_dir, pair, config, final_path,
                        scoped_config_payload=resume_config_payload,
                        allow_compatible_identity=allow_compatible_resume,
                    )
                if cached is not None and not _resume_mark_allows_cached(cached, requested_mark):
                    cached = None
                if cached is not None:
                    resumed += 1
                    compatible_hit = bool((cached.meta or {}).get("batch_resume_compatible_hit"))
                    if compatible_hit:
                        compatible_resumed += 1
                    record_page(cached, page_dir, pair)
                    logger.info(
                        "Resume skip page %d/%d: %s policy=%s compatible=%s",
                        idx, len(pairs), Path(pair.target_path).name, resume_policy, compatible_hit,
                    )
                    emit(idx, len(pairs), pair, "completed", True, "继续：保留已有结果，跳过")
                    save_checkpoint(idx, current_pair=pair, status="resumed")
                    _retire_old_stage_cache(idx)
                    continue

            logger.info("Processing page %d/%d: %s", idx, len(pairs), Path(pair.target_path).name)
            emit(idx, len(pairs), pair, "running", False, "正在处理")
            try:
                if use_decode_prefetch and idx < len(planned):
                    next_pair = planned[idx][1]
                    next_mark = resolve_mark(page_marks, next_pair) if page_marks is not None else None
                    if next_mark is None or next_mark.should_process:
                        decode_lookahead.schedule(idx + 1)
                process_kwargs = {"page_mark": requested_mark, "cancel_cb": cancel_cb}
                if prefetched_images is not None and process_accepts_prefetched:
                    process_kwargs["prefetched_images"] = prefetched_images
                if process_accepts_progress:
                    def _page_stage_progress(percent, stage, message, _idx=idx, _pair=pair):
                        detail = str(message or stage or "正在处理")
                        emit(_idx, len(pairs), _pair, "running", False, f"{int(percent)}% · {detail}")
                    process_kwargs["progress_cb"] = _page_stage_progress
                page = process_page(pair, page_dir, final_path, **process_kwargs)
                if bool(getattr(config.runtime, "detach_completed_page_arrays", True)):
                    released_array_bytes += release_page_heavy_arrays(page)
                record_page(page, page_dir, pair)
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
                save_checkpoint(idx - 1, current_pair=pair, status="cancelled", force=True)
                break
            except Exception as exc:
                row = {
                    "index": idx, "source": pair.source_path, "target": pair.target_path,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                failures.append(row)
                emit(idx, len(pairs), pair, "failed", False, row["error"])
                if config.batch.stop_on_error:
                    save_checkpoint(idx, current_pair=pair, status="failed", force=True)
                    raise

            _retire_old_stage_cache(idx)
            save_checkpoint(
                idx, current_pair=pair,
                status="failed" if failures and failures[-1].get("index") == idx else "completed",
            )
            if config.runtime.release_cache_every > 0 and idx % config.runtime.release_cache_every == 0:
                empty_accelerator_cache(config.runtime.device)
            if config.batch.save_manifest_every > 0 and idx % config.batch.save_manifest_every == 0:
                save_manifest(idx)
    finally:
        resume_window.close()
        decode_lookahead.close()

    elapsed = time.perf_counter() - started
    book_pages = (
        DiskBackedPageList(completed_project_paths, completed_pairs)
        if use_disk_backed_pages else pages
    )
    book = BookProject(
        source_dir=str(source_dir), target_dir=str(target_dir), output_dir=str(output), pages=book_pages,
        unmatched_source=unmatched_source, unmatched_target=unmatched_target,
        meta={
            "page_count": int(completed_count), "paired_count": len(pairs), "resumed_count": resumed,
            "compatible_resumed_count": int(compatible_resumed), "resume_policy": resume_policy,
            "skipped_count": int(skipped_count),
            "skipped_pages": skipped_pages,
            "page_management_updates": page_management_updates,
            "failed_count": len(failures), "failures": failures, "cancelled": cancelled,
            "elapsed_seconds": round(elapsed, 3), "runtime": runtime_summary(config.runtime.device),
            "registration_routes": route_counts, "stage_cache_hits": int(stage_cache_hits),
            "long_run_memory_released_bytes": int(released_array_bytes),
            "stage_cache_pruned_pages": int(pruned_cache_pages),
            "stage_cache_pruned_bytes": int(pruned_cache_bytes),
            # Compatibility: this retains the old meaning of eager all-page prefetch.
            "resume_prefetch": bool(use_resume_prefetch),
            "resume_prefetch_page_limit": int(resume_prefetch_limit),
            "resume_window_prefetch": bool(use_resume_window),
            "resume_prefetch_window": int(resume_window_size),
            "resume_prefetch_min_project_bytes": int(resume_window_min_bytes),
            "resume_window_worthwhile": bool(resume_window_worthwhile),
            "streaming_executor": bool(use_disk_backed_pages),
            "stream_book_page_threshold": int(stream_threshold),
            "disk_backed_page_count": int(len(book_pages)) if use_disk_backed_pages else 0,
            "decode_prefetch": bool(use_decode_prefetch),
            "decode_prefetch_pages": int(decode_prefetch_depth) if use_decode_prefetch else 0,
            "checkpoint_every": int(checkpoint_every),
            "checkpoint_max_interval_seconds": float(checkpoint_max_interval),
            "manifest_every": int(config.batch.save_manifest_every),
            "checkpoint_writes_before_final": int(checkpoint_writes),
            "manifest_writes_before_final": int(manifest_writes),
            "qa_errors": int(qa_errors),
            "qa_warnings": int(qa_warnings),
        },
    )
    _save_book_project_streaming(output / "book_project.json", book)
    save_manifest(completed_count + len(failures))
    _save_qa_summary_streaming(output / "qa_summary.json", book.pages, unmatched_source, unmatched_target)
    save_checkpoint(
        len(pairs) if not cancelled else min(len(pairs), completed_count + len(failures)),
        status="cancelled" if cancelled else "completed", force=True,
    )
    return book


__all__ = [
    "_planned_book_pages", "_resume_mark_allows_cached", "_prefetch_resume_hits",
    "_save_book_project_streaming", "_save_qa_summary_streaming",
    "run_book_orchestration",
]
