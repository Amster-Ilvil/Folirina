from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import load_json
from .models import PagePair
from .schema_compat import as_dict, normalize_project
from .result_state import resolve_result_state
from .pairing import pair_directories


@dataclass(slots=True)
class RestoredPage:
    page_id: str
    page_root: Path
    pair: PagePair
    project: dict[str, Any] = field(default_factory=dict)
    origin: str = "existing_result"
    warnings: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RestoredSession:
    selected_path: Path
    output_root: Path
    pages: list[RestoredPage]
    source_dir: str = ""
    target_dir: str = ""
    warnings: list[str] = field(default_factory=list)


def _safe_project(path: Path) -> dict[str, Any]:
    """Read only the metadata needed for restore, normalizing legacy schemas lazily.

    Current projects already store ``pair`` / ``artifacts`` / ``meta`` in the
    canonical shape. Avoiding a full compatibility normalization for every page
    materially speeds opening large processed-result folders. Legacy payloads
    still fall back to ``normalize_project`` when those keys are absent.
    """
    try:
        if path.exists():
            data = load_json(path)
            if isinstance(data, dict):
                if any(key in data for key in ("pair", "artifacts", "meta")):
                    return data
                return normalize_project(data)
    except Exception:
        pass
    return {}


def _existing_file(*values: Any) -> Path | None:
    for value in values:
        if not value:
            continue
        try:
            path = Path(str(value)).expanduser()
            if path.exists() and path.is_file():
                return path.resolve()
        except Exception:
            continue
    return None


def _common_parent(paths: list[Path]) -> str:
    if not paths:
        return ""
    try:
        import os
        return str(Path(os.path.commonpath([str(p.parent) for p in paths])))
    except Exception:
        return str(paths[0].parent)


def _local_published_result(page_root: Path) -> Path | None:
    """Cheap page-local publication probe used by processed-only restore.

    Opening an existing result must not parse hundreds of ``project.json`` files
    just to discover that most page workspaces were never completed.  A page is
    considered processed for this UI operation only when it has a page-local
    published ``final_reviewed.png`` or ``final.png``.
    """
    candidates = [page_root / "final_reviewed.png", page_root / "final.png"]
    rows: list[tuple[int, Path]] = []
    for candidate in candidates:
        try:
            if candidate.is_file():
                rows.append((int(candidate.stat().st_mtime_ns), candidate))
        except OSError:
            continue
    if not rows:
        return None
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows[0][1]


def _candidate_page_dirs(selected: Path, *, processed_only: bool = False) -> tuple[Path, list[Path], int]:
    selected = selected.resolve()

    def classify(children: list[Path]) -> tuple[list[Path], int]:
        if not processed_only:
            kept = [p for p in children if p.is_dir() and ((p / "project.json").exists() or (p / "target_original.png").exists())]
            return kept, 0
        kept: list[Path] = []
        skipped = 0
        for path in children:
            if not path.is_dir():
                continue
            if _local_published_result(path) is not None:
                kept.append(path)
            elif (path / "project.json").exists() or (path / "target_original.png").exists():
                skipped += 1
        return kept, skipped

    if (selected / "pages").is_dir():
        pages_root = selected / "pages"
        kept, skipped = classify(list(pages_root.iterdir()))
        return selected, sorted(kept), skipped
    if selected.name == "pages" and selected.is_dir():
        kept, skipped = classify(list(selected.iterdir()))
        return selected.parent, sorted(kept), skipped
    if processed_only:
        if _local_published_result(selected) is not None:
            output_root = selected.parent.parent if selected.parent.name == "pages" else selected.parent
            return output_root, [selected], 0
    elif (selected / "project.json").exists() or (selected / "target_original.png").exists():
        output_root = selected.parent.parent if selected.parent.name == "pages" else selected.parent
        return output_root, [selected], 0
    # Last-resort: accept a directory containing page-like subdirectories.
    kept, skipped = classify(list(selected.iterdir()))
    return selected, sorted(kept), skipped


def scan_existing_results(selected_path: str | Path, *, processed_only: bool = True) -> RestoredSession:
    """Read an existing CLI/Codex output tree without mutating it.

    By default this is intentionally a *processed-result-only* scan: page
    workspaces that do not have a published result are ignored. Opening an
    existing project must stay a cheap inspection operation and must never turn
    into an implicit whole-book pairing pass.

    Project metadata is preferred. Portable page-local ``source_original`` /
    ``target_original`` images are used when the original input paths no longer
    exist. Each restored pair carries its real page id in ``reasons`` so GUI
    workspace lookup remains stable even when a page-local fallback image is used.
    """
    selected = Path(selected_path).expanduser()
    if not selected.exists() or not selected.is_dir():
        raise FileNotFoundError(selected)
    output_root, page_dirs, skipped_unprocessed = _candidate_page_dirs(selected, processed_only=processed_only)
    if not page_dirs:
        raise ValueError("未找到可恢复的页面目录：请选择包含 pages/ 的输出目录，或具体页面目录。")

    restored: list[RestoredPage] = []
    source_files: list[Path] = []
    target_files: list[Path] = []
    # Only original input-path evidence is allowed to seed a future explicit
    # full-book pairing. Page-local portable fallbacks are for inspection only;
    # treating output/pages/... as a SOURCE/TARGET directory would pair the
    # workspace with itself when the user later presses “智能配对”.
    source_input_files: list[Path] = []
    target_input_files: list[Path] = []
    session_warnings: list[str] = []
    for ordinal, page_root in enumerate(page_dirs):
        # Processed-only restore already filtered page directories using two cheap
        # file-existence probes. Parse project metadata only for pages that will
        # actually be shown. This keeps opening a partially processed long book
        # proportional to the number of completed pages, not total page count.
        local_final = _local_published_result(page_root) if processed_only else None
        if processed_only and local_final is None:
            skipped_unprocessed += 1
            continue
        project = _safe_project(page_root / "project.json")
        pair_meta = as_dict(project.get("pair"))
        page_id = page_root.name
        warnings: list[str] = []
        final = local_final
        if not processed_only:
            result_state = resolve_result_state(page_root, as_dict(project.get("artifacts")))
            final = result_state.current
        original_source = _existing_file(pair_meta.get("source_path"))
        original_target = _existing_file(pair_meta.get("target_path"))
        source = original_source or _existing_file(page_root / "source_original.png", page_root / "source_authority_original.png")
        target = original_target or _existing_file(page_root / "target_original.png")
        run_state = {}
        try:
            state_path = page_root / "last_run_state.json"
            if state_path.exists():
                candidate = load_json(state_path)
                run_state = candidate if isinstance(candidate, dict) else {}
        except Exception:
            run_state = {}
        if str(run_state.get("status") or "") == "failed":
            restored_count = int(as_dict(run_state.get("previous_result_restored")).get("restored") or 0)
            warnings.append(
                f"上次处理失败；当前显示的是上次成功结果。已恢复 {restored_count} 个已发布产物。日志：{run_state.get('run_log') or 'run.log'}"
            )
        if source is None or target is None:
            session_warnings.append(f"{page_id}: 缺少 SOURCE/TARGET 原图，已跳过")
            continue
        if final is None:
            warnings.append("没有 final 结果；仅在 processed_only=False 的兼容读取中保留该工作区")
        source_files.append(source); target_files.append(target)
        if original_source is not None:
            source_input_files.append(original_source)
        if original_target is not None:
            target_input_files.append(original_target)
        reasons = [f"restored_page_id:{page_id}", "restored_existing_result", "pairing=restored"]
        pair = PagePair(
            source_path=str(source), target_path=str(target),
            source_index=int(pair_meta.get("source_index", ordinal) or ordinal),
            target_index=int(pair_meta.get("target_index", ordinal) or ordinal),
            confidence=float(pair_meta.get("confidence", 1.0) or 1.0),
            score=float(pair_meta.get("score", 1.0) or 1.0), reasons=reasons,
        )
        restored.append(RestoredPage(page_id=page_id, page_root=page_root.resolve(), pair=pair, project=project, warnings=warnings))

    if skipped_unprocessed:
        session_warnings.append(f"读取时跳过 {skipped_unprocessed} 个尚无已发布结果的页面工作区")
    if not restored:
        if processed_only:
            raise ValueError("找到页面目录，但没有任何可读取的已处理结果。请先处理页面，或选择正确的输出目录。")
        raise ValueError("找到页面目录，但没有任何页面同时具备可读取的 SOURCE 与 TARGET 原图。")
    return RestoredSession(
        selected_path=selected.resolve(), output_root=output_root.resolve(), pages=restored,
        source_dir=_common_parent(source_input_files), target_dir=_common_parent(target_input_files), warnings=session_warnings,
    )


def expand_restored_session_pairs(session: RestoredSession, pairing_config: Any) -> tuple[list[PagePair], list[str], list[str], bool]:
    """Explicitly rebuild the full-book pair list from a restored subset.

    This helper is retained for non-GUI callers/tests, but the GUI must call a
    pairing path only after an explicit user pairing action. Merely opening an
    existing result never invokes this function.
    """
    restored_pairs = [row.pair for row in session.pages]
    if not session.source_dir or not session.target_dir:
        return restored_pairs, [], [], False
    try:
        source_dir = Path(session.source_dir).expanduser()
        target_dir = Path(session.target_dir).expanduser()
        if not source_dir.is_dir() or not target_dir.is_dir():
            return restored_pairs, [], [], False
        pairs, us, ut = pair_directories(source_dir, target_dir, pairing_config)
    except Exception:
        return restored_pairs, [], [], False
    # Never replace a valid restored set with a smaller/empty speculative pair
    # result (portable page-local fallbacks can make source_dir/target_dir point
    # inside output/pages rather than the original book directories).
    if len(pairs) < len(restored_pairs):
        return restored_pairs, [], [], False
    return list(pairs), list(us), list(ut), len(pairs) > len(restored_pairs)
