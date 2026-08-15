from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_utils import load_json
from .models import PagePair
from .schema_compat import as_dict, normalize_project


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
    try:
        if path.exists():
            data = load_json(path)
            if isinstance(data, dict):
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


def _candidate_page_dirs(selected: Path) -> tuple[Path, list[Path]]:
    selected = selected.resolve()
    if (selected / "pages").is_dir():
        pages_root = selected / "pages"
        return selected, sorted([p for p in pages_root.iterdir() if p.is_dir()])
    if selected.name == "pages" and selected.is_dir():
        return selected.parent, sorted([p for p in selected.iterdir() if p.is_dir()])
    if (selected / "project.json").exists() or (selected / "target_original.png").exists():
        # A single page workspace can live outside the standard output/pages tree.
        output_root = selected.parent.parent if selected.parent.name == "pages" else selected.parent
        return output_root, [selected]
    # Last-resort: accept a directory containing page-like subdirectories.
    children = [p for p in selected.iterdir() if p.is_dir() and ((p / "project.json").exists() or (p / "target_original.png").exists())]
    return selected, sorted(children)


def scan_existing_results(selected_path: str | Path) -> RestoredSession:
    """Read an existing CLI/Codex output tree without mutating it.

    Project metadata is preferred.  Portable page-local ``source_original`` /
    ``target_original`` images are used when the original input paths no longer
    exist.  Each restored pair carries its real page id in ``reasons`` so GUI
    workspace lookup remains stable even when a page-local fallback image is used.
    """
    selected = Path(selected_path).expanduser()
    if not selected.exists() or not selected.is_dir():
        raise FileNotFoundError(selected)
    output_root, page_dirs = _candidate_page_dirs(selected)
    if not page_dirs:
        raise ValueError("未找到可恢复的页面目录：请选择包含 pages/ 的输出目录，或具体页面目录。")

    restored: list[RestoredPage] = []
    source_files: list[Path] = []
    target_files: list[Path] = []
    session_warnings: list[str] = []
    for ordinal, page_root in enumerate(page_dirs):
        project = _safe_project(page_root / "project.json")
        pair_meta = as_dict(project.get("pair"))
        page_id = page_root.name
        warnings: list[str] = []
        source = _existing_file(pair_meta.get("source_path"), page_root / "source_original.png", page_root / "source_authority_original.png")
        target = _existing_file(pair_meta.get("target_path"), page_root / "target_original.png")
        final = _existing_file(page_root / "final_reviewed.png", page_root / "final.png", as_dict(project.get("artifacts")).get("final"))
        if source is None or target is None:
            session_warnings.append(f"{page_id}: 缺少 SOURCE/TARGET 原图，已跳过")
            continue
        if final is None:
            warnings.append("没有 final 结果；仍可恢复原图和已有中间产物")
        source_files.append(source); target_files.append(target)
        reasons = [f"restored_page_id:{page_id}", "restored_existing_result"]
        pair = PagePair(
            source_path=str(source), target_path=str(target),
            source_index=int(pair_meta.get("source_index", ordinal) or ordinal),
            target_index=int(pair_meta.get("target_index", ordinal) or ordinal),
            confidence=float(pair_meta.get("confidence", 1.0) or 1.0),
            score=float(pair_meta.get("score", 1.0) or 1.0), reasons=reasons,
        )
        restored.append(RestoredPage(page_id=page_id, page_root=page_root.resolve(), pair=pair, project=project, warnings=warnings))

    if not restored:
        raise ValueError("找到页面目录，但没有任何页面同时具备可读取的 SOURCE 与 TARGET 原图。")
    return RestoredSession(
        selected_path=selected.resolve(), output_root=output_root.resolve(), pages=restored,
        source_dir=_common_parent(source_files), target_dir=_common_parent(target_files), warnings=session_warnings,
    )
