from __future__ import annotations

"""Persistent Folirina project workspace.

This module intentionally contains no Qt imports.  A project folder is portable:
all paths owned by the project are stored relative to the project root, while
legacy/external paths remain explicit and are never silently copied on restore.
"""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import os
import re
import shutil
import tempfile
import uuid
import zipfile

from .io_utils import load_json, save_json
from .models import PagePair

SCHEMA_VERSION = 1
PROJECT_FORMAT = "folirina-project"
PROJECT_FILE = ".folirina-project.json"
SESSION_FILE = "metadata/session.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _safe_component(value: str, fallback: str = "project") -> str:
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]+', "_", str(value or "")).strip(" .")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:120] or fallback


@dataclass(frozen=True)
class FolirinaProjectInfo:
    project_id: str
    name: str
    path: str
    created_at: str = ""
    updated_at: str = ""


class ProjectWorkspaceManager:
    """Own one workspace root and at most one active Folirina project."""

    LAYOUT = (
        "inputs/source",
        "inputs/target",
        "output/pages",
        "output/final",
        "exports",
        "metadata",
        "logs",
        "cache",
    )

    def __init__(self, workspace_root: str | Path | None = None):
        default = Path.home() / "Documents" / "Folirina Workspace"
        self.workspace_root = Path(workspace_root).expanduser() if workspace_root else default
        self.active_project: Path | None = None

    @property
    def projects_root(self) -> Path:
        # New projects live directly in the user-selected workspace directory.
        # Legacy <workspace>/projects/<name> projects remain discoverable.
        return self.workspace_root

    def set_workspace_root(self, path: str | Path, *, create: bool = True) -> Path:
        root = Path(path).expanduser().resolve()
        if create:
            root.mkdir(parents=True, exist_ok=True)
        self.workspace_root = root
        if self.active_project is not None:
            try:
                self.active_project.resolve().relative_to(root)
            except (ValueError, OSError):
                self.active_project = None
        return root

    def list_projects(self) -> list[FolirinaProjectInfo]:
        if not self.projects_root.exists():
            return []
        rows: list[FolirinaProjectInfo] = []
        roots = (self.projects_root, self.projects_root / "projects")
        for root in roots:
            if not root.is_dir():
                continue
            for folder in root.iterdir():
                if not folder.is_dir():
                    continue
                info = self._read_info(folder)
                if info is not None:
                    rows.append(info)
        rows.sort(key=lambda row: (row.updated_at, row.name.casefold()), reverse=True)
        return rows

    def create_project(self, name: str) -> FolirinaProjectInfo:
        self.projects_root.mkdir(parents=True, exist_ok=True)
        display = str(name or "").strip() or "未命名项目"
        target = self._unique_project_dir(display)
        target.mkdir(parents=True)
        self._ensure_layout(target)
        now = _now_iso()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "format": PROJECT_FORMAT,
            "project_id": uuid.uuid4().hex,
            "name": display,
            "created_at": now,
            "updated_at": now,
            "active_stage": "workspace",
            "inputs": {"source": "inputs/source", "target": "inputs/target"},
            "output": "output",
        }
        save_json(target / PROJECT_FILE, payload)
        self.active_project = target
        return self._info_from_payload(target, payload)

    def open_project(self, path: str | Path) -> FolirinaProjectInfo:
        project = Path(path).expanduser().resolve()
        info = self._read_info(project)
        if info is None:
            raise ValueError(f"不是有效的 Folirina 项目：{project}")
        self._ensure_layout(project)
        self.active_project = project
        return info

    def import_project_folder(self, path: str | Path, *, copy_into_workspace: bool = True) -> FolirinaProjectInfo:
        source = Path(path).expanduser().resolve()
        info = self._read_info(source)
        if info is None:
            raise ValueError(f"不是有效的 Folirina 项目：{source}")
        if not copy_into_workspace:
            return self.open_project(source)
        try:
            relative = source.relative_to(self.projects_root.resolve())
            if len(relative.parts) == 1 or (len(relative.parts) == 2 and relative.parts[0] == "projects"):
                return self.open_project(source)
        except (ValueError, OSError):
            pass
        self.projects_root.mkdir(parents=True, exist_ok=True)
        target = self._unique_project_dir(info.name)
        shutil.copytree(source, target, copy_function=shutil.copy2)
        payload = self._project_payload(target)
        payload["project_id"] = uuid.uuid4().hex
        payload["imported_from_project_id"] = info.project_id
        payload["created_at"] = _now_iso()
        payload["updated_at"] = payload["created_at"]
        save_json(target / PROJECT_FILE, payload)
        return self.open_project(target)

    def delete_project(self, path: str | Path) -> None:
        project = Path(path).expanduser().resolve()
        root = self.projects_root.resolve()
        try:
            relative = project.relative_to(root)
            if not (len(relative.parts) == 1 or (len(relative.parts) == 2 and relative.parts[0] == "projects")):
                raise ValueError("不是当前工作区的顶层项目目录")
        except ValueError as exc:
            raise ValueError("只能删除当前工作区中的命名项目目录") from exc
        if self._read_info(project) is None:
            raise ValueError("目标目录不是有效 Folirina 项目")
        if self.active_project is not None and self.active_project.resolve() == project:
            self.active_project = None
        shutil.rmtree(project)

    def export_project_backup(self, destination: str | Path, *, include_cache: bool = False) -> Path:
        """Create a portable project ZIP. Cache is excluded by default."""
        project = self._require_project().resolve()
        target = Path(destination).expanduser()
        if target.suffix.lower() != ".zip":
            target = target.with_suffix(".zip")
        target = target.resolve()
        try:
            target.relative_to(project)
        except ValueError:
            pass
        else:
            raise ValueError("项目备份不能写入项目目录内部")
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{target.name}.tmp-{uuid.uuid4().hex[:8]}")
        if temp.exists():
            temp.unlink()
        try:
            with zipfile.ZipFile(temp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in sorted(project.rglob("*")):
                    if not path.is_file():
                        continue
                    rel = path.relative_to(project)
                    if not include_cache and rel.parts and rel.parts[0] == "cache":
                        continue
                    if any(part.startswith(".output.import-") or part.startswith(".output.old-") for part in rel.parts):
                        continue
                    archive.write(path, arcname=str(Path(project.name) / rel))
            os.replace(temp, target)
        except Exception:
            if temp.exists():
                temp.unlink(missing_ok=True)
            raise
        return target

    def import_project_archive(self, archive_path: str | Path) -> FolirinaProjectInfo:
        """Safely import a Folirina project backup ZIP into the workspace."""
        source = Path(archive_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(str(source))
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="folirina-import-", dir=str(self.workspace_root)) as td:
            temp_root = Path(td)
            with zipfile.ZipFile(source, "r") as archive:
                for info in archive.infolist():
                    name = str(info.filename or "").replace("\\", "/")
                    candidate = (temp_root / name).resolve()
                    try:
                        candidate.relative_to(temp_root.resolve())
                    except ValueError as exc:
                        raise ValueError(f"备份包含不安全路径：{info.filename}") from exc
                archive.extractall(temp_root)
            manifests = list(temp_root.rglob(PROJECT_FILE))
            if len(manifests) != 1:
                raise ValueError("备份中必须且只能包含一个 Folirina 项目")
            project_root = manifests[0].parent
            return self.import_project_folder(project_root, copy_into_workspace=True)

    def audit_project(self) -> dict[str, Any]:
        """Return a lightweight integrity report without loading image pixels."""
        project = self._require_project()
        issues: list[str] = []
        info = self._read_info(project)
        if info is None:
            issues.append("项目清单无效")
        for rel in self.LAYOUT:
            if not (project / rel).exists():
                issues.append(f"缺少目录：{rel}")
        pair_count = 0
        missing_pair_files = 0
        try:
            session = self.load_session()
            pairs = list(session.get("pairs") or [])
            pair_count = len(pairs)
            for pair in pairs:
                for value in (getattr(pair, "source_path", ""), getattr(pair, "target_path", "")):
                    if value and not Path(value).exists():
                        missing_pair_files += 1
            if missing_pair_files:
                issues.append(f"配对记录中有 {missing_pair_files} 个文件路径不存在")
        except Exception as exc:
            issues.append(f"session 读取失败：{exc}")
        output_pages = 0
        page_root = project / "output" / "pages"
        if page_root.exists():
            try:
                output_pages = sum(1 for item in page_root.iterdir() if item.is_dir())
            except OSError as exc:
                issues.append(f"输出目录读取失败：{exc}")
        return {
            "ok": not issues,
            "issues": issues,
            "project": str(project),
            "pair_count": pair_count,
            "output_pages": output_pages,
            "active_stage": self.active_stage(),
        }

    def project_context(self) -> dict[str, str]:
        if self.active_project is None:
            return {}
        info = self._read_info(self.active_project)
        return asdict(info) if info is not None else {}

    def active_stage(self) -> str:
        if self.active_project is None:
            return ""
        try:
            payload = self._project_payload(self.active_project)
        except Exception:
            return ""
        return str(payload.get("active_stage") or "").strip()

    @property
    def source_dir(self) -> Path | None:
        return None if self.active_project is None else self.active_project / "inputs" / "source"

    @property
    def target_dir(self) -> Path | None:
        return None if self.active_project is None else self.active_project / "inputs" / "target"

    @property
    def output_dir(self) -> Path | None:
        return None if self.active_project is None else self.active_project / "output"

    @property
    def export_dir(self) -> Path | None:
        return None if self.active_project is None else self.active_project / "exports"

    def import_input_directory(self, kind: str, source_dir: str | Path) -> Path:
        """Copy an input tree into the active portable project.

        The copy is replaced transactionally at directory granularity: data is
        copied to a sibling staging directory first, then swapped in.  Existing
        project input survives if copying fails midway.
        """
        if kind not in {"source", "target"}:
            raise ValueError(f"unsupported input kind: {kind}")
        project = self._require_project()
        source = Path(source_dir).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(str(source))
        destination = project / "inputs" / kind
        staging = destination.with_name(f".{destination.name}.import-{uuid.uuid4().hex[:8]}")
        backup = destination.with_name(f".{destination.name}.old-{uuid.uuid4().hex[:8]}")
        if staging.exists():
            shutil.rmtree(staging)
        try:
            shutil.copytree(source, staging, copy_function=shutil.copy2)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staging, destination)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        self.touch(active_stage="pages")
        return destination

    def bind_existing_output(self, source_output: str | Path) -> Path:
        """Copy an existing Folirina output tree into the active project output."""
        project = self._require_project()
        source = Path(source_output).expanduser().resolve()
        if not source.is_dir():
            raise NotADirectoryError(str(source))
        destination = project / "output"
        staging = project / f".output.import-{uuid.uuid4().hex[:8]}"
        backup = project / f".output.old-{uuid.uuid4().hex[:8]}"
        try:
            shutil.copytree(source, staging, copy_function=shutil.copy2)
            if destination.exists():
                os.replace(destination, backup)
            os.replace(staging, destination)
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            if staging.exists():
                shutil.rmtree(staging, ignore_errors=True)
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        self._ensure_layout(project)
        self.touch(active_stage="pages")
        return destination

    def save_session(
        self,
        *,
        source_dir: str = "",
        target_dir: str = "",
        output_dir: str = "",
        pairs: Iterable[PagePair] = (),
        selected_index: int = 0,
        page_marks: dict[str, Any] | None = None,
        book_pairing_ready: bool = False,
        restored_processed_only: bool = False,
        config: dict[str, Any] | None = None,
        active_stage: str | None = None,
    ) -> None:
        project = self._require_project()
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_dir": self._encode_path(source_dir),
            "target_dir": self._encode_path(target_dir),
            # The project owns its output. A stale caller/legacy session may
            # still point to a free-standing Folirina_Output directory; never
            # persist that path as the destination of a managed project.
            "output_dir": self._encode_path(str(self.output_dir or "")),
            "selected_index": max(0, int(selected_index or 0)),
            "book_pairing_ready": bool(book_pairing_ready),
            "restored_processed_only": bool(restored_processed_only),
            "page_marks": dict(page_marks or {}),
            "config": dict(config or {}),
            "pairs": [self._pair_to_dict(pair) for pair in pairs],
            "saved_at": _now_iso(),
        }
        save_json(project / SESSION_FILE, payload)
        self.touch(active_stage=active_stage)

    def load_session(self) -> dict[str, Any]:
        project = self._require_project()
        path = project / SESSION_FILE
        if not path.exists():
            def populated_dir(value: Path | None) -> str:
                if value is None or not value.is_dir():
                    return ""
                try:
                    return str(value) if any(item.is_file() for item in value.rglob("*")) else ""
                except OSError:
                    return ""

            return {
                "source_dir": populated_dir(self.source_dir),
                "target_dir": populated_dir(self.target_dir),
                "output_dir": str(self.output_dir or ""),
                "pairs": [],
                "selected_index": 0,
                "page_marks": {},
                "config": {},
                "book_pairing_ready": False,
                "restored_processed_only": False,
            }
        raw = load_json(path)
        if not isinstance(raw, dict) or int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
            raise ValueError("项目 session.json 版本不受支持")
        old_output = self._decode_path(raw.get("output_dir"))
        managed_output = str(self.output_dir or "")
        return {
            "source_dir": self._decode_path(raw.get("source_dir")),
            "target_dir": self._decode_path(raw.get("target_dir")),
            "output_dir": managed_output,
            # Keep a breadcrumb for opt-in migration; NEVER move, overwrite or
            # silently delete pre-existing legacy outputs on project restore.
            "legacy_output_dir": old_output if old_output and Path(old_output).resolve() != Path(managed_output).resolve() else "",
            "pairs": [self._pair_from_dict(row) for row in list(raw.get("pairs") or []) if isinstance(row, dict)],
            "selected_index": max(0, int(raw.get("selected_index") or 0)),
            "page_marks": dict(raw.get("page_marks") or {}),
            "config": dict(raw.get("config") or {}),
            "book_pairing_ready": bool(raw.get("book_pairing_ready")),
            "restored_processed_only": bool(raw.get("restored_processed_only")),
        }

    def touch(self, *, active_stage: str | None = None) -> None:
        project = self._require_project()
        payload = self._project_payload(project)
        payload["updated_at"] = _now_iso()
        if active_stage is not None:
            payload["active_stage"] = str(active_stage)
        save_json(project / PROJECT_FILE, payload)

    def project_size_bytes(self) -> int:
        project = self._require_project()
        total = 0
        for path in project.rglob("*"):
            try:
                if path.is_file():
                    total += path.stat().st_size
            except OSError:
                pass
        return total

    def clear_project_cache(self) -> int:
        project = self._require_project()
        cache = project / "cache"
        removed = 0
        if cache.exists():
            for path in cache.rglob("*"):
                try:
                    if path.is_file():
                        removed += path.stat().st_size
                except OSError:
                    pass
            shutil.rmtree(cache, ignore_errors=True)
        cache.mkdir(parents=True, exist_ok=True)
        self.touch()
        return removed

    def _require_project(self) -> Path:
        if self.active_project is None:
            raise RuntimeError("当前没有活动工作区项目")
        return self.active_project

    def _project_payload(self, project: Path) -> dict[str, Any]:
        raw = load_json(project / PROJECT_FILE)
        if not isinstance(raw, dict):
            raise ValueError("项目元数据损坏")
        if raw.get("format") != PROJECT_FORMAT or int(raw.get("schema_version") or 0) != SCHEMA_VERSION:
            raise ValueError("项目格式或版本不受支持")
        return dict(raw)

    def _read_info(self, project: Path) -> FolirinaProjectInfo | None:
        try:
            if not project.is_dir() or not (project / PROJECT_FILE).is_file():
                return None
            payload = self._project_payload(project)
            return self._info_from_payload(project, payload)
        except Exception:
            return None

    @staticmethod
    def _info_from_payload(project: Path, payload: dict[str, Any]) -> FolirinaProjectInfo:
        return FolirinaProjectInfo(
            project_id=str(payload.get("project_id") or ""),
            name=str(payload.get("name") or project.name),
            path=str(project),
            created_at=str(payload.get("created_at") or ""),
            updated_at=str(payload.get("updated_at") or ""),
        )

    def _ensure_layout(self, project: Path) -> None:
        for rel in self.LAYOUT:
            (project / rel).mkdir(parents=True, exist_ok=True)

    def _unique_project_dir(self, name: str) -> Path:
        base = _safe_component(name)
        target = self.projects_root / base
        serial = 2
        legacy_root = self.projects_root / "projects"
        while target.exists() or (legacy_root / target.name).exists():
            target = self.projects_root / f"{base} {serial}"
            serial += 1
        return target

    def _encode_path(self, value: str | Path | None) -> dict[str, str]:
        text = str(value or "").strip()
        if not text:
            return {}
        path = Path(text).expanduser()
        project = self._require_project()
        try:
            rel = path.resolve().relative_to(project.resolve())
            return {"relative": rel.as_posix()}
        except (OSError, ValueError):
            return {"external": str(path)}

    def _decode_path(self, value: Any) -> str:
        if not isinstance(value, dict):
            return ""
        project = self._require_project()
        relative = str(value.get("relative") or "").strip()
        if relative:
            return str((project / relative).resolve())
        return str(value.get("external") or "")

    def _pair_to_dict(self, pair: PagePair) -> dict[str, Any]:
        return {
            "source_path": self._encode_path(pair.source_path),
            "target_path": self._encode_path(pair.target_path),
            "source_index": int(pair.source_index),
            "target_index": int(pair.target_index),
            "confidence": float(pair.confidence),
            "score": float(pair.score),
            "reasons": list(pair.reasons or []),
        }

    def _pair_from_dict(self, row: dict[str, Any]) -> PagePair:
        return PagePair(
            source_path=self._decode_path(row.get("source_path")),
            target_path=self._decode_path(row.get("target_path")),
            source_index=int(row.get("source_index") or 0),
            target_index=int(row.get("target_index") or 0),
            confidence=float(row.get("confidence") or 0.0),
            score=float(row.get("score") or 0.0),
            reasons=list(row.get("reasons") or []),
        )


__all__ = [
    "SCHEMA_VERSION", "PROJECT_FORMAT", "PROJECT_FILE", "SESSION_FILE",
    "FolirinaProjectInfo", "ProjectWorkspaceManager",
]
