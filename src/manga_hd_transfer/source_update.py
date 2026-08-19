from __future__ import annotations

"""Cross-platform source updater backed by a GitHub Git repository.

This updater is intentionally independent from Qt and from all image/OCR/model
modules.  It supports two installation layouts:

* a real Git checkout: fetch + fast-forward with a clean-working-tree guard;
* the portable ZIP/editable-install layout shipped by this project: clone (or
  download a repository archive when Git is unavailable), validate into a
  staging directory, transactionally replace only program-managed paths, then
  refresh the editable install.

User projects, output folders, model caches, QSettings and application logs are
outside the managed path set and are never removed by this module.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import base64
import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Callable, Mapping
from urllib.parse import quote, urlencode, urlparse
import urllib.request
import zipfile

from .version import __version__

logger = logging.getLogger(__name__)

DEFAULT_REPOSITORY = "Amster-Ilvil/Folirina"
DEFAULT_BRANCH = "main"
API_BASE = "https://api.github.com/repos"
SOURCE_STATE_FILE = ".folirina-source-state.json"
LEGACY_SOURCE_STATE_FILE = ".mhd-source-state.json"
MAX_ARCHIVE_FILES = 100_000
MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_DOWNLOAD_BYTES = 512 * 1024**2

# Keep the update surface deliberately small.  Runtime project/output/model data
# is never stored in these paths by the application.
MANAGED_PATHS = (
    "src",
    "scripts",
    "tools",
    ".github",
    ".gitignore",
    "config.example.json",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    "MULTIPLATFORM_INSTALL.md",
)

ProgressCallback = Callable[[str], None]


def _emit(progress: ProgressCallback | None, message: str) -> None:
    text = str(message).strip()
    logger.info("source update: %s", text)
    if progress is not None:
        progress(text)


def _version_tuple(value: str) -> tuple[int, int, int]:
    raw = str(value or "0").strip().lstrip("vV").split("-", 1)[0]
    parts: list[int] = []
    for token in raw.split("."):
        match = re.search(r"\d+", token)
        parts.append(int(match.group(0)) if match else 0)
    return tuple((parts + [0, 0, 0])[:3])  # type: ignore[return-value]


def normalize_repository(value: str | None) -> str:
    """Return the one repository this application is allowed to update from.

    The update target is release-critical configuration, not a user preference.
    v2.0.72 locks it to the official project repository so a typo, stale setting,
    environment variable, or edited UI field cannot redirect self-update code.
    """
    raw = str(value or "").strip().rstrip("/")
    if raw:
        if raw.startswith("git@github.com:"):
            raw = raw.split(":", 1)[1]
        elif "://" in raw:
            parsed = urlparse(raw)
            if parsed.hostname not in {"github.com", "www.github.com"}:
                raise ValueError("Git 更新仓库已锁定为官方 GitHub 仓库")
            raw = parsed.path.strip("/")
        raw = raw.removesuffix(".git").strip("/")
        if raw != DEFAULT_REPOSITORY:
            raise ValueError(f"Git 更新仓库已锁定：{DEFAULT_REPOSITORY}")
    return DEFAULT_REPOSITORY


def normalize_branch(value: str | None) -> str:
    """Return the locked release branch and reject alternate branches."""
    branch = str(value or "").strip()
    if branch and branch != DEFAULT_BRANCH:
        raise ValueError(f"Git 更新分支已锁定：{DEFAULT_BRANCH}")
    return DEFAULT_BRANCH


def github_repository_url(repo: str) -> str:
    return f"https://github.com/{normalize_repository(repo)}.git"


def discover_project_root(start: str | Path | None = None) -> Path:
    override = str(os.environ.get("FOLIRINA_PROJECT_ROOT", "") or os.environ.get("MHD_PROJECT_ROOT", "") or "").strip()
    candidates: list[Path] = []
    if start is not None:
        candidates.append(Path(start).expanduser().resolve())
    if override:
        candidates.append(Path(override).expanduser().resolve())
    candidates.append(Path(__file__).resolve())
    candidates.append(Path.cwd().resolve())
    seen: set[Path] = set()
    for candidate in candidates:
        probe = candidate if candidate.is_dir() else candidate.parent
        for root in (probe, *probe.parents):
            if root in seen:
                continue
            seen.add(root)
            if (root / "pyproject.toml").is_file() and (root / "src" / "manga_hd_transfer").is_dir():
                return root
    raise RuntimeError("无法定位本地程序源码目录；请从项目目录启动程序")


def read_project_version(root: str | Path) -> str:
    path = Path(root) / "pyproject.toml"
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    value = str(((data.get("project") or {}).get("version") or "")).strip()
    if not value:
        raise RuntimeError(f"无法读取项目版本：{path}")
    return value


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Folirina-Source-Updater",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = str(os.environ.get("FOLIRINA_GITHUB_TOKEN", "") or os.environ.get("MHD_GITHUB_TOKEN", "") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str, *, timeout: float = 12.0) -> object:
    request = urllib.request.Request(url, headers=_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _github_contents(repo: str, path: str, branch: str, *, timeout: float) -> bytes:
    query = urlencode({"ref": branch})
    data = _request_json(f"{API_BASE}/{repo}/contents/{quote(path)}?{query}", timeout=timeout)
    if not isinstance(data, dict):
        raise RuntimeError(f"远端文件响应异常：{path}")
    encoding = str(data.get("encoding") or "")
    content = str(data.get("content") or "")
    if encoding != "base64" or not content:
        raise RuntimeError(f"无法读取远端文件：{path}")
    return base64.b64decode(content, validate=False)


def _remote_project_version(repo: str, branch: str, *, timeout: float) -> str:
    raw = _github_contents(repo, "pyproject.toml", branch, timeout=timeout)
    data = tomllib.loads(raw.decode("utf-8"))
    version = str(((data.get("project") or {}).get("version") or "")).strip()
    if not version:
        raise RuntimeError("远端 pyproject.toml 没有项目版本")
    return version


def _run_git(args: list[str], *, cwd: str | Path | None = None, timeout: float = 90.0) -> str:
    git = shutil.which("git")
    if not git:
        raise RuntimeError("系统没有找到 Git")
    proc = subprocess.run(
        [git, *args], cwd=str(cwd) if cwd is not None else None,
        text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=timeout, check=False,
    )
    output = str(proc.stdout or "")
    if proc.returncode != 0:
        raise RuntimeError(f"Git 命令失败：git {' '.join(args)}\n{output[-5000:]}")
    return output.strip()


def _git_head(root: Path) -> str:
    if not (root / ".git").exists() or not shutil.which("git"):
        return ""
    try:
        return _run_git(["rev-parse", "HEAD"], cwd=root, timeout=12.0).strip()
    except Exception:
        logger.debug("git head detection failed", exc_info=True)
        return ""


def read_source_state(root: str | Path) -> dict[str, object]:
    base = Path(root)
    for name in (SOURCE_STATE_FILE, LEGACY_SOURCE_STATE_FILE):
        path = base / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except (OSError, ValueError, TypeError):
            continue
    return {}


def _write_source_state(root: Path, *, repo: str, branch: str, commit: str, version: str, method: str) -> None:
    payload = {
        "schema": "folirina.source_state.v1",
        "repository": repo,
        "branch": branch,
        "commit": commit,
        "version": version,
        "method": method,
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    path = root / SOURCE_STATE_FILE
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


@dataclass(frozen=True, slots=True)
class SourceUpdateInfo:
    repository: str
    branch: str
    local_version: str
    remote_version: str
    local_commit: str
    remote_commit: str
    remote_date: str
    remote_message: str
    available: bool
    reason: str
    project_root: Path

    @property
    def remote_short(self) -> str:
        return self.remote_commit[:10]

    @property
    def local_short(self) -> str:
        return self.local_commit[:10] if self.local_commit else "未记录"


@dataclass(frozen=True, slots=True)
class SourceUpdateResult:
    project_root: Path
    old_version: str
    new_version: str
    commit: str
    method: str
    restart_required: bool = True


def check_source_update(
    repo: str | None = None,
    branch: str | None = None,
    *,
    project_root: str | Path | None = None,
    timeout: float = 12.0,
) -> SourceUpdateInfo:
    repository = normalize_repository(repo)
    branch_name = normalize_branch(branch)
    root = discover_project_root(project_root)
    local_version = read_project_version(root)
    commit_data = _request_json(
        f"{API_BASE}/{repository}/commits/{quote(branch_name, safe='')}", timeout=timeout
    )
    if not isinstance(commit_data, dict):
        raise RuntimeError("GitHub commit 响应异常")
    remote_commit = str(commit_data.get("sha") or "").strip()
    commit_meta = commit_data.get("commit") if isinstance(commit_data.get("commit"), dict) else {}
    remote_message = str((commit_meta or {}).get("message") or "").splitlines()[0].strip()
    author = (commit_meta or {}).get("author") if isinstance((commit_meta or {}).get("author"), dict) else {}
    remote_date = str((author or {}).get("date") or "").strip()
    if not remote_commit:
        raise RuntimeError("GitHub 没有返回远端 commit")
    remote_version = _remote_project_version(repository, branch_name, timeout=timeout)

    local_commit = _git_head(root)
    if not local_commit:
        state = read_source_state(root)
        if str(state.get("repository") or "") == repository and str(state.get("branch") or "") == branch_name:
            local_commit = str(state.get("commit") or "").strip()

    local_v = _version_tuple(local_version)
    remote_v = _version_tuple(remote_version)
    if remote_v > local_v:
        available = True
        reason = f"发现新版本 v{remote_version}"
    elif remote_v < local_v:
        available = False
        reason = f"本地 v{local_version} 比仓库 v{remote_version} 更新；禁止自动降级"
    elif local_commit and local_commit != remote_commit:
        available = True
        reason = "版本号相同，但 Git 仓库已有新的代码提交"
    elif local_commit == remote_commit and local_commit:
        available = False
        reason = "当前代码已经与 Git 仓库同步"
    else:
        # Portable ZIP builds do not have a Git HEAD.  Allow a one-time verified
        # sync when the remote version is not older; after the first update the
        # source-state file makes future commit comparisons exact.
        available = True
        reason = "本地发布包没有 Git commit 基线；可安全同步同版本远端源码"

    return SourceUpdateInfo(
        repository=repository,
        branch=branch_name,
        local_version=local_version,
        remote_version=remote_version,
        local_commit=local_commit,
        remote_commit=remote_commit,
        remote_date=remote_date,
        remote_message=remote_message,
        available=available,
        reason=reason,
        project_root=root,
    )


def _safe_extract_zip(archive: zipfile.ZipFile, destination: Path) -> None:
    destination = destination.resolve()
    infos = archive.infolist()
    if len(infos) > MAX_ARCHIVE_FILES:
        raise RuntimeError("仓库归档文件数量异常，已停止更新")
    total = sum(max(0, int(item.file_size)) for item in infos)
    if total > MAX_ARCHIVE_BYTES:
        raise RuntimeError("仓库归档解压体积异常，已停止更新")
    for info in infos:
        name = str(info.filename).replace("\\", "/")
        candidate = Path(name)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise RuntimeError(f"仓库归档包含不安全路径：{info.filename}")
        output = (destination / candidate).resolve()
        try:
            output.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"仓库归档路径越界：{info.filename}") from exc
    archive.extractall(destination)


def _download_repository_archive(repo: str, branch: str, destination: Path, *, timeout: float = 90.0) -> None:
    # The GitHub API archive endpoint works for public repositories and for
    # private repositories when FOLIRINA_GITHUB_TOKEN (or legacy MHD_GITHUB_TOKEN) is provided.
    url = f"{API_BASE}/{repo}/zipball/{quote(branch, safe='')}"
    request = urllib.request.Request(url, headers=_headers())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as fh:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Git 仓库下载体积异常，已停止更新")
            fh.write(chunk)


def _candidate_root_from_archive(extract_root: Path) -> Path:
    roots = [p for p in extract_root.iterdir() if p.is_dir()]
    if len(roots) == 1:
        return roots[0]
    for candidate in roots:
        if (candidate / "pyproject.toml").exists() and (candidate / "src" / "manga_hd_transfer").is_dir():
            return candidate
    raise RuntimeError("下载的 Git 仓库中没有找到项目根目录")


def _validate_candidate(candidate: Path, *, current_version: str) -> str:
    required = (
        candidate / "pyproject.toml",
        candidate / "src" / "manga_hd_transfer" / "__init__.py",
        candidate / "src" / "manga_hd_transfer" / "gui_qt.py",
        candidate / "src" / "manga_hd_transfer" / "pipeline.py",
    )
    missing = [str(path.relative_to(candidate)) for path in required if not path.exists()]
    if missing:
        raise RuntimeError("远端源码结构不完整：" + ", ".join(missing))
    version = read_project_version(candidate)
    if _version_tuple(version) < _version_tuple(current_version):
        raise RuntimeError(f"远端源码版本 v{version} 低于本地 v{current_version}，已阻止降级")
    return version


def _portable_candidate(repo: str, branch: str, temp_root: Path, progress: ProgressCallback | None) -> tuple[Path, str]:
    checkout = temp_root / "checkout"
    if shutil.which("git"):
        _emit(progress, "正在从 Git 仓库克隆最新源码…")
        _run_git(
            ["clone", "--depth", "1", "--single-branch", "--branch", branch, github_repository_url(repo), str(checkout)],
            timeout=180.0,
        )
        commit = _run_git(["rev-parse", "HEAD"], cwd=checkout, timeout=12.0).strip()
        return checkout, commit

    _emit(progress, "系统未安装 Git，改用 GitHub 仓库归档下载…")
    archive_path = temp_root / "repository.zip"
    _download_repository_archive(repo, branch, archive_path)
    extract = temp_root / "archive"
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        _safe_extract_zip(archive, extract)
    candidate = _candidate_root_from_archive(extract)
    # Commit is obtained by the check API before install; caller stores it.
    return candidate, ""


def _copy_managed_to_stage(candidate: Path, stage: Path) -> list[str]:
    copied: list[str] = []
    stage.mkdir(parents=True, exist_ok=True)
    for rel in MANAGED_PATHS:
        source = candidate / rel
        if not source.exists():
            continue
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
        else:
            shutil.copy2(source, target)
        copied.append(rel)
    if "src" not in copied or "pyproject.toml" not in copied:
        raise RuntimeError("远端源码缺少核心程序目录，已停止更新")
    return copied


def _cleanup_packaging_artifacts(root: Path, *, progress: ProgressCallback | None = None) -> dict[str, list[str]]:
    """Remove disposable packaging outputs that can shadow freshly updated source.

    ``setuptools`` may reuse ``build/lib`` when its timestamps are newer than the
    edited source tree.  A source update must therefore remove packaging outputs
    before refreshing/installing the project, otherwise a wheel can silently
    contain code from the previous version.  Only reproducible build metadata is
    touched; virtualenvs, models, projects and user outputs are never included.
    """
    removed: list[str] = []
    candidates = [root / "build", root / "dist", root / ".pytest_cache"]
    candidates.extend(sorted(root.glob("*.egg-info")))
    src = root / "src"
    if src.is_dir():
        candidates.extend(sorted(src.glob("*.egg-info")))
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if not (path.exists() or path.is_symlink()):
            continue
        rel = path.relative_to(root).as_posix()
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path, ignore_errors=False)
        else:
            path.unlink(missing_ok=True)
        removed.append(rel)
    if removed:
        _emit(progress, "已清理旧打包缓存：" + "、".join(removed))
    return {"removed": removed}


def _refresh_editable_install(root: Path, *, progress: ProgressCallback | None) -> None:
    if bool(getattr(sys, "frozen", False)):
        _emit(progress, "当前为冻结应用；跳过 editable install 刷新。")
        return
    python = Path(sys.executable)
    if not python.exists():
        _emit(progress, "无法定位当前 Python；源码已更新，但没有刷新 editable install。")
        return
    _emit(progress, "正在刷新本地 Python editable install（不会下载 OCR 模型）…")
    proc = subprocess.run(
        [str(python), "-m", "pip", "install", "-e", ".[gui]"],
        cwd=str(root), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        timeout=600.0, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError("刷新本地安装失败：\n" + str(proc.stdout or "")[-8000:])
    logger.info("editable install refreshed: %s", str(proc.stdout or "")[-4000:])


def _install_portable(
    info: SourceUpdateInfo,
    *,
    progress: ProgressCallback | None,
    refresh_install: bool,
) -> SourceUpdateResult:
    root = info.project_root
    old_version = read_project_version(root)
    with tempfile.TemporaryDirectory(prefix="folirina-source-update-") as temp_name:
        temp_root = Path(temp_name)
        candidate, candidate_commit = _portable_candidate(info.repository, info.branch, temp_root, progress)
        new_version = _validate_candidate(candidate, current_version=old_version)
        commit = candidate_commit or info.remote_commit
        _emit(progress, f"远端源码校验通过：v{new_version} · {commit[:10]}")

        # Stage on the same filesystem as the project so directory renames remain
        # atomic.  Never stage inside src itself.
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.folirina-stage-", dir=str(root.parent)))
        backup = Path(tempfile.mkdtemp(prefix=f".{root.name}.folirina-backup-", dir=str(root.parent)))
        moved_existing: list[str] = []
        installed_new: list[str] = []
        try:
            copied = _copy_managed_to_stage(candidate, stage)
            _validate_candidate(stage, current_version=old_version)
            _emit(progress, "已建立事务更新区，正在替换程序代码…")
            for rel in copied:
                current = root / rel
                incoming = stage / rel
                saved = backup / rel
                if current.exists() or current.is_symlink():
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(current), str(saved))
                    moved_existing.append(rel)
                current.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(incoming), str(current))
                installed_new.append(rel)
            _cleanup_packaging_artifacts(root, progress=progress)
            if refresh_install:
                _refresh_editable_install(root, progress=progress)
            _write_source_state(
                root, repo=info.repository, branch=info.branch,
                commit=commit, version=new_version, method="portable-sync",
            )
        except Exception:
            logger.exception("portable source update failed; rolling back")
            _emit(progress, "更新失败，正在自动回滚到更新前代码…")
            for rel in reversed(installed_new):
                current = root / rel
                if current.is_dir() and not current.is_symlink():
                    shutil.rmtree(current, ignore_errors=True)
                else:
                    try:
                        current.unlink(missing_ok=True)
                    except OSError:
                        pass
            for rel in reversed(moved_existing):
                saved = backup / rel
                current = root / rel
                if saved.exists() or saved.is_symlink():
                    current.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(saved), str(current))
            raise
        finally:
            shutil.rmtree(stage, ignore_errors=True)
            shutil.rmtree(backup, ignore_errors=True)

    _emit(progress, "本地源码更新完成；重新启动程序后使用新代码。")
    return SourceUpdateResult(root, old_version, new_version, commit, "portable-sync")


def _git_clean(root: Path) -> bool:
    output = _run_git(["status", "--porcelain", "--untracked-files=no"], cwd=root, timeout=15.0)
    return not bool(output.strip())


def _install_git_checkout(
    info: SourceUpdateInfo,
    *,
    progress: ProgressCallback | None,
    refresh_install: bool,
) -> SourceUpdateResult:
    root = info.project_root
    if not _git_clean(root):
        raise RuntimeError("当前 Git 工作区有未提交修改。为避免覆盖本地代码，已停止自动更新。")
    old_commit = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=12.0).strip()
    old_version = read_project_version(root)
    _emit(progress, "正在 fetch Git 仓库…")
    _run_git(["fetch", "--no-tags", github_repository_url(info.repository), info.branch], cwd=root, timeout=180.0)
    target_commit = _run_git(["rev-parse", "FETCH_HEAD"], cwd=root, timeout=12.0).strip()
    ancestor = subprocess.run(
        [shutil.which("git") or "git", "merge-base", "--is-ancestor", old_commit, target_commit],
        cwd=str(root), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("远端更新不是当前代码的 fast-forward；为保护本地版本，已拒绝自动合并。")
    try:
        if target_commit != old_commit:
            _emit(progress, f"正在 fast-forward 到 {target_commit[:10]}…")
            _run_git(["merge", "--ff-only", target_commit], cwd=root, timeout=90.0)
        new_version = _validate_candidate(root, current_version=old_version)
        _cleanup_packaging_artifacts(root, progress=progress)
        if refresh_install:
            _refresh_editable_install(root, progress=progress)
        _write_source_state(
            root, repo=info.repository, branch=info.branch,
            commit=target_commit, version=new_version, method="git-fast-forward",
        )
    except Exception:
        logger.exception("git checkout update failed; resetting to previous commit")
        _emit(progress, "更新失败，正在恢复更新前 Git commit…")
        _run_git(["reset", "--hard", old_commit], cwd=root, timeout=60.0)
        raise
    _emit(progress, "Git fast-forward 更新完成；重新启动程序后使用新代码。")
    return SourceUpdateResult(root, old_version, new_version, target_commit, "git-fast-forward")


def install_source_update(
    info: SourceUpdateInfo,
    *,
    progress: ProgressCallback | None = None,
    refresh_install: bool = True,
) -> SourceUpdateResult:
    """Install ``info`` into its local project root with rollback protection."""
    repository = normalize_repository(info.repository)
    branch = normalize_branch(info.branch)
    if repository != info.repository or branch != info.branch:
        raise RuntimeError("更新信息的 Git 目标与锁定仓库不一致，已停止安装")
    root = discover_project_root(info.project_root)
    if root != info.project_root.resolve():
        raise RuntimeError("更新目标目录发生变化，已停止安装")
    current = read_project_version(root)
    if _version_tuple(info.remote_version) < _version_tuple(current):
        raise RuntimeError("远端版本低于当前版本，禁止自动降级")
    if (root / ".git").exists() and shutil.which("git"):
        return _install_git_checkout(info, progress=progress, refresh_install=refresh_install)
    return _install_portable(info, progress=progress, refresh_install=refresh_install)


__all__ = [
    "DEFAULT_REPOSITORY", "DEFAULT_BRANCH", "MANAGED_PATHS", "SOURCE_STATE_FILE", "LEGACY_SOURCE_STATE_FILE",
    "SourceUpdateInfo", "SourceUpdateResult", "check_source_update",
    "discover_project_root", "github_repository_url", "install_source_update",
    "normalize_branch", "normalize_repository", "read_project_version",
    "read_source_state", "_cleanup_packaging_artifacts",
]
