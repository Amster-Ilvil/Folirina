"""Small, dependency-free GitHub Release updater.

Release metadata checks are platform-aware so a multi-platform release never
feeds a Windows/Linux archive to the macOS installer. Automatic replacement is
currently limited to the macOS ``.app`` bundle; Windows/Linux users can still
check the latest matching release asset and update from the release page.
"""
from __future__ import annotations

import json
import os
import platform
import plistlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import __version__

DEFAULT_REPOSITORY = "Amster-Ilvil/Manga-HD-Translation-Transfer"
API_BASE = "https://api.github.com/repos"
MAX_UPDATE_FILES = 100_000
MAX_UPDATE_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_UPDATE_DOWNLOAD_BYTES = 2 * 1024**3


def repository() -> str:
    return os.environ.get("MHD_UPDATE_REPO", DEFAULT_REPOSITORY).strip().strip("/")


def _version_tuple(value: str) -> tuple[int, ...]:
    raw = value.strip().lstrip("vV").split("-", 1)[0]
    out: list[int] = []
    for item in raw.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        out.append(int(digits or 0))
    return tuple((out + [0, 0, 0])[:3])


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "Manga-HD-Transfer-Updater",
    }
    token = os.environ.get("MHD_GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _asset_matches_platform(name: str, system: str | None = None, machine: str | None = None) -> bool:
    lower = name.lower()
    system = (system or platform.system()).lower()
    machine = (machine or platform.machine()).lower()
    if system == "darwin":
        return "macos_universal" in lower and lower.endswith(".zip")
    if system == "windows":
        if machine in {"amd64", "x86_64"}:
            return "windows_x64" in lower and lower.endswith(".zip")
        return False
    if system == "linux":
        if machine in {"amd64", "x86_64"}:
            return "linux_x86_64" in lower and (lower.endswith(".tar.gz") or lower.endswith(".zip"))
        return False
    return False


@dataclass(frozen=True)
class UpdateInfo:
    version: str
    tag: str
    url: str
    notes: str = ""
    asset_name: str = ""

    @property
    def available(self) -> bool:
        return _version_tuple(self.version) > _version_tuple(__version__)


def _request_json(url: str, timeout: float = 8.0) -> Any:
    request = urllib.request.Request(url, headers=_github_headers())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_latest(repo: str | None = None, timeout: float = 8.0) -> UpdateInfo | None:
    repo = (repo or repository()).strip().strip("/")
    if not repo or "/" not in repo:
        raise ValueError("更新仓库必须是 owner/repository 格式")
    data = _request_json(f"{API_BASE}/{repo}/releases/latest", timeout)
    tag = str(data.get("tag_name") or "").strip()
    version = tag.lstrip("vV")
    assets = data.get("assets") or []
    asset = next(
        (x for x in assets if _asset_matches_platform(str(x.get("name", "")))),
        None,
    )
    if not tag or not asset or not asset.get("browser_download_url"):
        return None
    return UpdateInfo(
        version,
        tag,
        str(asset["browser_download_url"]),
        str(data.get("body") or ""),
        str(asset.get("name") or ""),
    )


def _download(url: str, destination: Path, timeout: float = 60.0) -> None:
    """Download one release asset atomically with a hard size ceiling."""
    request = urllib.request.Request(url, headers=_github_headers())
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".part", dir=str(destination.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, tmp.open("wb") as out:
            try:
                declared = int(response.headers.get("Content-Length", "0") or 0)
            except (TypeError, ValueError):
                declared = 0
            if declared > MAX_UPDATE_DOWNLOAD_BYTES:
                raise RuntimeError("更新包下载体积异常，已停止安装")
            total = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPDATE_DOWNLOAD_BYTES:
                    raise RuntimeError("更新包下载体积异常，已停止安装")
                out.write(chunk)
            out.flush()
        os.replace(tmp, destination)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _safe_extract_zip(bundle: zipfile.ZipFile, destination: Path) -> None:
    """Extract an update archive without path traversal or zip-bomb surprises."""
    destination = destination.resolve()
    infos = bundle.infolist()
    if len(infos) > MAX_UPDATE_FILES:
        raise RuntimeError("更新包文件数量异常，已停止安装")
    total = sum(max(0, int(info.file_size)) for info in infos)
    if total > MAX_UPDATE_UNCOMPRESSED_BYTES:
        raise RuntimeError("更新包解压体积异常，已停止安装")
    for info in infos:
        name = str(info.filename).replace("\\", "/")
        candidate = Path(name)
        if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
            raise RuntimeError(f"更新包包含不安全路径：{info.filename}")
        out = (destination / candidate).resolve()
        try:
            out.relative_to(destination)
        except ValueError as exc:
            raise RuntimeError(f"更新包路径越界：{info.filename}") from exc
    bundle.extractall(destination)


def _find_app(root: Path) -> Path:
    apps = sorted(
        set(root.rglob("*.app")),
        key=lambda p: (len(p.relative_to(root).parts), p.as_posix().casefold()),
    )
    if not apps:
        raise RuntimeError("更新包中没有找到 .app")
    return apps[0]


def _validate_app(app: Path) -> None:
    plist_path = app / "Contents" / "Info.plist"
    if not plist_path.exists():
        raise RuntimeError("更新包不是有效的 macOS 应用")
    with plist_path.open("rb") as stream:
        plist = plistlib.load(stream)
    if plist.get("CFBundleIdentifier") != "org.mangahd.transferstudio":
        raise RuntimeError("更新包的应用标识不匹配，已停止安装")


def install_update(info: UpdateInfo, app_path: str | Path) -> Path:
    """Download and transactionally replace a macOS app with rollback backup."""
    if platform.system() != "Darwin":
        raise RuntimeError("当前自动安装更新仅支持 macOS；请下载对应平台 Release 包手动更新")
    if not _asset_matches_platform(info.asset_name, "Darwin", platform.machine()):
        raise RuntimeError(f"更新资产与当前 macOS 平台不匹配：{info.asset_name}")
    target = Path(app_path).expanduser().resolve()
    if target.suffix != ".app" or not target.exists():
        raise RuntimeError(f"找不到当前应用：{target}")
    backup = target.with_name(target.name + ".previous")
    staged = target.with_name(target.name + ".new")
    with tempfile.TemporaryDirectory(prefix="mhd-update-") as tmp:
        archive = Path(tmp) / "release.zip"
        _download(info.url, archive)
        extract = Path(tmp) / "unpacked"
        extract.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            _safe_extract_zip(bundle, extract)
        incoming = _find_app(extract)
        _validate_app(incoming)

        if staged.exists():
            shutil.rmtree(staged)
        try:
            shutil.copytree(incoming, staged, symlinks=True)
            _validate_app(staged)
        except Exception:
            if staged.exists():
                shutil.rmtree(staged, ignore_errors=True)
            raise

        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
        try:
            staged.rename(target)
        except Exception:
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            if staged.exists():
                shutil.rmtree(staged, ignore_errors=True)
            backup.rename(target)
            raise
    return target


def app_bundle_from_environment() -> Path | None:
    value = os.environ.get("MHD_APP_BUNDLE", "").strip()
    return Path(value).expanduser() if value else None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manga HD Transfer 更新工具")
    parser.add_argument("command", choices=("check", "install"))
    parser.add_argument("--repo", default=None)
    parser.add_argument("--app", default=None, help="当前 .app 路径")
    args = parser.parse_args()
    info = check_latest(args.repo)
    if info is None:
        print(json.dumps({"available": False, "current": __version__}, ensure_ascii=False))
        return 0
    payload = {
        "available": info.available,
        "current": __version__,
        "version": info.version,
        "asset": info.asset_name,
    }
    if args.command == "check" or not info.available:
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    target = Path(args.app) if args.app else app_bundle_from_environment()
    if target is None:
        raise SystemExit("安装更新需要 --app /path/to/App.app")
    print(install_update(info, target))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
