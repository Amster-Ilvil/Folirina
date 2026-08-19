"""Small, dependency-free GitHub Release updater for the macOS app.

The updater intentionally performs metadata checks without importing the GUI or
any optional ML package.  Release assets are expected to contain one
``Folirina.app`` bundle in a zip archive.
"""
from __future__ import annotations

import json
import os
import plistlib
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .version import __version__

DEFAULT_REPOSITORY = "Amster-Ilvil/Folirina"
API_BASE = "https://api.github.com/repos"
MAX_UPDATE_FILES = 100_000
MAX_UPDATE_UNCOMPRESSED_BYTES = 4 * 1024**3
MAX_UPDATE_DOWNLOAD_BYTES = 2 * 1024**3


def repository() -> str:
    return DEFAULT_REPOSITORY


def _locked_repository(value: str | None = None) -> str:
    raw = str(value or DEFAULT_REPOSITORY).strip().strip("/").removesuffix(".git")
    if raw.startswith("https://github.com/"):
        raw = raw[len("https://github.com/"):].strip("/")
    if raw != DEFAULT_REPOSITORY:
        raise ValueError(f"Git 更新仓库已锁定：{DEFAULT_REPOSITORY}")
    return DEFAULT_REPOSITORY


def _version_tuple(value: str) -> tuple[int, ...]:
    raw = value.strip().lstrip("vV").split("-", 1)[0]
    out: list[int] = []
    for item in raw.split("."):
        digits = "".join(ch for ch in item if ch.isdigit())
        out.append(int(digits or 0))
    return tuple((out + [0, 0, 0])[:3])


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
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Folirina-Updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def check_latest(repo: str | None = None, timeout: float = 8.0) -> UpdateInfo | None:
    repo = _locked_repository(repo)
    data = _request_json(f"{API_BASE}/{repo}/releases/latest", timeout)
    tag = str(data.get("tag_name") or "").strip()
    version = tag.lstrip("vV")
    assets = data.get("assets") or []
    asset = next(
        (x for x in assets if str(x.get("name", "")).lower().endswith(".zip")), None
    )
    if not tag or not asset or not asset.get("browser_download_url"):
        return None
    return UpdateInfo(version, tag, str(asset["browser_download_url"]), str(data.get("body") or ""), str(asset.get("name") or ""))


def _download(url: str, destination: Path, timeout: float = 60.0) -> None:
    """Download one release asset atomically with a hard size ceiling."""
    request = urllib.request.Request(url, headers={"User-Agent": "Folirina-Updater"})
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
    """Download and transactionally replace an app bundle with rollback backup.

    The incoming app is fully copied and validated *before* the current app is
    moved aside. This avoids leaving a half-copied application at ``target`` if
    disk space, permissions, antivirus or a forced quit interrupts installation.
    """
    target = Path(app_path).expanduser().resolve()
    if target.suffix != ".app" or not target.exists():
        raise RuntimeError(f"找不到当前应用：{target}")
    backup = target.with_name(target.name + ".previous")
    staged = target.with_name(target.name + ".new")
    with tempfile.TemporaryDirectory(prefix="folirina-update-") as tmp:
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
    value = str(os.environ.get("FOLIRINA_APP_BUNDLE", "") or os.environ.get("MHD_APP_BUNDLE", "") or "").strip()
    return Path(value).expanduser() if value else None


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Folirina 更新工具")
    parser.add_argument("command", choices=("check", "install"))
    parser.add_argument("--repo", default=None)
    parser.add_argument("--app", default=None, help="当前 .app 路径")
    args = parser.parse_args()
    info = check_latest(args.repo)
    if info is None:
        print(json.dumps({"available": False, "current": __version__}, ensure_ascii=False))
        return 0
    payload = {"available": info.available, "current": __version__, "version": info.version, "asset": info.asset_name}
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
