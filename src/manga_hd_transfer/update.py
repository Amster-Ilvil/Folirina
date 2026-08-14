"""Small, dependency-free GitHub Release updater for the macOS app.

The updater intentionally performs metadata checks without importing the GUI or
any optional ML package.  Release assets are expected to contain one
``Manga HD Transfer Studio.app`` bundle in a zip archive.
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

from . import __version__

DEFAULT_REPOSITORY = "Amster-Ilvil/Manga-HD-Translation-Transfer"
API_BASE = "https://api.github.com/repos"


def repository() -> str:
    return os.environ.get("MHD_UPDATE_REPO", DEFAULT_REPOSITORY).strip().strip("/")


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
        headers={"Accept": "application/vnd.github+json", "User-Agent": "Manga-HD-Transfer-Updater"},
    )
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
        (x for x in assets if str(x.get("name", "")).lower().endswith(".zip")), None
    )
    if not tag or not asset or not asset.get("browser_download_url"):
        return None
    return UpdateInfo(version, tag, str(asset["browser_download_url"]), str(data.get("body") or ""), str(asset.get("name") or ""))


def _download(url: str, destination: Path, timeout: float = 60.0) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Manga-HD-Transfer-Updater"})
    with urllib.request.urlopen(request, timeout=timeout) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _find_app(root: Path) -> Path:
    apps = list(root.glob("*.app")) + list(root.rglob("*.app"))
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
    """Download and replace an app bundle, keeping a rollback backup."""
    target = Path(app_path).expanduser().resolve()
    if target.suffix != ".app" or not target.exists():
        raise RuntimeError(f"找不到当前应用：{target}")
    with tempfile.TemporaryDirectory(prefix="mhd-update-") as tmp:
        archive = Path(tmp) / "release.zip"
        _download(info.url, archive)
        extract = Path(tmp) / "unpacked"
        extract.mkdir()
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extract)
        incoming = _find_app(extract)
        _validate_app(incoming)
        backup = target.with_name(target.name + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        target.rename(backup)
        try:
            shutil.copytree(incoming, target)
        except Exception:
            if target.exists():
                shutil.rmtree(target)
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
