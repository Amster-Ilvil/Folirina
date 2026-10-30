from __future__ import annotations

"""Discover local publication engines. Installation is explicit, never per page."""

import hashlib
import os
from pathlib import Path
import platform
import shutil
import ssl
import tarfile
import tempfile
import urllib.request

from .storage_paths import runtime_home

CAESIUM_VERSION = "1.5.0"
CAESIUM_MAC_ASSETS = {
    "arm64": ("aarch64", "9fc192ee1f7190d238ff8bf684ce502ceb652040544ee4a1b4bd10a6bf1b783e"),
    "x86_64": ("x86_64", "a9e6dab8de3b3b024dd6bbf26fd99047355dad9235984faf8fa707b636b7e117"),
}


def _executable(paths) -> Path | None:
    for value in paths:
        if value:
            path = Path(value).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return path.resolve()
    return None


def find_caesium(explicit: str = "") -> Path | None:
    if str(explicit).strip():
        return _executable([explicit])
    return _executable([
        runtime_home() / "caesium" / "caesiumclt",
        shutil.which("caesiumclt"), "/opt/homebrew/bin/caesiumclt", "/usr/local/bin/caesiumclt",
    ])


def find_upscayl(explicit: str = "", models: str = "") -> tuple[Path | None, Path | None]:
    if str(explicit).strip():
        binary = _executable([explicit])
    else:
        binary = _executable([
            "/Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin",
            Path.home() / "Applications/Upscayl.app/Contents/Resources/bin/upscayl-bin",
            shutil.which("upscayl-bin"), shutil.which("upscayl"),
        ])
    if str(models).strip():
        model_dir = Path(models).expanduser()
    elif binary:
        candidates = [binary.parent.parent / "models", binary.parent / "models"]
        model_dir = next((p for p in candidates if p.is_dir()), candidates[0])
    else:
        model_dir = None
    return binary, model_dir


def install_caesium() -> Path:
    """Install the official, SHA-256-pinned macOS CLI (about 2.5 MB download)."""
    machine = platform.machine().lower()
    if platform.system() != "Darwin" or machine not in CAESIUM_MAC_ASSETS:
        raise RuntimeError("自动安装支持 macOS；其它平台请安装 caesiumclt 后在输出设置指定路径。")
    arch, expected = CAESIUM_MAC_ASSETS[machine]
    root = runtime_home() / "caesium"
    root.mkdir(parents=True, exist_ok=True)
    asset = f"caesiumclt-v{CAESIUM_VERSION}-{arch}-apple-darwin.tar.gz"
    url = f"https://github.com/Lymphatus/caesium-clt/releases/download/v{CAESIUM_VERSION}/{asset}"
    from .tls_support import build_runtime_ca_bundle
    ca = build_runtime_ca_bundle(root)
    context = ssl.create_default_context(cafile=str(ca) if ca else None)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=root) as td:
        archive = Path(td) / asset
        request = urllib.request.Request(url, headers={"User-Agent": "Folirina-publication-installer"})
        with urllib.request.urlopen(request, context=context, timeout=60) as response, archive.open("wb") as output:
            total = 0
            while chunk := response.read(1024 * 1024):
                total += len(chunk)
                if total > 16 * 1024 * 1024:
                    raise RuntimeError("Caesium 发行包大小超出预期。")
                output.write(chunk)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise RuntimeError("Caesium 发行包 SHA-256 校验失败。")
        with tarfile.open(archive) as tar:
            member = next((m for m in tar.getmembers() if Path(m.name).name == "caesiumclt" and m.isfile()), None)
            if member is None or member.size > 32 * 1024 * 1024:
                raise RuntimeError("Caesium 发行包缺少有效的命令行程序。")
            data = tar.extractfile(member)
            if data is None:
                raise RuntimeError("无法读取 Caesium 程序。")
            staged = Path(td) / "caesiumclt"
            with staged.open("wb") as output:
                shutil.copyfileobj(data, output)
            staged.chmod(0o755)
            os.replace(staged, root / "caesiumclt")
    return root / "caesiumclt"


if __name__ == "__main__":
    print(install_caesium())
