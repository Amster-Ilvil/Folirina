from __future__ import annotations

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
BUILD = ROOT / "build"
APP_NAME = "Folirina"
BUNDLE_ID = "io.github.amsterilvil.folirina"


def run(args: list[str]) -> None:
    print("+", " ".join(str(x) for x in args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def read_version() -> str:
    text = (ROOT / "src" / "manga_hd_transfer" / "version.py").read_text(encoding="utf-8")
    match = re.search(r'^\s*__version__\s*=\s*["\']([^"\']+)["\']', text, re.MULTILINE)
    if not match:
        raise SystemExit("Could not find __version__ in src/manga_hd_transfer/version.py")
    return match.group(1)


def add_data_args() -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    result: list[str] = []
    for directory in ("assets", "tools"):
        path = ROOT / directory
        if path.is_dir():
            result += ["--add-data", f"{path}{sep}{directory}"]
    for filename in ("README.md", "LICENSE"):
        path = ROOT / filename
        if path.is_file():
            result += ["--add-data", f"{path}{sep}."]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone Folirina desktop distribution")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    system = platform.system()
    if system not in {"Windows", "Darwin", "Linux"}:
        raise SystemExit(f"Unsupported packaging platform: {system}")

    if args.clean:
        shutil.rmtree(DIST, ignore_errors=True)
        shutil.rmtree(BUILD, ignore_errors=True)

    version = read_version()
    print(f"Building {APP_NAME} {version} for {system} {platform.machine()}", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        APP_NAME,
        "--paths",
        str(ROOT / "src"),
        "--collect-submodules",
        "manga_hd_transfer",
        "--exclude-module",
        "pytest",
        *add_data_args(),
    ]

    icon_candidates = [
        ROOT / "assets" / "icon.ico",
        ROOT / "assets" / "icon.icns",
        ROOT / "assets" / "icon.png",
    ]
    icon = next((path for path in icon_candidates if path.is_file()), None)
    if icon is not None:
        cmd += ["--icon", str(icon)]

    if system == "Darwin":
        cmd += ["--osx-bundle-identifier", BUNDLE_ID]

    cmd += [str(ROOT / "run_gui.py")]
    run(cmd)

    if system == "Darwin":
        app = DIST / f"{APP_NAME}.app"
        if not app.is_dir():
            raise SystemExit(f"Expected app bundle not found: {app}")
        executable = app / "Contents" / "MacOS" / APP_NAME
        if not executable.is_file():
            raise SystemExit(f"Expected app executable not found: {executable}")
        plist = app / "Contents" / "Info.plist"
        if plist.exists():
            subprocess.run(
                ["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleShortVersionString {version}", str(plist)],
                check=False,
            )
            subprocess.run(
                ["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleVersion {version}", str(plist)],
                check=False,
            )
            subprocess.run(["plutil", "-lint", str(plist)], check=True)
    else:
        app_dir = DIST / APP_NAME
        if not app_dir.is_dir():
            raise SystemExit(f"Expected distribution directory not found: {app_dir}")
        executable = app_dir / (f"{APP_NAME}.exe" if system == "Windows" else APP_NAME)
        if not executable.is_file():
            raise SystemExit(f"Expected executable not found: {executable}")

    print(f"Build complete: {DIST}")


if __name__ == "__main__":
    main()
