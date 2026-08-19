from __future__ import annotations

import argparse
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import sys

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
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', text, re.M)
    if not m:
        raise SystemExit("Folirina __version__ not found")
    return m.group(1)


def add_data_args() -> list[str]:
    sep = ";" if os.name == "nt" else ":"
    result: list[str] = []
    for src, dest in (
        (ROOT / "vendor", "vendor"),
        (ROOT / "README.md", "."),
        (ROOT / "LICENSE", "."),
    ):
        if src.exists():
            result += ["--add-data", f"{src}{sep}{dest}"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Build standalone Folirina desktop distribution")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    system = platform.system()
    if system not in {"Windows", "Darwin", "Linux"}:
        raise SystemExit(f"Unsupported platform: {system}")
    if args.clean:
        shutil.rmtree(DIST, ignore_errors=True)
        shutil.rmtree(BUILD, ignore_errors=True)

    run([sys.executable, "scripts/prepare_direct_vendor.py"])
    version = read_version()
    print(f"Building {APP_NAME} {version} for {system} {platform.machine()}")
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", APP_NAME,
        "--paths", str(ROOT / "src"),
        "--collect-submodules", "manga_hd_transfer",
        "--exclude-module", "pytest",
        *add_data_args(),
    ]
    if system == "Darwin":
        cmd += ["--osx-bundle-identifier", BUNDLE_ID]
    cmd += [str(ROOT / "run_gui.py")]
    run(cmd)

    if system == "Darwin":
        app = DIST / f"{APP_NAME}.app"
        if not app.is_dir():
            raise SystemExit(f"Expected app bundle not found: {app}")
        plist = app / "Contents" / "Info.plist"
        if plist.exists():
            subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleShortVersionString {version}", str(plist)], check=False)
            subprocess.run(["/usr/libexec/PlistBuddy", "-c", f"Set :CFBundleVersion {version}", str(plist)], check=False)
            subprocess.run(["plutil", "-lint", str(plist)], check=True)
    else:
        app_dir = DIST / APP_NAME
        if not app_dir.is_dir():
            raise SystemExit(f"Expected distribution directory not found: {app_dir}")
    print(f"Build complete: {DIST}")


if __name__ == "__main__":
    main()
