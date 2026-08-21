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
    # Runtime QApplication/QMainWindow icons are loaded through
    # importlib.resources, so package assets must be collected into the same
    # package-relative location inside PyInstaller bundles.
    package_assets = ROOT / "src" / "manga_hd_transfer" / "assets"
    if package_assets.is_dir():
        result += ["--add-data", f"{package_assets}{sep}manga_hd_transfer/assets"]
    for filename in ("README.md", "LICENSE"):
        path = ROOT / filename
        if path.is_file():
            result += ["--add-data", f"{path}{sep}."]
    return result


def platform_icon(system: str) -> Path:
    filename = {
        "Darwin": "icon.icns",
        "Windows": "icon.ico",
        "Linux": "icon.png",
    }[system]
    path = ROOT / "assets" / filename
    if not path.is_file():
        raise SystemExit(f"Required {system} icon is missing: {path}")
    return path


def write_windows_version_file(version: str) -> Path:
    parts = [int(p) if p.isdigit() else 0 for p in version.split(".")[:4]]
    while len(parts) < 4:
        parts.append(0)
    version_tuple = tuple(parts[:4])
    path = BUILD / "folirina-version-info.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={version_tuple!r},
    prodvers={version_tuple!r},
    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0, 0)
  ),
  kids=[
    StringFileInfo([StringTable('040904B0', [
      StringStruct('CompanyName', 'Amster-Ilvil'),
      StringStruct('FileDescription', 'Folirina Manga Translation Transfer Studio'),
      StringStruct('FileVersion', '{version}'),
      StringStruct('InternalName', 'Folirina'),
      StringStruct('OriginalFilename', 'Folirina.exe'),
      StringStruct('ProductName', 'Folirina'),
      StringStruct('ProductVersion', '{version}')
    ])]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])])
  ]
)
""", encoding="utf-8")
    return path


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

    icon = platform_icon(system)
    cmd += ["--icon", str(icon)]

    if system == "Darwin":
        cmd += ["--osx-bundle-identifier", BUNDLE_ID]
    elif system == "Windows":
        cmd += ["--version-file", str(write_windows_version_file(version))]

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
            commands = [
                f"Set :CFBundleShortVersionString {version}",
                f"Set :CFBundleVersion {version}",
                f"Set :CFBundleDisplayName {APP_NAME}",
                f"Set :CFBundleName {APP_NAME}",
            ]
            for command in commands:
                subprocess.run(["/usr/libexec/PlistBuddy", "-c", command, str(plist)], check=False)
            # PyInstaller creates CFBundleIconFile from --icon.  Fail the build
            # if the plist/resource linkage disappeared, because a window icon
            # alone is not sufficient for Finder/Dock branding.
            icon_name = subprocess.check_output(
                ["/usr/libexec/PlistBuddy", "-c", "Print :CFBundleIconFile", str(plist)],
                text=True,
            ).strip()
            if not icon_name:
                raise SystemExit("macOS bundle is missing CFBundleIconFile")
            resource_icon = app / "Contents" / "Resources" / icon_name
            if not resource_icon.is_file():
                raise SystemExit(f"macOS bundle icon resource missing: {resource_icon}")
            subprocess.run(["plutil", "-lint", str(plist)], check=True)
    else:
        app_dir = DIST / APP_NAME
        if not app_dir.is_dir():
            raise SystemExit(f"Expected distribution directory not found: {app_dir}")
        executable = app_dir / (f"{APP_NAME}.exe" if system == "Windows" else APP_NAME)
        if not executable.is_file():
            raise SystemExit(f"Expected executable not found: {executable}")
        if system == "Linux":
            shutil.copy2(ROOT / "assets" / "icon.png", app_dir / "folirina.png")
            desktop = app_dir / "Folirina.desktop"
            desktop.write_text(
                "[Desktop Entry]\nType=Application\nName=Folirina\n"
                "Exec=./Folirina\nIcon=folirina\nTerminal=false\n"
                "Categories=Graphics;Utility;\n",
                encoding="utf-8",
            )

    print(f"Build complete: {DIST}")


if __name__ == "__main__":
    main()
