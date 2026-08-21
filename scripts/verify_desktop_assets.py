from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def _image_info(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing icon asset: {path}")
    with Image.open(path) as image:
        image.load()
        return {"format": image.format, "size": list(image.size), "mode": image.mode}


def run() -> dict:
    root_assets = ROOT / "assets"
    pkg_assets = ROOT / "src" / "manga_hd_transfer" / "assets"
    png = root_assets / "icon.png"
    ico = root_assets / "icon.ico"
    icns = root_assets / "icon.icns"
    packaged_png = pkg_assets / "folirina_app_icon.png"
    packaged_ico = pkg_assets / "folirina_app_icon.ico"

    info = {
        "root_png": _image_info(png),
        "root_ico": _image_info(ico),
        "root_icns": _image_info(icns),
        "package_png": _image_info(packaged_png),
        "package_ico": _image_info(packaged_ico),
    }
    with Image.open(png).convert("RGBA") as image:
        if image.size != (1024, 1024):
            raise RuntimeError(f"root icon must be 1024x1024, got {image.size}")
        alpha = image.getchannel("A")
        corners = [alpha.getpixel((0, 0)), alpha.getpixel((1023, 0)),
                   alpha.getpixel((0, 1023)), alpha.getpixel((1023, 1023))]
        center = alpha.getpixel((512, 512))
        if any(value != 0 for value in corners):
            raise RuntimeError(f"rounded icon corners are not transparent: {corners}")
        if center < 250:
            raise RuntimeError(f"icon center unexpectedly transparent: {center}")
        info["transparent_corners"] = True

    build_text = (ROOT / "scripts" / "build_desktop.py").read_text(encoding="utf-8")
    required = [
        '"Darwin": "icon.icns"',
        '"Windows": "icon.ico"',
        '"Linux": "icon.png"',
        'manga_hd_transfer/assets',
        '--version-file',
        'CFBundleIconFile',
        'Folirina.desktop',
    ]
    missing = [token for token in required if token not in build_text]
    if missing:
        raise RuntimeError("desktop build contract missing: " + ", ".join(missing))
    info["build_contract"] = True
    return {"pass": True, "details": info}


def main() -> int:
    try:
        result = run()
    except Exception as exc:
        print(json.dumps({"pass": False, "error": str(exc)}, ensure_ascii=False, indent=2))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
