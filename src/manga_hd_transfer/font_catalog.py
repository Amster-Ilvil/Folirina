from __future__ import annotations

"""Small font catalog used by reletter UI.

No font files are bundled or copied. The catalog only discovers paths already
available on the user's machine or inside a user-managed project ``fonts`` folder.
"""

import os
from pathlib import Path

_FONT_EXTS = {".ttf", ".ttc", ".otf", ".otc"}


def font_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    raw = os.environ.get("MHD_FONT_DIRS", "")
    for part in raw.split(os.pathsep):
        if part.strip():
            dirs.append(Path(part).expanduser())
    project_root = Path(__file__).resolve().parents[2]
    dirs.extend([
        Path.cwd() / "fonts",
        project_root / "fonts",
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / ".fonts",
        Path("/usr/share/fonts"),
    ])
    out: list[Path] = []
    seen: set[str] = set()
    for d in dirs:
        try:
            key = str(d.resolve())
        except Exception:
            key = str(d)
        if key in seen:
            continue
        seen.add(key)
        if d.exists() and d.is_dir():
            out.append(d)
    return out


def discover_fonts(*, limit: int = 300) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in font_search_dirs():
        try:
            it = root.rglob("*")
            for p in it:
                if len(rows) >= limit:
                    break
                if not p.is_file() or p.suffix.lower() not in _FONT_EXTS:
                    continue
                try:
                    resolved = str(p.resolve())
                except Exception:
                    resolved = str(p)
                if resolved in seen:
                    continue
                seen.add(resolved)
                rows.append({"name": p.stem, "path": resolved, "root": str(root)})
        except OSError:
            continue
        if len(rows) >= limit:
            break
    rows.sort(key=lambda r: (r["name"].lower(), r["path"].lower()))
    return rows
