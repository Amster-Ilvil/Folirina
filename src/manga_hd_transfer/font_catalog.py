from __future__ import annotations

"""Folirina font discovery and persistent user font library.

The previous catalog only scanned font files that happened to exist on the
machine.  A font selected from Desktop therefore remained an external absolute
path; moving/deleting the source file made the project non-reproducible and the
same face could silently fall back to a different font later.

v2.3.85 adds a small application-owned font library.  Imported font bytes are
copied into the user's application data directory, keyed by SHA-256, and a JSON
manifest stores display metadata.  Rendering still consumes normal file paths,
so no mode renderer needs to know about the storage implementation.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from fontTools.ttLib import TTCollection, TTFont
except Exception:  # pragma: no cover - optional dependency
    TTCollection = None
    TTFont = None

_FONT_EXTS = {".ttf", ".ttc", ".otf", ".otc"}
_MANIFEST_SCHEMA = "folirina.font_library.v1"

# Font discovery is used by the project page, Workbench and OCR dialog.  A full
# recursive scan of macOS/System fonts at every widget construction made the new
# managed-library UI unnecessarily expensive.  Keep one process-local catalog
# snapshot; explicit Refresh buttons request a rescan, and importing a font
# invalidates it immediately.
_DISCOVERY_CACHE_KEY: tuple[str, str, str] | None = None
_DISCOVERY_CACHE_ROWS: list[dict[str, Any]] | None = None


def _discovery_cache_key() -> tuple[str, str, str]:
    try:
        lib = str(font_library_dir().resolve())
    except Exception:
        lib = str(font_library_dir())
    return (lib, os.environ.get("MHD_FONT_DIRS", ""), str(Path.cwd()))


def invalidate_font_discovery_cache() -> None:
    global _DISCOVERY_CACHE_KEY, _DISCOVERY_CACHE_ROWS
    _DISCOVERY_CACHE_KEY = None
    _DISCOVERY_CACHE_ROWS = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def font_library_dir() -> Path:
    """Return Folirina's persistent per-user font directory.

    ``FOLIRINA_FONT_LIBRARY_DIR`` is intentionally supported for tests,
    portable installs and advanced users.  The default follows each desktop
    platform's normal application-data convention and is independent of the
    source checkout / application bundle location.
    """
    override = os.environ.get("FOLIRINA_FONT_LIBRARY_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    system = platform.system().lower()
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / "Folirina" / "fonts"
    if system == "windows":
        root = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
        if root:
            return Path(root) / "Folirina" / "fonts"
        return Path.home() / "AppData" / "Roaming" / "Folirina" / "fonts"
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
    return root / "Folirina" / "fonts"


def font_library_manifest_path() -> Path:
    return font_library_dir() / "library.json"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_filename_part(value: str, *, fallback: str = "font") -> str:
    value = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", str(value or "")).strip(" ._")
    value = re.sub(r"\s+", "_", value)
    return (value[:80] or fallback)


def _name_from_table(font: Any, ids: tuple[int, ...]) -> str:
    try:
        table = font["name"]
    except Exception:
        return ""
    # Prefer typographic family/style and English records, then any Unicode name.
    candidates: list[tuple[int, str]] = []
    for rec in getattr(table, "names", []) or []:
        if int(getattr(rec, "nameID", -1)) not in ids:
            continue
        try:
            value = rec.toUnicode().strip()
        except Exception:
            continue
        if not value:
            continue
        score = 0
        if int(getattr(rec, "langID", -1)) in {0x409, 0}:
            score += 4
        if int(getattr(rec, "platformID", -1)) in {0, 3}:
            score += 2
        # Earlier ids in the tuple are preferred.
        score += max(0, 3 - ids.index(int(rec.nameID)))
        candidates.append((score, value))
    return max(candidates, default=(0, ""), key=lambda x: (x[0], len(x[1])))[1]


def inspect_font_file(path: str | Path) -> dict[str, Any]:
    """Validate a font file and return stable display/coverage metadata."""
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(str(p))
    ext = p.suffix.lower()
    if ext not in _FONT_EXTS:
        raise ValueError(f"不支持的字体格式：{p.suffix or '(无扩展名)'}")

    sha = _sha256_file(p)
    family = p.stem
    style = ""
    full_name = p.stem
    faces = 1
    glyph_count = 0
    cjk_codepoints = 0
    unicode_codepoints = 0

    if TTFont is not None:
        fonts: list[Any] = []
        collection = None
        try:
            if ext in {".ttc", ".otc"} and TTCollection is not None:
                collection = TTCollection(str(p), lazy=True)
                fonts = list(collection.fonts)
            else:
                fonts = [TTFont(str(p), lazy=True)]
            faces = max(1, len(fonts))
            if fonts:
                first = fonts[0]
                family = _name_from_table(first, (16, 1, 4)) or family
                style = _name_from_table(first, (17, 2))
                full_name = _name_from_table(first, (4, 16, 1)) or family
            cps: set[int] = set()
            for font in fonts:
                try:
                    cmap = font.getBestCmap() or {}
                    cps.update(int(cp) for cp in cmap)
                except Exception:
                    pass
                try:
                    glyph_count = max(glyph_count, int(font["maxp"].numGlyphs))
                except Exception:
                    pass
            unicode_codepoints = len(cps)
            cjk_codepoints = sum(
                1 for cp in cps
                if 0x3400 <= cp <= 0x9FFF or 0xF900 <= cp <= 0xFAFF
            )
        except Exception as exc:
            raise ValueError(f"字体文件无法解析：{p.name}: {exc}") from exc
        finally:
            for font in fonts:
                try:
                    font.close()
                except Exception:
                    pass
            if collection is not None:
                try:
                    collection.close()
                except Exception:
                    pass

    # Pillow uses the same FreeType backend as Folirina's OCR lettering.  A tiny
    # load test catches corrupt/unsupported files before we permanently import
    # them.  Import lazily so headless metadata tooling does not require Pillow.
    try:
        from PIL import ImageFont
        probe = ImageFont.truetype(str(p), 32)
        box = probe.getbbox("国Ag测试，。")
        if box is None or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("字体没有可用的字形度量")
        # Reject only truly pathological metrics. Decorative fonts may overshoot
        # an em, so keep the threshold intentionally generous.
        if (box[2] - box[0]) > 32 * 20 or (box[3] - box[1]) > 32 * 8:
            raise ValueError("字体字形度量异常，可能无法稳定排版")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"字体无法由 FreeType 加载：{p.name}: {exc}") from exc

    return {
        "sha256": sha,
        "source_name": p.name,
        "family": family,
        "style": style,
        "full_name": full_name,
        "faces": faces,
        "glyph_count": glyph_count,
        "unicode_codepoints": unicode_codepoints,
        "cjk_codepoints": cjk_codepoints,
        "cjk_capable": bool(cjk_codepoints >= 128),
        "size_bytes": int(p.stat().st_size),
        "extension": ext,
    }


def _read_manifest() -> dict[str, Any]:
    path = font_library_manifest_path()
    if not path.exists():
        return {"schema": _MANIFEST_SCHEMA, "updated_at": _utc_now(), "fonts": []}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        value = {}
    if not isinstance(value, dict):
        value = {}
    rows = value.get("fonts")
    if not isinstance(rows, list):
        rows = []
    return {"schema": _MANIFEST_SCHEMA, "updated_at": value.get("updated_at") or _utc_now(), "fonts": rows}


def _write_manifest(value: dict[str, Any]) -> None:
    root = font_library_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = font_library_manifest_path()
    payload = dict(value)
    payload["schema"] = _MANIFEST_SCHEMA
    payload["updated_at"] = _utc_now()
    fd, tmp_name = tempfile.mkstemp(prefix=".library-", suffix=".json", dir=str(root))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
            fh.flush(); os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        try:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)
        except OSError:
            pass


def import_font_to_library(path: str | Path) -> dict[str, Any]:
    """Copy one font into Folirina's persistent library and return its row.

    Content-addressing makes repeat imports idempotent and prevents a different
    file with the same human filename from silently replacing an existing face.
    """
    source = Path(path).expanduser().resolve()
    meta = inspect_font_file(source)
    root = font_library_dir()
    root.mkdir(parents=True, exist_ok=True)
    sha = str(meta["sha256"])
    stem = _safe_filename_part(str(meta.get("family") or source.stem))
    dest = root / f"{sha[:16]}_{stem}{source.suffix.lower()}"

    if not dest.exists() or _sha256_file(dest) != sha:
        fd, tmp_name = tempfile.mkstemp(prefix=".font-", suffix=source.suffix.lower(), dir=str(root))
        os.close(fd)
        try:
            shutil.copy2(source, tmp_name)
            if _sha256_file(Path(tmp_name)) != sha:
                raise IOError("字体复制后 SHA-256 校验失败")
            os.replace(tmp_name, dest)
        finally:
            try:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
            except OSError:
                pass

    manifest = _read_manifest()
    rows = [r for r in manifest.get("fonts", []) if isinstance(r, dict)]
    previous = next((r for r in rows if str(r.get("sha256") or "") == sha), None)
    row = {
        **meta,
        "id": sha,
        "path": str(dest),
        "name": str(meta.get("full_name") or meta.get("family") or dest.stem),
        "imported_at": str((previous or {}).get("imported_at") or _utc_now()),
        "last_seen_at": _utc_now(),
        "managed": True,
    }
    rows = [r for r in rows if str(r.get("sha256") or "") != sha]
    rows.append(row)
    rows.sort(key=lambda r: (str(r.get("name") or "").casefold(), str(r.get("sha256") or "")))
    manifest["fonts"] = rows
    _write_manifest(manifest)
    invalidate_font_discovery_cache()
    return row


def managed_fonts() -> list[dict[str, Any]]:
    """Return only healthy application-owned font rows, repairing stale paths."""
    manifest = _read_manifest()
    root = font_library_dir()
    out: list[dict[str, Any]] = []
    changed = False
    for raw in manifest.get("fonts", []):
        if not isinstance(raw, dict):
            changed = True; continue
        row = dict(raw)
        path = Path(str(row.get("path") or ""))
        sha = str(row.get("sha256") or "")
        if not path.is_file() and sha:
            matches = list(root.glob(f"{sha[:16]}_*")) if root.exists() else []
            if matches:
                path = matches[0]; row["path"] = str(path); changed = True
        if not path.is_file():
            # Keep the manifest clean; missing managed bytes must not remain a
            # selectable entry that later falls back to another font.
            changed = True; continue
        row["managed"] = True
        row.setdefault("name", row.get("full_name") or row.get("family") or path.stem)
        out.append(row)
    if changed:
        manifest["fonts"] = out
        _write_manifest(manifest)
    out.sort(key=lambda r: (str(r.get("name") or "").casefold(), str(r.get("path") or "")))
    return out


def _is_inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except Exception:
        return False


def persist_font_expression(value: str | None, *, strict: bool = False) -> str:
    """Import file-path tokens from a semicolon font chain into the library.

    Family aliases such as ``sans``/``serif`` and non-path tokens are preserved.
    This makes manual OCR text boxes and global Reletter settings durable even if
    the user picked a font from Desktop/Downloads.
    """
    raw = str(value or "").strip()
    if not raw:
        return ""
    tokens = [part.strip() for part in raw.replace("\n", ";").split(";")]
    root = font_library_dir()
    out: list[str] = []
    for token in tokens:
        if not token:
            continue
        p = Path(token).expanduser()
        if p.is_file() and p.suffix.lower() in _FONT_EXTS:
            if _is_inside(p, root):
                out.append(str(p.resolve()))
                continue
            try:
                out.append(str(import_font_to_library(p)["path"]))
            except Exception:
                if strict:
                    raise
                out.append(token)
        else:
            out.append(token)
    return ";".join(out)


def font_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    raw = os.environ.get("MHD_FONT_DIRS", "")
    for part in raw.split(os.pathsep):
        if part.strip():
            dirs.append(Path(part).expanduser())
    project_root = Path(__file__).resolve().parents[2]
    dirs.extend([
        font_library_dir(),
        Path.cwd() / "fonts",
        project_root / "fonts",
        Path.home() / "Library" / "Fonts",
        Path("/Library/Fonts"),
        Path("/System/Library/Fonts"),
        Path.home() / ".local" / "share" / "fonts",
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


def discover_fonts(*, limit: int = 300, force: bool = False) -> list[dict[str, Any]]:
    """Return managed fonts first, then machine/project fonts without duplicates.

    The expensive recursive system-font scan is cached for the lifetime of the
    process. ``force=True`` is reserved for the visible Refresh buttons so newly
    installed system fonts can still be discovered on demand.
    """
    global _DISCOVERY_CACHE_KEY, _DISCOVERY_CACHE_ROWS
    limit = max(1, int(limit))
    key = _discovery_cache_key()
    if not force and _DISCOVERY_CACHE_KEY == key and _DISCOVERY_CACHE_ROWS is not None:
        healthy = [row for row in _DISCOVERY_CACHE_ROWS if Path(str(row.get("path") or "")).is_file()]
        if len(healthy) != len(_DISCOVERY_CACHE_ROWS):
            _DISCOVERY_CACHE_ROWS = None
        else:
            return [dict(row) for row in healthy[:limit]]

    # Scan at least 300 entries on the first request. ProjectPage asks for 160
    # and Workbench/OCR dialogs ask for 180; this turns those into cheap slices
    # of the same snapshot rather than repeated directory walks.
    scan_limit = max(300, limit)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for row in managed_fonts():
        path = str(row.get("path") or "")
        if not path:
            continue
        try:
            resolved = str(Path(path).resolve())
        except Exception:
            resolved = path
        if resolved in seen:
            continue
        seen.add(resolved)
        item = dict(row)
        item["path"] = resolved
        item["root"] = str(font_library_dir())
        item["name"] = "★ " + str(item.get("name") or Path(resolved).stem)
        rows.append(item)
        if len(rows) >= scan_limit:
            break

    if len(rows) < scan_limit:
        for root in font_search_dirs():
            try:
                if _is_inside(root, font_library_dir()) or root.resolve() == font_library_dir().resolve():
                    # Already emitted from the manifest with real metadata.
                    continue
            except Exception:
                pass
            try:
                for p in root.rglob("*"):
                    if len(rows) >= scan_limit:
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
                    rows.append({"name": p.stem, "path": resolved, "root": str(root), "managed": False})
            except OSError:
                continue
            if len(rows) >= scan_limit:
                break
    rows.sort(key=lambda r: (0 if r.get("managed") else 1, str(r.get("name") or "").casefold(), str(r.get("path") or "").casefold()))
    _DISCOVERY_CACHE_KEY = key
    _DISCOVERY_CACHE_ROWS = [dict(row) for row in rows]
    return [dict(row) for row in rows[:limit]]
