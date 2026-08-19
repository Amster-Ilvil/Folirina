from __future__ import annotations

"""Folirina desktop entry point and frozen-runtime bootstrap helpers."""

import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from importlib.machinery import PathFinder

DIRECT_PROBE_FLAG = "--direct-v234-probe"
RELEASE_SELFCHECK_FLAG = "--release-selfcheck"
DIRECT_VENDOR_REL = Path("vendor") / "v2.3.4-direct-contract-guard"
EXPECTED_DIRECT_ARCHIVE = "63d1df8d9bff426f22362a777ce7fe33e25da4aa9f68ad6f31053847a5607bc5"


def _emit(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    if stream is not None:
        print(message, file=stream)


def _bundle_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root).resolve()
    return Path(__file__).resolve().parent


def _ensure_source_vendor(root: Path) -> None:
    if bool(getattr(sys, "frozen", False)):
        return
    vendor = root / DIRECT_VENDOR_REL
    if (vendor / "SOURCE_FILE_SHA256.json").is_file():
        return
    script = root / "scripts" / "prepare_direct_vendor.py"
    if script.is_file():
        subprocess.run([sys.executable, str(script)], cwd=root, check=True)


def _verify_direct_vendor(root: Path) -> tuple[Path, Path]:
    _ensure_source_vendor(root)
    vendor = root / DIRECT_VENDOR_REL
    vendor_src = vendor / "src"
    runner = vendor / "run_page.py"
    archive_txt = vendor / "SOURCE_ARCHIVE_SHA256.txt"
    manifest_path = vendor / "SOURCE_FILE_SHA256.json"
    if not (vendor_src / "manga_hd_transfer" / "pipeline.py").is_file() or not runner.is_file():
        raise RuntimeError(f"Folirina Direct helper vendor missing: {vendor}")
    declared = archive_txt.read_text(encoding="utf-8").strip().split()[0]
    if declared != EXPECTED_DIRECT_ARCHIVE:
        raise RuntimeError(f"Folirina Direct archive mismatch: {declared}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = dict(manifest.get("files") or {})
    if len(files) != 176:
        raise RuntimeError(f"Folirina Direct manifest count mismatch: {len(files)}")
    for rel, expected in files.items():
        path = vendor_src / "manga_hd_transfer" / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != str(expected):
            raise RuntimeError(f"Folirina Direct vendor integrity failed: {rel}")
    return vendor, vendor_src


def _install_vendored_manga_finder(vendor_src: Path) -> None:
    # PyInstaller's frozen finder precedes normal sys.path. Override only the
    # locked manga_hd_transfer package; third-party modules keep using the
    # normal PyInstaller importers.
    class _VendoredMangaFinder:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "manga_hd_transfer":
                return PathFinder.find_spec(fullname, [str(vendor_src)])
            if fullname.startswith("manga_hd_transfer."):
                return PathFinder.find_spec(fullname, path)
            return None

    for name in tuple(sys.modules):
        if name == "manga_hd_transfer" or name.startswith("manga_hd_transfer."):
            sys.modules.pop(name, None)
    sys.meta_path.insert(0, _VendoredMangaFinder())
    sys.path.insert(0, str(vendor_src))
    os.environ["PYTHONPATH"] = str(vendor_src)


def _looks_like_direct_runner(path: str) -> bool:
    try:
        p = Path(path)
        return p.name == "run_page.py" and p.parent.name == "v2.3.4-direct-contract-guard"
    except Exception:
        return False


def _run_direct_helper_if_requested() -> int | None:
    # The unchanged v2.3.16 Direct bridge launches
    #   [sys.executable, vendor/run_page.py, request.json, response.json].
    # In a PyInstaller app sys.executable is Folirina itself, so intercept that
    # invocation here before importing the current package.
    if len(sys.argv) != 4 or not _looks_like_direct_runner(sys.argv[1]):
        return None
    try:
        vendor, vendor_src = _verify_direct_vendor(_bundle_root())
        _install_vendored_manga_finder(vendor_src)
        runner = vendor / "run_page.py"
        sys.argv = [str(runner), sys.argv[2], sys.argv[3]]
        runpy.run_path(str(runner), run_name="__main__")
    except SystemExit as exc:
        code = exc.code
        return int(code) if isinstance(code, int) else (0 if code is None else 1)
    except Exception as exc:
        _emit(f"Folirina Direct helper startup failed: {type(exc).__name__}: {exc}", error=True)
        return 70
    return 0


def _run_direct_probe_if_requested() -> int | None:
    if len(sys.argv) != 2 or sys.argv[1] != DIRECT_PROBE_FLAG:
        return None
    try:
        _vendor, vendor_src = _verify_direct_vendor(_bundle_root())
        _install_vendored_manga_finder(vendor_src)
        import manga_hd_transfer.pipeline as vendor_pipeline
        import manga_hd_transfer.version as vendor_version
        module_file = Path(vendor_pipeline.__file__).resolve()
        if vendor_src.resolve() not in module_file.parents:
            raise RuntimeError(f"Direct probe resolved wrong package: {module_file}")
        _emit(f"Folirina Direct probe PASS: version={vendor_version.__version__} source={module_file}")
        return 0
    except Exception as exc:
        _emit(f"Folirina Direct probe failed: {type(exc).__name__}: {exc}", error=True)
        return 71


def _patch_frozen_direct_bridge() -> None:
    if not bool(getattr(sys, "frozen", False)):
        return
    # direct_v234_bridge.py uses Path(__file__).parents[2] as project root. A
    # PyInstaller module normally sits one level shallower, so give it the
    # equivalent source-layout path without modifying the locked Direct bridge.
    import manga_hd_transfer.direct_v234_bridge as bridge
    bridge.__file__ = str(_bundle_root() / "src" / "manga_hd_transfer" / "direct_v234_bridge.py")


def _run_release_selfcheck_if_requested() -> int | None:
    if len(sys.argv) != 2 or sys.argv[1] != RELEASE_SELFCHECK_FLAG:
        return None
    try:
        _verify_direct_vendor(_bundle_root())
        _patch_frozen_direct_bridge()
        from manga_hd_transfer.version import __version__
        from manga_hd_transfer import direct_v234_bridge as bridge
        from PySide6.QtCore import qVersion
        if bridge._vendor_root().resolve() != (_bundle_root() / DIRECT_VENDOR_REL).resolve():
            raise RuntimeError(f"frozen Direct root mismatch: {bridge._vendor_root()}")
        _emit(f"Folirina release self-check PASS: version={__version__} Qt={qVersion()}")
        return 0
    except Exception as exc:
        _emit(f"Folirina release self-check failed: {type(exc).__name__}: {exc}", error=True)
        return 72


def main() -> int:
    for handler in (_run_direct_helper_if_requested, _run_direct_probe_if_requested, _run_release_selfcheck_if_requested):
        rc = handler()
        if rc is not None:
            return rc

    root = Path(__file__).resolve().parent
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    _ensure_source_vendor(root)

    from manga_hd_transfer.launcher import main as launcher_main
    _patch_frozen_direct_bridge()
    return int(launcher_main())


if __name__ == "__main__":
    raise SystemExit(main())
