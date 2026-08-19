from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CURRENT = ROOT / "src" / "manga_hd_transfer"
VENDOR = ROOT / "vendor" / "v2.3.4-direct-contract-guard"
PATCH = ROOT / ".release" / "direct-v234-patch.tar.xz"
EXPECTED_ARCHIVE = "63d1df8d9bff426f22362a777ce7fe33e25da4aa9f68ad6f31053847a5607bc5"


def _safe_extract(tf: tarfile.TarFile, target: Path) -> None:
    target_resolved = target.resolve()
    for member in tf.getmembers():
        candidate = (target / member.name).resolve()
        if candidate != target_resolved and target_resolved not in candidate.parents:
            raise RuntimeError(f"unsafe archive member: {member.name}")
    try:
        tf.extractall(target, filter="data")
    except TypeError:  # Python < 3.12
        tf.extractall(target)


def reconstruct() -> None:
    if not CURRENT.is_dir():
        raise RuntimeError(f"current source missing: {CURRENT}")
    if not PATCH.is_file():
        raise RuntimeError(f"Direct vendor patch missing: {PATCH}")
    shutil.rmtree(VENDOR, ignore_errors=True)
    (VENDOR / "src").mkdir(parents=True, exist_ok=True)
    shutil.copytree(CURRENT, VENDOR / "src" / "manga_hd_transfer", dirs_exist_ok=True)
    with tarfile.open(PATCH, "r:xz") as tf:
        _safe_extract(tf, VENDOR)

    # Direct is an exact locked runtime: current files added after v2.3.4 must
    # not leak into the isolated vendor package.
    manifest_path = VENDOR / "SOURCE_FILE_SHA256.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    allowed = {Path(rel).as_posix() for rel in dict(manifest.get("files") or {})}
    source_root = VENDOR / "src" / "manga_hd_transfer"
    for path in sorted(source_root.rglob("*"), reverse=True):
        if path.is_file() and path.relative_to(source_root).as_posix() not in allowed:
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def verify() -> None:
    archive_txt = VENDOR / "SOURCE_ARCHIVE_SHA256.txt"
    manifest_path = VENDOR / "SOURCE_FILE_SHA256.json"
    runner = VENDOR / "run_page.py"
    if not archive_txt.is_file() or not manifest_path.is_file() or not runner.is_file():
        raise RuntimeError("Direct vendor metadata/runner missing")
    declared = archive_txt.read_text(encoding="utf-8").strip().split()[0]
    if declared != EXPECTED_ARCHIVE:
        raise RuntimeError(f"Direct archive fingerprint mismatch: {declared}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = dict(manifest.get("files") or {})
    if len(files) != 176:
        raise RuntimeError(f"unexpected Direct manifest file count: {len(files)}")
    source_root = VENDOR / "src" / "manga_hd_transfer"
    bad: list[str] = []
    for rel, expected in sorted(files.items()):
        path = source_root / rel
        if not path.is_file():
            bad.append(f"{rel}:missing")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != str(expected):
            bad.append(f"{rel}:sha256")
    if bad:
        raise RuntimeError("Direct vendor integrity failed: " + ", ".join(bad[:20]))
    print(f"Direct vendor verified: {len(files)} files; archive={declared}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        reconstruct()
    verify()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
