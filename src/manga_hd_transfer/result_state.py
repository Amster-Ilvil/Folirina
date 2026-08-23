from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable, Any

from .io_utils import load_json, save_json, write_image
from .schema_compat import as_dict, normalize_project
from .review_artifacts import STATIC_REVIEW_INPUTS, dynamic_review_artifact_names
from .mode_contracts import MODE_DERIVED_ARTIFACTS


@dataclass(frozen=True, slots=True)
class ResultState:
    current: Path | None
    reviewed: Path | None
    automatic: Path | None
    stable_manual_base: Path | None




# Files that represent the currently published/visible page state.  A fresh
# automatic run may delete or rewrite several of these before OCR/registration
# has fully succeeded.  Snapshot them so a failed run can restore the previous
# known-good page instead of leaving a half-cleared workspace visible in the GUI.
_RUN_SNAPSHOT_FILES = (
    "run_receipt.json", "project.json", "qa.json", "transfer_audit.json",
    "page_management.json", "last_run_state.json",
    "final.png", "final_reviewed.png", "final_auto.png", "manual_effect_base.png",
    "review_applied.json", "review_base.png", "review_preview.png", "review_sync.json",
    "text_layer.png", "text_layer_reviewed.png", "chinese_transfer_layer.png",
    "mask_transfer_layer.png", "mask_transfer_layer_reviewed.png", "mask_transfer_mask.png", "mask_transfer.json",
    "hybrid_transfer_layer.png", "hybrid_transfer_layer_reviewed.png", "hybrid_transfer_mask.png", "hybrid_transfer.json",
    "hybrid_text_layer.png", "hybrid_text_layer_reviewed.png", "reletter_text_layer.png", "reletter_text_layer_reviewed.png", "reletter.json",
    "direct_patch_layer.png", "direct_patch_layer_reviewed.png", "direct_patch_regions.png", "direct_patch.json",
    "aligned_overlay_reveal.json", "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
    "aligned_overlay_reveal_hole_mask.png", "aligned_overlay_reveal_erase_mask.png",
    "aligned_overlay_reveal_regions.png", "aligned_overlay_reveal_source_ink.png",
    "aligned_overlay_reveal_diff_mask.png", "aligned_overlay_reveal_judgment.png",
    "aligned_overlay_reveal_validation.json",
    "transparent_bubble_reveal.json", "final_rgba.png", "jp_layer_rgba.png", "cn_layer_rgb.png",
    "target_layer_erase_base.png", "target_layer_erase_effective_mask.png", "target_layer_erase_preview.png", "target_layer_erase.json",
    "target_layer_restore_base.png", "target_layer_restore_effective_mask.png", "target_layer_restore_preview.png", "target_layer_restore.json",
) + STATIC_REVIEW_INPUTS + tuple(
    name for names in MODE_DERIVED_ARTIFACTS.values() for name in names
)

# These paths are written in-place by the frozen Aligned Reveal compatibility
# renderer. They must be real copies in a rollback snapshot; a hard link would
# share the same inode and be truncated together with the live artifact.
_RUN_SNAPSHOT_COPY_NAMES = {
    "final.png", "review_preview.png",
    # The isolated Direct compatibility renderer still encodes directly into
    # these published files. Keep the frozen renderer untouched and make the
    # transaction boundary copy-on-snapshot instead.
    "direct_patch_layer.png", "direct_patch_regions.png",
    "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
    "aligned_overlay_reveal_hole_mask.png", "aligned_overlay_reveal_erase_mask.png",
    "aligned_overlay_reveal_regions.png", "aligned_overlay_reveal_source_ink.png",
    "aligned_overlay_reveal_diff_mask.png", "aligned_overlay_reveal_judgment.png",
    "jp_layer_rgba.png", "cn_layer_rgb.png",
}


def create_run_snapshot(page_dir: str | Path, run_id: str) -> Path:
    """Snapshot the full published-state *presence contract* for one page run.

    The caller must already hold :class:`PageRunGuard`.  This keeps the backup at
    one coherent point in time instead of racing a concurrent writer.  A failed
    snapshot creation is self-cleaning: a partial ``.run_backup`` directory must
    never accumulate or later be mistaken for a recoverable transaction.

    Large image artifacts use hard links because Folirina publishes them through
    atomic sibling replacement. JSON metadata is copied instead: a few frozen or
    compatibility writers still truncate JSON in place, which would mutate a
    hard-linked backup inode and make rollback ineffective.
    """
    page_dir = Path(page_dir)
    backup = page_dir / ".run_backup" / str(run_id)
    backup.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    try:
        snapshot_names = tuple(dict.fromkeys((*_RUN_SNAPSHOT_FILES, *dynamic_review_artifact_names(page_dir))))
        for name in snapshot_names:
            src = page_dir / name
            existed = src.exists() and src.is_file()
            row: dict[str, Any] = {"name": name, "existed": bool(existed), "method": "absent"}
            if existed:
                dst = backup / name
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.suffix.casefold() == ".json" or name in _RUN_SNAPSHOT_COPY_NAMES:
                    shutil.copy2(src, dst)
                    row["method"] = "copy"
                else:
                    try:
                        # Atomic image publication replaces the destination inode, so
                        # a hard link safely preserves the old image at near-zero cost.
                        os.link(src, dst)
                        row["method"] = "hardlink"
                    except OSError:
                        shutil.copy2(src, dst)
                        row["method"] = "copy"
            manifest.append(row)
        # The review archive is recovery state too. A mode switch may publish
        # an archive successfully and then fail later in the renderer/commit
        # path. Snapshot the whole small archive tree so rollback restores the
        # page *and* its recovery history to the exact pre-run state.
        archive_root = page_dir / "review_archive"
        archive_backup = backup / "review_archive"
        if archive_root.is_symlink():
            raise RuntimeError("unsafe review_archive symlink")
        archive_existed = archive_root.exists() and archive_root.is_dir()
        if archive_root.exists() and not archive_existed:
            raise RuntimeError("review_archive exists but is not a directory")
        archive_files: list[dict[str, Any]] = []
        if archive_existed:
            for src in sorted(archive_root.rglob("*")):
                if not src.is_file() or src.is_symlink():
                    continue
                rel = src.relative_to(archive_root)
                dst = archive_backup / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if src.suffix.casefold() == ".json":
                    shutil.copy2(src, dst)
                    method = "copy"
                else:
                    try:
                        os.link(src, dst)
                        method = "hardlink"
                    except OSError:
                        shutil.copy2(src, dst)
                        method = "copy"
                archive_files.append({"name": rel.as_posix(), "method": method})

        save_json(backup / "manifest.json", {
            "schema": "manga_hd_translation_transfer.run_snapshot.v3",
            "files": manifest,
            "review_archive": {
                "existed": bool(archive_existed),
                "files": archive_files,
            },
        })
        return backup
    except Exception:
        # A snapshot without a complete manifest is not a safe recovery point.
        shutil.rmtree(backup, ignore_errors=True)
        parent = backup.parent
        try:
            if parent.exists() and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            pass
        raise


def run_snapshot_has_existing(backup: str | Path | None) -> bool:
    if not backup:
        return False
    path = Path(backup) / "manifest.json"
    try:
        payload = load_json(path) if path.exists() else {}
        rows = payload.get("files", []) if isinstance(payload, dict) else []
        return any(bool(row.get("existed", True)) for row in rows if isinstance(row, dict))
    except Exception:
        return False


def restore_run_snapshot(page_dir: str | Path, backup: str | Path | None) -> dict[str, Any]:
    """Restore a run snapshot and report whether recovery was complete.

    A backup is discarded only when ``success`` is true.  Any unreadable
    manifest, missing expected backup artifact, failed replacement, or failed
    removal of a file that was absent before the run keeps the snapshot around
    for manual recovery and diagnostics.
    """
    page_dir = Path(page_dir)
    failed: list[dict[str, str]] = []
    if not backup:
        return {"restored": 0, "removed_new": 0, "failed": failed, "success": True, "backup": ""}
    backup = Path(backup)
    if not backup.exists():
        failed.append({"name": "manifest.json", "operation": "locate_backup", "error": "backup_missing"})
        return {"restored": 0, "removed_new": 0, "failed": failed, "success": False, "backup": str(backup)}

    manifest_path = backup / "manifest.json"
    try:
        if not manifest_path.exists():
            raise FileNotFoundError(manifest_path)
        payload = load_json(manifest_path)
        rows = payload.get("files", []) if isinstance(payload, dict) else []
        if not isinstance(rows, list):
            raise ValueError("snapshot manifest files must be a list")
    except Exception as exc:
        failed.append({"name": "manifest.json", "operation": "read_manifest", "error": f"{type(exc).__name__}: {exc}"})
        return {"restored": 0, "removed_new": 0, "failed": failed, "success": False, "backup": str(backup)}

    restored = 0
    removed_new = 0

    # v3 snapshots also cover the recovery archive itself. A failed mode switch
    # must not leave a newly-created or partially-overwritten review_archive
    # behind after the live review inputs have been restored.
    archive_contract = payload.get("review_archive") if isinstance(payload, dict) else None
    if isinstance(archive_contract, dict):
        archive_dst = page_dir / "review_archive"
        archive_src = backup / "review_archive"
        archive_existed = bool(archive_contract.get("existed", False))
        archive_rows = archive_contract.get("files", [])
        if not isinstance(archive_rows, list):
            failed.append({"name": "review_archive", "operation": "parse_manifest", "error": "archive files must be a list"})
        else:
            try:
                if archive_dst.exists():
                    if archive_dst.is_symlink() or not archive_dst.is_dir():
                        raise NotADirectoryError(archive_dst)
                    shutil.rmtree(archive_dst)
                if archive_existed:
                    archive_dst.mkdir(parents=True, exist_ok=True)
                    for archive_row in archive_rows:
                        rel_name = str(archive_row.get("name") or "") if isinstance(archive_row, dict) else ""
                        rel = Path(rel_name)
                        if (not rel_name or rel.is_absolute() or ".." in rel.parts):
                            raise ValueError(f"unsafe review_archive entry: {rel_name!r}")
                        src = archive_src / rel
                        dst = archive_dst / rel
                        if not src.is_file():
                            raise FileNotFoundError(src)
                        _atomic_copy(src, dst)
                elif archive_dst.exists():
                    shutil.rmtree(archive_dst)
            except (OSError, ValueError) as exc:
                failed.append({"name": "review_archive", "operation": "restore_tree", "error": f"{type(exc).__name__}: {exc}"})

    for row in rows:
        name = str(row.get("name") or "") if isinstance(row, dict) else ""
        if not name:
            failed.append({"name": "", "operation": "parse_manifest", "error": "empty artifact name"})
            continue
        dst = page_dir / name
        # v1 manifests did not carry ``existed``; every listed row represented
        # an existing source and therefore remains backward-compatible here.
        existed = bool(row.get("existed", True)) if isinstance(row, dict) else True
        if not existed:
            try:
                if dst.exists():
                    if not dst.is_file():
                        raise IsADirectoryError(dst)
                    dst.unlink()
                    removed_new += 1
            except OSError as exc:
                failed.append({"name": name, "operation": "remove_new", "error": f"{type(exc).__name__}: {exc}"})
            continue
        src = backup / name
        if not src.exists() or not src.is_file():
            failed.append({"name": name, "operation": "restore", "error": "backup_artifact_missing"})
            continue
        try:
            _atomic_copy(src, dst)
            restored += 1
        except OSError as exc:
            failed.append({"name": name, "operation": "restore", "error": f"{type(exc).__name__}: {exc}"})
    return {
        "restored": restored,
        "removed_new": removed_new,
        "failed": failed,
        "success": not failed,
        "backup": str(backup),
    }


def discard_run_snapshot(backup: str | Path | None) -> None:
    if not backup:
        return
    try:
        shutil.rmtree(Path(backup), ignore_errors=True)
        parent = Path(backup).parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass

def _existing_file(value: str | Path | None) -> Path | None:
    if not value:
        return None
    try:
        p = Path(value)
        return p if p.exists() and p.is_file() else None
    except (OSError, TypeError, ValueError):
        return None


def newest_existing(paths: Iterable[str | Path | None]) -> Path | None:
    rows: list[tuple[int, int, Path]] = []
    for order, value in enumerate(paths):
        p = _existing_file(value)
        if p is None:
            continue
        try:
            rows.append((int(p.stat().st_mtime_ns), -order, p))
        except OSError:
            continue
    if not rows:
        return None
    rows.sort(reverse=True)
    return rows[0][2]


def resolve_result_state(page_dir: str | Path, artifacts: Any = None, extra_candidates: Iterable[str | Path | None] = ()) -> ResultState:
    page_dir = Path(page_dir)
    art = as_dict(artifacts)
    reviewed = _existing_file(page_dir / "final_reviewed.png") or _existing_file(art.get("final_reviewed"))
    automatic = _existing_file(page_dir / "final.png") or _existing_file(art.get("final"))
    current = newest_existing([
        reviewed,
        automatic,
        art.get("book_final"),
        *list(extra_candidates),
    ])
    stable = _existing_file(page_dir / "final_auto.png") or _existing_file(page_dir / "manual_effect_base.png")
    return ResultState(current=current, reviewed=reviewed, automatic=automatic, stable_manual_base=stable)



def commit_automatic_result(
    page_dir: str | Path,
    image,
    final_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Write a fresh automatic result without creating manual-review state.

    Pixel modules must never own ``final.png``.  New automatic routes hand their
    rendered image to this result-state boundary; reviewed synchronization remains
    the responsibility of :func:`commit_reviewed_result`.
    """
    page_dir = Path(page_dir)
    page_dir.mkdir(parents=True, exist_ok=True)
    local_final = page_dir / "final.png"
    write_image(local_final, image)
    book_final = None
    if final_path is not None:
        book_final = Path(final_path)
        if book_final.resolve() != local_final.resolve():
            # The page-local PNG is already fully encoded. Publishing those exact
            # bytes avoids a second full-resolution PNG encode on Reveal routes and
            # keeps the book mirror byte-identical to the authoritative page result.
            atomic_copy_file(local_final, book_final)
    return local_final, book_final

def manual_baseline_path(page_dir: str | Path) -> Path:
    """Return the immutable pre-manual baseline, with legacy migration rules.

    New sessions use ``final_auto.png``.  For an old project that only has
    ``manual_effect_base.png``, a review-sync marker means that legacy file is
    still the trustworthy pre-manual image because ``final.png`` may already be a
    reviewed mirror.  Without that marker, a newer ``final.png`` is treated as a
    fresh automatic reprocess and wins over the stale legacy base.
    """
    page_dir = Path(page_dir)
    stable = page_dir / "final_auto.png"
    if stable.exists():
        return stable
    legacy = page_dir / "manual_effect_base.png"
    automatic = page_dir / "final.png"
    if legacy.exists():
        if (page_dir / "review_sync.json").exists():
            return legacy
        try:
            if automatic.exists() and automatic.stat().st_mtime_ns > legacy.stat().st_mtime_ns:
                return automatic
        except OSError:
            pass
        return legacy
    return automatic


def _atomic_copy(src: Path, dst: Path) -> None:
    """Copy one published result through a unique sibling temporary file.

    Review/apply and book-final synchronization can be triggered in quick
    succession by the GUI. A fixed ``.tmp-sync`` filename lets overlapping
    operations trample each other's staging file. Unique sibling temporaries
    preserve the same atomic-replace semantics without that race.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{dst.name}.", suffix=".tmp-sync", dir=str(dst.parent))
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        shutil.copyfile(src, tmp)
        try:
            with tmp.open("rb") as fh:
                os.fsync(fh.fileno())
        except OSError:
            pass
        os.replace(tmp, dst)
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def atomic_copy_file(src: str | Path, dst: str | Path) -> Path:
    """Atomically publish an existing file to another visible path."""
    source = Path(src)
    target = Path(dst)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(source)
    if source.resolve() != target.resolve():
        _atomic_copy(source, target)
    return target


def ensure_manual_baseline(page_dir: str | Path, preferred_source: str | Path | None = None) -> Path:
    """Freeze one stable image before the first manual omission edit.

    The stable image is independent from the mutable visible ``final.png``.  A
    legacy ``manual_effect_base.png`` alias is maintained for old projects and
    diagnostics, but all new code should consume ``final_auto.png``.
    """
    page_dir = Path(page_dir)
    stable = page_dir / "final_auto.png"
    legacy = page_dir / "manual_effect_base.png"
    if stable.exists():
        if not legacy.exists():
            _atomic_copy(stable, legacy)
        return stable

    source = _existing_file(preferred_source)
    if source is None:
        # Legacy frozen state has special migration semantics; without one, use
        # the freshest visible result so a clear-mask/manual review completed
        # before the first Reveal becomes part of the stable baseline.
        if legacy.exists():
            source = _existing_file(manual_baseline_path(page_dir))
        else:
            state = resolve_result_state(page_dir)
            source = state.current
    if source is None:
        source = _existing_file(page_dir / "target_original.png")
    if source is None:
        raise FileNotFoundError(f"No stable manual baseline source in {page_dir}")

    _atomic_copy(source, stable)
    if not legacy.exists():
        _atomic_copy(source, legacy)
    return stable


def invalidate_manual_review_state(page_dir: str | Path) -> None:
    """Invalidate only state that must never survive a fresh automatic process."""
    page_dir = Path(page_dir)
    for name in (
        "final_auto.png", "manual_effect_base.png", "review_sync.json", "manual_gui_flow.json",
        "target_layer_erase_base.png", "target_layer_erase_effective_mask.png",
        "target_layer_erase_chinese_protect_mask.png", "target_layer_erase_preview.png",
        "target_layer_erase.json",
        "target_layer_restore_base.png", "target_layer_restore_effective_mask.png",
        "target_layer_restore_preview.png", "target_layer_restore.json",
    ):
        try:
            (page_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    # OCR editor blocks are durable user data, but their cached base/render state
    # belongs to the previous automatic result. Keep blocks.json and invalidate
    # only derived images/metadata so a fresh OCR run cannot inherit old shadows.
    ocr_root = page_dir / "ocr_edit"
    if ocr_root.exists():
        for scope in ("mask_ocr", "review_ocr", "ocr_reletter"):
            scope_dir = ocr_root / scope
            for name in ("base.png", "base_state.json", "render_state.json", "final.png"):
                try:
                    (scope_dir / name).unlink(missing_ok=True)
                except OSError:
                    pass


def _file_sha256(path: Path) -> str:
    h = sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def commit_reviewed_result(page_dir: str | Path, final_reviewed: str | Path, *, update_project: bool = True) -> Path:
    """Expose reviewed pixels atomically to every reader and persist one state.

    This is the single synchronization contract shared by Qt, Web Review, CLI and
    the review compositor.  ``final_reviewed.png`` remains the explicit review
    artifact; ``final.png`` is its compatibility mirror; ``final_auto.png`` is
    immutable and never touched here.
    """
    page_dir = Path(page_dir)
    reviewed = Path(final_reviewed)
    if not reviewed.exists():
        raise FileNotFoundError(reviewed)
    local_final = page_dir / "final.png"
    if reviewed.resolve() != local_final.resolve():
        _atomic_copy(reviewed, local_final)

    reviewed_hash = _file_sha256(reviewed)
    final_hash = _file_sha256(local_final)
    if reviewed_hash != final_hash:
        raise RuntimeError("reviewed/final synchronization hash mismatch")

    payload = {
        "schema": "manga_hd_translation_transfer.review_sync.v3",
        "final_reviewed": str(reviewed),
        "page_local_final": str(local_final),
        "stable_manual_base": str(manual_baseline_path(page_dir)),
        "synced": True,
        "sha256": reviewed_hash,
        "bytes": int(reviewed.stat().st_size),
    }
    save_json(page_dir / "review_sync.json", payload)

    if update_project:
        project_path = page_dir / "project.json"
        if project_path.exists():
            project = normalize_project(load_json(project_path))
            artifacts = as_dict(project.get("artifacts"))
            artifacts["final"] = str(local_final)
            artifacts["final_reviewed"] = str(reviewed)
            artifacts["final_auto"] = str(page_dir / "final_auto.png") if (page_dir / "final_auto.png").exists() else ""
            project["artifacts"] = artifacts
            meta = as_dict(project.get("meta"))
            meta["review_sync"] = payload
            project["meta"] = meta
            save_json(project_path, project)
    return reviewed
