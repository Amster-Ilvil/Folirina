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
    "final.png", "final_reviewed.png", "final_auto.png", "manual_effect_base.png",
    "review_applied.json", "review_base.png", "review_preview.png", "review_sync.json",
    "text_layer.png", "text_layer_reviewed.png", "chinese_transfer_layer.png",
    "mask_transfer_layer.png", "mask_transfer_layer_reviewed.png", "mask_transfer_mask.png", "mask_transfer.json",
    "direct_patch_layer.png", "direct_patch_layer_reviewed.png", "direct_patch_regions.png", "direct_patch.json",
    "aligned_overlay_reveal.json", "aligned_overlay_reveal_layer.png", "aligned_overlay_reveal_mask.png",
    "transparent_bubble_reveal.json", "final_rgba.png", "jp_layer_rgba.png", "cn_layer_rgb.png",
    "target_layer_erase_base.png", "target_layer_erase_effective_mask.png", "target_layer_erase_preview.png", "target_layer_erase.json",
    "target_layer_restore_base.png", "target_layer_restore_effective_mask.png", "target_layer_restore_preview.png", "target_layer_restore.json",
)


def create_run_snapshot(page_dir: str | Path, run_id: str) -> Path | None:
    page_dir = Path(page_dir)
    existing = [page_dir / name for name in _RUN_SNAPSHOT_FILES if (page_dir / name).exists()]
    if not existing:
        return None
    backup = page_dir / ".run_backup" / str(run_id)
    backup.mkdir(parents=True, exist_ok=True)
    manifest = []
    for src in existing:
        dst = backup / src.name
        try:
            # Hard links make the temporary safety snapshot essentially free on
            # the normal same-volume workspace; copy is the portable fallback.
            os.link(src, dst)
            method = "hardlink"
        except OSError:
            shutil.copy2(src, dst)
            method = "copy"
        manifest.append({"name": src.name, "method": method})
    save_json(backup / "manifest.json", {"files": manifest})
    return backup


def restore_run_snapshot(page_dir: str | Path, backup: str | Path | None) -> dict[str, Any]:
    page_dir = Path(page_dir)
    if not backup:
        return {"restored": 0, "backup": ""}
    backup = Path(backup)
    if not backup.exists():
        return {"restored": 0, "backup": str(backup)}
    restored = 0
    try:
        payload = load_json(backup / "manifest.json") if (backup / "manifest.json").exists() else {}
        rows = payload.get("files", []) if isinstance(payload, dict) else []
    except Exception:
        rows = []
    for row in rows:
        name = str(row.get("name") or "") if isinstance(row, dict) else ""
        if not name:
            continue
        src = backup / name
        if not src.exists():
            continue
        dst = page_dir / name
        try:
            _atomic_copy(src, dst)
            restored += 1
        except OSError:
            pass
    return {"restored": restored, "backup": str(backup)}


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
            write_image(book_final, image)
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
        for scope in ("mask_ocr", "ocr_reletter"):
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
