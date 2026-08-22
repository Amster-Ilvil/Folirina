from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from .io_utils import load_json, save_json
from .schema_compat import as_dict, as_dict_rows


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _review_queue_present(project: dict[str, Any]) -> bool:
    meta = as_dict(project.get('meta'))
    for key in ('direct_patch', 'mask_replace', 'aligned_overlay_reveal', 'transparent_bubble_reveal'):
        row = as_dict(meta.get(key))
        if as_dict_rows(row.get('review_regions')) or as_dict_rows(row.get('manual_reletter_required')) or as_dict_rows(row.get('manual_effect_candidates')):
            return True
    return False


def cleanup_page_workspace(page_dir: str | Path, *, keep_review_preview: bool | None = None, keep_authority_alias: bool = False, aggressive_component_masks: bool = False) -> dict[str, int]:
    """Remove reproducible diagnostics while preserving every manual-edit contract.

    This is deliberately conservative: originals, final/result-state files,
    transfer/text layers, core clear masks, review overrides and every manual_*/
    target_layer_erase_* artifact remain.  Only artifacts that can be regenerated
    from those files are removed.
    """
    root = Path(page_dir)
    if not root.exists() or not root.is_dir():
        return {'files_removed': 0, 'dirs_removed': 0, 'bytes_freed': 0}
    project_path = root / 'project.json'
    try:
        project = as_dict(load_json(project_path)) if project_path.exists() else {}
    except Exception:
        project = {}
    if keep_review_preview is None:
        keep_review_preview = _review_queue_present(project)

    files: list[Path] = []
    files.extend(root.glob('debug_*.png'))
    files.extend([root / 'inpainted.png', root / 'editable.ora', root / 'editable.psd'])
    if not keep_review_preview:
        files.append(root / 'review_preview.png')

    # source_authority_original is commonly byte-identical to source_original.
    # Delete only when equality is proven; secondary-source workflows retain it.
    authority = root / 'source_authority_original.png'
    source = root / 'source_original.png'
    if not keep_authority_alias and authority.exists() and source.exists():
        try:
            if authority.stat().st_size == source.stat().st_size and _file_digest(authority) == _file_digest(source):
                files.append(authority)
                artifacts = as_dict(project.get('artifacts'))
                if artifacts:
                    artifacts['source_authority_original'] = str(source)
                    project['artifacts'] = artifacts
                    save_json(project_path, project)
        except OSError:
            pass

    removed = 0
    freed = 0
    seen: set[Path] = set()
    for path in files:
        if path in seen or not path.exists() or not path.is_file():
            continue
        seen.add(path)
        try:
            freed += int(path.stat().st_size)
            path.unlink()
            removed += 1
        except OSError:
            pass

    dirs_removed = 0
    # Component masks/bubble geometry are optional on disk, but when the user
    # explicitly generated them they remain valuable review/replay authority.
    # Automatic/page-finalization cleanup must never trade away that precision
    # just to save space.  Only an explicitly aggressive maintenance action may
    # remove them.
    if aggressive_component_masks:
        for name in ('masks', 'bubbles'):
            d = root / name
            if not d.exists() or not d.is_dir():
                continue
            try:
                for child in d.rglob('*'):
                    if child.is_file():
                        freed += int(child.stat().st_size)
                        removed += 1
                shutil.rmtree(d)
                dirs_removed += 1
            except OSError:
                pass
    return {'files_removed': removed, 'dirs_removed': dirs_removed, 'bytes_freed': freed}


def cleanup_output_workspace(output_dir: str | Path, *, aggressive_component_masks: bool = False) -> dict[str, int]:
    root = Path(output_dir)
    pages = root / 'pages' if (root / 'pages').is_dir() else root
    total = {'pages_scanned': 0, 'files_removed': 0, 'dirs_removed': 0, 'bytes_freed': 0}
    if not pages.exists():
        return total
    if (pages / 'project.json').exists() or (pages / 'target_original.png').exists():
        candidates = [pages]
    else:
        candidates = [p for p in pages.iterdir() if p.is_dir() and ((p / 'project.json').exists() or (p / 'target_original.png').exists())]
    for page in candidates:
        row = cleanup_page_workspace(page, aggressive_component_masks=aggressive_component_masks)
        total['pages_scanned'] += 1
        for key in ('files_removed', 'dirs_removed', 'bytes_freed'):
            total[key] += int(row.get(key, 0))
    return total


def page_has_active_review_state(page_dir: str | Path) -> bool:
    root = Path(page_dir)
    if (root / 'review_overrides.json').is_file():
        return True
    try:
        project = as_dict(load_json(root / 'project.json')) if (root / 'project.json').is_file() else {}
    except Exception:
        project = {}
    return _review_queue_present(project)


def prune_stage_cache(page_dir: str | Path, *, preserve_review_pages: bool = True) -> dict[str, int | bool]:
    """Delete only the recomputable ``.cache`` tree of a completed old page."""
    root = Path(page_dir)
    cache_dir = root / '.cache'
    if not cache_dir.is_dir():
        return {'pruned': False, 'bytes_freed': 0, 'files_removed': 0}
    if preserve_review_pages and page_has_active_review_state(root):
        return {'pruned': False, 'bytes_freed': 0, 'files_removed': 0}
    total = 0
    count = 0
    try:
        for child in cache_dir.rglob('*'):
            if child.is_file():
                try:
                    total += int(child.stat().st_size)
                    count += 1
                except OSError:
                    pass
        shutil.rmtree(cache_dir)
        return {'pruned': True, 'bytes_freed': total, 'files_removed': count}
    except OSError:
        return {'pruned': False, 'bytes_freed': 0, 'files_removed': 0}
