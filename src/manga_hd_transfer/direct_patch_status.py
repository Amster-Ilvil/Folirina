from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .io_utils import load_json
from .schema_compat import as_dict, normalize_project, normalize_route_meta

REQUIRED_DIRECT_FILES = (
    'direct_patch_layer.png',
    'direct_patch_regions.png',
    'direct_patch.json',
)


def load_direct_patch_payload(page_dir: str | Path) -> Dict[str, Any]:
    root = Path(page_dir)
    payload: Dict[str, Any] = {}
    direct_json = root / 'direct_patch.json'
    if direct_json.exists():
        try:
            payload = as_dict(load_json(direct_json))
        except Exception:
            payload = {}
    if payload:
        return payload
    project_json = root / 'project.json'
    if project_json.exists():
        try:
            project = normalize_project(load_json(project_json))
            meta = as_dict(project.get('meta'))
            payload = normalize_route_meta(meta.get('direct_patch'))
        except Exception:
            payload = {}
    return payload or {}


def summarize_direct_patch_payload(page_dir: str | Path) -> Dict[str, Any]:
    root = Path(page_dir)
    payload = load_direct_patch_payload(root)
    diagnostics = as_dict(payload.get('diagnostics'))
    regions = list(payload.get('regions') or payload.get('records') or payload.get('bubble_matches') or [])
    applied_count = int(
        payload.get('applied_count', 0)
        or diagnostics.get('applied_count', 0)
        or diagnostics.get('applied_region_count', 0)
        or len(regions)
    )
    used = bool(payload.get('used', False) or payload.get('requested_mode') == 'direct_patch' or payload.get('mode') == 'direct_patch')
    accepted = bool(payload.get('accepted', used and applied_count > 0))
    reason = str(payload.get('reason', '') or diagnostics.get('reason', '') or ('ok' if accepted else 'no_direct_patch_payload'))
    strategy = str(payload.get('strategy', '') or diagnostics.get('strategy', '') or 'direct_borderless_overlay')
    missing = [name for name in REQUIRED_DIRECT_FILES if not (root / name).exists()]
    return {
        'used': used,
        'accepted': accepted,
        'reason': reason,
        'strategy': strategy,
        'applied_count': applied_count,
        'region_count': len(regions),
        'missing_files': missing,
        'complete': not missing,
        'payload': payload,
    }
