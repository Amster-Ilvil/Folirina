from __future__ import annotations

"""Source-candidate discovery and rendition classification policies.

Kept outside the page orchestrator so sidecar/secondary-source discovery can be
tested without constructing ``TransferPipeline``.
"""

import json
from pathlib import Path

import cv2
import numpy as np

def _sidecar_with_suffix(path: str | Path, suffix: str) -> Path:
    p = Path(path)
    return p.with_suffix(suffix) if suffix.startswith(".") else p.parent / f"{p.stem}{suffix}"


def _load_additional_source_specs(source_path: str | Path, cfg) -> list[dict]:
    if not bool(getattr(cfg, "additional_source_enabled", True)):
        return []
    manifest = _sidecar_with_suffix(source_path, str(getattr(cfg, "additional_source_manifest_suffix", ".replace_sources.json")))
    if not manifest.exists():
        return []
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception:
        return []
    rows = payload.get("sources", payload) if isinstance(payload, dict) else payload
    out: list[dict] = []
    if not isinstance(rows, list):
        return out
    for row in rows[: max(0, int(getattr(cfg, "additional_source_max_candidates", 2)))]:
        if isinstance(row, str):
            item = {"path": row, "kind": "alternate"}
        elif isinstance(row, dict):
            item = {"path": row.get("path", ""), "kind": row.get("kind", "alternate")}
        else:
            continue
        rp = Path(item["path"])
        if not rp.is_absolute():
            rp = manifest.parent / rp
        if rp.exists():
            item["path"] = str(rp)
            out.append(item)
    return out


def _resolve_secondary_source_spec(primary_source_path: str | Path, target_path: str | Path, cfg) -> dict | None:
    if not bool(getattr(cfg, "enabled", False)):
        return None
    root_value = getattr(cfg, "secondary_source_dir", None)
    if not root_value:
        return None
    root = Path(root_value).expanduser()
    if root.is_file():
        return {"path": str(root), "kind": "secondary_dir", "origin": "dual_source"}
    if not root.is_dir():
        return None
    primary = Path(primary_source_path)
    target = Path(target_path)
    candidates = [root / primary.name, root / target.name]
    extensions = [primary.suffix, target.suffix, ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff"]
    for stem in (primary.stem, target.stem):
        for ext in extensions:
            if ext:
                candidates.append(root / f"{stem}{ext}")
    if bool(getattr(cfg, "recursive_lookup", False)):
        for stem in (primary.stem, target.stem):
            candidates.extend(root.rglob(f"{stem}.*"))
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            return {"path": str(candidate), "kind": "secondary_dir", "origin": "dual_source"}
    return None


def _saturation_p90(image: np.ndarray) -> float:
    if image.ndim != 3 or image.shape[2] < 3:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(np.percentile(hsv[..., 1], 90.0))


def _cross_rendition_monochrome_source(source: np.ndarray, target: np.ndarray) -> bool:
    """Detect BW/grayscale translated scan -> coloured master pairs."""
    return _saturation_p90(source) < 24.0 and _saturation_p90(target) >= 24.0


__all__ = ['_sidecar_with_suffix', '_load_additional_source_specs', '_resolve_secondary_source_spec', '_saturation_p90', '_cross_rendition_monochrome_source']
