from __future__ import annotations

"""Deterministic local cache and batch-resume helpers.

No source image pixels are copied into the cache. Keys use a fast sampled file
hash + relevant configuration. Stage payloads are JSON/NPZ and live under the
selected output directory, so deleting the output also removes the cache.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .models import (
    BubbleInstance,
    LetteringResult,
    PagePair,
    PageProject,
    QAItem,
    RegistrationResult,
    TextBlock,
    TextUnit,
    UnitMatch,
)
from .schema_compat import as_dict, as_dict_rows

CACHE_SCHEMA = "mhd-cache-v2"


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def file_signature(path: str | Path, sample_bytes: int = 65536) -> str:
    p = Path(path)
    st = p.stat()
    h = hashlib.sha256()
    h.update(f"{p.name}|{st.st_size}".encode())
    with p.open("rb") as f:
        size = st.st_size
        positions = [0]
        if size > sample_bytes * 2:
            positions.append(max(0, size // 2 - sample_bytes // 2))
        if size > sample_bytes:
            positions.append(max(0, size - sample_bytes))
        for pos in sorted(set(positions)):
            f.seek(pos)
            h.update(f.read(sample_bytes))
    return h.hexdigest()[:24]


def config_signature(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return _json_hash(value)


def page_job_fingerprint(pair: PagePair, config: Any) -> str:
    cfg = config.model_dump(mode="json") if hasattr(config, "model_dump") else dict(config)
    # Operational knobs do not alter page pixels. Excluding them means toggling
    # resume/cache/CPU threads/MPS does not invalidate already completed pages.
    if isinstance(cfg, dict):
        cfg = dict(cfg)
        for key in ("batch", "cache", "runtime"):
            cfg.pop(key, None)
    payload = {
        "schema": CACHE_SCHEMA,
        "source": file_signature(pair.source_path),
        "target": file_signature(pair.target_path),
        "pair": {"si": pair.source_index, "ti": pair.target_index},
        "config": cfg,
    }
    return _json_hash(payload)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class PageStageCache:
    def __init__(self, page_root: str | Path, enabled: bool = True):
        self.page_root = Path(page_root)
        self.root = self.page_root / ".cache"
        self.enabled = bool(enabled)

    def _path(self, name: str) -> Path:
        return self.root / name

    def load_registration(self, signature: str) -> RegistrationResult | None:
        if not self.enabled:
            return None
        obj = _load_json(self._path("registration.json"))
        if not obj or obj.get("schema") != CACHE_SCHEMA or obj.get("signature") != signature:
            return None
        p = obj.get("payload") or {}
        try:
            return RegistrationResult(
                matrix=np.asarray(p["matrix"], dtype=np.float64),
                method=str(p["method"]) + "+cache",
                confidence=float(p["confidence"]),
                inlier_ratio=float(p["inlier_ratio"]),
                reprojection_error=float(p["reprojection_error"]),
                spatial_coverage=float(p["spatial_coverage"]),
                num_matches=int(p["num_matches"]),
                source_size=tuple(map(int, p["source_size"])),
                target_size=tuple(map(int, p["target_size"])),
                diagnostics={**dict(p.get("diagnostics") or {}), "cache_hit": True},
            )
        except Exception:
            return None

    def save_registration(self, signature: str, result: RegistrationResult) -> None:
        if not self.enabled:
            return
        _write_json(self._path("registration.json"), {"schema": CACHE_SCHEMA, "signature": signature, "payload": result.to_dict()})

    def load_blocks(self, role: str, signature: str) -> list[TextBlock] | None:
        if not self.enabled:
            return None
        obj = _load_json(self._path(f"ocr_{role}.json"))
        if not obj or obj.get("schema") != CACHE_SCHEMA or obj.get("signature") != signature:
            return None
        try:
            return [TextBlock(**row) for row in obj.get("payload", [])]
        except Exception:
            return None

    def save_blocks(self, role: str, signature: str, blocks: list[TextBlock]) -> None:
        if not self.enabled:
            return
        _write_json(self._path(f"ocr_{role}.json"), {"schema": CACHE_SCHEMA, "signature": signature, "payload": [x.to_dict() for x in blocks]})

    def load_bubbles(self, role: str, signature: str) -> list[BubbleInstance] | None:
        if not self.enabled:
            return None
        meta = _load_json(self._path(f"bubbles_{role}.json"))
        npz_path = self._path(f"bubbles_{role}.npz")
        if not meta or meta.get("schema") != CACHE_SCHEMA or meta.get("signature") != signature or not npz_path.exists():
            return None
        try:
            arrays = np.load(npz_path, allow_pickle=False)
            out: list[BubbleInstance] = []
            for i, row in enumerate(meta.get("payload", [])):
                mask_key, safe_key = f"mask_{i}", f"safe_{i}"
                mask = arrays[mask_key].astype(np.uint8) if mask_key in arrays.files else None
                safe = arrays[safe_key].astype(np.uint8) if safe_key in arrays.files else None
                out.append(BubbleInstance(
                    id=str(row["id"]), polygon=[tuple(map(float, p)) for p in row["polygon"]],
                    confidence=float(row.get("confidence", 1.0)), kind=str(row.get("kind", "speech")),
                    block_ids=list(row.get("block_ids", [])), mask=mask, safe_mask=safe,
                    meta=dict(row.get("meta") or {}),
                ))
            return out
        except Exception:
            return None

    def save_bubbles(self, role: str, signature: str, bubbles: list[BubbleInstance]) -> None:
        if not self.enabled:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        for i, b in enumerate(bubbles):
            if b.mask is not None:
                arrays[f"mask_{i}"] = b.mask.astype(np.uint8)
            if b.safe_mask is not None:
                arrays[f"safe_{i}"] = b.safe_mask.astype(np.uint8)
        np.savez_compressed(self._path(f"bubbles_{role}.npz"), **arrays)
        _write_json(self._path(f"bubbles_{role}.json"), {"schema": CACHE_SCHEMA, "signature": signature, "payload": [b.to_dict() for b in bubbles]})


def registration_stage_signature(pair: PagePair, registration_config: Any) -> str:
    return _json_hash({
        "schema": CACHE_SCHEMA,
        "source": file_signature(pair.source_path),
        "target": file_signature(pair.target_path),
        "registration": registration_config.model_dump(mode="json") if hasattr(registration_config, "model_dump") else registration_config,
    })


def image_stage_signature(path: str | Path, config_value: Any, extra: Any = None) -> str:
    return _json_hash({
        "schema": CACHE_SCHEMA,
        "image": file_signature(path),
        "config": config_value.model_dump(mode="json") if hasattr(config_value, "model_dump") else config_value,
        "extra": extra,
    })


def blocks_signature(blocks: list[TextBlock]) -> str:
    return _json_hash([{"p": b.polygon, "t": b.text, "c": round(float(b.confidence), 4), "k": b.kind} for b in blocks])


def load_completed_page(page_dir: str | Path, pair: PagePair, config: Any, final_path: str | Path) -> PageProject | None:
    page_dir = Path(page_dir)
    project_path = page_dir / "project.json"
    if not project_path.exists() or not Path(final_path).exists():
        return None
    obj = as_dict(_load_json(project_path))
    if not obj:
        return None
    expected = page_job_fingerprint(pair, config)
    meta0 = as_dict(obj.get("meta"))
    if meta0.get("job_fingerprint") != expected:
        return None
    try:
        regp = as_dict(obj.get("registration"))
        reg = RegistrationResult(
            matrix=np.asarray(regp["matrix"], np.float64), method=str(regp["method"]),
            confidence=float(regp["confidence"]), inlier_ratio=float(regp["inlier_ratio"]),
            reprojection_error=float(regp["reprojection_error"]), spatial_coverage=float(regp["spatial_coverage"]),
            num_matches=int(regp["num_matches"]), source_size=tuple(regp["source_size"]), target_size=tuple(regp["target_size"]),
            diagnostics=dict(regp.get("diagnostics") or {}),
        )
        source_blocks = [TextBlock(**x) for x in as_dict_rows(obj.get("source_blocks"))]
        target_blocks = [TextBlock(**x) for x in as_dict_rows(obj.get("target_blocks"))]
        source_bubbles = [BubbleInstance(**{**x, "polygon": [tuple(p) for p in x.get("polygon", [])]}) for x in as_dict_rows(obj.get("source_bubbles"))]
        target_bubbles = [BubbleInstance(**{**x, "polygon": [tuple(p) for p in x.get("polygon", [])]}) for x in as_dict_rows(obj.get("target_bubbles"))]
        source_units = [TextUnit(**x) for x in as_dict_rows(obj.get("source_units"))]
        target_units = [TextUnit(**x) for x in as_dict_rows(obj.get("target_units"))]
        matches = [UnitMatch(**x) for x in as_dict_rows(obj.get("matches"))]
        lettering = [LetteringResult(**x) for x in as_dict_rows(obj.get("lettering"))]
        qa = [QAItem(**x) for x in as_dict_rows(obj.get("qa"))]
        meta = as_dict(obj.get("meta")); meta["batch_resume_hit"] = True
        return PageProject(
            page_id=str(obj["page_id"]), pair=pair, registration=reg,
            source_blocks=source_blocks, target_blocks=target_blocks,
            source_bubbles=source_bubbles, target_bubbles=target_bubbles,
            source_units=source_units, target_units=target_units,
            matches=matches, lettering=lettering, qa=qa,
            artifacts=as_dict(obj.get("artifacts")), meta=meta,
        )
    except Exception:
        return None
