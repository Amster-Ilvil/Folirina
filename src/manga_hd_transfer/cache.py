from __future__ import annotations

"""Deterministic local cache and batch-resume helpers.

No source image pixels are copied into the cache. Keys use a fast sampled file
hash + relevant configuration. Stage payloads are JSON/NPZ and live under the
selected output directory, so deleting the output also removes the cache.
"""

import hashlib
import json
import os
import tempfile
import threading
from collections import OrderedDict
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
from .mode_contracts import mode_scoped_config_payload
from .project_store import page_project_from_dict
from .layout_evidence_models import LayoutEvidence, LayoutEvidenceItem

CACHE_SCHEMA = "mhd-cache-v2"
RESUME_SCHEMA = "folirina-page-resume-v3"

_FILE_SIGNATURE_CACHE: OrderedDict[tuple[str, int, int, int, int], str] = OrderedDict()
_FILE_SIGNATURE_LOCK = threading.RLock()
_FILE_SIGNATURE_CACHE_MAX = 1024


def _json_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def file_signature(path: str | Path, sample_bytes: int = 65536) -> str:
    """Fast deterministic file signature with process-local memoization.

    The same SOURCE/TARGET file is fingerprinted several times per page (resume,
    registration, OCR and bubble stages). Re-reading three sampled chunks on each
    call is unnecessary. Cache by resolved path + size + mtime/ctime; any normal
    file replacement/edit invalidates the memoized entry while repeated stage-key
    construction becomes metadata-only.
    """
    p = Path(path)
    st = p.stat()
    try:
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p.absolute())
    key = (resolved, int(st.st_size), int(st.st_mtime_ns), int(getattr(st, 'st_ctime_ns', 0)), int(sample_bytes))
    with _FILE_SIGNATURE_LOCK:
        cached = _FILE_SIGNATURE_CACHE.get(key)
        if cached is not None:
            _FILE_SIGNATURE_CACHE.move_to_end(key)
            return cached
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
    value = h.hexdigest()[:24]
    with _FILE_SIGNATURE_LOCK:
        _FILE_SIGNATURE_CACHE[key] = value
        _FILE_SIGNATURE_CACHE.move_to_end(key)
        while len(_FILE_SIGNATURE_CACHE) > _FILE_SIGNATURE_CACHE_MAX:
            _FILE_SIGNATURE_CACHE.popitem(last=False)
    return value


def file_content_signature(path: str | Path, chunk_bytes: int = 1024 * 1024) -> str:
    """Exact content signature that deliberately ignores the filename.

    Resume mirrors often store the exact same PNG as ``pages/<id>/final.png`` and
    ``final/<page-name>.png``. ``file_signature`` includes the basename by design
    for cache identity, so it must not be used for mirror equality. New Folirina
    outputs normally hit the cheaper ``os.path.samefile`` path first; this full
    digest is the correctness fallback for legacy/cross-device copies.
    """
    p = Path(path)
    st = p.stat()
    try:
        resolved = str(p.resolve())
    except OSError:
        resolved = str(p.absolute())
    chunk = max(64 * 1024, int(chunk_bytes))
    key = ("content-exact:" + resolved, int(st.st_size), int(st.st_mtime_ns), int(getattr(st, 'st_ctime_ns', 0)), chunk)
    with _FILE_SIGNATURE_LOCK:
        cached = _FILE_SIGNATURE_CACHE.get(key)
        if cached is not None:
            _FILE_SIGNATURE_CACHE.move_to_end(key)
            return cached
    h = hashlib.sha256()
    h.update(str(int(st.st_size)).encode("ascii"))
    with p.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    value = h.hexdigest()[:24]
    with _FILE_SIGNATURE_LOCK:
        _FILE_SIGNATURE_CACHE[key] = value
        _FILE_SIGNATURE_CACHE.move_to_end(key)
        while len(_FILE_SIGNATURE_CACHE) > _FILE_SIGNATURE_CACHE_MAX:
            _FILE_SIGNATURE_CACHE.popitem(last=False)
    return value


def clear_file_signature_cache() -> None:
    """Clear only the cheap in-process file-signature memoization layer."""
    with _FILE_SIGNATURE_LOCK:
        _FILE_SIGNATURE_CACHE.clear()




def _signature_array(signature: str) -> np.ndarray:
    return np.frombuffer(str(signature).encode("ascii", errors="strict"), dtype=np.uint8).copy()


def _array_signature(arrays) -> str:
    if "__cache_signature__" not in arrays.files:
        return ""
    raw = np.asarray(arrays["__cache_signature__"], dtype=np.uint8).tobytes()
    try:
        return raw.decode("ascii")
    except UnicodeDecodeError:
        return ""




def _cache_bundle_matches(arrays, requested_signature: str, meta_path: Path, npz_path: Path) -> bool:
    """Validate JSON/NPZ pairing, including safe migration of v1.3.7 caches.

    New archives carry their own signature. Legacy archives did not, but the old
    writer always wrote NPZ first and JSON metadata second. A legacy bundle is
    accepted only when metadata is at least as new as the NPZ; an interrupted
    overwrite leaves the opposite ordering and is therefore discarded.
    """
    embedded = _array_signature(arrays)
    if embedded:
        return embedded == requested_signature
    try:
        return meta_path.stat().st_mtime_ns >= npz_path.stat().st_mtime_ns
    except OSError:
        return False

def _atomic_savez_compressed(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Atomically replace a recomputable NPZ cache bundle.

    JSON metadata and NPZ arrays are separate files.  A crash while overwriting
    the NPZ in-place can otherwise leave a valid old JSON signature pointing at
    partially/newly written arrays.  Write the complete archive to a sibling
    temporary file first, then rename it into place.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            np.savez_compressed(fh, **arrays)
            fh.flush()
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass

def config_signature(value: Any) -> str:
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    return _json_hash(value)


def page_job_fingerprint(pair: PagePair, config: Any, *, scoped_config_payload: dict[str, Any] | None = None) -> str:
    # Completion/resume identity is mode-scoped. A Reletter font change must not
    # invalidate Direct pages, and a Transparent-Reveal tuning change must not
    # invalidate Mask pages. Low-level stage caches keep their own signatures.
    # A whole-book run has one immutable configuration snapshot. The orchestrator
    # may precompute this fairly expensive model_dump/filter step once and reuse it
    # for every resume admission without changing the fingerprint schema/value.
    cfg = scoped_config_payload if scoped_config_payload is not None else mode_scoped_config_payload(config)
    payload = {
        # Completion identity intentionally has its own schema. Stage cache bundles
        # remain CACHE_SCHEMA=v2 and can still be reused after this safety upgrade.
        "schema": RESUME_SCHEMA,
        "source": file_signature(pair.source_path),
        "target": file_signature(pair.target_path),
        "config": cfg,
    }
    # External OCR is an input artifact, not merely a configuration string.
    # If the user replaces/corrects the JSON/MD in place, completed Reletter /
    # Mask/Hybrid pages must become stale instead of silently reusing old text.
    ocr_cfg = cfg.get("ocr") if isinstance(cfg, dict) else None
    if isinstance(ocr_cfg, dict):
        external_inputs = {}
        for key in ("external_source_ocr_path", "external_target_ocr_path"):
            raw = str(ocr_cfg.get(key) or "").strip()
            if raw:
                try:
                    external_inputs[key] = file_signature(raw)
                    # A selected Markdown can transparently use a structured JSON
                    # companion. Include that real geometry input too, otherwise
                    # replacing the JSON in place would incorrectly keep old
                    # completed-page results.
                    path_obj = Path(raw).expanduser()
                    if path_obj.suffix.lower() in {".md", ".markdown"}:
                        try:
                            from .external_ocr import find_json_companion
                            companion = find_json_companion(path_obj)
                        except Exception:
                            companion = None
                        if companion is not None:
                            external_inputs[key + "_companion_json"] = file_signature(companion)
                except Exception:
                    external_inputs[key] = {"path": raw, "missing": True}
        if external_inputs:
            payload["external_ocr_inputs"] = external_inputs
    return _json_hash(payload)


def _same_paired_files(saved_pair: Any, pair: PagePair) -> bool:
    """Allow resume after re-pairing the same image files.

    Pair indexes are ordering metadata, not page identity. Older project files
    included those indexes in their fingerprint, so compare the persisted paths
    and current file signatures as a compatibility fallback.
    """
    saved = as_dict(saved_pair)
    source_path = str(saved.get("source_path", "") or "")
    target_path = str(saved.get("target_path", "") or "")
    if not source_path or not target_path:
        return False
    try:
        return (
            file_signature(source_path) == file_signature(pair.source_path)
            and file_signature(target_path) == file_signature(pair.target_path)
        )
    except OSError:
        return False


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(value, ensure_ascii=False, indent=2, default=str).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(raw)
            fh.flush()
        os.replace(tmp_name, path)
        tmp_name = ""
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


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
            with np.load(npz_path, allow_pickle=False) as arrays:
                if not _cache_bundle_matches(arrays, signature, self._path(f"bubbles_{role}.json"), npz_path):
                    return None
                out: list[BubbleInstance] = []
                expected_shape: tuple[int, int] | None = None
                for i, row in enumerate(meta.get("payload", [])):
                    mask_key, safe_key = f"mask_{i}", f"safe_{i}"
                    mask = arrays[mask_key].astype(np.uint8, copy=False) if mask_key in arrays.files else None
                    safe = arrays[safe_key].astype(np.uint8, copy=False) if safe_key in arrays.files else None
                    for arr in (mask, safe):
                        if arr is None:
                            continue
                        if arr.ndim != 2:
                            return None
                        if expected_shape is None:
                            expected_shape = tuple(map(int, arr.shape))
                        elif tuple(arr.shape) != expected_shape:
                            return None
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
        arrays: dict[str, np.ndarray] = {"__cache_signature__": _signature_array(signature)}
        for i, b in enumerate(bubbles):
            if b.mask is not None:
                arrays[f"mask_{i}"] = np.asarray(b.mask, dtype=np.uint8)
            if b.safe_mask is not None:
                arrays[f"safe_{i}"] = np.asarray(b.safe_mask, dtype=np.uint8)
        _atomic_savez_compressed(self._path(f"bubbles_{role}.npz"), arrays)
        _write_json(self._path(f"bubbles_{role}.json"), {"schema": CACHE_SCHEMA, "signature": signature, "payload": [b.to_dict() for b in bubbles]})

    def load_layout_evidence(self, role: str, signature: str):
        if not self.enabled:
            return None
        meta = _load_json(self._path(f"layout_{role}.json"))
        npz_path = self._path(f"layout_{role}.npz")
        if not meta or meta.get("schema") != CACHE_SCHEMA or meta.get("signature") != signature or not npz_path.exists():
            return None
        try:
            with np.load(npz_path, allow_pickle=False) as arrays:
                if not _cache_bundle_matches(arrays, signature, self._path(f"layout_{role}.json"), npz_path):
                    return None
                payload = dict(meta.get("payload") or {})
                items = []
                expected_shape = None
                for i, row in enumerate(list(payload.get("items") or [])):
                    mk = f"mask_{i}"
                    if mk not in arrays.files:
                        return None
                    mask = arrays[mk].astype(np.uint8, copy=False)
                    if mask.ndim != 2:
                        return None
                    if expected_shape is None:
                        expected_shape = tuple(int(v) for v in mask.shape)
                    elif tuple(mask.shape) != expected_shape:
                        return None
                    items.append(LayoutEvidenceItem(
                        label=str(row.get("label") or "unknown"),
                        confidence=float(row.get("confidence", 0.0)),
                        polygon=[tuple(map(float, p)) for p in list(row.get("polygon") or [])],
                        mask=mask,
                        box=tuple(int(v) for v in list(row.get("box") or [])),
                        meta=dict(row.get("meta") or {}),
                    ))
                return LayoutEvidence(
                    available=bool(payload.get("available", True)),
                    backend=str(payload.get("backend") or "koharu_layout"),
                    items=items,
                    diagnostics=dict(payload.get("diagnostics") or {}),
                )
        except Exception:
            return None

    def save_layout_evidence(self, role: str, signature: str, evidence) -> None:
        if not self.enabled or evidence is None:
            return
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"__cache_signature__": _signature_array(signature)}
        items_payload = []
        for i, row in enumerate(list(getattr(evidence, "items", []) or [])):
            arrays[f"mask_{i}"] = np.asarray(row.mask, dtype=np.uint8)
            items_payload.append({
                "label": str(row.label),
                "confidence": float(row.confidence),
                "polygon": [[float(x), float(y)] for x, y in list(row.polygon)],
                "box": [int(v) for v in tuple(row.box)],
                "meta": dict(row.meta or {}),
            })
        payload = {
            "available": bool(getattr(evidence, "available", False)),
            "backend": str(getattr(evidence, "backend", "koharu_layout")),
            "items": items_payload,
            "diagnostics": dict(getattr(evidence, "diagnostics", {}) or {}),
        }
        _atomic_savez_compressed(self._path(f"layout_{role}.npz"), arrays)
        _write_json(self._path(f"layout_{role}.json"), {"schema": CACHE_SCHEMA, "signature": signature, "payload": payload})


    @staticmethod
    def _paired_bubble_to_row(b: BubbleInstance, arrays: dict[str, np.ndarray], prefix: str, index: int) -> dict[str, Any]:
        if b.mask is not None:
            arrays[f"{prefix}_mask_{index}"] = np.asarray(b.mask, dtype=np.uint8)
        if b.safe_mask is not None:
            arrays[f"{prefix}_safe_{index}"] = np.asarray(b.safe_mask, dtype=np.uint8)
        return b.to_dict()

    @staticmethod
    def _paired_bubble_from_row(row: dict[str, Any], arrays, prefix: str, index: int) -> BubbleInstance:
        mk, sk = f"{prefix}_mask_{index}", f"{prefix}_safe_{index}"
        mask = arrays[mk].astype(np.uint8, copy=False) if mk in arrays.files else None
        safe = arrays[sk].astype(np.uint8, copy=False) if sk in arrays.files else None
        return BubbleInstance(
            id=str(row["id"]),
            polygon=[tuple(map(float, p)) for p in row.get("polygon", [])],
            confidence=float(row.get("confidence", 1.0)),
            kind=str(row.get("kind", "speech")),
            block_ids=list(row.get("block_ids", [])),
            mask=mask,
            safe_mask=safe,
            meta=dict(row.get("meta") or {}),
        )

    def save_paired_diff(self, signature: str, result: Any) -> None:
        """Persist paired-diff geometry/aligned SOURCE for fast page re-review."""
        if not self.enabled:
            return
        from .paired_diff import PairedDiffResult
        if not isinstance(result, PairedDiffResult):
            return
        self.root.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {
            "__cache_signature__": _signature_array(signature),
            "change_mask": np.asarray(result.change_mask, dtype=np.uint8),
        }
        if result.aligned_source is not None:
            arrays["aligned_source"] = np.asarray(result.aligned_source, dtype=np.uint8)
        src_rows = [self._paired_bubble_to_row(b, arrays, "src", i) for i, b in enumerate(result.source_bubbles)]
        tgt_rows = [self._paired_bubble_to_row(b, arrays, "tgt", i) for i, b in enumerate(result.target_bubbles)]
        meta: dict[str, Any] = {
            "schema": CACHE_SCHEMA,
            "signature": signature,
            "payload": {
                "source_bubbles": src_rows,
                "target_bubbles": tgt_rows,
                "records": [r.to_dict() for r in result.records],
                "threshold": float(result.threshold),
                "noise_floor": float(result.noise_floor),
                "method": str(result.method),
                "safe_to_skip_ocr": bool(result.safe_to_skip_ocr),
                "diagnostics": dict(result.diagnostics or {}),
                "alignment_diagnostics": dict(result.alignment_diagnostics or {}),
                "has_supplemental": bool(result.supplemental is not None),
                "canvas_shape": [int(x) for x in result.change_mask.shape],
            },
        }
        if result.supplemental is not None:
            supp = result.supplemental
            arrays["supp_change_mask"] = np.asarray(supp.change_mask, dtype=np.uint8)
            if supp.aligned_source is not None:
                arrays["supp_aligned_source"] = np.asarray(supp.aligned_source, dtype=np.uint8)
            meta["payload"]["supplemental"] = {
                "source_bubbles": [self._paired_bubble_to_row(b, arrays, "supp_src", i) for i, b in enumerate(supp.source_bubbles)],
                "target_bubbles": [self._paired_bubble_to_row(b, arrays, "supp_tgt", i) for i, b in enumerate(supp.target_bubbles)],
                "records": [r.to_dict() for r in supp.records],
                "threshold": float(supp.threshold),
                "noise_floor": float(supp.noise_floor),
                "method": str(supp.method),
                "safe_to_skip_ocr": bool(supp.safe_to_skip_ocr),
                "diagnostics": dict(supp.diagnostics or {}),
                "alignment_diagnostics": dict(supp.alignment_diagnostics or {}),
            }
        # Paired-diff masks are mostly sparse/flat. Compressed NPZ reduces a
        # typical 50+ MB raw page cache to ~2 MB and is faster overall on rerun
        # because disk I/O dominates the small deflate/inflate cost.
        _atomic_savez_compressed(self._path("paired_diff.npz"), arrays)
        _write_json(self._path("paired_diff.json"), meta)

    def load_paired_diff(self, signature: str):
        if not self.enabled:
            return None
        meta = _load_json(self._path("paired_diff.json"))
        npz_path = self._path("paired_diff.npz")
        if not meta or meta.get("schema") != CACHE_SCHEMA or meta.get("signature") != signature or not npz_path.exists():
            return None
        try:
            from .paired_diff import DiffBubbleRecord, PairedDiffResult
            payload = dict(meta.get("payload") or {})
            with np.load(npz_path, allow_pickle=False) as arrays:
                if not _cache_bundle_matches(arrays, signature, self._path("paired_diff.json"), npz_path):
                    return None
                expected_canvas = tuple(map(int, payload.get("canvas_shape") or ()))

                def make_result(p: dict[str, Any], prefix: str, change_key: str, aligned_key: str):
                    src_rows = list(p.get("source_bubbles") or [])
                    tgt_rows = list(p.get("target_bubbles") or [])
                    source_bubbles = [self._paired_bubble_from_row(r, arrays, f"{prefix}src" if prefix else "src", i) for i, r in enumerate(src_rows)]
                    target_bubbles = [self._paired_bubble_from_row(r, arrays, f"{prefix}tgt" if prefix else "tgt", i) for i, r in enumerate(tgt_rows)]
                    records = []
                    for row in p.get("records") or []:
                        rr = dict(row)
                        if "bbox_target" in rr:
                            rr["bbox_target"] = tuple(map(int, rr["bbox_target"]))
                        records.append(DiffBubbleRecord(**rr))
                    if change_key not in arrays.files:
                        raise ValueError(f"paired-diff cache missing {change_key}")
                    change = arrays[change_key].astype(np.uint8, copy=False)
                    if change.ndim != 2:
                        raise ValueError("paired-diff change mask must be 2D")
                    if not prefix and expected_canvas and tuple(change.shape) != expected_canvas:
                        raise ValueError("paired-diff cache canvas shape mismatch")
                    aligned = arrays[aligned_key].astype(np.uint8, copy=False) if aligned_key in arrays.files else None
                    if aligned is not None and (aligned.ndim != 3 or aligned.shape[:2] != change.shape):
                        raise ValueError("paired-diff aligned source shape mismatch")
                    # TARGET bubbles live in TARGET/change-mask coordinates. SOURCE
                    # bubbles intentionally remain in SOURCE coordinates and may have
                    # a different canvas on cross-resolution / cross-rendition pairs.
                    # Reject malformed arrays, but never force SOURCE masks to the
                    # TARGET canvas (that silently disabled the real-page cache).
                    source_shape = None
                    for bubble in source_bubbles:
                        for arr in (bubble.mask, bubble.safe_mask):
                            if arr is None:
                                continue
                            if arr.ndim != 2:
                                raise ValueError("paired-diff source bubble mask must be 2D")
                            if source_shape is None:
                                source_shape = tuple(map(int, arr.shape))
                            elif tuple(arr.shape) != source_shape:
                                raise ValueError("paired-diff source bubble masks disagree on canvas")
                    for bubble in target_bubbles:
                        for arr in (bubble.mask, bubble.safe_mask):
                            if arr is not None and (arr.ndim != 2 or arr.shape != change.shape):
                                raise ValueError("paired-diff target bubble mask shape mismatch")
                    return PairedDiffResult(
                        source_bubbles, target_bubbles, change, records,
                        float(p.get("threshold", 0.0)), float(p.get("noise_floor", 0.0)),
                        method=str(p.get("method", "raw_diff")),
                        safe_to_skip_ocr=bool(p.get("safe_to_skip_ocr", True)),
                        diagnostics={**dict(p.get("diagnostics") or {}), "cache_hit": True},
                        aligned_source=aligned,
                        alignment_diagnostics={**dict(p.get("alignment_diagnostics") or {}), "cache_hit": True},
                    )

                result = make_result(payload, "", "change_mask", "aligned_source")
                supp_payload = payload.get("supplemental")
                if isinstance(supp_payload, dict) and "supp_change_mask" in arrays.files:
                    result.supplemental = make_result(supp_payload, "supp_", "supp_change_mask", "supp_aligned_source")
            return result
        except Exception:
            return None


def registration_stage_signature(pair: PagePair, registration_config: Any) -> str:
    return _json_hash({
        "schema": CACHE_SCHEMA,
        "source": file_signature(pair.source_path),
        "target": file_signature(pair.target_path),
        "registration": registration_config.model_dump(mode="json") if hasattr(registration_config, "model_dump") else registration_config,
    })


def paired_diff_stage_signature(pair: PagePair, registration: RegistrationResult, mask_replace_config: Any) -> str:
    """Fingerprint the expensive paired-difference geometry stage only.

    Runtime/export/resume settings are intentionally absent. Registration is part
    of the key because paired-diff operates in registered TARGET coordinates.
    """
    reg_matrix = np.asarray(registration.matrix, dtype=np.float64)
    raw_cfg = mask_replace_config.model_dump(mode="json") if hasattr(mask_replace_config, "model_dump") else dict(mask_replace_config or {})
    # Only geometry/detection knobs used by paired_diff.py / paired_diff_v08.py
    # enter this stage key. SR, inpainting, border write policy and review knobs
    # affect later stages and must not force the expensive detector to rerun.
    paired_cfg = {
        str(k): v for k, v in dict(raw_cfg).items()
        if str(k).startswith("paired_diff_") or str(k).startswith("photo_pair_")
    }
    return _json_hash({
        "schema": CACHE_SCHEMA,
        "source": file_signature(pair.source_path),
        "target": file_signature(pair.target_path),
        "registration": {
            "matrix": np.round(reg_matrix, 10).tolist(),
            "method": str(registration.method).replace("+cache", ""),
            "confidence": round(float(registration.confidence), 8),
            "source_size": list(registration.source_size),
            "target_size": list(registration.target_size),
        },
        "paired_diff_config": paired_cfg,
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


def _same_resume_input(saved_path: Any, current_path: str | Path, *, project_mtime_ns: int) -> bool:
    """Conservative same-input check for GUI continuation fallback.

    Older successful pages do not persist their raw file signatures separately
    from the job fingerprint.  For continuation we therefore require the exact
    same input file (samefile/resolved path) and reject it if that file was
    modified after the successful project.json was committed.
    """
    raw = str(saved_path or "").strip()
    if not raw:
        return False
    try:
        saved = Path(raw).expanduser()
        current = Path(current_path).expanduser()
        if saved.exists() and current.exists():
            try:
                if not os.path.samefile(saved, current):
                    return False
            except OSError:
                if saved.resolve(strict=False) != current.resolve(strict=False):
                    return False
            # If the input is newer than the durable completed-page transaction,
            # it may have been replaced in place. Fail closed and reprocess.
            if int(current.stat().st_mtime_ns) > int(project_mtime_ns):
                return False
            return True
        return saved.resolve(strict=False) == current.resolve(strict=False)
    except (OSError, TypeError, ValueError):
        return False


def _continue_identity_allows_completed(obj: dict[str, Any], project_path: Path, pair: PagePair, requested_mode: str) -> bool:
    meta0 = as_dict(obj.get("meta"))
    saved_mode = str(meta0.get("transfer_mode") or "").strip().lower()
    # Never continue across a renderer/mode boundary.
    if saved_mode and requested_mode and saved_mode != requested_mode:
        return False
    pair0 = as_dict(obj.get("pair"))
    try:
        project_mtime_ns = int(project_path.stat().st_mtime_ns)
    except OSError:
        return False
    return (
        _same_resume_input(pair0.get("source_path"), pair.source_path, project_mtime_ns=project_mtime_ns)
        and _same_resume_input(pair0.get("target_path"), pair.target_path, project_mtime_ns=project_mtime_ns)
    )


def load_completed_page(page_dir: str | Path, pair: PagePair, config: Any, final_path: str | Path, *, scoped_config_payload: dict[str, Any] | None = None, allow_compatible_identity: bool = False) -> PageProject | None:
    """Load a completed page only when its *mode-scoped* execution identity matches.

    v2.3.19 deliberately fails closed. Older code treated "same SOURCE/TARGET files"
    as sufficient compatibility after any job-fingerprint mismatch. That let a page
    completed in another mode, or with stale OCR/font/detector settings, be resumed
    as if it belonged to the newly selected mode. It could also publish the old
    page-local final into the book output *before* validating that fingerprint.

    Stage caches are still independently reusable; only the already-published
    completed-page shortcut is strict.
    """
    page_dir = Path(page_dir)
    project_path = page_dir / "project.json"
    final_path = Path(final_path)
    if not project_path.exists():
        return None
    obj = as_dict(_load_json(project_path))
    if not obj:
        return None

    expected = page_job_fingerprint(pair, config, scoped_config_payload=scoped_config_payload)
    meta0 = as_dict(obj.get("meta"))
    requested_mode = str(getattr(getattr(config, "transfer", None), "mode", "") or "").strip().lower()
    saved_mode = str(meta0.get("transfer_mode") or "").strip().lower()

    # Never resume across an explicit mode boundary. Legacy pages with no mode or
    # with an old resume fingerprint simply rebuild once, after which they carry
    # the v3 resume contract.
    if saved_mode and requested_mode and saved_mode != requested_mode:
        return None
    fingerprint_match = str(meta0.get("job_fingerprint") or "") == expected
    compatible_continue = False
    if not fingerprint_match:
        if not allow_compatible_identity:
            return None
        compatible_continue = _continue_identity_allows_completed(obj, project_path, pair, requested_mode)
        if not compatible_continue:
            return None

    # Only after identity validation may the book-level final be synchronized.
    # Prefer the page-local reviewed/automatic result over persisted absolute
    # artifact paths: an older project may point ``artifacts.final`` at the book
    # output itself, which is exactly the file we are trying to verify/repair.
    artifacts = as_dict(obj.get("artifacts"))
    candidates = [page_dir / "final_reviewed.png", page_dir / "final_auto.png", page_dir / "final.png"]
    for key in ("final_reviewed", "final"):
        val = str(artifacts.get(key, "") or "").strip()
        if val:
            candidates.append(Path(val))
    chosen = next((c for c in candidates if c.exists() and c.is_file() and c != final_path), None)
    if chosen is None and final_path.exists() and final_path.is_file():
        chosen = final_path
    if chosen is None:
        return None
    need_sync = not final_path.exists()
    if not need_sync and chosen != final_path:
        try:
            # Hard-linked mirrors are already identical even though their names
            # differ; otherwise compare bytes without basename-sensitive cache IDs.
            if os.path.samefile(chosen, final_path):
                need_sync = False
            else:
                need_sync = file_content_signature(chosen) != file_content_signature(final_path)
        except OSError:
            need_sync = True
    if need_sync:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from .result_state import atomic_copy_file
            atomic_copy_file(chosen, final_path)
        except OSError:
            return None
    # ``final`` remains the page-local artifact; the book mirror has its own key.
    previous_book_final = str(artifacts.get("book_final", "") or "")
    artifacts["book_final"] = str(final_path)
    obj["artifacts"] = artifacts
    # Normal resumes used to atomically rewrite project.json on every page even
    # when this persisted mirror path was already correct. Only legacy/repaired
    # projects need that write; ordinary resume admission is now read-mostly.
    if previous_book_final != str(final_path):
        try:
            _write_json(project_path, obj)
        except Exception:
            pass

    try:
        page = page_project_from_dict(obj, pair_override=pair, resume_hit=True)
        if compatible_continue:
            page.meta["batch_resume_compatible_hit"] = True
            page.meta["batch_resume_policy"] = "continue"
            page.meta["batch_resume_identity_reason"] = "same_inputs_same_mode_existing_success"
        return page
    except Exception:
        return None

