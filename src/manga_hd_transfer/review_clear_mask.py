from __future__ import annotations

"""Safe review-facing projection of broad automatic clear masks.

Automatic renderers sometimes publish a container/write-sized clear artifact.
That artifact is useful for renderer diagnostics, but it is too broad to expose
as an editable "erase Japanese" mask: a text box can overlap a face or other
artwork.  Review operations therefore project automatic masks down to compact
TARGET lettering.  Koharu/Layout text polygons act only as semantic seeds; they
are expanded through nearby compact dark-text groups so incomplete segmentation
still clears the full Japanese line while unrelated artwork stays protected.

Manual reviewer masks are never passed through this projection.  Human edits are
authority and are preserved exactly.
"""

from pathlib import Path
from typing import Any
import hashlib

import cv2
import numpy as np

from .io_utils import load_json, save_json, write_image, read_image
from .schema_compat import as_dict, as_dict_rows
from .text_only_transfer import _adaptive_ink_threshold, _compact_text_clusters, _gray

_CACHE_MASK = "review_safe_auto_clear_mask.png"
_CACHE_META = "review_safe_auto_clear_mask.json"


def _review_component_text_selector(
    ink: np.ndarray,
    unique: np.ndarray,
    region: np.ndarray,
    *,
    min_area: int = 2,
    min_unique_pixels: int = 2,
    min_unique_ratio: float = 0.055,
    max_component_fraction: float = 0.10,
) -> np.ndarray:
    """Vectorized, bit-equivalent component selector for review projection.

    The shared renderer helper historically built ``labels == i`` over the whole
    page once per connected component. A manga page can contain thousands of tiny
    components, turning a first review-mask projection into several seconds of
    CPU time. Review only needs the same acceptance rule, so count unique pixels
    per label once with ``bincount`` and materialize the accepted labels once.
    """
    binary=((ink>0)&(region>0)).astype(np.uint8)
    uniq=(unique>0)&(region>0)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1:
        return np.zeros_like(binary)
    flat_labels=labels.reshape(-1)
    uniq_flat=uniq.reshape(-1)
    unique_counts=np.bincount(flat_labels[uniq_flat],minlength=count) if np.any(uniq_flat) else np.zeros(count,dtype=np.int64)
    region_area=max(1,int(np.count_nonzero(region)))
    h,w=binary.shape
    keep=np.zeros(count,dtype=bool)
    for i in range(1,count):
        _x,_y,bw,bh,area=[int(v) for v in stats[i]]
        if area<int(min_area):
            continue
        up=int(unique_counts[i])
        ratio=float(up/max(1,area)); area_fraction=float(area/region_area)
        span_x=float(bw/max(1,w)); span_y=float(bh/max(1,h)); fill=float(area/max(1,bw*bh))
        if area_fraction>float(max_component_fraction) and (ratio<0.42 or fill<0.16):
            continue
        if (span_x>0.82 or span_y>0.82) and fill<0.16 and ratio<0.55:
            continue
        needed=max(int(min_unique_pixels),int(round(area*float(min_unique_ratio))))
        if up<needed:
            continue
        keep[i]=True
    return keep[labels].astype(np.uint8)*255


def _review_target_text_mask_in_container(target: np.ndarray, region_mask: np.ndarray) -> np.ndarray:
    use=region_mask>0
    if not np.any(use):
        return np.zeros(region_mask.shape,np.uint8)
    tg=_gray(target); tth=_adaptive_ink_threshold(tg,use)
    t_all=((tg<tth)&use).astype(np.uint8)*255
    out=_review_component_text_selector(
        t_all,t_all,region_mask,min_unique_pixels=1,min_unique_ratio=0.0,max_component_fraction=0.07,
    )
    return _compact_text_clusters(out,region_mask)


def _stat_signature(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    st = path.stat()
    return {"exists": True, "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _layout_text_mask(page_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    path = page_dir / ".cache" / "layout_target.json"
    out = np.zeros(shape, np.uint8)
    if not path.exists():
        return out, {"available": False, "reason": "layout_target_missing", "items": 0}
    try:
        data = load_json(path)
        payload = as_dict(data.get("payload"))
        items = as_dict_rows(payload.get("items"))
    except Exception as exc:
        return out, {"available": False, "reason": f"layout_target_invalid:{type(exc).__name__}", "items": 0}

    used = 0
    for row in items:
        if str(row.get("label") or row.get("kind") or "").strip().lower() != "text":
            continue
        pts = np.asarray(row.get("polygon") or [], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
            continue
        pts[:, 0] = np.clip(pts[:, 0], 0, max(0, shape[1] - 1))
        pts[:, 1] = np.clip(pts[:, 1], 0, max(0, shape[0] - 1))
        cv2.fillPoly(out, [np.rint(pts).astype(np.int32)], 255)
        used += 1
    return out, {
        "available": bool(used and cv2.countNonZero(out)),
        "reason": "ok" if used else "no_text_items",
        "items": int(used),
        "pixels": int(cv2.countNonZero(out)),
    }


def _semantic_seeded_text_groups(
    compact_text: np.ndarray,
    semantic_seed: np.ndarray,
    *,
    join_radius: int = 10,
    seed_halo: int = 4,
) -> np.ndarray:
    text = (np.asarray(compact_text, dtype=np.uint8) > 0).astype(np.uint8) * 255
    seed = (np.asarray(semantic_seed, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(text) == 0 or cv2.countNonZero(seed) == 0:
        return np.zeros_like(text)
    jr = max(2, min(18, int(join_radius)))
    grouped = cv2.dilate(
        text,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (jr * 2 + 1, jr * 2 + 1)),
        iterations=1,
    )
    _count, labels = cv2.connectedComponents((grouped > 0).astype(np.uint8), 8)
    sh = max(0, min(10, int(seed_halo)))
    if sh:
        seed = cv2.dilate(
            seed,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sh * 2 + 1, sh * 2 + 1)),
            iterations=1,
        )
    touched = np.unique(labels[(seed > 0) & (labels > 0)])
    if touched.size == 0:
        return np.zeros_like(text)
    return np.where((text > 0) & np.isin(labels, touched), 255, 0).astype(np.uint8)


def project_automatic_clear_mask(
    page_dir: str | Path,
    raw_mask: np.ndarray,
    *,
    target: np.ndarray | None = None,
    source_name: str = "automatic",
    use_cache: bool = True,
) -> tuple[np.ndarray, dict]:
    """Project a renderer-owned automatic mask to safe TARGET text evidence.

    If semantic text layout is unavailable, return the original mask unchanged;
    this preserves compatibility for old workspaces.  When layout evidence is
    available, semantic seeds select complete nearby compact Japanese text groups.
    The result is additionally clipped to the original automatic authority.
    """
    page_dir = Path(page_dir)
    raw = (np.asarray(raw_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255
    if cv2.countNonZero(raw) == 0:
        return raw, {"projected": False, "reason": "empty_raw", "raw_pixels": 0, "safe_pixels": 0}
    target_path = page_dir / "target_original.png"
    layout_path = page_dir / ".cache" / "layout_target.json"
    cache_path = page_dir / _CACHE_MASK
    meta_path = page_dir / _CACHE_META
    signature = {
        "schema": "folirina.review_safe_auto_clear.v1",
        "source_name": str(source_name),
        "raw_pixels": int(cv2.countNonZero(raw)),
        # Pixel count alone is not a cache identity: an edited mask can move the
        # same number of pixels. A short BLAKE2 digest keeps stale projections
        # from surviving such edits without introducing a heavy dependency.
        "raw_digest": hashlib.blake2b(raw.tobytes(), digest_size=8).hexdigest(),
        "target": _stat_signature(target_path),
        "layout": _stat_signature(layout_path),
    }
    if use_cache and cache_path.exists() and meta_path.exists():
        try:
            meta = load_json(meta_path)
            cached = cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
            if (
                as_dict(meta.get("signature")) == signature
                and cached is not None and cached.shape == raw.shape
            ):
                return (cached > 0).astype(np.uint8) * 255, as_dict(meta.get("diagnostics"))
        except Exception:
            pass

    semantic, semantic_diag = _layout_text_mask(page_dir, raw.shape)
    if not semantic_diag.get("available"):
        diag = {
            "projected": False,
            "reason": str(semantic_diag.get("reason") or "semantic_unavailable"),
            "raw_pixels": int(cv2.countNonZero(raw)),
            "safe_pixels": int(cv2.countNonZero(raw)),
            "semantic": semantic_diag,
        }
        return raw, diag

    if target is None:
        target = read_image(target_path)
    if target.shape[:2] != raw.shape:
        raise ValueError("review clear mask / TARGET size mismatch")

    # First isolate compact dark glyph candidates inside renderer authority.
    compact = _review_target_text_mask_in_container(target, raw)
    selected = _semantic_seeded_text_groups(compact, semantic, join_radius=10, seed_halo=4)
    if cv2.countNonZero(selected) == 0:
        # Semantic segmentation is safer than restoring the broad artwork-sized
        # mask when it exists but grouping failed.
        selected = cv2.bitwise_and(raw, semantic)
        fallback = "semantic_intersection"
    else:
        fallback = "none"

    # One-pixel fringe removes antialiasing without growing into unrelated art.
    selected = cv2.dilate(selected, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
    safe = cv2.bitwise_and(selected, raw)
    diag = {
        "projected": True,
        "reason": "layout_seeded_compact_target_text",
        "source_name": str(source_name),
        "raw_pixels": int(cv2.countNonZero(raw)),
        "compact_pixels": int(cv2.countNonZero(compact)),
        "safe_pixels": int(cv2.countNonZero(safe)),
        "removed_from_review_mask_pixels": int(cv2.countNonZero(raw) - cv2.countNonZero(safe)),
        "semantic": semantic_diag,
        "join_radius": 10,
        "seed_halo": 4,
        "fallback": fallback,
        "artwork_policy": "layout_text_seed_required_for_automatic_review_mask",
    }
    if use_cache:
        write_image(cache_path, safe)
        save_json(meta_path, {"schema": "folirina.review_safe_auto_clear_cache.v1", "signature": signature, "diagnostics": diag})
    return safe, diag


def safe_automatic_clear_seed(page_dir: str | Path, shape: tuple[int, int]) -> tuple[np.ndarray | None, dict]:
    """Load the first automatic clear artifact and safely project it for review."""
    page_dir = Path(page_dir)
    for name in ("target_clear_mask.png", "clear_mask.png"):
        path = page_dir / name
        if not path.exists():
            continue
        raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if raw is None or raw.shape != shape:
            continue
        safe, diag = project_automatic_clear_mask(page_dir, raw, source_name=name)
        return safe, {**diag, "source": name}
    return None, {"projected": False, "reason": "automatic_clear_missing"}


__all__ = [
    "project_automatic_clear_mask", "safe_automatic_clear_seed",
]
