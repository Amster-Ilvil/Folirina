from __future__ import annotations

"""Shared Koharu Layout evidence adapters for visual transfer modes.

The visual modes in Folirina intentionally remain OCR-independent: Direct,
Precise Mask, Transparent Bubble Reveal, and Experimental Aligned Reveal must
not depend on text recognition just to locate likely text containers.

Koharu Layout is therefore treated here as *layout evidence only*:
- ``bubble`` items can become BubbleInstance hints;
- ``text`` / ``onomatopoeia`` items can become TextBlock seeds;
- ``panel`` items can be consumed as safety context.

Nothing in this module performs OCR or relettering.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from .cache import PageStageCache, image_stage_signature
from .config import BubbleConfig
from .geometry import polygon_bbox, polygon_centroid, rasterize_polygon
from .model_downloads import discovered_model_path
from .models import BubbleInstance, TextBlock


logger = logging.getLogger(__name__)


_LABEL_ALIASES = {
    "onomatopoeia": "sfx",
    "sound_effect": "sfx",
    "sound-effect": "sfx",
    "speech_bubble": "bubble",
}


def _normalize_label(label: str) -> str:
    key = str(label or "").strip().lower().replace("-", "_")
    return _LABEL_ALIASES.get(key, key)


@dataclass(slots=True)
class LayoutEvidenceItem:
    label: str
    confidence: float
    polygon: list[tuple[float, float]]
    mask: np.ndarray
    box: tuple[int, int, int, int]
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LayoutAuthorityDecision:
    state: str  # ALLOW | PROTECT | UNKNOWN
    reason: str
    bubble_overlap: float = 0.0
    text_overlap: float = 0.0
    sfx_overlap: float = 0.0
    panel_overlap: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "reason": self.reason,
            "bubble_overlap": round(float(self.bubble_overlap), 4),
            "text_overlap": round(float(self.text_overlap), 4),
            "sfx_overlap": round(float(self.sfx_overlap), 4),
            "panel_overlap": round(float(self.panel_overlap), 4),
        }


@dataclass(slots=True)
class LayoutEvidence:
    available: bool
    backend: str
    items: list[LayoutEvidenceItem] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def by_label(self, *labels: str) -> list[LayoutEvidenceItem]:
        wanted = {_normalize_label(label) for label in labels}
        return [row for row in self.items if row.label in wanted]

    def combined_mask(self, labels: Iterable[str], *, dilate_px: int = 0) -> np.ndarray:
        shape = self.diagnostics.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("LayoutEvidence diagnostics missing shape")
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros((h, w), np.uint8)
        wanted = {_normalize_label(label) for label in labels}
        for row in self.items:
            if row.label not in wanted:
                continue
            if row.mask.shape != out.shape:
                mask = cv2.resize(row.mask, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                mask = row.mask
            out = np.maximum(out, mask)
        radius = max(0, int(dilate_px))
        if radius > 0 and cv2.countNonZero(out) > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            out = cv2.dilate(out, kernel)
        return out

    def text_blocks(
        self, *, include_sfx: bool = True, backend_name: str = "koharu_layout",
        source_only: bool = False, target_only: bool = False,
    ) -> list[TextBlock]:
        out: list[TextBlock] = []
        for idx, row in enumerate(self.items):
            if row.label == "text":
                kind = "speech"
            elif include_sfx and row.label == "sfx":
                kind = "sfx"
            else:
                continue
            meta = dict(row.meta)
            meta.update({
                "backend": backend_name,
                "layout_label": row.label,
                "layout_evidence": True,
                "text_seed": True,
            })
            if source_only:
                meta["source_only"] = True
                meta.pop("target_only", None)
            if target_only:
                meta["target_only"] = True
                meta.pop("source_only", None)
            out.append(TextBlock(
                id=f"koharu-text-{idx:04d}",
                polygon=list(row.polygon),
                text="",
                confidence=float(row.confidence),
                kind=kind,
                reading_order=idx,
                meta=meta,
            ))
        return out

    def bubble_instances(self, *, backend_name: str = "koharu_layout", target_only: bool = False, source_only: bool = False) -> list[BubbleInstance]:
        out: list[BubbleInstance] = []
        for idx, row in enumerate(self.items):
            if row.label != "bubble":
                continue
            mask = row.mask.copy()
            safe = mask.copy()
            meta = dict(row.meta)
            meta.update({
                "backend": backend_name,
                "layout_label": row.label,
                "layout_evidence": True,
            })
            if target_only:
                meta["target_only"] = True
                meta.pop("source_only", None)
            if source_only:
                meta["source_only"] = True
                meta.pop("target_only", None)
            out.append(BubbleInstance(
                id=f"koharu-bubble-{idx:04d}",
                polygon=list(row.polygon),
                confidence=float(row.confidence),
                kind="speech",
                block_ids=[],
                mask=mask,
                safe_mask=safe,
                meta=meta,
            ))
        return out

    def authority_map(self, *, allow_dilate_px: int = 0) -> np.ndarray:
        """Return the page semantic authority map (0 UNKNOWN, 1 ALLOW, 2 PROTECT).

        ALLOW has priority over panel protection.  The map is intentionally a
        first-layer semantic prior, not a replacement for candidate-specific
        geometry checks: fallback detectors may operate in UNKNOWN but must not
        destructively override PROTECT.
        """
        shape = self.diagnostics.get("shape")
        if not isinstance(shape, (list, tuple)) or len(shape) != 2:
            raise ValueError("LayoutEvidence diagnostics missing shape")
        h, w = int(shape[0]), int(shape[1])
        out = np.zeros((h, w), np.uint8)
        if not self.available:
            return out
        allow = self.combined_mask(("bubble", "text", "sfx"), dilate_px=allow_dilate_px)
        panel = self.combined_mask(("panel",), dilate_px=0)
        out[panel > 0] = 2
        out[allow > 0] = 1
        return out


_LAYOUT_EVIDENCE_MEMO: OrderedDict[str, LayoutEvidence] = OrderedDict()
_LAYOUT_EVIDENCE_MEMO_MAX = 128


def _array_signature(image: np.ndarray) -> str:
    h, w = image.shape[:2]
    sample = np.ascontiguousarray(image[:: max(1, h // 16), :: max(1, w // 16)])
    token = cv2.imencode('.png', sample)[1].tobytes() if sample.size else b''
    import hashlib
    return hashlib.sha256(token + f'|{image.shape}|{image.dtype}'.encode('utf-8')).hexdigest()[:24]


def _canonical_cache_role(role: str) -> str:
    key = str(role or "page").strip().lower()
    if "source" in key or key.startswith("src"):
        return "source"
    if "target" in key or key.startswith("tgt") or key.startswith("dst"):
        return "target"
    return "page"


def _model_revision_token(cfg: BubbleConfig) -> str:
    try:
        resolved = discovered_model_path("koharu_layout", getattr(cfg, "koharu_layout_model_path", None))
    except Exception:
        resolved = None
    if resolved is None:
        return "missing"
    p = Path(resolved).expanduser()
    parts = [str(p)]
    for name in ("inference_config.json", "load_model.py", "model.safetensors"):
        q = p / name
        try:
            st = q.stat()
            parts.append(f"{name}:{st.st_size}:{st.st_mtime_ns}")
        except OSError:
            parts.append(f"{name}:missing")
    import hashlib
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _layout_signature(
    image: np.ndarray,
    bubble_cfg: BubbleConfig | None,
    *,
    role: str,
    image_path: str | Path | None = None,
) -> str:
    cfg = bubble_cfg or BubbleConfig()
    payload = {
        'backend': 'koharu_layout',
        'page_role': _canonical_cache_role(role),
        'thresholds': {
            'text': float(getattr(cfg, 'koharu_layout_text_threshold', 0.25)),
            'sfx': float(getattr(cfg, 'koharu_layout_sfx_threshold', 0.20)),
            'bubble': float(getattr(cfg, 'koharu_layout_bubble_threshold', 0.50)),
            'panel': float(getattr(cfg, 'koharu_layout_panel_threshold', 0.50)),
            'shape': int(getattr(cfg, 'koharu_layout_shape', 1152)),
            'postprocess_max_side': int(getattr(cfg, 'koharu_layout_postprocess_max_side', 1152)),
            'postprocess_downscale_trigger_side': int(getattr(cfg, 'koharu_layout_postprocess_downscale_trigger_side', 2048)),
            'device': str(getattr(cfg, 'device', 'auto')),
            'model_path': str(getattr(cfg, 'koharu_layout_model_path', '') or ''),
            'model_revision': _model_revision_token(cfg),
        },
    }
    if image_path is not None:
        try:
            if Path(image_path).exists():
                return image_stage_signature(image_path, payload)
        except OSError:
            pass
    cache_role = _canonical_cache_role(role)
    return f"array:{cache_role}:{_array_signature(image)}:{payload['thresholds']['shape']}:{payload['thresholds']['postprocess_max_side']}:{payload['thresholds']['postprocess_downscale_trigger_side']}:{payload['thresholds']['text']}:{payload['thresholds']['bubble']}"


def collect_koharu_layout_evidence_cached(
    image: np.ndarray,
    bubble_cfg: BubbleConfig | None = None,
    *,
    role: str = 'page',
    image_path: str | Path | None = None,
    cache: PageStageCache | None = None,
    cache_enabled: bool = True,
    stats: dict[str, Any] | None = None,
    allow_missing: bool = True,
) -> LayoutEvidence:
    cache_role = _canonical_cache_role(role)
    signature = _layout_signature(image, bubble_cfg, role=cache_role, image_path=image_path)
    if cache is not None and cache_enabled and image_path is not None and hasattr(cache, "load_layout_evidence"):
        cached = cache.load_layout_evidence(cache_role, signature)
        if cached is not None:
            if stats is not None:
                stats[f'layout_{role}'] = 'hit_disk'
                stats[f'layout_{cache_role}_cache'] = 'hit_disk'
            return cached
    memo = _LAYOUT_EVIDENCE_MEMO.get(signature)
    if memo is not None:
        _LAYOUT_EVIDENCE_MEMO.move_to_end(signature)
        if stats is not None:
            token = 'hit_memory' if memo.available else 'hit_memory_unavailable'
            stats[f'layout_{role}'] = token
            stats[f'layout_{cache_role}_cache'] = token
            if not memo.available:
                reason = str((memo.diagnostics or {}).get("reason") or "unavailable")
                error = str((memo.diagnostics or {}).get("error") or "")
                stats[f'layout_{role}_reason'] = reason
                if error:
                    stats[f'layout_{role}_error'] = error[:800]
        return memo

    evidence = collect_koharu_layout_evidence(image, bubble_cfg, role=role, allow_missing=allow_missing)
    # Successful evidence is always memoized. Deterministic *runtime* failures
    # (for example RF-DETR postprocess OOM) are also memoized for this process so
    # bubble/Direct/transfer/QA consumers do not repeat the same multi-second crash
    # on one page. Missing-model state remains non-sticky because the user may
    # install/download a model while the GUI process is still open.
    memo_reason = str((evidence.diagnostics or {}).get("reason") or "")
    if evidence.available or memo_reason == "runtime_error":
        _LAYOUT_EVIDENCE_MEMO[signature] = evidence
        _LAYOUT_EVIDENCE_MEMO.move_to_end(signature)
        while len(_LAYOUT_EVIDENCE_MEMO) > _LAYOUT_EVIDENCE_MEMO_MAX:
            _LAYOUT_EVIDENCE_MEMO.popitem(last=False)
    if cache is not None and cache_enabled and image_path is not None and evidence.available and hasattr(cache, "save_layout_evidence"):
        cache.save_layout_evidence(cache_role, signature, evidence)
    if stats is not None:
        stats[f'layout_{role}'] = 'miss_built' if evidence.available else 'miss_unavailable'
        stats[f'layout_{cache_role}_cache'] = 'miss_built' if evidence.available else 'miss_unavailable'
        if not evidence.available:
            reason = str((evidence.diagnostics or {}).get("reason") or "unavailable")
            error = str((evidence.diagnostics or {}).get("error") or "")
            stats[f'layout_{role}_reason'] = reason
            if error:
                stats[f'layout_{role}_error'] = error[:800]
    return evidence

def prepare_page_layout_evidence(
    mode: str,
    source: np.ndarray,
    target: np.ndarray,
    *,
    source_path: str | Path | None,
    target_path: str | Path | None,
    bubble_cfg: BubbleConfig | None,
    cache: PageStageCache | None = None,
    cache_enabled: bool = True,
    stats: dict[str, Any] | None = None,
) -> dict[str, LayoutEvidence]:
    """Warm the shared per-page Layout Evidence Cache before mode logic.

    Koharu Layout is a geometry detector, not OCR.  Every transfer mode may use
    it, while each mode remains free to consume only the SOURCE/TARGET evidence
    allowed by its own contract.  Missing models are a safe fallback, not an
    implicit reason to instantiate OCR.
    """
    cfg = bubble_cfg or BubbleConfig()
    key = str(mode or "auto").strip().lower()
    # v2.0.90 authority-first contract: every automatic transfer mode warms
    # SOURCE + TARGET once, before any mode-specific detector.  Individual
    # modes may consume only one side, but later fallbacks must see the same
    # first semantic authority instead of silently instantiating their own
    # independent detector ordering.
    need_source = True
    need_target = True
    out: dict[str, LayoutEvidence] = {}
    if need_source:
        out["source"] = collect_koharu_layout_evidence_cached(
            source, cfg, role="source", image_path=source_path, cache=cache,
            cache_enabled=cache_enabled, stats=stats, allow_missing=True,
        )
    if need_target:
        out["target"] = collect_koharu_layout_evidence_cached(
            target, cfg, role="target", image_path=target_path, cache=cache,
            cache_enabled=cache_enabled, stats=stats, allow_missing=True,
        )
    if stats is not None:
        available = [role for role, ev in out.items() if ev.available]
        unavailable = [role for role, ev in out.items() if not ev.available]
        stats["layout_prefetch"] = "koharu_layout" if available else "koharu_unavailable"
        stats["layout_prefetch_available"] = ",".join(available)
        stats["layout_prefetch_unavailable"] = ",".join(unavailable)
        stats["layout_authority"] = "koharu_first_allow_protect_unknown" if available else "fallback_unknown_only"
        stats["layout_authority_mode"] = key
    return out


def _candidate_mask(candidate: Any, shape: tuple[int, int]) -> np.ndarray:
    mask = getattr(candidate, "mask", None)
    if isinstance(mask, np.ndarray) and tuple(mask.shape[:2]) == tuple(shape):
        return (mask > 0).astype(np.uint8)
    polygon = getattr(candidate, "polygon", None) or []
    if polygon:
        return (rasterize_polygon(polygon, shape) > 0).astype(np.uint8)
    return np.zeros(shape, np.uint8)


def _max_item_overlap(mask: np.ndarray, items: Iterable[LayoutEvidenceItem]) -> float:
    area = int(np.count_nonzero(mask))
    if area <= 0:
        return 0.0
    best = 0.0
    mb = mask > 0
    for item in items:
        im = item.mask > 0
        if im.shape != mb.shape:
            continue
        ia = int(np.count_nonzero(im))
        if ia <= 0:
            continue
        inter = int(np.count_nonzero(mb & im))
        best = max(best, float(inter / max(1, min(area, ia))))
    return best


def classify_layout_authority(
    evidence: LayoutEvidence | None,
    candidate: Any,
    shape: tuple[int, int],
    *,
    region_kind: str = "bubble",
    cfg: Any | None = None,
) -> LayoutAuthorityDecision:
    """Classify one proposed writable region against Koharu's first authority.

    ``ALLOW`` means Koharu positively supports the candidate semantics.
    ``PROTECT`` means the proposal is panel/artwork-only and ordinary automatic
    fallbacks must not write it. ``UNKNOWN`` means Koharu is unavailable or did
    not give enough evidence either way; conservative fallback logic may inspect
    it without outranking a positive PROTECT decision.
    """
    if evidence is None or not evidence.available:
        return LayoutAuthorityDecision("UNKNOWN", "koharu_layout_unavailable")
    mask = _candidate_mask(candidate, shape)
    if int(np.count_nonzero(mask)) <= 0:
        return LayoutAuthorityDecision("UNKNOWN", "empty_candidate_mask")
    bo = _max_item_overlap(mask, evidence.by_label("bubble"))
    to = _max_item_overlap(mask, evidence.by_label("text"))
    so = _max_item_overlap(mask, evidence.by_label("sfx"))
    po = _max_item_overlap(mask, evidence.by_label("panel"))
    bubble_min = float(getattr(cfg, "paired_diff_koharu_layout_bubble_min_overlap", 0.25) if cfg is not None else 0.25)
    text_min = float(getattr(cfg, "paired_diff_koharu_layout_text_min_overlap", 0.35) if cfg is not None else 0.35)
    sfx_min = float(getattr(cfg, "paired_diff_koharu_layout_sfx_min_overlap", 0.35) if cfg is not None else 0.35)
    panel_min = float(getattr(cfg, "paired_diff_koharu_layout_panel_only_min_overlap", 0.20) if cfg is not None else 0.20)
    kind = str(region_kind or "bubble").strip().lower()
    if kind in {"complex_text", "free_text", "text", "sfx", "onomatopoeia"}:
        supported = bool(to >= text_min or so >= sfx_min)
    elif kind in {"bubble", "speech", "narration", "container"}:
        supported = bool(bo >= bubble_min or to >= text_min)
    else:
        supported = bool(bo >= bubble_min or to >= text_min or so >= sfx_min)
    if supported:
        return LayoutAuthorityDecision("ALLOW", "koharu_layout_semantic_support", bo, to, so, po)
    if po >= panel_min:
        return LayoutAuthorityDecision("PROTECT", "koharu_layout_panel_only_artwork", bo, to, so, po)
    return LayoutAuthorityDecision("UNKNOWN", "koharu_layout_no_semantic_support", bo, to, so, po)


def filter_candidates_by_layout_authority(
    candidates: Iterable[Any],
    evidence: LayoutEvidence | None,
    shape: tuple[int, int],
    *,
    region_kind: str = "bubble",
    cfg: Any | None = None,
    allow_unknown: bool = True,
    meta_key: str = "koharu_layout_authority",
) -> tuple[list[Any], list[dict[str, Any]]]:
    kept: list[Any] = []
    audit: list[dict[str, Any]] = []
    for row in list(candidates or []):
        decision = classify_layout_authority(evidence, row, shape, region_kind=region_kind, cfg=cfg)
        meta = dict(getattr(row, "meta", {}) or {})
        meta[meta_key] = decision.to_dict()
        try:
            row.meta = meta
        except Exception:
            pass
        accepted = decision.state == "ALLOW" or (allow_unknown and decision.state == "UNKNOWN")
        audit.append({
            "candidate_id": str(getattr(row, "id", "")),
            "accepted": bool(accepted),
            **decision.to_dict(),
        })
        if accepted:
            kept.append(row)
    return kept, audit


def collect_koharu_layout_evidence(
    image: np.ndarray,
    bubble_cfg: BubbleConfig | None = None,
    *,
    role: str = "page",
    allow_missing: bool = True,
) -> LayoutEvidence:
    cfg = bubble_cfg or BubbleConfig()
    shape = tuple(int(v) for v in image.shape[:2])
    resolved = discovered_model_path("koharu_layout", getattr(cfg, "koharu_layout_model_path", None))
    if resolved is None:
        detail = {
            "role": role,
            "available": False,
            "reason": "model_missing",
            "shape": shape,
        }
        if allow_missing:
            return LayoutEvidence(False, "koharu_layout", [], detail)
        raise ValueError("Koharu Layout 模型缺失；请在模型中心主动下载/离线导入。")

    from .vision_runtime import run_koharu_layout

    try:
        payload = run_koharu_layout(
            image,
            model_dir=str(resolved),
            device=str(getattr(cfg, "device", "auto")),
            text_threshold=float(getattr(cfg, "koharu_layout_text_threshold", 0.25)),
            sfx_threshold=float(getattr(cfg, "koharu_layout_sfx_threshold", 0.20)),
            bubble_threshold=float(getattr(cfg, "koharu_layout_bubble_threshold", 0.50)),
            panel_threshold=float(getattr(cfg, "koharu_layout_panel_threshold", 0.50)),
            shape=int(getattr(cfg, "koharu_layout_shape", 1152)),
            postprocess_max_side=int(getattr(cfg, "koharu_layout_postprocess_max_side", 1152)),
            postprocess_downscale_trigger_side=int(getattr(cfg, "koharu_layout_postprocess_downscale_trigger_side", 2048)),
        )
    except Exception as exc:
        detail = {
            "role": role,
            "available": False,
            "reason": "runtime_error",
            "error": str(exc),
            "shape": shape,
        }
        logger.exception(
            "Koharu Layout inference failed role=%s shape=%sx%s model=%s",
            role, shape[1], shape[0], resolved,
        )
        if allow_missing:
            return LayoutEvidence(False, "koharu_layout", [], detail)
        raise

    h, w = shape
    items: list[LayoutEvidenceItem] = []
    counts: dict[str, int] = {}
    for idx, raw in enumerate(list(payload.get("items") or [])):
        polygon_raw = raw.get("polygon") or []
        polygon = [(float(x), float(y)) for x, y in polygon_raw if len([x, y]) == 2]
        if len(polygon) < 3:
            box = raw.get("box") or []
            if len(box) == 4:
                x0, y0, x1, y1 = [float(v) for v in box]
                polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if len(polygon) < 3:
            continue
        mask = rasterize_polygon(polygon, (h, w))
        if cv2.countNonZero(mask) <= 0:
            continue
        label = _normalize_label(str(raw.get("label") or "unknown"))
        counts[label] = counts.get(label, 0) + 1
        meta = {
            "raw_label": str(raw.get("label") or label),
            "raw_box": list(raw.get("box") or []),
            "role": role,
            "isolated_runtime": True,
            "device": str(payload.get("device") or "isolated"),
            "item_index": idx,
        }
        items.append(LayoutEvidenceItem(
            label=label,
            confidence=float(raw.get("confidence", 0.0)),
            polygon=polygon,
            mask=mask,
            box=tuple(int(v) for v in polygon_bbox(polygon)),
            meta=meta,
        ))

    diagnostics = {
        "role": role,
        "available": True,
        "shape": shape,
        "counts": counts,
        "device": str(payload.get("device") or "isolated"),
        "item_count": len(items),
        "isolated_runtime": bool(payload.get("isolated_runtime", True)),
        "original_shape": list(payload.get("original_shape") or shape),
        "inference_shape": list(payload.get("inference_shape") or shape),
        "postprocess_max_side": int(payload.get("postprocess_max_side") or getattr(cfg, "koharu_layout_postprocess_max_side", 1152)),
        "postprocess_downscale_trigger_side": int(payload.get("postprocess_downscale_trigger_side") or getattr(cfg, "koharu_layout_postprocess_downscale_trigger_side", 2048)),
        "downscaled_for_postprocess": bool(payload.get("downscaled_for_postprocess", False)),
    }
    return LayoutEvidence(True, "koharu_layout", items, diagnostics)


def _mask_overlap(a: np.ndarray, b: np.ndarray) -> float:
    aa=np.asarray(a)>0; bb=np.asarray(b)>0
    denom=max(1,int(np.count_nonzero(aa)))
    return float(np.count_nonzero(aa & bb)/denom)


def collect_ysg_obb_layout_evidence(
    image: np.ndarray,
    bubble_cfg: BubbleConfig | None = None,
    *,
    role: str = "page",
    allow_missing: bool = True,
) -> LayoutEvidence:
    """Normalize the optional YSG YOLO OBB detector into positive layout evidence.

    YSG contributes *positive* bubble/open-text hints only. It has no panel class
    and therefore never creates PROTECT.  When merged with Koharu, Koharu panel
    protection is evaluated first and can veto every YSG candidate.
    """
    cfg = bubble_cfg or BubbleConfig()
    shape = tuple(int(v) for v in image.shape[:2])
    resolved = discovered_model_path("ysg_obb", getattr(cfg, "ysg_obb_model_path", None))
    if resolved is None:
        detail = {"role": role, "available": False, "reason": "model_missing", "shape": shape}
        if allow_missing:
            return LayoutEvidence(False, "ysg_obb", [], detail)
        raise ValueError("YSG YOLO OBB 模型缺失；请在模型中心主动下载/离线导入。")
    from .vision_runtime import run_ysg_obb
    try:
        payload = run_ysg_obb(
            image, model_path=str(resolved),
            confidence=float(getattr(cfg, "ysg_obb_confidence", .25)),
            iou=float(getattr(cfg, "ysg_obb_iou", .50)),
            imgsz=int(getattr(cfg, "ysg_obb_imgsz", 1600)),
            device=str(getattr(cfg, "device", "auto")),
        )
    except Exception as exc:
        logger.exception("YSG YOLO OBB inference failed role=%s shape=%sx%s model=%s", role, shape[1], shape[0], resolved)
        if allow_missing:
            return LayoutEvidence(False, "ysg_obb", [], {"role":role,"available":False,"reason":"runtime_error","error":str(exc),"shape":shape})
        raise
    h,w = shape; items=[]; counts={}
    include_other = bool(getattr(cfg, "ysg_obb_include_other", False))
    mapping = {"balloon":"bubble","qipao":"bubble","fangkuai":"bubble","changfangtiao":"bubble","kuangwai":"text","other":"text"}
    for idx, raw in enumerate(list(payload.get("items") or [])):
        raw_label = str(raw.get("label") or "").lower().strip()
        if raw_label == "other" and not include_other:
            continue
        label = mapping.get(raw_label)
        if label is None:
            continue
        polygon = [(float(x),float(y)) for x,y in list(raw.get("polygon") or [])]
        if len(polygon) < 3:
            continue
        mask = rasterize_polygon(polygon, (h,w))
        if cv2.countNonZero(mask) <= 0:
            continue
        counts[label] = counts.get(label,0)+1
        items.append(LayoutEvidenceItem(
            label=label, confidence=float(raw.get("confidence",0.0)), polygon=polygon,
            mask=mask, box=tuple(int(v) for v in polygon_bbox(polygon)),
            meta={"backend":"ysg_obb","raw_label":raw_label,"role":role,"item_index":idx,"obb":True,"device":str(payload.get("device") or "isolated")},
        ))
    return LayoutEvidence(True, "ysg_obb", items, {
        "role":role,"available":True,"shape":shape,"counts":counts,"item_count":len(items),
        "device":str(payload.get("device") or "isolated"),"positive_evidence_only":True,
    })


def collect_ysg_obb_layout_evidence_cached(
    image: np.ndarray, bubble_cfg: BubbleConfig | None = None, *, role: str = "page",
    image_path: str | Path | None = None, cache: PageStageCache | None = None,
    cache_enabled: bool = True, stats: dict[str, Any] | None = None, allow_missing: bool = True,
) -> LayoutEvidence:
    cfg = bubble_cfg or BubbleConfig()
    payload = {
        "backend":"ysg_obb", "role":str(role),
        "confidence":float(getattr(cfg,"ysg_obb_confidence",.25)),
        "iou":float(getattr(cfg,"ysg_obb_iou",.50)),
        "imgsz":int(getattr(cfg,"ysg_obb_imgsz",1600)),
        "include_other":bool(getattr(cfg,"ysg_obb_include_other",False)),
        "model_path":str(getattr(cfg,"ysg_obb_model_path","") or ""),
    }
    if image_path is not None:
        try:
            signature = image_stage_signature(image_path, payload) if Path(image_path).exists() else f"ysg-array:{_array_signature(image)}:{payload}"
        except OSError:
            signature = f"ysg-array:{_array_signature(image)}:{payload}"
    else:
        signature = f"ysg-array:{_array_signature(image)}:{payload}"
    cache_role = f"ysg_{_canonical_cache_role(role)}"
    if cache is not None and cache_enabled and image_path is not None and hasattr(cache,"load_layout_evidence"):
        hit = cache.load_layout_evidence(cache_role, signature)
        if hit is not None:
            if stats is not None: stats[f"layout_{cache_role}"] = "hit_disk"
            return hit
    evidence = collect_ysg_obb_layout_evidence(image, cfg, role=role, allow_missing=allow_missing)
    if cache is not None and cache_enabled and image_path is not None and evidence.available and hasattr(cache,"save_layout_evidence"):
        cache.save_layout_evidence(cache_role, signature, evidence)
    if stats is not None: stats[f"layout_{cache_role}"] = "miss_built" if evidence.available else "miss_unavailable"
    return evidence


def merge_positive_layout_evidence(
    authority: LayoutEvidence | None, auxiliary: LayoutEvidence | None, shape: tuple[int,int], *, cfg: Any | None = None,
) -> LayoutEvidence | None:
    """Merge positive auxiliary evidence without allowing it to override Koharu PROTECT."""
    if auxiliary is None or not auxiliary.available:
        return authority
    if authority is None or not authority.available:
        return auxiliary
    items = list(authority.items)
    rejected=0; added=0
    for item in auxiliary.items:
        kind = "free_text" if item.label in {"text","sfx"} else "bubble"
        proxy = BubbleInstance(id=f"aux-{added}", polygon=item.polygon, confidence=item.confidence, mask=item.mask, safe_mask=item.mask.copy(), meta={"region_kind":kind})
        decision = classify_layout_authority(authority, proxy, shape, region_kind=kind, cfg=cfg)
        if decision.state == "PROTECT":
            rejected += 1; continue
        # Avoid duplicate evidence; auxiliaries fill holes rather than inflate an existing region.
        if any(_mask_overlap(item.mask, old.mask) >= .62 for old in items if old.label in {item.label,"bubble","text"}):
            continue
        meta=dict(item.meta or {}); meta["authority_merge"] = decision.to_dict(); item.meta=meta
        items.append(item); added += 1
    diagnostics = dict(authority.diagnostics or {})
    diagnostics["auxiliary_positive_evidence"] = {"backend":auxiliary.backend,"added":added,"protect_rejected":rejected}
    return LayoutEvidence(True, f"{authority.backend}+{auxiliary.backend}", items, diagnostics)


__all__ = [
    "LayoutEvidenceItem",
    "LayoutAuthorityDecision",
    "LayoutEvidence",
    "classify_layout_authority",
    "filter_candidates_by_layout_authority",
    "collect_koharu_layout_evidence",
    "collect_koharu_layout_evidence_cached",
    "prepare_page_layout_evidence",
    "collect_ysg_obb_layout_evidence", "collect_ysg_obb_layout_evidence_cached",
    "merge_positive_layout_evidence",
]
