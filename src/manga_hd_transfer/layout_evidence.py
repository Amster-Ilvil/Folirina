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
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .cache import PageStageCache, image_stage_signature
from .config import BubbleConfig
from .geometry import polygon_bbox, polygon_centroid, rasterize_polygon
from .model_downloads import discovered_model_path
from .m6_runtime_plan import choose_detector_size, m6_cache_token, m6_layout_cache_token
from .models import BubbleInstance, TextBlock


logger = logging.getLogger(__name__)


from .layout_evidence_models import LayoutAuthorityDecision, LayoutEvidence, LayoutEvidenceItem, normalize_layout_label

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


def _koharu_requested_shape(cfg: BubbleConfig) -> int:
    """Resolve the explicit cloud/test Koharu shape without changing product defaults.

    Folirina's product default remains ``cfg.koharu_layout_shape`` (1152).  The
    cloud acceptance harness historically exports ``KOHARU_SHAPE=624``; honoring
    that explicit environment override keeps GUI runs on the same detector scale
    as the verified headless Koharu acceptance path.
    """
    base = int(getattr(cfg, "koharu_layout_shape", 1152) or 1152)
    raw = str(os.environ.get("KOHARU_SHAPE", "") or "").strip()
    if not raw:
        return base
    try:
        value = int(raw)
    except ValueError:
        return base
    return max(512, min(2048, value))


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
    legacy: bool = False,
) -> str:
    cfg = bubble_cfg or BubbleConfig()
    cache_role = _canonical_cache_role(role)
    payload = {
        'backend': 'koharu_layout',
        'mask_transport': 'foreground_png_v1',
        # New v5 layout signatures are intentionally role-neutral: SOURCE/TARGET
        # are cache-file scopes, not model-output semantics. This also lets the
        # in-process memo reuse identical image content across cooperating stages.
        'page_role': cache_role if legacy else 'shared',
        'thresholds': {
            'text': float(getattr(cfg, 'koharu_layout_text_threshold', 0.25)),
            'sfx': float(getattr(cfg, 'koharu_layout_sfx_threshold', 0.20)),
            'bubble': float(getattr(cfg, 'koharu_layout_bubble_threshold', 0.50)),
            'panel': float(getattr(cfg, 'koharu_layout_panel_threshold', 0.50)),
            'shape': _koharu_requested_shape(cfg),
            'effective_shape': int(choose_detector_size('koharu_layout', _koharu_requested_shape(cfg), image.shape[:2], config=cfg)),
            'postprocess_max_side': int(getattr(cfg, 'koharu_layout_postprocess_max_side', 1152)),
            'postprocess_downscale_trigger_side': int(getattr(cfg, 'koharu_layout_postprocess_downscale_trigger_side', 2048)),
            'device': str(getattr(cfg, 'device', 'auto')),
            'model_path': str(getattr(cfg, 'koharu_layout_model_path', '') or ''),
            'm6_koharu_weight_fp16_auto': bool(getattr(cfg, 'm6_koharu_weight_fp16_auto', True)),
            # v5 only binds the Koharu-relevant part of the M6 policy. The broad
            # token remains available for unrelated pipeline stages.
            'm6_policy': m6_cache_token(cfg) if legacy else m6_layout_cache_token(cfg),
        },
    }
    if image_path is not None:
        try:
            if Path(image_path).exists():
                return image_stage_signature(image_path, payload)
        except OSError:
            pass
    return (
        f"array:foreground_png_v1:{cache_role if legacy else 'shared'}:{_array_signature(image)}:"
        f"{payload['thresholds']['shape']}:{payload['thresholds']['postprocess_max_side']}:"
        f"{payload['thresholds']['postprocess_downscale_trigger_side']}:"
        f"{payload['thresholds']['text']}:{payload['thresholds']['bubble']}:{payload['thresholds']['m6_policy']}"
    )


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
    legacy_signature = _layout_signature(image, bubble_cfg, role=cache_role, image_path=image_path, legacy=True)

    def _mark_stats(state: str, *, origin: str, migrated: bool = False) -> None:
        if stats is None:
            return
        stats[f'layout_{role}'] = state
        # Later consumers can report hit_disk after the prefetch actually ran
        # the model. Preserve the first lookup for honest cold/warm timings.
        stats.setdefault(f'layout_{cache_role}_first_cache', state)
        stats.setdefault(f'layout_{cache_role}_first_cache_origin', origin)
        stats[f'layout_{cache_role}_cache'] = state
        stats[f'layout_{cache_role}_cache_origin'] = origin
        stats[f'layout_{cache_role}_cache_lookup'] = 'hit' if state.startswith('hit') else 'miss'
        if migrated:
            stats[f'layout_{cache_role}_cache_migrated'] = True

    if cache is not None and cache_enabled and image_path is not None and hasattr(cache, 'load_layout_evidence'):
        cached = cache.load_layout_evidence(cache_role, signature)
        if cached is not None:
            _mark_stats('hit_disk', origin='disk')
            return cached
        # One-time compatibility bridge for v4/M6 caches. The old signature
        # mixed unrelated pipeline parameters into the layout key, so v5 first
        # tries the exact old key and then rewrites the cache under the narrower
        # stage-specific key after a successful load.
        if legacy_signature != signature:
            cached = cache.load_layout_evidence(cache_role, legacy_signature)
            if cached is not None:
                _mark_stats('hit_disk_migrated', origin='disk_legacy', migrated=True)
                if hasattr(cache, 'save_layout_evidence'):
                    try:
                        cache.save_layout_evidence(cache_role, signature, cached)
                    except Exception:
                        logger.debug('Koharu cache migration write failed role=%s', role, exc_info=True)
                return cached

    memo = _LAYOUT_EVIDENCE_MEMO.get(signature)
    if memo is not None:
        _LAYOUT_EVIDENCE_MEMO.move_to_end(signature)
        token = 'hit_memory' if memo.available else 'hit_memory_unavailable'
        _mark_stats(token, origin='memory')
        if stats is not None and not memo.available:
            reason = str((memo.diagnostics or {}).get('reason') or 'unavailable')
            error = str((memo.diagnostics or {}).get('error') or '')
            stats[f'layout_{role}_reason'] = reason
            if error:
                stats[f'layout_{role}_error'] = error[:800]
        return memo

    started = time.perf_counter()
    evidence = collect_koharu_layout_evidence(image, bubble_cfg, role=role, allow_missing=allow_missing)
    elapsed = time.perf_counter() - started
    try:
        evidence.diagnostics['cache_build_seconds'] = round(float(elapsed), 4)
    except Exception:
        pass
    memo_reason = str((evidence.diagnostics or {}).get('reason') or '')
    if evidence.available or memo_reason == 'runtime_error':
        _LAYOUT_EVIDENCE_MEMO[signature] = evidence
        _LAYOUT_EVIDENCE_MEMO.move_to_end(signature)
        while len(_LAYOUT_EVIDENCE_MEMO) > _LAYOUT_EVIDENCE_MEMO_MAX:
            _LAYOUT_EVIDENCE_MEMO.popitem(last=False)
    if cache is not None and cache_enabled and image_path is not None and evidence.available and hasattr(cache, 'save_layout_evidence'):
        cache.save_layout_evidence(cache_role, signature, evidence)
    _mark_stats('miss_built' if evidence.available else 'miss_unavailable', origin='inference' if evidence.available else 'none')
    if stats is not None:
        stats[f'layout_{cache_role}_build_seconds'] = round(float(elapsed), 4)
        if not evidence.available:
            reason = str((evidence.diagnostics or {}).get('reason') or 'unavailable')
            error = str((evidence.diagnostics or {}).get('error') or '')
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
    # Run TARGET before SOURCE. Photographed/translation SOURCE pages are often
    # 4K--6K while TARGET is the canonical publication-size page. On CPU RF-DETR
    # a large SOURCE-first segmentation call can leave substantial postprocess
    # state resident and stall the immediately following request. SOURCE-last is
    # semantically equivalent, and all later stages consume these cached results.
    if need_target:
        out["target"] = collect_koharu_layout_evidence_cached(
            target, cfg, role="target", image_path=target_path, cache=cache,
            cache_enabled=cache_enabled, stats=stats, allow_missing=True,
        )
    if need_source:
        out["source"] = collect_koharu_layout_evidence_cached(
            source, cfg, role="source", image_path=source_path, cache=cache,
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
        stats["layout_device_receipts"] = {
            role: {name: ev.diagnostics.get(name, "unknown") for name in (
                "device", "requested_device", "model_device", "input_device", "precision", "fallback_reason"
            )}
            for role, ev in out.items() if ev.available
        }
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



def koharu_semantic_text_ink_mask(
    image: np.ndarray,
    evidence: LayoutEvidence | None,
    *,
    labels: tuple[str, ...] = ("text",),
    min_confidence: float = 0.50,
    pad_px: int = 3,
    dilate_px: int = 1,
    region_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Recover complete text ink inside Koharu RF-DETR semantic boxes.

    Koharu segmentation polygons are excellent *anchors* but can hug only a
    subset of a glyph/line.  The RF-DETR ``raw_box`` is a better semantic
    corridor for recovering the complete raster text.  This helper therefore
    uses the raw box only as a search window, derives foreground polarity from
    the trusted segmentation-vs-surrounding contrast, filters structural
    rules/hatching, and always retains the original segmentation support.

    The result is layout evidence, not OCR.  Callers should still intersect it
    with an authoritative writable container (bubble/text region) before any
    destructive operation.
    """
    h, w = image.shape[:2]
    out = np.zeros((h, w), np.uint8)
    if evidence is None or not evidence.available:
        return out
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    wanted = {normalize_layout_label(x) for x in labels}
    restrict = None
    if isinstance(region_mask, np.ndarray) and tuple(region_mask.shape[:2]) == (h, w):
        restrict = region_mask > 0

    for row in evidence.items:
        if row.label not in wanted or float(row.confidence) < float(min_confidence):
            continue
        raw_box = list((row.meta or {}).get("raw_box") or row.box or [])
        if len(raw_box) != 4:
            continue
        try:
            x0, y0, x1, y1 = [int(round(float(v))) for v in raw_box]
        except Exception:
            continue
        pad = max(0, int(pad_px))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            continue
        if restrict is not None and not np.any(restrict[y0:y1, x0:x1]):
            continue

        roi = gray[y0:y1, x0:x1]
        seg = np.zeros(roi.shape, np.uint8)
        pts = np.array(
            [[int(round(float(x))) - x0, int(round(float(y))) - y0] for x, y in row.polygon],
            dtype=np.int32,
        )
        if len(pts) >= 3:
            cv2.fillPoly(seg, [pts], 255)
        if cv2.countNonZero(seg) <= 0:
            continue

        seed_values = roi[seg > 0]
        seed_median = float(np.median(seed_values)) if seed_values.size else 128.0
        ring = cv2.dilate(seg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)))
        bg_values = roi[ring == 0]
        if bg_values.size < 32:
            bg_values = roi.reshape(-1)
        bg_median = float(np.median(bg_values)) if bg_values.size else 128.0
        dark_foreground = seed_median < bg_median
        if abs(seed_median - bg_median) < 8.0:
            dark_foreground = seed_median < 128.0

        flat = roi.reshape(-1)
        if dark_foreground:
            background = float(np.percentile(flat, 78))
            spread = float(np.percentile(flat, 90) - np.percentile(flat, 15))
            contrast = max(14.0, min(50.0, 14.0 + spread * 0.12))
            threshold = int(np.clip(background - contrast, 45, 210))
            candidate = (roi <= threshold).astype(np.uint8) * 255
        else:
            background = float(np.percentile(flat, 22))
            spread = float(np.percentile(flat, 85) - np.percentile(flat, 10))
            contrast = max(14.0, min(50.0, 14.0 + spread * 0.12))
            threshold = int(np.clip(background + contrast, 45, 245))
            candidate = (roi >= threshold).astype(np.uint8) * 255

        # Keep glyph-like components but reject long panel rules, hatching and
        # huge sparse structural blobs.  Exact Koharu segmentation is restored
        # afterwards so the detector remains the semantic anchor.
        n, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
        use = np.zeros_like(candidate)
        rh, rw = use.shape
        crop_area = max(1, rh * rw)
        for k in range(1, n):
            cx, cy, cw, ch, area = [int(v) for v in stats_cc[k]]
            if area < 2:
                continue
            fill = area / max(1, cw * ch)
            aspect = max(cw, ch) / max(1, min(cw, ch))
            if area / crop_area > 0.30 and fill < 0.30:
                continue
            if cw / max(1, rw) > 0.82 and ch < max(5, int(rh * 0.10)) and fill < 0.60:
                continue
            if ch / max(1, rh) > 0.92 and cw < max(5, int(rw * 0.06)) and fill < 0.35:
                continue
            if aspect > 14.0 and area > 40 and fill < 0.45:
                continue
            use[labels_cc == k] = 255

        use = cv2.bitwise_or(use, seg)
        n, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats((use > 0).astype(np.uint8), 8)
        clean = np.zeros_like(use)
        for k in range(1, n):
            if int(stats_cc[k, cv2.CC_STAT_AREA]) >= 2:
                clean[labels_cc == k] = 255
        radius = max(0, int(dilate_px))
        if radius > 0 and cv2.countNonZero(clean) > 0:
            clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
        if restrict is not None:
            local_restrict = restrict[y0:y1, x0:x1]
            clean[~local_restrict] = 0
        out[y0:y1, x0:x1] = cv2.bitwise_or(out[y0:y1, x0:x1], clean)
    return out

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
        requested_shape = _koharu_requested_shape(cfg)
        dynamic_shape = choose_detector_size(
            "koharu_layout", requested_shape, shape, config=cfg
        )
        post_max = min(int(getattr(cfg, "koharu_layout_postprocess_max_side", 1152)), dynamic_shape)
        payload = run_koharu_layout(
            image,
            model_dir=str(resolved),
            device=str(getattr(cfg, "device", "auto")),
            text_threshold=float(getattr(cfg, "koharu_layout_text_threshold", 0.25)),
            sfx_threshold=float(getattr(cfg, "koharu_layout_sfx_threshold", 0.20)),
            bubble_threshold=float(getattr(cfg, "koharu_layout_bubble_threshold", 0.50)),
            panel_threshold=float(getattr(cfg, "koharu_layout_panel_threshold", 0.50)),
            shape=dynamic_shape,
            postprocess_max_side=post_max,
            postprocess_downscale_trigger_side=max(int(getattr(cfg, "koharu_layout_postprocess_downscale_trigger_side", 2048)), dynamic_shape),
            optimize_weights_fp16=bool(getattr(cfg, "m6_koharu_weight_fp16_auto", True)),
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
        foreground = raw.get("foreground_mask")
        if foreground:
            from .koharu_mask_codec import decode_mask
            mask = decode_mask(foreground, (h, w))
            full_box = raw.get("box") or []
            if len(full_box) == 4:
                x0,y0,x1,y1 = map(float, full_box)
                polygon = [(x0,y0),(x1,y0),(x1,y1),(x0,y1)]
        else:
            mask = rasterize_polygon(polygon, (h, w))
        if cv2.countNonZero(mask) <= 0:
            continue
        label = normalize_layout_label(str(raw.get("label") or "unknown"))
        counts[label] = counts.get(label, 0) + 1
        meta = {
            "raw_label": str(raw.get("label") or label),
            "raw_box": list(raw.get("box") or []),
            "role": role,
            "isolated_runtime": True,
            "device": str(payload.get("device") or "isolated"),
            "item_index": idx,
        }
        if foreground:
            meta["foreground_mask"] = foreground
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
        "requested_inference_shape": _koharu_requested_shape(cfg),
        "counts": counts,
        "device": str(payload.get("device") or "isolated"),
        "item_count": len(items),
        "isolated_runtime": bool(payload.get("isolated_runtime", True)),
        "original_shape": list(payload.get("original_shape") or shape),
        "inference_shape": list(payload.get("inference_shape") or shape),
        "postprocess_max_side": int(payload.get("postprocess_max_side") or getattr(cfg, "koharu_layout_postprocess_max_side", 1152)),
        "postprocess_downscale_trigger_side": int(payload.get("postprocess_downscale_trigger_side") or getattr(cfg, "koharu_layout_postprocess_downscale_trigger_side", 2048)),
        "downscaled_for_postprocess": bool(payload.get("downscaled_for_postprocess", False)),
        "model_cache_hit": bool(payload.get("model_cache_hit", False)),
        "model_load_seconds": float(payload.get("model_load_seconds", 0.0) or 0.0),
        "optimize_seconds": float(payload.get("optimize_seconds", 0.0) or 0.0),
        "optimized_weights_fp16": bool(payload.get("optimized_weights_fp16", False)),
        "optimize_weights_fp16_requested": bool(payload.get("optimize_weights_fp16_requested", False)),
        "predict_seconds": float(payload.get("predict_seconds", 0.0) or 0.0),
        "requested_device": str(payload.get("requested_device", "auto")),
        "precision": str(payload.get("precision", "unknown")),
        "model_device": str(payload.get("model_device", "unknown")),
        "input_device": str(payload.get("input_device", "unknown")),
        "fallback_reason": str(payload.get("fallback_reason", "")),
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
            imgsz=choose_detector_size("ysg_obb", max(640, int(getattr(cfg, "ysg_obb_imgsz", 1600))), shape, config=cfg),
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
        "imgsz":choose_detector_size("ysg_obb", max(640, int(getattr(cfg,"ysg_obb_imgsz",1600))), image.shape[:2], config=cfg),
        "m6_policy":m6_cache_token(cfg),
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
    "koharu_semantic_text_ink_mask",
    "prepare_page_layout_evidence",
    "collect_ysg_obb_layout_evidence", "collect_ysg_obb_layout_evidence_cached",
    "merge_positive_layout_evidence",
]
