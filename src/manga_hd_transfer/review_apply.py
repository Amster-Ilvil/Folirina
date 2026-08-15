from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import cv2
import numpy as np

from .config import PipelineConfig
from .export import export_openraster, export_psd_imagemagick, make_text_layer_rgba, write_rgba
from .inpainting import inpaint_image
from .io_utils import load_json, read_image, save_json, write_image
from .lettering import composite_text, fit_text, polygon_safe_mask
from .masking import build_clear_mask
from .manual_effect import build_manual_effect_masks, apply_reveal_window, estimate_source_background, composite_source_text_delta, strip_border_ring_components, clean_manual_target_text, white_container_safe_mask
from .models import BubbleInstance, PagePair, TextBlock, TextUnit, UnitMatch
from .text_only_transfer import clear_to_target_paper, target_text_mask_in_container
from .schema_compat import as_dict, as_dict_rows, as_list, normalize_project, normalize_overrides, normalize_route_meta
from .result_state import ensure_manual_baseline, manual_baseline_path, commit_reviewed_result, resolve_result_state


def _dict_or_empty(value):
    """Return a plain dict for mixed/legacy JSON schema values."""
    return as_dict(value)


def _route_meta(meta, key: str) -> dict:
    return normalize_route_meta(_dict_or_empty(meta).get(key))


def _dict_rows(value) -> list[dict]:
    """Normalize a stale list-like review field to dictionary rows only."""
    return as_dict_rows(value)


def _text_block(row: dict) -> TextBlock:
    return TextBlock(**row)


def _text_unit(row: dict) -> TextUnit:
    return TextUnit(**row)


def _load_target_bubbles(page_dir: Path, rows: list[dict]) -> list[BubbleInstance]:
    out = []
    for row in rows:
        b = BubbleInstance(
            id=row["id"],
            polygon=row["polygon"],
            confidence=row.get("confidence", 1.0),
            kind=row.get("kind", "speech"),
            block_ids=list(row.get("block_ids", [])),
            meta=as_dict(row.get("meta")),
        )
        mp = page_dir / "bubbles" / f"{b.id}.png"
        sp = page_dir / "bubbles" / f"{b.id}_safe.png"
        if mp.exists():
            b.mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if sp.exists():
            b.safe_mask = cv2.imread(str(sp), cv2.IMREAD_GRAYSCALE)
        out.append(b)
    return out


def _rect_mask(shape: tuple[int, int], bbox: list[int] | tuple[int, int, int, int], inset: int = 2) -> np.ndarray:
    x0, y0, x1, y1 = map(int, bbox)
    x0 = max(0, min(shape[1], x0 + inset)); y0 = max(0, min(shape[0], y0 + inset))
    x1 = max(0, min(shape[1], x1 - inset)); y1 = max(0, min(shape[0], y1 - inset))
    mask = np.zeros(shape, np.uint8)
    if x1 > x0 and y1 > y0:
        cv2.rectangle(mask, (x0, y0), (x1, y1), 255, -1)
    return mask


def _clear_region_to_paper(rendered: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = rendered.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    sel = mask > 0
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    bright = sel & (gray >= 205)
    if np.count_nonzero(bright) >= 20:
        paper = np.median(target[bright], axis=0).astype(np.uint8)
    else:
        paper = np.array([255, 255, 255], np.uint8)
    out[sel] = paper
    return out


def _load_effective_clear_mask(page_dir: Path, shape: tuple[int, int]) -> tuple[np.ndarray, str]:
    """Load base clear mask plus the additive Japanese-cleanup brush.

    ``manual_clear_mask.png`` is the legacy authoritative full-mask editor.  The
    v1.3 ``manual_japanese_clear_mask.png`` is deliberately additive so a reviewer
    can brush missed Japanese without replacing the automatic detector mask.
    """
    base = np.zeros(shape, np.uint8)
    source = "none"
    candidates = [
        (page_dir / "manual_clear_mask.png", "manual_clear_mask"),
        (page_dir / "target_clear_mask.png", "target_clear_mask"),
        (page_dir / "clear_mask.png", "clear_mask"),
    ]
    for path, label in candidates:
        if not path.exists():
            continue
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != shape:
            if label == "manual_clear_mask":
                raise ValueError("manual_clear_mask.png has invalid dimensions")
            continue
        base = (mask > 0).astype(np.uint8) * 255
        source = label
        break

    additive = page_dir / "manual_japanese_clear_mask.png"
    if additive.exists():
        extra = cv2.imread(str(additive), cv2.IMREAD_GRAYSCALE)
        if extra is None or extra.shape != shape:
            raise ValueError("manual_japanese_clear_mask.png has invalid dimensions")
        base = np.maximum(base, (extra > 0).astype(np.uint8) * 255)
        source = f"{source}+manual_japanese_clear_mask" if source != "none" else "manual_japanese_clear_mask"
    return base, source


def _load_clear_brush_settings(page_dir: Path) -> dict:
    """Read page-local Japanese-cleanup behavior without mutating user config."""
    path = page_dir / "clear_brush_settings.json"
    data = load_json(path) if path.exists() else {}
    mode = str(data.get("mode", "strict")).strip().lower()
    if mode not in {"strict", "aggressive"}:
        mode = "strict"
    default_dilate = 1 if mode == "strict" else 3
    try:
        dilate = int(data.get("dilate_px", default_dilate))
    except Exception:
        dilate = default_dilate
    dilate = max(0, min(8, dilate))
    return {"mode": mode, "dilate_px": dilate}


def _apply_manual_brush_expansion(page_dir: Path, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Expand only the additive brush; never shrink/crop explicit reviewer intent."""
    settings = _load_clear_brush_settings(page_dir)
    additive_path = page_dir / "manual_japanese_clear_mask.png"
    if not additive_path.exists():
        return mask.copy(), {**settings, "manual_pixels": 0, "expanded_pixels": 0}
    extra = cv2.imread(str(additive_path), cv2.IMREAD_GRAYSCALE)
    if extra is None or extra.shape != mask.shape:
        raise ValueError("manual_japanese_clear_mask.png has invalid dimensions")
    raw = (extra > 0).astype(np.uint8) * 255
    expanded = raw.copy()
    r = int(settings["dilate_px"])
    if r > 0 and cv2.countNonZero(expanded) > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (r * 2 + 1, r * 2 + 1))
        expanded = cv2.dilate(expanded, k, iterations=1)
    out = np.maximum(mask, expanded)
    return out, {
        **settings,
        "manual_pixels": int(cv2.countNonZero(raw)),
        "expanded_pixels": int(cv2.countNonZero(expanded)),
        "explicit_mask_never_clipped": True,
    }


def _residual_dark_heatmap(target: np.ndarray, cleaned: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, dict]:
    """Visualize dark TARGET pixels that remain dark after the remove-text stage."""
    if target.shape != cleaned.shape or target.shape[:2] != mask.shape:
        raise ValueError("residual heatmap inputs must share shape")
    tgray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    cgray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    use = mask > 0
    target_dark = use & (tgray <= 205)
    residual = target_dark & (cgray <= 205)
    denom = max(1, int(np.count_nonzero(target_dark)))
    ratio = float(np.count_nonzero(residual) / denom)
    heat = target.copy()
    if np.any(residual):
        overlay = heat.copy()
        overlay[residual] = (0, 0, 255)
        heat = cv2.addWeighted(heat, 0.48, overlay, 0.52, 0.0)
    return heat, {
        "target_dark_pixels": int(np.count_nonzero(target_dark)),
        "residual_dark_pixels": int(np.count_nonzero(residual)),
        "residual_dark_ratio": ratio,
    }


def _read_layer_alpha(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if layer is None or layer.shape[:2] != shape:
        return None
    if layer.ndim == 2:
        return (layer > 0).astype(np.uint8) * 255
    if layer.ndim == 3 and layer.shape[2] >= 4:
        return layer[:, :, 3].astype(np.uint8)
    return None


def _protected_chinese_mask(page_dir: Path, shape: tuple[int, int], *, extra_masks: list[np.ndarray] | None = None, margin_px: int = 1) -> tuple[np.ndarray, dict]:
    """Collect visible Chinese ink that TARGET-only erasing must never touch.

    Transfer-layer alpha is sometimes a *write* mask: it may include pixels used
    only to clear old Japanese, not just Chinese glyphs. Protecting that raw alpha
    would defeat this tool exactly at Japanese AA fringes. Therefore automatic
    raster layers are reduced to likely Chinese ink: alpha outside the automatic
    clear mask is trusted directly, while overlap with the clear mask must also
    carry local stroke detail. True text-only reviewed/manual layers keep their
    exact alpha.
    """
    protect = np.zeros(shape, np.uint8)
    sources: list[str] = []
    clear = np.zeros(shape, np.uint8)
    for clear_name in ('effective_clear_mask.png', 'target_clear_mask.png', 'clear_mask.png'):
        cp = page_dir / clear_name
        if cp.exists():
            cm = cv2.imread(str(cp), cv2.IMREAD_GRAYSCALE)
            if cm is not None and cm.shape == shape:
                clear = (cm > 0).astype(np.uint8) * 255
                break

    def add_raster_layer(path: Path) -> bool:
        layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED) if path.exists() else None
        if layer is None or layer.shape[:2] != shape or layer.ndim != 3 or layer.shape[2] < 4:
            return False
        alpha = layer[:, :, 3].astype(np.uint8)
        if cv2.countNonZero(alpha) == 0:
            return False
        bgr = layer[:, :, :3]
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        local = cv2.GaussianBlur(gray, (0, 0), sigmaX=2.2, sigmaY=2.2)
        detail = cv2.absdiff(gray, local)
        edge = cv2.Canny(gray, 24, 72, L2gradient=True)
        ink_like = (detail >= 7) | (edge > 0)
        # Outside the JP clear footprint, nonzero transfer alpha is overwhelmingly
        # Chinese write alpha. Inside it, require actual stroke/detail evidence so
        # clear-only paper pixels never shield a missed Japanese fragment.
        keep = (alpha > 0) & ((clear == 0) | ink_like)
        if np.any(keep):
            protect[keep] = 255
        sources.append(path.name)
        return True

    canonical = page_dir / 'chinese_transfer_layer.png'
    if not add_raster_layer(canonical):
        for name in ('direct_patch_layer_reviewed.png', 'mask_transfer_layer_reviewed.png', 'direct_patch_layer.png', 'mask_transfer_layer.png'):
            add_raster_layer(page_dir / name)
        # Automatic reletter path is already text-only.
        a = _read_layer_alpha(page_dir / 'text_layer.png', shape)
        if a is not None:
            protect = np.maximum(protect, a)
            sources.append('text_layer.png')

    # These are explicit Chinese-only layers created after automatic transfer.
    for name in ('text_layer_reviewed.png', 'manual_effect_transfer_layer.png'):
        path = page_dir / name
        a = _read_layer_alpha(path, shape)
        if a is not None:
            protect = np.maximum(protect, a)
            sources.append(name)
    for i, mask in enumerate(extra_masks or []):
        arr = np.asarray(mask)
        if arr.shape != shape:
            continue
        protect = np.maximum(protect, (arr > 0).astype(np.uint8) * 255)
        sources.append(f'extra_mask_{i}')
    raw_pixels = int(cv2.countNonZero(protect))
    margin = max(0, min(4, int(margin_px)))
    if raw_pixels and margin > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2 + 1, margin * 2 + 1))
        protect = cv2.dilate(protect, k, iterations=1)
    return (protect > 0).astype(np.uint8) * 255, {
        'sources': sources,
        'clear_reference_pixels': int(cv2.countNonZero(clear)),
        'raw_chinese_pixels': raw_pixels,
        'protected_chinese_pixels': int(cv2.countNonZero(protect)),
        'protect_margin_px': margin,
    }

def _clean_target_under_erase_mask(target: np.ndarray, mask: np.ndarray, cfg: PipelineConfig) -> tuple[np.ndarray, dict]:
    """Reconstruct TARGET underneath an explicit erase mask.

    General colour/artwork areas use the configured inpaint backend. Components
    surrounded by neutral bright paper are normalized to the local paper median,
    which avoids Telea/JPEG black specks in ordinary speech balloons.
    """
    if cv2.countNonZero(mask) == 0:
        return target.copy(), {'paper_components': 0, 'inpaint_pixels': 0}
    cleaned = inpaint_image(target, mask, cfg.inpainting).image
    binary = (mask > 0).astype(np.uint8)
    gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    paper_components = 0
    paper_pixels = 0
    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for lab in range(1, count):
        comp = labels == lab
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        dil = cv2.dilate(comp.astype(np.uint8), ring_kernel, iterations=1) > 0
        ring = dil & ~comp
        neutral = ring & (gray >= 212) & (hsv[..., 1] <= 52)
        vals = target[neutral]
        if vals.shape[0] < max(12, min(80, area // 4)):
            continue
        paper = np.median(vals, axis=0).astype(np.uint8)
        # Only normalize when the surrounding ring itself proves a white/neutral
        # container. Coloured/gradient regions retain inpainting.
        if float(np.mean(cv2.cvtColor(paper.reshape(1,1,3), cv2.COLOR_BGR2GRAY))) < 215:
            continue
        cleaned[comp] = paper
        paper_components += 1
        paper_pixels += area
    return cleaned, {
        'paper_components': int(paper_components),
        'paper_pixels': int(paper_pixels),
        'inpaint_pixels': int(cv2.countNonZero(mask)),
    }


def _apply_target_layer_erase_to_rendered(
    page_dir: Path,
    rendered: np.ndarray,
    target: np.ndarray,
    cfg: PipelineConfig,
    *,
    refresh_base: bool = False,
    extra_protect_masks: list[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict]:
    """Erase only TARGET-layer pixels while preserving Chinese byte-for-byte.

    The reviewer is allowed to paint broadly across residual JP strokes, lines or
    punctuation. The explicit erase mask is lightly expanded for antialiasing,
    then Chinese alpha (plus a safety margin) is subtracted before any pixel is
    replaced. Thus a brush stroke may cross translated Chinese without altering
    those Chinese pixels.
    """
    mask_path = page_dir / 'manual_target_layer_erase_mask.png'
    if not mask_path.exists():
        return rendered.copy(), {'used': False, 'reason': 'mask_missing'}
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != target.shape[:2]:
        raise ValueError('manual_target_layer_erase_mask.png has invalid dimensions')
    raw = (raw > 0).astype(np.uint8) * 255
    if refresh_base:
        write_image(page_dir / 'target_layer_erase_base.png', rendered)
    settings_path = page_dir / 'target_layer_erase_settings.json'
    settings = as_dict(load_json(settings_path)) if settings_path.exists() else {}
    dilate_px = max(0, min(4, int(settings.get('dilate_px', 1) or 0)))
    protect_margin = max(0, min(6, int(settings.get('protect_chinese_margin_px', 1) or 0)))
    expanded = raw.copy()
    if cv2.countNonZero(expanded) and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        expanded = cv2.dilate(expanded, k, iterations=1)
    protect, protect_diag = _protected_chinese_mask(
        page_dir, target.shape[:2], extra_masks=extra_protect_masks, margin_px=protect_margin
    )
    effective = expanded.copy()
    effective[protect > 0] = 0
    fill_mode = str(settings.get('fill_mode', 'auto') or 'auto').strip().lower()
    if fill_mode == 'pure_white':
        cleaned_target = target.copy()
        cleaned_target[effective > 0] = 255
        clean_diag = {'mode': 'pure_white', 'pure_white_pixels': int(cv2.countNonZero(effective))}
    else:
        cleaned_target, clean_diag = _clean_target_under_erase_mask(target, effective, cfg)
        fill_mode = 'auto'
    out = rendered.copy()
    sel = effective > 0
    out[sel] = cleaned_target[sel]
    # Hard contract: even if future refactors change mask operations, protected
    # Chinese pixels are restored exactly from the incoming rendered page.
    psel = protect > 0
    out[psel] = rendered[psel]
    write_image(page_dir / 'target_layer_erase_effective_mask.png', effective)
    write_image(page_dir / 'target_layer_erase_chinese_protect_mask.png', protect)
    write_image(page_dir / 'target_layer_erase_preview.png', out)
    diag = {
        'schema': 'manga_hd_translation_transfer.target_layer_erase.v1',
        'used': bool(cv2.countNonZero(raw)),
        'raw_mask_pixels': int(cv2.countNonZero(raw)),
        'expanded_mask_pixels': int(cv2.countNonZero(expanded)),
        'effective_erase_pixels': int(cv2.countNonZero(effective)),
        'blocked_by_chinese_pixels': int(cv2.countNonZero(cv2.bitwise_and(expanded, protect))),
        'dilate_px': int(dilate_px),
        'fill_mode': fill_mode,
        'chinese_protection': protect_diag,
        'cleanup': clean_diag,
        'background_policy': 'target_original_only',
        'chinese_pixel_policy': 'byte_for_byte_preserve',
    }
    save_json(page_dir / 'target_layer_erase.json', diag)
    return out, diag


def apply_target_layer_erase_review(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    """Fast final-stage TARGET-only erase without rerunning OCR/registration."""
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    target = read_image(page_dir / 'target_original.png')
    base_path = page_dir / 'target_layer_erase_base.png'
    if base_path.exists():
        base = read_image(base_path)
    else:
        state = resolve_result_state(page_dir)
        source = state.current or (page_dir / 'final.png')
        if not Path(source).exists():
            raise FileNotFoundError('current final result is missing')
        base = read_image(source)
        write_image(base_path, base)
    rendered, diag = _apply_target_layer_erase_to_rendered(page_dir, base, target, cfg, refresh_base=False)
    final_path = page_dir / 'final_reviewed.png'
    write_image(final_path, rendered)
    diag['final'] = str(final_path)
    save_json(page_dir / 'target_layer_erase.json', diag)
    return commit_reviewed_result(page_dir, final_path)


def reset_target_layer_erase_review(page_dir: str | Path) -> Path | None:
    """Remove the dedicated TARGET erase mask and restore its pre-erase base."""
    page_dir = Path(page_dir)
    base_path = page_dir / 'target_layer_erase_base.png'
    restored: Path | None = None
    if base_path.exists():
        base = read_image(base_path)
        restored = page_dir / 'final_reviewed.png'
        write_image(restored, base)
    for name in (
        'manual_target_layer_erase_mask.png', 'target_layer_erase_effective_mask.png',
        'target_layer_erase_chinese_protect_mask.png', 'target_layer_erase_preview.png',
        'target_layer_erase.json', 'target_layer_erase_settings.json', 'target_layer_erase_base.png',
    ):
        try:
            (page_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    return commit_reviewed_result(page_dir, restored) if restored is not None else None


def generate_remove_text_preview(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    """Run only the target-text removal stage for the current page.

    This is deliberately independent from Chinese raster transfer. It lets the
    user inspect/repair the clear mask without rerunning registration, detection,
    OCR or glyph placement. The result is safe to regenerate repeatedly.
    """
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    target = read_image(page_dir / "target_original.png")
    raw_mask, source = _load_effective_clear_mask(page_dir, target.shape[:2])
    mask, brush_diag = _apply_manual_brush_expansion(page_dir, raw_mask)
    # Persist exactly what the remove/apply stage will use. This is the
    # reviewer-facing "真实生效 mask", independent from the brush source file.
    effective_path = page_dir / "effective_clear_mask.png"
    write_image(effective_path, mask)
    if cv2.countNonZero(mask) == 0:
        preview = target.copy()
        backend = "none"
    else:
        result = inpaint_image(target, mask, cfg.inpainting)
        preview = result.image
        backend = str(getattr(result, "method", getattr(cfg.inpainting, "backend", "auto")))
    out = page_dir / "removed_text_preview.png"
    write_image(out, preview)
    heat, residual_diag = _residual_dark_heatmap(target, preview, mask)
    heat_path = page_dir / "japanese_residual_heatmap.png"
    write_image(heat_path, heat)
    save_json(page_dir / "remove_text_stage.json", {
        "schema": "manga_hd_translation_transfer.remove_text_stage.v2",
        "mask_source": source,
        "raw_mask_pixels": int(cv2.countNonZero(raw_mask)),
        "mask_pixels": int(cv2.countNonZero(mask)),
        "effective_mask": str(effective_path),
        "brush": brush_diag,
        "inpainting_backend": backend,
        **residual_diag,
        "residual_review_recommended": bool(residual_diag["target_dark_pixels"] >= 8 and residual_diag["residual_dark_ratio"] > 0.08),
        "residual_heatmap": str(heat_path),
        "output": str(out),
    })
    return out



def _source_for_review(page_dir: Path, project: dict) -> np.ndarray:
    local = page_dir / "source_original.png"
    if local.exists():
        return read_image(local)
    pair = dict(project.get("pair", {}) or {})
    source_path = str(pair.get("source_path", "") or "")
    if not source_path:
        raise FileNotFoundError("manual effect transfer needs source_original.png or pair.source_path")
    return read_image(source_path)


def _write_bgra(path: Path, bgra: np.ndarray) -> None:
    ok, data = cv2.imencode(".png", bgra)
    if not ok:
        raise ValueError(f"could not encode {path.name}")
    data.tofile(path)


def _alpha_over_bgra(bottom: np.ndarray, top: np.ndarray) -> np.ndarray:
    """Alpha-composite one BGRA manual layer over an existing BGRA layer."""
    if bottom.shape != top.shape:
        raise ValueError("manual effect layer size mismatch")
    ba = bottom[:, :, 3].astype(np.float32) / 255.0
    ta = top[:, :, 3].astype(np.float32) / 255.0
    out_a = ta + ba * (1.0 - ta)
    out = np.zeros_like(bottom)
    denom = np.maximum(out_a, 1e-6)
    for c in range(3):
        bc = bottom[:, :, c].astype(np.float32)
        tc = top[:, :, c].astype(np.float32)
        out[:, :, c] = np.clip((tc * ta + bc * ba * (1.0 - ta)) / denom, 0, 255).astype(np.uint8)
    out[:, :, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)
    return out


def _load_reveal_commit_patch(page_dir: Path, row: dict, shape: tuple[int, int]) -> np.ndarray | None:
    """Load an exact sparse preview patch saved by the Qt Reveal editor.

    Older projects do not have this artifact and transparently fall back to
    recomputing the effect from SOURCE/TARGET masks.
    """
    name = str(row.get("reveal_patch_file", "") or "").strip()
    if not name:
        return None
    path = page_dir / name
    if not path.exists():
        return None
    patch = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if patch is None or patch.ndim != 3 or patch.shape[2] != 4 or patch.shape[:2] != shape:
        return None
    return patch


def _apply_manual_effect_regions(
    rendered: np.ndarray,
    target: np.ndarray,
    page_dir: Path,
    project: dict,
    overrides: dict,
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """Apply arbitrary detector-independent open/SFX recovery regions.

    This path intentionally does not depend on OCR or a detected speech bubble.
    A human supplies the small target rectangle; paired-image evidence then
    separates SOURCE-supported Chinese strokes from TARGET-supported Japanese
    strokes.  The target-only strokes are inpainted and the Chinese source raster
    is composited afterwards.  ``full_patch`` is also available for genuinely
    safe white/paper regions.
    """
    rows = [dict(x) for x in _dict_rows(overrides.get("manual_effect_regions")) if bool(x.get("enabled", True))]
    h, w = target.shape[:2]
    empty_layer = np.zeros((h, w, 4), np.uint8)
    empty_mask = np.zeros((h, w), np.uint8)
    if not rows:
        return rendered, empty_layer, empty_mask, []

    source = _source_for_review(page_dir, project)
    out = rendered.copy()
    effect_layer = np.zeros((h, w, 4), np.uint8)
    all_clear = np.zeros((h, w), np.uint8)
    applied: list[dict] = []
    for index, row in enumerate(rows):
        try:
            masks = build_manual_effect_masks(source, target, project, row, cfg)
        except Exception as exc:
            applied.append({"id": str(row.get("id", f"manual-effect-{index:03d}")), "success": False, "reason": str(exc)})
            continue
        full_source_mask = masks.source_mask.copy()
        source_mask = full_source_mask.copy()
        clear_mask = masks.target_clear_mask.copy()
        mode = str(row.get("mode", "effect_text") or "effect_text")
        reveal_patch = None
        if mode == "reveal_text":
            mask_name = str(row.get("reveal_mask_file", "") or "")
            mask_path = page_dir / mask_name if mask_name else None
            reveal = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) if mask_path is not None and mask_path.exists() else None
            if reveal is None or reveal.shape != target.shape[:2]:
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": False,
                    "reason": "reveal mask missing or size mismatch",
                    "mode": mode,
                })
                continue
            source_mask, clear_mask = apply_reveal_window(source_mask, clear_mask, reveal)
            reveal_patch = _load_reveal_commit_patch(page_dir, row, target.shape[:2])
            if reveal_patch is not None:
                # The sparse preview patch is authoritative. Empty patches are a
                # no-op: never clear TARGET just because an old/stale reveal mask
                # exists. This preserves the transactional "save only if Chinese
                # pixels were actually produced" contract.
                patch_alpha = np.minimum(reveal_patch[:, :, 3], reveal).astype(np.uint8)
                patch_sel = patch_alpha > 0
                if np.any(patch_sel) and cv2.countNonZero(clear_mask) > 0:
                    # Clear TARGET Japanese first, then place the sparse Chinese
                    # Reveal patch. Flat colour bubbles use TARGET fill
                    # reconstruction instead of a blurry full-ROI inpaint.
                    cleaned, clean_diag = clean_manual_target_text(
                        target, clear_mask, bbox=as_list(row.get("target_bbox"))
                    )
                    clear_sel = clear_mask > 0
                    out[clear_sel] = cleaned[clear_sel]
                    masks.diagnostics.setdefault("manual_target_cleanup", {}).update(clean_diag)
                if np.any(patch_sel):
                    out[patch_sel] = reveal_patch[:, :, :3][patch_sel]
                    top = reveal_patch.copy(); top[:, :, 3] = patch_alpha
                    effect_layer = _alpha_over_bgra(effect_layer, top)
                all_clear = np.maximum(all_clear, clear_mask)
                diag = dict(masks.diagnostics)
                applied.append({
                    "id": str(row.get("id", f"manual-effect-{index:03d}")),
                    "success": bool(np.any(patch_sel)),
                    "mode": mode,
                    "target_bbox": as_list(row.get("target_bbox")),
                    "source_pixels": int(cv2.countNonZero(source_mask)),
                    "target_clear_pixels": int(cv2.countNonZero(clear_mask)),
                    "preview_patch_applied": True,
                    "preview_patch_pixels": int(cv2.countNonZero(patch_alpha)),
                    "preview_patch_exact": bool(np.array_equal(out[patch_sel], reveal_patch[:, :, :3][patch_sel])) if np.any(patch_sel) else False,
                    "diagnostics": diag,
                })
                continue
        if mode in {"full_patch", "white_bubble_text"}:
            # Manual white-bubble correction is replacement, not additive.  The
            # current base may already contain an automatically transferred CN
            # layer, so clear both TARGET JP text and any existing rendered text
            # inside the confirmed white container before drawing the nudged CN.
            region = np.zeros((h, w), np.uint8)
            safe = np.zeros((h, w), np.uint8)
            bx = as_list(row.get("target_bbox"))
            inset = 4
            if len(bx) == 4:
                rx0, ry0, rx1, ry1 = [int(v) for v in bx]
                rx0 = max(0, min(w, rx0)); rx1 = max(0, min(w, rx1))
                ry0 = max(0, min(h, ry0)); ry1 = max(0, min(h, ry1))
                if rx1 > rx0 and ry1 > ry0:
                    region[ry0:ry1, rx0:rx1] = 255
                    lo = max(0, int(getattr(cfg.mask_replace, "white_container_manual_inset_min_px", 1)))
                    hi = max(lo, int(getattr(cfg.mask_replace, "white_container_manual_inset_max_px", 4)))
                    ratio = max(0.0, float(getattr(cfg.mask_replace, "white_container_manual_inset_ratio", 0.02)))
                    safe, safe_diag = white_container_safe_mask(
                        target, region,
                        inset_min_px=lo,
                        inset_max_px=hi,
                        inset_ratio=ratio,
                    )
                    inset = int(safe_diag.get("container_border_inset_px", 0) or 0)
                    masks.diagnostics.setdefault("white_container_safe_mask", safe_diag)
            current_text = target_text_mask_in_container(out, safe) if cv2.countNonZero(safe) else np.zeros((h, w), np.uint8)
            authority = cv2.bitwise_or(clear_mask, source_mask)
            if cv2.countNonZero(current_text):
                current_text, current_diag = strip_border_ring_components(current_text, safe)
                masks.diagnostics.setdefault("current_text_border_ring_removed", current_diag)
            white_clear = cv2.bitwise_or(authority, current_text)
            if cv2.countNonZero(white_clear):
                white_clear = cv2.dilate(white_clear, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
                white_clear[safe == 0] = 0
                if cv2.countNonZero(safe):
                    white_clear, white_diag = strip_border_ring_components(white_clear, safe)
                    masks.diagnostics.setdefault("white_clear_border_ring_removed", white_diag)
                out = clear_to_target_paper(out, target, white_clear)
                all_clear = np.maximum(all_clear, white_clear)
        elif cv2.countNonZero(clear_mask) > 0:
            cleaned, clean_diag = clean_manual_target_text(
                target, clear_mask, bbox=as_list(row.get("target_bbox"))
            )
            sel = clear_mask > 0
            out[sel] = cleaned[sel]
            all_clear = np.maximum(all_clear, clear_mask)
            masks.diagnostics.setdefault("manual_target_cleanup", {}).update(clean_diag)

        alpha = source_mask.astype(np.float32) / 255.0
        feather = max(0, min(4, int(row.get("feather_px", 0) or 0)))
        if feather > 0 and cv2.countNonZero(source_mask) > 0 and mode != "full_patch":
            sigma = max(0.35, feather * 0.55)
            alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=sigma, sigmaY=sigma)
            alpha[source_mask > 0] = np.maximum(alpha[source_mask > 0], 0.92)
            alpha = np.clip(alpha, 0.0, 1.0)
        if np.any(alpha > 0):
            source_bg = estimate_source_background(masks.aligned_source, full_source_mask)
            out, delta_diag = composite_source_text_delta(
                out,
                masks.aligned_source,
                source_mask,
                source_background=source_bg,
                alpha=alpha,
            )
            # Keep an approximate editable layer for inspection/export.  The
            # final published render is the delta composite above; this layer is
            # only a reviewer aid and therefore may not fully reproduce the
            # darkening/lightening blend by itself.
            top = np.zeros_like(effect_layer)
            top[:, :, :3] = masks.aligned_source
            top[:, :, 3] = np.clip(alpha * 255.0, 0, 255).astype(np.uint8)
            effect_layer = _alpha_over_bgra(effect_layer, top)
            masks.diagnostics.setdefault("delta_composite", {}).update(delta_diag)

        diag = dict(masks.diagnostics)
        applied.append({
            "id": str(row.get("id", f"manual-effect-{index:03d}")),
            "success": bool(cv2.countNonZero(source_mask) > 0 or cv2.countNonZero(clear_mask) > 0),
            "mode": mode,
            "target_bbox": as_list(row.get("target_bbox")),
            "source_pixels": int(cv2.countNonZero(source_mask)),
            "target_clear_pixels": int(cv2.countNonZero(clear_mask)),
            "diagnostics": diag,
        })

    write_image(page_dir / "manual_effect_transfer_mask.png", effect_layer[:, :, 3])
    write_image(page_dir / "manual_effect_clear_mask.png", all_clear)
    _write_bgra(page_dir / "manual_effect_transfer_layer.png", effect_layer)
    return out, effect_layer, all_clear, applied


def _ensure_manual_effect_stable_base(page_dir: Path) -> Path:
    """Compatibility wrapper around the shared result-state contract."""
    return ensure_manual_baseline(page_dir)


def _manual_effect_overlay_base_path(page_dir: Path) -> Path:
    """Compatibility wrapper retained for older tests/plugins."""
    return manual_baseline_path(page_dir)


def _commit_reviewed_result(page_dir: Path, final_path: Path) -> Path:
    """Compatibility wrapper; all result synchronization is centralized."""
    return commit_reviewed_result(page_dir, final_path)


def _manual_effect_can_overlay_final(page_dir: Path, overrides: dict) -> bool:
    """True when manual omission repair is the only visual review operation.

    In this common workflow the already-rendered automatic ``final.png`` is the
    authoritative base.  Reconstructing the whole page from one transfer layer
    can drop successful replacements produced by another route/supplement.
    """
    if not _dict_rows(overrides.get("manual_effect_regions")):
        return False
    visual_keys = (
        "restore_target_bubbles", "manual_reletter", "text_overrides",
        "match_overrides", "unit_actions",
    )
    if any(bool(overrides.get(key)) for key in visual_keys):
        return False
    if (page_dir / "manual_clear_mask.png").exists():
        return False
    if (page_dir / "manual_transfer_mask.png").exists() or (page_dir / "manual_direct_patch_regions.png").exists():
        return False
    return True


def _apply_manual_effect_only_review(page_dir: Path, project: dict, overrides: dict, cfg: PipelineConfig) -> Path:
    """Manual recovery also works on pages where the automatic route passed through.

    This matters for a page containing only open/SFX text: the detector may emit
    zero speech bubbles, yet the reviewer can still box the missed text and finish
    the page without re-enabling OCR or rerunning the page pipeline.
    """
    target = read_image(page_dir / "target_original.png")
    base_path = _manual_effect_overlay_base_path(page_dir)
    base = read_image(base_path) if base_path.exists() else target.copy()
    rendered = base.copy()
    rendered, layer, clear_mask, applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    erase_extra = [layer[:, :, 3]] if layer is not None and layer.ndim == 3 and layer.shape[2] >= 4 else []
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    final_path = page_dir / "final_reviewed.png"
    write_image(final_path, rendered)
    # Preserve every pre-existing successful replacement in the flattened base.
    # Only the newly requested TARGET text-clear pixels are updated here.
    clean_base = base.copy()
    if cv2.countNonZero(clear_mask) > 0:
        cleaned = inpaint_image(target, clear_mask, cfg.inpainting).image
        clean_base[clear_mask > 0] = cleaned[clear_mask > 0]
    clean_path = page_dir / "review_base.png"
    write_image(clean_path, clean_base)
    transfer_path = page_dir / "manual_effect_transfer_layer.png"
    empty_text = make_text_layer_rgba(target.shape[:2], [], color=cfg.lettering.fill)
    text_path = page_dir / "text_layer_reviewed.png"
    write_rgba(text_path, empty_text)
    export_openraster(page_dir / "editable_reviewed.ora", target, clean_base, empty_text, cv2.cvtColor(layer, cv2.COLOR_BGRA2RGBA))
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", clean_path, text_path, transfer_path)
    save_json(page_dir / "review_applied.json", {
        "mode": "manual_effect_only",
        "status": overrides.get("status", "reviewed_with_manual_effect"),
        "manual_effect_applied": applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(clear_mask)),
        "target_layer_erase": target_erase_diag,
        "manual_effect_preview_patch_verified": bool(all(
            (not bool(x.get("preview_patch_applied"))) or bool(x.get("preview_patch_exact"))
            for x in applied if bool(x.get("success"))
        )),
        "manual_effect_base": str(base_path),
        "psd_exported": psd_ok,
        "final": str(final_path),
    })
    return final_path


def _apply_manual_reletters(rendered: np.ndarray, target: np.ndarray, page_dir: Path, project: dict, overrides: dict, cfg: PipelineConfig) -> tuple[np.ndarray, list[np.ndarray], list[dict]]:
    rows = _dict_rows(overrides.get("manual_reletter"))
    if not rows:
        return rendered, [], []
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    bubbles_by_id = {b.id: b for b in target_bubbles}
    meta = _dict_or_empty(project.get("meta"))
    direct_meta = _route_meta(meta, "direct_patch")
    active_meta = direct_meta if bool(direct_meta.get("used")) else _route_meta(meta, "mask_replace")
    manual_queue = _dict_rows(active_meta.get("manual_reletter_required"))
    queue_by_target = {str(x.get("target_bubble_id", "")): x for x in manual_queue if x.get("target_bubble_id")}
    out = rendered.copy()
    masks: list[np.ndarray] = []
    applied: list[dict] = []
    for i, row in enumerate(rows):
        text = str(row.get("text", "")).strip()
        if not text:
            continue
        bubble_id = str(row.get("target_bubble_id", "") or "")
        bubble = bubbles_by_id.get(bubble_id)
        safe = None
        polygon = None
        if bubble is not None:
            bubble_id = bubble.id
            polygon = bubble.polygon
            safe = bubble.safe_mask if bubble.safe_mask is not None and cv2.countNonZero(bubble.safe_mask) > 0 else bubble.mask
        bbox = row.get("target_bbox")
        if (not bbox) and bubble_id and bubble_id in queue_by_target:
            bbox = queue_by_target[bubble_id].get("target_bbox")
        if (safe is None or cv2.countNonZero(safe) == 0) and bbox:
            safe = _rect_mask(target.shape[:2], bbox, inset=3)
            x0, y0, x1, y1 = map(int, bbox)
            polygon = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        if safe is None or cv2.countNonZero(safe) == 0 or not polygon:
            continue
        out = _clear_region_to_paper(out, target, safe)
        orientation = str(row.get("orientation", "auto") or "auto")
        lcfg = cfg.lettering.model_copy(deep=True) if hasattr(cfg.lettering, 'model_copy') else cfg.lettering.copy(deep=True)
        lcfg.orientation = orientation
        unit = TextUnit(
            id=f"manual-reletter-{i:03d}",
            polygon=polygon,
            block_ids=[],
            text=text,
            confidence=1.0,
            kind=str(row.get("kind", "speech") or "speech"),
            reading_order=i,
            bubble_id=bubble_id,
            meta={"review_manual": True},
        )
        lr = fit_text(target.shape[:2], safe, unit, text, lcfg)
        if not lr.success or lr.text_mask is None:
            continue
        out = composite_text(out, lr, lcfg)
        masks.append(lr.text_mask)
        applied.append({
            "target_bubble_id": bubble_id or "",
            "text": text,
            "orientation": orientation,
            "bbox": list(lr.bbox),
        })
    return out, masks, applied


def _apply_mask_replace_review(page_dir: Path, project: dict, cfg: PipelineConfig) -> Path:
    target = read_image(page_dir / "target_original.png")
    overrides_probe = normalize_overrides(load_json(page_dir / "review_overrides.json") if (page_dir / "review_overrides.json").exists() else {})
    automatic_path = _manual_effect_overlay_base_path(page_dir) if _dict_rows(overrides_probe.get("manual_effect_regions")) else page_dir / "final.png"
    automatic = read_image(automatic_path) if automatic_path.exists() else None
    if automatic is not None and automatic.shape != target.shape:
        automatic = None
    review_change_mask = np.zeros(target.shape[:2], np.uint8)
    meta = _dict_or_empty(project.get("meta"))
    direct_meta = _route_meta(meta, "direct_patch")
    direct_used = bool(direct_meta.get("used"))
    route = "direct_patch" if direct_used else "mask_replace"
    layer_path = page_dir / ("direct_patch_layer.png" if direct_used else "mask_transfer_layer.png")
    if not layer_path.exists():
        raise FileNotFoundError(f"{layer_path.name} is missing")
    bgra = cv2.imread(str(layer_path), cv2.IMREAD_UNCHANGED)
    if bgra is None or bgra.ndim != 3 or bgra.shape[2] != 4:
        raise ValueError("mask_transfer_layer.png must be RGBA")
    if bgra.shape[:2] != target.shape[:2]:
        raise ValueError("mask transfer layer size mismatch")

    overrides_path = page_dir / "review_overrides.json"
    overrides = normalize_overrides(load_json(overrides_path) if overrides_path.exists() else {})
    transfer_meta = _route_meta(meta, route)
    review_queue = _dict_rows(transfer_meta.get("review_regions") or transfer_meta.get("manual_reletter_required"))
    queue_by_target = {str(x.get("target_bubble_id", "")): x for x in review_queue if x.get("target_bubble_id")}
    restore_ids = set(map(str, overrides.get("restore_target_bubbles", []) or []))
    accept_ids = set(map(str, overrides.get("accept_candidate_targets", []) or []))
    manual_rows = [x for x in _dict_rows(overrides.get("manual_reletter")) if str(x.get("text", "")).strip()]
    edit_ids = {str(x.get("target_bubble_id", "")) for x in manual_rows if x.get("target_bubble_id")}

    patch_bgr = bgra[:, :, :3]
    original_alpha = bgra[:, :, 3]
    manual = page_dir / ("manual_direct_patch_regions.png" if direct_used else "manual_transfer_mask.png")
    if manual.exists():
        m = cv2.imread(str(manual), cv2.IMREAD_GRAYSCALE)
        if m is None or m.shape != target.shape[:2]:
            raise ValueError("manual_transfer_mask.png has invalid dimensions")
        alpha = np.minimum(original_alpha, m)
        review_change_mask[alpha != original_alpha] = 255
    else:
        alpha = original_alpha.copy()

    # v0.8.25: the clear mask is an independent editable overlay. When present,
    # run only inpainting on that mask first; the Chinese transfer layer is
    # composited afterwards. This mirrors the detector -> mask -> remove -> write
    # separation used by mature comic-translation editors.
    effective_clear_raw, clear_source = _load_effective_clear_mask(page_dir, target.shape[:2])
    effective_clear, brush_diag = _apply_manual_brush_expansion(page_dir, effective_clear_raw)
    write_image(page_dir / "effective_clear_mask.png", effective_clear)
    manual_clear = page_dir / "manual_clear_mask.png"
    manual_japanese_clear = page_dir / "manual_japanese_clear_mask.png"
    manual_clear_present = manual_clear.exists() or manual_japanese_clear.exists()
    if manual_clear_present:
        auto_clear = np.zeros(target.shape[:2], np.uint8)
        for candidate in (page_dir / "target_clear_mask.png", page_dir / "clear_mask.png"):
            if not candidate.exists():
                continue
            probe = cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
            if probe is not None and probe.shape == target.shape[:2]:
                auto_clear = (probe > 0).astype(np.uint8) * 255
                break
        clear_delta = cv2.absdiff((effective_clear > 0).astype(np.uint8) * 255, auto_clear)
        if cv2.countNonZero(clear_delta) > 0:
            # Small halo covers antialiased glyph edges affected by local inpaint.
            clear_delta = cv2.dilate(clear_delta, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)
            review_change_mask = np.maximum(review_change_mask, clear_delta)
    residual_diag = {"target_dark_pixels": 0, "residual_dark_pixels": 0, "residual_dark_ratio": 0.0}
    if manual_clear_present and cv2.countNonZero(effective_clear) > 0:
        base = inpaint_image(target, effective_clear, cfg.inpainting).image
        heat, residual_diag = _residual_dark_heatmap(target, base, effective_clear)
        write_image(page_dir / "japanese_residual_heatmap.png", heat)
        # The original transfer layer alpha contains both Chinese glyphs and the
        # automatically cleared paper. When the user erases part of the clear
        # overlay, keep only real dark Chinese raster there; otherwise the old
        # white clear patch would silently override the manual erase.
        pgray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
        glyph_alpha = np.where((original_alpha > 0) & (pgray <= 232), original_alpha, 0).astype(np.uint8)
        alpha = np.maximum(glyph_alpha, np.minimum(alpha, effective_clear))
    else:
        base = target.copy()

    # Restore means the candidate disappears and the exact untouched Japanese
    # master becomes visible again. Edit also removes the candidate, but clears the
    # old Japanese text on a review base before drawing new HD Chinese text.
    for tid in sorted(restore_ids | edit_ids):
        row = queue_by_target.get(tid, {})
        box = row.get("target_bbox") or []
        if len(box) != 4:
            continue
        x0, y0, x1, y1 = map(int, box)
        x0=max(0,x0); y0=max(0,y0); x1=min(target.shape[1],x1); y1=min(target.shape[0],y1)
        if x1 <= x0 or y1 <= y0:
            continue
        review_change_mask[y0:y1, x0:x1] = 255
        alpha[y0:y1, x0:x1] = 0
        if tid in restore_ids:
            base[y0:y1, x0:x1] = target[y0:y1, x0:x1]
        elif tid in edit_ids:
            clear = _rect_mask(target.shape[:2], (x0,y0,x1,y1), inset=2)
            base = _clear_region_to_paper(base, target, clear)

    a = (alpha.astype(np.float32) / 255.0)[..., None]
    rendered = np.clip(patch_bgr.astype(np.float32) * a + base.astype(np.float32) * (1.0 - a), 0, 255).astype(np.uint8)
    if automatic is not None:
        # The automatic final may contain successful replacements from several
        # routes (Direct + supplemental Mask + safe lettering).  A review of one
        # local region must not flatten the whole page back to TARGET merely
        # because the chosen editable layer represents only one of those routes.
        untouched = review_change_mask == 0
        rendered[untouched] = automatic[untouched]
    final_path = page_dir / "final_reviewed.png"
    transfer_reviewed = page_dir / ("direct_patch_layer_reviewed.png" if direct_used else "mask_transfer_layer_reviewed.png")
    reviewed_base_path = page_dir / "review_base.png"
    write_image(reviewed_base_path, base)

    reviewed_bgra = bgra.copy(); reviewed_bgra[:, :, 3] = alpha
    ok, data = cv2.imencode(".png", reviewed_bgra)
    if not ok: raise ValueError("could not encode reviewed transfer layer")
    data.tofile(transfer_reviewed)

    rendered, manual_masks, manual_applied = _apply_manual_reletters(rendered, target, page_dir, project, overrides, cfg)
    rendered, effect_layer, effect_clear_mask, effect_applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    erase_extra = list(manual_masks)
    if effect_layer is not None and effect_layer.ndim == 3 and effect_layer.shape[2] >= 4:
        erase_extra.append(effect_layer[:, :, 3])
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    write_image(final_path, rendered)

    # Keep editable exports faithful to the reviewed result: target-only Japanese
    # strokes removed by a manual open-text region also disappear from the base,
    # while the recovered Chinese raster is merged into the transfer layer.
    if cv2.countNonZero(effect_clear_mask) > 0:
        effect_clean = inpaint_image(target, effect_clear_mask, cfg.inpainting).image
        base[effect_clear_mask > 0] = effect_clean[effect_clear_mask > 0]
        write_image(reviewed_base_path, base)
    merged_bgra = _alpha_over_bgra(reviewed_bgra, effect_layer) if np.any(effect_layer[:, :, 3] > 0) else reviewed_bgra
    _write_bgra(transfer_reviewed, merged_bgra)
    transfer_rgba = cv2.cvtColor(merged_bgra, cv2.COLOR_BGRA2RGBA)
    text_rgba = make_text_layer_rgba(target.shape[:2], manual_masks, color=cfg.lettering.fill)
    text_path = page_dir / "text_layer_reviewed.png"
    write_rgba(text_path, text_rgba)
    export_openraster(page_dir / "editable_reviewed.ora", target, base, text_rgba, transfer_rgba)
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", reviewed_base_path, text_path, transfer_reviewed)

    unresolved = [
        x for x in review_queue
        if str(x.get("target_bubble_id", "")) not in (restore_ids | accept_ids | edit_ids)
        and str(x.get("review_level", "required")) == "required"
    ]
    unreviewed_recommended = [
        x for x in review_queue
        if str(x.get("target_bubble_id", "")) not in (restore_ids | accept_ids | edit_ids)
        and str(x.get("review_level", "required")) != "required"
    ]
    save_json(page_dir / "review_applied.json", {
        "mode": route,
        "status": overrides.get("status", "reviewed"),
        "notes": overrides.get("notes", ""),
        "manual_transfer_mask": manual.exists(),
        "manual_clear_mask": manual_clear.exists(),
        "manual_japanese_clear_mask": manual_japanese_clear.exists(),
        "clear_mask_source": clear_source,
        "clear_brush": brush_diag,
        "target_layer_erase": target_erase_diag,
        "effective_clear_pixels": int(cv2.countNonZero(effective_clear)),
        "residual_dark_pixels": int(residual_diag.get("residual_dark_pixels", 0)),
        "residual_dark_ratio": float(residual_diag.get("residual_dark_ratio", 0.0)),
        "residual_review_recommended": bool(int(residual_diag.get("target_dark_pixels", 0)) >= 8 and float(residual_diag.get("residual_dark_ratio", 0.0)) > 0.08),
        "review_change_pixels": int(cv2.countNonZero(review_change_mask)),
        "automatic_final_preserved_outside_review": bool(automatic is not None),
        "manual_reletter_applied": manual_applied,
        "manual_effect_applied": effect_applied,
        "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
        "manual_effect_preview_patch_verified": bool(all(
            (not bool(x.get("preview_patch_applied"))) or bool(x.get("preview_patch_exact"))
            for x in effect_applied if bool(x.get("success"))
        )),
        "restored_targets": sorted(restore_ids),
        "accepted_candidate_targets": sorted(accept_ids),
        "unresolved_candidates": unresolved,
        "unreviewed_recommended": unreviewed_recommended,
        "psd_exported": psd_ok,
        "final": str(final_path),
    })
    return final_path


def rerun_page_with_force(page_dir: str | Path, mode: str, config: PipelineConfig | None = None) -> Path:
    """Re-run one page through an explicitly selected Direct/Mask route."""
    page_dir = Path(page_dir)
    if mode not in {"direct_patch", "mask_replace"}:
        raise ValueError(f"unsupported force mode: {mode}")
    project = normalize_project(load_json(page_dir / "project.json"))
    pair = PagePair(**project["pair"])
    cfg = (config or PipelineConfig()).model_copy(deep=True)
    cfg.transfer.mode = mode
    # Remove stale route-specific products before regenerating. This prevents a
    # failed Direct force from leaving an old Mask layer that Review could mistake
    # for the new result.
    for name in (
        "direct_patch_layer.png", "direct_patch_regions.png", "direct_patch.json",
        "mask_transfer_layer.png", "mask_transfer_mask.png", "mask_transfer.json",
        "direct_patch_layer_reviewed.png", "mask_transfer_layer_reviewed.png",
        "final_reviewed.png",
    ):
        path = page_dir / name
        if path.exists():
            path.unlink()
    final_path = None
    book_final = str(as_dict(project.get("artifacts")).get("book_final", "") or "").strip()
    if book_final:
        final_path = Path(book_final)
    from .pipeline import TransferPipeline
    regenerated = TransferPipeline(cfg).process_page(pair, page_dir, final_path=final_path)
    result = page_dir / "final.png"
    save_json(page_dir / "force_action_result.json", {
        "schema": "manga_hd_translation_transfer.force_action.v1",
        "action": f"force_{mode}",
        "passthrough": bool((regenerated.meta or {}).get("passthrough")),
        "planner": (regenerated.meta or {}).get("transfer_planner", {}),
        "final": str(result),
    })
    return result


def apply_review_page(page_dir: str | Path, config: PipelineConfig | None = None) -> Path:
    page_dir = Path(page_dir)
    cfg = config or PipelineConfig()
    override_path = page_dir / "review_overrides.json"
    overrides = normalize_overrides(load_json(override_path) if override_path.exists() else {})
    force_action = str(overrides.get("page_force_action", "") or "")
    if force_action in {"force_direct_patch", "force_mask_replace"}:
        forced_mode = "direct_patch" if force_action == "force_direct_patch" else "mask_replace"
        forced_final = rerun_page_with_force(page_dir, forced_mode, cfg)
        overrides["page_force_action_consumed"] = force_action
        overrides["page_force_action"] = ""
        save_json(override_path, overrides)
        project = normalize_project(load_json(page_dir / "project.json"))
        if bool(_dict_or_empty(project.get("meta")).get("passthrough")):
            return forced_final
    else:
        project = normalize_project(load_json(page_dir / "project.json"))
    manual_effect_rows = _dict_rows(overrides.get("manual_effect_regions"))
    if manual_effect_rows:
        _ensure_manual_effect_stable_base(page_dir)
    meta = _dict_or_empty(project.get("meta"))
    # Manual omission repair is an additive overlay.  When it is the only
    # visual review operation, never rebuild the whole page from a single
    # Direct/Mask layer; use the already-good automatic final as the base.
    if manual_effect_rows and (
        _manual_effect_can_overlay_final(page_dir, overrides)
        or bool(meta.get("passthrough"))
        or not ((page_dir / "direct_patch_layer.png").exists() or (page_dir / "mask_transfer_layer.png").exists())
    ):
        return _commit_reviewed_result(page_dir, _apply_manual_effect_only_review(page_dir, project, overrides, cfg))
    # Raster-review is the fast/default route for Auto/Mask/Direct pages, but an
    # actual unit-level text/match override means the reviewer explicitly wants
    # regenerated lettering. normalize_overrides() always inserts empty keys, so
    # route on *non-empty values* rather than key presence.
    unit_actions_probe = {str(k): str(v) for k, v in dict(overrides.get("unit_actions", {}) or {}).items()}
    unit_level_override = bool(dict(overrides.get("text_overrides", {}) or {})) or bool(dict(overrides.get("match_overrides", {}) or {})) or bool(list(overrides.get("accepted_source_units", []) or [])) or any(
        action in {"force_match", "skip_unit"} for action in unit_actions_probe.values()
    )
    if meta.get("transfer_mode") in {"mask_replace", "direct_patch", "auto"} and not unit_level_override:
        return _commit_reviewed_result(page_dir, _apply_mask_replace_review(page_dir, project, cfg))

    source_units = [_text_unit(x) for x in project.get("source_units", [])]
    target_units = [_text_unit(x) for x in project.get("target_units", [])]
    target_blocks = [_text_block(x) for x in project.get("target_blocks", [])]
    target_bubbles = _load_target_bubbles(page_dir, project.get("target_bubbles", []))
    target = read_image(page_dir / "target_original.png")

    source_by_id = {u.id: u for u in source_units}
    target_by_id = {u.id: u for u in target_units}
    bubbles_by_id = {b.id: b for b in target_bubbles}

    for source_id, text in dict(overrides.get("text_overrides", {})).items():
        if source_id in source_by_id:
            source_by_id[source_id].text = str(text)

    existing = {}
    for row in project.get("matches", []):
        if row.get("relation") == "one_to_one":
            existing[row["source_unit_id"]] = row["target_unit_id"]
    existing.update({str(k): str(v) for k, v in dict(overrides.get("match_overrides", {})).items()})

    if "accepted_source_units" in overrides:
        accepted_ids = set(map(str, overrides.get("accepted_source_units", [])))
    else:
        accepted_ids = {
            x.split("->", 1)[0]
            for x in _dict_or_empty(project.get("meta")).get("auto_applied_match_ids", [])
            if "->" in x
        }
    unit_actions = {str(k): str(v) for k, v in dict(overrides.get("unit_actions", {}) or {}).items()}
    for source_id, action in unit_actions.items():
        if action == "force_match":
            accepted_ids.add(source_id)
        elif action == "skip_unit":
            accepted_ids.discard(source_id)
    matches: list[UnitMatch] = []
    for source_id in accepted_ids:
        target_id = existing.get(source_id)
        if source_id in source_by_id and target_id in target_by_id:
            matches.append(UnitMatch(source_id, target_id, 1.0, 0.0, "one_to_one", ["review_accepted"]))

    manual_mask = page_dir / "manual_clear_mask.png"
    additive_mask = page_dir / "manual_japanese_clear_mask.png"
    if manual_mask.exists() or additive_mask.exists():
        mask, _clear_source = _load_effective_clear_mask(page_dir, target.shape[:2])
        mask, _brush_diag = _apply_manual_brush_expansion(page_dir, mask)
        write_image(page_dir / "effective_clear_mask.png", mask)
        from .masking import MaskBuildResult
        mask_result = MaskBuildResult((mask > 0).astype("uint8") * 255, {}, 0, int(cv2.countNonZero(mask)))
    else:
        mask_result = build_clear_mask(
            target.shape[:2], target_blocks, target_units, target_bubbles, matches, cfg.masking,
            min_match_confidence=0.0, allow_relations={"one_to_one"}, target_image=target
        )

    inpainted = inpaint_image(target, mask_result.mask, cfg.inpainting).image
    rendered = inpainted.copy()
    lettering = []
    masks = []
    for match in matches:
        src = source_by_id[match.source_unit_id]
        dst = target_by_id[match.target_unit_id]
        safe = bubbles_by_id.get(dst.bubble_id).safe_mask if dst.bubble_id and dst.bubble_id in bubbles_by_id else None
        if safe is None or cv2.countNonZero(safe) == 0:
            safe = polygon_safe_mask(dst, target.shape[:2], margin=max(2, cfg.bubbles.safe_margin_px // 2))
        lr = fit_text(target.shape[:2], safe, dst, src.text, cfg.lettering)
        lettering.append(lr)
        if lr.success and lr.text_mask is not None:
            rendered = composite_text(rendered, lr, cfg.lettering)
            masks.append(lr.text_mask)

    rendered, effect_layer, effect_clear_mask, effect_applied = _apply_manual_effect_regions(rendered, target, page_dir, project, overrides, cfg)
    if cv2.countNonZero(effect_clear_mask) > 0:
        effect_clean = inpaint_image(target, effect_clear_mask, cfg.inpainting).image
        inpainted[effect_clear_mask > 0] = effect_clean[effect_clear_mask > 0]
    erase_extra = list(masks)
    if effect_layer is not None and effect_layer.ndim == 3 and effect_layer.shape[2] >= 4:
        erase_extra.append(effect_layer[:, :, 3])
    rendered, target_erase_diag = _apply_target_layer_erase_to_rendered(
        page_dir, rendered, target, cfg, refresh_base=True, extra_protect_masks=erase_extra
    )
    final_path = page_dir / "final_reviewed.png"
    inpainted_path = page_dir / "inpainted_reviewed.png"
    text_path = page_dir / "text_layer_reviewed.png"
    write_image(final_path, rendered)
    write_image(inpainted_path, inpainted)
    text_rgba = make_text_layer_rgba(target.shape[:2], masks, color=cfg.lettering.fill)
    write_rgba(text_path, text_rgba)
    transfer_rgba = cv2.cvtColor(effect_layer, cv2.COLOR_BGRA2RGBA) if np.any(effect_layer[:, :, 3] > 0) else None
    export_openraster(page_dir / "editable_reviewed.ora", target, inpainted, text_rgba, transfer_rgba)
    transfer_path = page_dir / "manual_effect_transfer_layer.png" if transfer_rgba is not None else None
    psd_ok = export_psd_imagemagick(page_dir / "editable_reviewed.psd", page_dir / "target_original.png", inpainted_path, text_path, transfer_path)
    save_json(
        page_dir / "review_applied.json",
        {
            "status": overrides.get("status", "reviewed"),
            "notes": overrides.get("notes", ""),
            "accepted_source_units": sorted(accepted_ids),
            "matches": [m.to_dict() for m in matches],
            "lettering": [x.to_dict() for x in lettering],
            "manual_mask": manual_mask.exists(),
            "manual_effect_applied": effect_applied,
            "manual_effect_clear_pixels": int(cv2.countNonZero(effect_clear_mask)),
            "target_layer_erase": target_erase_diag,
            "psd_exported": psd_ok,
            "final": str(final_path),
        },
    )
    return _commit_reviewed_result(page_dir, final_path)
