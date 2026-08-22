from __future__ import annotations

"""TARGET-layer erase/restore review service.

This service owns the reviewer operation that removes residual Japanese or
restores original TARGET pixels.  It is independent from the large review
application dispatcher and remains protected by page-level write guards.
"""

from pathlib import Path

import cv2
import numpy as np

from .config import PipelineConfig
from .inpainting import inpaint_image
from .io_utils import load_json, read_image, save_json, write_image
from .result_state import commit_reviewed_result, resolve_result_state
from .schema_compat import as_dict
from .workspace_guard import guarded_page_write
from .review_clear_mask import safe_automatic_clear_seed

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
    for name in ('text_layer_reviewed.png', 'manual_effect_transfer_layer.png', 'manual_force_transfer_layer.png'):
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


def _load_binary_mask(path: Path, shape: tuple[int, int]) -> np.ndarray | None:
    if not path.exists():
        return None
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.shape != shape:
        return None
    return (mask > 0).astype(np.uint8) * 255


def _subtract_mask_file(path: Path, remove_mask: np.ndarray) -> dict:
    current = _load_binary_mask(path, remove_mask.shape)
    if current is None:
        return {'path': path.name, 'changed': False, 'reason': 'missing_or_invalid'}
    before = int(cv2.countNonZero(current))
    current[remove_mask > 0] = 0
    after = int(cv2.countNonZero(current))
    write_image(path, current)
    return {'path': path.name, 'changed': before != after, 'before_pixels': before, 'after_pixels': after, 'removed_pixels': before - after}


def _project_restore_back_into_masks(page_dir: Path, effective: np.ndarray) -> dict:
    """Persist TARGET-restore intent back into the driving masks.

    A reviewer uses TARGET restore to say "this area should stay Japanese".
    Merely covering the final image is not enough, because a later re-apply would
    bring Chinese back.  Therefore the restore footprint is subtracted from the
    relevant clear/force masks.  If there is no explicit manual clear override
    yet, one is materialized from the current automatic clear artifact with the
    restored area removed.
    """
    if cv2.countNonZero(effective) == 0:
        return {'used': False, 'reason': 'empty_restore'}
    shape = effective.shape
    touched: list[dict] = []
    # Existing explicit override masks should lose the restored pixels.
    for name in ('manual_force_auto_target_override.png', 'manual_clear_mask.png', 'manual_japanese_clear_mask.png', 'manual_force_transfer_mask.png'):
        touched.append(_subtract_mask_file(page_dir / name, effective))

    # If no manual clear override exists yet, materialize one from the effective
    # automatic clear artifact so future remove/apply reruns keep this restore.
    manual_clear = page_dir / 'manual_clear_mask.png'
    if not manual_clear.exists():
        seeded, seed_diag = safe_automatic_clear_seed(page_dir, shape)
        if seeded is not None:
            before = int(cv2.countNonZero(seeded))
            seeded[effective > 0] = 0
            write_image(manual_clear, seeded)
            touched.append({
                'path': manual_clear.name,
                'changed': True,
                'created_from': str(seed_diag.get('source') or 'automatic_clear'),
                'automatic_projection': seed_diag,
                'before_pixels': before,
                'after_pixels': int(cv2.countNonZero(seeded)),
                'removed_pixels': before - int(cv2.countNonZero(seeded)),
            })

    return {
        'used': True,
        'effective_restore_pixels': int(cv2.countNonZero(effective)),
        'touched_masks': touched,
    }


def _apply_target_layer_restore_to_rendered(
    page_dir: Path,
    rendered: np.ndarray,
    target: np.ndarray,
    *,
    refresh_base: bool = False,
) -> tuple[np.ndarray, dict]:
    """Restore TARGET pixels in explicit reviewer-selected regions.

    This is the inverse of TARGET-only erase: when the automatic transfer showed
    Chinese in a region that should stay Japanese, the reviewer paints that
    region and the original TARGET pixels are restored exactly.
    """
    mask_path = page_dir / 'manual_target_layer_restore_mask.png'
    if not mask_path.exists():
        return rendered.copy(), {'used': False, 'reason': 'mask_missing'}
    raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if raw is None or raw.shape != target.shape[:2]:
        raise ValueError('manual_target_layer_restore_mask.png has invalid dimensions')
    raw = (raw > 0).astype(np.uint8) * 255
    if refresh_base:
        write_image(page_dir / 'target_layer_restore_base.png', rendered)
    settings_path = page_dir / 'target_layer_restore_settings.json'
    settings = as_dict(load_json(settings_path)) if settings_path.exists() else {}
    dilate_px = max(0, min(4, int(settings.get('dilate_px', 0) or 0)))
    effective = raw.copy()
    if cv2.countNonZero(effective) and dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px * 2 + 1, dilate_px * 2 + 1))
        effective = cv2.dilate(effective, k, iterations=1)
    out = rendered.copy()
    sel = effective > 0
    out[sel] = target[sel]
    persist_diag = _project_restore_back_into_masks(page_dir, effective)
    write_image(page_dir / 'target_layer_restore_effective_mask.png', effective)
    write_image(page_dir / 'target_layer_restore_preview.png', out)
    diag = {
        'schema': 'manga_hd_translation_transfer.target_layer_restore.v1',
        'used': bool(cv2.countNonZero(raw)),
        'raw_mask_pixels': int(cv2.countNonZero(raw)),
        'effective_restore_pixels': int(cv2.countNonZero(effective)),
        'dilate_px': int(dilate_px),
        'background_policy': 'target_original_exact_restore',
        'chinese_pixel_policy': 'allow_replace_with_target',
        'mask_back_projection': persist_diag,
    }
    save_json(page_dir / 'target_layer_restore.json', diag)
    return out, diag


@guarded_page_write("target_layer_erase")
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


@guarded_page_write("target_layer_erase_reset")
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


@guarded_page_write("target_layer_restore")
def apply_target_layer_restore_review(page_dir: str | Path) -> Path:
    """Fast final-stage TARGET restore without rerunning OCR/registration."""
    page_dir = Path(page_dir)
    target = read_image(page_dir / 'target_original.png')
    base_path = page_dir / 'target_layer_restore_base.png'
    if base_path.exists():
        base = read_image(base_path)
    else:
        state = resolve_result_state(page_dir)
        source = state.current or (page_dir / 'final.png')
        if not Path(source).exists():
            raise FileNotFoundError('current final result is missing')
        base = read_image(source)
        write_image(base_path, base)
    rendered, diag = _apply_target_layer_restore_to_rendered(page_dir, base, target, refresh_base=False)
    final_path = page_dir / 'final_reviewed.png'
    write_image(final_path, rendered)
    diag['final'] = str(final_path)
    save_json(page_dir / 'target_layer_restore.json', diag)
    return commit_reviewed_result(page_dir, final_path)


@guarded_page_write("target_layer_restore_reset")
def reset_target_layer_restore_review(page_dir: str | Path) -> Path | None:
    """Remove the dedicated TARGET restore mask and restore its pre-restore base."""
    page_dir = Path(page_dir)
    base_path = page_dir / 'target_layer_restore_base.png'
    restored: Path | None = None
    if base_path.exists():
        base = read_image(base_path)
        restored = page_dir / 'final_reviewed.png'
        write_image(restored, base)
    for name in (
        'manual_target_layer_restore_mask.png', 'target_layer_restore_effective_mask.png',
        'target_layer_restore_preview.png', 'target_layer_restore.json',
        'target_layer_restore_settings.json', 'target_layer_restore_base.png',
    ):
        try:
            (page_dir / name).unlink(missing_ok=True)
        except OSError:
            pass
    return commit_reviewed_result(page_dir, restored) if restored is not None else None


__all__ = ['_read_layer_alpha', '_protected_chinese_mask', '_clean_target_under_erase_mask', '_apply_target_layer_erase_to_rendered', '_apply_target_layer_restore_to_rendered', 'apply_target_layer_erase_review', 'reset_target_layer_erase_review', 'apply_target_layer_restore_review', 'reset_target_layer_restore_review']
