from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import tempfile
import time
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .config import OCRConfig
from .model_downloads import discovered_paddle_model_dirs
from .geometry import bbox_polygon, polygon_centroid
from .models import TextBlock
from .ocr_base import OCRBackend

logger = logging.getLogger(__name__)


class NullOCRBackend(OCRBackend):
    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        return []


class SidecarOCRBackend(OCRBackend):
    supports_crop_recognition = False
    supports_rectified_input = False

    """Reads deterministic OCR data from `<image>.ocr.json` or a configured sidecar.

    Accepted payloads:
      {"blocks": [{"polygon": [[x,y],...], "text": "...", "confidence": .99}]}
      [{...}, {...}]
    """

    def __init__(self, suffix: str = ".ocr.json") -> None:
        self.suffix = suffix

    def sidecar_path(self, image_path: str | Path) -> Path:
        p = Path(image_path)
        # page.png -> page.ocr.json
        if self.suffix.startswith("."):
            return p.with_suffix(self.suffix)
        return p.parent / f"{p.stem}{self.suffix}"

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if image_path is None:
            raise ValueError("SidecarOCRBackend requires image_path")
        sidecar = self.sidecar_path(image_path)
        if not sidecar.exists():
            raise FileNotFoundError(f"OCR sidecar not found: {sidecar}")
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        rows = payload.get("blocks", []) if isinstance(payload, dict) else payload
        blocks: list[TextBlock] = []
        for i, row in enumerate(rows):
            poly = row.get("polygon")
            if not poly and row.get("bbox"):
                poly = bbox_polygon(row["bbox"])
            if not poly:
                continue
            blocks.append(
                TextBlock(
                    id=str(row.get("id") or f"ocr-{i:04d}"),
                    polygon=[(float(x), float(y)) for x, y in poly],
                    text=str(row.get("text", "")),
                    confidence=float(row.get("confidence", 1.0)),
                    kind=str(row.get("kind", "unknown")),
                    reading_order=int(row.get("reading_order", i)),
                    bubble_id=row.get("bubble_id"),
                    meta=dict(row.get("meta", {})),
                )
            )
            mask_path = row.get("mask_path") or blocks[-1].meta.get("mask_path")
            if mask_path:
                mp = Path(mask_path)
                if not mp.is_absolute():
                    mp = sidecar.parent / mp
                blocks[-1].meta["mask_path"] = str(mp)
        return blocks


def _find_ocr_dict(obj: Any) -> dict[str, Any] | None:
    if isinstance(obj, dict):
        if "rec_texts" in obj and ("rec_polys" in obj or "dt_polys" in obj or "rec_boxes" in obj):
            return obj
        for v in obj.values():
            found = _find_ocr_dict(v)
            if found is not None:
                return found
    return None


def _result_to_dict(res: Any) -> dict[str, Any]:
    candidates = []
    for attr in ("json", "res", "result"):
        if hasattr(res, attr):
            value = getattr(res, attr)
            try:
                value = value() if callable(value) else value
            except Exception:
                pass
            candidates.append(value)
    if hasattr(res, "to_dict"):
        try:
            candidates.append(res.to_dict())
        except Exception:
            pass
    candidates.append(res)
    for candidate in candidates:
        found = _find_ocr_dict(candidate)
        if found is not None:
            return found
    raise ValueError(f"Unsupported PaddleOCR result structure: {type(res)!r}")




def _tesseract_language(lang: str) -> str:
    key = str(lang or '').strip().lower()
    if key in {'japan', 'ja', 'jp', 'japanese', 'ja_jp'}:
        return 'Japanese_vert+Japanese'
    if key in {'ch', 'zh', 'zh-hant', 'zh_hant', 'cht', 'zht', 'traditional_chinese', 'chinese_traditional'}:
        return 'chi_tra_vert+chi_tra'
    if key in {'zh-hans', 'zh_hans', 'chs', 'zh-cn', 'zh_cn', 'simplified_chinese', 'chinese_simplified'}:
        return 'chi_sim_vert+chi_sim'
    return 'Japanese_vert+Japanese'


class TesseractOCRBackend(OCRBackend):
    """Lightweight manga OCR fallback for cloud/headless runtimes.

    The fallback deliberately separates whole-page OCR from manga-region OCR.
    Vertical Chinese/Japanese glyphs remain upright and are recognized with the
    native ``*_vert`` traineddata (PSM 5). Before recognition, border/large-art
    components are removed from the crop and a tight ink ROI is used. Tesseract's
    line order on vertical data is already right-to-left by column. If the whole
    ROI misses one or more visible columns, only then are column bands retried.
    """

    supports_crop_recognition = True
    supports_region_query = False
    supports_rectified_input = True
    retry_crops = False
    region_text_only = True

    def __init__(self, lang: str, config: OCRConfig | None = None) -> None:
        self.config = config or OCRConfig()
        self.lang = str(lang)
        self.tess_lang = _tesseract_language(lang)
        key = str(lang or '').strip().lower()
        if key in {'japan', 'ja', 'jp', 'japanese', 'ja_jp'}:
            self.vertical_lang, self.horizontal_lang = 'Japanese_vert', 'Japanese'
        elif key in {'ch', 'zh', 'zh-hant', 'zh_hant', 'cht', 'zht', 'traditional_chinese', 'chinese_traditional'}:
            self.vertical_lang, self.horizontal_lang = 'chi_tra_vert', 'chi_tra'
        elif key in {'zh-hans', 'zh_hans', 'chs', 'zh-cn', 'zh_cn', 'simplified_chinese', 'chinese_simplified'}:
            self.vertical_lang, self.horizontal_lang = 'chi_sim_vert', 'chi_sim'
        else:
            self.vertical_lang = self.tess_lang.split('+')[0]
            self.horizontal_lang = self.tess_lang.split('+')[-1]
        try:
            import pytesseract  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f'Tesseract backend unavailable: {exc}') from exc
        self._pytesseract = pytesseract

    @staticmethod
    def _has_cjk(value: str) -> bool:
        return any(
            '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u9fff'
            for ch in str(value or '')
        )

    @staticmethod
    def _clean_token(value: str) -> str:
        import re
        value = str(value or '').strip()
        if not value:
            return ''
        # For CJK fallback OCR, ASCII bars/slashes/Latin fragments are usually
        # bubble borders or screentone noise. Keep punctuation and CJK content.
        if TesseractOCRBackend._has_cjk(value):
            value = re.sub(r'[^\u3040-\u30ff\u3400-\u9fff，。！？：；、…—～・「」『』（）()!?]', '', value)
            return value
        if all(ch in '，。！？：；、…—～・「」『』（）()!?' for ch in value):
            return value
        return ''

    @staticmethod
    def _tight_ink_roi(image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        if image is None or image.size == 0:
            return image, (0, 0)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        ink = (gray < 190).astype(np.uint8)
        border = max(2, int(round(min(h, w) * 0.04)))
        if h > border * 2 and w > border * 2:
            ink[:border, :] = 0; ink[-border:, :] = 0
            ink[:, :border] = 0; ink[:, -border:] = 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        keep = np.zeros_like(ink)
        page_area = float(max(1, h * w))
        for i in range(1, n):
            x, y, cw, ch, area = [int(v) for v in stats[i]]
            if area < max(3, int(page_area * 0.00002)):
                continue
            if area > page_area * 0.08 or cw > w * 0.50 or ch > h * 0.55:
                continue
            keep[labels == i] = 1
        ys, xs = np.where(keep > 0)
        if xs.size == 0:
            return image, (0, 0)
        pad = max(4, min(10, int(round(min(h, w) * 0.035))))
        x0 = max(0, int(xs.min()) - pad); x1 = min(w, int(xs.max()) + 1 + pad)
        y0 = max(0, int(ys.min()) - pad); y1 = min(h, int(ys.max()) + 1 + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return image, (0, 0)
        return image[y0:y1, x0:x1].copy(), (x0, y0)

    @staticmethod
    def _glyph_keep_mask(image: np.ndarray) -> np.ndarray:
        if image is None or image.size == 0:
            return np.zeros((0, 0), dtype=np.uint8)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        ink = (gray < 190).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, 8)
        keep = np.zeros_like(ink)
        area_total = float(max(1, h * w))
        for i in range(1, n):
            x, y, cw, ch, area = [int(v) for v in stats[i]]
            if area < 3 or area > area_total * 0.08 or cw > w * 0.50 or ch > h * 0.55:
                continue
            keep[labels == i] = 1
        return keep

    @staticmethod
    def _vertical_column_bands(image: np.ndarray) -> list[tuple[int, int]]:
        if image is None or image.size == 0:
            return []
        h, w = image.shape[:2]
        keep = TesseractOCRBackend._glyph_keep_mask(image)
        if keep.size == 0:
            return []
        proj = keep.sum(axis=0).astype(np.float32)
        if not np.any(proj > 0):
            return []
        smooth = np.convolve(proj, np.ones(5, np.float32) / 5.0, mode='same')
        positive = smooth[smooth > 0]
        threshold = max(1.0, float(np.percentile(positive, 30.0)) * 0.35) if positive.size else 1.0
        active = smooth >= threshold
        bands: list[list[int]] = []
        i = 0
        while i < w:
            if not active[i]:
                i += 1; continue
            j = i + 1
            while j < w and active[j]:
                j += 1
            bands.append([i, j]); i = j
        gap = max(4, int(round(w * 0.025)))
        merged: list[list[int]] = []
        for band in bands:
            if merged and band[0] - merged[-1][1] <= gap:
                merged[-1][1] = band[1]
            else:
                merged.append(band)
        # Components glued to the exact crop edge are commonly bubble borders or
        # panel art rather than text. Keep near-edge real columns (x=5 etc.), but
        # reject bands that literally touch the crop boundary.
        return [(a, b) for a, b in merged if b - a >= 5 and a > 0 and b < w]

    @staticmethod
    def _component_column_bands(image: np.ndarray) -> list[tuple[int, int]]:
        """Recover vertical text columns from glyph-like connected components.

        Projection bands are fast but can merge neighbouring CJK columns when
        punctuation or antialiasing bridges the gap.  Component centres are a
        second, independent geometry cue.  This is OCR-only geometry and never
        changes publication masks.
        """
        if image is None or image.size == 0:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        ink = (gray < 185).astype(np.uint8)
        n, _labels, stats, centroids = cv2.connectedComponentsWithStats(ink, 8)
        comps = []
        page_area = float(max(1, h * w))
        for i in range(1, n):
            x, y, cw, ch, area = [int(v) for v in stats[i]]
            cx, _cy = centroids[i]
            if area < 12 or area > page_area * 0.025:
                continue
            if cw < 2 or ch < 3 or cw > w * 0.30 or ch > h * 0.35:
                continue
            if max(cw / max(1.0, ch), ch / max(1.0, cw)) > 4.5 and area > 80:
                continue
            comps.append({"cx": float(cx), "x0": x, "x1": x + cw, "w": cw})
        if len(comps) < 4:
            return []
        median_w = float(np.median([c["w"] for c in comps]))
        tolerance = max(8.0, median_w * 1.10)
        comps.sort(key=lambda c: c["cx"], reverse=True)
        groups: list[list[dict]] = []
        for comp in comps:
            if not groups:
                groups.append([comp]); continue
            center = float(np.mean([row["cx"] for row in groups[-1]]))
            if abs(comp["cx"] - center) > tolerance:
                groups.append([comp])
            else:
                groups[-1].append(comp)
        bands = []
        for group in groups:
            if len(group) < 3:
                continue
            x0 = max(0, min(int(c["x0"]) for c in group) - 4)
            x1 = min(w, max(int(c["x1"]) for c in group) + 4)
            if x1 - x0 >= 6 and (x1 - x0) <= max(18.0, median_w * 2.05):
                bands.append((x0, x1, len(group)))
        # Remove a broad low-support band when a denser band sits inside it.
        kept: list[tuple[int, int, int]] = []
        for band in bands:
            a, b, support = band
            dominated = False
            for oa, ob, osupport in bands:
                if (oa, ob, osupport) == band or osupport <= support:
                    continue
                inter = max(0, min(b, ob) - max(a, oa))
                if inter / max(1, b - a) >= 0.45 and (b - a) > (ob - oa) * 1.35:
                    dominated = True; break
            if not dominated:
                kept.append(band)
        kept.sort(key=lambda row: row[0], reverse=True)
        return [(a, b) for a, b, _support in kept if a > 0 and b < w]

    @staticmethod
    def _align_candidate_to_reference(reference: str, candidate: str):
        """Align one OCR candidate onto a reference without inventing glyphs."""
        import difflib
        reference = str(reference or "")
        candidate = str(candidate or "")
        mapped = [""] * len(reference)
        insertions = [""] * (len(reference) + 1)
        matcher = difflib.SequenceMatcher(None, reference, candidate, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                for off in range(i2 - i1):
                    mapped[i1 + off] = candidate[j1 + off]
            elif tag == "replace":
                common = min(i2 - i1, j2 - j1)
                for off in range(common):
                    mapped[i1 + off] = candidate[j1 + off]
                if j2 - j1 > common:
                    insertions[i1 + common] += candidate[j1 + common:j2]
            elif tag == "insert":
                insertions[i1] += candidate[j1:j2]
        return mapped, insertions

    @staticmethod
    def _vote_insertion_prefix(values: list[str], *, min_routes: int) -> str:
        """Keep only insertion glyphs independently supported by multiple routes."""
        from collections import Counter
        active = [str(v or "") for v in values if v]
        out = []
        while active:
            counts = Counter(v[0] for v in active if v)
            if not counts:
                break
            char, count = counts.most_common(1)[0]
            if int(count) < int(min_routes):
                break
            out.append(char)
            active = [v[1:] for v in active if len(v) > 1 and v[0] == char]
        return "".join(out)

    @staticmethod
    def _consensus_text(variants: list[tuple[str, float, str]]):
        """Conservative character-level consensus for one vertical OCR column.

        A glyph is accepted only when at least two independent recognition routes
        support it.  This fixes substitutions such as 戴/載 without using a
        vocabulary or silently guessing a disputed character.
        """
        import difflib
        from collections import Counter

        rows = [(str(t or ""), float(c), str(l or "")) for t, c, l in variants if str(t or "")]
        if not rows:
            return "", 0.0, 0, 0.0, ""
        if len(rows) == 1:
            text, conf, lang = rows[0]
            return text, conf, 1, 1.0, lang

        # Medoid string: agreement first, confidence/vertical route only break ties.
        best_idx, best_score = 0, -1.0
        for i, (text, conf, lang) in enumerate(rows):
            sims = [
                difflib.SequenceMatcher(None, text, other, autojunk=False).ratio()
                for j, (other, _c, _l) in enumerate(rows) if j != i
            ]
            mean_sim = float(sum(sims) / max(1, len(sims)))
            score = mean_sim + min(1.0, max(0.0, conf)) * 0.08 + (0.025 if "vert" in lang.lower() else 0.0)
            if score > best_score:
                best_idx, best_score = i, score
        reference = rows[best_idx][0]
        aligned = []
        for text, conf, lang in rows:
            mapped, insertions = TesseractOCRBackend._align_candidate_to_reference(reference, text)
            aligned.append((mapped, insertions, conf, lang))

        min_routes = 2
        output = []
        supported_positions = 0
        participating_routes: set[int] = set()

        def choose_char(position: int) -> str:
            votes = []
            for route_idx, (mapped, _ins, conf, lang) in enumerate(aligned):
                ch = mapped[position] if position < len(mapped) else ""
                if not ch:
                    continue
                weight = 1.0 + float(conf) * 0.20 + (0.04 if "vert" in lang.lower() else 0.0)
                votes.append((ch, route_idx, weight))
            counts = Counter(ch for ch, _idx, _weight in votes)
            if not counts:
                return ""
            max_count = max(counts.values())
            if max_count < min_routes:
                return ""
            tied = {ch for ch, count in counts.items() if count == max_count}
            chosen = max(tied, key=lambda ch: sum(w for c, _idx, w in votes if c == ch))
            for ch, route_idx, _weight in votes:
                if ch == chosen:
                    participating_routes.add(route_idx)
            return chosen

        before = TesseractOCRBackend._vote_insertion_prefix(
            [ins[0] for _mapped, ins, _conf, _lang in aligned], min_routes=min_routes,
        )
        if before:
            output.append(before)
        for pos in range(len(reference)):
            chosen = choose_char(pos)
            if chosen:
                output.append(chosen)
                supported_positions += 1
            suffix = TesseractOCRBackend._vote_insertion_prefix(
                [ins[pos + 1] for _mapped, ins, _conf, _lang in aligned], min_routes=min_routes,
            )
            if suffix:
                output.append(suffix)

        text = "".join(output)
        coverage = supported_positions / max(1, len(reference))
        route_count = len(participating_routes)
        if route_count:
            conf = float(sum(rows[i][1] for i in participating_routes) / route_count)
            langs = "+".join(rows[i][2] for i in sorted(participating_routes))
        else:
            conf, langs = 0.0, ""
        return text, conf, route_count, float(coverage), langs

    def _column_consensus(self, image: np.ndarray, *, offset=(0, 0), deadline: float | None = None) -> TextBlock | None:
        """Choose a column reading only when independent Tesseract routes agree."""
        variants = []
        alt_vertical = 'HanT_vert' if self.vertical_lang == 'chi_tra_vert' else self.vertical_lang
        alt_horizontal = 'HanT' if self.horizontal_lang == 'chi_tra' else self.horizontal_lang
        # PSM 5 is the main vertical-column route, but narrow manga/burst
        # columns are often materially more stable under PSM 12. Keep both
        # vertical segmenters in the consensus pool, then use horizontal PSM 6
        # only as an independent tie-break. This remains column-local and is
        # invoked only for suspicious columns, so it does not turn into a
        # whole-page OCR multiplier.
        for lang, psm in (
            (self.vertical_lang, 5),
            (alt_vertical, 5),
            (self.vertical_lang, 12),
            (alt_vertical, 12),
            (self.horizontal_lang, 6),
            (alt_horizontal, 6),
        ):
            if deadline is not None and time.monotonic() >= deadline:
                break
            try:
                rows = self._run_lines(image, lang=lang, psm=psm, offset=offset, deadline=deadline)
            except Exception:
                continue
            rows = sorted(rows, key=lambda r: (float(r.bbox[1]), float(r.bbox[0])))
            text = self._clean_token(''.join(r.text for r in rows if self._has_cjk(r.text)))
            cjk = sum(1 for ch in text if self._has_cjk(ch))
            if cjk < 2:
                continue
            weights = [max(1, sum(1 for ch in r.text if self._has_cjk(ch))) for r in rows]
            conf = float(np.average([r.confidence for r in rows], weights=weights)) if rows else 0.0
            variants.append((text, conf, lang))
        if not variants:
            return None
        if len(variants) > 1:
            import difflib
            def _pairwise(rows):
                values = [
                    difflib.SequenceMatcher(None, rows[i][0], rows[j][0], autojunk=False).ratio()
                    for i in range(len(rows)) for j in range(i + 1, len(rows))
                ]
                return float(sum(values) / max(1, len(values))) if values else 1.0
            pairwise_agreement_all = _pairwise(variants)
            vertical_variants = [row for row in variants if 'vert' in str(row[2]).lower()]
            # Vertical manga OCR is the primary evidence. Horizontal routes are
            # included only as character tie-breakers and must not be allowed to
            # veto a strong agreement among independent vertical segmenters.
            pairwise_agreement = (
                _pairwise(vertical_variants) if len(vertical_variants) >= 2
                else pairwise_agreement_all
            )
        else:
            pairwise_agreement_all = 1.0
            pairwise_agreement = 1.0
        if len(variants) == 1:
            text, conf, lang = variants[0]
            if conf < 0.72:
                return None
            box = (offset[0], offset[1], offset[0] + image.shape[1], offset[1] + image.shape[0])
            return TextBlock(
                id='tesseract-column-consensus', polygon=bbox_polygon(box), text=text,
                confidence=conf, kind='unknown', reading_order=0, bubble_id=None,
                meta={'backend':'tesseract','tesseract_lang':lang,'column_consensus':True,
                      'consensus_routes':1,'consensus_char_coverage':1.0,'consensus_variant_count':1,
                      'consensus_pairwise_agreement':pairwise_agreement,
                      'consensus_pairwise_agreement_all_routes':pairwise_agreement_all},
            )

        text, conf, route_count, coverage, langs = self._consensus_text(variants)
        text = self._clean_token(text)
        if sum(1 for ch in text if self._has_cjk(ch)) < 2:
            return None
        box = (offset[0], offset[1], offset[0] + image.shape[1], offset[1] + image.shape[0])
        return TextBlock(
            id='tesseract-column-consensus', polygon=bbox_polygon(box), text=text,
            confidence=float(conf), kind='unknown', reading_order=0, bubble_id=None,
            meta={'backend':'tesseract','tesseract_lang':langs,'column_consensus':True,
                  'consensus_routes':int(route_count),'consensus_char_coverage':float(coverage),
                  'consensus_variant_count':len(variants),
                  'consensus_pairwise_agreement':pairwise_agreement,
                  'consensus_pairwise_agreement_all_routes':pairwise_agreement_all},
        )

    def _run_lines(self, image: np.ndarray, *, lang: str, psm: int, offset=(0, 0), deadline: float | None = None) -> list[TextBlock]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.ndim == 3 else image
        try:
            call_timeout = max(0.75, float(getattr(self.config, 'tesseract_call_timeout_seconds', 3.0)))
            if deadline is not None:
                remaining = float(deadline) - time.monotonic()
                if remaining <= 0.20:
                    return []
                call_timeout = max(0.20, min(call_timeout, remaining))
            data = self._pytesseract.image_to_data(
                rgb, lang=lang,
                config=f'--oem 1 --psm {int(psm)} -c preserve_interword_spaces=0',
                output_type=self._pytesseract.Output.DICT,
                timeout=call_timeout,
            )
        except RuntimeError:
            return []
        n = len(data.get('text', []))
        groups: dict[tuple[int, int, int], list[tuple[int, str, float, tuple[int,int,int,int]]]] = {}
        for i in range(n):
            raw = str((data.get('text') or [''])[i] or '').strip()
            token = self._clean_token(raw)
            if not token:
                continue
            try:
                conf = float((data.get('conf') or ['-1'])[i])
            except Exception:
                conf = -1.0
            x = int((data.get('left') or [0])[i]); y = int((data.get('top') or [0])[i])
            ww = int((data.get('width') or [0])[i]); hh = int((data.get('height') or [0])[i])
            if ww <= 1 or hh <= 1:
                continue
            key = (
                int((data.get('block_num') or [0])[i] or 0),
                int((data.get('par_num') or [0])[i] or 0),
                int((data.get('line_num') or [0])[i] or 0),
            )
            word = int((data.get('word_num') or [0])[i] or 0)
            groups.setdefault(key, []).append((word, token, conf, (x,y,x+ww,y+hh)))
        ox, oy = [int(v) for v in offset]
        rows = []
        for key, words in groups.items():
            words.sort(key=lambda z: z[0])
            text = ''.join(z[1] for z in words)
            if not text:
                continue
            # Keep CJK tokens even at low confidence when they belong to a line
            # with strong neighbours; the geometry is already a verified manga
            # text region and dropping a single low-confidence glyph creates worse
            # omissions than retaining it for review.
            confs = [z[2] for z in words if z[2] >= 0]
            conf = float(sum(confs) / max(1, len(confs))) if confs else 55.0
            x0=min(z[3][0] for z in words)+ox; y0=min(z[3][1] for z in words)+oy
            x1=max(z[3][2] for z in words)+ox; y1=max(z[3][3] for z in words)+oy
            rows.append((key, text, conf, (x0,y0,x1,y1)))
        # Tesseract PSM 5 line numbers already follow right-to-left columns.
        rows.sort(key=lambda z: (z[0][0], z[0][1], z[0][2], -z[3][0], z[3][1]))
        out=[]
        for order, (_key,text,conf,box) in enumerate(rows):
            out.append(TextBlock(
                id=f'tesseract-line-{order:04d}', polygon=bbox_polygon(box), text=text,
                confidence=max(0.0,min(1.0,conf/100.0)), kind='unknown',
                reading_order=order, bubble_id=None,
                meta={'backend':'tesseract','tesseract_lang':lang,'grouped_line':True},
            ))
        return out

    @staticmethod
    def _arbitrate_vertical_psm_rows(
        primary: list[TextBlock],
        alternate: list[TextBlock],
        *,
        min_conf_gain: float = 0.08,
    ) -> list[TextBlock]:
        """Take a better same-length column from the alternate PSM result.

        PSM 5 and PSM 12 often succeed on different columns of the same manga
        balloon.  Page-level winner-takes-all arbitration throws that evidence
        away.  This cheap pass swaps only a geometrically corresponding column
        with the same CJK glyph count and a material confidence gain; additions
        and deletions remain the job of the stricter multi-route consensus.
        """
        out = list(primary)
        used: set[int] = set()
        for i, row in enumerate(list(out)):
            rx0, _ry0, rx1, _ry1 = row.bbox
            rw = max(1.0, float(rx1 - rx0))
            current_cjk = sum(1 for ch in str(row.text or '') if TesseractOCRBackend._has_cjk(ch))
            if current_cjk <= 0:
                continue
            best = None
            for j, alt in enumerate(alternate):
                if j in used:
                    continue
                ax0, _ay0, ax1, _ay1 = alt.bbox
                aw = max(1.0, float(ax1 - ax0))
                overlap = max(0.0, min(float(rx1), float(ax1)) - max(float(rx0), float(ax0)))
                ownership = overlap / max(1.0, min(rw, aw))
                if ownership < 0.60:
                    continue
                alt_cjk = sum(1 for ch in str(alt.text or '') if TesseractOCRBackend._has_cjk(ch))
                if alt_cjk != current_cjk:
                    continue
                gain = float(alt.confidence) - float(row.confidence)
                candidate = (ownership, gain, j, alt)
                if best is None or (candidate[0], candidate[1]) > (best[0], best[1]):
                    best = candidate
            if best is None:
                continue
            _ownership, gain, j, alt = best
            if gain < float(min_conf_gain):
                continue
            if str(alt.text or '') == str(row.text or ''):
                continue
            alt.meta = dict(getattr(alt, 'meta', {}) or {})
            alt.meta['psm_column_arbitrated'] = True
            alt.meta['psm_column_previous_text'] = str(row.text or '')
            alt.meta['psm_column_previous_confidence'] = float(row.confidence)
            out[i] = alt
            used.add(j)
        out.sort(key=lambda r: (-float((r.bbox[0] + r.bbox[2]) * 0.5), float(r.bbox[1])))
        for idx, row in enumerate(out):
            row.reading_order = idx
        return out

    def recognize_manga_region(self, image: np.ndarray, *, orientation: str = 'vertical') -> list[TextBlock]:
        if image is None or image.size == 0:
            return []
        roi, (ox, oy) = self._tight_ink_roi(image)
        vertical = str(orientation or '').strip().lower() == 'vertical'
        region_deadline = time.monotonic() + max(1.0, float(
            getattr(self.config, 'tesseract_region_budget_seconds', 7.0)
        ))
        if not vertical:
            return self._run_lines(
                roi, lang=self.horizontal_lang, psm=6, offset=(ox,oy), deadline=region_deadline
            )

        # Manga vertical text has two common geometries: compact speech columns
        # (PSM 5 is strongest) and multi-column burst/narration blocks where PSM
        # 12 often recovers a column that PSM 5 drops. Run both on the *same tight
        # ROI* and choose one candidate set deterministically. This is layout
        # arbitration, not repeated OCR over expanding crops.
        def _clean_whole(rows):
            cleaned=[]
            roi_w=max(1, roi.shape[1])
            for row in rows:
                if not self._has_cjk(row.text):
                    continue
                x0,_y0,x1,_y1=row.bbox
                # Very low-confidence fragments glued to the ROI edge are almost
                # always burst rays/bubble borders. They were the source of the
                # trailing “多/舉多多” noise on open speech bubbles.
                edge = (float(x0) <= ox + 2.0) or (float(x1) >= ox + roi_w - 2.0)
                if edge and float(row.confidence) < 0.30:
                    continue
                cleaned.append(row)
            return cleaned

        def _candidate_score(rows):
            if not rows:
                return -1.0
            chars=[max(1,sum(1 for ch in r.text if self._has_cjk(ch))) for r in rows]
            total=sum(chars)
            mean_conf=float(np.average([r.confidence for r in rows],weights=chars))
            # Coverage dominates, confidence breaks close ties. A longer result
            # with catastrophically low confidence still loses to a compact
            # high-confidence speech candidate.
            return float(total) + mean_conf * 3.0

        whole5=_clean_whole(self._run_lines(roi, lang=self.vertical_lang, psm=5, offset=(ox,oy), deadline=region_deadline))
        whole12=_clean_whole(self._run_lines(roi, lang=self.vertical_lang, psm=12, offset=(ox,oy), deadline=region_deadline))
        for row in whole5:
            row.meta['tesseract_vertical_psm'] = 5
        for row in whole12:
            row.meta['tesseract_vertical_psm'] = 12
        if _candidate_score(whole12) > _candidate_score(whole5) + 0.10:
            whole = self._arbitrate_vertical_psm_rows(
                whole12, whole5,
                min_conf_gain=float(getattr(self.config, 'tesseract_psm_column_conf_gain', 0.08)),
            )
        else:
            whole = self._arbitrate_vertical_psm_rows(
                whole5, whole12,
                min_conf_gain=float(getattr(self.config, 'tesseract_psm_column_conf_gain', 0.08)),
            )
        if not bool(getattr(self.config, 'tesseract_column_refine_enabled', True)):
            return whole

        component_bands = self._component_column_bands(roi)
        if component_bands:
            accepted_consensus: list[TextBlock] = []
            replaced_indexes: set[int] = set()
            projection_bands = self._vertical_column_bands(roi)
            reliable_component_bands = []
            for x0, x1 in component_bands:
                width = max(1, x1 - x0)
                best_overlap = 0.0
                for pa, pb in projection_bands:
                    overlap = max(0, min(x1, pb) - max(x0, pa))
                    best_overlap = max(best_overlap, overlap / width)
                if not projection_bands or best_overlap >= 0.45:
                    reliable_component_bands.append((x0, x1))
            # Recheck only suspicious columns. Whole-page PSM5/PSM12 already
            # provide two independent observations; healthy agreement must not
            # trigger four more Tesseract processes per column.
            trigger_conf = float(getattr(self.config, 'tesseract_consensus_trigger_confidence', 0.72))
            max_consensus = max(0, int(getattr(self.config, 'tesseract_max_consensus_columns', 3)))

            def _band_rows(rows, x0, x1):
                gx0, gx1 = float(x0 + ox), float(x1 + ox)
                picked=[]
                for row in rows:
                    rx0,_ry0,rx1,_ry1=row.bbox
                    rcx=(float(rx0)+float(rx1))*0.5
                    if gx0-2.0 <= rcx <= gx1+2.0:
                        picked.append(row)
                return picked

            suspicious=[]
            for x0,x1 in reliable_component_bands:
                selected_rows=_band_rows(whole,x0,x1)
                p5_rows=_band_rows(whole5,x0,x1)
                p12_rows=_band_rows(whole12,x0,x1)
                selected_text=self._clean_token(''.join(r.text for r in selected_rows))
                p5_text=self._clean_token(''.join(r.text for r in p5_rows))
                p12_text=self._clean_token(''.join(r.text for r in p12_rows))
                selected_conf=(
                    float(np.mean([r.confidence for r in selected_rows])) if selected_rows else 0.0
                )
                missing = not selected_text
                low_conf = bool(selected_rows) and selected_conf < trigger_conf
                disagreement = bool(p5_text and p12_text and p5_text != p12_text)
                if missing or low_conf or disagreement:
                    # Spend the bounded consensus budget on the highest-value
                    # uncertainty first: a missing column, then a direct PSM5/12
                    # disagreement, and only then an otherwise low-confidence
                    # column.  A high-confidence but contradictory OCR reading is
                    # more dangerous than a merely soft score because it can
                    # silently overwrite a correct alternate segmentation.
                    priority=(0 if missing else 1 if disagreement else 2, selected_conf, -int(x0))
                    suspicious.append((priority,x0,x1))
            suspicious.sort(key=lambda z:z[0])

            for _priority, x0, x1 in suspicious[:max_consensus]:
                if time.monotonic() >= region_deadline:
                    break
                sx0=max(0,int(x0)-3); sx1=min(roi.shape[1],int(x1)+3)
                if sx1-sx0 < 6:
                    continue
                block=self._column_consensus(
                    roi[:,sx0:sx1], offset=(ox+sx0,oy), deadline=region_deadline
                )
                if block is None:
                    continue
                bx0,_by0,bx1,_by1=block.bbox
                bw=max(1.0,float(bx1-bx0))
                matches=[]
                for idx,row in enumerate(whole):
                    rx0,_ry0,rx1,_ry1=row.bbox
                    rcx=(float(rx0)+float(rx1))*0.5
                    # Use column ownership (centre inside the consensus band)
                    # instead of raw bbox overlap. Tesseract sometimes returns a
                    # very wide bbox for a neighbouring vertical column; overlap
                    # alone then incorrectly groups two independent columns.
                    if float(bx0)-2.0 <= rcx <= float(bx1)+2.0:
                        matches.append(idx)
                if not matches:
                    for idx,row in enumerate(whole):
                        rx0,_ry0,rx1,_ry1=row.bbox
                        rw=max(1.0,float(rx1-rx0))
                        overlap=max(0.0,min(float(rx1),float(bx1))-max(float(rx0),float(bx0)))
                        if overlap/min(bw,rw) >= 0.72:
                            matches.append(idx)
                if matches:
                    old_chars=sum(max(1,sum(1 for ch in whole[i].text if self._has_cjk(ch))) for i in matches)
                    old_conf=float(np.average(
                        [whole[i].confidence for i in matches],
                        weights=[max(1,sum(1 for ch in whole[i].text if self._has_cjk(ch))) for i in matches],
                    ))
                    new_chars=sum(1 for ch in block.text if self._has_cjk(ch))
                    consensus_coverage=float(block.meta.get('consensus_char_coverage',0.0) or 0.0)
                    consensus_routes=int(block.meta.get('consensus_routes',0) or 0)
                    consensus_agreement=float(block.meta.get('consensus_pairwise_agreement',0.0) or 0.0)
                    accepted = (
                        consensus_routes >= int(getattr(self.config,'tesseract_consensus_min_routes',2))
                        and consensus_coverage >= float(getattr(self.config,'tesseract_consensus_min_char_coverage',0.74))
                        and consensus_agreement >= float(getattr(self.config,'tesseract_consensus_min_pairwise_agreement',0.50))
                        and block.confidence >= old_conf-0.12
                        and new_chars >= max(2,int(round(old_chars*float(getattr(self.config,'tesseract_consensus_min_length_ratio',0.88)))))
                        and new_chars <= old_chars
                    )
                    if accepted:
                        replaced_indexes.update(matches)
                        accepted_consensus.append(block)
                    else:
                        old_text = self._clean_token(''.join(whole[i].text for i in matches))
                        candidate_text = self._clean_token(str(block.text or ''))
                        meaningful_conflict = (
                            candidate_text and old_text and candidate_text != old_text
                            and candidate_text not in old_text and old_text not in candidate_text
                        )
                        if meaningful_conflict:
                            for idx in matches:
                                whole[idx].meta = dict(getattr(whole[idx], 'meta', {}) or {})
                                whole[idx].meta['column_consensus_review_required'] = True
                                whole[idx].meta['column_consensus_candidate_text'] = candidate_text
                                whole[idx].meta['column_consensus_candidate_confidence'] = float(block.confidence)
                                whole[idx].meta['column_consensus_candidate_agreement'] = float(consensus_agreement)
                                whole[idx].meta['column_consensus_candidate_coverage'] = float(consensus_coverage)
                else:
                    consensus_coverage=float(block.meta.get('consensus_char_coverage',0.0) or 0.0)
                    consensus_routes=int(block.meta.get('consensus_routes',0) or 0)
                    consensus_agreement=float(block.meta.get('consensus_pairwise_agreement',0.0) or 0.0)
                    if (
                        consensus_routes >= int(getattr(self.config,'tesseract_consensus_min_routes',2))
                        and consensus_coverage >= float(getattr(self.config,'tesseract_consensus_min_char_coverage',0.74))
                        and consensus_agreement >= float(getattr(self.config,'tesseract_consensus_min_pairwise_agreement',0.50))
                        and block.confidence >= 0.62
                    ):
                        accepted_consensus.append(block)
            if accepted_consensus:
                whole=[row for i,row in enumerate(whole) if i not in replaced_indexes] + accepted_consensus
                whole.sort(key=lambda r:(-float((r.bbox[0]+r.bbox[2])*0.5),float(r.bbox[1])))
                for idx,row in enumerate(whole):
                    row.reading_order=idx
                    row.id=f'tesseract-consensus-{idx:04d}'

        bands = [b for b in self._vertical_column_bands(roi) if b[1]-b[0] >= 5]
        max_retries = max(1, int(getattr(self.config, 'tesseract_max_column_retries', 8)))
        if not bands or len(bands) > max_retries:
            return whole

        h, w = roi.shape[:2]
        retries: list[TextBlock] = []
        for x0, x1 in sorted(bands, key=lambda b: b[0], reverse=True):
            gx0, gx1 = x0 + ox, x1 + ox
            band_w = max(1.0, float(gx1-gx0))
            # Consider the band already covered only when a whole-region OCR box
            # occupies most of its horizontal span.  This preserves reliable
            # columns and confines extra OCR to genuinely missing text.
            covered = False
            for row in whole:
                rx0, _ry0, rx1, _ry1 = row.bbox
                overlap = max(0.0, min(float(rx1), float(gx1)) - max(float(rx0), float(gx0)))
                if overlap / band_w >= 0.55:
                    covered = True
                    break
            if covered:
                continue

            pad = 4
            sx0=max(0,x0-pad); sx1=min(w,x1+pad)
            if sx1 - sx0 < 6:
                continue
            sub=roi[:,sx0:sx1]
            if time.monotonic() >= region_deadline:
                break
            rows=[r for r in self._run_lines(sub, lang=self.vertical_lang, psm=5, offset=(ox+sx0,oy), deadline=region_deadline)
                  if self._has_cjk(r.text)]
            horizontal_fallback = False
            if not rows:
                # Short 2–3 glyph vertical columns are a known weak spot of the
                # *_vert Tesseract models.  On the same upright narrow crop,
                # horizontal HanT/HanS often emits one row per glyph; joining by
                # Y restores the column without rotating/resampling the source.
                rows=[r for r in self._run_lines(sub, lang=self.horizontal_lang, psm=6, offset=(ox+sx0,oy), deadline=region_deadline)
                      if self._has_cjk(r.text)]
                horizontal_fallback = bool(rows)
            if not rows:
                # ``image_to_data`` can return no word boxes for a tiny vertical
                # 2–3 glyph column even when ``image_to_string`` reads it
                # correctly.  Use a bounded string-only fallback on the *same*
                # already-verified column crop.  Upscaling changes only OCR
                # sampling, never publication geometry.
                try:
                    scaled=cv2.resize(sub, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                    raw_text=self._pytesseract.image_to_string(
                        cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB) if scaled.ndim == 3 else scaled,
                        lang=self.horizontal_lang,
                        config='--oem 1 --psm 6 -c preserve_interword_spaces=0',
                        timeout=max(0.20, min(
                            float(getattr(self.config, 'tesseract_call_timeout_seconds', 3.0)),
                            max(0.20, region_deadline - time.monotonic()),
                        )),
                    )
                    parts=[self._clean_token(x) for x in str(raw_text or '').splitlines()]
                    text=''.join(x for x in parts if self._has_cjk(x))
                    if sum(1 for ch in text if self._has_cjk(ch)) >= 2:
                        rows=[TextBlock(
                            id='tesseract-short-column-string',
                            polygon=bbox_polygon((ox+sx0, oy, ox+sx1, oy+h)),
                            text=text, confidence=0.58, kind='unknown', reading_order=0,
                            meta={
                                'backend':'tesseract',
                                'tesseract_lang':self.horizontal_lang,
                                'short_column_string_fallback':True,
                            },
                        )]
                        horizontal_fallback=True
                except Exception:
                    rows=[]
            if not rows:
                continue
            rows=sorted(rows,key=lambda r:(float(r.bbox[1]),float(r.bbox[0])))
            text=''.join(r.text for r in rows if self._has_cjk(r.text))
            if not text:
                continue
            weights=[max(1,sum(1 for ch in r.text if self._has_cjk(ch))) for r in rows]
            conf=float(np.average([r.confidence for r in rows],weights=weights))
            min_retry_conf=max(0.30, float(getattr(self.config,'tesseract_min_confidence',0.42))-0.08)
            if conf < min_retry_conf:
                continue
            xlo=min(r.bbox[0] for r in rows); ylo=min(r.bbox[1] for r in rows)
            xhi=max(r.bbox[2] for r in rows); yhi=max(r.bbox[3] for r in rows)
            retries.append(TextBlock(
                id=f'tesseract-column-retry-{len(retries):04d}',
                polygon=bbox_polygon((xlo,ylo,xhi,yhi)), text=text,
                confidence=conf, kind='unknown', reading_order=0,
                meta={
                    'backend':'tesseract',
                    'tesseract_lang':self.horizontal_lang if horizontal_fallback else self.vertical_lang,
                    'column_retry':True,
                    'short_column_horizontal_fallback':horizontal_fallback,
                },
            ))

        merged=whole+retries
        merged.sort(key=lambda r:(-float((r.bbox[0]+r.bbox[2])*0.5), float(r.bbox[1])))
        for idx,row in enumerate(merged):
            row.reading_order=idx
            row.id=f'tesseract-manga-{idx:04d}'
        return merged

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if image is None or image.size == 0:
            return []
        if image_path is None:
            h, w = image.shape[:2]
            aspect = float(getattr(self.config, 'tesseract_region_vertical_aspect', 1.12))
            orientation = 'vertical' if h >= w * aspect else 'horizontal'
            return self.recognize_manga_region(image, orientation=orientation)
        # Full pages stay sparse and never enter expensive per-column retries.
        return self._run_lines(image, lang=self.tess_lang, psm=11, offset=(0,0))


class PaddleOCRBackend(OCRBackend):
    """PaddleOCR backend hosted in an isolated compatible Python venv.

    Current macOS Paddle wheels are arm64 CPython 3.9-3.13 only.  The desktop
    GUI may run on a newer Python (or under Rosetta), so importing PaddleOCR in
    process is deliberately avoided.  A persistent JSONL worker keeps model
    startup cost out of per-page recognition.
    """

    def __init__(self, lang: str, config: OCRConfig | None = None, device: str | None = None, profile_override: str | None = None, *, onnx_coreml: bool = False) -> None:
        self.config = config or OCRConfig()
        self.lang = str(lang)
        self.profile_override = profile_override
        self.onnx_coreml = bool(onnx_coreml)
        self.selection: dict[str, Any] = {}
        self.retry_crops = True
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_queue: queue.Queue[str | None] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._start_worker(device=device)

    def _start_worker(self, device: str | None = None) -> None:
        from .paddle_runtime import require_runtime_python as require_ocr_python, worker_script as ocr_worker_script, runtime_root as ocr_runtime_root
        if self.onnx_coreml:
            from .paddle_onnx_coreml_runtime import ensure_onnxruntime
            bridge_status = ensure_onnxruntime()
            if not bridge_status.ready:
                raise RuntimeError(bridge_status.detail or "Paddle ONNXRuntime/CoreML bridge not ready")
        from .paddle_doc_runtime import ensure_runtime as ensure_doc_runtime, require_runtime_python as require_doc_python, worker_script as doc_worker_script, runtime_root as doc_runtime_root
        from .paddle_sources import paddle_model_source_attempts, paddle_source_environment, PADDLE_MODEL_SOURCE_LABELS
        from .paddle_profiles import resolve_paddle_model_selection
        from .tls_support import apply_runtime_tls_environment

        selection = resolve_paddle_model_selection(self.config, self.profile_override)
        self.selection = dict(selection)
        pipeline_name = str(selection.get("pipeline") or "ocr")
        local_det, local_rec = discovered_paddle_model_dirs()
        det_dir = (self.config.paddle_text_detection_model_dir or (str(local_det) if local_det else None)) if pipeline_name == "ocr" else None
        rec_dir = (self.config.paddle_text_recognition_model_dir or (str(local_rec) if local_rec else None)) if pipeline_name == "ocr" else None
        if pipeline_name in {"vl", "structure"}:
            # VL / Structure have their own dependency surface and venv.  A repair
            # must never mutate the classic PP-OCRv6 runtime.
            ensure_doc_runtime()
            py = require_doc_python()
            worker = doc_worker_script()
            tls_root = doc_runtime_root()
        else:
            py = require_ocr_python()
            worker = ocr_worker_script()
            tls_root = ocr_runtime_root()
        self.retry_crops = pipeline_name == "ocr"
        cmd = [
            str(py), "-u", str(worker), "--lang", self.lang,
            "--pipeline", pipeline_name,
            "--ocr-version", str(selection["ocr_version"] or self.config.ocr_version),
            "--model-profile", str(selection["profile"]),
        ]
        if selection.get("pipeline_version"):
            cmd += ["--pipeline-version", str(selection["pipeline_version"])]
        if selection.get("det_name"):
            cmd += ["--det-name", str(selection["det_name"])]
        if selection.get("rec_name"):
            cmd += ["--rec-name", str(selection["rec_name"])]
        if det_dir:
            cmd += ["--det-dir", str(det_dir)]
        if rec_dir:
            cmd += ["--rec-dir", str(rec_dir)]
        if device and str(device).lower() in {"cpu", "gpu"}:
            cmd += ["--device", str(device).lower()]
        if self.onnx_coreml:
            cmd += ["--onnx-engine", "--coreml-units", str(getattr(self.config, "paddle_coreml_compute_units", "ane") or "ane")]

        # With complete offline model dirs, never touch a model hub. Otherwise
        # retry sources only during worker initialization, before any page is
        # emitted; this avoids duplicate OCR output on fallback.
        sources = ("local",) if det_dir and rec_dir else paddle_model_source_attempts(getattr(self.config, "paddle_model_source", "auto"))
        failures: list[str] = []
        self.model_source = ""
        for source in sources:
            base = os.environ.copy()
            env = base if source == "local" else paddle_source_environment(source, base)
            proxy = str(env.get("MHD_MODEL_PROXY", "") or "").strip()
            if proxy:
                for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
                    env[key] = proxy
            env = apply_runtime_tls_environment(env, tls_root)
            try:
                self._proc = subprocess.Popen(
                    cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
                )
                self._start_stdout_pump()
                ready = self._read_json(expected={"ready", "probe", "init"})
                if ready.get("ok"):
                    self.model_source = source
                    return
                detail = str(ready.get("error", "未知错误"))
            except Exception as exc:
                detail = str(exc)
            label = "本地离线模型" if source == "local" else PADDLE_MODEL_SOURCE_LABELS.get(source, source)
            failures.append(f"{label}: {detail[-3500:]}")
            self.close()

        raise RuntimeError(
            "PaddleOCR 独立运行环境初始化失败；已在 OCR 输出前尝试全部允许的模型源。\n"
            + "\n\n".join(failures[-4:])[-10000:]
        )

    def _start_stdout_pump(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("PaddleOCR worker stdout 不可用")
        q: queue.Queue[str | None] = queue.Queue()
        self._stdout_queue = q

        def pump() -> None:
            try:
                for line in proc.stdout:
                    q.put(line)
            finally:
                q.put(None)

        thread = threading.Thread(target=pump, name="mhd-paddle-worker-output", daemon=True)
        self._stdout_thread = thread
        thread.start()

    def _read_json(self, *, expected: set[str], max_lines: int = 400) -> dict[str, Any]:
        proc = self._proc
        q = self._stdout_queue
        if proc is None or q is None:
            raise RuntimeError("PaddleOCR worker 未启动")
        diagnostics: list[str] = []
        startup = bool(expected.intersection({"ready", "probe", "init"}))
        env_name = "MHD_PADDLE_STARTUP_TIMEOUT" if startup else "MHD_PADDLE_REQUEST_TIMEOUT"
        default_timeout = 900.0 if startup else 300.0
        try:
            timeout = max(30.0, float(os.environ.get(env_name, default_timeout)))
        except Exception:
            timeout = default_timeout
        deadline = time.monotonic() + timeout
        lines_seen = 0
        while lines_seen < max_lines:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(
                    f"PaddleOCR worker {'初始化/模型下载' if startup else '识别'}等待超过 {int(timeout)} 秒。"
                    + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                )
            try:
                line = q.get(timeout=min(1.0, remaining))
            except queue.Empty:
                code = proc.poll()
                if code is not None:
                    raise RuntimeError(
                        f"PaddleOCR worker 提前退出 (code={code})"
                        + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                    )
                continue
            if line is None:
                code = proc.poll()
                raise RuntimeError(
                    "PaddleOCR worker 输出结束" + (f" (code={code})" if code is not None else "")
                    + ("\n" + "\n".join(diagnostics[-30:]) if diagnostics else "")
                )
            lines_seen += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                diagnostics.append(raw)
                continue
            if isinstance(payload, dict) and str(payload.get("type", "")) in expected:
                return payload
            diagnostics.append(raw)
        raise RuntimeError("PaddleOCR worker 输出过多非协议日志，无法取得结果。\n" + "\n".join(diagnostics[-30:]))

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            self.close()
            self._start_worker()
            proc = self._proc
        assert proc is not None and proc.stdin is not None
        fd, tmp_name = tempfile.mkstemp(prefix="mhd-paddle-", suffix=".png")
        os.close(fd)
        try:
            if not cv2.imwrite(tmp_name, image):
                raise RuntimeError("无法写入 PaddleOCR 临时图像")
            proc.stdin.write(json.dumps({"cmd": "predict", "path": tmp_name}, ensure_ascii=False) + "\n")
            proc.stdin.flush()
            payload = self._read_json(expected={"result"})
            if not payload.get("ok"):
                pipeline_name = str(self.selection.get("pipeline") or "ocr")
                profile_name = str(self.selection.get("profile") or self.config.paddle_model_profile or "")
                stage = str(payload.get("stage") or "predict")
                raise RuntimeError(
                    f"PaddleOCR 识别失败 [{pipeline_name}/{profile_name}/{stage}]："
                    + str(payload.get("error", "未知错误"))
                )
            blocks: list[TextBlock] = []
            for i, row in enumerate(payload.get("blocks") or []):
                poly = row.get("polygon") or []
                if len(poly) < 3:
                    continue
                blocks.append(
                    TextBlock(
                        id=str(row.get("id") or f"ocr-{i:04d}"),
                        polygon=[(float(x), float(y)) for x, y in poly],
                        text=str(row.get("text", "")),
                        confidence=float(row.get("confidence", 1.0)),
                        reading_order=i,
                        kind=str(row.get("kind") or row.get("label") or "unknown"),
                        meta={
                            "backend": "paddle",
                            "paddle_pipeline": str(self.selection.get("pipeline") or "ocr"),
                            "paddle_profile": str(self.selection.get("profile") or ""),
                            "paddle_pipeline_version": str(self.selection.get("pipeline_version") or ""),
                            "ocr_version": str(self.selection.get("ocr_version") or self.config.ocr_version),
                            "isolated_runtime": True, "model_source": getattr(self, "model_source", ""),
                            "onnx_engine": bool(self.onnx_coreml),
                            "coreml_compute_units": str(getattr(self.config, "paddle_coreml_compute_units", "ane") or "ane"),
                            "coreml_requested": bool(self.onnx_coreml),
                            **dict(row.get("meta") or {}),
                        },
                    )
                )
            return sort_reading_order(blocks)
        finally:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        self._stdout_queue = None
        self._stdout_thread = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write('{"cmd":"close"}\n'); proc.stdin.flush()
                proc.wait(timeout=3)
        except Exception:
            try:
                proc.terminate(); proc.wait(timeout=2)
            except Exception:
                try: proc.kill()
                except Exception: pass
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None: stream.close()
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class RetryingOCRBackend(OCRBackend):
    """Re-runs low-confidence crops at higher scale and keeps evidence for arbitration."""

    def __init__(self, primary: OCRBackend, threshold: float = 0.80, scale: float = 2.0) -> None:
        self.primary = primary
        self.threshold = threshold
        self.scale = max(1.0, scale)

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        blocks = self.primary.recognize(image, image_path=image_path)
        h, w = image.shape[:2]
        for block in blocks:
            if block.confidence >= self.threshold:
                continue
            x0, y0, x1, y1 = block.bbox
            pad = max(3, round(min(x1 - x0, y1 - y0) * 0.12))
            ix0, iy0 = max(0, int(x0) - pad), max(0, int(y0) - pad)
            ix1, iy1 = min(w, int(x1) + pad), min(h, int(y1) + pad)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            crop = image[iy0:iy1, ix0:ix1]
            up = cv2.resize(crop, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_CUBIC)
            gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
            sharpen = cv2.addWeighted(gray, 1.7, cv2.GaussianBlur(gray, (0, 0), 1.0), -0.7, 0)
            variants = [up, cv2.cvtColor(sharpen, cv2.COLOR_GRAY2BGR)]
            if getattr(self.primary, "config", None) is not None and getattr(self.primary.config, "photo_ocr_preprocess", True):
                clip = float(getattr(self.primary.config, "photo_ocr_clahe_clip", 2.2))
                clahe = cv2.createCLAHE(clipLimit=max(1.0, clip), tileGridSize=(8, 8)).apply(gray)
                local = cv2.adaptiveThreshold(
                    clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 31, 11,
                )
                variants.extend([
                    cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(local, cv2.COLOR_GRAY2BGR),
                ])
            candidates = [{"text": block.text, "confidence": block.confidence, "variant": "original"}]
            for vi, variant in enumerate(variants):
                try:
                    reread = self.primary.recognize(variant, image_path=None)
                except Exception:
                    continue
                if not reread:
                    continue
                # Prefer the most confident, text-bearing region in the crop.
                cand = max(reread, key=lambda b: (b.confidence, len(b.text.strip())))
                candidates.append({"text": cand.text, "confidence": cand.confidence, "variant": f"retry_{vi}"})
            best = max(candidates, key=lambda x: (x["confidence"], len(x["text"].strip())))
            block.meta["ocr_candidates"] = candidates
            if best["confidence"] > block.confidence + 0.015 and best["text"].strip():
                block.meta["ocr_original"] = {"text": block.text, "confidence": block.confidence}
                block.text = str(best["text"])
                block.confidence = float(best["confidence"])
                block.meta["ocr_selected"] = best["variant"]
        return blocks


class InjectedOCRBackend(OCRBackend):
    """Test/embedding adapter that returns caller-provided blocks by image path or one static list."""

    def __init__(self, blocks: list[TextBlock] | dict[str, list[TextBlock]]) -> None:
        self.blocks = blocks

    def recognize(self, image: np.ndarray, *, image_path: str | Path | None = None) -> list[TextBlock]:
        if isinstance(self.blocks, dict):
            key = str(image_path) if image_path is not None else ""
            values = self.blocks.get(key, [])
        else:
            values = self.blocks
        return [
            TextBlock(
                id=b.id,
                polygon=list(b.polygon),
                text=b.text,
                confidence=b.confidence,
                kind=b.kind,
                reading_order=b.reading_order,
                bubble_id=b.bubble_id,
                meta=dict(b.meta),
            )
            for b in values
        ]


def sort_reading_order(blocks: list[TextBlock], vertical: bool = False) -> list[TextBlock]:
    if vertical:
        ordered = sorted(blocks, key=lambda b: (-b.centroid[0], b.centroid[1]))
    else:
        # Row-aware sort: quantize Y by median block height to avoid tiny OCR jitter.
        heights = [max(1.0, b.bbox[3] - b.bbox[1]) for b in blocks]
        band = float(np.median(heights)) * 0.7 if heights else 16.0
        ordered = sorted(blocks, key=lambda b: (round(b.centroid[1] / max(1.0, band)), b.centroid[0]))
    for i, block in enumerate(ordered):
        block.reading_order = i
    return ordered


def filter_blocks(blocks: list[TextBlock], min_confidence: float) -> tuple[list[TextBlock], list[TextBlock]]:
    accepted, review = [], []
    for b in blocks:
        (accepted if b.confidence >= min_confidence else review).append(b)
    return accepted, review


def build_backend(config: OCRConfig, lang: str, backend: str | None = None, *, role: str | None = None) -> OCRBackend:
    name = (backend or config.backend).lower().strip()
    if name in {"manga_ocr", "manga-ocr", "baberu_ocr", "baberu-ocr", "ocr48px", "48px", "hayai_ocr", "hayai-ocr"}:
        from .koharu_ocr import KoharuCropOCRBackend
        normalized = {"manga-ocr":"manga_ocr", "baberu-ocr":"baberu_ocr", "48px":"ocr48px", "hayai-ocr":"hayai_ocr"}.get(name,name)
        return KoharuCropOCRBackend(normalized, lang, config)
    if name in {"ndlocr_lite", "ndlocr-lite", "ndl"}:
        from .ndlocr_lite_runtime import NDLOCRLiteBackend
        return NDLOCRLiteBackend(lang, config)
    if name in {"m6_japanese_auto", "m6-japanese-auto", "japanese_auto", "m6_ja_auto"}:
        from .m6_japanese_ocr import M6JapaneseAutoOCRBackend
        return M6JapaneseAutoOCRBackend(lang, config)
    if name == "paddle":
        return PaddleOCRBackend(lang, config)
    if name in {"tesseract", "cloud_tesseract", "runtime_tesseract"}:
        return TesseractOCRBackend(lang, config)
    if name in {"paddle_onnx_coreml", "paddle-coreml", "paddle_onnx"}:
        return PaddleOCRBackend(lang, config, onnx_coreml=True)
    from .paddle_profiles import backend_profile_key
    paddle_profile = backend_profile_key(name)
    if paddle_profile:
        return PaddleOCRBackend(lang, config, profile_override=paddle_profile)
    if name in {"external", "external_ocr", "external_json", "external_md"}:
        from .external_ocr import ExternalOCRBackend
        selected_role = str(role or "source").strip().lower()
        if selected_role == "target":
            path = getattr(config, "external_target_ocr_path", None)
            start_page = getattr(config, "external_target_start_page", 1)
        else:
            path = getattr(config, "external_source_ocr_path", None)
            start_page = getattr(config, "external_source_start_page", 1)
        if not path:
            raise RuntimeError(f"{selected_role.upper()} 已选择外部 OCR，但尚未指定 JSON/MD 文件。")
        return ExternalOCRBackend(path, start_page=start_page, ignored_labels=getattr(config, "external_ocr_ignore_labels", ()))
    # Apple OCR uses the same user-facing route as Novel Formatter:
    # Swift VisionKit Live Text first, macOS ExtractText Shortcut fallback.
    if name in {"apple", "apple_live_text", "apple_auto", "live_text"}:
        from .apple_live_text import AppleAutoLiveTextBackend
        return AppleAutoLiveTextBackend(lang, config)
    if name in {"apple_visionkit", "visionkit"}:
        from .apple_live_text import AppleVisionKitLiveTextBackend
        return AppleVisionKitLiveTextBackend(lang, config)
    if name in {"apple_shortcut", "shortcut", "shortcuts"}:
        from .apple_live_text import AppleShortcutLiveTextBackend
        return AppleShortcutLiveTextBackend(lang, config)
    if name == "sidecar":
        return SidecarOCRBackend(config.sidecar_suffix)
    if name == "none":
        return NullOCRBackend()
    raise ValueError(f"Unknown OCR backend: {name}")
