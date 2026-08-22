from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

from .config import PairingConfig
from .io_utils import list_images, read_image
from .models import PageFingerprint, PagePair
from .remake_pairing import verify_remake_pair


def _dhash(gray: np.ndarray, hash_size: int = 8) -> int:
    small = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    value = 0
    for bit in diff.flatten():
        value = (value << 1) | int(bit)
    return value


def _hamming64(a: int, b: int) -> float:
    return (a ^ b).bit_count() / 64.0


def _edge_descriptor(gray: np.ndarray) -> list[float]:
    thumb = cv2.resize(gray, (96, 128), interpolation=cv2.INTER_AREA)
    thumb = cv2.GaussianBlur(thumb, (5, 5), 0)
    gx = cv2.Sobel(thumb, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(thumb, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out: list[float] = []
    for yy in range(4):
        for xx in range(4):
            cell = mag[yy * 32 : (yy + 1) * 32, xx * 24 : (xx + 1) * 24]
            out.append(float(np.mean(cell)))
    arr = np.asarray(out, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-8:
        arr /= norm
    return arr.tolist()


def fingerprint_image(path: str | Path, index: int) -> PageFingerprint:
    img = read_image(path)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Blurring/downsampling reduces the influence of different language glyphs.
    smooth = cv2.GaussianBlur(gray, (7, 7), 1.5)
    return PageFingerprint(
        path=str(path),
        index=index,
        width=w,
        height=h,
        dhash=_dhash(smooth),
        edge_hist=_edge_descriptor(smooth),
    )


@dataclass(slots=True)
class PairCost:
    total: float
    hash_cost: float
    aspect_cost: float
    edge_cost: float
    order_cost: float


def fingerprint_cost(
    a: PageFingerprint,
    b: PageFingerprint,
    source_count: int,
    target_count: int,
    config: PairingConfig,
) -> PairCost:
    hash_cost = _hamming64(a.dhash, b.dhash)
    aspect_cost = min(1.0, abs(np.log(max(a.aspect, 1e-6) / max(b.aspect, 1e-6))))
    ea = np.asarray(a.edge_hist, dtype=np.float32)
    eb = np.asarray(b.edge_hist, dtype=np.float32)
    edge_cost = float(min(1.0, np.linalg.norm(ea - eb) / np.sqrt(2.0)))
    pa = a.index / max(1, source_count - 1)
    pb = b.index / max(1, target_count - 1)
    order_cost = min(1.0, abs(pa - pb))
    total = (
        config.hash_weight * hash_cost
        + config.aspect_weight * aspect_cost
        + config.edge_weight * edge_cost
        + config.order_weight * order_cost
    )
    return PairCost(total, hash_cost, aspect_cost, edge_cost, order_cost)


def _build_page_pairs_from_alignment(
    source: list[PageFingerprint], target: list[PageFingerprint], cfg: PairingConfig,
    pairs_rev: list[tuple[int, int]], unmatched_source: list[int], unmatched_target: list[int],
    cost_lookup,
) -> tuple[list[PagePair], list[int], list[int]]:
    page_pairs: list[PagePair] = []
    for si, tj in reversed(pairs_rev):
        c = cost_lookup(si, tj)
        if c.total > cfg.max_pair_cost:
            unmatched_source.append(si)
            unmatched_target.append(tj)
            continue
        confidence = float(np.clip(1.0 - c.total / max(cfg.max_pair_cost, 1e-6), 0.0, 1.0))
        reasons = [
            "pairing=smart",
            f"dhash={c.hash_cost:.3f}",
            f"edge={c.edge_cost:.3f}",
            f"aspect={c.aspect_cost:.3f}",
            f"order={c.order_cost:.3f}",
        ]
        if confidence < cfg.confidence_floor:
            reasons.append("review_recommended")
        page_pairs.append(
            PagePair(
                source_path=source[si].path,
                target_path=target[tj].path,
                source_index=si,
                target_index=tj,
                confidence=confidence,
                score=c.total,
                reasons=reasons,
            )
        )
    return page_pairs, sorted(set(unmatched_source)), sorted(set(unmatched_target))


def _pair_fingerprints_full(
    source: list[PageFingerprint], target: list[PageFingerprint], cfg: PairingConfig,
) -> tuple[list[PagePair], list[int], list[int]]:
    """Reference full-matrix sequence alignment used by existing projects."""
    n, m = len(source), len(target)
    costs = [[fingerprint_cost(source[i], target[j], n, m, cfg) for j in range(m)] for i in range(n)]

    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    prev: list[list[tuple[int, int, str] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + cfg.gap_penalty
        prev[i][0] = (i - 1, 0, "skip_source")
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + cfg.gap_penalty
        prev[0][j] = (0, j - 1, "skip_target")

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = [
                (dp[i - 1, j - 1] + costs[i - 1][j - 1].total, (i - 1, j - 1, "pair")),
                (dp[i - 1, j] + cfg.gap_penalty, (i - 1, j, "skip_source")),
                (dp[i, j - 1] + cfg.gap_penalty, (i, j - 1, "skip_target")),
            ]
            val, step = min(choices, key=lambda x: x[0])
            dp[i, j] = val
            prev[i][j] = step

    pairs_rev: list[tuple[int, int]] = []
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    i, j = n, m
    while i > 0 or j > 0:
        step = prev[i][j]
        if step is None:
            break
        pi, pj, op = step
        if op == "pair":
            pairs_rev.append((i - 1, j - 1))
        elif op == "skip_source":
            unmatched_source.append(i - 1)
        else:
            unmatched_target.append(j - 1)
        i, j = pi, pj

    return _build_page_pairs_from_alignment(
        source, target, cfg, pairs_rev, unmatched_source, unmatched_target,
        lambda si, tj: costs[si][tj],
    )


def _pair_fingerprints_banded(
    source: list[PageFingerprint], target: list[PageFingerprint], cfg: PairingConfig,
) -> tuple[list[PagePair], list[int], list[int]] | None:
    """Memory-bounded sequence alignment for very long unresolved intervals.

    The path is constrained around the normalized reading-order diagonal.  The
    effective band always covers the source/target count delta.  If the endpoint
    cannot be reached, callers fail safe by using the full reference algorithm.
    """
    n, m = len(source), len(target)
    base_band = max(8, int(getattr(cfg, "smart_alignment_band", 64) or 64))
    band = max(base_band, abs(n - m) + 8)
    # If the band is already effectively the whole matrix there is no benefit.
    if band >= max(n, m):
        return None

    dp_prev: dict[int, float] = {0: 0.0}
    prev: dict[tuple[int, int], tuple[int, int, str]] = {}
    costs: dict[tuple[int, int], PairCost] = {}

    def get_cost(si: int, tj: int) -> PairCost:
        key = (si, tj)
        value = costs.get(key)
        if value is None:
            value = fingerprint_cost(source[si], target[tj], n, m, cfg)
            costs[key] = value
        return value

    for i in range(0, n + 1):
        center = int(round((i * m) / max(1, n)))
        j_lo = max(0, center - band)
        j_hi = min(m, center + band)
        if i == 0:
            row: dict[int, float] = {0: 0.0}
            for j in range(max(1, j_lo), j_hi + 1):
                left = row.get(j - 1)
                if left is not None:
                    row[j] = left + cfg.gap_penalty
                    prev[(0, j)] = (0, j - 1, "skip_target")
            dp_prev = row
            continue

        row = {}
        for j in range(j_lo, j_hi + 1):
            choices: list[tuple[float, tuple[int, int, str]]] = []
            up = dp_prev.get(j)
            if up is not None:
                choices.append((up + cfg.gap_penalty, (i - 1, j, "skip_source")))
            left = row.get(j - 1)
            if left is not None:
                choices.append((left + cfg.gap_penalty, (i, j - 1, "skip_target")))
            diag = dp_prev.get(j - 1)
            if j > 0 and diag is not None:
                choices.append((diag + get_cost(i - 1, j - 1).total, (i - 1, j - 1, "pair")))
            if choices:
                value, step = min(choices, key=lambda x: x[0])
                row[j] = value
                prev[(i, j)] = step
        dp_prev = row

    if m not in dp_prev:
        return None

    pairs_rev: list[tuple[int, int]] = []
    unmatched_source: list[int] = []
    unmatched_target: list[int] = []
    i, j = n, m
    touched_band_edge = False
    while i > 0 or j > 0:
        center = int(round((i * m) / max(1, n)))
        if 0 < i < n and abs(j - center) >= max(1, band - 2):
            touched_band_edge = True
        step = prev.get((i, j))
        if step is None:
            return None
        pi, pj, op = step
        if op == "pair":
            pairs_rev.append((i - 1, j - 1))
        elif op == "skip_source":
            unmatched_source.append(i - 1)
        else:
            unmatched_target.append(j - 1)
        i, j = pi, pj

    # A best path riding the imposed boundary is evidence that the configured
    # band may be too narrow for this edition. Fail safe to the full reference
    # solver rather than silently accepting a resource-constrained pairing.
    if touched_band_edge:
        return None

    return _build_page_pairs_from_alignment(
        source, target, cfg, pairs_rev, unmatched_source, unmatched_target, get_cost,
    )


def pair_fingerprints(
    source: list[PageFingerprint], target: list[PageFingerprint], config: PairingConfig | None = None
) -> tuple[list[PagePair], list[int], list[int]]:
    cfg = config or PairingConfig()
    n, m = len(source), len(target)
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))

    max_full_cells = max(1, int(getattr(cfg, "smart_alignment_full_matrix_max_cells", 250000) or 1))
    if n * m > max_full_cells:
        banded = _pair_fingerprints_banded(source, target, cfg)
        if banded is not None:
            return banded
    return _pair_fingerprints_full(source, target, cfg)


_COPY_SUFFIX_RE = re.compile(r"(?:\s*\(\s*\d+\s*\)|\s*\[\s*\d+\s*\])$")
_ROLE_SUFFIX_RE = re.compile(
    r"(?:[\s._-]*(?:cn|zh|zhcn|chs|sc|translated|translation|汉化|漢化|中文|"
    r"jp|ja|jpn|raw|original|原版|日文|日本語))+$",
    re.IGNORECASE,
)


def _normalized_stem(path: str | Path) -> str:
    """Normalize edition/copy decorations while preserving the actual page name."""
    stem = unicodedata.normalize("NFKC", Path(path).stem).strip().casefold()
    # Finder/browser duplicate suffixes like 006(2) or p-006(1) should not become
    # page numbers. Strip repeated copy suffixes before extracting the page key.
    previous = None
    while previous != stem:
        previous = stem
        stem = _COPY_SUFFIX_RE.sub("", stem).strip()
    stem = _ROLE_SUFFIX_RE.sub("", stem).strip(" ._-")
    return stem


def _exact_name_key(path: str | Path) -> str:
    stem = _normalized_stem(path)
    # Ignore punctuation/separators and extension differences, but do not strip CJK.
    return re.sub(r"[^0-9a-z\u3040-\u30ff\u3400-\u9fff]+", "", stem)


def _page_number_key(path: str | Path) -> str | None:
    stem = _normalized_stem(path)
    numbers = re.findall(r"\d+", stem)
    if not numbers:
        return None
    # The final number is normally the page number in names such as vol1_page_006.
    # Finder copy suffixes were already removed above.
    token = numbers[-1]
    try:
        return str(int(token))
    except ValueError:
        return token.lstrip("0") or "0"


def _unique_key_pairs(
    source_paths: list[Path], target_paths: list[Path], source_ids: set[int], target_ids: set[int],
    key_fn,
) -> list[tuple[int, int, str]]:
    smap: dict[str, list[int]] = {}
    tmap: dict[str, list[int]] = {}
    for i in sorted(source_ids):
        key = key_fn(source_paths[i])
        if key:
            smap.setdefault(str(key), []).append(i)
    for j in sorted(target_ids):
        key = key_fn(target_paths[j])
        if key:
            tmap.setdefault(str(key), []).append(j)
    out: list[tuple[int, int, str]] = []
    for key in sorted(set(smap) & set(tmap)):
        if len(smap[key]) == 1 and len(tmap[key]) == 1:
            out.append((smap[key][0], tmap[key][0], key))
    return out


def _monotonic_subset(candidates: list[tuple[int, int, str, str]]) -> list[tuple[int, int, str, str]]:
    """Keep the strongest maximum-size monotonic set of filename anchors."""
    if not candidates:
        return []
    rows = sorted(candidates, key=lambda row: (row[0], row[1]))
    # DP objective: first maximize pair count, then prefer exact-name anchors.
    best: list[tuple[int, int]] = [(1, 3 if row[3] == "name_exact" else 1) for row in rows]
    prev = [-1] * len(rows)
    for i, row in enumerate(rows):
        for j in range(i):
            if rows[j][1] >= row[1]:
                continue
            candidate = (best[j][0] + 1, best[j][1] + (3 if row[3] == "name_exact" else 1))
            if candidate > best[i]:
                best[i] = candidate
                prev[i] = j
    idx = max(range(len(rows)), key=lambda i: best[i])
    keep: list[tuple[int, int, str, str]] = []
    while idx >= 0:
        keep.append(rows[idx])
        idx = prev[idx]
    return list(reversed(keep))


def _name_anchors(source_paths: list[Path], target_paths: list[Path]) -> list[tuple[int, int, str, str]]:
    sleft = set(range(len(source_paths)))
    tleft = set(range(len(target_paths)))
    candidates: list[tuple[int, int, str, str]] = []

    for si, tj, key in _unique_key_pairs(source_paths, target_paths, sleft, tleft, _exact_name_key):
        candidates.append((si, tj, key, "name_exact"))
        sleft.discard(si); tleft.discard(tj)

    # If exact names differ (006.png vs p-006.jpeg), use a unique normalized page
    # number as the next-best filename anchor.
    for si, tj, key in _unique_key_pairs(source_paths, target_paths, sleft, tleft, _page_number_key):
        candidates.append((si, tj, key, "name_number"))

    return _monotonic_subset(candidates)


def _segments(n: int, m: int, anchors: list[tuple[int, int]]) -> list[tuple[list[int], list[int]]]:
    """Return source/target ranges between already locked monotonic anchors."""
    clean = sorted(anchors)
    out: list[tuple[list[int], list[int]]] = []
    prev_s = prev_t = -1
    for si, tj in clean + [(n, m)]:
        out.append((list(range(prev_s + 1, si)), list(range(prev_t + 1, tj))))
        prev_s, prev_t = si, tj
    return out


def _make_direct_pair(
    source_paths: list[Path], target_paths: list[Path], si: int, tj: int,
    method: str, detail: str = "",
) -> PagePair:
    if method == "name_exact":
        confidence, score = 1.0, 0.0
        reasons = ["pairing=name", "name_match=exact"]
    elif method == "name_number":
        confidence, score = 0.995, 0.005
        reasons = ["pairing=name", "name_match=page_number"]
    elif method == "order":
        confidence, score = 0.99, 0.01
        reasons = ["pairing=order", "order_match=natural_sort"]
    else:
        confidence, score = 0.98, 0.02
        reasons = [f"pairing={method}"]
    if detail:
        reasons.append(f"pair_key={detail}")
    return PagePair(
        source_path=str(source_paths[si]), target_path=str(target_paths[tj]),
        source_index=si, target_index=tj, confidence=confidence, score=score, reasons=reasons,
    )


def pairing_method(pair: PagePair) -> str:
    for reason in pair.reasons:
        if reason.startswith("pairing="):
            return reason.split("=", 1)[1]
    return "smart"


def pair_directories(
    source_dir: str | Path, target_dir: str | Path, config: PairingConfig | None = None
) -> tuple[list[PagePair], list[str], list[str]]:
    """Pair pages with visual matching by default and optional deterministic anchors.

    Priority when both optional switches are enabled:
      1. unique normalized filename / page-number anchors;
      2. natural-sort order inside equal-length intervals between those anchors;
      3. legacy visual+order sequence alignment only for unresolved intervals.

    This makes well-organized books near-instant to pair while still keeping the
    old robust visual matcher for missing/extra/reordered pages.
    """
    cfg = config or PairingConfig()
    source_paths = list_images(source_dir)
    target_paths = list_images(target_dir)
    n, m = len(source_paths), len(target_paths)
    if n == 0 or m == 0:
        return [], [str(p) for p in source_paths], [str(p) for p in target_paths]

    pairs: list[PagePair] = []
    anchors: list[tuple[int, int]] = []

    if bool(getattr(cfg, "prefer_name_pairing", False)):
        for si, tj, key, method in _name_anchors(source_paths, target_paths):
            pairs.append(_make_direct_pair(source_paths, target_paths, si, tj, method, key))
            anchors.append((si, tj))

    # Order mode is deliberately conservative when there are extra pages. Only lock
    # an interval when the two sides contain the same number of still-unmatched pages.
    # Unequal intervals are left to the visual matcher so one insert does not shift
    # every subsequent page by one.
    if bool(getattr(cfg, "prefer_order_pairing", False)):
        order_new: list[tuple[int, int]] = []
        for sseg, tseg in _segments(n, m, anchors):
            if sseg and len(sseg) == len(tseg):
                for si, tj in zip(sseg, tseg):
                    order_new.append((si, tj))
                    pairs.append(_make_direct_pair(source_paths, target_paths, si, tj, "order"))
        anchors.extend(order_new)
        anchors.sort()

    paired_s = {p.source_index for p in pairs}
    paired_t = {p.target_index for p in pairs}
    unmatched_s: set[int] = set()
    unmatched_t: set[int] = set()

    # Smart fallback stays inside deterministic-anchor intervals. That prevents a
    # visually similar page before a filename anchor from crossing to the other side.
    for sseg, tseg in _segments(n, m, anchors):
        sseg = [i for i in sseg if i not in paired_s]
        tseg = [j for j in tseg if j not in paired_t]
        if not sseg:
            unmatched_t.update(tseg)
            continue
        if not tseg:
            unmatched_s.update(sseg)
            continue
        source_fps = [fingerprint_image(source_paths[i], local) for local, i in enumerate(sseg)]
        target_fps = [fingerprint_image(target_paths[j], local) for local, j in enumerate(tseg)]
        smart_pairs, us, ut = pair_fingerprints(source_fps, target_fps, cfg)
        for p in smart_pairs:
            osi = sseg[p.source_index]
            otj = tseg[p.target_index]
            pair_confidence = float(p.confidence)
            pair_reasons = list(p.reasons)
            if (
                bool(getattr(cfg, "remake_pair_verifier_enabled", False))
                and pair_confidence <= float(getattr(cfg, "remake_pair_verify_confidence_ceiling", 0.78))
            ):
                try:
                    evidence = verify_remake_pair(
                        read_image(source_paths[osi]), read_image(target_paths[otj]),
                        max_side=int(getattr(cfg, "remake_pair_verify_max_side", 1000)),
                        ratio_test=float(getattr(cfg, "remake_pair_verify_ratio_test", 0.76)),
                        min_good_matches=int(getattr(cfg, "remake_pair_verify_min_good_matches", 18)),
                        min_inlier_ratio=float(getattr(cfg, "remake_pair_verify_min_inlier_ratio", 0.45)),
                        min_spatial_coverage=float(getattr(cfg, "remake_pair_verify_min_spatial_coverage", 0.08)),
                        max_median_error=float(getattr(cfg, "remake_pair_verify_max_median_error", 4.5)),
                    )
                    if evidence.confirmed:
                        cap = float(getattr(cfg, "remake_pair_verify_max_boost_confidence", 0.92))
                        pair_confidence = max(pair_confidence, min(cap, float(evidence.confidence)))
                        pair_reasons.append("remake_verify=confirmed")
                        pair_reasons.append(f"remake_inliers={evidence.inliers}/{evidence.good_matches}")
                        pair_reasons.append(f"remake_coverage={evidence.spatial_coverage:.3f}")
                        pair_reasons.append(f"remake_error={evidence.median_reprojection_error:.3f}")
                    else:
                        pair_reasons.append(f"remake_verify={evidence.diagnostics.get('reason', 'inconclusive')}")
                except Exception as exc:
                    # Pairing must never fail because the optional verifier failed.
                    pair_reasons.append(f"remake_verify=soft_failure:{type(exc).__name__}")
            pairs.append(PagePair(
                source_path=str(source_paths[osi]), target_path=str(target_paths[otj]),
                source_index=osi, target_index=otj, confidence=pair_confidence, score=p.score,
                reasons=pair_reasons,
            ))
            paired_s.add(osi); paired_t.add(otj)
        unmatched_s.update(sseg[i] for i in us)
        unmatched_t.update(tseg[j] for j in ut)

    # Anchored pairs are already accounted for; include any page that somehow did not
    # enter a segment (defensive against future non-monotonic custom anchors).
    paired_s = {p.source_index for p in pairs}
    paired_t = {p.target_index for p in pairs}
    unmatched_s.update(i for i in range(n) if i not in paired_s)
    unmatched_t.update(j for j in range(m) if j not in paired_t)

    pairs.sort(key=lambda p: (p.target_index, p.source_index))
    return pairs, [str(source_paths[i]) for i in sorted(unmatched_s)], [str(target_paths[j]) for j in sorted(unmatched_t)]
