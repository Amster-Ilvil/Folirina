# Publication Gate

## Product contract

Manga-HD-Translation-Transfer is a **Chinese translated old edition → Japanese HD master migration tool**. It is not a Japanese-to-Chinese machine translator.

Hard contracts:

| Metric | v1.0 gate |
|---|---:|
| Page pairing accuracy | ≥ 99.5% |
| Ordinary dialogue/narration identity matching accuracy | ≥ 99% |
| Visible Japanese residue on auto-pass pages | 0 |
| Bubble-border / character-line damage on auto-pass pages | 0 |
| Glyph safe-area overflow | 0 |
| Explicit Direct silently becoming Mask | 0 |
| Initial Review-rate target | ≤ 15% after a representative baseline exists |

`review_rate` is a production-efficiency target, not permission to auto-pass unsafe pages. Safety gates dominate automation rate.

## Real paired benchmark layout

See `benchmarks/README.md`. Private/copyrighted pages should live outside the release tree and be supplied to the gate by path.

Required coverage across the full private benchmark should include at least:

- `same_size`
- `scale_diff`
- `crop_diff`
- `photo_scan`
- `color_burst`
- `sfx`
- `vertical_text`
- `open_text`
- `splash`

Target v1.0 validation set: **3–5 works / 100–300 paired pages**.

## Running the gate

```bash
PYTHONPATH=src python scripts/publication_gate.py /path/to/private-benchmarks \
  --config config.publication.json \
  --output gate_output
```

Outputs:

- `publication_gate.json`: machine-readable metrics and per-page evidence.
- `GATE_REPORT_<version>.md`: release comparison report.
- `failures/<work>/<page>/`: compact failure archive containing source/target/final, overlays, masks, project and QA evidence.

## Metric definitions in the v0.9 gate runner

- **Pair accuracy**: automatic `pair_directories` mapping versus `labels.json` expected source→target mapping.
- **Identity match accuracy**: required golden unit-pairs found among one-to-one `UnitMatch` results. N/A until unit IDs are labeled.
- **Japanese residual**: maximum target-only residual ratio reported by transfer records.
- **Border damage**: pixels changed in a narrow guard ring immediately outside the transfer mask.
- **Safe-area overflow**: maximum transfer-record spill ratio.
- **Review rate**: pages with Review/Reject transfer records or processing failure.
- **Direct fallback violation**: an explicit `direct_patch` page producing Mask artifacts.

The runner is deliberately conservative. It does not claim the project meets publication gates unless a real benchmark set is actually supplied and passes.

## Failure archive requirement

Every hard failure is archived with enough evidence to reproduce and force-review it. A release candidate must never replace a failed page silently.

## v0.9.0-alpha.1 implementation status

Implemented now: benchmark discovery, automatic pair-map comparison, per-page Pipeline execution, residual / border / spill / review / timing metrics, Markdown+JSON reports, failure evidence archive.

Not claimed yet: the repository does **not** include a 100+ page private benchmark, so this alpha does not claim the hard v1.0 numbers have been achieved. Those numbers must be established by running the gate on real paired books.


## v0.9.0-alpha.2 real-pair acceptance

`publication_gate.py` remains the book-level benchmark gate. For a single private failure pair, use `scripts/real_pair_acceptance.py` with a manually confirmed expected-region JSON. The single-pair gate additionally checks that no final pixel changes outside the active transfer mask and can cap write pixels outside approved translation bboxes.


## v0.9.0-alpha.3 protected target-border ring

Rigid white-container transfers now expose `meta.target_border_preservation`. Publication acceptance treats any non-zero `changed_after_restore` as a failure/review condition. This catches border changes *inside* the container geometry that a simple outside-write-mask damage metric can miss. Gap-fill is constrained to the same border-safe envelope and may not grow back across it.
