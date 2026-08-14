# v0.8.1 + v0.8 merge notes

This package uses v0.8.1 as the primary implementation and absorbs the non-conflicting v0.8 paired-transfer capabilities.

## Kept from v0.8.1

- `photo_pair` photographed-page route and its conservative geometry gates.
- Photo illumination normalization / ink reconstruction / OCR re-letter fallback.
- Source-resolution-preserving rectified OCR.
- Publication-blocking QA for empty mask-replace output and missing OCR evidence.
- macOS Apple Vision OCR defaults and current GUI wording/behavior.

## Restored from v0.8

- Structural ink mismatch detector for photographed pages.
- Low-frequency OpenCV DIS dense-flow local alignment.
- Barrier/enclosed-region detection.
- Free-text / SFX residual-region detection.
- Target-driven paired-diff compositor (`transfer_paired_diff_regions`).
- v0.8 tuning fields, benchmark, plan document, and regression coverage.

## Conflict policy

The v0.8.1 route remains authoritative. The v0.8 structural engine is isolated in `paired_diff_v08.py` and is used as a fallback only when the v0.8.1 high-noise photo route cannot find a safe region. Structural fallback does not skip OCR, preserving v0.8.1 completeness safeguards.

## Validation

- Python compileall: passed.
- Pytest: 25/25 passed.
- Built-in CLI selftest: passed; 0 QA errors, 0 warnings.
- Config example parsing and compatibility API checks: passed.
- Release audit: passed.
