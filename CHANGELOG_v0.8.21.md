# v0.8.21 — Complex Text Recovery / Transfer Audit

## Processing fixes

- Added structural detection for text changes on coloured/dark backgrounds, covering burst balloons, open captions and SFX-like text that cannot be represented as a normal closed white balloon.
- Structural recovery is supplemental and non-overlapping: ordinary precise bubble transfer remains the first route, while free/complex regions use an ink-only compositor.
- Complex/open regions erase only selected Japanese glyph component groups plus a tiny antialias fringe, then composite only registered source Chinese raster ink. No rectangular coloured-background copy and no OCR re-lettering is performed.
- Added OCR-guided component transfer for matched OCR text regions missed by paired-diff/container geometry. OCR supplies only correspondence/gating; source raster glyphs remain authoritative in strict Precise Mask mode.
- Low-confidence open/complex/OCR-guided regions are retained as reversible review candidates rather than silently discarded.
- Target clear-mask expansion now accepts tall narrow vertical Japanese columns and wide short captions, while rejecting long panel/balloon outlines.
- Fixed a duplicate `_component_from_seed()` source lookup in the raw paired-diff path.

## Per-page evidence

Every processed page now writes explicit review artifacts:

- `source_original.png`
- `target_original.png`
- `target_clear_mask.png`
- `chinese_transfer_layer.png`
- `final.png`
- `review_preview.png`
- `transfer_audit.json`

`transfer_audit.json` records registration confidence, candidate counts/kinds, OCR evidence, ambiguous matches, applied/rejected/review counts, OCR-guided/complex-text counts, target-clear/write pixel counts, rejection reasons and QA issue codes.

## Compatibility

- New paired pages still default to `content`.
- No post-pair automatic bubble/text-box scan was reintroduced.
- Page-manager exclusions remain passthrough pages and still publish one final page per paired target.
- `strict_mask_replace_no_ocr_reletter` remains authoritative.
- v0.8.16 photo glyph-footprint rescue remains enabled and covered by regression testing.
