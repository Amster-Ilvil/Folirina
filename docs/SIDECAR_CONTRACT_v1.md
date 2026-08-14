# Sidecar Contract v1

This release formalizes the lightweight sidecar contract used by OCR / bubble / replace_translation workflows.

## OCR
- `page.ocr.json`: `{ "blocks": [...] }` or a list of block rows
- each block may include `polygon`, `text`, `confidence`, `kind`, `bubble_id`, `mask_path`

## Bubbles
- `page.bubbles.json`: `{ "bubbles": [...] }` or a list
- each bubble may include `polygon`, `kind`, `mask_path`, `safe_mask_path`

## Additional SOURCE evidence
- `page.replace_sources.json`: `{ "sources": [{"path": "...", "kind": "high_translated"}] }`
- current v0.8.34.5 uses these candidates for Direct Patch retry in auto/direct workflows.

## Replace Translation export bundle
Every processed page may emit `replace_translation/` containing:
- `source_ocr.json`
- `target_ocr.json`
- `unit_matches.json`
- `summary.json`
