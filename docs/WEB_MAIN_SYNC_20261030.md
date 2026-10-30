# Folirina Web Lite — 2026-10-30 main sync (v0.3.7)

Base: desktop/core commit `a1516de602e99e5098431ff32864c0578d797a9f`.

## Adapted to a static browser (no local Python process)

- Optional publication encoding: PNG (lossless default), WebP or JPEG. Browser-native codec; user-controlled quality for WebP/JPEG. TARGET resolution is unchanged.
- The publication choice applies to single-page and whole-book output **only**. Original source/target pictures and reversible workspace edits remain uncompressed in workspace ZIP.
- Optional preview-only black-ink/white-outline TARGET candidate evidence on colored artwork, derived conservatively from `outlined_text_mask.py`. Default OFF. All red TARGET masks and green SOURCE masks require manual review before any destructive edit.
- The existing light palette, 3-column editor, manual zoom, registration, review safety, and lossless workspace structure are preserved.

## Not ported (desktop-only capabilities)

The Oct 30 desktop commit also changes Koharu / OCR / LaMa and Python OpenCV routines, smooth colored-paper fitting, shadow-free white-paper enhancement, local Upscayl and Caesium CLI invocation, and restart/recovery logic. They **do not** run as GitHub Pages features. WebP/JPEG use browser codecs, not Caesium. No native super-resolution is claimed.

## Validation

- Chromium rendered the white 3-column UI at 1440×900 and checked central editor width.
- Browser image encoding successfully produced proper PNG/WebP/JPEG MIME signatures and preserved image dimensions.
- Full UI boot, import, single page exports, ZIP export, and lossless workspace ZIP flow were tested with synthetic 480×650 images using an in-memory module loader because direct localhost navigation is blocked in this environment.
- White-outline mask unit tests: colored outlined glyph accepted; a broad white speech panel and plain colored background rejected.
- These tests do **not** prove publication-grade accuracy on a real comic. Users must review transferred lettering and background structure.
