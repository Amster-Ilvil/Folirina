# Folirina Web Lite — 2026-11-03 latest main sync (v0.3.9)

Baseline: `37a3fff6bfac13f7aee551e06cd7ff1584d28a96` (2026-11-01).

## Browser-safe capabilities adapted from latest desktop main

- Local verified-flat repair of *manually reviewed* TARGET erasure mask islands (new `verified_flat_repair.js`) using the nearby untouched TARGET pixel distribution. Flat components must satisfy strict colour consistency; uncertain textured components retain the previous four-ray interpolation. All writes remain inside the explicitly approved masks.
- Preview-only support for three or more aligned white-outlined glyph candidates on neutral black/white manga, based on the desktop `outlined_text_mask.py` advances. A single bright eye or isolated white highlight is rejected. Colour manga preview detection continues unchanged.
- All browser features preserved: local import, page pairing, affine/homography, safe clipped brush transfer, reversible workspace, manual red/green mask review and export.
- UI version v0.3.9; service worker cache v24; workspace export provenance now references latest main SHA without changing v1 workspace import format.

## Desktop-only capabilities not ported

Koharu/AI inference, OCR, LaMa, Python image-processing runtimes, native model workers and desktop window scheduling cannot run on static GitHub Pages without a separately running local/native service. The browser cannot claim functional equivalence to the desktop version. No automatic destructive approval is introduced.

## Validation scope

JavaScript syntax and deterministic synthetic pixel tests for flat vs textured backgrounds, mask-only write authority, neutral glyph runs, isolated highlights and existing coloured glyph evidence. These tests do not guarantee art-quality accuracy on every comic panel.
