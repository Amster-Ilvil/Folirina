# v0.8.22

- Separate raster `applied` state from independent content-completeness verification.
- Add source-ink coverage and target-language residual metrics per transfer record.
- Block cross-rendition/complex records in QA when content is incomplete or unverifiable.
- Add cross-rendition high-white balloon refinement/false-positive rejection.
- Add strict saturated burst-container overlap + source-brightness guards.
- Use structural dense/local flow for candidate discovery only; use global registration for final supplemental Chinese glyph pixels when available.
- Upgrade `transfer_audit.json` schema to v2 with geometry/content counters and verification scope.
- Refresh offline self-test invariants for multi-region structural detection.
- Add v0.8.22 synthetic regression tests and a generic real-pair acceptance runner.
