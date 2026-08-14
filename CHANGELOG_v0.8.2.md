# v0.8.2

- Added antialiased `photo-crisp-ink` reconstruction for photographed Chinese lettering.
- Removed the small-photo-bubble hard rejection; small regions now try deterministic ink recovery.
- Added source-mask salvage expansion for slightly under-segmented photographed bubbles.
- Relaxed photographed target-driven geometry gates while retaining target-mask write constraints.
- Prevented photographed balloon borders from being copied into the HD master.
- Downgraded weak pairing/OCR-evidence blockers to warnings only after strong registration + fully applied photo geometry.
- Added regression tests for small photographed text, crisp transfer, and QA geometry verification.
- 29/29 tests pass.
