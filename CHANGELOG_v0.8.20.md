# v0.8.20

## Page Manager

- Removed the post-pair automatic balloon/text-box scan from the Studio UI and pairing workflow.
- Every newly paired page now defaults to `content`. Legacy automatic `auto_no_text` marks migrate back to default content.
- “恢复自动” is replaced by “恢复正文”; resetting a page no longer launches a scanner.
- Manual non-content page types remain authoritative and are emitted as unchanged HD target pages.

## Transfer admission

- Added a runtime source-text gate: when source OCR actually ran and no OCR-backed Chinese speech/narration container exists, the page becomes an unchanged-target passthrough instead of being forcibly inpainted/replaced.
- The gate is fail-open when OCR is disabled or deliberately skipped, avoiding false-negative page loss.
- OCR evidence may be associated with precise paired-diff candidate geometry, preserving photographed and edge-clipped valid translations even when the ordinary seeded bubble detector cannot reconstruct the whole bubble.
- Runtime no-text passthrough results are resume-safe and remain valid content results while the job fingerprint matches.

## Thumbnail performance

- Replaced whole-book eager thumbnail decoding with viewport-only lazy loading.
- Thumbnail decoding pauses during active wheel/trackpad scrolling and resumes after an 80 ms quiet period.
- Loads at most three scaled images per UI event-loop slice, with one-row prefetch.
- Added a bounded 128-entry LRU of already-scaled thumbnail pixmaps; page-type changes reuse image pixels instead of decoding files again.

## Output integrity

- Skipped pages continue to write their unchanged HD target into `final/`; skipping transfer never removes a page.
- Added collision-safe final filenames when multiple target pages share the same output stem.

## Regression

- 92/92 pytest tests pass.
- Precise-mask acceptance benchmark: 24/24, pass rate 1.0.
