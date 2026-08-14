# Unseeded Completion Dedup (v0.9.0a4)

A real colour target regression page exposed that the OCR-free unseeded white completion pass still executed even when the primary photo-pair mask transfer had already finished cleanly. That produced duplicate rigid-container records on already handled bubbles.

## Fixes
- Completion now runs only when there is an actual gap: no records, unapplied records, or review-required records.
- Already applied target bubbles are passed into the completion detector as `existing_target_bubbles`, so only missed containers remain eligible.

## Effect on the new real pair
- before: 10 applied mask-transfer records (6 legitimate + 4 duplicate rediscoveries)
- after: 6 applied records, all legitimate
- final image: unchanged in authored regions
- risk: lower accidental overpaint / lower runtime / cleaner review evidence
