# Colour Burst / Artwork False-Positive / Residual Safety

## Publication rule

Colourization differences are not translation evidence.  A black-and-white SOURCE
and colour TARGET can differ strongly in RGB/brightness while keeping the same line
art.  Automatic writes therefore require *ink identity change*, not merely a large
photometric difference.

### v0.9.0a2 safeguards

1. **Unseeded white-container completion**
   - still requires white-container geometry;
   - now also compares registered SOURCE/TARGET dark-ink identity;
   - rejects same-artwork regions when ink remains highly overlapping;
   - rejects extreme SOURCE/TARGET ink-density imbalance (common in foliage,
     paving and clothing rather than translated text).

2. **Structural free/complex-text supplement**
   - computes an OCR-free ink-change score on the aligned SOURCE and TARGET;
   - complex/open regions with mostly identical ink are treated as artwork;
   - sparse SOURCE evidence is not allowed to erase TARGET-only headers/signage.

3. **Coloured burst transfer**
   - coloured/halftone TARGET geometry is preserved;
   - only translated ink / verified target-glyph clear pixels are changed;
   - Direct Patch remains disallowed when preserving TARGET colour texture would
     require mask semantics.

## Real-pair regression (user-provided, not packaged)

A 2048×1440 monochrome Chinese SOURCE and 1600×1117 colour Japanese TARGET were
used as a private real-pair regression.  The page contains a yellow burst panel,
three ordinary text containers, white clothing, foliage and textured paving.

Baseline v0.9.0a1 behavior:
- 8 transfer records;
- 4 true translated regions + 4 false artwork writes;
- 21,849 write-mask pixels fell outside the four manually confirmed text regions;
- one visible rectangular structural artifact appeared on the paving;
- white clothing was also modified by unseeded-white completion.

After the v0.9.0a2 gates:
- 4 transfer records, all four manually confirmed translated regions;
- SAFE=4, REVIEW=0, REJECT=0 (OCR disabled for this sandbox run);
- TARGET residual ratio reported 0.0 for all four detected translation regions;
- only 43 feather/edge pixels fall outside the four manually confirmed bboxes
  (acceptance allowance: 100);
- all previously identified shirt/paving false-positive ROIs are pixel-identical
  to TARGET;
- no pixel outside the actual transfer mask differs from TARGET;
- yellow-region HSV median remains H=30, S=135, V=255 before/after, confirming
  that the colour burst background was preserved rather than replaced by SOURCE white.

The input images are deliberately not stored in the release ZIP. Use
`scripts/real_pair_acceptance.py` with private/local files to reproduce this gate.
