# v0.8.35 吸收与落地

## 1. Direct / Mask 匹配与失败处理
- Matching: centroid + overlap + projected IoU + text-length + shape + order + kind + registration-confidence penalty.
- Diagnostics: unmatched source Top-3 targets, rejected-over-max-cost, ambiguity evidence, force actions.
- Planner: evidence + force_actions.

## 2. Dual SOURCE
`dual_source.secondary_source_dir` 已进入真实 Direct 执行链：secondary load → same-page verification → registration → source detector chain → Direct plan → invariant QA → selection.
主 SOURCE 保持 authority；secondary 只在 Direct 路径安全/显式 Direct 可用时成为 raster source。

## 3. White bubble fast clear
`_fast_dark_pixel_clear` 仅对白度达标区域清暗像素，局部纸色填充 + tiny Telea；纹理/彩色区域拒绝 fast path。

## 4. Replace Translation interoperability
Schema: `manga-hd-transfer/replace_translation/v1`
Compatible marker: `manga-translator-ui/replace_translation`
Region evidence includes translated_text, source/target bbox, overlap, matched, relation, confidence and cost.

## 5. Review actions
Web Review supports page-level force Direct/Mask (real page rerun) and unit-level force_match/skip_unit.
