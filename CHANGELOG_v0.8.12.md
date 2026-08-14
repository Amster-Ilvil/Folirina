# v0.8.12

## 修复

- 修复 **清晰旧中文版保留原字号/分列** 模式下，局部转移虽然清晰但实际发生了
  **列缺失 / 行缺失 / 首字缺失** 时仍被错误保留的问题。
- 新增 **保留版式完整性检查**：在跳过 OCR 高清重排前，会对比
  源中文版与目标页转移结果的版式画像（列数 / 行数 / 墨迹占比 / 字距）。
- 若检测到版式塌缩（如 2 列变 1 列、行数明显减少、墨迹宽度异常缩小），
  将自动放弃“直接保留旧中文字形”，改走 OCR 高清重排 fallback，
  避免发布缺字、漏字、半列丢失的结果。
- 为该回归补充测试：覆盖 **竖排列缺失** 场景。

## 验证

已通过：

- `tests/test_layout_preservation.py`
- `tests/test_no_apple_vision_legacy.py`
- `tests/test_pairing.py`
- `tests/test_photo_pair_regressions.py`
- `tests/test_paired_diff_photo_v08.py`
- `tests/test_mask_transfer.py`
