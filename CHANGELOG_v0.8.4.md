# v0.8.4 (2026-08-11)

## 本次改进

### 1) 摄影页优先高清重排，减少模糊
- 新增 photographed-page OCR 优先高清重排策略：
  - 当 `mask_replace` 处理摄影/旧版翻拍页，且 OCR 证据可用时，已匹配的对白气泡会优先走 **HD reletter**，而不是直接发布手机拍摄文字像素。
  - 这能显著降低“替换后字体模糊”的情况。
- 直接蒙版转移仍保留，用于：
  - OCR 缺失区域
  - 非标准区域
  - 作为几何/身份保底层

### 2) 新增 mask_replace 页面“手动补字”闭环
- 对 `source_text_region_clipped_at_page_edge` 这类 **源图边缘裁切** 导致的残缺气泡：
  - 自动输出 `review_overrides.template.json`
  - 支持在 `review_overrides.json` 中填入 `manual_reletter` 内容
  - `apply_review_page()` 现可直接在 `mask_replace` 页面上执行 **清底 + 高清重排补字**
- 这解决了“不能全替换”的最后一公里问题：
  - 自动做不全时，不再假成功
  - 可以通过最小人工输入补完成品

### 3) review 应用增强
- `mask_replace` 复核页现在支持：
  - `target_bubble_id`
  - `target_bbox`
  两种方式定位补字区域。
- 即使目标泡泡不在 OCR 泡泡列表里，只要 project meta 中存在待补字记录，也能定位并完成补字。

## 回归测试
- 全量 pytest：35/35 通过。
- 新增测试：`test_mask_replace_review_can_apply_manual_reletter`
- 更新摄影页回归测试，验证 OCR 存在时可对已转移气泡执行高清重排。
