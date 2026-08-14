# v0.8.15

## 精准蒙版模式语义修复

- **精准蒙版替换不再允许 OCR 改写最终文字。**
- Apple Live Text / OCR 在该模式中只用于区域检测、匹配证据和复核。
- 禁止以下旧自动路径在精准蒙版中进入最终输出：
  - 漏区 OCR reletter fallback；
  - 模糊区 OCR reletter fallback；
  - OCR 可用时自动优先高清重排。
- 因此 `……`、`!?`、引号、破折号、全/半角符号及特殊字形均来自旧中文版真实图像，不再被 transcript 规范化或改写。
- 老配置即便保存过旧的 fallback=true，`strict_mask_replace_no_ocr_reletter=true` 仍会强制执行模式边界。
- 需要 OCR 改字时使用 **智能混合 / 高清重排**，或在复核工作台手动“重新编辑并高清排字”。

## GUI

- 更新精准蒙版说明和低清保护说明，明确 OCR 不会重写最终文本。
- 工作台同步时不再重新打开旧 OCR fallback 标志。
