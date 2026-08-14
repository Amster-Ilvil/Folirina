# v0.5 成对差异蒙版替换 Plan（已实现）

目标：当旧中文版与高清日文版是同源/近同源页面时，不依赖 OCR 猜“哪些气泡被翻译”，而直接利用两页之间的真实视觉变化，提取**只有中文翻译实际出现的气泡与文本框**。

## 路径 A：同源 / 近同源页面（优先）

1. 页面配准：SIFT / LightGlue / LoFTR 得到 source → target 几何变换。
2. 配准置信度门槛：低于阈值禁止 paired-diff，回退通用路径。
3. 轻度模糊后做灰度差，利用页面噪声分位数动态提高阈值，避免 JPEG / 扫描噪声被当成翻译。
4. 对变化字符做 morphology 聚合，形成候选区域。
5. 候选区域分别吸附到旧中文版与目标日文版的封闭白色气泡 / 文本框实例。
6. 检查 source/target 实例 IoU、面积、变化密度与一对一去重。
7. 只保留真正发生翻译变化的实例；未翻译气泡、拟声词和画面元素自然被排除。
8. 同尺寸近 identity 页面启用 `pixel-exact`：不 resize、不 feather、不背景归一化，直接复制旧中文版目标实例的像素。
9. 替换蒙版之外严格保留高清日文母版。
10. 输出 `debug_paired_diff.png`、`mask_transfer_layer.png`、`mask_transfer_mask.png`、结构 JSON 与 QA。

## 路径 B：真正的低清旧版 → 高清新版（回退）

当页面裁边、重绘、扫描形变或分辨率差异导致 paired-diff 不安全时：

`全局配准 → MangaLens / Sidecar 气泡实例 → 中文 OCR / 结构辅助 → Hungarian 实例匹配 → 局部 bbox/ECC → 可选超分 → 目标 mask 裁切 → QA`

超分只提高源 patch 采样质量，不参与几何尺寸计算；最终覆盖范围始终由目标高清气泡/文本框 mask 决定。

## 安全门槛

- 页面配准置信度
- source/target mask IoU
- 目标覆盖率
- spill ratio
- 局部缩放异常
- 匹配歧义
- 气泡边线保护

任一关键门槛失败：纯蒙版模式进入 Review；智能混合模式回退高清重排。

## 用户提供真实页面回归

本次测试自动提取 **5 个**实际翻译区域：4 个对白气泡 + 1 个右下说明文本框；顶部 3 个未翻译小气泡与画面拟声词未被选择。

结果：

- 5/5 应用成功
- QA 0 error / 0 warning
- transfer mask 内：100% 与中文参考逐像素一致
- transfer mask 外：100% 与高清日文母版逐像素一致

详见 `REAL_PAIR_ACCEPTANCE_V05.md`。
