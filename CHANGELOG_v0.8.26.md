# v0.8.26 — 实心气泡内部 / 日文残留修复

## 根因修复

v0.8.25 把亮色区域检测得到的 mask 直接当成完整气泡内部。小对白中，如果日文字连接/切入亮区边缘，mask 会出现由日文字造成的黑色缺口。随后：

- 清除阶段跳过这些黑色缺口，因此原日文残留；
- 中文迁移阶段又被同一个缺口裁切，可能缺笔；
- QA 使用同一错误 mask 检查残留，可能出现 `target_residual_ratio=0` 的假通过。

v0.8.26 将“检测 mask”和“真正可写入的容器内部”拆开。

## 新增：solid container interior

- `rigid_container_solidify_enabled`
- `rigid_container_solidify_radius_ratio`
- `rigid_container_solidify_min_radius_px`
- `rigid_container_solidify_max_radius_px`
- `rigid_container_solidify_boundary_guard_px`
- `rigid_container_solidify_max_added_ratio`

流程：

1. 原始 detector mask 仍负责定位容器；
2. 在容器 bbox 内闭合窄文字缺口；
3. 填补真正的内部孔洞；
4. 用原始 mask 的凸包内缩边界作为保护区，只允许在安全内侧补像素；
5. 最终 `clear mask` 与 `Chinese raster clip mask` 使用这个实心内部。

这样能填掉日文字造成的缺口，同时不把 065 上方爆发式黑色气泡边缘刷白。

## 额外安全修复

- OCR-free 白色容器补漏的最小 dark ratio 从 `0.008` 提高到 `0.012`，避免 006 上方屋檐/天空白块被当成对白框。
- `target_residual_ratio` 现在在实心容器内部统计，不能再因为日文恰好落在 detector mask 黑洞里而假装为 0。
- source 中文栅格也使用实心容器内部裁切，减少中文字因源侧 mask 缺口而缺笔。

## 性能

在相同测试机、OCR=none、cache disabled 的实页验收中：

- 006：v0.8.25 `31.32s` → v0.8.26 `23.14s`，约快 26%。
- 065：约 `20.00s`。
- 066：约 `22.23s`。
- 007：约 `30.43s`。

性能提升主要来自 OCR-free 补漏候选更严格，减少无效结构候选进入后续刚性迁移。
