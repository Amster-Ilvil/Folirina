# v1.0.0 真实用户页验收报告

> 测试素材：本轮会话中用户提供的同页黑白中文版 / 彩色日文版截图。原始图片不打包进发行 ZIP。

## 配准

- SOURCE：黑白中文版截图（1270×911）
- TARGET：彩色日文版截图（1297×911）
- registration confidence：`0.8092560539752747`
- reprojection error：`0.8404296636581421 px`
- inlier ratio：`0.7551546391752577`
- spatial coverage：`0.668906221198073`

## Direct Patch

旧版默认 `source_direct_min_registration_confidence=0.82` 会直接拒绝本页，尽管几何误差已经小于 1 px。v1.0.0 使用受约束 cross-rendition gate：

- 黑白 SOURCE / 彩色 TARGET 条件：满足
- relaxed gate：`true`
- Direct Patch：`used=true`
- accepted records：`2`
- accepted white containers：`2`
- artwork-like candidates：继续拒绝，不因放宽配准而被写入

## Reveal / 紫色爆发框

真实 TARGET ROI：`[95, 965, 320, 1190]`。

- cross_rendition：`true`
- edge correspondence radius：`4 px`
- source text mask pixels：`7915`
- target clear mask pixels：`11508`
- ROI 内实际改变像素：`12519`
- ROI 外改变像素：`0`

新版流程：

1. SOURCE/TARGET 结构差分只生成候选文字边缘种子；
2. 在局部 black-hat / top-hat 笔画证据中受限生长，恢复完整字形内部；
3. 拒绝贴边长刺和大面积低填充组件；
4. 清理 TARGET 日文笔画；
5. SOURCE 背景先重建，再根据文字组件极性只传递黑字/白字的局部增量；
6. 最终背景始终来自 TARGET 彩图。

## 自动测试

`PYTHONPATH=src pytest -q` → **48 passed**。

新增回归包括：

- 黑白→彩图低于 0.82 时，只有几何证据同时达标才使用 relaxed gate；
- 同样 0.80 confidence 但 inlier / reprojection / coverage 不达标时必须拒绝；
- 暗字 delta composite 不允许生成亮色 halo；
- 原有 Direct / Mask / Review / Reveal 测试全部继续通过。
