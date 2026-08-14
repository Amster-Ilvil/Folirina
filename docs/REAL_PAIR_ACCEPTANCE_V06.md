# v0.6 真实成对页面回归

测试日期：2026-08-11

输入仍为用户此前提供的同一漫画页日文原图 / 已嵌字中文参考图。原始测试图片不进入发布 ZIP。

## 结果

- Auto route：`fast-phase-identity`
- 配准置信度：`0.953869`
- 自动翻译区域：5
- 实际应用：5/5
- QA：0 error / 0 warning
- transfer mask 内与中文参考逐像素一致：100%
- transfer mask 外与日文母版逐像素一致：100%
- whole-page MAE vs 中文参考：`0.0035695`
- v0.6 final vs v0.5 final：逐像素完全一致

## 降本结果

相同页面仅测配准阶段，5 次中位：

- v0.6 Auto cheap path：`0.01605 s`
- 强制 OpenCV SIFT：`1.08333 s`
- 加速：约 `67.5×`

快速路线只在严格 blurred correlation / phase response / shift 门槛通过时使用；否则自动支付 SIFT 成本。困难页才继续升级 LightGlue / LoFTR，因此这里的 67.5× 不能泛化为所有跨版本页面。

## 结论

v0.6 在这组真实同源页面上实现了“算法成本下降，但最终像素结果不变”。这是 cheap-first 路线的核心验收条件，而不是单纯追求更快的配准分数。
