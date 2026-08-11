# Implementation plan and status

## 已实现 P0

- [x] 测试/合成基线框架。
- [x] 页面视觉指纹 + 保序列配对 + extra/missing page。
- [x] SIFT/ORB + RANSAC similarity/affine/homography。
- [x] LightGlue optional backend。
- [x] LoFTR optional fallback。
- [x] PP-OCRv5 adapter。
- [x] 旧中文低置信度多次 crop 复识与候选保留。
- [x] 外部 OCR/text segmentation sidecar。
- [x] seeded-white bubble fallback。
- [x] 外部 bubble instance/safe mask sidecar。
- [x] bubble-level TextUnit 聚合。
- [x] Hungarian cross-edition identity matching。
- [x] one-to-many / many-to-one detection + Review gate。
- [x] 像素 text mask 优先、polygon fallback。
- [x] bubble border protection + adaptive dilation。
- [x] solid / OpenCV / external LaMa inpainting。
- [x] CJK 动态规划断行、标点禁则、最大字号搜索。
- [x] horizontal / vertical / auto lettering。
- [x] glyph safe-mask 验证。
- [x] publication QA。
- [x] PNG / JSON / debug overlays。
- [x] OpenRaster layer export。
- [x] ImageMagick layered PSD export。
- [x] 本地 Review：文字 / 匹配 / mask 编辑。
- [x] Review 回写并从高清母版重新生成。
- [x] Linux/macOS GitHub Actions。

## 已实现自验收

- [x] 8 个 unit/integration tests。
- [x] 内置端到端 `mhd-transfer selftest`。
- [x] 20 组随机几何扰动 registration benchmark。
- [x] 多长度中文 safe-area lettering benchmark。
- [x] 离线 editable install 验证（no build isolation）。
- [x] ORA/PSD 本机导出验证。

## 真实出版验收仍需数据

程序实现完成不等于“对未知漫画已经获得出版认证”。必须用真实成对漫画建立 100–300 页基准：

- 页面配对准确率 ≥ 99.5%；
- ordinary speech/narration identity match ≥ 99%；
- 自动通过页面中可见日文残留为 0；
- 自动通过页面中 bubble border/人物线稿误伤为 0；
- glyph safe-area overflow 为 0；
- 统计需要人工 Review 的页面率与每页平均修改次数。

真实数据无法由仓库凭空生成，因此这部分是项目上线前唯一剩余的“数据验收”，不是代码 TODO。
