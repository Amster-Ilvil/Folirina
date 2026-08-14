# Implementation plan and status

## v0.8.34 已实现：Direct Patch / Mask Transfer 分离

- [x] `transfer.mode = auto | direct_patch | mask_replace | hybrid | reletter`。
- [x] `direct_patch` 为独立整块 SOURCE 栅格贴图路线，不再从 `mask_replace` 偷用开关。
- [x] Direct 失败严格拒绝 / passthrough，不静默回退到 Mask/OCR。
- [x] `auto` 才允许 Direct 不安全时继续进入 Mask。
- [x] 新增 OCR-free same-page precheck 与统一 transfer planner。
- [x] Direct 专用 layer / region / JSON 产物与独立 GUI 模式。
- [x] Direct/Mask 防回归 pytest。

详细契约：`docs/DIRECT_VS_MASK_V0834.md`。

## v0.4 已实现：气泡 / 文本框蒙版替换

- [x] 历史 v0.4 模式：`reletter | mask_replace | hybrid`；v0.8.34 已由上方五模式取代。
- [x] 只选择含 OCR 文字的旧中文 speech / narration 实例。
- [x] source bubble → target bubble Hungarian 实例匹配。
- [x] 整页配准后，按目标 bubble mask bbox 做局部尺寸收敛。
- [x] ECC 小范围平移精校；只有 IoU 改善才采用。
- [x] 局部 scale correction 超阈值自动拒绝，防止硬拉伸错气泡。
- [x] source bubble patch 独立 Lanczos / external SR；SR 不参与最终坐标计算。
- [x] 默认保留高清 target bubble 边线，只替换内部。
- [x] Mask IoU / target coverage / spill ratio 三重 gate。
- [x] `hybrid` 模式自动回退到 OCR 高清重排。
- [x] 独立 transfer RGBA layer / mask / JSON / debug / ORA / PSD layer。
- [x] GUI 直接控制模式、局部对齐、SR、边线保护与几何门槛。
- [x] 新增蒙版迁移单元 + pipeline 集成测试。

详细设计：`docs/MASK_REPLACE_PLAN.md`。


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
