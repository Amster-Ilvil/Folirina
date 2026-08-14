# Architecture

## 1. 原则

系统围绕三个完全分离的坐标/语义对象设计：

- **Source translation geometry**：旧中文版上中文译文在哪里，只是来源证据。
- **Target erase geometry**：高清日文版哪些像素属于需要删除的日文。
- **Target lettering geometry**：最终中文可以出现在哪个气泡/文本框安全区。

任何两个对象都不能用一个 bbox 代替。

## 2. 数据对象

### PagePair

保存 source/target 页、视觉指纹成本、置信度与理由。页面配对采用顺序约束动态规划，可显式跳过 source 或 target 的额外页。

### RegistrationResult

保存：

- 3x3 source→target 变换；
- backend/method；
- RANSAC inlier ratio；
- median reprojection error；
- spatial coverage；
- confidence；
- 采样匹配点与候选模型诊断。

### TextBlock

OCR/检测的最小文字区域：polygon、text、confidence、kind、reading order、可选 `mask_path`。

### BubbleInstance

实例 polygon、完整 mask、safe mask、类型、包含的 TextBlock。

### TextUnit

真正参与跨版本身份匹配的单位。气泡内多条 OCR line 会先合成一个 bubble-level TextUnit，从根源上减少“日文多竖列 vs 中文重新断行”造成的一对多噪声。

### UnitMatch

记录 source unit → target unit 的成本、置信度、relation 与证据。`one_to_many` / `many_to_one` 会被识别，但默认不能自动出版。

## 3. 页面配对

指纹成本包含：

- 模糊缩略图 dHash；
- 边缘空间分布；
- 宽高比；
- 卷内相对顺序。

使用 sequence alignment 而不是全局 Hungarian，防止漫画页顺序交叉，同时允许 staff/广告/缺页。

## 4. 配准

### OpenCV 离线基线

SIFT 优先，ORB 后备；KNN ratio test 后执行弱一对一去重。

分别估计：

1. similarity；
2. affine；
3. homography。

候选按：inlier ratio + reprojection error + feature spatial coverage - complexity penalty 打分，因此能满足条件时优先更简单、稳定的变换。

### LightGlue

可选 `SIFT / ALIKED / DISK + LightGlue`。只有安装对应 extra 后才启用，不是核心离线测试硬依赖。

### LoFTR

困难页后备。`auto` 模式不会因为没有 kornia 而失败。

### 安全 fallback

所有特征法失败时只能退回 resize + phase correlation，并把 confidence 限制为低值；它不能越过 publication gate 自动修改高清母版。

## 5. OCR 与文本结构

默认 source 中文使用 PP-OCRv5。低于 `retry_confidence` 的块会：

- 扩边 crop；
- 2x 高质量放大；
- 原图 / sharpen 两个候选复识；
- 保存所有候选；
- 只有置信度明确提升才替换原 OCR。

外部 detector/OCR 可以通过 sidecar 接入；`mask_path` 让 comic text segmentation 直接进入清字。

## 6. 气泡与 safe area

核心自带 `seeded_white` fallback：以 OCR 文字中心寻找封闭白色连通区域，然后用外轮廓填补原文字造成的黑洞。

safe area 使用 distance/erosion 向内收缩，保护气泡边线；窄气泡尾部通常在腐蚀后与主体断开。

专业 speech-bubble instance segmentation 推荐通过 `*.bubbles.json` 输入完整实例 mask。

## 7. 跨版本身份匹配

source TextUnit 先经 Registration 投影到 target 坐标。

成本包含：

- 中心位置；
- bbox overlap-over-smaller；
- 面积/长宽形状；
- reading order；
- speech/narration/free-text kind。

一对一使用 Hungarian assignment。assignment best-vs-second margin 进入 confidence。

未匹配区域额外检查 one-to-many/many-to-one，但这些 relation 一律进入 Review，不自动复制/合并译文。

## 8. 日文清字

优先级：

1. `TextBlock.meta.mask_path` 像素级 segmentation；
2. OCR polygon rasterization。

之后使用与字高相关的自适应 dilation。若 target unit 属于 bubble，最终 mask 还必须与“气泡向内保护后的 mask”相交，因此不会因为 dilation 擦穿边线。

## 9. Inpainting

`auto` 根据 mask 周边像素方差选择：

- 低方差：robust local median 的确定性填充；
- 高方差：OpenCV Telea；
- 配有 `lama_command`：复杂区域优先外部 LaMa，失败会记录并退回 OpenCV。

LaMa 通过命令模板调用，不在项目里复制模型代码/权重。

## 10. 中文排字

排字不是 bbox 字号缩放，而是约束搜索：

- 从 `max_font_size` 向下搜索；
- CJK char-level 动态规划断行；
- 禁止典型中文关闭标点出现在行首；
- 禁止打开标点留在行尾；
- 惩罚孤字和极不均衡行宽；
- 根据 safe-mask 质心居中；
- 小范围位置搜索；
- 实际渲染 glyph mask；
- `glyph ∩ safe / glyph >= min_safe_coverage` 才算成功。

支持 horizontal / vertical / auto。

## 11. Publication gate

自动应用某个文本至少同时要求：

- page pair confidence 通过；
- registration confidence 通过；
- source OCR confidence 通过；
- target geometry confidence 通过；
- unit match confidence 通过；
- relation 必须是 one_to_one；
- 中文文本非空。

任一失败：保留高清母版，不自动清字/排字。

## 12. QA

逐页检查：

- 页面配对低置信度；
- 配准低置信度；
- source OCR 低置信度；
- target geometry 低置信度；
- match 低置信度；
- split/merge；
- source unmatched；
- mask 被气泡边界大量裁剪；
- 清字后 dark-pixel 残留启发式；
- lettering missing/failed；
- glyph 越 safe area；
- 字号过小。

## 13. Review

`mhd-transfer review output` 启动仅绑定 `127.0.0.1` 的本地 HTTP 编辑器。

Review 保存：

- `text_overrides`；
- `match_overrides`；
- `accepted_source_units`；
- `manual_clear_mask.png`；
- status / notes。

`apply_review_page()` 从高清母版重新执行清字和排字，不在已经压过文字的 final 上二次加工。

## 14. 可编辑输出

每页最少保留：

- Original HD Japanese；
- Inpainted；
- Chinese text transparent layer；
- clear mask；
- bubble/safe masks；
- ORA；
- 有 ImageMagick 时 PSD；
- project/QA JSON；
- registration/structure/matching/mask overlays。
