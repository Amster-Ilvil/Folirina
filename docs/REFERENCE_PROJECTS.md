# Reference Projects / 参考项目

本项目优先吸收**架构思想、算法路线和交互设计**。任何第三方代码、模型和权重在真正纳入前都要单独核对许可证、模型许可和分发限制。

## 1. hgmzhn/manga-translator-ui

https://github.com/hgmzhn/manga-translator-ui

最直接相关的参考。

值得吸收：

- `replace_translation` 双图工作流。
- 生肉图/翻译图各自检测 + OCR。
- 翻译图 OCR 文本作为 translation，不再调用机器翻译。
- 匹配后只清除需要迁移的日文区域。
- inpainting / rendering / JSON / PSD 等现成流水线思想。
- debug artifact 思路。
- balloon-aware rendering、中文断行方向。

不能直接沿用为核心：

- 主要按整页宽高比例缩放区域。
- 固定 overlap threshold 的区域配对。
- 直接粘贴路径依赖翻译图 mask 与 resize 后像素。
- 单一中心模板匹配只能描述平移，无法可靠处理裁边、旋转、非等比缩放、局部变形。
- 模板对齐函数即使存在，也不能替代真正的特征配准。

## 2. zyddnys/manga-image-translator

https://github.com/zyddnys/manga-image-translator

值得吸收：

- 漫画翻译完整 stage pipeline：检测 → OCR → mask → inpaint → render。
- comic text detector / manga OCR / LaMa 等模块化组合。
- 高分辨率时提高 detection/inpainting size 的工程经验。
- 可编辑输出与人工排字准备思路。

不直接采用：

- 本项目不需要翻译阶段。
- 文本渲染区域不能只由文字检测框决定；必须由气泡/文本框安全区约束。

## 3. dmMaze/BallonsTranslator

https://github.com/dmMaze/BallonsTranslator

值得吸收：

- 计算机辅助而非纯黑箱全自动的产品定位。
- 自动文本检测、OCR、清字、翻译、排字后仍允许人工编辑。
- mask 编辑、inpaint 编辑。
- rich text、样式预设、批量格式、自动布局。
- 基于气泡区域改进排字。
- 原文格式估计后用于目标文字样式。

对本项目尤其重要：

- 最终必须有人工复核编辑器。
- 自动结果要能在文本层和 mask 层继续修改。

## 4. thomaswantstobeaskeleton/BallonsTranslator-Pro

https://github.com/thomaswantstobeaskeleton/BallonsTranslator-Pro

值得研究的方向：

- shape-aware safe area。
- overflow checks。
- density-aware font scaling。
- block-level QA warnings。
- 多 OCR、多 detector、多 inpainter 后端抽象。

这些能力非常接近本项目“出版级排字 + 自动 QA”的目标，但应独立验证代码质量与许可证后再决定是否复用实现。

## 5. dmMaze/comic-text-detector

https://github.com/dmMaze/comic-text-detector

价值：

- 漫画/漫画风格文本检测。
- 输出文本 block、text line 与 segmentation。
- 像素级文字分割特别适合生成精准清字 mask。

本项目用法：

- 旧中文版：辅助找中文文字块。
- 高清日文版：产生文字 mask，避免整框清除。

## 6. ogkalu2/comic-translate

https://github.com/ogkalu2/comic-translate

值得吸收：

- speech bubble detection + text segmentation 分开处理。
- 日文和其它语言使用不同 OCR 路线。
- PPOCRv5 用于通用/中文 OCR 的工程思路。
- manga/anime 适配的 LaMa 清字。

## 7. PaddlePaddle/PaddleOCR

https://github.com/PaddlePaddle/PaddleOCR

本项目旧中文版 OCR 的主要候选。

重点评估：

- PP-OCRv5 中文识别。
- 简体/繁体/日文覆盖。
- 竖排、复杂字体和低清扫描表现。
- detection 与 recognition 分离部署。

策略：高置信度单模型直出，低置信度才多尺度/多模型复核。

## 8. cvg/LightGlue

https://github.com/cvg/LightGlue

本项目跨版本页面配准的重点候选。

用途：

- 匹配旧中文版与高清日文版中不受文字变化影响的局部画面特征。
- 与 SIFT / DISK / ALIKED 等特征结合。
- RANSAC 后估计 affine / homography。

注意：不同 feature extractor 的许可证并不相同，正式打包前逐项核对。

## 9. zju3dv/LoFTR

https://github.com/zju3dv/LoFTR

用途：困难页面的 detector-free matching 后备路径。

适合：

- 传统 sparse feature 匹配不足。
- 低纹理或跨版本差异较大的页面。

不建议所有页面默认使用；应作为低置信度升级路径。

## 10. advimman/lama

https://github.com/advimman/lama

用途：复杂背景下日文文字移除。

价值：高分辨率大 mask inpainting 能力较成熟。

本项目不能只靠 LaMa：

- 纯白气泡应优先 deterministic fill。
- 网点应尽量做纹理保持。
- 线稿穿字区域必须检测风险。

## 11. facebookresearch/sam2

https://github.com/facebookresearch/sam2

用途：

- 困难气泡/文本框的交互式 mask 精修。
- 人工 review 时点/框提示快速修正 segmentation。

定位：后备/交互式工具，不把通用 SAM 当作唯一漫画气泡模型。

## 12. MangaLens Bubble Segmentation

https://huggingface.co/huyvux3005/manga109-segmentation-bubble

用途：专门漫画 speech bubble instance segmentation。

本项目重点关注输出实例 mask，而不是只有 bbox；后续由 mask 向内生成排字 safe area。

## 13. ScanR/TypeR

https://github.com/ScanR/TypeR

这是偏人工排字生产力工具的参考，而非自动翻译器。

值得吸收：

- 自动居中。
- 气泡形状辅助定位。
- 样式管理。
- 面向真正 typesetter 的快捷交互。

本项目编辑器不应只做“拖文字框”，而应参考专业排字工作流。

## 14. xulihang/ImageTrans_plugins

https://github.com/xulihang/ImageTrans_plugins

价值：模块化 OCR / mask generation / inpainting 插件架构，包括 PaddleOCR、RapidOCR、mangaOCR、SAM mask、LaMa/MIGAN 等。

本项目可以借鉴 provider interface，但核心数据模型必须围绕“双版本迁移”而不是一般图像翻译。

## 15. 推荐组合

第一版建议：

```text
Page Pairing:
  pHash + feature score

Registration:
  SIFT/ALIKED + LightGlue + RANSAC
  LoFTR fallback

Old Chinese:
  comic text detector + PP-OCRv5

HD Japanese:
  comic text segmentation + bubble instance segmentation
  optional manga-specific OCR only for structure/debug

Cleanup:
  deterministic bubble fill
  + LaMa for complex regions

Lettering:
  custom bubble-safe-area optimizer

Review:
  custom editor + optional SAM2 mask refinement
```

## 16. 最重要的架构结论

没有发现一个成熟开源项目已经完整解决“从低清旧汉化版准确提取现成中文译文，再迁移到几何并不完全一致的高清日文版并达到出版级嵌字”这一整条链路。

因此正确路线不是 fork 某一个项目后不断打补丁，而是：

- 以 `manga-translator-ui` 的 replace-translation 作为业务基线；
- 以 LightGlue/LoFTR 补上跨版本几何配准；
- 以漫画文本分割 + 气泡实例分割把“文字在哪里”和“文字应该排在哪里”分离；
- 以 PaddleOCR 做旧中文版文本内容提取；
- 以 LaMa/确定性修复做清字；
- 以 BallonsTranslator/TypeR 的专业编辑工作流作为人工复核与排字交互参考；
- 自己实现跨版本 matcher、safe-area lettering 和 QA 层。
