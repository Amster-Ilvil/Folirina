# Reference projects and absorption boundaries

本项目只吸收架构/算法思路或通过外部适配器调用，不复制第三方项目实现。

## hgmzhn/manga-translator-ui

参考点：`replace_translation` 的业务定义——两套同作品图片、两侧检测/OCR、把已有译文迁移到生肉/高清版。

不照搬：

- 按宽高比例直接 scale region；
- 固定 overlap 阈值作为主要身份匹配；
- 中心模板单平移；
- 普通对白低清文字像素直接贴图。

本项目把它升级为“页面配准 → bubble-level identity matching → target pixel mask → rerender”。

## zyddnys/manga-image-translator

参考点：检测/OCR/翻译/Mask/Inpaint/Render 分阶段管线，以及调试中间产物思想。

本项目差异：不做机器翻译；source 已经是中文翻译；核心任务是 cross-edition transfer。

## dmMaze/BallonsTranslator

参考点：人工编辑器、mask/inpaint 可编辑、rich lettering、漫画翻译生产流程。

本项目的本地 Review 保持三栏证据结构，并允许文字、匹配、mask 回写。

## dmMaze/comic-text-detector

参考点：漫画文字不仅有 bbox，还应该有 text line / segmentation。它的像素级 mask 能显著降低清字误伤。

接入方式：把 detector 输出转成 `page.ocr.json`，每个 block 可包含 `mask_path`。

## PaddlePaddle/PaddleOCR

使用点：PP-OCRv5 中文/日文 OCR。项目显式传 `ocr_version="PP-OCRv5"`，关闭文档方向、文档去畸变等不适合漫画页的通用文档模块。

## cvg/LightGlue

使用点：SIFT / ALIKED / DISK 局部特征 + LightGlue 匹配。代码通过可选 extra 动态 import；权重与上游许可证保持独立。

注意：上游 README 对不同 feature/weights 的许可证有区别，部署前应确认所选 extractor 的许可。

## zju3dv/LoFTR

使用点：在传统 keypoint/LightGlue 难以获得足够匹配时，作为 dense-ish feature matching 后备。

## advimman/lama

使用点：复杂纹理/画面文字的高质量 inpainting。

本项目只提供外部 command adapter，不内置权重，不自动下载。

## Speech bubble segmentation / SAM-like models

专用漫画气泡模型优先于通用分割模型。SAM/SAM2 更适合作为人工 Review 的困难区域辅助，而不是把整页每个候选都交给通用 prompt segmentation。

接入统一用 `page.bubbles.json` + instance mask / safe mask，核心流水线不依赖某个具体模型仓库。
