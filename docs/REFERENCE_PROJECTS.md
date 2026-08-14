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

## thomaswantstobeaskeleton/BallonsTranslator-Pro

参考点：shape-aware safe area、balanced line break、density-aware font scaling、overflow safety check、block-level QA。当前核心排字器已经使用 bubble safe mask、实际 glyph-mask safe coverage、最大可行字号搜索和块级 QA；不是复制上游实现。

## ogkalu2/comic-translate

参考点：speech bubble detection 与 text segmentation 独立，PPOCRv5 用于通用 OCR，漫画适配 LaMa 负责清字。当前项目同样把 bubble instance、text mask、OCR 与 inpainter 分为独立接口。

## facebookresearch/sam2

参考点：Review 中对困难气泡 / 文字区域做 point-prompt interactive segmentation。当前 ZIP **尚未直接运行 SAM2ImagePredictor**，只保留接口方向，避免误称“已接入”。

## MangaLens / manga109-segmentation-bubble

本版新增直接可选后端：本地 Ultralytics YOLO11 segmentation `.pt` → bubble instance mask → safe area → OCR block 关联。模型权重不随 ZIP 分发，也不会自动下载。

## ScanR/TypeR

参考点：稳定 auto-centering、bubble-aware auto-centering、样式管理、字号变化时行距同步、adaptive size、紧凑 typesetter 属性面板。本项目不依赖 Photoshop，只吸收交互与排字原则。

## Amster-Ilvil/Novel-formatter

v0.6 直接吸收其“运行环境检测必须无副作用”的工程原则：GUI 只做包/本地路径浅探测，不因为打开设置页就 import 大型模型、创建虚拟环境或下载权重。`runtime_catalog.py` 把安装、模型就绪、MPS 推定分开显示；真正的深度验证只在任务启动或 `doctor` 时发生。

## Amster-Ilvil/Colortina

吸收批量桌面应用的资源策略：OpenCV / OMP / PyTorch CPU 线程限制在合理比例，避免整册任务占满所有 CPU 核导致 Qt 卡顿；GPU 推理与 CPU 预处理分离。v0.6 新增统一 `runtime.py`，Apple Silicon 上自动选择 MPS，并定期释放 MPS allocator cache，但不反复卸载模型。

## Amster-Ilvil/KCC-Kindle-CHS

吸收 PySide6 的紧凑卡片布局、单主操作入口、后台 `QThread`、真实进度、取消与失败清理模式。v0.6 的整册处理不再把 GUI 阻塞在主线程，并支持安全取消：已经完成的页面/缓存保留，可下次断点继续。

## meangrinch/MangaTranslator

参考其跨平台 device helper、模型单例与按图像内容哈希缓存思路。v0.6 将 LightGlue / LoFTR / MangaLens / Torch 超分模型改为进程级复用；配准、OCR、气泡结构改为输出目录内的确定性阶段缓存，且缓存 key 同时包含输入文件与相关配置，避免跨配置误命中。

## BallonsTranslator lazy registry

参考其“启动时只扫描模块元数据、不 import 重模型”的 lazy registry 思路。v0.6 模型中心的日常刷新使用浅探测；即使已经安装 PyTorch，也不会仅因为切换到“模型/配准”页面就加载 Torch 模型。

## v0.8 photographed-page paired transfer references
### zyddnys/manga-image-translator / dmMaze/comic-text-detector

继续吸收“**先生成 text mask / segmentation，再做清除与迁移**”这一分层思想。v0.8 不再假设所有翻译都位于 speech bubble：结构差分先找 changed ink，随后分别生成 enclosed bubble mask 与 free-text/SFX mask。没有复制 GPL 项目实现，核心仍为本项目 OpenCV/NumPy 自有实现。

### ily-R/ImageCoregistration / OpenCV

全局层沿用 SIFT + RANSAC 的稳健配准思路；对手机拍照页特有的局部页弯，v0.8 新增 OpenCV DIS dense optical flow 作为**低频局部 refinement**。flow 只在强高斯模糊后的灰度页上估计，并限制最大位移，避免把中日文字差异本身当作形变。

### 设计边界

- 不引入第三方模型权重；
- 不下载 GitHub 项目代码到发行包；
- 不把第三方 GPL 实现复制进 MIT 核心；
- GitHub 项目仅用于算法/架构调研，最终实现、阈值和回归测试均在本项目内独立完成。
