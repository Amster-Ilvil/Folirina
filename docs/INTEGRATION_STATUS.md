# 第三方项目吸收 / 接入状态

本文件只描述当前 ZIP 的真实实现状态。`参考` 不等于复制第三方代码；除明确标注“直接后端”外，本项目不会把第三方源码或模型权重打包进来。

| 项目 | 当前状态 | 实际吸收内容 | 当前边界 |
|---|---|---|---|
| manga-image-translator | 架构吸收 | detection → OCR → mask → inpaint → render 的分层、逐阶段中间产物 | 不直接 import 上游代码 |
| BallonsTranslator | 深度参考 | mask 可编辑、inpaint 后重生成、文字/匹配回写、三栏 Review、样式面板思路 | 还没有做到其完整富文本编辑器与所有快捷键 |
| BallonsTranslator-Pro | 核心算法吸收 | shape-aware safe area、字形实际覆盖率、overflow 拦截、块级 QA、最大字号搜索 | 未复制其模块系统；density scaling 由“最大可行字号搜索 + safe coverage”实现，而不是同一套公式 |
| comic-text-detector | 外部适配器 | 支持 text polygon / pixel text mask / block sidecar，清字优先像素 mask | 当前不直接加载 CTD 权重；通过 sidecar 接入 |
| comic-translate | 架构吸收 | bubble detection 与 text segmentation 分离；OCR 与 inpaint 独立后端 | 不直接 import 上游应用代码 |
| PP-OCRv5 | 直接后端 | PaddleOCR 直接调用，中文/日文、置信度、低置信度放大复识 | 需要用户可选安装 Paddle 依赖/模型 |
| LightGlue | 直接可选后端 | SIFT / ALIKED / DISK + LightGlue；RANSAC 估计变换 | 不自动安装/下载权重 |
| LoFTR | 直接可选后端 | Kornia LoFTR 困难页后备 | 仅在安装 torch/kornia 时可用 |
| LaMa | 外部可调用后端 | 复杂背景通过 command template 调用任意 LaMa wrapper | 不内置 LaMa 权重 |
| SAM 2.1 | 尚未直连 | 当前只保留“Review 点选分割最适合 SAM2”设计位 | **本 ZIP 还没有 SAM2ImagePredictor 点选接口** |
| MangaLens | 直接可选后端 | 新增 Ultralytics YOLO11 instance segmentation，本地 `.pt` 模型 → bubble instance mask → safe area | 不自动下载模型；需配置 `bubbles.mangalens_model_path` |
| TypeR | 深度 UI/排字参考 | 自动居中、字号与行距联动、样式预设/属性面板、紧凑 typesetter 工作流 | 不依赖 Photoshop，不复制其扩展代码 |

## 为什么不把所有第三方直接嵌进一个包

1. 模型体积与许可证不同，默认捆绑会显著增大 Mac 包并带来分发许可问题。
2. 用户明确要求首次启动不要偷偷下载模型，因此重模型均保持显式安装/显式路径。
3. 出版级流程需要可替换后端：同一工程 JSON 应能换 OCR、检测器、气泡模型与修补器，而不是和某一仓库绑死。
4. GPL / Apache / MIT 等许可证边界不同，所以这里优先使用 API/sidecar/command adapter，避免无意复制不兼容代码。

## 下一步最重要的未完成项

- SAM 2.1：在 Review 中点选气泡/文字区域后生成可编辑 mask。
- comic-text-detector：增加可选的本地 runner，而不是仅 sidecar。
- 更完整的 rich text：逐块字体、粗细、描边、旋转、字距、行距、样式批处理和撤销栈。

## v0.4 本地新增：Mask Replace Engine

状态：**已实现 / 非外部占位**。

- 旧中文气泡/旁白框实例 → 高清日文目标实例的专用 Hungarian 匹配；
- source→target 全局配准之后的 bubble bbox 局部收敛；
- ECC 小范围平移校正；
- Lanczos 超采样与 external SR command；
- SR 与几何坐标解耦；
- 默认保留高清气泡边线；
- Mask IoU / target coverage / spill / local-scale gate；
- `mask_replace` 与 `hybrid` 两种直接使用方式；
- 独立 RGBA layer / mask / JSON / debug / ORA / PSD layer；
- Review 可缩小/擦除纯 mask-replace 的覆盖区域并重新合成。

详见 `MASK_REPLACE_PLAN.md`。
