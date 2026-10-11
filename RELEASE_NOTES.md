# Folirina v3.0.0

**跨平台漫画高清重制与文字迁移桌面工作台**

本次 v3.0.0 基于 Folirina `main` 的 3.0.0 代码线制作，重点为迁移模式分离、整册流程与人工复核体验。与上一公开 Release v2.3.18 相比，包含后续多轮核心模块及 GUI 工作流改进。

## 主要更新

- **迁移模式与模块化**：Direct Patch、Mask Replace、对齐挖孔 Reveal、透明 Reveal、Hybrid 与 OCR 重排共六种活动模式；增强模式间输出和复核状态隔离。
- **页面配对和整册处理**：支持智能配对、非正文页跳过、批量继续、页面级缓存和恢复处理。
- **图像质量**：增强文字迁移区域控制、配准和遮罩处理，优先减少对高清 TARGET 背景、网点和边框的影响。
- **复核工作台**：支持局部复合、人工补漏、TARGET 日文层擦除与恢复、Reveal 画笔、OCR 文字重排和项目状态保存。
- **模型运行时**：保留 Koharu 检测器及可选 OCR / Inpainting 路径，支持按设备实际可用性选择 CPU、Apple MPS 或 NVIDIA CUDA。
- **发布与说明**：README 精简，增加 Star History；整合跨平台构建、版本校验、隐私审计和 SHA-256 下载校验。

## 下载对应平台

| 平台 | 文件 |
| --- | --- |
| macOS Apple Silicon (arm64) | `Folirina_3.0.0_macOS_AppleSilicon.dmg`、`.app.zip` |
| macOS Intel (x86_64) | `Folirina_3.0.0_macOS_Intel.dmg`、`.app.zip` |
| Windows x64 | `Folirina_3.0.0_Windows_x64.zip` |
| Linux x64 | `Folirina_3.0.0_Linux_x64.tar.gz` |

另附 `SHA256SUMS.txt` 和 `BUILD-INFO.txt`；可据此校验文件及对应构建提交。

## 安装须知

- 安装包不捆绑大型模型权重，也不包含用户图片、私人项目或本机缓存。可选模型需要自行在模型中心配置。
- macOS 构建采用 ad-hoc 签名，并未经过 Apple 公证；首次运行可能需要在「隐私与安全性」中允许。
- GPU 加速和可选 OCR / 检测能力取决于平台、驱动、模型和独立运行时，基础桌面包不保证所有可选模型开箱即用。
- 提交问题时，请附操作系统、处理模式和匿名化诊断信息；不要上传私人原图、凭据或项目缓存。

完整源码与最新使用指南见 [Folirina README](https://github.com/Amster-Ilvil/Folirina#readme)。
