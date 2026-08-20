<p align="center">
  <img src="assets/folirina_banner/part_00.jpg" alt="Folirina — Manga Translation Transfer Studio" width="12.5%"><img src="assets/folirina_banner/part_01.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_02.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_03.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_04.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_05.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_06.jpg" alt="" width="12.5%"><img src="assets/folirina_banner/part_07.jpg" alt="" width="12.5%">
</p>

<h1 align="center">Folirina</h1>

<p align="center">面向漫画本地化与重制流程的桌面文字迁移工具，支持页面配准、文字区域迁移、清理、重排、人工复核。</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="Linux" src="https://img.shields.io/badge/Linux-x64-FCC624?logo=linux&logoColor=black">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Local Processing" src="https://img.shields.io/badge/Processing-Local-7867D9">
  <a href="https://github.com/Amster-Ilvil/Folirina/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Amster-Ilvil/Folirina?label=Release"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-d76aa5"></a>
</p>

> 本地处理 · 页面配准 · 直接贴图 / 精准蒙版 / 整页对齐显中文 · OCR 与模型中心 · 人工补漏 · 可恢复编辑 · 批量处理 ·

## 下载

优先从仓库 **[Releases](https://github.com/Amster-Ilvil/Folirina/releases)** 下载对应平台的最新桌面构建：

- **macOS Apple Silicon**：原生 arm64 `.dmg` + `.app.zip`
- **macOS Intel**：原生 x86_64 `.dmg` + `.app.zip`
- **Windows x64**：便携 ZIP
- **Linux x64**：`tar.gz`
- **SHA-256**：`SHA256SUMS.txt`
- **构建信息**：`BUILD-INFO.txt`

当前发布核心版本：**v2.3.18**

> macOS 公共构建采用 ad-hoc 签名，并未使用 Apple Developer ID 公证。首次启动若被 Gatekeeper 拦截，可在系统“隐私与安全性”中确认打开。

## 主要特性

- **漫画文字迁移工作流**：面向日文原图与中文翻译图 / 重制图之间的页面对应、区域迁移与结果复核。
- **五种活动处理模式**：提供直接贴图、精准蒙版、整页对齐显中文、智能组合与高清重排五条当前可用处理路径。
- **页面配准与结构对齐**：通过几何、特征与可选深度配准能力，使来源页与目标页尽可能保持一致坐标关系。
- **安全区域处理**：优先限制在气泡、文本框和确认过的文字区域内工作，减少对人物、网点、背景与边框的误伤。
- **文字层与恢复编辑**：支持人工补漏、目标层擦除 / 恢复、历史状态与复核工作流，便于对自动结果做最后修正。
- **OCR 与模型中心**：可按需准备 PaddleOCR、Apple Live Text 及其他可选视觉 / 配准运行时，不强制把大型模型打进基础安装包。
- **整册与恢复处理**：支持批量页面处理、已有工作区恢复、处理中断后继续，以及页面级结果管理。
- **QA 与导出**：保留项目状态、页面结果与复核信息，并提供面向最终成品的导出流程。
- **跨平台桌面版**：Windows x64、macOS Apple Silicon、macOS Intel、Linux x64 均有独立 GitHub Actions 构建。
- **本地处理优先**：漫画图片、项目数据、模型缓存、日志和输出默认保存在本机，不作为公开仓库内容提交。

## v2.3.18 更新重点

- **正式补齐四平台发行链**：Windows x64、Linux x64、macOS Apple Silicon 与 macOS Intel 均由 GitHub Actions 在原生 runner 上构建。
- **macOS 双架构发布**：Apple Silicon 与 Intel 分别生成原生 `.app.zip` 和 `.dmg`，并执行架构检查、ad-hoc 签名与 DMG 校验。
- **Release 自动化**：主线发布自动汇总四个平台产物，并生成 `SHA256SUMS.txt` 与 `BUILD-INFO.txt`。
- **版本统一**：项目元数据与 Folirina 核心版本统一到 `2.3.18`。
- **基础包轻量化**：不捆绑大型 Paddle / Torch / OCR / 检测模型，继续使用 Folirina 现有按需模型中心与隔离运行时。
- **隐私审计**：发布前检查常见凭据、本机绝对路径、日志、数据库、缓存、项目输出与模型权重，降低误提交风险。
- **核心功能保持不变**：本次多平台适配集中在打包与发布层，没有重写已稳定的直接贴图、精准蒙版、整页对齐显中文、智能组合与高清重排核心流程。

## 使用流程

1. 导入来源漫画与目标 / 翻译页面。
2. 建立或检查页面配对与结构对齐。
3. 根据页面特点选择直接贴图、精准蒙版、整页对齐显中文、智能组合或高清重排。
4. 运行页面或整册处理。
5. 在复核界面检查文字、气泡、边框、背景与配准结果。
6. 使用人工补漏、擦除 / 恢复与区域编辑修正少量异常。
7. 保存工作区并在需要时继续处理。
8. 导出最终页面或整册结果。

## 处理模式

当前界面提供以下五种活动处理模式：

| 模式 | 用途 |
|---|---|
| **直接贴图** | 对结构高度一致、适合原样迁移的文字或气泡区域进行直接覆盖，尽量保留来源中文字与排版表现。 |
| **精准蒙版** | 通过安全蒙版限制替换区域，强调原字保真与边缘保护，尽量避免破坏气泡边框、人物和背景。 |
| **整页对齐显中文** | 将来源页与目标页整页对齐，通过透明 / 挖洞方式局部显露中文内容，适合高度一致的页面。 |
| **智能组合** | 优先迁移原有中文字形，并在必要位置使用 OCR / 重排能力补漏，由组合策略处理不同区域。 |
| **高清重排** | 对文字区域进行 OCR、重新排字和高清重建，适合无法直接安全迁移原字的区域。 |

## 平台与能力

| 能力 | macOS Apple Silicon | macOS Intel | Windows x64 | Linux x64 |
|---|:---:|:---:|:---:|:---:|
| Folirina GUI / Core | ✓ | ✓ | ✓ | ✓ |
| 直接贴图 / 精准蒙版 / 对齐显中文 | ✓ | ✓ | ✓ | ✓ |
| 人工复核与编辑 | ✓ | ✓ | ✓ | ✓ |
| PaddleOCR 可选运行时 | ✓ | ✓* | ✓ | ✓ |
| Apple Live Text | ✓ | ✓* | — | — |
| Apple MPS 可选加速 | ✓ | — | — | — |
| NVIDIA CUDA 可选运行时 | — | — | ✓ | ✓ |
| CPU 路径 | ✓ | ✓ | ✓ | ✓ |
| 官方 Release 构建 | ✓ | ✓ | ✓ | ✓ |

`*` 具体能力取决于 macOS 版本、机器架构以及对应第三方运行时是否提供兼容构建。

## 模型中心与可选运行时

Folirina 的桌面基础包刻意保持轻量，不直接捆绑大型 OCR、Torch、检测、配准与超分模型。

需要相关功能时，由应用按需准备对应运行环境。当前代码包含或预留了 PaddleOCR / Paddle 文档解析、LightGlue / LoFTR、RT-DETR、SAM、漫画 OCR、布局检测与超分等可选能力。不同模型对 Python 版本、平台架构、GPU 与第三方 wheel 的要求不同，因此 Folirina 尽量将这些组件放入隔离运行环境，而不是污染 GUI 自身 Python。

模型权重、缓存与运行时保存在本机应用数据 / 模型目录，不属于公开源码仓库。

## 从源码安装

需要 **Python 3.11+**。

### macOS

```bash
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[gui]"
python run_gui.py
```

### Windows

```powershell
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[gui]"
python run_gui.py
```

### Linux

```bash
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[gui]"
python run_gui.py
```

也可以使用命令行入口：

```bash
folirina --help
```

兼容旧入口：

```bash
mhd-transfer --help
```

## 本地构建

先安装项目和 PyInstaller：

```bash
python -m pip install -U pip setuptools wheel pyinstaller
python -m pip install -e ".[gui]"
python scripts/build_desktop.py --clean
```

macOS 可进一步生成 DMG：

```bash
bash scripts/build_macos_dmg.sh
```

## 构建与发布

GitHub Actions 发布流程覆盖：

- Windows x64
- macOS Apple Silicon / arm64
- macOS Intel / x86_64
- Linux x64
- Python 编译与基础 import smoke check
- macOS 原生架构检查
- macOS ad-hoc 签名与 DMG 校验
- 发布文件 SHA-256 校验
- 发布前隐私审计
- GitHub Release 自动创建 / 更新

公开仓库不应包含模型权重、用户漫画、工作区、项目输出、缓存、日志、凭据、令牌、私钥、数据库或真实用户主目录路径。

## 隐私说明

- 漫画图片、项目文件与生成结果默认在本机处理。
- 模型权重和可写运行数据保存在本地模型 / 应用数据目录。
- 不将用户素材作为项目源码提交。
- 基础 Release 不捆绑用户数据和本地模型缓存。
- 发布流程执行隐私审计，检查常见密钥、本机路径和不应进入公开仓库的运行数据。

## 致谢与许可

Folirina 是一个面向漫画本地化工作流的整合型桌面项目。OCR、视觉模型、配准、图像处理与 GUI 能力依赖或参考多个开源生态，包括但不限于：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) / PaddlePaddle
- [PaddleX](https://github.com/PaddlePaddle/PaddleX)
- [LightGlue](https://github.com/cvg/LightGlue)
- [Kornia](https://github.com/kornia/kornia) / LoFTR
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Transformers](https://github.com/huggingface/transformers)
- [Segment Anything 2](https://github.com/facebookresearch/sam2)
- PySide6 / Qt、OpenCV、NumPy、Pillow、SciPy、Pydantic 等 Python 生态项目

Folirina 原创代码以 [MIT License](LICENSE) 开源。第三方代码、模型、权重、数据文件及可选依赖仍按其各自许可证授权；重新分发或商业使用前，请自行核对相应上游许可要求。

## 免责声明

请仅处理你有权使用、修改或本地化的内容。使用 Folirina 及其生成结果时，应遵守适用法律、第三方开源许可与原作品版权要求。项目作者不对不当使用产生的侵权或其他法律责任负责。
