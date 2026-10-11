<p align="center">
  <img src="assets/folirina_banner.png" alt="Folirina" width="100%">
</p>

<h1 align="center">Folirina</h1>

<p align="center">跨平台漫画高清重制与中文文字迁移工作台 · Manga Translation Transfer Studio</p>

<p align="center">
  <a href="https://github.com/Amster-Ilvil/Folirina/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Amster-Ilvil/Folirina?label=Download"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/macOS%20%7C%20Windows%20%7C%20Linux-Desktop-3686C9">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue"></a>
  <a href="https://github.com/Amster-Ilvil/Folirina/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Amster-Ilvil/Folirina?style=flat"></a>
</p>

将旧版中文漫画（SOURCE）上的译文迁移到高清日文原图（TARGET），尽量保留原图的网点、线条、人物和彩色背景。提供自动处理、人工复核、可恢复的批量工作区，不是简单地整块覆盖气泡。

## 下载

[**前往 GitHub Releases 下载最新版**](https://github.com/Amster-Ilvil/Folirina/releases)

| 平台 | 下载文件 |
| --- | --- |
| macOS Apple Silicon（M 系列） | `.dmg` / `.app.zip`（arm64） |
| macOS Intel | `.dmg` / `.app.zip`（x86_64） |
| Windows x64 | 便携 `.zip` |
| Linux x64 | `.tar.gz` |

安装包不预装大型检测 / OCR / Inpainting 模型；需在模型中心按需配置。macOS 公共版未经 Apple 公证，首次打开可能需要在「隐私与安全性」允许启动。以 Release 页面实际列出的文件为准。

## 核心功能

- **智能配对与页面管理**：中文 / 日文页面匹配、跳过封面与章节页、整册处理及断点恢复。
- **六种处理模式**：Direct Patch、精准蒙版、对齐挖孔 Reveal、透明 Reveal、Hybrid（蒙版 + OCR）、OCR 重排。
- **高清画面保护**：文字区域差异处理、遮罩和配准，减少对线稿、网点及背景的破坏。
- **人工复核与补漏**：区域复合、日文层擦除 / 恢复、画笔 Reveal、可编辑 OCR 排版、撤销与恢复。
- **可选模型与加速**：Koharu 等版面检测、OCR / Inpainting、Apple Silicon MPS、NVIDIA CUDA 或 CPU（取决于运行时和模型支持）。
- **独立工作区**：按项目保存中间状态、最终图、人工修改和诊断信息。

## 使用流程

```text
导入中文版 SOURCE + 日文高清版 TARGET
  → 页面配对、检查与跳过非正文页
  → 选择迁移模式、按需配置检测器
  → 处理当前页 / 整本 / 继续未完成页面
  → 在复核工作台人工修正
  → 导出最终页面
```

## 从源码运行

需要 Python 3.11+，在 macOS / Linux 终端执行：

```bash
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui]"
python run_gui.py
```

Windows PowerShell 将激活命令改为 `.venv\Scripts\Activate.ps1`，其余步骤相同（`python3` 改为 `python`）。

自行构建桌面包：`python -m pip install pyinstaller`，然后 `python scripts/build_desktop.py --clean`。macOS 可再运行 `bash scripts/build_macos_dmg.sh`。

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Amster-Ilvil/Folirina&type=Date)](https://www.star-history.com/#Amster-Ilvil/Folirina&Date)

## 许可

原创代码以 [MIT License](LICENSE) 发布。请仅处理拥有合法使用、修改或本地化权限的作品；第三方模型、权重、依赖与素材分别遵守各自许可。
