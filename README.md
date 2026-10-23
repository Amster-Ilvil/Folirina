<p align="center">
  <img src="assets/folirina_banner.png" alt="Folirina" width="100%">
</p>

<h1 align="center">Folirina</h1>

<p align="center">跨平台漫画高清重制与中文文字迁移工作台 · Manga Translation Transfer Studio</p>

<p align="center">
  <a href="https://amster-ilvil.github.io/Folirina/"><img alt="Web 在线使用" src="https://img.shields.io/badge/Web-Online-0F887F"></a>
  <a href="https://github.com/Amster-Ilvil/Folirina/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Amster-Ilvil/Folirina?label=Download"></a>
  <img alt="Platforms" src="https://img.shields.io/badge/macOS%20%7C%20Windows%20%7C%20Linux-Desktop-3686C9">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-blue"></a>
  <a href="https://github.com/Amster-Ilvil/Folirina/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/Amster-Ilvil/Folirina?style=flat"></a>
</p>

将旧版中文漫画（SOURCE）上的译文迁移到高清日文原图（TARGET），尽量保留原图的网点、线条、人物和彩色背景。提供自动处理、人工复核、可恢复的批量工作区，不是简单地整块覆盖气泡。

## 在线网页版（免安装）

**[▶ 打开 Folirina Web 在线工作台](https://amster-ilvil.github.io/Folirina/)**

这是公开的 **GitHub Pages** 网页版：无需注册 GitHub 账号，也无需安装 Folirina、Python 或 AI 模型。用户导入的漫画和编辑内容在**自己的浏览器本地**处理，无须上传至 Folirina 服务器；你不需要保持电脑开机。

**快速开始**

1. 打开 [Folirina Web](https://amster-ilvil.github.io/Folirina/)，分别导入中文 **SOURCE** 和高清日文 **TARGET**（图片、ZIP/CBZ）。
2. 检查页面配对并执行自动配准，使用「叠加检查」核对画框、人物等是否重合；配准失败的页面请人工检查。
3. 对简单气泡使用涂抹迁移；复杂背景可通过「差异检查」复核**绿色 SOURCE 中文字形掩码**与**红色 TARGET 日文清除掩码**，预览后再确认应用。
4. 使用撤销/恢复和局部修复检查结果，导出当前页 PNG、整册 ZIP，或保存工作区以便下次继续编辑。

| 功能 | 在线 Web Lite | 桌面版 |
| --- | --- | --- |
| 页面导入、配对、仿射/透视配准 | 支持 | 支持 |
| 自由涂抹、双掩码人工复核、预览与导出 | 支持 | 支持 |
| Koharu 等 AI 检测、OCR、完整自动批处理 | 不提供 | 按依赖与模型配置提供 |
| 复杂背景高质量修复 | 需要人工检查，不保证等同桌面版 | 提供更完整的处理流程 |

网页版是**轻量工作台，不是桌面版全部功能的在线复制**。大批高清图片的速度和稳定性受各用户设备内存、浏览器限制；多人可各自在本地独立处理，但网站的访问量仍受 [GitHub Pages 托管限制](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits)约束。请勿将网页上的“配准成功”视为成品已自动通过画质复核。

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

<a href="https://www.star-history.com/?repos=Amster-Ilvil%2FFolirina&type=date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Amster-Ilvil/Folirina&type=date&theme=dark">
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Amster-Ilvil/Folirina&type=date">
    <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Amster-Ilvil/Folirina&type=date">
  </picture>
</a>

## 许可

原创代码以 [MIT License](LICENSE) 发布。请仅处理拥有合法使用、修改或本地化权限的作品；第三方模型、权重、依赖与素材分别遵守各自许可。
