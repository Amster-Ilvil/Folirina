<p align="center">
  <img src="assets/folirina_banner.png" alt="Folirina — Manga Translation Transfer Studio" width="100%">
</p>

<h1 align="center">Folirina</h1>

<p align="center">面向漫画本地化、高清重制与文字迁移的桌面工作台：把旧版中文页中的文字安全迁移到高清日文母版，并提供配准、蒙版、Reveal、OCR 重排、区域复合、人工补漏与可恢复整册处理。</p>

<p align="center">
  <img alt="macOS" src="https://img.shields.io/badge/macOS-Apple%20Silicon%20%2F%20Intel-black?logo=apple">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-x64-0078D4?logo=windows11&logoColor=white">
  <img alt="Linux" src="https://img.shields.io/badge/Linux-x64-FCC624?logo=linux&logoColor=black">
  <img alt="PySide6" src="https://img.shields.io/badge/UI-PySide6-41CD52?logo=qt&logoColor=white">
  <img alt="Local Processing" src="https://img.shields.io/badge/Processing-Local-7867D9">
  <a href="https://github.com/Amster-Ilvil/Folirina/releases"><img alt="Release" src="https://img.shields.io/github/v/release/Amster-Ilvil/Folirina?label=Release"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-MIT-d76aa5"></a>
</p>

> **本 README 以当前 `main` 分支实际界面与代码能力为准。** `main` 可能先于 GitHub Release；只想直接使用时，优先下载 Releases 中的稳定桌面构建。

## 下载

优先从 **[GitHub Releases](https://github.com/Amster-Ilvil/Folirina/releases)** 下载对应平台的桌面构建：

- **macOS Apple Silicon**：原生 arm64 `.dmg` + `.app.zip`
- **macOS Intel**：原生 x86_64 `.dmg` + `.app.zip`
- **Windows x64**：便携 ZIP
- **Linux x64**：`tar.gz`
- **完整性校验**：`SHA256SUMS.txt`
- **构建信息**：`BUILD-INFO.txt`

> macOS 公共构建采用 ad-hoc 签名，并未使用 Apple Developer ID 公证。首次启动若被 Gatekeeper 拦截，可在系统“隐私与安全性”中确认打开。

## Folirina 能做什么

Folirina 不是单纯的 OCR 排字器，也不是把一个气泡整块贴到另一张图上的脚本。它的核心目标是：

1. 将 **旧版中文 / 翻译页（SOURCE）** 与 **高清日文页（TARGET）** 对应起来；
2. 先做页面配准，让 SOURCE 与 TARGET 落到统一坐标系；
3. 根据页面结构选择 Direct、精准蒙版、两种 Reveal、Hybrid 或 OCR 重排；
4. 尽量只迁移文字相关像素，保护 TARGET 的人物、网点、边框、彩图背景与分格线；
5. 自动结果不足时，在同一页继续人工补漏、擦除日文、恢复日文、OCR 重排或区域复合；
6. 保存页面工程与人工状态，支持之后继续处理、重新处理与整册恢复。

## 当前主流程

```text
导入 SOURCE / TARGET
        ↓
智能配对 + 页面检查
        ↓
选择整页处理模式
        ↓
处理当前页 / 从头处理整本 / 继续处理整本
        ↓
当前页 QA 与结果检查
        ↓
区域复合 / 人工补漏 / TARGET 擦除与恢复 / 人工 OCR 排版
        ↓
保存工作区 → 导出最终页面
```

## 六种活动处理模式

当前主界面提供 **6 个活动模式**。旧 `auto` 路由仅为历史项目兼容保留，不再作为主界面的活动模式。

| 模式 | 适合场景 | 核心思路 |
|---|---|---|
| **直接贴图 · 无边框内层贴图** | SOURCE / TARGET 结构高度一致，中文字可以直接安全迁移 | 在配准后从 SOURCE 提取可迁移文字/内部像素，尽量不带气泡边框与无关背景。 |
| **精准蒙版 · 原字保真 / 拍照边缘保护** | 希望尽量保留旧中文版原字形，同时保护高清母版边缘 | 由配准、结构与差异证据生成精确写入 Mask，只在允许区域迁移 SOURCE 原字。 |
| **整页对齐挖孔显中文** | 两版页面高度一致，适合通过硬边开孔显出下层中文 | TARGET 与 SOURCE 整页对齐后生成 hole mask，以挖孔方式显露中文。 |
| **整页对齐透明显中文** | 彩页、复杂背景或需要软边 Reveal 的页面 | 生成 TARGET 上层 / SOURCE 下层结构，通过透明度局部显露中文，尽量保持 TARGET 彩图背景。 |
| **精准蒙版+OCR** | 大部分区域可原字迁移，但少量区域需要重新排字 | 先走 Hybrid 自有的精准蒙版流程，再对需要的区域使用 OCR / 重排补漏；两阶段仍属于同一模式工作区。 |
| **OCR重排** | SOURCE 字形无法安全直接迁移，或需要重新设计排版 | OCR 识别文字后在 TARGET 上重新清字、排版与渲染，支持后续人工文字编辑。 |

模式之间有独立的产物与复核状态。重新处理时会清理上一自动模式的派生产物；切换已处理页面的模式时，不兼容的人工 Review 状态会进入归档，避免不同模式互相污染。

## 项目与页面管理

### 智能配对

Folirina 会对两套页面建立视觉指纹并做顺序约束配对，而不是只按文件名硬对齐。当前项目页还提供三项可选辅助：

- **优先名称 / 页码配对**：先锁定唯一同名或唯一页码对应页面；
- **优先文件夹自然顺序**：适合两套页数和顺序都非常规整的项目；
- **重制增强配对 · AKAZE 二次核验**：只对低置信配对做局部特征 + RANSAC 核验，不会自动删除或重排页面。

长册在候选矩阵过大时会尝试内存受限的带状序列对齐；如果约束不足，则回退到完整参考算法，而不是静默接受可疑配对。

### 页面类型

每个新配对页默认都按 **“正文 / 需替换”** 处理。页面管理器允许人工标记：

- 封面
- 扉页 / 书名页
- 目录
- 单话首页 / 章节页
- 插图 / 纯图片
- 卷首插画
- 空白页
- 后记 / 版权 / 广告
- 手动跳过

这些被明确标记为非正文的页面仍保留在阅读顺序中，但最终直接保留高清 TARGET，不进入漫画文字迁移。

### 读取已有运行结果

“**读取已有运行结果…**”可以恢复命令行、旧会话或之前已经处理出的页面工程，包括 SOURCE / TARGET、final、中间产物与人工复核状态，随后直接进入 GUI 继续检查和补漏。

需要注意：读取已有结果只恢复 **已经处理过的页面子集**，不会偷偷执行整本智能配对。若要继续批量处理整本，仍需先点击“智能配对”，确保未处理页面也已经加入完整书册映射。

## 批量处理与断点恢复

项目页当前提供四个明确的处理动作：

- **处理当前页**
- **从头处理整本**
- **继续处理整本**
- **停止**

“继续处理整本”会使用恢复语义跳过仍然有效的已完成页面，并复用页面级配准、OCR、气泡等缓存。完成状态使用模式相关的配置指纹，尽量避免修改一个模式的无关参数后把其它已完成页面全部判定为失效。

重新处理当前页时，Folirina 会重新执行自动流程，并重新应用该页已经保存的人工强制迁移、人工补漏、清除 Mask、TARGET 擦除 / 恢复等人工结果，避免一次自动重跑把人工修正全部抹掉。

## 替换工作台与人工收尾

自动流程只负责第一遍。当前工作台已经把“难页最后 5%”拆成独立工具，而不是要求不断切换整页模式重跑。

### 区域复合工具

在同一个选区内，可以连续叠加：

- Direct
- 精准蒙版
- 挖孔 Reveal
- 透明 Reveal / 透明文字
- OCR

选区支持矩形、椭圆、爆炸框智能闭合与自由闭合。每个区域动作独立保存，选区外禁止写入；处理完一个动作后可以保留当前选区继续叠加下一种工具。

### 开放文字框选与人工补漏

对于自动检测漏掉的开放式文字、彩色文字、效果字或普通白气泡，可以直接在 TARGET 上框选：

- **开放文字框选**：精准蒙版 / Hybrid 专用，只迁移 SOURCE 中文原始笔画，不调用 OCR，不整块贴背景；
- **彩色开放式文字 · 擦除显字**：优先清除 TARGET 日文字形，再显露 SOURCE 中文；
- **彩色开放式文字 · 自动迁移**：在复杂背景上只处理文字差异；
- **白色气泡 · 文字迁移 + X/Y 微调**：迁移文字本身，可手动微调 SOURCE 对应位置。

人工框只是搜索/约束范围，不等于最终整块写入范围。

### TARGET 日文层擦除 / 恢复

工作台提供两个方向相反的收尾工具：

- **仅擦 TARGET 日文层**：处理残留日文、黑点、短线、符号等；保护中文迁移 / 重排 / 人工补漏层；
- **恢复 TARGET 日文层 / 擦蒙版**：对不应该显示中文的区域恢复 TARGET 原始日文图层和背景，并反向收回相关清除 / 强制 Mask。

Reveal 编辑还支持画笔式 **透明揭示 / 挖孔揭示 / 恢复日文**，可调画笔、透明羽化与强度，并支持逐笔撤销 / 重做。

## 人工 OCR 实时排版编辑器

人工 OCR 现在是独立于整页自动模式的 **页面复核叠加层**。Direct、精准蒙版、两种 Reveal、精准蒙版+OCR、OCR重排都可以在处理后继续框选 ROI 做人工 OCR 与排版，不会偷偷切换当前整页模式。

编辑器支持：

- 在 TARGET 或实时成品上直接框选 OCR 区域；
- 重新 OCR 当前定位框，或直接手工输入 / 修正中文；
- 双击实时文字框直接编辑；
- 自动 / 竖排 / 横排；
- 智能断句 / 均衡断句 / 保留手工换行；
- 智能缩放、严格文本框、智能气泡等布局策略；
- 自动气泡 / 文本框与手动排版框分离；
- 文字对齐；
- 字号、列数、行距、**字距**、列距；
- 文字旋转与文字颜色；
- 实时最终效果预览与安全覆盖提示；
- 自动排版 / 恢复自动框 / 删除文本块 / 清空人工 OCR。

定位框只负责“原日文在哪里”，排版框只负责“中文放在哪里”；两者分离可以避免移动中文时把 TARGET 的清除范围一起拖走。

## 持久字体库

Folirina 已不再只记住某个临时绝对路径。人工 OCR / 重排界面导入外部字体时，可以把字体复制进应用自己的持久字体库，之后项目仍可复用。

当前支持：

- `.ttf`
- `.otf`
- `.ttc`
- `.otc`

字体导入会做基础解析与 FreeType 可用性检查，并记录稳定元数据；重复字体以内容哈希识别。字体库按操作系统保存到用户应用数据目录，而不是塞进源码仓库或当前漫画项目。

## 模型中心与视觉管线

Folirina 把“页面配准、主检测器、OCR、清字 / Inpainting”明确拆开。不同自动模式只调用自己合同允许的子系统，避免为了一个 Mask 模式顺手启动不必要的 OCR 或重排器。

### OCR

当前界面可选择或接入：

- Apple Live Text
- Apple 快捷指令 ExtractText
- PaddleOCR v6 Medium / Small
- PaddleOCR-VL 1.6
- Manga OCR
- Baberu OCR
- 48px AR OCR
- PP-StructureV3
- 外部 OCR JSON / Markdown
- Sidecar `.ocr.json`
- 关闭 OCR

### 页面配准

- Auto / SIFT
- LightGlue
- LoFTR

### 主检测器

- Koharu Layout RF-DETR Seg 2XL
- MangaLens
- Comic Translate RT-DETR-v2

主检测器单选；辅助检测器可以按需或始终参与。辅助来源包括白色容器 / 结构几何、MangaLens、RT-DETR-v2、YSG YOLO OBB、Koharu Layout、CTD 外部像素 Mask 与 Sidecar 外部蒙版，并可按需使用 SAM 2 / 2.1 做边界精修。

### 清字 / Inpainting

当前可选：

- Auto 轻量安全路径
- LaMa Manga
- AOT Inpainting
- FLUX.2 Klein
- RORem Mixed
- OpenCV Telea

基础桌面包刻意不捆绑大型 OCR、Torch、检测、配准、超分或 Inpainting 权重。需要时由模型中心准备本地模型 / 依赖，并尽量放在隔离运行环境中，避免污染 GUI 自身 Python 环境。

## 输出、工作区与空间控制

默认使用 **精简工作区**：保留最终结果、恢复处理与人工补漏需要的文件，把大量 Debug 图、逐组件 Mask 和可编辑图层改成按需输出。

可选项包括：

- 保存配准 / 结构 / 蒙版 Debug 图；
- 保存逐气泡 / 逐文本单元 Mask；
- 输出 OpenRaster / PSD 可编辑图层；
- 清理 `pages/` 中可重新生成的冗余诊断文件。

工作区清理不会删除 `final`、原始 SOURCE / TARGET、中文迁移层、Review 或人工补漏状态；界面会统计扫描页数、删除文件数与释放空间。

## 设置、更新与诊断

### 界面

- 浅色 / 深色主题；
- 主题记忆；
- 针对小屏幕、高 DPI 与大窗口的整体响应式缩放。

### Git 源码更新

设置页内置 Git 更新检查。更新目标锁定为 Folirina 官方仓库和 `main` 分支，界面不可改写：

- Git 工作树只允许 **clean + fast-forward**；
- 不自动降级；
- 不覆盖存在未提交修改的 Git 工作树；
- ZIP / portable 场景先在临时目录取得并校验新代码，再事务式替换，失败自动回滚；
- 更新程序文件时不删除 `.venv`、用户漫画、输出目录、模型或日志。

### 日志与诊断

- 程序级运行日志目录可直接打开；
- 当前页可查看 `run.log` / `last_run_state.json`；
- 可一键导出当前页诊断 ZIP，包含运行状态、QA、核心 Mask / transfer 元数据等排查文件。

## 平台说明

| 能力 | macOS Apple Silicon | macOS Intel | Windows x64 | Linux x64 |
|---|:---:|:---:|:---:|:---:|
| Folirina GUI / Core | ✓ | ✓ | ✓ | ✓ |
| 6 个活动迁移模式 | ✓ | ✓ | ✓ | ✓ |
| 页面管理 / 智能配对 / 批处理 | ✓ | ✓ | ✓ | ✓ |
| 人工补漏 / 区域复合 / OCR 编辑 | ✓ | ✓ | ✓ | ✓ |
| Apple Live Text | ✓ | ✓* | — | — |
| Apple MPS 可选加速 | ✓ | — | — | — |
| NVIDIA CUDA 可选运行时 | — | — | ✓ | ✓ |
| CPU 路径 | ✓ | ✓ | ✓ | ✓ |
| 官方 Release 构建流程 | ✓ | ✓ | ✓ | ✓ |

`*` 具体能力取决于 macOS 版本、机器架构以及 Apple / 第三方运行时是否提供兼容能力。

## 从源码运行

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

命令行入口：

```bash
folirina --help
```

兼容旧入口：

```bash
mhd-transfer --help
```

## 本地构建

安装项目与 PyInstaller：

```bash
python -m pip install -U pip setuptools wheel pyinstaller
python -m pip install -e ".[gui]"
python scripts/build_desktop.py --clean
```

macOS 可进一步生成 DMG：

```bash
bash scripts/build_macos_dmg.sh
```

## CI 与 Release

GitHub Actions 当前包含核心质量检查、隐私审计、运行时自检以及四个平台的桌面构建：

- Windows x64
- macOS Apple Silicon / arm64
- macOS Intel / x86_64
- Linux x64
- Python 编译 / import / release quality gate
- Runtime self-test
- macOS 原生架构检查
- macOS ad-hoc 签名与 DMG 校验
- 发布文件 SHA-256 校验
- GitHub Release 自动创建 / 更新

公开仓库不应包含模型权重、用户漫画、工作区、项目输出、缓存、日志、凭据、令牌、私钥、数据库或真实用户主目录路径。

## 隐私说明

- 漫画图片、页面工程与生成结果默认在本机处理；
- 模型权重和可写运行数据保存在本地模型 / 应用数据目录；
- 不将用户素材作为项目源码提交；
- 基础 Release 不捆绑用户数据和本地模型缓存；
- 发布流程执行隐私审计，检查常见密钥、本机路径与不应进入公开仓库的运行数据。

## 开源生态

Folirina 是一个面向漫画本地化工作流的整合型桌面项目。OCR、视觉模型、配准、图像处理与 GUI 能力依赖或参考多个开源生态，包括但不限于：

- PaddleOCR / PaddlePaddle / PaddleX
- PySide6 / Qt
- OpenCV / NumPy / Pillow / SciPy / Pydantic
- LightGlue / Kornia / LoFTR
- Ultralytics / RT-DETR / Transformers
- Segment Anything 2
- Manga OCR 及其它可选漫画 OCR / Inpainting / Layout 运行时

具体第三方代码、模型、权重与数据仍遵循各自上游许可证。

## 许可与免责声明

Folirina 原创代码以 [MIT License](LICENSE) 开源。

请只处理你有权使用、修改或本地化的内容。使用 Folirina 及其生成结果时，应遵守适用法律、第三方开源许可与原作品版权要求。第三方代码、模型、权重、数据文件及可选依赖仍按其各自许可证授权；重新分发或商业使用前，请自行核对对应上游许可。
