# Manga HD Transfer Studio

将已有中文漫画译文从低清/旧版页面迁移到高清日文原图的桌面工具。项目重点不是重新 OCR 排字，而是尽可能保留高清 TARGET 的人物、网点、背景、气泡边界和色彩，只迁移需要的中文内容，并提供自动处理、精准蒙版、Direct、Reveal 与人工补漏工作流。

当前版本：**v1.3.14**

## 主要能力

- SOURCE / TARGET 页面自动配对与配准。
- 精准蒙版迁移：清除 TARGET 日文后迁移 SOURCE 中文内容。
- Direct / 白气泡 / 彩色复杂区域处理。
- Aligned Overlay Reveal 实验路线：整页配准后按安全区域“挖洞显中文”。
- 人工补漏工作台：白气泡补漏、Reveal、TARGET 日文层擦除、纯白涂抹、位置微调。
- 页面管理：跳过页、人工页类型、批量处理、结果复核。
- 已运行工程恢复：可以读取 CLI/Codex 已生成的页面工作区，重新匹配中日文并继续人工补漏。
- 缓存、结果状态、QA 与异常恢复。
- Apple Live Text / PaddleOCR 等 OCR 后端按需使用；OCR/ML 模型不会在启动时自动下载。

## 多平台 Release

v1.3.14 起采用与 Novel Formatter 相同的发布思路：**发布包只包含受 Git 管理的程序源码和启动器，独立 Python 运行时与主程序依赖在用户机器首次运行时准备；OCR/ML 专用依赖和模型仍保持按需、经用户确认后安装。**

| 平台 | Release 文件 | 状态 |
| --- | --- | --- |
| Windows x64 | `MangaHDTransfer_<version>_Windows_x64.zip` | GitHub Actions 实测 |
| macOS Apple Silicon | `MangaHDTransfer_<version>_macOS_universal.dmg` / `.zip` | GitHub Actions 实测 |
| macOS Intel | 同一个 universal macOS 包 | GitHub Actions 实测 |
| Linux x86_64 | `MangaHDTransfer_<version>_Linux_x86_64.tar.gz` | Ubuntu GitHub Actions 实测 |

每个正式 Release 同时提供：

- `SHA256SUMS.txt`：下载文件 SHA-256 校验值。
- `BUILD-INFO.txt`：构建提交、CI 验证结果和隐私边界说明。

### Windows

1. 下载并完整解压 Windows ZIP。
2. 双击根目录的 `启动Windows.bat`。
3. 启动器优先使用兼容的 Python 3.11–3.13。
4. 如果没有兼容 Python，会下载并校验独立 Python 运行时。
5. 自动创建项目专用虚拟环境，然后只安装 GUI/基础运行依赖并启动程序。

运行时目录、日志和虚拟环境都被 Git 忽略，不属于发布源码。

### macOS

1. 下载 DMG 或 ZIP。
2. 将 `Manga HD Transfer Studio.app` 放到“应用程序”或其他普通位置。
3. 首次打开会根据当前机器自动准备 Apple Silicon 或 Intel 对应的独立 Python。
4. macOS 如果提示未验证开发者，可在 Finder 中右键 App →“打开”。

macOS App 是轻量运行壳：真正可写的应用源码副本、Python、依赖、缓存和日志位于本机 Application Support 目录，不会修改只读/签名 App 内容。

Apple Live Text 路线使用仓库内的 Swift Helper 源码。需要编译 Helper 时，系统需具备 Xcode Command Line Tools；不满足条件时，程序会按现有后端策略降级，而不是把私有或预编译开发机产物打进 Release。

### Linux

1. 解压 Linux tar.gz。
2. 在解压目录运行：

```bash
bash ./启动Linux.sh
```

3. 启动器优先使用兼容的系统 Python 3.11–3.13；否则下载并校验独立 Python。
4. 首次运行会创建项目专用虚拟环境。

PySide6 在部分发行版上仍可能需要系统提供常见 XCB/桌面运行库。正式 Release 目前在 Ubuntu x86_64 CI 上验证。

## 首次启动不会做什么

程序启动阶段**不会**：

- 下载 PaddleOCR 模型；
- 下载 Torch / RT-DETR / YOLO / LightGlue 等模型权重；
- 自动安装全部 OCR/ML 专用依赖；
- 自动扫描或上传你的漫画；
- 把 API Key、模型缓存、用户图片或处理结果写回 Git 仓库。

只有当用户实际进入对应 OCR/模型功能并确认安装时，现有的依赖/模型管理流程才会执行。

## 从源码运行

需要 Python 3.11 或更高版本。

```bash
git clone <repository-url>
cd Manga-HD-Translation-Transfer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[gui]"
python run_gui.py
```

Windows PowerShell 可使用：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[gui]"
.\.venv\Scripts\python.exe .\run_gui.py
```

### CLI

安装基础包后可使用：

```bash
mhd-transfer --help
```

GUI 是当前完整人工补漏、页面管理和结果复核功能的主要入口。

## 可选依赖

`pyproject.toml` 将大体积/专用组件拆成可选依赖：

- `gui`：PySide6 GUI。Qt for Python Community Edition 使用 LGPLv3/GPLv3 许可体系，也提供 Qt 商业许可；主项目 MIT 不会覆盖 Qt/PySide6 的许可义务。
- `ocr`：PaddleOCR / PaddlePaddle（Apache-2.0）。
- `lightglue`：Torch + Kornia + LightGlue；LightGlue 代码和其自身预训练权重为 Apache-2.0，但具体特征提取器/权重仍应核对各自许可。
- `bubbles`：Ultralytics 检测组件。Ultralytics 当前提供 AGPL-3.0 与 Enterprise 两种许可路径，启用或再分发时必须遵守适用许可。
- `rtdetr`：Torch + Transformers RT-DETR；模型权重的许可可能与 Transformers 软件许可不同。
- `accel`：Torch / Kornia / Spandrel / Ultralytics 加速组件；其中 Ultralytics 仍受其 AGPL/Enterprise 许可约束。
- `dev`：pytest / coverage 开发测试依赖。

Release 首次启动只安装 `.[gui]`，不会一次性安装这些可选 ML 栈，也不会把第三方模型权重打进仓库 Release。

完整第三方许可证、上游链接和分发边界见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

## 推荐工作流

1. 选择已有中文 SOURCE 页面目录。
2. 选择高清日文 TARGET 页面目录。
3. 检查页面配对结果，必要时调整配对方式或页面标记。
4. 选择 Auto / 精准蒙版 / Direct / Reveal 等适合的迁移路线。
5. 批量处理并查看 QA、残留提示和处理状态。
6. 在替换工作台中对漏字、特殊气泡、开放式效果字等区域进行人工补漏。
7. 输出最终页面；需要时可再次读取既有结果继续校对，而不必重新跑整本。

## 结果与缓存

程序会在用户指定的输出目录保存页面工作区、最终图、复核状态和必要的 QA/诊断文件。大型中间产物、模型缓存和运行环境不应提交到 Git。

如果已经通过命令行或其他前端完成过一次处理，可在 GUI 的页面管理中读取已有运行结果，然后继续人工补漏与复核。

## 隐私与发布安全

本仓库把“不能把开发者或用户隐私带进 Release”作为发布阻断条件，而不只是 README 约定。

### 1. Git 忽略

`.gitignore` 会排除常见本地内容，例如：

- `.env` / 本地凭据文件；
- `.runtime` / 虚拟环境；
- OCR/模型缓存与权重；
- 日志、数据库、调试目录；
- 用户漫画、文档、PSD/ORA、输出工作区；
- 本机构建产物、DMG、ZIP。

### 2. tracked-file 隐私审计

`scripts/privacy_audit.py` 会在 PR、push 和 Release 构建前扫描所有 Git tracked files，并阻止：

- 私钥、常见 API Key / GitHub / Hugging Face / AWS / Google Token；
- 真实的 macOS `/Users/<name>`、Linux `/home/<name>`、Windows 用户目录；
- 非占位个人邮箱地址；
- `.env`、credential 文件；
- 模型权重；
- 日志、数据库、用户文档/漫画容器；
- runtime、venv、cache、output 等生成目录。

任何一项命中都会让 CI 和 Release 失败。

### 3. License / attribution 审计

`scripts/license_audit.py` 会检查：

- `LICENSE` 仍为标准 MIT 文本；
- `pyproject.toml` 使用明确的 PEP 639 `MIT` 元数据并包含 `LICENSE`；
- `VERSION`、包版本和 `CITATION.cff` 版本一致；
- 第三方许可清单没有遗漏关键直接/可选依赖；
- PySide6/Qt 与 Ultralytics 等许可证敏感依赖仍被显式标注；
- LightGlue、LoFTR、RT-DETR、PaddleOCR 学术引用仍保留。

这样许可证/引用文件不会只靠人工记忆维护。

### 4. Release 只从 tracked files 构建

Windows/macOS/Linux 的打包脚本都以 `git archive HEAD` 为输入。即使开发机器目录中存在未跟踪的图片、配置、模型、日志或私有文件，它们也不会进入发布包。

GitHub Actions 还会从干净 checkout 构建，因此 Release 不依赖开发者本机工作目录。

### 5. 最小发布权限

普通 CI 只有 `contents: read`。只有合并到 `main` 后执行的 `publish` job 临时获得 `contents: write`，用于创建/更新 GitHub Release。

## GitHub Actions

- `Privacy audit`：每个 PR/push 执行隐私扫描、许可证/引用一致性审计和 Python 语法编译。
- `Build and publish Manga HD Transfer releases`：
  - 检查 `VERSION`、`pyproject.toml`、运行时版本、许可证和引用元数据一致；
  - Windows x64 安装基础依赖并导入验证；
  - macOS Apple Silicon 安装基础依赖并导入验证；
  - macOS Intel 安装基础依赖并导入验证；
  - Linux x86_64 安装基础依赖并导入验证；
  - 构建多平台发布包；
  - 生成 SHA-256；
  - 合并到 `main` 后发布正式 Release。

## 更新器

更新检查现在会按当前操作系统选择对应的 Release 资产，不再简单取“第一个 ZIP”，因此多平台 Release 不会让 macOS 错拿 Windows 包。

- macOS：保留 `.app` 事务更新与回滚能力。
- Windows / Linux：可检查到本平台最新资产；当前使用 Release 包手动替换程序目录更安全。
- 私有仓库环境如需程序直接访问私有 Release，可通过运行时环境变量提供 GitHub Token；不要把 Token 写入源码、配置模板或 Release 包。

## 版本发布

版本号统一维护在：

- `VERSION`
- `pyproject.toml`
- `src/manga_hd_transfer/version.py`
- `CITATION.cff`

Release CI 会强制检查这些版本元数据一致。

当前首个多平台发布版本：**v1.3.14**。

## 许可证、第三方声明与引用

- **本项目原创代码/文档**：MIT License，详见 [`LICENSE`](LICENSE)。
- **第三方软件与许可清单**：详见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。MIT 不会覆盖或替代第三方许可证。
- **实现来源、设计致谢与学术引用**：详见 [`REFERENCES.md`](REFERENCES.md)。
- **引用本项目**：仓库提供 [`CITATION.cff`](CITATION.cff)，GitHub 可据此生成引用信息。

已明确标注的实现/设计来源包括 Novel Formatter（Apple OCR 与跨平台/隐私发布架构）和 KCC-Kindle-CHS（UI 配色灵感）；开发研究参考与论文引用则单独列在 `REFERENCES.md`，避免把“概念参考”“代码依赖”“代码改写”混为一谈。

模型权重、字体、扫描图、译文、第三方数据和系统框架均不因本项目采用 MIT 而获得新的授权。使用或再分发前应分别检查它们自己的许可证/授权条件。

## 使用范围

请只处理你有权编辑、翻译或重新排版的素材，并遵守源作品、译本、字体、模型和相关站点的授权条款。
