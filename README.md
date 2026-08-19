# Folirina

Folirina 是一个用于漫画页面文字迁移、清理、重排和人工复核的桌面工具。

## 桌面版下载

GitHub Releases 提供可直接运行的桌面发行包：

- Windows x64：ZIP 便携版
- macOS Apple Silicon：DMG + `.app.zip`
- macOS Intel：DMG + `.app.zip`
- Linux x64：`tar.gz`

Release 同时提供 `SHA256SUMS.txt` 和 `BUILD-INFO.txt`，用于核对文件完整性和构建环境。

> macOS 公共构建采用 ad-hoc 签名，不包含 Apple Developer ID 公证。首次启动时若被 Gatekeeper 拦截，可在系统“隐私与安全性”中确认打开。

基础桌面包只包含 Folirina GUI/Core 所需运行时，不捆绑体积较大的 Paddle/Torch/OCR/检测模型。可选 OCR、配准、检测和超分运行时仍由 Folirina 的模型中心按需准备，并尽量使用隔离运行环境，避免污染桌面程序本身。

## 平台支持

- **Windows 10/11 x64**：CPU；支持可选 CUDA/Torch 与 PaddleOCR 等运行时。
- **macOS Apple Silicon (arm64)**：CPU / MPS；支持 Apple Live Text/系统能力以及可选隔离模型运行时。
- **macOS Intel (x86_64)**：CPU；可使用平台存在的 OCR/外部工具能力。
- **Linux x64**：CPU；支持可选 CUDA/Torch 与 PaddleOCR 等运行时。

不同平台、GPU 驱动与第三方模型的可用性可能不同；Folirina 会按当前机器能力显示并准备可用后端。

## 从源码安装

```bash
python -m pip install -e ".[gui]"
```

## 启动

```bash
python run_gui.py
```

也可以使用命令行：

```bash
mhd-transfer --help
```

## 支持模式

- `auto`
- `direct_patch`
- `mask_replace`
- `aligned_overlay_reveal`
- `transparent_bubble_reveal`
- `hybrid`
- `reletter`

## 本地构建

先安装项目和 PyInstaller：

```bash
python -m pip install -U pip setuptools wheel pyinstaller
python -m pip install ".[gui]"
python scripts/build_desktop.py --clean
```

在 macOS 上还可以生成 DMG：

```bash
bash scripts/build_macos_dmg.sh
```

程序运行时生成的模型、缓存、日志和项目结果保存在本地工作区，不属于代码仓库内容。仓库的 GitHub Actions 会在发布前执行隐私审计，并分别在 Windows、macOS Apple Silicon、macOS Intel 和 Linux 上构建发行包。
