# Folirina

> 面向漫画本地化与高清重制的跨平台工作台：将**既有中文版本**中的译文、对白气泡内容与排版信息，安全迁移到**高清日文母版**，并提供配准、精准替换、人工复核、QA 与出版输出能力。

**当前代码版本：v2.3.8**  
**Python：3.11+**  
**平台：macOS / Windows / Linux**  
**许可证：MIT**

---

## 1. Folirina 是什么

Folirina 的目标不是“重新翻译漫画”，而是解决另一类更实际的问题：

- 已经有一套低清或较旧的中文汉化版本；
- 同时拥有同一作品的高清日文原图；
- 希望尽可能保留现有中文译文、字形和气泡内容；
- 又不希望因为简单缩放、OCR 重排或整块覆盖而破坏高清原图的人物、网点、彩色背景和气泡边缘。

Folirina 会先建立 SOURCE 与 TARGET 的页面对应关系和几何坐标，再根据所选模式决定是直接迁移原中文像素、使用精准蒙版、进行整页对齐显中文，还是通过 OCR 重排补漏。

### SOURCE / TARGET / OUTPUT

- **SOURCE**：已有中文版本，通常分辨率较低，但中文译文已经存在。
- **TARGET**：高清日文母版，是最终画面、人物、网点、背景和线条的视觉权威。
- **OUTPUT**：在尽可能保持 TARGET 画质的前提下，将 SOURCE 中需要的中文内容迁移进去。

---

## 2. 设计目标

Folirina 重点解决以下问题：

- 尽可能保留既有中文版本中的原始字形；
- 不把“识别文字”和“迁移像素”强行混成一条路线；
- 对人物、背景、照片、彩色区域和气泡边缘保持保守策略；
- 在不同处理模式之间建立明确边界，减少串线；
- 支持整册处理、断点续跑和页面级人工复核；
- 对自动化无法可靠处理的页面明确交给 QA 和人工处理；
- 保持 GUI 与 CLI 使用同一套核心处理逻辑；
- 允许逐步接入不同的 OCR、布局检测、气泡检测和配准后端。

Folirina 的核心原则是：

> **TARGET 决定最终画面，SOURCE 提供需要迁移的中文内容。**

---

## 3. 核心能力

Folirina 当前包含一套完整的漫画迁移流水线：

- **整册页面配对**：根据页序、图像特征和配置对 SOURCE / TARGET 页面进行匹配。
- **几何配准**：支持 Auto、OpenCV、LightGlue、LoFTR 等路线。
- **视觉布局证据**：可利用气泡、文字、SFX、页面结构等视觉信息约束自动处理区域。
- **多种迁移模式**：Direct、精准蒙版、整页对齐显中文、Hybrid、高清重排彼此隔离。
- **0 OCR 原字迁移**：Direct 与精准蒙版可以完全不依赖 OCR，直接保留原中文像素形态。
- **OCR 补漏与重排**：Hybrid / Reletter 可在需要时识别文本并重新排版。
- **人物与背景保护**：优先保护 TARGET 的人物、彩色区域、网点和非文字画面。
- **人工复核**：支持页面级工作区、手工蒙版、恢复/擦除、补漏与重新应用结果。
- **断点续跑**：整册处理中断后可继续，避免从第一页重新计算。
- **模式隔离**：不同处理模式拥有独立能力契约和产物所有权。
- **QA 与出版阻断**：高风险页面可以被标记为必须人工复核。
- **工作区清理**：可删除能够重新生成的中间诊断文件，同时尽量保留人工编辑所需内容。
- **跨平台 GUI 与 CLI**：既可以图形界面操作，也可以脚本化批处理。

---

## 4. 处理模式

Folirina 当前 GUI 主流程提供 5 个活动模式。

| 模式 | 内部名称 | OCR | 适合场景 | 主要思路 |
|---|---|---:|---|---|
| 直接贴图 | `direct_patch` | 否 | 中日版气泡和版式高度一致 | SOURCE 在上、TARGET 在下，仅迁移安全的气泡内部中文像素 |
| 精准蒙版 | `mask_replace` | 否 | 希望最大限度保留原中文字形，同时严格保护边缘 | 基于配准、差异和视觉区域生成精准迁移蒙版 |
| 整页对齐显中文 | `aligned_overlay_reveal` | 不用于正文重排 | SOURCE / TARGET 整页结构高度一致 | 对齐两页后在受控区域显露中文内容，并保护 TARGET 画面 |
| 智能组合 | `hybrid` | 是 | 大部分区域可原字迁移，少量区域需要 OCR 补漏 | 原字迁移优先，必要时进入 OCR + Reletter |
| 高清重排 | `reletter` | 是 | 旧中文字质较差、需要重新排字 | OCR 获取文字，再按高清目标区域重新排版 |

### 4.1 直接贴图 `direct_patch`

特点：

- 不依赖 OCR；
- 优先保留 SOURCE 中现成中文字形；
- 适用于气泡内部形状高度一致的页面；
- 对候选区域进行安全约束；
- 避免整块贴图覆盖人物或复杂背景；
- 适合结构几乎一致的中日版本。

主要产物通常包括：

```text
direct_patch_layer.png
direct_patch_regions.png
direct_patch.json
```

### 4.2 精准蒙版 `mask_replace`

这是 Folirina 中最强调“原字保真”的路线之一。

特点：

- **0 OCR**；
- 使用视觉候选、配准与蒙版完成原中文字迁移；
- SOURCE 中文像素是迁移内容权威；
- TARGET 是背景与画面权威；
- 重点保护气泡边框、人物、照片和彩色区域；
- 支持 paired-diff 等精细视觉路线；
- 适合希望尽量原样保留旧汉化文字的页面。

主要产物通常包括：

```text
mask_transfer_layer.png
mask_transfer_mask.png
mask_transfer.json
```

### 4.3 整页对齐显中文 `aligned_overlay_reveal`

适合 SOURCE 与 TARGET 几乎是同一版式、但需要更大范围对齐处理的场景。

特点：

- 先整页配准，再规划允许处理的区域；
- 通过 hole / erase / source-ink 等蒙版控制实际写入；
- 不使用 OCR 去重新生成正文；
- 可以结合布局或存在性证据限制高风险区域；
- 对彩色 TARGET 使用额外画面保护逻辑；
- 更适合两版图像结构高度一致的情况。

主要产物通常包括：

```text
aligned_overlay_reveal_layer.png
aligned_overlay_reveal_mask.png
aligned_overlay_reveal_erase_mask.png
```

### 4.4 智能组合 `hybrid`

特点：

- 原中文字迁移优先；
- 允许 OCR；
- 当视觉迁移不足时，再由 Hybrid 自己管理 Reletter 补漏；
- 不允许随意消费其他模式的旧产物；
- 适合整册自动化后再人工复核。

### 4.5 高清重排 `reletter`

特点：

- 使用 OCR 获取中文内容；
- 根据目标区域重新计算布局；
- 支持字体、字号、换行、缩放等排版参数；
- 适合 SOURCE 字体质量较差、无法直接保留原字的页面；
- 提供人工 Reletter 编辑能力。

### 4.6 兼容模式

代码仍保留：

```text
auto
transparent_bubble_reveal
```

它们属于兼容/旧路线，不是当前 GUI 的主要推荐入口。新项目建议优先明确选择 5 个活动模式，以获得更稳定的模式隔离和可预测行为。

---

## 5. 一页漫画如何处理

典型处理流程：

```text
SOURCE 中文页 ─┐
               ├─ 页面配对
TARGET 日文页 ─┘
        ↓
      几何配准
        ↓
  视觉布局 / 气泡 / 文字区域证据
        ↓
     选择迁移模式
        ↓
 Direct / Mask / Reveal / Hybrid / Reletter
        ↓
   合成与页面级产物
        ↓
       QA 检查
        ↓
     人工复核 / 补漏
        ↓
      最终输出
```

Folirina 的架构原则是：

**共享底层几何能力，但隔离不同 Renderer 与人工复核状态。**

这样修改某一个模式时，不应悄悄改变另一个已经稳定的模式。

---

## 6. 平台支持

### macOS

支持：

- CPU
- Apple Silicon MPS（可选）
- Apple Live Text / 本地相关能力（视系统与配置而定）
- LightGlue / 其他深度模型按需安装

### Windows

支持：

- CPU
- NVIDIA CUDA（可选）
- PaddleOCR（可选）
- LightGlue / RT-DETR / Ultralytics 等可选模型

### Linux

支持：

- CPU
- NVIDIA CUDA（可选）
- PaddleOCR（可选）
- CLI、批处理和服务器环境

基础图像处理依赖 OpenCV、NumPy、Pillow、SciPy 等。深度模型并不是启动 Folirina 的强制条件。

---

## 7. 安装

### 7.1 环境要求

- Python **3.11 或更高版本**
- 建议使用虚拟环境
- GUI 需要 PySide6

### 7.2 macOS / Linux

```bash
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[gui]"
```

### 7.3 Windows PowerShell

```powershell
git clone https://github.com/Amster-Ilvil/Folirina.git
cd Folirina

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[gui]"
```

### 7.4 启动 GUI

安装后：

```bash
folirina-studio
```

也可以在源码目录运行：

```bash
python run_gui.py
```

---

## 8. 可选依赖

`pyproject.toml` 将较重的运行时拆成可选依赖，建议按实际需要安装。

| Extra | 用途 |
|---|---|
| `gui` | PySide6 桌面界面 |
| `ocr` | PaddleOCR / PaddlePaddle |
| `semantic` | PaddleX 语义/版面相关能力 |
| `lightglue` | Torch + Kornia + LightGlue 配准 |
| `bubbles` | Ultralytics 气泡/视觉模型相关能力 |
| `rtdetr` | RT-DETR / Transformers 视觉检测 |
| `accel` | Torch、Kornia、Spandrel、Ultralytics 等加速组件 |
| `dev` | pytest / pytest-cov 开发测试工具 |

例如：

```bash
pip install -e ".[gui,lightglue]"
```

或：

```bash
pip install -e ".[gui,ocr]"
```

Paddle OCR 与文档/版面运行时可以使用独立环境管理。遇到 Paddle 依赖冲突时，优先使用 GUI 的模型/运行时管理能力，而不是强行把所有 Paddle 组件安装到主 Python 环境。

---

## 9. GUI 推荐工作流

### 第一步：准备图片

建议准备两个目录：

```text
book/
├─ zh_source/
│  ├─ 001.png
│  ├─ 002.png
│  └─ ...
└─ jp_target/
   ├─ 001.png
   ├─ 002.png
   └─ ...
```

不要求文件名完全一致，但页序越可靠，自动配对越稳定。

### 第二步：载入 SOURCE / TARGET

在项目/页面管理区域选择两套图片目录并建立页面配对。

建议先检查：

- 封面是否对应；
- 是否存在多出来的广告页、版权页、扉页；
- 两边是否有漏页；
- 自动配对后页码是否发生偏移。

### 第三步：选择处理模式

可以按照目标选择：

- 想最大程度保留旧中文字：`mask_replace`
- 气泡高度一致、希望直接迁移：`direct_patch`
- 两张整页高度一致、希望对齐显露中文：`aligned_overlay_reveal`
- 希望自动迁移并让 OCR 补漏：`hybrid`
- 希望重新排字：`reletter`

### 第四步：检查配准与模型

建议先处理少量代表性页面，观察：

- 配准是否存在整体偏移；
- 气泡边缘是否被破坏；
- 人脸、头发、网点、彩色背景是否被误处理；
- 中文是否完整；
- 是否出现旧缓存或其他模式产物串入当前页面。

### 第五步：整册处理

确认样例页没有明显问题后再跑整册。

Folirina 支持断点续跑。默认批处理会尝试复用已经完成且配置指纹一致的页面；需要完全重跑时可以明确禁用 resume。

### 第六步：人工复核

自动化处理后仍建议逐页检查：

- 漏迁移文字；
- 气泡边框；
- 彩页人物肤色；
- 开放式效果字；
- 拟声词；
- 复杂照片页；
- SOURCE 与 TARGET 本身存在绘制差异的区域。

### 第七步：导出

通过 QA 后再导出最终结果。出版级用途不建议跳过人工复核。

---

## 10. 命令行使用

安装后：

```bash
folirina --help
```

兼容旧命令名：

```bash
mhd-transfer --help
```

### 10.1 生成完整配置

```bash
folirina init-config config.json
```

然后：

```bash
folirina run ./zh_source ./jp_target ./output --config config.json
```

### 10.2 只检查页面配对

```bash
folirina pair ./zh_source ./jp_target
```

### 10.3 处理整册

```bash
folirina run ./zh_source ./jp_target ./output \
  --mode mask_replace \
  --registration-backend auto \
  --device auto
```

可用模式：

```text
direct_patch
mask_replace
aligned_overlay_reveal
hybrid
reletter
auto
transparent_bubble_reveal
```

设备：

```text
auto
mps
cuda
cpu
```

配准后端：

```text
auto
opencv
lightglue
loftr
```

CLI 的 OCR 覆盖参数支持：

```text
paddle
sidecar
none
```

是否真正允许 OCR，最终仍受当前模式契约限制。`direct_patch` 和 `mask_replace` 不会因为配置里选择了 OCR 就自动跨模式调用 OCR。

### 10.4 禁用断点续跑

```bash
folirina run ./zh_source ./jp_target ./output --no-resume
```

### 10.5 检查运行环境

```bash
folirina doctor
```

用于检查 OpenCV、SIFT、Paddle 运行时、LightGlue、Kornia、ImageMagick、Spandrel、Ultralytics、字体和设备状态等。

### 10.6 启动复核编辑器

```bash
folirina review ./output
```

默认仅绑定：

```text
127.0.0.1:8765
```

如果明确需要远程访问：

```bash
folirina review ./output --host 0.0.0.0 --allow-remote
```

远程绑定属于安全敏感操作，不建议在不可信网络中直接暴露。

### 10.7 应用单页复核结果

```bash
folirina apply-review ./output/pages/001
```

### 10.8 清理工作区

```bash
folirina cleanup-workspace ./output
```

用于删除可重复生成的页面诊断数据，并尽量保留 GUI / 人工恢复所需文件。

### 10.9 内置自检

```bash
folirina selftest
```

用于运行离线合成出版流水线验收。

### 10.10 架构隔离检查

```bash
folirina architecture-audit
```

用于检查：

- 模式能力契约；
- Renderer 所有权；
- 产物隔离；
- Review 状态隔离；
- 不允许的跨模式调用。

---

## 11. 输出目录

实际文件会根据模式和配置变化，典型结构类似：

```text
output/
├─ pages/
│  ├─ 001/
│  │  ├─ project.json
│  │  ├─ qa.json
│  │  ├─ ...当前模式的中间层与蒙版
│  │  └─ ...人工复核文件
│  ├─ 002/
│  └─ ...
└─ final/
   ├─ 001.png
   ├─ 002.png
   └─ ...
```

其中：

- `project.json`：页面处理状态、模式、产物路径和诊断信息；
- `qa.json`：页面 QA 信息；
- 模式专属 layer / mask / json：由当前 Renderer 生成；
- 人工蒙版：由复核界面生成并在后续应用中复用；
- `final/`：整册最终页输出位置之一。

不要依赖某个中间文件名作为长期 API。需要自动化集成时，优先读取 `project.json` 中的 `artifacts` / `meta`。

---

## 12. 模式切换与旧结果

Folirina 会尽量避免不同模式互相污染。

当同一页从一个模式切换到另一个模式时：

- 自动生成的旧 Renderer 产物会被清理；
- 不兼容的人工复核输入不会直接混入新模式；
- 用户编辑内容会尽量归档到 `review_archive/<old>_to_<new>/`；
- 新模式只允许消费自己契约中允许的子系统。

例如：如果先用 Mask 处理一页，之后切换到 Reletter，不应该因为目录中遗留 `mask_transfer_layer.png` 就继续使用旧 Mask 结果。

---

## 13. 断点续跑与缓存

Folirina 的批处理支持 resume。

默认行为重点是：

- 已完成页面可跳过；
- 配置发生真正相关变化时重新处理；
- 不相关模式的参数变化不应让所有旧页面失效；
- 中间阶段可以使用自己的缓存键；
- 模型和模式发生实质变化时重新计算对应阶段。

如果怀疑缓存污染，可以：

1. 关闭当前任务；
2. 对目标输出目录执行 `cleanup-workspace`；
3. 对问题页重新处理；
4. 必要时使用 `--no-resume` 做一次完整重跑。

---

## 14. QA 与人工复核原则

Folirina 的自动化目标是减少人工工作量，而不是声称任何漫画都可以 100% 无人值守。

尤其建议人工检查：

- SOURCE / TARGET 实际画面并不相同；
- 中文气泡尺寸明显不同；
- 开放式文字压在人脸、头发、衣服或复杂纹理上；
- 彩色页面；
- 文字穿过气泡边缘；
- 旋转文字、艺术字、拟声词；
- 低清 SOURCE 已经严重压缩或模糊；
- OCR 对罕见字、符号、竖排或特殊字体识别不稳定。

如果 CLI 结束时出现出版阻断级 QA 项，程序会明确报告，并可能以非零退出码结束。此时应先 review，而不是直接把结果当作最终稿。

---

## 15. 仓库隐私与安全

本仓库只应保存运行所需源码与公开文档，不应提交：

- SOURCE / TARGET 漫画原图；
- 用户项目工作区；
- 输出页和 QA 产物；
- 模型权重；
- `.env`；
- API Key / Token / 密码；
- 私钥 / 证书；
- 带有个人绝对路径的运行时配置；
- 打包 ZIP / `.app`；
- 其他只属于本机的临时文件或敏感数据。

`.gitignore` 已对常见本地敏感文件、模型、缓存、输出目录和打包产物进行拦截，但 `.gitignore` 不能替代提交前检查。

发布前建议至少执行：

```bash
git status
git diff --cached
```

如果 Token、密码或私钥曾经真正提交到 Git 历史中，仅删除当前文件并不足够：应立即撤销或轮换凭据，并清理 Git 历史。

---

## 16. 常见问题

### GUI 启动时报缺少 PySide6

安装 GUI extra：

```bash
pip install -e ".[gui]"
```

### PaddleOCR 无法运行

先检查：

```bash
folirina doctor
```

Paddle OCR 与文档解析运行时可能使用独立环境。优先在 GUI 的模型/运行时管理区域进行安装或修复。

### macOS 没有 CUDA

正常。Apple Silicon 优先使用：

```text
auto / mps / cpu
```

### Windows / Linux 没有 NVIDIA 显卡

使用：

```text
auto / cpu
```

### 页面整体偏移

优先检查：

1. SOURCE / TARGET 是否配错页；
2. 自动配准是否失败；
3. 尝试显式切换 `opencv` / `lightglue` / `loftr`；
4. SOURCE 和 TARGET 是否被裁边、缩放或重新排版过。

### 为什么精准蒙版没有走 OCR

这是设计行为。`mask_replace` 的模式契约就是 0 OCR 原字迁移。需要 OCR 补漏请使用 `hybrid`；需要重新排字请使用 `reletter`。

### 为什么切换模式后旧人工蒙版不直接生效

不同模式的人工复核状态具有所有权。为了避免旧坐标和旧 Renderer 污染新模式，Folirina 会在模式切换时归档不兼容的 review 输入。

### 为什么工作区很大

图像处理会生成蒙版、图层、QA 和诊断文件。处理完成并确认无需进一步调试后，可以运行：

```bash
folirina cleanup-workspace ./output
```

---

## 17. 项目结构

核心源码位于：

```text
src/manga_hd_transfer/
```

主要模块职责：

```text
cli.py                         CLI 入口
launcher.py                    GUI 启动入口
config.py                      全局配置模型
pairing.py / page_pairing.py   页面配对
registration.py                基础配准
structure_registration.py      结构配准
mode_contracts.py              模式能力与隔离契约
modes/                         各模式声明与私有逻辑
direct_containers.py           Direct 区域/贴图核心
mask_transfer.py               精准蒙版迁移
aligned_overlay_reveal*.py     整页对齐显中文
reletter_*.py                  OCR 后重排与排版
pipeline*.py                   主处理流水线及拆分阶段
review*.py                     人工复核与应用
qa.py                          质量检查
workspace.py                   页面工作区解析
workspace_cleanup.py           工作区清理
runtime*.py                    设备与可选运行时
model_downloads.py             模型下载管理
selftest.py                    离线合成自检
architecture_audit.py          架构/模式隔离检查
```

`pipeline.py` 主要负责组织主流程，具体职责逐步拆分到 `pipeline_*` 模块，以降低主 Pipeline 的耦合度，同时避免随意重写已经稳定的 Direct / Mask / Reletter Renderer。

---

## 18. 开发与验证

推荐在提交影响处理结果的改动前至少运行：

```bash
folirina selftest
folirina architecture-audit
```

安装开发依赖：

```bash
pip install -e ".[dev]"
```

如果当前分支包含 pytest 测试：

```bash
python -m pytest
```

对图像算法的修改还应使用真实代表性页面做回归，重点比较：

- 最终像素是否发生非预期变化；
- 人物或彩色背景保护是否退化；
- 模式是否串线；
- 旧项目是否仍能恢复；
- GUI 与 CLI 的行为是否一致。

---

## 19. 版权与使用范围

Folirina 是图像处理与本地化辅助工具。请只处理你拥有合法访问权、编辑权或获得授权的素材。

项目本身不提供漫画资源、翻译内容或第三方模型版权授权。

---

## 20. License

Folirina 使用 **MIT License**。

详见仓库中的 [`LICENSE`](LICENSE)。

---

## 21. 项目地址与反馈

GitHub：

```text
https://github.com/Amster-Ilvil/Folirina
```

提交 Issue 时建议提供：

- Folirina 版本；
- 操作系统与 Python 版本；
- 所选模式；
- 配准后端；
- `folirina doctor` 的相关结果；
- 可复现步骤；
- 必要且已经脱敏的截图或最小复现素材说明。
