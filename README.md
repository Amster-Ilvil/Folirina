# Manga HD Translation Transfer

把**旧版低清中文汉化漫画**中的既有译文可靠迁移到**高清日文原图**，完成跨版本页面配准、中文 OCR、日文清字、气泡安全区排版、自动 QA、人工复核与分层导出。

项目不是机器翻译器。它不重新翻译日文；中文内容来自你已有的旧汉化版。

## 目标

输入：

- `source_cn/`：旧版中文汉化图，允许低清、压缩、扫描偏移、裁边、额外 staff 页。
- `target_jp/`：同一作品的高清日文图，作为唯一画面母版。

输出：

- `final/`：自动通过安全门槛的最终 PNG。
- `pages/.../target_original.png`：高清母版，永不覆盖。
- `inpainted.png` / `clear_mask.png`：清字结果与精确 mask。
- `text_layer.png`：透明中文文字层。
- `editable.ora`：OpenRaster 分层工程。
- `editable.psd`：系统有 ImageMagick 时自动生成分层 PSD。
- `project.json`：配准矩阵、OCR、气泡、匹配、排字、置信度和可追溯信息。
- `qa.json` / debug overlays：出版 QA 和视觉证据。
- 本地 Review 编辑器：可改译文、改目标匹配、画/擦清字 mask，再重新生成。

## 核心设计

```text
旧中文版 ─ 页面指纹/顺序配对 ─┐
                            ├─ SIFT/LightGlue/LoFTR + RANSAC 配准
高清日文版 ─ 页面指纹/顺序配对 ┘

旧中文版 ─ 中文 OCR / 多次低置信度复识 ─ 中文 TextUnit
高清日文版 ─ 日文文字区域 + 气泡实例 / safe mask ─ TargetUnit

TextUnit + Registration + TargetUnit
              ↓
      跨版本最小成本身份匹配
              ↓
      日文像素级 text mask 清字
              ↓
  solid / OpenCV / external LaMa 修复
              ↓
  中文约束排字（字号、断行、禁则、安全区）
              ↓
       QA 安全门槛 + Review Queue
              ↓
       PNG / ORA / PSD / JSON
```

### 与简单“替换翻译”方案的关键区别

1. **先做视觉配准，再匹配文字身份**，不依赖整页 resize 后固定 IoU。
2. 普通对白迁移的是**中文文本内容**，不是把低清文字像素放大贴到高清图。
3. **清字 mask 与排字 safe area 分离**：前者只描述要删除的日文，后者描述中文允许出现的位置。
4. 气泡边界有保护带；中文最终字形 mask 必须通过 safe-area 覆盖验证。
5. 低页面配对、低配准、低 OCR、低身份匹配、拆分/合并关系都会阻止自动覆盖，进入 Review。
6. 每页有完整 evidence/debug，不做不可追溯的黑箱整页重绘。

## 安装

Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

核心安装只需要 OpenCV / NumPy / Pillow / SciPy / Pydantic / Typer，不下载 OCR 或深度模型。

### 中文 / 日文 OCR

```bash
pip install -e '.[ocr]'
```

默认显式使用 `PP-OCRv5`。PaddleOCR 的模型可能在首次实际推理时下载。若当前平台不方便运行 Paddle，也可使用 `sidecar`，把任何外部 OCR/漫画文字检测器的结果接进来。

### LightGlue / LoFTR 增强配准

```bash
pip install -e '.[lightglue]'
```

`registration.backend=auto` 默认可以只靠 OpenCV SIFT 离线运行；已安装 LightGlue 时可先尝试 LightGlue。LoFTR 作为困难页后备。

### LaMa

项目不复制/绑定 LaMa 模型。把任意可用 LaMa wrapper 配成：

```json
{
  "inpainting": {
    "backend": "lama",
    "lama_command": "python lama_wrapper.py --input {input} --mask {mask} --output {output}"
  }
}
```

## 首次运行

```bash
mhd-transfer doctor
mhd-transfer init-config config.json
mhd-transfer run source_cn target_jp output --config config.json
```

如果使用外部 OCR sidecar：

```bash
mhd-transfer run source_cn target_jp output --ocr-backend sidecar
```

运行结束若存在出版阻断级 QA，CLI 返回非 0，并提示进入 Review：

```bash
mhd-transfer review output
```

浏览器打开三栏编辑器：旧中文版 / 高清日文与清字 mask / 当前输出。可以：

- 修改 OCR 中文译文；
- 改某个中文 TextUnit 对应的目标气泡/文本框；
- 勾选是否应用该文本；
- 画笔增加或擦除清字 mask；
- 保存并点击“应用复核并重新生成”。

也可以命令行应用：

```bash
mhd-transfer apply-review output/pages/0001_xxx
```

## Sidecar 接口

### OCR / 漫画文字分割

`page.png` 对应 `page.ocr.json`：

```json
{
  "blocks": [
    {
      "id": "b0",
      "text": "已经存在的中文译文",
      "confidence": 0.99,
      "polygon": [[100,100],[260,100],[260,180],[100,180]],
      "kind": "speech",
      "mask_path": "masks/b0.png"
    }
  ]
}
```

如果提供 `mask_path`，清字优先使用像素级 segmentation；没有时才回退到 polygon。

### 气泡 / 旁白框实例分割

`page.bubbles.json`：

```json
{
  "bubbles": [
    {
      "id": "bubble-0",
      "kind": "speech",
      "confidence": 0.99,
      "polygon": [[80,70],[300,70],[300,220],[80,220]],
      "mask_path": "bubbles/0.png",
      "safe_mask_path": "bubbles/0_safe.png"
    }
  ]
}
```

没有 `safe_mask_path` 时会自动向内腐蚀实例 mask，窄尾部通常会自然脱离排版安全区。

## 自验收

完全离线、无模型下载：

```bash
pytest
mhd-transfer selftest
python benchmarks/synthetic_acceptance.py
```

本次实现时结果：

- 单元/集成测试：**8/8 通过**。
- 内置端到端 selftest：**通过**，无 error/warning。
- 20 组随机几何扰动配准：**20/20 通过**。
- 控制点中位误差：约 **0.051 px**；P95：约 **0.117 px**。
- 4 组不同长度中文气泡排字：**4/4 通过**，字形安全区覆盖均 ≥ 0.997。
- 本机层导出验证：ORA 成功；ImageMagick 存在时 PSD 成功。

详见 `docs/SELF_ACCEPTANCE.md`。

> 合成验收不能替代真实出版数据验收。最终“出版级”门槛仍要求建立同作品的 100–300 页人工真值集，统计页面配对、区域身份匹配、残留日文、误伤线稿、排字越界和人工复核率。程序已把这些证据与 QA 接口保留下来。

## 目录

```text
src/manga_hd_transfer/
  pairing.py          页面指纹与顺序配对
  registration.py     SIFT / LightGlue / LoFTR + RANSAC
  ocr.py              Paddle、sidecar、低置信度复识
  bubbles.py          气泡实例 / safe area / TextUnit
  matching.py         跨版本身份匹配、拆分/合并检测
  masking.py          像素级/多边形日文清字 mask
  inpainting.py       solid / OpenCV / external LaMa
  lettering.py        中文约束排字
  qa.py               出版 QA
  pipeline.py         全流水线
  review.py           本地人工复核编辑器
  review_apply.py     Review 回写与重生成
  export.py           PNG / ORA / PSD layer export
```

## 参考与许可

本仓库没有复制下列项目代码；它们作为算法/产品设计参考或可选外部后端。接入前请分别确认项目和模型权重许可证：

- `hgmzhn/manga-translator-ui`：Replace Translation 业务原型。
- `zyddnys/manga-image-translator`：漫画翻译流水线。
- `dmMaze/BallonsTranslator`：编辑器、mask/inpaint/排字工作流。
- `dmMaze/comic-text-detector`：漫画文本框、文本行、segmentation 思路。
- `PaddlePaddle/PaddleOCR`：PP-OCRv5。
- `cvg/LightGlue`：局部特征匹配。
- `zju3dv/LoFTR`：困难图像匹配后备。
- `advimman/lama`：复杂背景 inpainting。

详细吸收边界见 `docs/REFERENCE_PROJECTS.md`。
