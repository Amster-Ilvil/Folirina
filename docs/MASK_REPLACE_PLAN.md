# 气泡 / 文本框蒙版替换设计与实现

## 目标

新增一条与“中文 OCR 后重新排字”平行的迁移路径：

> 从旧版中文汉化图中只提取**包含中文译文的气泡和旁白/文本框**，将其作为局部 patch 迁移到高清日文版的对应气泡/文本框。

最终画面仍以高清日文版为母版。默认只替换气泡/文本框**边线以内**，保留高清版原始气泡边线与页面线稿。

## 为什么不能直接整页 resize 后贴

旧中文版与高清版通常存在：

- 分辨率不同；
- 裁边、扫描偏移、轻微旋转；
- 气泡尺寸有几像素差异；
- 汉化版可能重新绘制过气泡；
- 旧文字像素较糊。

因此蒙版替换必须把“几何对齐”和“清晰度增强”彻底分离。

## 实际实现流水线

```text
旧中文版                     高清日文版
   │                              │
中文 OCR / 气泡实例              日文 OCR / 气泡实例
   │                              │
只保留含中文文字的气泡/文本框     目标气泡/文本框
   └──────────────┬───────────────┘
                  │
        整页 source→target 配准
       SIFT / LightGlue / LoFTR
                  │
          气泡实例 Hungarian 匹配
       mask overlap + centroid + shape
                  │
          目标气泡 bbox 局部收敛
                  │
     ECC 只允许小范围 translation 精校
                  │
      旧中文 bubble patch 单独超采样
      （坐标仍使用原始分辨率计算）
                  │
        target bubble mask 最终裁切
                  │
    默认向内 3 px，保留高清气泡边线
                  │
           alpha composite
                  │
         Mask Replace QA Gate
```

## 三种工作模式

### `reletter`

原有高质量路径：OCR 中文内容 → 清除高清日文 → 重新排字。

适合：

- 旧汉化字体质量差；
- 希望统一字体；
- 需要重新做出版排版。

### `mask_replace`

只做旧中文气泡/文本框 patch 迁移。

优点：

- 保留旧译版原来的断行、字形、强调与排版；
- 不需要重新决定每个气泡如何断行；
- 日文文字由旧中文版对应的气泡内部直接覆盖，不需要生成式清字。

默认：`preserve_target_border=true`，因此不会把旧版低清气泡边线一起覆盖。

### `hybrid`

推荐整册自动处理：

1. 优先尝试蒙版替换；
2. 如果 bubble match / mask IoU / coverage / spill 任一不达标，该气泡不强贴；
3. 自动退回 `reletter` 高清重排路径。

## 清晰度增强不会改变几何

这是本实现最重要的约束。

超分/放大只发生在**源气泡 crop 的采样层**：

```text
source crop --(Lanczos / external SR)--> high-sampling patch
       │
       └── SR patch coordinates -> original source coordinates
                                  -> target registration
```

因此即使源 patch 放大 2x/4x，最终位置与大小仍由：

- 原始 source 坐标；
- 页面 registration；
- target bubble 实例；

共同决定。SR 不参与目标 bbox 的计算，不会让气泡“越放越大”。

### 内置清晰度策略

- `off`：不增强；
- `lanczos`：Lanczos4 + 轻微 unsharp；
- `auto`：配置了外部 SR 时优先外部，否则 Lanczos；
- `external`：可连接 Real-ESRGAN 等本地 wrapper。

外部命令占位符：

```json
{
  "mask_replace": {
    "sr_backend": "external",
    "sr_command": "python realesrgan_wrapper.py --input {input} --output {output} --scale {scale}"
  }
}
```

程序不捆绑、不自动下载 SR 模型。

## 精准对齐策略

### 1. Page registration

旧中文版 → 高清日文版的整体映射仍使用项目现有 SIFT / LightGlue / LoFTR。

### 2. Bubble identity matching

不依赖文字 bbox IoU。匹配成本使用：

- 全局配准后 bubble mask IoU；
- 气泡中心距离；
- contour shape similarity；
- speech / narration 类型一致性。

使用 Hungarian 一对一分配。

### 3. Local size convergence

`local_fit=bbox/ecc` 时，根据**目标气泡真实 mask bbox**对已配准的源气泡做第二级 x/y 尺寸收敛。

局部尺寸修正超过 `max_local_scale_change` 时拒绝自动应用，避免把错误气泡硬拉伸到目标框。

### 4. ECC micro alignment

`local_fit=ecc` 在 bbox 收敛之后，只允许很小的 x/y translation 修正。

只有当 IoU 实际提升时才采用；偏移超过气泡尺寸比例门槛时自动丢弃。

## 默认高清边线保护

`preserve_target_border=true` 时：

- 目标 bubble mask 向内腐蚀 `border_inset_px`；
- source 中文 patch 只能出现在该内部区域；
- 高清原版气泡边线、尾巴、外部线稿保持原样。

这是默认推荐路径。

如果后续确实遇到“汉化版重新绘制了特殊文本框”的情况，可以关闭边线保护，但该模式应进入更严格 QA。

## 自动 QA Gate

每个 bubble patch 必须同时满足：

- bubble match confidence ≥ `min_match_confidence`；
- mask IoU ≥ `min_mask_iou`；
- target interior coverage ≥ `min_target_coverage`；
- source spill ratio ≤ `max_spill_ratio`；
- local scale correction 不超过门槛；
- 页面 registration 通过。

不满足时：

- `mask_replace`：标记为 error，等待 Review；
- `hybrid`：蒙版替换被拒绝，同时自动退回高清 OCR 重排，拒绝原因保留为 warning。

## 输出证据

每页新增：

- `mask_transfer_layer.png`：透明旧中文气泡迁移层；
- `mask_transfer_mask.png`：最终实际覆盖区域；
- `mask_transfer.json`：每个气泡的匹配、SR、IoU、coverage、spill、局部修正；
- `debug_mask_replace.png`：覆盖范围叠图；
- ORA / PSD 中独立 `Chinese bubble patch transfer` 层。

这使最终成品可回溯，也方便出版人工复核。

## 当前代码位置

- `src/manga_hd_transfer/mask_transfer.py`：完整蒙版替换核心；
- `config.py::MaskReplaceConfig`：参数；
- `pipeline.py`：`reletter / mask_replace / hybrid` 三模式调度；
- `qa.py::run_mask_replace_qa`：专用 QA；
- `gui.py`：工作台模式和精确度/超分设置；
- `tests/test_mask_transfer.py`：蒙版迁移与 pipeline 集成测试。
