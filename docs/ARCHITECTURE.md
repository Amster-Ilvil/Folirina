# Architecture

## 1. 设计目标

系统要解决的不是单图 OCR，而是两个版本之间的结构化内容迁移：

- Source A：旧版低清中文汉化图。
- Source B：高清日文原图。
- Output：以 B 为画面母版，保留 A 的中文译文内容，并重新完成清字、排字和出版级 QA。

所有阶段必须可独立测试、可替换、可回滚。

## 2. 总体模块

```text
                         ┌────────────────────┐
Old Chinese Pages ──────▶│ Page Pairing       │◀────── HD Japanese Pages
                         └─────────┬──────────┘
                                   │ paired pages
                                   ▼
                         ┌────────────────────┐
                         │ Registration        │
                         │ global + local      │
                         └──────┬────────┬────┘
                                │        │
             ┌──────────────────┘        └──────────────────┐
             ▼                                                ▼
┌────────────────────────┐                         ┌────────────────────────┐
│ Chinese Extraction     │                         │ HD Structure Analysis  │
│ detect + OCR + group   │                         │ JP text + bubbles      │
└────────────┬───────────┘                         └────────────┬───────────┘
             │                                                │
             └──────────────────┬─────────────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ Cross-Version Matcher  │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Text Removal / Inpaint │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Lettering Optimizer    │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ QA + Review Queue      │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Export                 │
                    └────────────────────────┘
```

## 3. Page Pairing

职责：确定两套图片中哪些页面对应。

输入：文件序列、缩略图。

输出：

```json
{
  "source_page": "old/001.jpg",
  "target_page": "hd/003.png",
  "confidence": 0.998,
  "signals": {
    "sequence": 0.8,
    "phash": 0.96,
    "feature_match": 0.99
  }
}
```

不能假设两边页码永远一一对应；必须处理封面、广告、staff 页、双页拆分等情况。

## 4. Registration

### 4.1 Global registration

优先估计：

- translation + scale
- similarity
- affine
- homography

模型复杂度逐级增加；只有低阶模型无法解释匹配点时才升级。

### 4.2 Feature backends

Backend 接口：

```python
class FeatureMatcher:
    def match(self, image_a, image_b) -> MatchSet:
        ...
```

实现候选：

- OpenCV SIFT/ORB baseline
- ALIKED + LightGlue
- DISK + LightGlue
- LoFTR fallback

### 4.3 Registration quality

至少计算：

- match count
- RANSAC inlier ratio
- median reprojection error
- spatial coverage
- forward/backward consistency

若控制点只集中在一个角落，即使误差低也不能判为高置信度。

### 4.4 Local registration

对于局部裁剪、纸张弯曲、汉化重绘：按 panel/bubble neighborhood 建局部映射。

## 5. Chinese Extraction

### 5.1 Detection data model

```json
{
  "id": "cn_001",
  "polygon": [[0,0],[1,0],[1,1],[0,1]],
  "text": "……",
  "ocr_confidence": 0.0,
  "reading_order": 0,
  "kind": "speech",
  "bubble_id": "old_bubble_01",
  "candidates": []
}
```

### 5.2 OCR ensemble

OCR provider 使用统一接口：

```python
class OcrEngine:
    def recognize(self, crop, hints=None) -> OcrCandidate:
        ...
```

默认单模型即可完成高置信度区域；只有低置信度区域才触发 ensemble。

## 6. HD Structure Analysis

输出两类几何：

1. **text mask / text polygon**：用于准确清除日文。
2. **bubble / narration safe region**：用于重新排版中文。

不能把两者混为一个矩形框。

### 6.1 Bubble representation

```json
{
  "id": "bubble_01",
  "mask_ref": "...",
  "bbox": [0,0,0,0],
  "safe_polygon": [],
  "tail_exclusion": [],
  "kind": "speech",
  "confidence": 0.0
}
```

## 7. Cross-Version Matcher

### 7.1 Candidate generation

旧中文版区域通过 registration transform 投影到高清版坐标，生成局部候选。

### 7.2 Cost function

```text
cost =
  w1 * normalized_center_distance
+ w2 * shape_difference
+ w3 * bubble_mismatch
+ w4 * reading_order_difference
+ w5 * local_visual_inconsistency
+ w6 * class_mismatch
```

不使用固定 IoU 0.3 作为唯一判断。

### 7.3 Assignment

- 一对一：Hungarian/min-cost matching。
- 一对多/多对一：建立 group node，再进行匹配。
- unmatched：保留并进入 Review Queue，不能强行塞到最近框。

## 8. Masking and Inpainting

### 8.1 Mask policy

```text
text segmentation mask
  -> remove tiny noise
  -> adaptive dilation
  -> clip to allowed repair region
  -> protect bubble border / line art
  -> risk classification
```

### 8.2 Repair backend

```python
class Inpainter:
    def inpaint(self, image, mask, context) -> InpaintResult:
        ...
```

后端：

- flat-fill
- texture/patch fill
- LaMa
- manual-required

QA 不通过时可换后端重跑，而无需重新 OCR/配准。

## 9. Lettering Engine

### 9.1 Layout representation

```json
{
  "text": "中文译文",
  "font_family": "...",
  "font_size": 42,
  "tracking": 0,
  "line_spacing": 1.05,
  "alignment": "center",
  "rotation": 0,
  "lines": [],
  "fit_score": 0.0
}
```

### 9.2 Constraint solver

硬约束：

- 所有 glyph 落在 safe area。
- 不跨越禁止区域。
- 不低于最小可读字号（除非人工确认）。

软目标：

- 字号尽量大。
- 行宽均衡。
- 视觉中心接近原文本中心。
- 避免单字孤行。
- 中文标点禁则合理。

## 10. SFX / 艺术字独立路径

拟声词不能与对白共用普通字体排版逻辑。

三档：

1. 保留日文拟声词，不迁移。
2. OCR 中文后人工字体/变形排版。
3. 旧中文版像素风格迁移：必须先做局部 registration 和 alpha/mask 提取，再进行高分辨率重建。

默认不自动执行第 3 档。

## 11. QA Engine

每个 stage 输出自己的 confidence 和 diagnostics；最终 QA 只聚合，不重新猜测。

`PageQA` 示例：

```json
{
  "status": "review_required",
  "registration": {"score": 0.94},
  "ocr": {"low_confidence_blocks": ["cn_04"]},
  "matching": {"ambiguous": ["cn_09"]},
  "cleanup": {"residual_japanese": ["jp_07"]},
  "lettering": {"overflow": []}
}
```

## 12. Artifact-first 调试

每页都应可选输出：

```text
debug/page_0001/
  01_pairing.jpg
  02_registration_matches.jpg
  03_registration_warp.jpg
  04_cn_text_regions.jpg
  05_hd_text_regions.jpg
  06_bubbles.jpg
  07_cross_version_matches.jpg
  08_cleanup_mask.png
  09_inpainted.png
  10_lettering_safe_area.jpg
  11_final_overlay.jpg
  page.json
  qa.json
```

这比单纯打印日志更适合定位出版级视觉问题。

## 13. 可测试性

每个模块必须支持离线 golden test：

- registration：固定页面对，比较 transform/error。
- OCR：固定 crop，比较字符级结果。
- matcher：固定区域图，比较 assignment。
- mask：比较 IoU + protected-pixel violation。
- lettering：检查 glyph mask 是否越界。
- export：像素尺寸、色彩空间、alpha、文件完整性。

## 14. 性能策略

质量优先，但避免所有页面都跑最重模型：

```text
fast path success
  -> continue
fast path low confidence
  -> robust model
still uncertain
  -> review queue
```

缓存所有阶段结果，以页面内容 hash + 配置 hash 作为 cache key。