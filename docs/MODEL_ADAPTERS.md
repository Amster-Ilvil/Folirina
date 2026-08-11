# Model adapters

## 原则

模型是可替换 backend，工程 JSON/schema 才是稳定接口。这样：

- Apple Silicon、CUDA、CPU 可以使用不同模型；
- 模型未安装时核心算法和测试仍可工作；
- 不把下载权重、许可证和隐私文件打进主包。

## OCR

### Paddle

配置：

```json
{
  "ocr": {
    "source_backend": "paddle",
    "target_backend": "paddle",
    "source_lang": "ch",
    "target_lang": "japan",
    "ocr_version": "PP-OCRv5"
  }
}
```

source 会默认低置信度 crop retry；target 不改写日文内容，只利用 polygon/confidence 做 geometry。

### Sidecar

适合：comic-text-detector、自有 OCR、Apple Vision OCR、Paddle 独立批处理、云 OCR。

每个 block 最好包含 polygon + text + confidence + kind + mask_path。

## Bubble segmentation

核心 fallback：`seeded_white`。

生产建议：专用 speech-bubble instance segmentation 输出 sidecar。每个实例可提供：

- `mask_path`：完整 bubble interior；
- `safe_mask_path`：若模型/后处理已经生成；
- `kind`：speech / narration。

若没有 safe mask，核心会自动向内腐蚀。

## LightGlue

配置：

```json
{
  "registration": {
    "backend": "lightglue",
    "feature": "aliked"
  }
}
```

也可用 `sift` / `disk`。默认 `auto` 在没安装 LightGlue 时不会阻塞。

## LoFTR

```json
{
  "registration": {
    "backend": "loftr"
  }
}
```

通过 Kornia 调用；第一次使用预训练模型时可能需要取得权重。

## LaMa

任何 wrapper 只要符合：

```text
command --input input.png --mask mask.png --output output.png
```

即可使用。占位符 `{input}` `{mask}` `{output}` 会替换成临时文件。
