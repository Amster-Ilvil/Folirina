# Review workflow

## 为什么必须有 Review

“出版级”不能把不确定性隐藏掉。自动系统的任务是把 90%+ 普通页做得稳定，并把剩余困难页压缩成明确可复核的队列。

## 页面证据

Review 三栏：

1. 旧中文版：验证 OCR 中文内容；
2. 高清日文母版 + clear mask：验证要删的像素；
3. 当前输出：验证排字/修复。

下方每个 source TextUnit 可以：

- 勾选是否应用；
- 编辑中文；
- 重新选择目标 TargetUnit。

mask canvas：

- 增加：把漏掉的日文补入 clear mask；
- 擦除：保护人物线稿、气泡边线、网点。

保存后 `review_overrides.json` 和 `manual_clear_mask.png` 不会修改原图。

## 重新生成

点击应用或运行：

```bash
mhd-transfer apply-review output/pages/0001_page
```

系统重新从 `target_original.png` 开始：

1. 使用 manual/自动 mask；
2. 重新 inpaint；
3. 应用人工确认匹配和文本；
4. 重新 fit safe area；
5. 导出 `final_reviewed.png`；
6. 导出 `editable_reviewed.ora/.psd`。

因此反复修改不会产生“在旧 final 上覆盖新文字”的累积损伤。
