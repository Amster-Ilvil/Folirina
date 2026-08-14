# v0.8.14

## 修复

- 修复 v0.8.13 新增版式完整性检查中的 `NameError: _safe_bbox is not defined`。
- `pipeline.py` 改用本模块自有 `_mask_bbox()`，不再依赖 `lettering.py` 私有函数。
- 单页处理成功后自动跳转到 **替换工作台 → 最终结果**，直接展示生成的 `final.png`。
- 整册批处理完成后不强制切换结果页，避免打断批处理工作流。
