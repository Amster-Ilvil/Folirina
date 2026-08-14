# v0.8.10

- 完全移除 Apple Vision Legacy OCR 后端与 GUI 入口。
- 删除旧 Legacy 安装脚本、可选依赖、运行时状态探测与专用测试。
- macOS Apple OCR 仅保留：Swift VisionKit Live Text（默认）与 ExtractText 快捷指令回退。
- `apple` / `apple_live_text` / `apple_shortcut` 为当前 Apple OCR 路线；旧 Legacy backend 名称不再接受。
- 保留漫画 paired-diff 内部历史算法的 `legacy` 命名；这与已删除的 Apple OCR 后端无关。
