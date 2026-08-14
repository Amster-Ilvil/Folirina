# v0.8.9 — Apple Live Text / Novel Formatter route

- macOS `apple` OCR 默认改为 Swift VisionKit ImageAnalyzer / Live Text，不再默认使用 PyObjC VNRecognizeTextRequest。
- VisionKit Helper 失败后自动在本次任务熔断并回退 `ExtractText` macOS 快捷指令。
- 新增 transcript-only paired-region OCR：paired-diff 负责气泡 geometry，Apple OCR 只读取每个源中文气泡中的 Unicode 文本。
- 目标页不再为 Apple Live Text 重跑 OCR bbox；目标几何由高清气泡直接生成，普通对白可走 HD re-letter。
- 新增 `apple_shortcut`、`apple_visionkit`、`apple_legacy` backend；旧 PyObjC 路线明确降级为 Legacy。
- 新增 Swift Helper 源码与 `build_apple_live_text_helper.command`。
- `首次安装_Apple_Vision_OCR.command` 改为编译 Live Text Helper并检查 ExtractText，不再安装 PyObjC。
- 新增 `首次安装_Apple_Vision_Legacy.command` 供旧路线兼容。
- 修复 GUI 切换 OCR 时只改 `backend`、却遗留 `source_backend/target_backend` 的问题。
- 新增 Apple Live Text fallback/paired-region 回归测试。
- Apple Live Text 两条路线都失败时默认软降级：批处理继续使用可还原的中文字形候选并产生 OCR 证据 QA，不再让 209 页逐页异常终止。
- OCR 高清重排改为“气泡局部 4× 超采样”，不再为每个气泡创建整页 4× PIL 画布；真实大页的 OCR 重排批量性能显著改善。
- 黑白中文版 → 彩色母版且 photo-pair 已稳定找到 ≥3 个区域时，跳过通常为空但非常耗时的 v0.8 barrier supplement；真实 006 回归从长时间 barrier 扫描降到约 15 秒整页，仍保持 6/6 区域。

## Live Text 漫画排版补强

- transcript-only Apple OCR 不再用整颗气泡的长宽比猜方向；逐气泡从旧中文版的实际墨迹连通域推断竖排/横排，并把 `orientation_hint` 写入 OCR block。
- 对证据模糊的中日文漫画对白，默认倾向竖排；明显宽向文字仍保持横排。
- OCR region cache 加入方向策略版本，旧缓存不会把错误方向带进新版本。
- 高清文字 4× 超采样改为只在气泡局部画布执行，避免整页每个气泡反复创建超大画布。
- 黑白中文版→彩色日文版已有足够 photo-pair 区域时跳过高成本且常为空的结构差分补扫。
