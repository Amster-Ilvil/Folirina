# 运行卡死 / 残线 / 黑点 / 清日文画笔 修复计划（2026-08-14）

## 1. 问题归因

### A. 运行中容易“卡死”
- 主流程整页处理已经在线程里，但“仅执行去字 / 应用蒙版到最终结果”仍直接在 GUI 主线程执行。
- 在较大彩图页上，这两步会阻塞 Qt 事件循环，表现为窗口假死、按钮无响应。

### B. 白色气泡替换后残留线条 / 黑点 / 多余小色块
- 白底完整气泡属于 text-only transfer，但后处理只做了基础清字与回写，没有额外做：
  1. WHITE 容器外溢像素回退；
  2. 白纸区细碎残留点清理；
  3. 气泡内浅灰短线清理。
- 因此会出现：
  - 日文小残笔；
  - 黑点；
  - 气泡上缘/角落细线；
  - 个别白底区域误沾到附近彩图像素。

### C. 需要专用“清日文画笔”
- 当前已有清除蒙版编辑器，但入口表达不够明确。
- 需要把它明确成“只清日文”的专用画笔工作流，避免用户误以为会改坏底图或重排文字。

## 2. 本次落实的修复

### 2.1 GUI 不卡死
- 新增 `PageActionWorker`，把以下操作移出主线程：
  - `仅执行去字`
  - `应用蒙版到最终结果`
- 操作运行期间禁用相关按钮并显示状态栏提示，避免重复点击造成假死观感。

### 2.2 白底气泡后处理增强
- 新增 `white_container_paper_mask()`：
  - 建立白底气泡内部稳定纸面区域，供白底专用清理逻辑复用。
- 新增 `cleanup_white_container_line_artifacts()`：
  - 清理白底内部、不受中文文字支持的浅灰短线/细碎残痕。
- 白底 path 现在额外执行：
  1. spill component 回退（把不该写到彩图区域的小块改动还原成 TARGET）；
  2. TARGET-only residual speck cleanup；
  3. rendered white-container artifact cleanup；
  4. faint line artifact cleanup。

### 2.3 专用“清日文画笔”入口
- 将按钮文案从“编辑清除蒙版…”调整为：
  - `日文清除画笔…`
- 让流程语义更清楚：
  - 这支画笔只是在 `manual_clear_mask.png` 上画；
  - 不直接破坏原图；
  - 适合清理残留日文、黑点、浅灰线、小边角异常。

## 3. 回归测试

### 单元测试
- 全量：`114 passed`
- 新增覆盖：
  - `tests/test_white_container_cleanup.py`
    - 白底纸面 mask 不误入彩图区域
    - 白底浅灰线清理不会伤到受 SOURCE 支持的中文笔画

### 实图验收（用户提供后两张图）
- 测试输入：
  - SOURCE：`858249cf-804f-5cc5-8b7d-8c0af0e3b754.jpg`
  - TARGET：`7423d6d1-512a-529a-a457-ca65fe8bc660.jpg`
- 运行命令：
  - `python scripts/real_pair_acceptance.py --mode auto --ocr-backend none ...`
- 验收结果：
  - `pass: true`
  - `outside_transfer_changed_pixels: 0`
  - `protected_border_changed_pixels: 0`
  - `qa_errors: 0`

## 4. 建议使用流程
1. 自动处理当前页
2. 查看最终结果
3. 如仍有少量残留日文 / 黑点 / 顶边浅灰线：
   - 点 `日文清除画笔…`
   - 只刷需要清掉的区域
   - 先 `仅执行去字` 检查
   - 再 `应用蒙版到最终结果`

## 5. 这版的目标
- 自动路径先尽量减少残留与外溢；
- 对自动仍难 100% 清干净的少量边角，用专用清日文画笔收尾；
- 保持 TARGET 彩图背景权威，不再把 SOURCE 的底色/纸面/肤色贴回去。
