# v0.5 自验收

日期：2026-08-11

## 核心回归

- `python -m compileall src tests benchmarks`：PASS
- `pytest -q`：**12/12 PASS**
- `python -m manga_hd_transfer.selftest`：PASS
- 原随机配准 / 排字 benchmark：**20/20 PASS**
- 蒙版替换随机 benchmark：**24/24 PASS**

## 用户提供真实页面对

输入：同一页日文 JPG + 已正确嵌字中文 PNG。

- 页面配准置信度约 0.999811
- 自动提取实际翻译区域：5
- 应用替换：5/5
- QA：0 error / 0 warning
- 替换蒙版内对中文参考逐像素一致：100%
- 替换蒙版外对日文母版逐像素一致：100%

自动找到的是 4 个对白气泡 + 右下角说明文本框；未翻译顶部小气泡和画面拟声词未被迁移。

## UI

GUI 已重写为 PySide6/Qt，并通过 Python 静态编译检查。当前 Linux 执行容器没有 PySide6 且禁止联网，因此这里无法声称已经实际启动 macOS Qt 窗口；Mac 启动脚本会在首次运行时安装 `.[gui]`。核心算法验收与 GUI 运行时验收明确分开记录。
