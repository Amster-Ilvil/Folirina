# v0.6 Self Acceptance

日期：2026-08-11

## 自动测试

- `pytest -q`: 17/17 PASS
- `python -m compileall`: PASS
- `mhd-transfer selftest`: PASS

新增覆盖：

- cheap-first fast identity registration；
- 低清同版 fast-resize phase route；
- runtime CPU/device policy；
- 整册断点续跑；
- registration stage cache 命中。

## 真实页面回归

输入使用用户此前提供的同页日文/中文对照，仅用于本地测试，不进入 ZIP。

- registration: `fast-phase-identity`
- confidence: 0.953869
- detected/applied: 5/5
- QA: 0 error / 0 warning
- transfer mask 内与中文参考逐像素一致：1.0
- transfer mask 外与日文母版逐像素一致：1.0
- v0.6 final 与 v0.5 final：逐像素完全一致

## 配准性能

5 次重复中位数：

- cheap Auto: 0.01605 s
- forced SIFT: 1.08333 s
- speedup: ~67.5×

该指标只代表这组“同源/近 identity”页面；困难跨版本页仍会升级到 SIFT / LightGlue / LoFTR。

## 批量

6 页合成整册：

- first run: 3.8878 s
- resume run: 0.1205 s
- resume hit: 6/6
- resume speedup: 32.27×
- QA errors: 0

## MPS 边界

当前构建/验收容器是 Linux，因此不能声称已经在本机实际执行 Apple Metal kernel。代码路径、设备选择、模型复用、MPS cache/memory policy 和 Mac 安装脚本已经实现；最终 Apple Silicon 实机吞吐/峰值统一内存仍需在 Mac 上跑同一 benchmark 验证。

## 发布包卫生

新增 `scripts/release_audit.py`，在 ZIP 前检查虚拟环境、缓存、模型权重、日志/数据库、常见凭据和私人 home 绝对路径。该检查不依赖 git checkout。
