# v0.6 降本增效 / 批量 / Apple MPS 设计

## 1. Cheap-first 自动路线

`registration.backend=auto`：

1. 模糊结构 + phase correlation 同源/近同源预判；
2. OpenCV SIFT/ORB；
3. 只有低置信度页才考虑 LightGlue；
4. 仍低置信度再考虑 LoFTR。

Auto 默认 `allow_model_downloads=false`。即使 LightGlue/Kornia 已安装，如果对应权重不在本地缓存，也不会为了 Auto 路线静默联网下载。用户显式选择 LightGlue/LoFTR 时仍由上游行为负责模型准备。

同源但分辨率不同也可走 `fast-resize-phase`：先在缩略图坐标验证结构一致性，最终矩阵仍使用真实 source→target 尺寸，因此不会牺牲最终几何精度。

## 2. Apple Silicon / MPS

统一 `runtime.py`：

- `auto` 在 Apple Silicon + 可用 PyTorch 时选择 `mps`；
- LightGlue / LoFTR / MangaLens / Torch 局部超分共享同一 device policy；
- MPS 推理用全局 accelerator lock 串行，避免统一内存并发尖峰；
- `PYTORCH_ENABLE_MPS_FALLBACK=1` 在 macOS、Torch import 前设置；
- 每 N 页清 allocator cache，但模型对象继续常驻；
- 可设置 MPS per-process memory fraction。

`首次安装_MPS_AI加速.command` 只安装运行库/LightGlue 代码，不内置漫画模型或超分权重。

GUI 提供三档运行策略，但底层仍保存为显式参数：

- **省资源**：CPU 线程约 35%，MPS 内存 fraction 0.68，批量预检 1 线程；
- **智能平衡**：CPU 线程约 50%，MPS 内存 fraction 0.82，批量预检 2 线程；
- **性能优先**：CPU 线程约 75%，MPS 内存 fraction 0.90，批量预检 4 线程。

MPS 深度推理仍通过统一 accelerator lock 受控串行，不会因为选择“性能优先”就在统一内存上并发堆多个模型推理。

## 3. 模型常驻

整册不再每页重复初始化：

- LightGlue extractor + matcher：按 feature/device 缓存；
- LoFTR：按 device 缓存；
- MangaLens/YOLO：按本地 checkpoint 路径缓存；
- Spandrel/Torch 超分：按 checkpoint/device 缓存。

## 4. 三层缓存

每页 `pages/<target-stem>/.cache/` 可保存：

- registration；
- source / target OCR；
- source / target bubble instance + mask/safe-mask。

完成页面另有 `project.json` job fingerprint。重跑整册时，如果输入和影响成品的配置没有变化，直接断点跳过；运行线程数、MPS/CPU 选择、cache 开关和 resume 开关不会错误地使成品 fingerprint 失效。

## 5. 批量处理

- 页面目录采用 target stem，插入/删除前面页面不会让后续缓存整体失效；
- 多线程只用于 resume 文件哈希/JSON 预检等独立 I/O；
- CV/MPS 主推理保持受控串行，避免显存/统一内存抖动；
- 单页失败默认记录并继续下一页；可设置 stop-on-error；
- 每页写 `batch_manifest.json`，崩溃后仍可继续；
- GUI 显示当前页、缓存命中、处理路线，并提供安全取消。

## 6. 实测

用户提供的同源真实页面：

- v0.6 Auto：`fast-phase-identity`，配准中位约 0.016 s；
- 强制 SIFT：约 1.083 s；
- 单配准阶段约 67.5× 加速；
- 最终仍为 5/5 区域替换、QA 0/0；
- 替换 mask 内 100% 与中文参考一致；mask 外 100% 与日文母版一致；
- 与 v0.5 最终输出逐像素完全一致。

6 页合成批量测试：首次约 3.89 s，第二次断点重跑约 0.12 s，6/6 resume hit，约 32.27×；实际时间随磁盘、图片尺寸与模型不同而变化。
