# Self acceptance report

日期：2026-08-11

## 环境

- Python 3.13
- OpenCV 4.13.0
- SIFT available
- Noto Sans CJK available
- ImageMagick available
- PaddleOCR / LightGlue / Kornia deliberately **not installed** for core offline acceptance

这样可以证明核心测试不依赖在线权重下载。

## 1. Unit / integration tests

命令：

```bash
pytest -q
```

结果：

```text
........ [100%]
8 passed
```

覆盖：

1. 气泡安全区 + 中文排字；
2. target text mask 受 bubble border protection 限制；
3. Hungarian unit matching；
4. extra-page directory pairing；
5. 完整单页 pipeline；
6. SIFT affine/similarity registration；
7. OCR/bubble sidecar + pixel segmentation mask；
8. Review overrides 回写和重生成。

## 2. Built-in end-to-end selftest

```bash
mhd-transfer selftest
```

结果：PASS。

关键值：

- registration method: `opencv-sift+similarity`
- registration confidence: ~0.740
- inlier ratio: ~0.566
- median reprojection error: ~0.842 px
- spatial coverage: ~0.357
- auto match: PASS
- lettering: PASS
- QA errors: 0
- QA warnings: 0

## 3. Random registration benchmark

```bash
python benchmarks/synthetic_acceptance.py
```

20 组随机：

- scale 0.86–1.16；
- rotation ±3°；
- x/y translation ±35 px；
- source/target 气泡内加入不同文字形状，模拟语言变化。

结果：

- 20/20 PASS；
- median control-point error: **0.0508 px**；
- P95: **0.1166 px**；
- minimum registration confidence: **0.8380**。

原始 JSON：`benchmarks/latest_report.json`。

## 4. Lettering benchmark

四种中文长度：3 / 11 / 39 / 26 字符。

结果：4/4 PASS。

- chosen font size: 54 / 54 / 31 / 38 px；
- glyph safe coverage: 1.0000 / 0.99962 / 0.99793 / 0.99876。

门槛：≥ 0.997。

## 5. Packaging

由于执行容器无外网，普通 PEP517 build isolation 无法重新下载 setuptools；改用现有本地构建依赖：

```bash
pip install -e . --no-deps --no-build-isolation
```

成功；`mhd-transfer doctor` 和 `mhd-transfer selftest` 均可从安装后的 console script 运行。

这是执行环境网络限制，不是 package metadata 错误。正常联网环境可直接 `pip install -e .`。

## 6. Layer export

本机实际生成：

- `editable.ora`：成功；
- `editable.psd`：ImageMagick 存在时成功，文件非空。

## 7. 结论

**代码级自验收通过。**

但这里刻意不把“合成测试通过”写成“真实出版认证通过”。真实出版验收需要用户拥有版权/授权的真实成对旧中文与高清日文页面，并建立人工真值集。代码已经具备用于该验收的 project JSON、QA、debug overlay、Review Queue 与安全 gate。
