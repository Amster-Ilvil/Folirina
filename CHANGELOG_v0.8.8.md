# v0.8.8 (2026-08-11)

## Apple Vision OCR 稳定性修复

- 修复 macOS 上 `VNImageRequestHandler.alloc().initWithCGImage_options_(cgimg, {})` 可能抛出 `NSInvalidArgumentException - key does not exist`。
- `options:` 改为原生 `Foundation.NSDictionary.dictionary()`，不再依赖 Python dict 的 Objective-C 代理。
- Vision handler 新增多路线初始化：URL → NSData → CGImage + orientation → CGImage；单一路线异常会自动尝试下一条。
- 每条初始化路线使用新的 Objective-C `alloc()` 对象，避免失败 initializer 后复用无效实例。
- Apple Vision 安装脚本改为显式升级 PyObjC 12.2.1+，避免旧版桥接元数据在新版 macOS 上继续被复用。
- 新增 3 项 Apple Vision handler 回归测试，包括用户实际遇到的 `NSInvalidArgumentException` 模拟。

## 回归

- 全量 pytest：45/45 通过。
- Linux 环境无法真正执行 macOS Vision.framework，因此 Apple Vision 真机路径通过桥接模拟回归测试覆盖；核心图像处理与既有测试保持通过。
