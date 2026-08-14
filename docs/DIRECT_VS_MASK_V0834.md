# v0.8.34 Direct Patch 与 Mask Transfer 设计契约

本文件锁定两个概念，防止后续开发再次把它们混成同一种“精准蒙版”。

## 1. Direct Patch：直接扣整块 SOURCE 原像素

适用条件：SOURCE 与 TARGET 是同一页，并且版面、气泡位置和局部结构一致或可通过受限注册可靠对应。

处理语义：

1. 对页面做 OCR-free 同页预检和注册。
2. 在 SOURCE 中找到可迁移的完整白色气泡 / 白底文本框内部。
3. `geometry` 只用于确定“扣哪里”和保护气泡边线；最终写入内容是 SOURCE 的原始 raster patch。
4. SOURCE 的**白色背景 + 中文字形**作为一个整体迁移，不把中文拆成 OCR 字符，也不单独生成 target clear mask。
5. identity 页面 1:1；非同尺寸同版页最终只允许 local similarity（uniform scale + small rotation + translation）。Affine / homography 只负责定位。
6. 写入前不对 TARGET 做 OCR、清字或 inpaint。
7. 若页面/区域不能证明安全，显式 `direct_patch` 必须拒绝并保留 TARGET，不能偷偷切换到 Mask/OCR。

因此，Direct 可以内部使用一个“提取区域的二值几何形状”，但这个形状的角色是 **patch extraction / border guard**，不是“蒙版清字合成”的 transfer mask。两者不能因为都出现 mask 数组就被视为同一算法。

## 2. Mask Transfer：受安全蒙版约束的目标感知合成

适用条件：SOURCE / TARGET 虽然对应，但目标容器底色、纹理、网点、渐变、局部结构或版本细节需要保留。

核心必须分开保存：

- `geometry_mask`：描述气泡 / 文本框的真实几何，包括边界证据。
- `transfer_mask / clear_mask`：描述最终允许改写 TARGET 的安全内部区域。
- `ink_mask / artwork_mask`：描述 SOURCE 中文墨迹或需要迁移的完整特效字形。

处理语义：

1. 保护 TARGET 气泡边框、人物与背景。
2. 在 transfer/clear mask 内清除 TARGET 日文。
3. 根据容器类型选择 `interior / ink-only / artwork / hybrid` 合成策略。
4. 彩色、网点、渐变容器优先保留 TARGET background/texture，再迁移 SOURCE 中文 ink。
5. QA 检查日文残留、SOURCE 字形完整度、边框损伤、mask 外改写和颜色保留。

## 3. 二者的硬边界

| 项目 | Direct Patch | Mask Transfer |
|---|---|---|
| 目标 | 同版页整块原像素恢复 | 目标感知的安全局部合成 |
| SOURCE 白底 | 与中文一起直接覆盖 | 视策略决定是否覆盖 |
| TARGET 日文清除 | 不单独清除，靠 SOURCE 整块覆盖 | clear mask 内主动清除 |
| TARGET 彩色/纹理保留 | 不适合；默认拒绝 | 适合，优先保留 |
| OCR | 0 | 精准路线原则上 0，极端 fallback 由更高层决定 |
| Inpaint | 0 | 可按策略使用 |
| 最终字形变换 | identity/local similarity | identity/local similarity；高级变换只定位 |
| 失败行为 | 显式 Direct：拒绝，不 fallback | Review / 上层 hybrid fallback |
| 独立产物 | `direct_patch_*` | `mask_transfer_*` |

## 4. Auto Planner

`auto` 的职责是选择路线，不是把路线重新混在一起：

```text
same-page precheck
        ↓
registration
        ↓
Direct 安全？ ── YES → Direct Patch → QA
        │
        NO
        ↓
Mask Transfer → QA / Review
```

显式 `direct_patch` 没有从 Direct → Mask 的箭头；只有 `auto` 才有。

## 5. v0.8.34 回归约束

自动测试至少要锁住：

- `direct_patch` 与 `mask_replace` 配置命名空间互不污染。
- Direct 成功时 `direct_patch_used=true` 且 `mask_route_used=false`。
- Direct 成功时 SOURCE OCR 状态为 skipped。
- Direct 被强制判定不安全时，输出与 TARGET 完全一致。
- Direct 被拒绝时不生成 `mask_transfer_layer/mask`，也不进入 OCR/reletter。
- Direct 有独立 JSON / layer / region debug 产物。
