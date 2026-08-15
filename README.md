
## v1.3.11：读取已有结果并继续人工补漏

GUI 的“页面管理”新增 **读取已有运行结果…**。可以选择命令行/Codex 已生成的输出根目录、`pages/` 或单页工作区；程序优先读取 `project.json` 的 SOURCE/TARGET 配对，原输入路径失效时回退到页目录内的 `source_original.png` / `target_original.png`。读取不会重新处理页面或覆盖已有 final，进入“替换工作台”后可继续白气泡补漏、Reveal、自动 clear mask 和 TARGET 日文层擦除。

“仅擦 TARGET 日文层”现在有两种模式：**智能恢复 TARGET**（纸色/复杂背景 inpaint）和 **纯白涂抹**（有效笔刷区域直接写成 255 白色）。两种模式都先扣除中文保护层，并在最终一步 byte-for-byte 恢复中文保护像素。

## v1.3.8 — 深度稳定性 / 缓存一致性 / 无损加速

- 不改变 TARGET 背景权威、中文保护、Direct/Mask/Review 像素语义；用户 005 实页 `final.png` 与 v1.3.7 SHA-256 完全一致。
- 修复跨尺寸 SOURCE/TARGET 的 paired-diff 持久缓存误判：SOURCE bubble mask 保持 SOURCE 坐标，TARGET bubble mask 才要求 TARGET canvas；此前会造成“缓存显示大多命中但 paired_diff 仍重跑”。
- paired-diff 白色连通域标签按“图像+阈值”在单次请求内精确复用，减少重复 `connectedComponentsWithStats`，不改变候选/像素。
- stage cache JSON/NPZ 增加事务签名与原子写入；旧 v1.3.7 缓存安全兼容，半写入/错配缓存自动 miss 重算。
- 图像、JSON、OpenRaster、review result 同步改为唯一临时文件 + 原子替换；并发/强退时不再容易留下半文件或固定 `.tmp` 竞争。
- PSD 改为临时 `.psd` 成功后原子替换；ImageMagick 超时/失败不会破坏已有有效 PSD。
- 更新器增加安全解压、下载体积上限、staging 校验和失败回滚；当前 `.app` 在新版本完整复制验证前不移动。
- 本地 MangaLens / SR / RT-DETR / SAM2 权重缓存会感知同路径权重替换，避免长会话继续使用旧模型。
- 发布质量审计增加版本/CHANGELOG/policy/lineage 一致性和重复类方法检测；v1.3.8 稳定性测试被纳入永久 regression contract。

## v1.3.3 贴图清字回归修复 / 恢复 v1.3.0 残留清理能力

## v1.3.7 — 真实成对页 QA 与 paired-diff 加速

- 完整代码级核对优化 Plan：A1–A6、B1–B6、C1–C5、D1–D5、E1–E4、F1–F6 均已有执行路径，详见 `docs/OPTIMIZATION_PLAN_STATUS_v1.3.7.md`。
- 修复跨尺寸 SOURCE/TARGET 的真实验收脚本崩溃；验收现在在 TARGET 坐标系评估 registration、applied regions、content completeness、TARGET 背景保持和 residual heatmap。
- `paired_diff_v08` 对同一页的 illumination flatten / barrier connected-components 做请求内精确复用；不改变检测结果。
- 新增 paired-diff 阶段缓存，二刷无需重跑成对结构分析；cache key 只包含 `paired_diff_*` / `photo_pair_*` 像素相关配置。
- 用户提供的 005 实页测试：6/6 区域应用、QA error=0；v1.3.7 `final.png` 与 v1.3.6 基线 SHA-256 完全一致，首次处理约从 18–20s 降至 10.6s，二刷约 4.1s（测试容器）。


- 修复 v1.3.2 合并时意外丢失的 v1.3.0 清字逻辑：Direct 彩色/复杂容器重新补抓日文抗锯齿边缘，并恢复第二轮 TARGET 残留局部清理。
- 白色气泡/旁白框恢复 v1.3.0 的纸面识别与较宽容的真实清字 envelope：1px inset/guard + 小范围 clear halo，不再把日文边缘因为过度安全裁剪重新留回结果。
- 白底容器重新使用完整 TARGET 紧凑文字 + 小标点清理，并恢复 v1.3.0 的残留日文与浅灰短线后处理。
- 成对差异的彩色/高饱和容器恢复日文 AA fringe 捕获；渐变/网点底色自动改用 Telea，平坦底色才使用中值填充。
- 恢复 v1.3.0 的人工日文清除增量扩张、`effective_clear_mask.png`、`japanese_residual_heatmap.png` 与残留 QA。
- 保留 v1.3.2 的人工白气泡边框剥离、人工模式持久记忆、页面管理阻尼滚动、实验挖洞状态提示等 UI/Review 改进。
- 继续保持 TARGET 背景/色彩权威：只迁移 SOURCE 中文墨迹，不把低清 SOURCE 白底或彩图背景覆盖回高清 TARGET。

## v1.3.2 人工补漏 / 白气泡 / 实验挖洞 / 页面管理稳定性

- 白气泡人工模式新增“预览实际文字 Mask”：绿色为最终写入的 SOURCE 中文，红色为最终清除的 TARGET 日文，橙色为重叠；预览使用真实 safe inset、边框剥离和 X/Y 微调后的生效蒙版。
- 人工模式选择改为持久记忆：成功应用后写入 Qt `QSettings`，关闭并重启程序后仍恢复上次人工模式。
- 白气泡 text-only 主流程补齐纸面识别、边框保护、TARGET 残留日文清理、浅灰短线清理，选框/气泡边线不会被当成中文文字迁移。
- “整页对齐挖洞显中文”增加当前页真实执行状态：直接检查 `aligned_overlay_reveal*.png/json` 核心产物、应用区域数与拒绝原因，避免只看开关误以为已经生效。
- 页面管理缩略图使用独立的阻尼滚轮/触控板处理，限制 macOS `pixelDelta` 大幅跳动；滚动只移动视图，不改变当前页选择。
- 低清文字保护重新按 `reject_blurry_source` 本身生效；照片边缘裁切的中文继续以低置信候选进入 Review，而不是静默当成完整翻译。
- Auto/Mask/Direct 页面存在实际 `text_overrides` / `match_overrides` / unit-level override 时，会真正进入高清重新排字路径，不再被 raster-review 路由吞掉。
- 继续保持 TARGET 背景/色彩权威：不恢复旧版“整块 SOURCE 白底/彩图覆盖 TARGET”的行为。

## v1.2.3 实验模式真实页可见结果 / GUI 显式启用修复

- “显示实验模式”改为“显示并启用实验模式”：勾选后自动切换到 `aligned_overlay_reveal`，避免仍停在 Auto 导致结果原样保留日文。
- 按真实页放宽实验路线的配准、文字连通域和面积阈值；默认仍 `enabled=false / allow_in_auto=false / require_explicit_mode=true`。
- 默认 `erase_source=hybrid`，但 SOURCE 背景仍不得成为彩色 TARGET 的像素权威。
- 白容器恢复改为局部 ROI 计算，避免 hybrid 在高分辨率真实页逐区域全页连通域分析造成卡顿。
- 单区域面积超限改为 REVIEW 并交给 Reveal，而不是直接丢失候选；配准失败诊断现在同时记录实测指标与阈值。
- 对用户提供的真实 SOURCE/TARGET 页验证：实验路线 accepted，18 个区域实际应用，最终彩图保持 TARGET 背景并出现中文文字。

## v1.2.2 实验整页对齐挖洞显中文 / Aligned Overlay Reveal

- 新增实验路线 `aligned_overlay_reveal`，默认关闭且 GUI 默认隐藏；必须显式打开“显示实验模式”后才可选择。
- 复用既有整页配准，但采用更严格的 confidence / inlier / reprojection / spatial coverage 门槛。注册不达标或页面复核失败时，高清日文 TARGET 原样保留。
- 默认采用 **SOURCE ink-only**：配准后的中日差异墨迹只作为文字种子，再回收到完整文字连通组件；先清 TARGET 日文墨迹，再仅写入 SOURCE 中文墨迹，TARGET 始终是背景/色彩权威。
- 新增 text corridor、共享长线/大结构 guard、边界渐进保护、单区域/整页擦除面积上限，避免人物线稿、分镜线、气泡边框被当成文字。
- 可选 `full_raster_white` 只允许在高白度、低饱和、边框内缩后的局部走；彩色 TARGET 禁止 SOURCE 黑白背景露底。
- `auto` 默认完全不会选择该实验路线；即使 `enabled=true`，还必须同时设置 `allow_in_auto=true` 且 `require_explicit_mode=false`。Direct 仍保持优先级。
- 新增 `aligned_overlay_reveal_layer.png` / `aligned_overlay_reveal_mask.png` / `aligned_overlay_reveal_source_ink.png` / `aligned_overlay_reveal_regions.png` / `aligned_overlay_reveal.json` 诊断与复核产物。
- REVIEW / REJECT 区域会写入现有 `manual_effect_candidates`，可继续使用 Reveal 人工补漏；重新自动处理仍由 `result_state` 统一失效旧人工状态。
- `final.png` 仍只由 Pipeline → `result_state.commit_automatic_result()` 写入，新像素模块不拥有结果状态。

## v1.2.0 架构统一 / 全项目状态链收口

- 保持已经验证正常的 Direct / 精准蒙版 / Reveal / 白气泡 X/Y 微调行为不变，本轮主要做职责拆分和状态统一。
- 新增 `result_state.py`：统一管理 `final.png` / `final_reviewed.png` / `final_auto.png`、review sync、工程 artifacts 和重新处理后的状态失效；GUI、Web Review、Workspace 不再各维护一套“哪个结果最新”的规则。
- 新增 `manual_review_service.py`：人工补漏提交成为 Qt 无关事务，负责保存 Reveal、更新 overrides、调用 review compositor、最终像素校验；Qt 只负责交互和刷新。
- `schema_compat.py` 继续作为历史 JSON 唯一兼容入口，并增加 Web Review 非破坏合并，Web 保存复核不会再清掉 Qt 的 `manual_effect_regions`。
- Web Review、Workspace、Review Apply、Pipeline 全部接入共享状态契约；重新自动处理页面时统一清理旧 manual baseline / review sync。
- 修复旧 Page Manager / cache 状态为异常 bool 时可能报错的问题；异常旧状态现在安全降级而不是影响主流程。
- 统一运行时版本来源为 `version.py`，`__init__`、GUI、`pyproject.toml` 统一到 v1.2.0。
- 新增架构说明：`docs/ARCHITECTURE_V1.2.md`。

## v1.1.7 人工补漏 schema 归一化 / bool.get 全链修复

- 新增 `schema_compat.py`，在人工补漏/Review/Workspace 入口统一把历史 `bool / dict / list` JSON 形态归一化。
- `meta.direct_patch=true/false`、`meta.mask_replace=true/false` 会转换为稳定的 `{used: ...}` 路由结构。
- `manual_effect_regions`、`manual_reletter`、`review_regions`、`manual_effect_candidates`、`matches` 等列表会自动过滤布尔值和异常项。
- `artifacts`、`qa_summary`、`review_sync`、`text_overrides`、`match_overrides`、`unit_actions` 如果被旧缓存写成布尔值，将安全恢复为空对象，而不是调用 `.get()` 崩溃。
- GUI 的 Reveal 提交校验、人工补漏候选、撤销、人工重排字、Review audit 都使用相同的类型安全归一化。
- Workspace 解析同样归一化，切回工作台刷新时不会因为旧候选/旧 route schema 再次触发 `bool has no attribute get`。
- 真实 044 压力测试：故意把多个 project/review 字段改成布尔值并混入 bool 行，真实 Reveal 仍可成功应用，`preview_patch_exact=true`，`final.png == final_reviewed.png`。

## v1.1.3 人工补漏落盘链强化 / review_sync 可追踪

- Reveal 选框现在只是**搜索范围**，不再直接决定写入范围；实际写入受 SOURCE 中文文字层和 TARGET 日文文字层约束。
- 大范围 Reveal 增加 text corridor：人物、皮肤、网点、背景和未关联线条不进入最终补丁。
- Reveal 必须检测到 SOURCE 中文；若只能检测到 TARGET 日文，会抑制清除，禁止“只擦日文、不显中文”。
- 空 Reveal 补丁不再允许保存成功；GUI 会明确提示重新框选、微调或改模式。
- 复杂彩色区域增加 relaxed SOURCE 文字提取 fallback，但仍经过文字组件和 corridor 门禁。
- `full_patch` 旧工程兼容名已改为 text-only 语义：不再复制 SOURCE 背景 RGB；GUI 改为“白色气泡 · 文字迁移 + X/Y 微调”。
- 白气泡 SOURCE X/Y 微调改成注册完成后的确定性图层平移，实测 `+3/-2 px` 即得到 `+3.0/-2.0 px`。
- 白气泡文字提取增加边框内缩，气泡边线不会进入中文迁移 mask。
- 手动白气泡修正会先清理该气泡内旧的自动中文/日文文字，再写入新的微调中文字，避免双层重影。
- 过大的自动白色 contour 若文字写入占比异常，会被视为 artwork overreach，不再把人物/彩底清成白块；真实嵌套气泡由独立气泡/人工白气泡路径处理。
- 工作台/Web Review 继续按 mtime 读取 `final.png` / `final_reviewed.png` 中最新结果，旧结果不再覆盖新复核。
- Reveal / 人工补漏保存后，除了生成 `final_reviewed.png`，还会**同步覆盖 page-local `final.png`**；因此替换工作台、旧导出链、后续人工流程和任何仍读取 `final.png` 的路径都会立即看到最新人工结果。
- Reveal / 人工补漏保存后，除了同步覆盖 page-local `final.png`，还会回写 `project.json` 的 `artifacts.final` / `artifacts.final_reviewed`，并生成 `review_sync.json`，便于排查“编辑器里成功、工作台没变”的同步链问题。
- 工作台在人工补漏成功后会明确提示：**已同步 `final_reviewed.png → final.png`**，减少误判。

### 模式选择

| 区域 | 推荐模式 | 背景来源 | 可微调 |
|---|---|---|---|
| 普通白色对白气泡 | 白色气泡文字迁移 / Direct / Mask | TARGET | X/Y |
| 紫色、粉色、人物背景开放式文字 | 擦除显字 Reveal | TARGET | X/Y + Reveal 画笔 |
| 自动检测可靠的白框 | Direct / 精准蒙版 | TARGET | 自动局部配准 |

## v1.0.6 隐藏门禁清除 / 小白气泡补全 / 结果缓存失效修复

本版针对“关闭出版安全后结果仍然没变化”做了逐段真实页排查，确认 v1.0.5 仍有三处隐藏阻断/缓存问题，并全部移除。

- **出版安全彻底退出运行时**：旧配置字段 `publication_safety_enabled` 仅保留反序列化兼容，Direct/Mask/QA 不再读取它作为写入门禁；GUI 中开关变为不可用的兼容提示。
- **Auto 不再被内容完整性 invariant 反向挡回 Mask**：`content_incomplete:*` 仅保留诊断；只有写到边框、Direct 路线错误启用 OCR、重新依赖 TARGET 气泡匹配等架构级 invariant 才能阻断 Direct。
- **修复遗漏小白气泡的隐藏 0.84 白度阈值**：真实页中央圆形气泡 `white_ratio≈0.810` 已被 OCR-free completion 检出，但旧过滤要求 `>=0.84`，导致最后一步静默消失。现在使用 detector 自身阈值，密集 CJK 小气泡可以正常补全。
- **彩色浅底不再被白气泡 completion 覆盖成白块**：新增中性纸色判定。浅紫/粉色爆发框即使高亮，也不会进入 rigid white completion；继续走 TARGET-aware 彩色墨迹迁移或人工 Reveal。
- **修复“重新处理了但界面还是旧图”**：工作台与 Web Review 不再无条件优先 `final_reviewed.png`，而是比较更新时间，显示最新的 `final.png/final_reviewed.png`。
- **修复旧人工底图回滚**：如果重新处理后 `final.png` 比 `manual_effect_base.png` 更新，旧 frozen base 自动失效；人工补漏始终叠加到本轮最新自动结果。
- **真实页 45 验收**：Auto 现在直接选择 Direct；中央小圆气泡被补全；普通白底气泡保持中文；紫色开放式区域人工 Reveal 后，补漏框外改变像素为 0。

## v1.0.5 激进替换优先 / 出版安全可关闭（默认关闭）

本版把 v1.0.x 过去的“出版安全”从 Direct / Mask 的硬阻断契约改为**可选策略层**。默认 `publication_safety_enabled = false`：只要区域具备可计算的配准/几何和可用文字证据，旧版的彩色区域、跨版本大区域、IoU、coverage、spill、低置信度、内容完整性等出版级门槛不再直接阻止写入；这些指标仍写入诊断与 QA，供人工检查。

- **Direct Patch**：默认允许 TARGET-aware 彩色合成，并保留 TARGET 底色/纹理；旧 `artwork_like` / 彩色 / spiky / 大区域门禁不再作为发布阻断。仍保留最低限度的“这是可定位文字候选”有效性判断，避免把完全没有文字证据的头发、衣服、任意轮廓当成贴图。
- **Mask Transfer**：匹配置信度、目标尺寸比、局部尺度、IoU、coverage、spill、模糊源图以及内容审计在安全关闭时降为诊断/警告；实际已经完成几何写入的记录不会再因为出版 QA 被降成 REVIEW/REJECT。
- **彩色/复杂容器**：whole-raster rigid route 若会把黑白 SOURCE 背景盖到彩色 TARGET，会自动转入 TARGET-aware 墨迹迁移；这是“合成方式选择”，不是出版安全拒绝。
- **UI**：工作台新增“启用出版安全门禁（关闭 = 激进替换优先）”。默认关闭；如果需要恢复 v1.0.4 及以前的保守行为，可手动重新勾选。
- **真实黑白→彩图回归**：029 Direct 得到 13 个 SAFE 写入、QA error 0；029 Mask 得到 6 个 applied/SAFE、QA error 0。028 Mask 的大黄底效果框走 `saturated-container-ink-transfer`，保留 TARGET 黄色背景并迁移中文墨迹。

> 注意：关闭出版安全不等于“无条件把任意矩形贴上去”。完全不可用的配准、没有可定位文字证据、无法建立几何映射等基础有效性失败仍会被跳过，否则输出会变成随机破坏。

## v1.0.4 普通白气泡完整性补全

- 修复“已有 Direct/Mask 记录全部 SAFE，但页面仍有普通白气泡保留日文”的逻辑错误。
- 旧逻辑只有在已发现记录失败/需复核时才运行 OCR-free 白气泡补全；现在跨版本黑白中文→彩色日文页面无论已有记录是否全部成功，都会执行一次受安全门禁约束的未覆盖白气泡扫描。
- Direct 模式也接入同一补全链，不再因为走 fast direct 而跳过普通白气泡完整性检查。
- 新增 `_filter_uncovered_white_completion_pairs()`：只接受普通 `unseeded_white` 容器；拒绝彩色/人物候选、与已写入区域大幅重叠的大白块，并要求高白度、低暗纹密度。
- 已写入区域保持不动；补全只对真正未覆盖的白气泡追加 rigid-container raster。
- 真实页 029 回归：Direct 原 5 个已替换区域保持不变，额外找回 2 个遗漏白气泡，最终 7 个记录全部 SAFE。

# v0.9.0-alpha.4 — Exact TARGET Border Preservation / Real Pair Border Gate


## v1.0.3 安全拒绝候选导流 / 开放式效果字复核闭环

- **真正实现“白底自动、彩底人工”闭环**：Direct/精准蒙版继续只自动写入安全白底区域；彩色/复杂区域被 Direct 安全拒绝后，其诊断不会再因为 Auto 回退到 Mask 而丢失。
- Direct 诊断新增 `manual_effect_candidates` / `manual_effect_candidate_count`，记录可疑开放式/爆发文字的 SOURCE/TARGET bbox、拒绝原因和建议模式。
- **防止把人物误当文字候选**：只有“文本种子 + 彩色拒绝 + 足够紧凑文字组件”同时成立才标记 `auto_actionable=true` 并允许一键预填；普通 artwork-like 拒绝只显示统计，不自动框选。
- 工作台“人工补漏 / 开放式效果字”新增候选下拉框和“使用候选区域…”；候选会自动预填 TARGET 框并默认进入 `reveal_text`，仍可人工拖动微调。
- 若安全策略有 `rejected_artwork_like/review_candidates_skipped` 但没有高置信候选，UI 明确提示用户使用“手动框选遗漏区域…”，不会伪装成自动可修复。
- 同一区域被 contour 与 detector hint 重复发现时按 IoU/覆盖率去重，优先保留更紧、更可信的 text-seeded bbox。
- 修复 `ManualEffectDialog` 已存在但未接通的 `initial_bbox/initial_mode` 参数接口，避免点击候选时报错。
- Review Preview 无论最终采用 Direct 还是 Mask，都会保留 Direct 安全拒绝产生的人工补漏候选。
- 兼容旧页面缓存：若候选只存在 `direct_patch.diagnostics.manual_effect_candidates`，工作台仍可读取。
- 保留 v1.0.2 的非破坏人工 Overlay：补漏区域之外的自动中文结果不会回归日文。


## v1.0.0 — 真实黑白→彩图验收 / Reveal 文字笔画重建

这是 1.x 系列起点。v1.0.0 不再只依赖合成测试，已用本轮用户提供的真实“黑白中文版截图 → 彩色日文版截图”做完整验收。

- **Direct Patch 黑白→彩图门禁修复**：保留原 `0.82` 高置信度门槛，同时新增严格受约束的 cross-rendition 例外。只有 SOURCE 明显黑白、TARGET 明显彩色，并且 inlier / 重投影误差 / 空间覆盖同时达标时，才允许最低到 `0.78` 的综合置信度；不是简单降低安全阈值。
- **真实页 Direct 验收**：本轮真实页配准 confidence `0.8093`、reprojection `0.8404 px`、inlier `0.7552`、coverage `0.6689`，默认 Direct Patch 成功，自动接受 2 个白底文字容器。
- **Reveal 从“边缘膨胀”改为“结构种子→局部笔画生长”**：成对差分只负责找不同文字的种子，再在局部 black-hat / top-hat 笔画证据中受限生长，补齐文字内部，同时拒绝爆发框长刺、边框和大面积背景。
- **黑白/白字极性感知合成**：每个文字连通组件独立判断暗字、亮字或混合描边。黑字只允许压暗 TARGET，白字只允许提亮 TARGET，消除旧版紫边/白边/底色晕圈。
- **真实紫色爆发框验收**：人工 ROI `[95,965,320,1190]` 中完成日文清除 + 中文显字；相对 Direct 结果，ROI 外改变像素 `0`。
- **隐私**：真实验收图片不进入发行 ZIP，只保留数值报告与通用测试。


## v0.9.0a10 擦除显字去底 / 黑白→彩图贴图回归

- 修复“擦除显字”把旧中文版紫底/白底/低清背景一起露出来的问题。
- Reveal/Effect 合成改为 **source background delta composite**：先在 SOURCE 内估计局部背景，再只把中文文字相对背景的亮化/暗化贡献叠加到高清 TARGET 上。
- Reveal 预览与最终导出现在共享同一套 delta 合成逻辑，所见即所得。
- 新增开放式文字组件过滤，优先保留文本状组件，抑制爆发框边缘、长刺边等明显非文字残留。
- 新增黑白 SOURCE → 彩色 TARGET 的 Direct Patch 回归测试，锁定“贴图仍可用于黑白替换彩图”的路线。

## v0.9.0a9 擦除显字 / 双层人工校对

- 新增“擦除显字”人工模式：页面配准后生成中文透明文字候选层 + TARGET 日文清除层 + Reveal Mask。
- Reveal 编辑器支持左键显字/擦日文、右键恢复，实时预览始终使用高清 TARGET 背景。
- Reveal Mask 按页持久化，可反复应用/导出，不重新 OCR、不重新配准。
- 彩色 TARGET / 黑白 SOURCE 的文字差分改为“未匹配边缘结构”优先，避免仅因颜色不同把人物/背景误判成文字。
- 保留原“开放式效果字自动模式”和“整块贴图模式”，互不冲突。

自动检测之后可以在“替换工作台”直接框选任意遗漏区域，不依赖 OCR，也不要求存在对白气泡边界。默认“开放式效果字”模式会在人工框内利用已保存的 SOURCE→TARGET 配准和成对差异，把旧中文版支持的中文笔画与高清日文版支持的日文笔画分开：先清除 TARGET 日文笔画，再只合成 SOURCE 中文笔画，从而尽量保留彩色背景、网点和人物画面。白底/确定安全的区域还可切换为“整块贴图”。

人工区域记录在当前页 `review_overrides.json` 的 `manual_effect_regions` 中，可连续添加和撤销；即使自动流程因为“没有检测到可迁移气泡”而把整页原样透传，人工补漏仍可单独生成 `final_reviewed.png`，无需重新启用 OCR 或重跑整页检测。复核阶段另外输出 `manual_effect_transfer_layer.png`、`manual_effect_transfer_mask.png` 和 `manual_effect_clear_mask.png`，便于检查中文写入与日文清除范围。

## v0.9.0-alpha.4 TARGET 框线精确保护

真实页回归发现：rigid-container 虽先内缩保护 TARGET 框线，但后续 gap fill 曾以完整 target mask 为安全包络，可能重新覆盖框线抗锯齿，使旁白框顶部线条变黑/变粗。

本版将 clear / full-patch / gap-fill 统一到同一个 **border-safe write envelope**，默认保护 TARGET 容器边缘 2px；合成结束后受保护 ring 会逐像素恢复 TARGET，且从最终 write mask 中移除。每条 rigid record 记录 `target_border_preservation.changed_before_restore / changed_after_restore`，后者非 0 时强制 Review。

用户提供的黑白中文→彩色日文真实页已重新验收：左下气泡顶部细线确认是 TARGET 原始气泡轮廓，应保留；右下旁白框顶部线本来也是合法框线，但旧版会局部加深，本版恢复为 TARGET 原始像素。


## v0.9.0-alpha.2 真实页误伤门禁

- **真实成对页回归**：新增 `scripts/real_pair_acceptance.py`，可对私有 SOURCE/TARGET 跑真实 Pipeline，并用人工确认的 target bbox 检查越界写入、Direct/Mask 命名空间和 QA。输入图片不会进入发行 ZIP。
- **unseeded white artwork guard**：白衣服、窗户、路面等“白色 + 暗线”不再仅靠容器形状判定为文字；新增注册后 SOURCE/TARGET ink identity change、ink density 与 density-ratio 门禁。
- **structural supplement artwork guard**：free/complex text 补充候选必须证明 SOURCE/TARGET ink 身份真正变化；同一线稿因彩色化产生的光度差异按 artwork 拒绝。
- **彩色爆发框验证**：Mask 继续保持 ink-only / target-colour-preserving 语义；真实黄底爆发页测试中 TARGET HSV 中位数保持不变。
- 用户提供的 2048×1440 中文黑白页 → 1600×1117 日文彩页真实测试：Baseline 8 个写入记录（含 4 个 artwork 误写）降为 **4 个真实翻译记录，SAFE=4、REVIEW=0、REJECT=0**；人工确认区域外写入从 **21,849 px 降至 43 px**（羽化容差 100 px），已识别白衬衫/道路误伤 ROI 与 TARGET 像素完全一致。


## v0.8.35 本轮重点

- **Dual SOURCE Direct 真正执行**：`dual_source.secondary_source_dir` 会对候选高清汉化 SOURCE 独立执行同页预检、配准、SOURCE 容器检测、Direct 计划与 invariant QA。主 SOURCE 仍是译文权威；次 SOURCE 只有在 Direct 计划可用时才成为像素源。
- **Matching Evidence**：Hungarian 成本增加文字长度与配准置信度惩罚，并保留 overlap / projected IoU soft prior；`MatchResult.diagnostics` 输出未匹配 Top-3、超 `max_cost` 拒配、歧义和 `force_actions`。
- **Planner 可操作证据**：`TransferDecision` 输出 `evidence` 和 `force_actions`，Direct 失败可明确提示二次 SOURCE、强制 Mask、跳过等动作。
- **白底 Fast Dark Clear**：白气泡优先只清目标暗字，局部纸色回填后仅对暗字 mask 做小半径 Telea；非白底/纹理区域自动回原组件/Mask 路线。
- **Review Force Actions**：Review Web UI 新增“强制 Direct / 强制 Mask”，单元级增加“强制匹配 / 跳过”；页面强制动作会真实重跑当前页，而不是只写 JSON。
- **Replace Translation v1 对齐**：页级 meta / `replace_translation/summary.json` 使用 `manga-hd-transfer/replace_translation/v1`，标记 `compatible_with: manga-translator-ui/replace_translation`，并导出 `translated_text / bbox / overlap / matched / matching_diagnostics`。


## v0.8.34.4 效果优先增强

- **Region 亚像素微调**：在原有局部 ECC 后追加受限 0.5px 级 soft-mask 搜索，只改平移，不拉伸/扭曲中文字形。
- **Pixel Enhance**：低清 SOURCE 在进入二值 Ink Reconstruction/OCR 前，先尝试保留原字形与排版的抗锯齿锐化增强；白底与非文字区域保持不变。
- **Auto 像素结果优先**：Mask 已得到 SAFE/REVIEW 的可逆像素结果后，OCR 仅作证据，不再覆盖成重新排字；只有 REJECT 或未覆盖区域可进入重 fallback。
- **Structure Map v2**：融合 Sobel、Canny、分镜/长直线和低频结构，提高黑白中文版到彩色/高清日文版的配准稳定性。
- 保留 v0.8.34.3 的内容完整性自动修复、日文残留二次清除、渐进式 mask 扩张和 SAFE/REVIEW/REJECT 三态。


## v0.8.34.3 本轮重点

- **内容完整性从“只报警”升级为自动修复闭环**：Mask 写入后如果 SOURCE ink coverage 不足或 TARGET residual 偏高，会在同一安全气泡包络内自动尝试一次受控补救，再重新审计。
- **中文缺笔画自动补偿**：写入 mask 可在安全 envelope 内渐进扩张，优先补回因 1–5px 几何差异/内缩造成的 SOURCE 边缘笔画缺失。
- **日文残片二次清除**：只针对已被 compact-component 识别为 TARGET-only 的文字墨迹做局部清除，不拿气泡边框/分镜线当成日文。
- **修复失败会回滚**：如果二次修复没有带来足够 coverage/residual 改善，不采用更差结果。
- **每个区域统一 SAFE / REVIEW / REJECT**：SAFE 必须是已写入、内容已独立验证且达到置信门槛；REVIEW 保留可逆候选；REJECT 不作为可直接发布结果。
- `transfer_audit.transfer` 新增 `auto_repair_attempted / auto_repair_succeeded / triage_safe / triage_review / triage_reject`，方便 Review 与批量质量统计。

- 新增独立 **`direct_patch`（直接贴图）**：面向完全一致或可证明为同版同布局的页面，直接从 SOURCE 扣取完整气泡/白底文本框内部的**原始栅格（白色背景 + 中文字形一起）**，经 identity / local similarity 对齐后硬覆盖到 TARGET。它不是“蒙版模式的一个参数”。
- **Direct Patch 严格不做 OCR、不清字、不 inpaint、不重新排字，也不偷偷回退到 Mask。** 同页预检、配准或容器安全性不满足时，显式 Direct 模式保留 TARGET 原图并标记 Review；`auto` 才允许继续走 Mask。
- `mask_replace` 重新限定为真正的 **蒙版迁移**：geometry mask / transfer(clear) mask 分离，按安全边界清除 TARGET 日文并合成 SOURCE 内容；彩色、网点、渐变等需要保留 TARGET 纹理的容器只能走 Mask/Auto fallback，不再冒充 Direct。
- 新增 `page_pairing.py` OCR-free 同页预检，使用低频结构、边缘重合和配准质量联合判断，降低相邻页错贴风险。
- 新增 `transfer_planner.py`：`auto → direct_patch（安全时）→ mask_replace`；显式模式保持严格语义。
- Direct 输出新增 `direct_patch_layer.png`、`direct_patch_regions.png`、`direct_patch.json`，同时保留旧 review 兼容产物。
- GUI 新增“自动 · Direct 优先 / 直接贴图 · 整气泡/文本框 / 精准蒙版迁移 / 智能混合 / 高清重排”，并把 Direct 与 Mask 的参数说明拆开。
- 新增 Direct/Mask 独立配置命名空间与回归测试，专门验证 Direct 不会重新混回 Mask/OCR。

# v0.8.32 — Source Direct Provider Integration

- 新增统一 ProviderRegistry 与 SOURCE-only provider chain：pseudo barrier、sidecar/CTD、Comic Translate RT-DETR-v2、SAM2、MangaLens、Debubble-style white overlay。
- 新增 `coordinate_space.py`：所有检测结果固定在 SOURCE 原始像素坐标，affine/homography 只做定位，最终中文 raster 始终 local similarity。
- 新增 Cotrans-inspired coloured clear-mask connected-component refiner；PanelCleaner-inspired progressive border fitter 已正式注册为 mask-refiner provider。
- Manga-Overlay-Translator 的原图坐标/缩放映射与缓存思路已适配；DebubbleBot 的独立 editable mask overlay 思路已适配。
- RT-DETR-v2 / SAM2 真正接入但默认不下载模型；成功 source-direct 路径仍为 0 OCR、0 TARGET bubble matching、0 边框写入。
- cheap SOURCE hints 先跑、direct plan 只构建一次，避免上一开发版重复整页构建。
- 详见 `INTEGRATION_STATUS_v0.8.32.md` 与 `TEST_REPORT_v0.8.32.md`。

# Manga HD Translation Transfer



## v0.8.31 — 结构配准 + 渐进安全蒙版 + 彩色框保色 + source-only 兜底

- **仍以 source-direct 为主流程。** 中文版源页是唯一文字/容器内容来源；日文高清页不再承担“重新检测气泡再配对”的职责，目标页轮廓仅用于坐标配准与 QA。
- 页面自动配准升级为 **quick SIFT/ORB + RANSAC → 结构图 ECC 残差精修**。ECC 只在约 900px 长边缩略结构图上运行，并受相关性提升、位移/旋转上限和特征重投影门槛约束；affine/homography 仍只负责找位置，最终中文字始终 local similarity。
- 吸收 PanelCleaner 的“渐进式 mask 拟合 / 低置信宁可跳过”思路，独立实现 **动态边框厚度估计 + progressive inset + safe/review/rejected 三态**。几何容器与最终 transfer mask 分离，边框永远只用于辅助对齐。
- 渐进蒙版计算改为 **候选 ROI 局部 morphology / distance transform**，不再每个轮廓都对 2K/4K 整页做距离变换；文字组件统计同样只在候选 ROI 运行。
- 吸收 manga-image-translator 的 mask dilation 思路：彩色框中的目标日文清除 mask 可受控扩张，减少残字；但不进入 OCR / 重排。
- **彩色尖角框不再把目标黄色/红色填充刷成白色。** 目标填充和高清边线保留，只在安全内部清除目标日文字形，再迁移已配准的中文墨迹 alpha。普通白气泡仍使用完整白底 + 中文整块硬覆盖，保证无目标日文残留。
- 增加 Comic Translate 风格的 **source-only detector fallback** 接口：默认关闭，仅当普通 source-direct 无法安全完成、用户明确配置本地 MangaLens/Ultralytics 权重时才调用；模型只补源页容器提示，绝不检测目标气泡或恢复双边匹配。
- 第三方参考源码按原许可证放入 `third_party_reference/`，与 MIT 运行时代码隔离；核心 `src/` 不导入 GPL 参考模块。
- 实页 `014.jpg` → `p-014.jpeg` 用于 v0.8.30 / v0.8.31 A/B：继续命中 6 个真实容器（3 白 + 3 彩色尖角），OCR=0、目标气泡匹配=0、辅助边框写入=0；v0.8.31 同时保持彩色填充并减少局部候选计算成本。

## v0.8.30 — 不同尺寸自动配准 / 仿射只定位 / 字形始终等比

- **不同像素尺寸、不同裁边、轻微旋转/扫描比例差现在自动处理。** 页面级配准可以使用 similarity / affine / homography 来确定“中文版坐标 → 日文版坐标”，不再要求两张图尺寸完全相同。
- 新增 **A0/A1/A2/A3 自动对齐模式**：同尺寸 1:1、全局等比、affine 位置映射 + 局部等比栅格、homography 位置映射 + 局部等比栅格。程序自动选择，不需要手工指定。
- **页面 affine/homography 永远不直接作用到中文字形。** 每个中文版气泡/文本框在对应位置只使用 local similarity（单一 scale + rotation + translation）渲染，因此 X/Y 拉伸、shear、透视都不会把中文压扁或扭曲。
- 每个容器增加局部边线微调：在目标页黑色/彩色框轮廓附近自动微调少量等比 scale、rotation、X/Y，边框仍然只用于对齐，最终 `border_pixels_written=0`。
- 注册阶段增加 **quick OpenCV** 预检：同版页面优先在约 1000px 长边进行 SIFT/ORB 配准，只有置信度/覆盖不足才升级到原来的高分辨率配准或深度后端，减少不同分辨率页面的等待。
- 使用 `014.jpg`（1440×2048）+ `p-014.jpeg`（1117×1600）回归：自动识别为 `A2_affine_location_local_similarity_raster`，普通白框和彩色尖角框继续直接从中文版整体覆盖，OCR/目标气泡双边匹配均未使用。

## v0.8.29 — 同版整气泡直接覆盖 / 彩色尖角框 / 边框仅对齐

- 新增 **source-direct whole-container 快速路线**：不再先检测“中文版气泡 → 日文版气泡”再做配对。直接从中文版提取完整对白框内部，按页面对应位置覆盖到日文高清页。
- **同尺寸/同坐标页面直接 1:1 拷贝**；分辨率不同的同版页面只使用一个全局等比缩放 + 小范围平移对齐，禁止 X/Y 非等比缩放、shear 和 OCR 重排。
- 气泡/尖角框的黑色边线和任何检测矩形都只作为定位证据，**最终不写入辅助边框**；最终写入区域先向内收缩，保留高清日文母版自己的轮廓。
- 历史版本曾加入 **彩色尖角拟声/对白框** 的 source-direct 实验路线；**v0.8.34.1 起此行为已收紧**：需要保留 TARGET 彩色/纹理的区域归入 Mask/Auto fallback，显式 Direct 不再使用这条彩色覆盖语义。
- 加入 artwork guard：人物皮肤、头发、建筑等即使是彩色区域，也必须同时通过源白底、紧凑文字、边界对齐和中日墨迹变化门槛，避免把画面误当彩色对白框。
- 高置信同版页命中快速路线后直接跳过 paired-diff、OCR、气泡双边匹配和重复 completion，减少整页处理时间。
- 使用用户提供的 `014.jpg` + `p-014.jpeg` 实测：自动选中 6 个真实对白/尖角框（3 个普通白框 + 3 个彩色尖角框），未把人物/建筑误覆盖。

## v0.8.28 — 整气泡覆盖强化 / 白色爆发框 / 页面级性能优化

- **整气泡 / 白底文本框覆盖现在在 GUI 中有明确开关，默认开启。** 白色容器优先复制旧中文版的完整白底 + 中文字补丁，不拆字、不 OCR 重排。
- 新增 **白色爆发框 / 锯齿对白框** 安全路线：允许低 bbox 填充率的星爆/锯齿白框进入整容器覆盖，但仍要求中日两页都满足白度、文字墨迹、饱和度、长宽比和配对一致性。
- 修复实心容器补洞对锯齿爆发框的潜在破坏：低 solidity 的星爆形状不再使用大核 closing / convex-envelope 补洞，只填真正的内部文字孔洞，保留每个尖角和凹槽。
- 页面级性能优化：source/target 的 Gray/HSV 转换在整页只计算一次，气泡越多收益越明显；图层写入也不再为每个气泡整页执行一次 BGR→RGB 转换。
- 真实 006 / 065 / 066 / 007 再次回归，QA 均为 0 error；006 小对白继续走完整 source container patch。

## v0.8.27 — 整个中文版气泡 / 文本框覆盖

- 白色气泡、白底旁白框和白底文本框默认使用 `rigid-container-full-patch`：直接迁移旧中文版的白底 + 中文字整体补丁。
- 目标容器仅提供位置与边界；中文字不使用 X/Y 非等比缩放，不通过 OCR 重排。
- 目标日文文字被完整白底补丁覆盖，避免“中文字已经写入但日文仍残留”的混合结果。
- 原 `rigid-container-raster` 仍作为兼容回退路径。

## v0.8.26 — 实心气泡内部 / 日文残留修复

- 修复白色气泡 detector mask 被日文字切出黑缺口后，清除阶段反而跳过日文字的根因。
- 检测 mask 与真正的容器内部 mask 分离：闭合窄缺口、填内部孔洞，并保留边界保护带。
- `clear mask` 与中文字栅格裁切都使用实心容器内部，避免日文残留与中文缺笔。
- 065 的爆发式黑色气泡边缘通过 convex-envelope boundary guard 保留，不会因补洞被刷白。
- OCR-free 白色容器补漏提高文字密度门槛，减少建筑/天空白块误判。
- 真实 006 小对白已重新验证：原日文清空，中文完整，QA pass。

## v0.8.25 — 白色容器补漏与独立清除蒙版

- 同版黑白中文 → 彩色日文页新增 OCR-free 白色容器补漏。目标气泡/文本框作为几何真值，反向映射到原始中文页取完整中文字形，避免源扫描气泡连到纸面或被翻译组扩框时被错误拒绝。
- 中文最终字形继续严格 `uniform scale + translation`，不继承页面 affine/dense-flow 的非等比变形。
- 工作台新增 `清除蒙版 / 中文迁移层 / 只清日文` 三个独立视图，以及可画笔增删的清除蒙版 overlay 编辑器。
- `仅执行去字` 不重跑配准、OCR、检测或中文字形迁移。
- 可独立选择气泡 detector、检测分辨率、清除 mask 扩张和 inpainting backend。
- 006 / 065 / 066 用户真实页均重新回归；006 右下角此前漏掉的对白已由自动路径恢复。


## v0.8.23 锁定栅格气泡迁移：位置可以仿射，字形绝不仿射

本版重新定义黑白旧中文版 → 彩色/高清日文页的最终写入规则：**页面配准负责找位置，中文字栅格只允许单一等比缩放 + 平移**。全局 affine / homography、局部 dense-flow 都不能再直接作用到最终 CJK 字形。

- 对白色对白气泡、旁白框、白底文本框，目标高清页的完整容器是清除边界：保留目标轮廓/尾巴，清空内部日文，再从原始旧中文版取得整块中文字栅格。
- 字符和标点不拆成 connected components 重组；整块灰度/抗锯齿栅格作为一个连续 alpha 场迁移，避免缺笔、断标点和字符重心漂移。
- 分辨率不同只允许一个 scalar scale；禁止 X/Y 独立 resize、shear 和把页面 affine 的非等比校正传给字形。
- 结构候选只用于发现区域；若目标 mask 恰好切到日文字边缘，只清除字符尺寸的残余组件，不进行可能泄漏到画面的白区 flood-fill。
- 黄色/彩色爆炸框保留目标彩色背景，只迁移中文墨迹，不复制黑白版的爆炸框底色/线稿。
- 复杂/开放文字在真正写入最终页之前先做内容完整性检查；检查失败就不发布该候选，避免出现“写入成功但字残缺”的半成品。
- 065 / 066 两组真实图在 OCR=none 下重新回归：065 为 5/5 内容完整，066 为 4/4 内容完整；已检测区域目标日文残留率为 0。


## v0.8.22 真实 065/066 验证：内容完整性与跨版本复杂文字

本版针对黑白旧中文版 → 彩色/高清日文原页的真实 065、066 配对做了回归，不再把 `applied=true` 当成“翻译完整”。主要变化：

- **几何写入与内容完整性分离**：每条替换记录新增 `content_check`、`source_ink_coverage`、`target_residual_ratio`、`content_complete`。跨版本白气泡、开放文字、复杂文字、OCR 几何兜底若内容未验证/不完整，会由 QA 阻止作为成功页。
- **跨版本白气泡候选守门**：黑白→彩色时用高白度、填充率、内部文字墨迹和边界证据收紧气泡，防止列车车身、招牌、浅色画面块被当成对白框。
- **彩色爆炸框独立路线**：高饱和黄色/红色文字框必须与结构候选有足够真实重叠，并要求注册后的中文源区域主要是亮底，避免邻近蓝色 SFX/招牌误触发。只擦目标文字笔画，只写源中文笔画，不复制黑白底纹。
- **结构局部流只负责检测**：结构补充仍可用局部/稠密流发现变化，但最终中文字统一优先取全局注册源，避免局部光流扭曲 CJK 字形、把爆炸框边线拖入最终图。
- **`transfer_audit.v2`**：同时记录 `geometry_applied`、`content_checked`、`content_complete`、`content_incomplete`、`content_unverified`、最差源墨迹覆盖率和最大目标残留率。无 OCR 时明确标记验证范围为 `detected_regions_only_no_ocr`，不假装做了整页语言识别。
- **真实样本**：065 共 5 个区域（3 白气泡 + 2 黄色爆炸框）全部内容验证通过；066 共 4 个区域（列车小气泡、右侧气泡、中间气泡、左下大气泡）全部内容验证通过，未再修改右侧白色招牌/画面块。

## v0.8.21 复杂文字 / 漏检恢复 / 逐页迁移审计

- 精准蒙版增加 **复杂文字区域** 路线：黄色爆发框、开放式文字、彩色背景文字不再被“近白底”门槛直接丢弃；必须同时满足源/目标两侧紧凑文字墨迹、结构差异和面积上限，最终仍只搬运源中文栅格墨迹，不做 OCR 重排。
- 成对差异的闭合气泡检测之外增加结构补充：raw-diff 没有建立容器时可回退到 structural-v08；raw-diff 已找到普通气泡时也可补充不重叠的 free/complex text。低置信度文本不再静默丢弃，而是生成可恢复、可编辑的复核候选。
- 新增 **OCR 几何引导组件迁移**：OCR 只用于确认“源中文区域 ↔ 目标日文区域”的对应和候选范围；真正清除的是目标日文字形组件，真正写回的是配准后的源中文像素。`strict_mask_replace_no_ocr_reletter=true` 时绝不拿 OCR 字符串重打字。
- 目标清除蒙版允许窄长竖排日文列和横向标题组件跨出不完整气泡内框；仅对经过组件形状约束的目标文字先行清除，避免残留半截日文，同时保护气泡轮廓、格线和彩色背景。
- 每页新增 `transfer_audit.json`、`source_original.png`、`target_clear_mask.png`、`chinese_transfer_layer.png`，并保证 `review_preview.png` 总是存在；可直接核对配准、候选类型、OCR 证据、应用/拒绝/复核数量、清除像素和 QA。
- 修复 raw-diff 中重复执行一次源白区组件搜索的隐藏性能问题；精准蒙版 0.8.16 的 glyph-footprint rescue 保持不变。
- 页面管理/跳过逻辑保持 v0.8.20：所有新页默认正文，不做配对后自动气泡扫描；无中文 OCR 文本的正文可原样输出高清日文页；手动跳过只跳过替换，不从最终整册删页。

## v0.8.20 默认正文 / 按需替换 / 缩略图性能修复

- **删除配对后的自动气泡/文本框检查**：新配对页面一律默认为“正文 / 需替换”，页面类型只由用户手动分类；v0.8.19 保存的 `auto_no_text` 自动标记会迁移回默认正文。
- “正文”现在表示**允许进入迁移**，不再表示“必须改图”。实际处理时若中文源页 OCR + 气泡结构确认没有可迁移的 speech/narration 文本框，会直接输出未修改的高清日文页。精准蒙版模式仍禁止 OCR 改写原中文字形。
- 手动标记为封面、扉页、目录、单话首页、插图、空白页或“跳过”的页面仍是完整输出页：只跳过替换步骤，不从 `final/` 和整册页序删除。
- 缩略图改为**可视区域懒加载 + 滚动静默期加载 + 128 张缩放图 LRU 缓存**。滚轮/触控板滚动时不再后台持续解码整本漫画，停止滚动后才分小批加载当前视口及相邻一行。
- 切换页面类型不再重新解码已经缓存的缩略图；“刷新”才主动清空缩略图缓存。
- 整册最终输出文件名增加同名保护，避免极端情况下两个目标页同 stem 时互相覆盖。

## v0.8.19 可视化页面管理

- 页面管理默认改为 **缩略图画廊**：每一页直接显示高清日文页面缩略图、页码、页面类型、手动/自动来源和类型色标；可切换为旧版中文缩略图，不再靠文件名猜封面/目录/插图。
- **双击任意缩略图或列表行**打开可缩放大图窗口，旧版中文与高清日文左右并排；支持上一页/下一页、滚轮缩放、拖拽、适合窗口和 100% 查看。
- 缩略图支持 Ctrl / Command / Shift 多选，右键可直接批量标记正文、封面、扉页、目录、单话首页、纯图片等，也可恢复自动判断或只检查所选页。
- 新增“日文缩略图 / 中文缩略图”切换、页面筛选和当前选择计数；右侧详情同步显示当前页大预览、分类原因、气泡/自由文字数量、配对方法和置信度。
- 大图窗口的分类状态刷新不再重复解码两张全分辨率图片，避免批量处理/自动检查时因 UI 状态更新造成卡顿。
- `page_management.json` 改用同目录临时文件 + `fsync` + 原子替换保存；强制退出或写盘异常时不会把已有页面分类文件写成半截 JSON。

## v0.8.18 页面管理 / 安全停止

- 顶部加入始终可见的红色 **“■ 停止”** 按钮，页面自动检查、单页处理和整册处理共用同一套安全取消信号；单页 Pipeline 也在配准、成对差异、OCR/气泡、迁移和导出等安全边界检查取消。
- “页面”重构为 **页面管理**：每个已配对页面都有独立处理准入状态，可批量标记为正文、封面、扉页/书名页、目录、单话首页/章节页、插图/纯图片、卷首插画、空白页、后记/版权/广告或手动跳过。
- 配对后默认运行 OCR-free 页面检查：只使用下采样 OpenCV 配准 + paired-diff 几何，不读取文字内容；高置信度且没有气泡/文本框的页面自动标为“无气泡/文本框（自动）”并跳过迁移。检测失败或配准不足时**绝不自动丢页**，保持进入正常处理。
- 跳过页仍保留在整册页序中，并把高清日文原页原样写入 `final/`；不会进入 OCR、气泡迁移、清字或重排。手动“正文 / 需替换”可强制恢复处理。
- 页面标记持久化到输出目录 `page_management.json`，重新打开项目后继续生效；断点续跑也服从最新页面标记，避免把旧的已处理结果错误恢复到后来标记为封面/插图的页面。
- 页面类型使用不同颜色，表格同时显示“处理/跳过、页面类型、配对方法、状态和路线”，并支持多选批量标记、恢复自动、只检查所选页。

## v0.8.17 工作台同步与 UI 修复

- **替换工作台统一页状态**：日文原图、旧中文版、最终结果、复核标注和替换蒙版都从同一个当前页解析，不再复用“上一张最后结果”。
- 工作台新增 **上一页 / 下一页 / 当前页计数**，最终结果可连续翻页；当前页没有结果时明确显示为空，不会误显示其它页面。
- 整册处理完成后，每页工程会进入按页索引；同时可直接从 `pages/<page_id>/project.json` 和 `final/` 恢复预览，断点续跑页也能正常浏览。
- 复核后的 `final_reviewed.png` 优先显示，并同步回整册 `final/` 对应页，避免预览结果和出版输出不一致。
- “优先名称 / 页码配对”和“优先文件夹自然顺序”默认关闭，并增加明显间距与说明；默认走智能视觉配对。
- 修复 CJK 文件名页 ID：日文/中文纯文字文件名不再全部坍缩成同一个 `page` 目录；同时兼容 v0.8.16 单页处理留下的旧目录。
- UI 继续参考 KCC-Kindle-CHS 的卡片化、双栏、右侧滚动设置区和紧凑 macOS 控件设计，但不改动迁移核心信号与处理流程。

## v0.8.11 排版质量策略

默认采用“**OCR 认字，源图定排版**”：Apple Live Text 成功并不意味着一定重新排字。旧中文版迁移后的字形已经清晰时，程序保留原字号、分列和相对位置；只有模糊/低清/不安全区域才交给 OCR 高清重排。这样避免真实 Mac 批处理里出现巨大字体、分列错乱、白块和空白气泡。

把**旧版低清中文汉化漫画**中的既有译文可靠迁移到**高清日文原图**，完成跨版本页面配准、中文 OCR、日文清字、气泡安全区排版、自动 QA、人工复核与分层导出。

项目不是机器翻译器。它不重新翻译日文；中文内容来自你已有的旧汉化版。

## 目标

输入：

- `source_cn/`：旧版中文汉化图，允许低清、压缩、扫描偏移、裁边、额外 staff 页。
- `target_jp/`：同一作品的高清日文图，作为唯一画面母版。

输出：

- `final/`：自动通过安全门槛的最终 PNG。
- `pages/.../target_original.png`：高清母版，永不覆盖。
- `inpainted.png` / `clear_mask.png`：清字结果与精确 mask。
- `text_layer.png`：透明中文文字层。
- `editable.ora`：OpenRaster 分层工程。
- `editable.psd`：系统有 ImageMagick 时自动生成分层 PSD。
- `project.json`：配准矩阵、OCR、气泡、匹配、排字、置信度和可追溯信息。
- `qa.json` / debug overlays：出版 QA 和视觉证据。
- 本地 Review 编辑器：可改译文、改目标匹配、画/擦清字 mask，再重新生成。

## 核心设计

```text
旧中文版 ─ 页面指纹/顺序配对 ─┐
                            ├─ cheap-first / SIFT / LightGlue / LoFTR + RANSAC 配准
高清日文版 ─ 页面指纹/顺序配对 ┘

旧中文版 ─ 中文 OCR / 多次低置信度复识 ─ 中文 TextUnit
高清日文版 ─ 日文文字区域 + 气泡实例 / safe mask ─ TargetUnit

TextUnit + Registration + TargetUnit
              ↓
      跨版本最小成本身份匹配
              ↓
      日文像素级 text mask 清字
              ↓
  solid / OpenCV / external LaMa 修复
              ↓
  中文约束排字（字号、断行、禁则、安全区）
              ↓
       QA 安全门槛 + Review Queue
              ↓
       PNG / ORA / PSD / JSON
```

## 五种译文迁移模式

### 1. 自动 `auto`（默认）

先做 OCR-free 同页预检和页面配准。若页面属于同版同布局、容器可安全直接迁移，则走 `direct_patch`；Direct 不满足条件时才进入 `mask_replace`。这条路线用于批量处理时自动“先便宜、后复杂”，避免所有模型全跑。

### 2. 直接贴图 `direct_patch`

这是和蒙版迁移**语义完全不同**的一条路线，专门针对完全一致或高度一致的两张图。SOURCE 的完整气泡/白底文本框内部被当作一个原始栅格 patch，**白色背景与中文文字一起扣出、一起对齐、一起覆盖**。

- identity 页面优先 1:1 覆盖；尺寸/裁边略有变化时只允许 local similarity（统一缩放 + 小旋转 + 平移）。
- 页面 affine/homography 只负责“找到 TARGET 位置”，不会把中文字形做 X/Y 拉伸或透视变形。
- 不调用 OCR，不重新输入中文，不清 TARGET 日文，不 inpaint，不走 target-aware 背景重建。
- 显式 `direct_patch` 只要同页预检、配准或容器完整性不够安全，就拒绝该页/区域并保留 TARGET；**绝不静默切换到蒙版或 OCR**。
- 彩色/网点/渐变容器如果需要保留 TARGET 原纹理，应选择 `auto` 或 `mask_replace`，而不是 Direct。

### 3. 精准蒙版迁移 `mask_replace`

蒙版模式不是“整块贴图”的别名。它以 `geometry_mask` 表示容器几何，以独立的 transfer/clear mask 表示真正允许修改 TARGET 的区域：先保护边框和画面，再在安全区域内清除 TARGET 日文、迁移 SOURCE 中文 ink / interior / artwork。

对于白色普通气泡可以迁移 SOURCE interior；对于彩色、网点、渐变容器则优先保留 TARGET background/texture，只清日文墨迹并迁移中文。局部几何、内容完整度、spill、边框保护等 QA 不满足时进入 Review。

### 4. 智能混合 `hybrid`

优先使用蒙版迁移保留旧汉化排字；某个区域无法安全完成时，允许退回 OCR → 清字 → 高清重排。适合确实需要传统 fallback 的跨版本页面。

### 5. 高清重排 `reletter`

旧中文版提供中文文本内容，高清日文版负责清字与重新排版。适用于 SOURCE 字形本身太糊、需要统一出版级字体的情况。

CLI：

```bash
mhd-transfer run source_cn target_jp output --mode auto
mhd-transfer run source_cn target_jp output --mode direct_patch
mhd-transfer run source_cn target_jp output --mode mask_replace
mhd-transfer run source_cn target_jp output --mode hybrid
mhd-transfer run source_cn target_jp output --mode reletter
```

Direct / Mask 的边界和回归验收见 `docs/DIRECT_VS_MASK_V0834.md`；蒙版算法细节见 `docs/MASK_REPLACE_PLAN.md`。

### 与简单“替换翻译”方案的关键区别

1. **先做视觉配准，再匹配文字身份**，不依赖整页 resize 后固定 IoU。
2. **Direct Patch** 迁移 SOURCE 的整块原始栅格；**蒙版迁移**在独立 clear/transfer mask 内做 TARGET-aware 合成；**高清重排**才迁移文本内容并重新渲染，三种语义严格分离。
3. **清字 mask 与排字 safe area 分离**：前者只描述要删除的日文，后者描述中文允许出现的位置。
4. 气泡边界有保护带；中文最终字形 mask 必须通过 safe-area 覆盖验证。
5. 低页面配对、低配准、低 OCR、低身份匹配、拆分/合并关系都会阻止自动覆盖，进入 Review。
6. 每页有完整 evidence/debug，不做不可追溯的黑箱整页重绘。

## 安装

Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

核心安装只需要 OpenCV / NumPy / Pillow / SciPy / Pydantic / Typer，不下载 OCR 或深度模型。

### 中文 / 日文 OCR

```bash
pip install -e '.[ocr]'
```

默认显式使用 `PP-OCRv5`。PaddleOCR 的模型可能在首次实际推理时下载。若当前平台不方便运行 Paddle，也可使用 `sidecar`，把任何外部 OCR/漫画文字检测器的结果接进来。

### LightGlue / LoFTR 增强配准

```bash
pip install -e '.[lightglue]'
```

`registration.backend=auto` 采用 cheap-first：同源快速结构验证 → OpenCV SIFT/ORB → 本地已有权重时 LightGlue → LoFTR。Auto 默认不允许隐藏下载权重。Apple Silicon 可用 `首次安装_MPS_AI加速.command` 安装 MPS 推理运行库；模型权重仍由用户显式准备。

### LaMa

项目不复制/绑定 LaMa 模型。把任意可用 LaMa wrapper 配成：

```json
{
  "inpainting": {
    "backend": "lama",
    "lama_command": "python lama_wrapper.py --input {input} --mask {mask} --output {output}"
  }
}
```

## 首次运行

```bash
mhd-transfer doctor
mhd-transfer init-config config.json
mhd-transfer run source_cn target_jp output --config config.json
```

如果使用外部 OCR sidecar：

```bash
mhd-transfer run source_cn target_jp output --ocr-backend sidecar
```

运行结束若存在出版阻断级 QA，CLI 返回非 0，并提示进入 Review：

```bash
mhd-transfer review output
```

浏览器打开三栏编辑器：旧中文版 / 高清日文与清字 mask / 当前输出。可以：

- 修改 OCR 中文译文；
- 改某个中文 TextUnit 对应的目标气泡/文本框；
- 勾选是否应用该文本；
- 画笔增加或擦除清字 mask；
- 保存并点击“应用复核并重新生成”。

也可以命令行应用：

```bash
mhd-transfer apply-review output/pages/0001_xxx
```

## v0.6 批量与 Apple MPS

- 整册处理支持断点续跑、失败页继续、实时进度和安全取消；
- 配准/OCR/气泡实例按输入+配置做本地阶段缓存；
- LightGlue、LoFTR、MangaLens 和 Torch 超分模型整册常驻，不再每页重复初始化；
- Apple Silicon 上可选择 `mps`，MPS GPU 推理受控串行，CPU 图像预处理继续使用受限线程；
- 同源页面先跑约缩略图级的结构/phase 快速验证，只有不确定页才支付 SIFT/深度模型成本；
- 打开 GUI 的模型页不会因此 import 大型模型或下载权重。

详细见 `docs/PERFORMANCE_V06.md`。

## Sidecar 接口

### OCR / 漫画文字分割

`page.png` 对应 `page.ocr.json`：

```json
{
  "blocks": [
    {
      "id": "b0",
      "text": "已经存在的中文译文",
      "confidence": 0.99,
      "polygon": [[100,100],[260,100],[260,180],[100,180]],
      "kind": "speech",
      "mask_path": "masks/b0.png"
    }
  ]
}
```

如果提供 `mask_path`，清字优先使用像素级 segmentation；没有时才回退到 polygon。

### 气泡 / 旁白框实例分割

`page.bubbles.json`：

```json
{
  "bubbles": [
    {
      "id": "bubble-0",
      "kind": "speech",
      "confidence": 0.99,
      "polygon": [[80,70],[300,70],[300,220],[80,220]],
      "mask_path": "bubbles/0.png",
      "safe_mask_path": "bubbles/0_safe.png"
    }
  ]
}
```

没有 `safe_mask_path` 时会自动向内腐蚀实例 mask，窄尾部通常会自然脱离排版安全区。

## 自验收

完全离线、无模型下载：

```bash
pytest
mhd-transfer selftest
python benchmarks/synthetic_acceptance.py
```

本次实现时结果：

- v0.8.1 合并版单元/集成测试：**25/25 通过**，新增 cheap-first 配准、断点续跑与阶段缓存测试。
- 内置端到端 selftest：**通过**，无 error/warning。
- 20 组随机几何扰动配准：**20/20 通过**。
- 控制点中位误差：约 **0.051 px**；P95：约 **0.117 px**。
- 4 组不同长度中文气泡排字：**4/4 通过**，字形安全区覆盖均 ≥ 0.997。
- 本机层导出验证：ORA 成功；ImageMagick 存在时 PSD 成功。

详见 `docs/SELF_ACCEPTANCE.md`。

> 合成验收不能替代真实出版数据验收。最终“出版级”门槛仍要求建立同作品的 100–300 页人工真值集，统计页面配对、区域身份匹配、残留日文、误伤线稿、排字越界和人工复核率。程序已把这些证据与 QA 接口保留下来。

## 目录

```text
src/manga_hd_transfer/
  pairing.py          页面指纹与顺序配对
  registration.py     SIFT / LightGlue / LoFTR + RANSAC
  ocr.py              Paddle、sidecar、低置信度复识
  bubbles.py          气泡实例 / safe area / TextUnit
  matching.py         跨版本身份匹配、拆分/合并检测
  masking.py          像素级/多边形日文清字 mask
  mask_transfer.py     旧中文气泡/文本框 patch 对齐、超采样与蒙版替换
  inpainting.py       solid / OpenCV / external LaMa
  lettering.py        中文约束排字
  qa.py               出版 QA
  pipeline.py         全流水线
  review.py           本地人工复核编辑器
  review_apply.py     Review 回写与重生成
  export.py           PNG / ORA / PSD layer export
```

## 参考与许可

v0.8.32 已将部分经许可的关键上游源码/摘录放入 `third_party_reference/` 作隔离参考；运行时适配状态见 `INTEGRATION_STATUS_v0.8.32.md`，GPL 参考文件不从 MIT 核心导入。

- `hgmzhn/manga-translator-ui`：Replace Translation 业务原型。
- `zyddnys/manga-image-translator`：漫画翻译流水线。
- `dmMaze/BallonsTranslator`：编辑器、mask/inpaint/排字工作流。
- `dmMaze/comic-text-detector`：漫画文本框、文本行、segmentation；当前通过 pixel-mask sidecar 接入。
- `ogkalu2/comic-translate`：气泡检测与文本分割分层、PPOCR/LaMa 模块化。
- `PaddlePaddle/PaddleOCR`：PP-OCRv5 直接后端。
- `cvg/LightGlue`：局部特征匹配直接可选后端。
- `zju3dv/LoFTR`：困难图像匹配直接可选后备。
- `advimman/lama`：复杂背景 external command 后端。
- `facebookresearch/sam2`：v0.8.32 已接入 SOURCE-only 可选分割 provider；默认不下载模型。
- `huyvux3005/manga109-segmentation-bubble`（MangaLens）：本版新增 Ultralytics 本地模型直接后端。
- `ScanR/TypeR`：自动居中、行距联动、样式预设与紧凑排字工作流参考。

精确状态见 `docs/INTEGRATION_STATUS.md`，不要把“参考”与“直接调用”混为一谈。

---

## macOS Studio GUI v0.6（KCC 风格淡蓝精简版 + 批量/MPS）

本地 ZIP 版继续使用 `Manga HD Transfer Studio` PySide6 GUI：低饱和淡蓝、白色卡片、细边框、紧凑布局，并吸收 KCC-Kindle-CHS 的后台 QThread、进度/取消与资源清理方式：

- 页面管理：成对导入旧中文/高清日文，自动配对、批量状态、处理路线、断点续跑与安全取消；
- 配准/OCR：Apple Vision OCR、PP-OCRv5、cheap-first/SIFT/LightGlue/LoFTR、MangaLens 与 Sidecar；模型中心只做无副作用浅探测；
- 替换工作台：页面配准 → 中文气泡/文本框 → 实例匹配 → 局部精对齐 → 可选 MPS Torch 超分 → 蒙版替换/高清重排 → 出版 QA；
- Publication Builder：整册构建、输出统计、QA 状态。

Mac 推荐先双击 `启动_Manga_HD_Transfer.command`。需要 Apple Silicon AI 加速时再运行 `首次安装_MPS_AI加速.command`；该脚本安装运行库，但不捆绑漫画模型/超分权重。


### v0.6 降本增效

Auto 配准改成 `同源快速验证 → OpenCV SIFT/ORB → LightGlue/MPS → LoFTR/MPS`，不再默认先支付深度匹配成本；Auto 且 `allow_model_downloads=false` 时不会为了自动升级而隐藏下载模型权重。LightGlue、LoFTR、MangaLens、Torch 超分整册常驻复用。完成页可通过 job fingerprint 断点跳过，配准/OCR/气泡结构另有阶段缓存。

用户真实同源测试页上，v0.6 快速配准 5 次中位约 `0.016 s`，强制 SIFT 中位约 `1.083 s`，单配准约 `67.5×` 加速；最终图与 v0.5 逐像素一致，仍为 5/5 替换、QA 0/0。详见 `docs/PERFORMANCE_V06.md` 与 `docs/SELF_ACCEPTANCE_V06.md`。

### v0.5 真实成对页面验收

使用一组同页“日文原图 / 已正确嵌字中文图”进行端到端回归：SIFT 配准置信度约 `0.9998`，自动找到正好 5 个真正发生翻译变化的区域（4 个对白气泡 + 1 个说明文本框），顶部未翻译小气泡与画面拟声词不会被误选。最终替换蒙版内与中文参考逐像素一致，蒙版外与日文母版逐像素一致，QA 为 0 error / 0 warning。详见 `docs/REAL_PAIR_ACCEPTANCE_V05.md`。

### 蒙版替换 GUI

工作台右侧可直接切换 `高清重排 / 蒙版替换 / 智能混合`。v0.5 新增同源页面 `paired-diff` 快速路径：先通过页面配准定位真正发生翻译变化的气泡/文本框，同源页面可进行像素级精确覆盖；跨版本页面仍使用气泡实例匹配、局部 ECC、可选超分、Mask IoU 与目标覆盖率安全门槛。


## v0.8 手机翻拍旧版：光照归一化 + OCR 完整性保护

使用真实 `2400×3650` 手机翻拍中文版与 `850×1200` 日文母版做 5 页回归后，摄影版处理被单独拆成 `photo_pair` 路线：配准仍使用高清日文页作为几何真值，但不再依赖原始像素差；中文照片先做透视/尺寸映射，再在白色气泡内进行**确定性光照归一化**，去除反光、灰底和色温漂移，同时保留旧译文字形的真实抗锯齿。归一化后仍偏软才进入墨迹重建；极小字、开放爆炸框、覆盖不足或直接层漏检区域自动交给 OCR 高清重排。

Mac Studio 默认优先 Apple Vision OCR；摄影页固定 `safe_to_skip_ocr=False`，所以即使直接蒙版层已经成功若干气泡，也不会因此跳过 OCR。零候选/无 OCR 证据会产生 blocking QA error，避免把残缺页静默当成成品。5 页真实压力测试的配准 confidence 为 `0.867–0.890`；关闭 OCR 仅测直接层时 27 个安全候选中 21 个直接应用，其余明确转 OCR/几何拒绝。详见 `docs/PHOTO_PAIR_V08.md`。

## v0.8.1 合并版：保留 v0.8 结构差分回退

本合并包以 v0.8.1 为主体，不回退 `photo_pair`、摄影页 OCR 完整性保护、源分辨率保留和出版阻断 QA；同时恢复 v0.8 的结构墨迹差分、低频 DIS 局部光流、enclosed barrier 气泡检测、free-text/SFX 检测和 target-driven transfer。默认只在 v0.8.1 的高噪声摄影回退未能找到安全候选时启用 `structural_v08`，避免与新版 `photo_pair` 争抢同一区域。

兼容开关为 `mask_replace.paired_diff_structural_fallback_enabled`；旧版细粒度阈值字段也继续接受。结构回退始终保持 OCR，不会绕过 v0.8.1 的摄影页完整性保护。

## v0.7 低清中文文字保护

针对手机拍摄、反光、失焦和旧扫描版，蒙版替换新增 **Pixels → Ink reconstruction → OCR re-letter** 自动清晰度门控。模糊源文字不再被静默贴到高清母版；生成式图像模型不参与中文文字绘制。配准可靠时，旧中文版会先按高清页几何矫正后再做 Apple Vision / PP-OCRv5 OCR，并对低置信度块尝试 CLAHE、锐化和自适应二值版本。详见 `docs/TEXT_FIDELITY_V07.md`。

## v0.8.2：摄影版高清文字重建与小气泡完整替换

本版继续以 v0.8.1 + v0.8 合并版为基础，重点解决“替换后文字发糊”和摄影版小气泡被直接拒绝的问题。新增 `photo-crisp-ink` 路径：不再把手机照片里的灰底、眩光和模糊文字块原样贴进高清页，而是从已配准的源图中提取真实中文暗部细节，生成抗锯齿中性墨迹，再覆盖到高清日文页的干净纸面。该路径不依赖 OCR，也不会重新生成字符。

主要变化：

- 摄影版默认启用 `photo_pair_crisp_text_enabled`，清除相机灰雾、色偏和采样模糊；
- 通过 `photo_pair_crisp_border_guard_px` 排除源气泡边框，避免替换后出现“双重气泡轮廓”；
- 取消小于 88px 的摄影气泡硬拒绝，优先执行高清墨迹恢复；
- 对轻微欠分割的摄影气泡加入 1–3px 源掩膜扩张 salvage；
- 摄影页目标几何门槛调整为 target-driven 安全策略：IoU 0.74、coverage 0.84、spill 0.27；最终写入仍受高清目标掩膜约束；
- 当初始页面配对分数较低、但 SIFT/仿射配准高置信且所有摄影候选均成功替换时，页面配对 QA 降为 warning，不再误阻断；
- 无 OCR 时仍保留“可能漏掉开放式气泡/SFX”的 QA warning，但完整成功的闭合气泡替换不再返回失败退出码；
- 本版回归测试为 **29/29 通过**。

实图 `009` 复测：8 个摄影候选全部替换，`8/8 applied`，QA `0 error / 2 warning`；文字均走 `photo-crisp-ink`，不再复制摄影灰底或模糊像素。

## v0.8.3：源照片裁字保护，禁止“残缺中文假成功”

真实 `009` 的右上气泡暴露了一个重要完整性问题：中文版手机照片在画面右边缘已经把气泡和部分中文字符物理裁掉，但 v0.8.2 仍以约 `0.864` 的 target coverage 接受该区域，随后清空整块高清日文气泡，造成“只贴进去一部分中文，却显示替换成功”。

v0.8.3 新增摄影源边缘完整性门控：如果源气泡掩膜触碰照片边缘，就不能只使用普通 `coverage=0.84` 门槛，而必须达到 `photo_pair_edge_clip_min_target_coverage=0.94`。达不到时程序保持高清目标气泡原样，并输出 blocking QA `mask_replace_source_translation_clipped`，同时在 `mask_transfer.json` 写入 `source_edge_clipped` 和 `source_edge_sides`。这条规则只针对实际不完整的边缘源；本页右下同样触碰照片右边缘的「哼！」覆盖率约 `0.9889`，仍会正常替换，不会被误杀。

真实回归：`009` 现在为 7 个安全区域直接替换 + 1 个明确边缘裁切拒绝，不再出现残缺中文假成功；`007` 保持 4/4 成功。自动测试 **33/33** 通过。

## v0.8.5：摄影主路线 + 结构差分补漏 + GUI 待补字

v0.8.5 继续以 `photo_pair` 作为摄影旧版的主路线，同时把 v0.8 的结构差分限制为**非重叠补漏器**：只补主路线没有覆盖的开放爆炸框 / free text，并过滤邻近已确认气泡、页脚、低亮度和变化不足区域。补漏区域同样使用 `photo-crisp-ink`，不会直接发布模糊摄影像素。

普通对白在 OCR 可用时优先进入高清 reletter；排版方向会参考源 OCR 文字块几何，圆形气泡中的竖排中文不再仅由气泡长宽比决定。

对于源照片边缘物理裁掉的译文，替换工作台新增“待补文字”：选择待补气泡、输入完整中文、选择自动/竖排/横排后，可直接生成高清复核图，不必手工编辑 JSON。裁切源仍然不会被旧结构差分绕过。

彩色拟声词形状迁移保留为实验能力，默认关闭；真实摄影页上如果字形阴影重、源图本身模糊，强行迁移可能降低质量，因此正式默认策略优先保证成品清晰和可复核。

## v0.8.6：先给中文候选 + 模糊标注 + 可编辑/可还原

v0.8.6 将摄影旧版的失败策略改成 **review-first**：只要源中文版仍有可恢复中文字形，就先输出中文候选，而不是直接把高清日文留在最终页。候选不会被当作出版通过结果，会在 `review_preview.png` 和替换工作台中标注为“可能不完整/不准确”或“可能模糊/扭曲”，并支持接受、重新编辑高清排字或一键还原日文。

同时，摄影页局部几何不再使用可能压扁中文字形的 X/Y 独立 bbox 拉伸；默认使用等比例局部拟合。普通 OCR 重排则使用 4× 超采样字体渲染后一次缩回目标尺寸，改善小字号清晰度。

## v0.8.7：黑白中文版 → 彩色母版跨版本替换

v0.8.7 增加跨色彩版本路由。程序检测到低饱和/黑白中文版与彩色目标母版时，会优先信任高清目标页的气泡几何，不再因为两个版本的气泡尺寸、留白或排字位置不同而直接拒绝。对于目标页中可靠的白色文字容器，只要已配准源图和目标图都存在文字墨迹且差异足够，就可以先生成可复核的中文结果。

小气泡会从目标气泡附近恢复完整的中文字形块，过滤气泡边线和长画面线条后，以单一等比例缩放重新居中；同时清除目标 mask 边缘遗漏的旧日文字形和抗锯齿灰边。普通彩色摄影页仍保持 v0.8.6 的路线，不会因为跨版本增强而被强制切换。
.2 — Direct Patch / Mask Transfer 正式分离

- 新增独立 **`direct_patch`（直接贴图）**：面向完全一致或可证明为同版同布局的页面，直接从 SOURCE 扣取完整气泡/白底文本框内部的**原始栅格（白色背景 + 中文字形一起）**，经 identity / local similarity 对齐后硬覆盖到 TARGET。它不是“蒙版模式的一个参数”。
- **Direct Patch 严格不做 OCR、不清字、不 inpaint、不重新排字，也不偷偷回退到 Mask。** 同页预检、配准或容器安全性不满足时，显式 Direct 模式保留 TARGET 原图并标记 Review；`auto` 才允许继续走 Mask。
- `mask_replace` 重新限定为真正的 **蒙版迁移**：geometry mask / transfer(clear) mask 分离，按安全边界清除 TARGET 日文并合成 SOURCE 内容；彩色、网点、渐变等需要保留 TARGET 纹理的容器只能走 Mask/Auto fallback，不再冒充 Direct。
- 新增 `page_pairing.py` OCR-free 同页预检，使用低频结构、边缘重合和配准质量联合判断，降低相邻页错贴风险。
- 新增 `transfer_planner.py`：`auto → direct_patch（安全时）→ mask_replace`；显式模式保持严格语义。
- Direct 输出新增 `direct_patch_layer.png`、`direct_patch_regions.png`、`direct_patch.json`，同时保留旧 review 兼容产物。
- GUI 新增“自动 · Direct 优先 / 直接贴图 · 整气泡/文本框 / 精准蒙版迁移 / 智能混合 / 高清重排”，并把 Direct 与 Mask 的参数说明拆开。
- 新增 Direct/Mask 独立配置命名空间与回归测试，专门验证 Direct 不会重新混回 Mask/OCR。

# v0.8.32 — Source Direct Provider Integration

- 新增统一 ProviderRegistry 与 SOURCE-only provider chain：pseudo barrier、sidecar/CTD、Comic Translate RT-DETR-v2、SAM2、MangaLens、Debubble-style white overlay。
- 新增 `coordinate_space.py`：所有检测结果固定在 SOURCE 原始像素坐标，affine/homography 只做定位，最终中文 raster 始终 local similarity。
- 新增 Cotrans-inspired coloured clear-mask connected-component refiner；PanelCleaner-inspired progressive border fitter 已正式注册为 mask-refiner provider。
- Manga-Overlay-Translator 的原图坐标/缩放映射与缓存思路已适配；DebubbleBot 的独立 editable mask overlay 思路已适配。
- RT-DETR-v2 / SAM2 真正接入但默认不下载模型；成功 source-direct 路径仍为 0 OCR、0 TARGET bubble matching、0 边框写入。
- cheap SOURCE hints 先跑、direct plan 只构建一次，避免上一开发版重复整页构建。
- 详见 `INTEGRATION_STATUS_v0.8.32.md` 与 `TEST_REPORT_v0.8.32.md`。

# Manga HD Translation Transfer



## v0.8.31 — 结构配准 + 渐进安全蒙版 + 彩色框保色 + source-only 兜底

- **仍以 source-direct 为主流程。** 中文版源页是唯一文字/容器内容来源；日文高清页不再承担“重新检测气泡再配对”的职责，目标页轮廓仅用于坐标配准与 QA。
- 页面自动配准升级为 **quick SIFT/ORB + RANSAC → 结构图 ECC 残差精修**。ECC 只在约 900px 长边缩略结构图上运行，并受相关性提升、位移/旋转上限和特征重投影门槛约束；affine/homography 仍只负责找位置，最终中文字始终 local similarity。
- 吸收 PanelCleaner 的“渐进式 mask 拟合 / 低置信宁可跳过”思路，独立实现 **动态边框厚度估计 + progressive inset + safe/review/rejected 三态**。几何容器与最终 transfer mask 分离，边框永远只用于辅助对齐。
- 渐进蒙版计算改为 **候选 ROI 局部 morphology / distance transform**，不再每个轮廓都对 2K/4K 整页做距离变换；文字组件统计同样只在候选 ROI 运行。
- 吸收 manga-image-translator 的 mask dilation 思路：彩色框中的目标日文清除 mask 可受控扩张，减少残字；但不进入 OCR / 重排。
- **彩色尖角框不再把目标黄色/红色填充刷成白色。** 目标填充和高清边线保留，只在安全内部清除目标日文字形，再迁移已配准的中文墨迹 alpha。普通白气泡仍使用完整白底 + 中文整块硬覆盖，保证无目标日文残留。
- 增加 Comic Translate 风格的 **source-only detector fallback** 接口：默认关闭，仅当普通 source-direct 无法安全完成、用户明确配置本地 MangaLens/Ultralytics 权重时才调用；模型只补源页容器提示，绝不检测目标气泡或恢复双边匹配。
- 第三方参考源码按原许可证放入 `third_party_reference/`，与 MIT 运行时代码隔离；核心 `src/` 不导入 GPL 参考模块。
- 实页 `014.jpg` → `p-014.jpeg` 用于 v0.8.30 / v0.8.31 A/B：继续命中 6 个真实容器（3 白 + 3 彩色尖角），OCR=0、目标气泡匹配=0、辅助边框写入=0；v0.8.31 同时保持彩色填充并减少局部候选计算成本。

## v0.8.30 — 不同尺寸自动配准 / 仿射只定位 / 字形始终等比

- **不同像素尺寸、不同裁边、轻微旋转/扫描比例差现在自动处理。** 页面级配准可以使用 similarity / affine / homography 来确定“中文版坐标 → 日文版坐标”，不再要求两张图尺寸完全相同。
- 新增 **A0/A1/A2/A3 自动对齐模式**：同尺寸 1:1、全局等比、affine 位置映射 + 局部等比栅格、homography 位置映射 + 局部等比栅格。程序自动选择，不需要手工指定。
- **页面 affine/homography 永远不直接作用到中文字形。** 每个中文版气泡/文本框在对应位置只使用 local similarity（单一 scale + rotation + translation）渲染，因此 X/Y 拉伸、shear、透视都不会把中文压扁或扭曲。
- 每个容器增加局部边线微调：在目标页黑色/彩色框轮廓附近自动微调少量等比 scale、rotation、X/Y，边框仍然只用于对齐，最终 `border_pixels_written=0`。
- 注册阶段增加 **quick OpenCV** 预检：同版页面优先在约 1000px 长边进行 SIFT/ORB 配准，只有置信度/覆盖不足才升级到原来的高分辨率配准或深度后端，减少不同分辨率页面的等待。
- 使用 `014.jpg`（1440×2048）+ `p-014.jpeg`（1117×1600）回归：自动识别为 `A2_affine_location_local_similarity_raster`，普通白框和彩色尖角框继续直接从中文版整体覆盖，OCR/目标气泡双边匹配均未使用。

## v0.8.29 — 同版整气泡直接覆盖 / 彩色尖角框 / 边框仅对齐

- 新增 **source-direct whole-container 快速路线**：不再先检测“中文版气泡 → 日文版气泡”再做配对。直接从中文版提取完整对白框内部，按页面对应位置覆盖到日文高清页。
- **同尺寸/同坐标页面直接 1:1 拷贝**；分辨率不同的同版页面只使用一个全局等比缩放 + 小范围平移对齐，禁止 X/Y 非等比缩放、shear 和 OCR 重排。
- 气泡/尖角框的黑色边线和任何检测矩形都只作为定位证据，**最终不写入辅助边框**；最终写入区域先向内收缩，保留高清日文母版自己的轮廓。
- 历史版本曾加入 **彩色尖角拟声/对白框** 的 source-direct 实验路线；**v# v0.8.34.1 — Direct Patch / Mask Transfer 正式分离

- 新增独立 **`direct_patch`（直接贴图）**：面向完全一致或可证明为同版同布局的页面，直接从 SOURCE 扣取完整气泡/白底文本框内部的**原始栅格（白色背景 + 中文字形一起）**，经 identity / local similarity 对齐后硬覆盖到 TARGET。它不是“蒙版模式的一个参数”。
- **Direct Patch 严格不做 OCR、不清字、不 inpaint、不重新排字，也不偷偷回退到 Mask。** 同页预检、配准或容器安全性不满足时，显式 Direct 模式保留 TARGET 原图并标记 Review；`auto` 才允许继续走 Mask。
- `mask_replace` 重新限定为真正的 **蒙版迁移**：geometry mask / transfer(clear) mask 分离，按安全边界清除 TARGET 日文并合成 SOURCE 内容；彩色、网点、渐变等需要保留 TARGET 纹理的容器只能走 Mask/Auto fallback，不再冒充 Direct。
- 新增 `page_pairing.py` OCR-free 同页预检，使用低频结构、边缘重合和配准质量联合判断，降低相邻页错贴风险。
- 新增 `transfer_planner.py`：`auto → direct_patch（安全时）→ mask_replace`；显式模式保持严格语义。
- Direct 输出新增 `direct_patch_layer.png`、`direct_patch_regions.png`、`direct_patch.json`，同时保留旧 review 兼容产物。
- GUI 新增“自动 · Direct 优先 / 直接贴图 · 整气泡/文本框 / 精准蒙版迁移 / 智能混合 / 高清重排”，并把 Direct 与 Mask 的参数说明拆开。
- 新增 Direct/Mask 独立配置命名空间与回归测试，专门验证 Direct 不会重新混回 Mask/OCR。

# v0.8.32 — Source Direct Provider Integration

- 新增统一 ProviderRegistry 与 SOURCE-only provider chain：pseudo barrier、sidecar/CTD、Comic Translate RT-DETR-v2、SAM2、MangaLens、Debubble-style white overlay。
- 新增 `coordinate_space.py`：所有检测结果固定在 SOURCE 原始像素坐标，affine/homography 只做定位，最终中文 raster 始终 local similarity。
- 新增 Cotrans-inspired coloured clear-mask connected-component refiner；PanelCleaner-inspired progressive border fitter 已正式注册为 mask-refiner provider。
- Manga-Overlay-Translator 的原图坐标/缩放映射与缓存思路已适配；DebubbleBot 的独立 editable mask overlay 思路已适配。
- RT-DETR-v2 / SAM2 真正接入但默认不下载模型；成功 source-direct 路径仍为 0 OCR、0 TARGET bubble matching、0 边框写入。
- cheap SOURCE hints 先跑、direct plan 只构建一次，避免上一开发版重复整页构建。
- 详见 `INTEGRATION_STATUS_v0.8.32.md` 与 `TEST_REPORT_v0.8.32.md`。

# Manga HD Translation Transfer



## v0.8.31 — 结构配准 + 渐进安全蒙版 + 彩色框保色 + source-only 兜底

- **仍以 source-direct 为主流程。** 中文版源页是唯一文字/容器内容来源；日文高清页不再承担“重新检测气泡再配对”的职责，目标页轮廓仅用于坐标配准与 QA。
- 页面自动配准升级为 **quick SIFT/ORB + RANSAC → 结构图 ECC 残差精修**。ECC 只在约 900px 长边缩略结构图上运行，并受相关性提升、位移/旋转上限和特征重投影门槛约束；affine/homography 仍只负责找位置，最终中文字始终 local similarity。
- 吸收 PanelCleaner 的“渐进式 mask 拟合 / 低置信宁可跳过”思路，独立实现 **动态边框厚度估计 + progressive inset + safe/review/rejected 三态**。几何容器与最终 transfer mask 分离，边框永远只用于辅助对齐。
- 渐进蒙版计算改为 **候选 ROI 局部 morphology / distance transform**，不再每个轮廓都对 2K/4K 整页做距离变换；文字组件统计同样只在候选 ROI 运行。
- 吸收 manga-image-translator 的 mask dilation 思路：彩色框中的目标日文清除 mask 可受控扩张，减少残字；但不进入 OCR / 重排。
- **彩色尖角框不再把目标黄色/红色填充刷成白色。** 目标填充和高清边线保留，只在安全内部清除目标日文字形，再迁移已配准的中文墨迹 alpha。普通白气泡仍使用完整白底 + 中文整块硬覆盖，保证无目标日文残留。
- 增加 Comic Translate 风格的 **source-only detector fallback** 接口：默认关闭，仅当普通 source-direct 无法安全完成、用户明确配置本地 MangaLens/Ultralytics 权重时才调用；模型只补源页容器提示，绝不检测目标气泡或恢复双边匹配。
- 第三方参考源码按原许可证放入 `third_party_reference/`，与 MIT 运行时代码隔离；核心 `src/` 不导入 GPL 参考模块。
- 实页 `014.jpg` → `p-014.jpeg` 用于 v0.8.30 / v0.8.31 A/B：继续命中 6 个真实容器（3 白 + 3 彩色尖角），OCR=0、目标气泡匹配=0、辅助边框写入=0；v0.8.31 同时保持彩色填充并减少局部候选计算成本。

## v0.8.30 — 不同尺寸自动配准 / 仿射只定位 / 字形始终等比

- **不同像素尺寸、不同裁边、轻微旋转/扫描比例差现在自动处理。** 页面级配准可以使用 similarity / affine / homography 来确定“中文版坐标 → 日文版坐标”，不再要求两张图尺寸完全相同。
- 新增 **A0/A1/A2/A3 自动对齐模式**：同尺寸 1:1、全局等比、affine 位置映射 + 局部等比栅格、homography 位置映射 + 局部等比栅格。程序自动选择，不需要手工指定。
- **页面 affine/homography 永远不直接作用到中文字形。** 每个中文版气泡/文本框在对应位置只使用 local similarity（单一 scale + rotation + translation）渲染，因此 X/Y 拉伸、shear、透视都不会把中文压扁或扭曲。
- 每个容器增加局部边线微调：在目标页黑色/彩色框轮廓附近自动微调少量等比 scale、rotation、X/Y，边框仍然只用于对齐，最终 `border_pixels_written=0`。
- 注册阶段增加 **quick OpenCV** 预检：同版页面优先在约 1000px 长边进行 SIFT/ORB 配准，只有置信度/覆盖不足才升级到原来的高分辨率配准或深度后端，减少不同分辨率页面的等待。
- 使用 `014.jpg`（1440×2048）+ `p-014.jpeg`（1117×1600）回归：自动识别为 `A2_affine_location_local_similarity_raster`，普通白框和彩色尖角框继续直接从中文版整体覆盖，OCR/目标气泡双边匹配均未使用。

## v0.8.29 — 同版整气泡直接覆盖 / 彩色尖角框 / 边框仅对齐

- 新增 **source-direct whole-container 快速路线**：不再先检测“中文版气泡 → 日文版气泡”再做配对。直接从中文版提取完整对白框内部，按页面对应位置覆盖到日文高清页。
- **同尺寸/同坐标页面直接 1:1 拷贝**；分辨率不同的同版页面只使用一个全局等比缩放 + 小范围平移对齐，禁止 X/Y 非等比缩放、shear 和 OCR 重排。
- 气泡/尖角框的黑色边线和任何检测矩形都只作为定位证据，**最终不写入辅助边框**；最终写入区域先向内收缩，保留高清日文母版自己的轮廓。
- 历史版本曾加入 **彩色尖角拟声/对白框** 的 source-direct 实验路线；**v0.8.34.1 起此行为已收紧**：需要保留 TARGET 彩色/纹理的区域归入 Mask/Auto fallback，显式 Direct 不再使用这条彩色覆盖语义。
- 加入 artwork guard：人物皮肤、头发、建筑等即使是彩色区域，也必须同时通过源白底、紧凑文字、边界对齐和中日墨迹变化门槛，避免把画面误当彩色对白框。
- 高置信同版页命中快速路线后直接跳过 paired-diff、OCR、气泡双边匹配和重复 completion，减少整页处理时间。
- 使用用户提供的 `014.jpg` + `p-014.jpeg` 实测：自动选中 6 个真实对白/尖角框（3 个普通白框 + 3 个彩色尖角框），未把人物/建筑误覆盖。

## v0.8.28 — 整气泡覆盖强化 / 白色爆发框 / 页面级性能优化

- **整气泡 / 白底文本框覆盖现在在 GUI 中有明确开关，默认开启。** 白色容器优先复制旧中文版的完整白底 + 中文字补丁，不拆字、不 OCR 重排。
- 新增 **白色爆发框 / 锯齿对白框** 安全路线：允许低 bbox 填充率的星爆/锯齿白框进入整容器覆盖，但仍要求中日两页都满足白度、文字墨迹、饱和度、长宽比和配对一致性。
- 修复实心容器补洞对锯齿爆发框的潜在破坏：低 solidity 的星爆形状不再使用大核 closing / convex-envelope 补洞，只填真正的内部文字孔洞，保留每个尖角和凹槽。
- 页面级性能优化：source/target 的 Gray/HSV 转换在整页只计算一次，气泡越多收益越明显；图层写入也不再为每个气泡整页执行一次 BGR→RGB 转换。
- 真实 006 / 065 / 066 / 007 再次回归，QA 均为 0 error；006 小对白继续走完整 source container patch。

## v0.8.27 — 整个中文版气泡 / 文本框覆盖

- 白色气泡、白底旁白框和白底文本框默认使用 `rigid-container-full-patch`：直接迁移旧中文版的白底 + 中文字整体补丁。
- 目标容器仅提供位置与边界；中文字不使用 X/Y 非等比缩放，不通过 OCR 重排。
- 目标日文文字被完整白底补丁覆盖，避免“中文字已经写入但日文仍残留”的混合结果。
- 原 `rigid-container-raster` 仍作为兼容回退路径。

## v0.8.26 — 实心气泡内部 / 日文残留修复

- 修复白色气泡 detector mask 被日文字切出黑缺口后，清除阶段反而跳过日文字的根因。
- 检测 mask 与真正的容器内部 mask 分离：闭合窄缺口、填内部孔洞，并保留边界保护带。
- `clear mask` 与中文字栅格裁切都使用实心容器内部，避免日文残留与中文缺笔。
- 065 的爆发式黑色气泡边缘通过 convex-envelope boundary guard 保留，不会因补洞被刷白。
- OCR-free 白色容器补漏提高文字密度门槛，减少建筑/天空白块误判。
- 真实 006 小对白已重新验证：原日文清空，中文完整，QA pass。

## v0.8.25 — 白色容器补漏与独立清除蒙版

- 同版黑白中文 → 彩色日文页新增 OCR-free 白色容器补漏。目标气泡/文本框作为几何真值，反向映射到原始中文页取完整中文字形，避免源扫描气泡连到纸面或被翻译组扩框时被错误拒绝。
- 中文最终字形继续严格 `uniform scale + translation`，不继承页面 affine/dense-flow 的非等比变形。
- 工作台新增 `清除蒙版 / 中文迁移层 / 只清日文` 三个独立视图，以及可画笔增删的清除蒙版 overlay 编辑器。
- `仅执行去字` 不重跑配准、OCR、检测或中文字形迁移。
- 可独立选择气泡 detector、检测分辨率、清除 mask 扩张和 inpainting backend。
- 006 / 065 / 066 用户真实页均重新回归；006 右下角此前漏掉的对白已由自动路径恢复。


## v0.8.23 锁定栅格气泡迁移：位置可以仿射，字形绝不仿射

本版重新定义黑白旧中文版 → 彩色/高清日文页的最终写入规则：**页面配准负责找位置，中文字栅格只允许单一等比缩放 + 平移**。全局 affine / homography、局部 dense-flow 都不能再直接作用到最终 CJK 字形。

- 对白色对白气泡、旁白框、白底文本框，目标高清页的完整容器是清除边界：保留目标轮廓/尾巴，清空内部日文，再从原始旧中文版取得整块中文字栅格。
- 字符和标点不拆成 connected components 重组；整块灰度/抗锯齿栅格作为一个连续 alpha 场迁移，避免缺笔、断标点和字符重心漂移。
- 分辨率不同只允许一个 scalar scale；禁止 X/Y 独立 resize、shear 和把页面 affine 的非等比校正传给字形。
- 结构候选只用于发现区域；若目标 mask 恰好切到日文字边缘，只清除字符尺寸的残余组件，不进行可能泄漏到画面的白区 flood-fill。
- 黄色/彩色爆炸框保留目标彩色背景，只迁移中文墨迹，不复制黑白版的爆炸框底色/线稿。
- 复杂/开放文字在真正写入最终页之前先做内容完整性检查；检查失败就不发布该候选，避免出现“写入成功但字残缺”的半成品。
- 065 / 066 两组真实图在 OCR=none 下重新回归：065 为 5/5 内容完整，066 为 4/4 内容完整；已检测区域目标日文残留率为 0。


## v0.8.22 真实 065/066 验证：内容完整性与跨版本复杂文字

本版针对黑白旧中文版 → 彩色/高清日文原页的真实 065、066 配对做了回归，不再把 `applied=true` 当成“翻译完整”。主要变化：

- **几何写入与内容完整性分离**：每条替换记录新增 `content_check`、`source_ink_coverage`、`target_residual_ratio`、`content_complete`。跨版本白气泡、开放文字、复杂文字、OCR 几何兜底若内容未验证/不完整，会由 QA 阻止作为成功页。
- **跨版本白气泡候选守门**：黑白→彩色时用高白度、填充率、内部文字墨迹和边界证据收紧气泡，防止列车车身、招牌、浅色画面块被当成对白框。
- **彩色爆炸框独立路线**：高饱和黄色/红色文字框必须与结构候选有足够真实重叠，并要求注册后的中文源区域主要是亮底，避免邻近蓝色 SFX/招牌误触发。只擦目标文字笔画，只写源中文笔画，不复制黑白底纹。
- **结构局部流只负责检测**：结构补充仍可用局部/稠密流发现变化，但最终中文字统一优先取全局注册源，避免局部光流扭曲 CJK 字形、把爆炸框边线拖入最终图。
- **`transfer_audit.v2`**：同时记录 `geometry_applied`、`content_checked`、`content_complete`、`content_incomplete`、`content_unverified`、最差源墨迹覆盖率和最大目标残留率。无 OCR 时明确标记验证范围为 `detected_regions_only_no_ocr`，不假装做了整页语言识别。
- **真实样本**：065 共 5 个区域（3 白气泡 + 2 黄色爆炸框）全部内容验证通过；066 共 4 个区域（列车小气泡、右侧气泡、中间气泡、左下大气泡）全部内容验证通过，未再修改右侧白色招牌/画面块。

## v0.8.21 复杂文字 / 漏检恢复 / 逐页迁移审计

- 精准蒙版增加 **复杂文字区域** 路线：黄色爆发框、开放式文字、彩色背景文字不再被“近白底”门槛直接丢弃；必须同时满足源/目标两侧紧凑文字墨迹、结构差异和面积上限，最终仍只搬运源中文栅格墨迹，不做 OCR 重排。
- 成对差异的闭合气泡检测之外增加结构补充：raw-diff 没有建立容器时可回退到 structural-v08；raw-diff 已找到普通气泡时也可补充不重叠的 free/complex text。低置信度文本不再静默丢弃，而是生成可恢复、可编辑的复核候选。
- 新增 **OCR 几何引导组件迁移**：OCR 只用于确认“源中文区域 ↔ 目标日文区域”的对应和候选范围；真正清除的是目标日文字形组件，真正写回的是配准后的源中文像素。`strict_mask_replace_no_ocr_reletter=true` 时绝不拿 OCR 字符串重打字。
- 目标清除蒙版允许窄长竖排日文列和横向标题组件跨出不完整气泡内框；仅对经过组件形状约束的目标文字先行清除，避免残留半截日文，同时保护气泡轮廓、格线和彩色背景。
- 每页新增 `transfer_audit.json`、`source_original.png`、`target_clear_mask.png`、`chinese_transfer_layer.png`，并保证 `review_preview.png` 总是存在；可直接核对配准、候选类型、OCR 证据、应用/拒绝/复核数量、清除像素和 QA。
- 修复 raw-diff 中重复执行一次源白区组件搜索的隐藏性能问题；精准蒙版 0.8.16 的 glyph-footprint rescue 保持不变。
- 页面管理/跳过逻辑保持 v0.8.20：所有新页默认正文，不做配对后自动气泡扫描；无中文 OCR 文本的正文可原样输出高清日文页；手动跳过只跳过替换，不从最终整册删页。

## v0.8.20 默认正文 / 按需替换 / 缩略图性能修复

- **删除配对后的自动气泡/文本框检查**：新配对页面一律默认为“正文 / 需替换”，页面类型只由用户手动分类；v0.8.19 保存的 `auto_no_text` 自动标记会迁移回默认正文。
- “正文”现在表示**允许进入迁移**，不再表示“必须改图”。实际处理时若中文源页 OCR + 气泡结构确认没有可迁移的 speech/narration 文本框，会直接输出未修改的高清日文页。精准蒙版模式仍禁止 OCR 改写原中文字形。
- 手动标记为封面、扉页、目录、单话首页、插图、空白页或“跳过”的页面仍是完整输出页：只跳过替换步骤，不从 `final/` 和整册页序删除。
- 缩略图改为**可视区域懒加载 + 滚动静默期加载 + 128 张缩放图 LRU 缓存**。滚轮/触控板滚动时不再后台持续解码整本漫画，停止滚动后才分小批加载当前视口及相邻一行。
- 切换页面类型不再重新解码已经缓存的缩略图；“刷新”才主动清空缩略图缓存。
- 整册最终输出文件名增加同名保护，避免极端情况下两个目标页同 stem 时互相覆盖。

## v0.8.19 可视化页面管理

- 页面管理默认改为 **缩略图画廊**：每一页直接显示高清日文页面缩略图、页码、页面类型、手动/自动来源和类型色标；可切换为旧版中文缩略图，不再靠文件名猜封面/目录/插图。
- **双击任意缩略图或列表行**打开可缩放大图窗口，旧版中文与高清日文左右并排；支持上一页/下一页、滚轮缩放、拖拽、适合窗口和 100% 查看。
- 缩略图支持 Ctrl / Command / Shift 多选，右键可直接批量标记正文、封面、扉页、目录、单话首页、纯图片等，也可恢复自动判断或只检查所选页。
- 新增“日文缩略图 / 中文缩略图”切换、页面筛选和当前选择计数；右侧详情同步显示当前页大预览、分类原因、气泡/自由文字数量、配对方法和置信度。
- 大图窗口的分类状态刷新不再重复解码两张全分辨率图片，避免批量处理/自动检查时因 UI 状态更新造成卡顿。
- `page_management.json` 改用同目录临时文件 + `fsync` + 原子替换保存；强制退出或写盘异常时不会把已有页面分类文件写成半截 JSON。

## v0.8.18 页面管理 / 安全停止

- 顶部加入始终可见的红色 **“■ 停止”** 按钮，页面自动检查、单页处理和整册处理共用同一套安全取消信号；单页 Pipeline 也在配准、成对差异、OCR/气泡、迁移和导出等安全边界检查取消。
- “页面”重构为 **页面管理**：每个已配对页面都有独立处理准入状态，可批量标记为正文、封面、扉页/书名页、目录、单话首页/章节页、插图/纯图片、卷首插画、空白页、后记/版权/广告或手动跳过。
- 配对后默认运行 OCR-free 页面检查：只使用下采样 OpenCV 配准 + paired-diff 几何，不读取文字内容；高置信度且没有气泡/文本框的页面自动标为“无气泡/文本框（自动）”并跳过迁移。检测失败或配准不足时**绝不自动丢页**，保持进入正常处理。
- 跳过页仍保留在整册页序中，并把高清日文原页原样写入 `final/`；不会进入 OCR、气泡迁移、清字或重排。手动“正文 / 需替换”可强制恢复处理。
- 页面标记持久化到输出目录 `page_management.json`，重新打开项目后继续生效；断点续跑也服从最新页面标记，避免把旧的已处理结果错误恢复到后来标记为封面/插图的页面。
- 页面类型使用不同颜色，表格同时显示“处理/跳过、页面类型、配对方法、状态和路线”，并支持多选批量标记、恢复自动、只检查所选页。

## v0.8.17 工作台同步与 UI 修复

- **替换工作台统一页状态**：日文原图、旧中文版、最终结果、复核标注和替换蒙版都从同一个当前页解析，不再复用“上一张最后结果”。
- 工作台新增 **上一页 / 下一页 / 当前页计数**，最终结果可连续翻页；当前页没有结果时明确显示为空，不会误显示其它页面。
- 整册处理完成后，每页工程会进入按页索引；同时可直接从 `pages/<page_id>/project.json` 和 `final/` 恢复预览，断点续跑页也能正常浏览。
- 复核后的 `final_reviewed.png` 优先显示，并同步回整册 `final/` 对应页，避免预览结果和出版输出不一致。
- “优先名称 / 页码配对”和“优先文件夹自然顺序”默认关闭，并增加明显间距与说明；默认走智能视觉配对。
- 修复 CJK 文件名页 ID：日文/中文纯文字文件名不再全部坍缩成同一个 `page` 目录；同时兼容 v0.8.16 单页处理留下的旧目录。
- UI 继续参考 KCC-Kindle-CHS 的卡片化、双栏、右侧滚动设置区和紧凑 macOS 控件设计，但不改动迁移核心信号与处理流程。

## v0.8.11 排版质量策略

默认采用“**OCR 认字，源图定排版**”：Apple Live Text 成功并不意味着一定重新排字。旧中文版迁移后的字形已经清晰时，程序保留原字号、分列和相对位置；只有模糊/低清/不安全区域才交给 OCR 高清重排。这样避免真实 Mac 批处理里出现巨大字体、分列错乱、白块和空白气泡。

把**旧版低清中文汉化漫画**中的既有译文可靠迁移到**高清日文原图**，完成跨版本页面配准、中文 OCR、日文清字、气泡安全区排版、自动 QA、人工复核与分层导出。

项目不是机器翻译器。它不重新翻译日文；中文内容来自你已有的旧汉化版。

## 目标

输入：

- `source_cn/`：旧版中文汉化图，允许低清、压缩、扫描偏移、裁边、额外 staff 页。
- `target_jp/`：同一作品的高清日文图，作为唯一画面母版。

输出：

- `final/`：自动通过安全门槛的最终 PNG。
- `pages/.../target_original.png`：高清母版，永不覆盖。
- `inpainted.png` / `clear_mask.png`：清字结果与精确 mask。
- `text_layer.png`：透明中文文字层。
- `editable.ora`：OpenRaster 分层工程。
- `editable.psd`：系统有 ImageMagick 时自动生成分层 PSD。
- `project.json`：配准矩阵、OCR、气泡、匹配、排字、置信度和可追溯信息。
- `qa.json` / debug overlays：出版 QA 和视觉证据。
- 本地 Review 编辑器：可改译文、改目标匹配、画/擦清字 mask，再重新生成。

## 核心设计

```text
旧中文版 ─ 页面指纹/顺序配对 ─┐
                            ├─ cheap-first / SIFT / LightGlue / LoFTR + RANSAC 配准
高清日文版 ─ 页面指纹/顺序配对 ┘

旧中文版 ─ 中文 OCR / 多次低置信度复识 ─ 中文 TextUnit
高清日文版 ─ 日文文字区域 + 气泡实例 / safe mask ─ TargetUnit

TextUnit + Registration + TargetUnit
              ↓
      跨版本最小成本身份匹配
              ↓
      日文像素级 text mask 清字
              ↓
  solid / OpenCV / external LaMa 修复
              ↓
  中文约束排字（字号、断行、禁则、安全区）
              ↓
       QA 安全门槛 + Review Queue
              ↓
       PNG / ORA / PSD / JSON
```

## 五种译文迁移模式

### 1. 自动 `auto`（默认）

先做 OCR-free 同页预检和页面配准。若页面属于同版同布局、容器可安全直接迁移，则走 `direct_patch`；Direct 不满足条件时才进入 `mask_replace`。这条路线用于批量处理时自动“先便宜、后复杂”，避免所有模型全跑。

### 2. 直接贴图 `direct_patch`

这是和蒙版迁移**语义完全不同**的一条路线，专门针对完全一致或高度一致的两张图。SOURCE 的完整气泡/白底文本框内部被当作一个原始栅格 patch，**白色背景与中文文字一起扣出、一起对齐、一起覆盖**。

- identity 页面优先 1:1 覆盖；尺寸/裁边略有变化时只允许 local similarity（统一缩放 + 小旋转 + 平移）。
- 页面 affine/homography 只负责“找到 TARGET 位置”，不会把中文字形做 X/Y 拉伸或透视变形。
- 不调用 OCR，不重新输入中文，不清 TARGET 日文，不 inpaint，不走 target-aware 背景重建。
- 显式 `direct_patch` 只要同页预检、配准或容器完整性不够安全，就拒绝该页/区域并保留 TARGET；**绝不静默切换到蒙版或 OCR**。
- 彩色/网点/渐变容器如果需要保留 TARGET 原纹理，应选择 `auto` 或 `mask_replace`，而不是 Direct。

### 3. 精准蒙版迁移 `mask_replace`

蒙版模式不是“整块贴图”的别名。它以 `geometry_mask` 表示容器几何，以独立的 transfer/clear mask 表示真正允许修改 TARGET 的区域：先保护边框和画面，再在安全区域内清除 TARGET 日文、迁移 SOURCE 中文 ink / interior / artwork。

对于白色普通气泡可以迁移 SOURCE interior；对于彩色、网点、渐变容器则优先保留 TARGET background/texture，只清日文墨迹并迁移中文。局部几何、内容完整度、spill、边框保护等 QA 不满足时进入 Review。

### 4. 智能混合 `hybrid`

优先使用蒙版迁移保留旧汉化排字；某个区域无法安全完成时，允许退回 OCR → 清字 → 高清重排。适合确实需要传统 fallback 的跨版本页面。

### 5. 高清重排 `reletter`

旧中文版提供中文文本内容，高清日文版负责清字与重新排版。适用于 SOURCE 字形本身太糊、需要统一出版级字体的情况。

CLI：

```bash
mhd-transfer run source_cn target_jp output --mode auto
mhd-transfer run source_cn target_jp output --mode direct_patch
mhd-transfer run source_cn target_jp output --mode mask_replace
mhd-transfer run source_cn target_jp output --mode hybrid
mhd-transfer run source_cn target_jp output --mode reletter
```

Direct / Mask 的边界和回归验收见 `docs/DIRECT_VS_MASK_V0834.md`；蒙版算法细节见 `docs/MASK_REPLACE_PLAN.md`。

### 与简单“替换翻译”方案的关键区别

1. **先做视觉配准，再匹配文字身份**，不依赖整页 resize 后固定 IoU。
2. **Direct Patch** 迁移 SOURCE 的整块原始栅格；**蒙版迁移**在独立 clear/transfer mask 内做 TARGET-aware 合成；**高清重排**才迁移文本内容并重新渲染，三种语义严格分离。
3. **清字 mask 与排字 safe area 分离**：前者只描述要删除的日文，后者描述中文允许出现的位置。
4. 气泡边界有保护带；中文最终字形 mask 必须通过 safe-area 覆盖验证。
5. 低页面配对、低配准、低 OCR、低身份匹配、拆分/合并关系都会阻止自动覆盖，进入 Review。
6. 每页有完整 evidence/debug，不做不可追溯的黑箱整页重绘。

## 安装

Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

核心安装只需要 OpenCV / NumPy / Pillow / SciPy / Pydantic / Typer，不下载 OCR 或深度模型。

### 中文 / 日文 OCR

```bash
pip install -e '.[ocr]'
```

默认显式使用 `PP-OCRv5`。PaddleOCR 的模型可能在首次实际推理时下载。若当前平台不方便运行 Paddle，也可使用 `sidecar`，把任何外部 OCR/漫画文字检测器的结果接进来。

### LightGlue / LoFTR 增强配准

```bash
pip install -e '.[lightglue]'
```

`registration.backend=auto` 采用 cheap-first：同源快速结构验证 → OpenCV SIFT/ORB → 本地已有权重时 LightGlue → LoFTR。Auto 默认不允许隐藏下载权重。Apple Silicon 可用 `首次安装_MPS_AI加速.command` 安装 MPS 推理运行库；模型权重仍由用户显式准备。

### LaMa

项目不复制/绑定 LaMa 模型。把任意可用 LaMa wrapper 配成：

```json
{
  "inpainting": {
    "backend": "lama",
    "lama_command": "python lama_wrapper.py --input {input} --mask {mask} --output {output}"
  }
}
```

## 首次运行

```bash
mhd-transfer doctor
mhd-transfer init-config config.json
mhd-transfer run source_cn target_jp output --config config.json
```

如果使用外部 OCR sidecar：

```bash
mhd-transfer run source_cn target_jp output --ocr-backend sidecar
```

运行结束若存在出版阻断级 QA，CLI 返回非 0，并提示进入 Review：

```bash
mhd-transfer review output
```

浏览器打开三栏编辑器：旧中文版 / 高清日文与清字 mask / 当前输出。可以：

- 修改 OCR 中文译文；
- 改某个中文 TextUnit 对应的目标气泡/文本框；
- 勾选是否应用该文本；
- 画笔增加或擦除清字 mask；
- 保存并点击“应用复核并重新生成”。

也可以命令行应用：

```bash
mhd-transfer apply-review output/pages/0001_xxx
```

## v0.6 批量与 Apple MPS

- 整册处理支持断点续跑、失败页继续、实时进度和安全取消；
- 配准/OCR/气泡实例按输入+配置做本地阶段缓存；
- LightGlue、LoFTR、MangaLens 和 Torch 超分模型整册常驻，不再每页重复初始化；
- Apple Silicon 上可选择 `mps`，MPS GPU 推理受控串行，CPU 图像预处理继续使用受限线程；
- 同源页面先跑约缩略图级的结构/phase 快速验证，只有不确定页才支付 SIFT/深度模型成本；
- 打开 GUI 的模型页不会因此 import 大型模型或下载权重。

详细见 `docs/PERFORMANCE_V06.md`。

## Sidecar 接口

### OCR / 漫画文字分割

`page.png` 对应 `page.ocr.json`：

```json
{
  "blocks": [
    {
      "id": "b0",
      "text": "已经存在的中文译文",
      "confidence": 0.99,
      "polygon": [[100,100],[260,100],[260,180],[100,180]],
      "kind": "speech",
      "mask_path": "masks/b0.png"
    }
  ]
}
```

如果提供 `mask_path`，清字优先使用像素级 segmentation；没有时才回退到 polygon。

### 气泡 / 旁白框实例分割

`page.bubbles.json`：

```json
{
  "bubbles": [
    {
      "id": "bubble-0",
      "kind": "speech",
      "confidence": 0.99,
      "polygon": [[80,70],[300,70],[300,220],[80,220]],
      "mask_path": "bubbles/0.png",
      "safe_mask_path": "bubbles/0_safe.png"
    }
  ]
}
```

没有 `safe_mask_path` 时会自动向内腐蚀实例 mask，窄尾部通常会自然脱离排版安全区。

## 自验收

完全离线、无模型下载：

```bash
pytest
mhd-transfer selftest
python benchmarks/synthetic_acceptance.py
```

本次实现时结果：

- v0.8.1 合并版单元/集成测试：**25/25 通过**，新增 cheap-first 配准、断点续跑与阶段缓存测试。
- 内置端到端 selftest：**通过**，无 error/warning。
- 20 组随机几何扰动配准：**20/20 通过**。
- 控制点中位误差：约 **0.051 px**；P95：约 **0.117 px**。
- 4 组不同长度中文气泡排字：**4/4 通过**，字形安全区覆盖均 ≥ 0.997。
- 本机层导出验证：ORA 成功；ImageMagick 存在时 PSD 成功。

详见 `docs/SELF_ACCEPTANCE.md`。

> 合成验收不能替代真实出版数据验收。最终“出版级”门槛仍要求建立同作品的 100–300 页人工真值集，统计页面配对、区域身份匹配、残留日文、误伤线稿、排字越界和人工复核率。程序已把这些证据与 QA 接口保留下来。

## 目录

```text
src/manga_hd_transfer/
  pairing.py          页面指纹与顺序配对
  registration.py     SIFT / LightGlue / LoFTR + RANSAC
  ocr.py              Paddle、sidecar、低置信度复识
  bubbles.py          气泡实例 / safe area / TextUnit
  matching.py         跨版本身份匹配、拆分/合并检测
  masking.py          像素级/多边形日文清字 mask
  mask_transfer.py     旧中文气泡/文本框 patch 对齐、超采样与蒙版替换
  inpainting.py       solid / OpenCV / external LaMa
  lettering.py        中文约束排字
  qa.py               出版 QA
  pipeline.py         全流水线
  review.py           本地人工复核编辑器
  review_apply.py     Review 回写与重生成
  export.py           PNG / ORA / PSD layer export
```

## 参考与许可

v0.8.32 已将部分经许可的关键上游源码/摘录放入 `third_party_reference/` 作隔离参考；运行时适配状态见 `INTEGRATION_STATUS_v0.8.32.md`，GPL 参考文件不从 MIT 核心导入。

- `hgmzhn/manga-translator-ui`：Replace Translation 业务原型。
- `zyddnys/manga-image-translator`：漫画翻译流水线。
- `dmMaze/BallonsTranslator`：编辑器、mask/inpaint/排字工作流。
- `dmMaze/comic-text-detector`：漫画文本框、文本行、segmentation；当前通过 pixel-mask sidecar 接入。
- `ogkalu2/comic-translate`：气泡检测与文本分割分层、PPOCR/LaMa 模块化。
- `PaddlePaddle/PaddleOCR`：PP-OCRv5 直接后端。
- `cvg/LightGlue`：局部特征匹配直接可选后端。
- `zju3dv/LoFTR`：困难图像匹配直接可选后备。
- `advimman/lama`：复杂背景 external command 后端。
- `facebookresearch/sam2`：v0.8.32 已接入 SOURCE-only 可选分割 provider；默认不下载模型。
- `huyvux3005/manga109-segmentation-bubble`（MangaLens）：本版新增 Ultralytics 本地模型直接后端。
- `ScanR/TypeR`：自动居中、行距联动、样式预设与紧凑排字工作流参考。

精确状态见 `docs/INTEGRATION_STATUS.md`，不要把“参考”与“直接调用”混为一谈。

---

## macOS Studio GUI v0.6（KCC 风格淡蓝精简版 + 批量/MPS）

本地 ZIP 版继续使用 `Manga HD Transfer Studio` PySide6 GUI：低饱和淡蓝、白色卡片、细边框、紧凑布局，并吸收 KCC-Kindle-CHS 的后台 QThread、进度/取消与资源清理方式：

- 页面管理：成对导入旧中文/高清日文，自动配对、批量状态、处理路线、断点续跑与安全取消；
- 配准/OCR：Apple Vision OCR、PP-OCRv5、cheap-first/SIFT/LightGlue/LoFTR、MangaLens 与 Sidecar；模型中心只做无副作用浅探测；
- 替换工作台：页面配准 → 中文气泡/文本框 → 实例匹配 → 局部精对齐 → 可选 MPS Torch 超分 → 蒙版替换/高清重排 → 出版 QA；
- Publication Builder：整册构建、输出统计、QA 状态。

Mac 推荐先双击 `启动_Manga_HD_Transfer.command`。需要 Apple Silicon AI 加速时再运行 `首次安装_MPS_AI加速.command`；该脚本安装运行库，但不捆绑漫画模型/超分权重。


### v0.6 降本增效

Auto 配准改成 `同源快速验证 → OpenCV SIFT/ORB → LightGlue/MPS → LoFTR/MPS`，不再默认先支付深度匹配成本；Auto 且 `allow_model_downloads=false` 时不会为了自动升级而隐藏下载模型权重。LightGlue、LoFTR、MangaLens、Torch 超分整册常驻复用。完成页可通过 job fingerprint 断点跳过，配准/OCR/气泡结构另有阶段缓存。

用户真实同源测试页上，v0.6 快速配准 5 次中位约 `0.016 s`，强制 SIFT 中位约 `1.083 s`，单配准约 `67.5×` 加速；最终图与 v0.5 逐像素一致，仍为 5/5 替换、QA 0/0。详见 `docs/PERFORMANCE_V06.md` 与 `docs/SELF_ACCEPTANCE_V06.md`。

### v0.5 真实成对页面验收

使用一组同页“日文原图 / 已正确嵌字中文图”进行端到端回归：SIFT 配准置信度约 `0.9998`，自动找到正好 5 个真正发生翻译变化的区域（4 个对白气泡 + 1 个说明文本框），顶部未翻译小气泡与画面拟声词不会被误选。最终替换蒙版内与中文参考逐像素一致，蒙版外与日文母版逐像素一致，QA 为 0 error / 0 warning。详见 `docs/REAL_PAIR_ACCEPTANCE_V05.md`。

### 蒙版替换 GUI

工作台右侧可直接切换 `高清重排 / 蒙版替换 / 智能混合`。v0.5 新增同源页面 `paired-diff` 快速路径：先通过页面配准定位真正发生翻译变化的气泡/文本框，同源页面可进行像素级精确覆盖；跨版本页面仍使用气泡实例匹配、局部 ECC、可选超分、Mask IoU 与目标覆盖率安全门槛。


## v0.8 手机翻拍旧版：光照归一化 + OCR 完整性保护

使用真实 `2400×3650` 手机翻拍中文版与 `850×1200` 日文母版做 5 页回归后，摄影版处理被单独拆成 `photo_pair` 路线：配准仍使用高清日文页作为几何真值，但不再依赖原始像素差；中文照片先做透视/尺寸映射，再在白色气泡内进行**确定性光照归一化**，去除反光、灰底和色温漂移，同时保留旧译文字形的真实抗锯齿。归一化后仍偏软才进入墨迹重建；极小字、开放爆炸框、覆盖不足或直接层漏检区域自动交给 OCR 高清重排。

Mac Studio 默认优先 Apple Vision OCR；摄影页固定 `safe_to_skip_ocr=False`，所以即使直接蒙版层已经成功若干气泡，也不会因此跳过 OCR。零候选/无 OCR 证据会产生 blocking QA error，避免把残缺页静默当成成品。5 页真实压力测试的配准 confidence 为 `0.867–0.890`；关闭 OCR 仅测直接层时 27 个安全候选中 21 个直接应用，其余明确转 OCR/几何拒绝。详见 `docs/PHOTO_PAIR_V08.md`。

## v0.8.1 合并版：保留 v0.8 结构差分回退

本合并包以 v0.8.1 为主体，不回退 `photo_pair`、摄影页 OCR 完整性保护、源分辨率保留和出版阻断 QA；同时恢复 v0.8 的结构墨迹差分、低频 DIS 局部光流、enclosed barrier 气泡检测、free-text/SFX 检测和 target-driven transfer。默认只在 v0.8.1 的高噪声摄影回退未能找到安全候选时启用 `structural_v08`，避免与新版 `photo_pair` 争抢同一区域。

兼容开关为 `mask_replace.paired_diff_structural_fallback_enabled`；旧版细粒度阈值字段也继续接受。结构回退始终保持 OCR，不会绕过 v0.8.1 的摄影页完整性保护。

## v0.7 低清中文文字保护

针对手机拍摄、反光、失焦和旧扫描版，蒙版替换新增 **Pixels → Ink reconstruction → OCR re-letter** 自动清晰度门控。模糊源文字不再被静默贴到高清母版；生成式图像模型不参与中文文字绘制。配准可靠时，旧中文版会先按高清页几何矫正后再做 Apple Vision / PP-OCRv5 OCR，并对低置信度块尝试 CLAHE、锐化和自适应二值版本。详见 `docs/TEXT_FIDELITY_V07.md`。

## v0.8.2：摄影版高清文字重建与小气泡完整替换

本版继续以 v0.8.1 + v0.8 合并版为基础，重点解决“替换后文字发糊”和摄影版小气泡被直接拒绝的问题。新增 `photo-crisp-ink` 路径：不再把手机照片里的灰底、眩光和模糊文字块原样贴进高清页，而是从已配准的源图中提取真实中文暗部细节，生成抗锯齿中性墨迹，再覆盖到高清日文页的干净纸面。该路径不依赖 OCR，也不会重新生成字符。

主要变化：

- 摄影版默认启用 `photo_pair_crisp_text_enabled`，清除相机灰雾、色偏和采样模糊；
- 通过 `photo_pair_crisp_border_guard_px` 排除源气泡边框，避免替换后出现“双重气泡轮廓”；
- 取消小于 88px 的摄影气泡硬拒绝，优先执行高清墨迹恢复；
- 对轻微欠分割的摄影气泡加入 1–3px 源掩膜扩张 salvage；
- 摄影页目标几何门槛调整为 target-driven 安全策略：IoU 0.74、coverage 0.84、spill 0.27；最终写入仍受高清目标掩膜约束；
- 当初始页面配对分数较低、但 SIFT/仿射配准高置信且所有摄影候选均成功替换时，页面配对 QA 降为 warning，不再误阻断；
- 无 OCR 时仍保留“可能漏掉开放式气泡/SFX”的 QA warning，但完整成功的闭合气泡替换不再返回失败退出码；
- 本版回归测试为 **29/29 通过**。

实图 `009` 复测：8 个摄影候选全部替换，`8/8 applied`，QA `0 error / 2 warning`；文字均走 `photo-crisp-ink`，不再复制摄影灰底或模糊像素。

## v0.8.3：源照片裁字保护，禁止“残缺中文假成功”

真实 `009` 的右上气泡暴露了一个重要完整性问题：中文版手机照片在画面右边缘已经把气泡和部分中文字符物理裁掉，但 v0.8.2 仍以约 `0.864` 的 target coverage 接受该区域，随后清空整块高清日文气泡，造成“只贴进去一部分中文，却显示替换成功”。

v0.8.3 新增摄影源边缘完整性门控：如果源气泡掩膜触碰照片边缘，就不能只使用普通 `coverage=0.84` 门槛，而必须达到 `photo_pair_edge_clip_min_target_coverage=0.94`。达不到时程序保持高清目标气泡原样，并输出 blocking QA `mask_replace_source_translation_clipped`，同时在 `mask_transfer.json` 写入 `source_edge_clipped` 和 `source_edge_sides`。这条规则只针对实际不完整的边缘源；本页右下同样触碰照片右边缘的「哼！」覆盖率约 `0.9889`，仍会正常替换，不会被误杀。

真实回归：`009` 现在为 7 个安全区域直接替换 + 1 个明确边缘裁切拒绝，不再出现残缺中文假成功；`007` 保持 4/4 成功。自动测试 **33/33** 通过。

## v0.8.5：摄影主路线 + 结构差分补漏 + GUI 待补字

v0.8.5 继续以 `photo_pair` 作为摄影旧版的主路线，同时把 v0.8 的结构差分限制为**非重叠补漏器**：只补主路线没有覆盖的开放爆炸框 / free text，并过滤邻近已确认气泡、页脚、低亮度和变化不足区域。补漏区域同样使用 `photo-crisp-ink`，不会直接发布模糊摄影像素。

普通对白在 OCR 可用时优先进入高清 reletter；排版方向会参考源 OCR 文字块几何，圆形气泡中的竖排中文不再仅由气泡长宽比决定。

对于源照片边缘物理裁掉的译文，替换工作台新增“待补文字”：选择待补气泡、输入完整中文、选择自动/竖排/横排后，可直接生成高清复核图，不必手工编辑 JSON。裁切源仍然不会被旧结构差分绕过。

彩色拟声词形状迁移保留为实验能力，默认关闭；真实摄影页上如果字形阴影重、源图本身模糊，强行迁移可能降低质量，因此正式默认策略优先保证成品清晰和可复核。

## v0.8.6：先给中文候选 + 模糊标注 + 可编辑/可还原

v0.8.6 将摄影旧版的失败策略改成 **review-first**：只要源中文版仍有可恢复中文字形，就先输出中文候选，而不是直接把高清日文留在最终页。候选不会被当作出版通过结果，会在 `review_preview.png` 和替换工作台中标注为“可能不完整/不准确”或“可能模糊/扭曲”，并支持接受、重新编辑高清排字或一键还原日文。

同时，摄影页局部几何不再使用可能压扁中文字形的 X/Y 独立 bbox 拉伸；默认使用等比例局部拟合。普通 OCR 重排则使用 4× 超采样字体渲染后一次缩回目标尺寸，改善小字号清晰度。

## v0.8.7：黑白中文版 → 彩色母版跨版本替换

v0.8.7 增加跨色彩版本路由。程序检测到低饱和/黑白中文版与彩色目标母版时，会优先信任高清目标页的气泡几何，不再因为两个版本的气泡尺寸、留白或排字位置不同而直接拒绝。对于目标页中可靠的白色文字容器，只要已配准源图和目标图都存在文字墨迹且差异足够，就可以先生成可复核的中文结果。

小气泡会从目标气泡附近恢复完整的中文字形块，过滤气泡边线和长画面线条后，以单一等比例缩放重新居中；同时清除目标 mask 边缘遗漏的旧日文字形和抗锯齿灰边。普通彩色摄影页仍保持 v0.8.6 的路线，不会因为跨版本增强而被强制切换。
.2 起此行为已收紧**：需要保留 TARGET 彩色/纹理的区域归入 Mask/Auto fallback，显式 Direct 不再使用这条彩色覆盖语义。
- 加入 artwork guard：人物皮肤、头发、建筑等即使是彩色区域，也必须同时通过源白底、紧凑文字、边界对齐和中日墨迹变化门槛，避免把画面误当彩色对白框。
- 高置信同版页命中快速路线后直接跳过 paired-diff、OCR、气泡双边匹配和重复 completion，减少整页处理时间。
- 使用用户提供的 `014.jpg` + `p-014.jpeg` 实测：自动选中 6 个真实对白/尖角框（3 个普通白框 + 3 个彩色尖角框），未把人物/建筑误覆盖。

## v0.8.28 — 整气泡覆盖强化 / 白色爆发框 / 页面级性能优化

- **整气泡 / 白底文本框覆盖现在在 GUI 中有明确开关，默认开启。** 白色容器优先复制旧中文版的完整白底 + 中文字补丁，不拆字、不 OCR 重排。
- 新增 **白色爆发框 / 锯齿对白框** 安全路线：允许低 bbox 填充率的星爆/锯齿白框进入整容器覆盖，但仍要求中日两页都满足白度、文字墨迹、饱和度、长宽比和配对一致性。
- 修复实心容器补洞对锯齿爆发框的潜在破坏：低 solidity 的星爆形状不再使用大核 closing / convex-envelope 补洞，只填真正的内部文字孔洞，保留每个尖角和凹槽。
- 页面级性能优化：source/target 的 Gray/HSV 转换在整页只计算一次，气泡越多收益越明显；图层写入也不再为每个气泡整页执行一次 BGR→RGB 转换。
- 真实 006 / 065 / 066 / 007 再次回归，QA 均为 0 error；006 小对白继续走完整 source container patch。

## v0.8.27 — 整个中文版气泡 / 文本框覆盖

- 白色气泡、白底旁白框和白底文本框默认使用 `rigid-container-full-patch`：直接迁移旧中文版的白底 + 中文字整体补丁。
- 目标容器仅提供位置与边界；中文字不使用 X/Y 非等比缩放，不通过 OCR 重排。
- 目标日文文字被完整白底补丁覆盖，避免“中文字已经写入但日文仍残留”的混合结果。
- 原 `rigid-container-raster` 仍作为兼容回退路径。

## v0.8.26 — 实心气泡内部 / 日文残留修复

- 修复白色气泡 detector mask 被日文字切出黑缺口后，清除阶段反而跳过日文字的根因。
- 检测 mask 与真正的容器内部 mask 分离：闭合窄缺口、填内部孔洞，并保留边界保护带。
- `clear mask` 与中文字栅格裁切都使用实心容器内部，避免日文残留与中文缺笔。
- 065 的爆发式黑色气泡边缘通过 convex-envelope boundary guard 保留，不会因补洞被刷白。
- OCR-free 白色容器补漏提高文字密度门槛，减少建筑/天空白块误判。
- 真实 006 小对白已重新验证：原日文清空，中文完整，QA pass。

## v0.8.25 — 白色容器补漏与独立清除蒙版

- 同版黑白中文 → 彩色日文页新增 OCR-free 白色容器补漏。目标气泡/文本框作为几何真值，反向映射到原始中文页取完整中文字形，避免源扫描气泡连到纸面或被翻译组扩框时被错误拒绝。
- 中文最终字形继续严格 `uniform scale + translation`，不继承页面 affine/dense-flow 的非等比变形。
- 工作台新增 `清除蒙版 / 中文迁移层 / 只清日文` 三个独立视图，以及可画笔增删的清除蒙版 overlay 编辑器。
- `仅执行去字` 不重跑配准、OCR、检测或中文字形迁移。
- 可独立选择气泡 detector、检测分辨率、清除 mask 扩张和 inpainting backend。
- 006 / 065 / 066 用户真实页均重新回归；006 右下角此前漏掉的对白已由自动路径恢复。


## v0.8.23 锁定栅格气泡迁移：位置可以仿射，字形绝不仿射

本版重新定义黑白旧中文版 → 彩色/高清日文页的最终写入规则：**页面配准负责找位置，中文字栅格只允许单一等比缩放 + 平移**。全局 affine / homography、局部 dense-flow 都不能再直接作用到最终 CJK 字形。

- 对白色对白气泡、旁白框、白底文本框，目标高清页的完整容器是清除边界：保留目标轮廓/尾巴，清空内部日文，再从原始旧中文版取得整块中文字栅格。
- 字符和标点不拆成 connected components 重组；整块灰度/抗锯齿栅格作为一个连续 alpha 场迁移，避免缺笔、断标点和字符重心漂移。
- 分辨率不同只允许一个 scalar scale；禁止 X/Y 独立 resize、shear 和把页面 affine 的非等比校正传给字形。
- 结构候选只用于发现区域；若目标 mask 恰好切到日文字边缘，只清除字符尺寸的残余组件，不进行可能泄漏到画面的白区 flood-fill。
- 黄色/彩色爆炸框保留目标彩色背景，只迁移中文墨迹，不复制黑白版的爆炸框底色/线稿。
- 复杂/开放文字在真正写入最终页之前先做内容完整性检查；检查失败就不发布该候选，避免出现“写入成功但字残缺”的半成品。
- 065 / 066 两组真实图在 OCR=none 下重新回归：065 为 5/5 内容完整，066 为 4/4 内容完整；已检测区域目标日文残留率为 0。


## v0.8.22 真实 065/066 验证：内容完整性与跨版本复杂文字

本版针对黑白旧中文版 → 彩色/高清日文原页的真实 065、066 配对做了回归，不再把 `applied=true` 当成“翻译完整”。主要变化：

- **几何写入与内容完整性分离**：每条替换记录新增 `content_check`、`source_ink_coverage`、`target_residual_ratio`、`content_complete`。跨版本白气泡、开放文字、复杂文字、OCR 几何兜底若内容未验证/不完整，会由 QA 阻止作为成功页。
- **跨版本白气泡候选守门**：黑白→彩色时用高白度、填充率、内部文字墨迹和边界证据收紧气泡，防止列车车身、招牌、浅色画面块被当成对白框。
- **彩色爆炸框独立路线**：高饱和黄色/红色文字框必须与结构候选有足够真实重叠，并要求注册后的中文源区域主要是亮底，避免邻近蓝色 SFX/招牌误触发。只擦目标文字笔画，只写源中文笔画，不复制黑白底纹。
- **结构局部流只负责检测**：结构补充仍可用局部/稠密流发现变化，但最终中文字统一优先取全局注册源，避免局部光流扭曲 CJK 字形、把爆炸框边线拖入最终图。
- **`transfer_audit.v2`**：同时记录 `geometry_applied`、`content_checked`、`content_complete`、`content_incomplete`、`content_unverified`、最差源墨迹覆盖率和最大目标残留率。无 OCR 时明确标记验证范围为 `detected_regions_only_no_ocr`，不假装做了整页语言识别。
- **真实样本**：065 共 5 个区域（3 白气泡 + 2 黄色爆炸框）全部内容验证通过；066 共 4 个区域（列车小气泡、右侧气泡、中间气泡、左下大气泡）全部内容验证通过，未再修改右侧白色招牌/画面块。

## v0.8.21 复杂文字 / 漏检恢复 / 逐页迁移审计

- 精准蒙版增加 **复杂文字区域** 路线：黄色爆发框、开放式文字、彩色背景文字不再被“近白底”门槛直接丢弃；必须同时满足源/目标两侧紧凑文字墨迹、结构差异和面积上限，最终仍只搬运源中文栅格墨迹，不做 OCR 重排。
- 成对差异的闭合气泡检测之外增加结构补充：raw-diff 没有建立容器时可回退到 structural-v08；raw-diff 已找到普通气泡时也可补充不重叠的 free/complex text。低置信度文本不再静默丢弃，而是生成可恢复、可编辑的复核候选。
- 新增 **OCR 几何引导组件迁移**：OCR 只用于确认“源中文区域 ↔ 目标日文区域”的对应和候选范围；真正清除的是目标日文字形组件，真正写回的是配准后的源中文像素。`strict_mask_replace_no_ocr_reletter=true` 时绝不拿 OCR 字符串重打字。
- 目标清除蒙版允许窄长竖排日文列和横向标题组件跨出不完整气泡内框；仅对经过组件形状约束的目标文字先行清除，避免残留半截日文，同时保护气泡轮廓、格线和彩色背景。
- 每页新增 `transfer_audit.json`、`source_original.png`、`target_clear_mask.png`、`chinese_transfer_layer.png`，并保证 `review_preview.png` 总是存在；可直接核对配准、候选类型、OCR 证据、应用/拒绝/复核数量、清除像素和 QA。
- 修复 raw-diff 中重复执行一次源白区组件搜索的隐藏性能问题；精准蒙版 0.8.16 的 glyph-footprint rescue 保持不变。
- 页面管理/跳过逻辑保持 v0.8.20：所有新页默认正文，不做配对后自动气泡扫描；无中文 OCR 文本的正文可原样输出高清日文页；手动跳过只跳过替换，不从最终整册删页。

## v0.8.20 默认正文 / 按需替换 / 缩略图性能修复

- **删除配对后的自动气泡/文本框检查**：新配对页面一律默认为“正文 / 需替换”，页面类型只由用户手动分类；v0.8.19 保存的 `auto_no_text` 自动标记会迁移回默认正文。
- “正文”现在表示**允许进入迁移**，不再表示“必须改图”。实际处理时若中文源页 OCR + 气泡结构确认没有可迁移的 speech/narration 文本框，会直接输出未修改的高清日文页。精准蒙版模式仍禁止 OCR 改写原中文字形。
- 手动标记为封面、扉页、目录、单话首页、插图、空白页或“跳过”的页面仍是完整输出页：只跳过替换步骤，不从 `final/` 和整册页序删除。
- 缩略图改为**可视区域懒加载 + 滚动静默期加载 + 128 张缩放图 LRU 缓存**。滚轮/触控板滚动时不再后台持续解码整本漫画，停止滚动后才分小批加载当前视口及相邻一行。
- 切换页面类型不再重新解码已经缓存的缩略图；“刷新”才主动清空缩略图缓存。
- 整册最终输出文件名增加同名保护，避免极端情况下两个目标页同 stem 时互相覆盖。

## v0.8.19 可视化页面管理

- 页面管理默认改为 **缩略图画廊**：每一页直接显示高清日文页面缩略图、页码、页面类型、手动/自动来源和类型色标；可切换为旧版中文缩略图，不再靠文件名猜封面/目录/插图。
- **双击任意缩略图或列表行**打开可缩放大图窗口，旧版中文与高清日文左右并排；支持上一页/下一页、滚轮缩放、拖拽、适合窗口和 100% 查看。
- 缩略图支持 Ctrl / Command / Shift 多选，右键可直接批量标记正文、封面、扉页、目录、单话首页、纯图片等，也可恢复自动判断或只检查所选页。
- 新增“日文缩略图 / 中文缩略图”切换、页面筛选和当前选择计数；右侧详情同步显示当前页大预览、分类原因、气泡/自由文字数量、配对方法和置信度。
- 大图窗口的分类状态刷新不再重复解码两张全分辨率图片，避免批量处理/自动检查时因 UI 状态更新造成卡顿。
- `page_management.json` 改用同目录临时文件 + `fsync` + 原子替换保存；强制退出或写盘异常时不会把已有页面分类文件写成半截 JSON。

## v0.8.18 页面管理 / 安全停止

- 顶部加入始终可见的红色 **“■ 停止”** 按钮，页面自动检查、单页处理和整册处理共用同一套安全取消信号；单页 Pipeline 也在配准、成对差异、OCR/气泡、迁移和导出等安全边界检查取消。
- “页面”重构为 **页面管理**：每个已配对页面都有独立处理准入状态，可批量标记为正文、封面、扉页/书名页、目录、单话首页/章节页、插图/纯图片、卷首插画、空白页、后记/版权/广告或手动跳过。
- 配对后默认运行 OCR-free 页面检查：只使用下采样 OpenCV 配准 + paired-diff 几何，不读取文字内容；高置信度且没有气泡/文本框的页面自动标为“无气泡/文本框（自动）”并跳过迁移。检测失败或配准不足时**绝不自动丢页**，保持进入正常处理。
- 跳过页仍保留在整册页序中，并把高清日文原页原样写入 `final/`；不会进入 OCR、气泡迁移、清字或重排。手动“正文 / 需替换”可强制恢复处理。
- 页面标记持久化到输出目录 `page_management.json`，重新打开项目后继续生效；断点续跑也服从最新页面标记，避免把旧的已处理结果错误恢复到后来标记为封面/插图的页面。
- 页面类型使用不同颜色，表格同时显示“处理/跳过、页面类型、配对方法、状态和路线”，并支持多选批量标记、恢复自动、只检查所选页。

## v0.8.17 工作台同步与 UI 修复

- **替换工作台统一页状态**：日文原图、旧中文版、最终结果、复核标注和替换蒙版都从同一个当前页解析，不再复用“上一张最后结果”。
- 工作台新增 **上一页 / 下一页 / 当前页计数**，最终结果可连续翻页；当前页没有结果时明确显示为空，不会误显示其它页面。
- 整册处理完成后，每页工程会进入按页索引；同时可直接从 `pages/<page_id>/project.json` 和 `final/` 恢复预览，断点续跑页也能正常浏览。
- 复核后的 `final_reviewed.png` 优先显示，并同步回整册 `final/` 对应页，避免预览结果和出版输出不一致。
- “优先名称 / 页码配对”和“优先文件夹自然顺序”默认关闭，并增加明显间距与说明；默认走智能视觉配对。
- 修复 CJK 文件名页 ID：日文/中文纯文字文件名不再全部坍缩成同一个 `page` 目录；同时兼容 v0.8.16 单页处理留下的旧目录。
- UI 继续参考 KCC-Kindle-CHS 的卡片化、双栏、右侧滚动设置区和紧凑 macOS 控件设计，但不改动迁移核心信号与处理流程。

## v0.8.11 排版质量策略

默认采用“**OCR 认字，源图定排版**”：Apple Live Text 成功并不意味着一定重新排字。旧中文版迁移后的字形已经清晰时，程序保留原字号、分列和相对位置；只有模糊/低清/不安全区域才交给 OCR 高清重排。这样避免真实 Mac 批处理里出现巨大字体、分列错乱、白块和空白气泡。

把**旧版低清中文汉化漫画**中的既有译文可靠迁移到**高清日文原图**，完成跨版本页面配准、中文 OCR、日文清字、气泡安全区排版、自动 QA、人工复核与分层导出。

项目不是机器翻译器。它不重新翻译日文；中文内容来自你已有的旧汉化版。

## 目标

输入：

- `source_cn/`：旧版中文汉化图，允许低清、压缩、扫描偏移、裁边、额外 staff 页。
- `target_jp/`：同一作品的高清日文图，作为唯一画面母版。

输出：

- `final/`：自动通过安全门槛的最终 PNG。
- `pages/.../target_original.png`：高清母版，永不覆盖。
- `inpainted.png` / `clear_mask.png`：清字结果与精确 mask。
- `text_layer.png`：透明中文文字层。
- `editable.ora`：OpenRaster 分层工程。
- `editable.psd`：系统有 ImageMagick 时自动生成分层 PSD。
- `project.json`：配准矩阵、OCR、气泡、匹配、排字、置信度和可追溯信息。
- `qa.json` / debug overlays：出版 QA 和视觉证据。
- 本地 Review 编辑器：可改译文、改目标匹配、画/擦清字 mask，再重新生成。

## 核心设计

```text
旧中文版 ─ 页面指纹/顺序配对 ─┐
                            ├─ cheap-first / SIFT / LightGlue / LoFTR + RANSAC 配准
高清日文版 ─ 页面指纹/顺序配对 ┘

旧中文版 ─ 中文 OCR / 多次低置信度复识 ─ 中文 TextUnit
高清日文版 ─ 日文文字区域 + 气泡实例 / safe mask ─ TargetUnit

TextUnit + Registration + TargetUnit
              ↓
      跨版本最小成本身份匹配
              ↓
      日文像素级 text mask 清字
              ↓
  solid / OpenCV / external LaMa 修复
              ↓
  中文约束排字（字号、断行、禁则、安全区）
              ↓
       QA 安全门槛 + Review Queue
              ↓
       PNG / ORA / PSD / JSON
```

## 五种译文迁移模式

### 1. 自动 `auto`（默认）

先做 OCR-free 同页预检和页面配准。若页面属于同版同布局、容器可安全直接迁移，则走 `direct_patch`；Direct 不满足条件时才进入 `mask_replace`。这条路线用于批量处理时自动“先便宜、后复杂”，避免所有模型全跑。

### 2. 直接贴图 `direct_patch`

这是和蒙版迁移**语义完全不同**的一条路线，专门针对完全一致或高度一致的两张图。SOURCE 的完整气泡/白底文本框内部被当作一个原始栅格 patch，**白色背景与中文文字一起扣出、一起对齐、一起覆盖**。

- identity 页面优先 1:1 覆盖；尺寸/裁边略有变化时只允许 local similarity（统一缩放 + 小旋转 + 平移）。
- 页面 affine/homography 只负责“找到 TARGET 位置”，不会把中文字形做 X/Y 拉伸或透视变形。
- 不调用 OCR，不重新输入中文，不清 TARGET 日文，不 inpaint，不走 target-aware 背景重建。
- 显式 `direct_patch` 只要同页预检、配准或容器完整性不够安全，就拒绝该页/区域并保留 TARGET；**绝不静默切换到蒙版或 OCR**。
- 彩色/网点/渐变容器如果需要保留 TARGET 原纹理，应选择 `auto` 或 `mask_replace`，而不是 Direct。

### 3. 精准蒙版迁移 `mask_replace`

蒙版模式不是“整块贴图”的别名。它以 `geometry_mask` 表示容器几何，以独立的 transfer/clear mask 表示真正允许修改 TARGET 的区域：先保护边框和画面，再在安全区域内清除 TARGET 日文、迁移 SOURCE 中文 ink / interior / artwork。

对于白色普通气泡可以迁移 SOURCE interior；对于彩色、网点、渐变容器则优先保留 TARGET background/texture，只清日文墨迹并迁移中文。局部几何、内容完整度、spill、边框保护等 QA 不满足时进入 Review。

### 4. 智能混合 `hybrid`

优先使用蒙版迁移保留旧汉化排字；某个区域无法安全完成时，允许退回 OCR → 清字 → 高清重排。适合确实需要传统 fallback 的跨版本页面。

### 5. 高清重排 `reletter`

旧中文版提供中文文本内容，高清日文版负责清字与重新排版。适用于 SOURCE 字形本身太糊、需要统一出版级字体的情况。

CLI：

```bash
mhd-transfer run source_cn target_jp output --mode auto
mhd-transfer run source_cn target_jp output --mode direct_patch
mhd-transfer run source_cn target_jp output --mode mask_replace
mhd-transfer run source_cn target_jp output --mode hybrid
mhd-transfer run source_cn target_jp output --mode reletter
```

Direct / Mask 的边界和回归验收见 `docs/DIRECT_VS_MASK_V0834.md`；蒙版算法细节见 `docs/MASK_REPLACE_PLAN.md`。

### 与简单“替换翻译”方案的关键区别

1. **先做视觉配准，再匹配文字身份**，不依赖整页 resize 后固定 IoU。
2. **Direct Patch** 迁移 SOURCE 的整块原始栅格；**蒙版迁移**在独立 clear/transfer mask 内做 TARGET-aware 合成；**高清重排**才迁移文本内容并重新渲染，三种语义严格分离。
3. **清字 mask 与排字 safe area 分离**：前者只描述要删除的日文，后者描述中文允许出现的位置。
4. 气泡边界有保护带；中文最终字形 mask 必须通过 safe-area 覆盖验证。
5. 低页面配对、低配准、低 OCR、低身份匹配、拆分/合并关系都会阻止自动覆盖，进入 Review。
6. 每页有完整 evidence/debug，不做不可追溯的黑箱整页重绘。

## 安装

Python 3.11+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

核心安装只需要 OpenCV / NumPy / Pillow / SciPy / Pydantic / Typer，不下载 OCR 或深度模型。

### 中文 / 日文 OCR

```bash
pip install -e '.[ocr]'
```

默认显式使用 `PP-OCRv5`。PaddleOCR 的模型可能在首次实际推理时下载。若当前平台不方便运行 Paddle，也可使用 `sidecar`，把任何外部 OCR/漫画文字检测器的结果接进来。

### LightGlue / LoFTR 增强配准

```bash
pip install -e '.[lightglue]'
```

`registration.backend=auto` 采用 cheap-first：同源快速结构验证 → OpenCV SIFT/ORB → 本地已有权重时 LightGlue → LoFTR。Auto 默认不允许隐藏下载权重。Apple Silicon 可用 `首次安装_MPS_AI加速.command` 安装 MPS 推理运行库；模型权重仍由用户显式准备。

### LaMa

项目不复制/绑定 LaMa 模型。把任意可用 LaMa wrapper 配成：

```json
{
  "inpainting": {
    "backend": "lama",
    "lama_command": "python lama_wrapper.py --input {input} --mask {mask} --output {output}"
  }
}
```

## 首次运行

```bash
mhd-transfer doctor
mhd-transfer init-config config.json
mhd-transfer run source_cn target_jp output --config config.json
```

如果使用外部 OCR sidecar：

```bash
mhd-transfer run source_cn target_jp output --ocr-backend sidecar
```

运行结束若存在出版阻断级 QA，CLI 返回非 0，并提示进入 Review：

```bash
mhd-transfer review output
```

浏览器打开三栏编辑器：旧中文版 / 高清日文与清字 mask / 当前输出。可以：

- 修改 OCR 中文译文；
- 改某个中文 TextUnit 对应的目标气泡/文本框；
- 勾选是否应用该文本；
- 画笔增加或擦除清字 mask；
- 保存并点击“应用复核并重新生成”。

也可以命令行应用：

```bash
mhd-transfer apply-review output/pages/0001_xxx
```

## v0.6 批量与 Apple MPS

- 整册处理支持断点续跑、失败页继续、实时进度和安全取消；
- 配准/OCR/气泡实例按输入+配置做本地阶段缓存；
- LightGlue、LoFTR、MangaLens 和 Torch 超分模型整册常驻，不再每页重复初始化；
- Apple Silicon 上可选择 `mps`，MPS GPU 推理受控串行，CPU 图像预处理继续使用受限线程；
- 同源页面先跑约缩略图级的结构/phase 快速验证，只有不确定页才支付 SIFT/深度模型成本；
- 打开 GUI 的模型页不会因此 import 大型模型或下载权重。

详细见 `docs/PERFORMANCE_V06.md`。

## Sidecar 接口

### OCR / 漫画文字分割

`page.png` 对应 `page.ocr.json`：

```json
{
  "blocks": [
    {
      "id": "b0",
      "text": "已经存在的中文译文",
      "confidence": 0.99,
      "polygon": [[100,100],[260,100],[260,180],[100,180]],
      "kind": "speech",
      "mask_path": "masks/b0.png"
    }
  ]
}
```

如果提供 `mask_path`，清字优先使用像素级 segmentation；没有时才回退到 polygon。

### 气泡 / 旁白框实例分割

`page.bubbles.json`：

```json
{
  "bubbles": [
    {
      "id": "bubble-0",
      "kind": "speech",
      "confidence": 0.99,
      "polygon": [[80,70],[300,70],[300,220],[80,220]],
      "mask_path": "bubbles/0.png",
      "safe_mask_path": "bubbles/0_safe.png"
    }
  ]
}
```

没有 `safe_mask_path` 时会自动向内腐蚀实例 mask，窄尾部通常会自然脱离排版安全区。

## 自验收

完全离线、无模型下载：

```bash
pytest
mhd-transfer selftest
python benchmarks/synthetic_acceptance.py
```

本次实现时结果：

- v0.8.1 合并版单元/集成测试：**25/25 通过**，新增 cheap-first 配准、断点续跑与阶段缓存测试。
- 内置端到端 selftest：**通过**，无 error/warning。
- 20 组随机几何扰动配准：**20/20 通过**。
- 控制点中位误差：约 **0.051 px**；P95：约 **0.117 px**。
- 4 组不同长度中文气泡排字：**4/4 通过**，字形安全区覆盖均 ≥ 0.997。
- 本机层导出验证：ORA 成功；ImageMagick 存在时 PSD 成功。

详见 `docs/SELF_ACCEPTANCE.md`。

> 合成验收不能替代真实出版数据验收。最终“出版级”门槛仍要求建立同作品的 100–300 页人工真值集，统计页面配对、区域身份匹配、残留日文、误伤线稿、排字越界和人工复核率。程序已把这些证据与 QA 接口保留下来。

## 目录

```text
src/manga_hd_transfer/
  pairing.py          页面指纹与顺序配对
  registration.py     SIFT / LightGlue / LoFTR + RANSAC
  ocr.py              Paddle、sidecar、低置信度复识
  bubbles.py          气泡实例 / safe area / TextUnit
  matching.py         跨版本身份匹配、拆分/合并检测
  masking.py          像素级/多边形日文清字 mask
  mask_transfer.py     旧中文气泡/文本框 patch 对齐、超采样与蒙版替换
  inpainting.py       solid / OpenCV / external LaMa
  lettering.py        中文约束排字
  qa.py               出版 QA
  pipeline.py         全流水线
  review.py           本地人工复核编辑器
  review_apply.py     Review 回写与重生成
  export.py           PNG / ORA / PSD layer export
```

## 参考与许可

v0.8.32 已将部分经许可的关键上游源码/摘录放入 `third_party_reference/` 作隔离参考；运行时适配状态见 `INTEGRATION_STATUS_v0.8.32.md`，GPL 参考文件不从 MIT 核心导入。

- `hgmzhn/manga-translator-ui`：Replace Translation 业务原型。
- `zyddnys/manga-image-translator`：漫画翻译流水线。
- `dmMaze/BallonsTranslator`：编辑器、mask/inpaint/排字工作流。
- `dmMaze/comic-text-detector`：漫画文本框、文本行、segmentation；当前通过 pixel-mask sidecar 接入。
- `ogkalu2/comic-translate`：气泡检测与文本分割分层、PPOCR/LaMa 模块化。
- `PaddlePaddle/PaddleOCR`：PP-OCRv5 直接后端。
- `cvg/LightGlue`：局部特征匹配直接可选后端。
- `zju3dv/LoFTR`：困难图像匹配直接可选后备。
- `advimman/lama`：复杂背景 external command 后端。
- `facebookresearch/sam2`：v0.8.32 已接入 SOURCE-only 可选分割 provider；默认不下载模型。
- `huyvux3005/manga109-segmentation-bubble`（MangaLens）：本版新增 Ultralytics 本地模型直接后端。
- `ScanR/TypeR`：自动居中、行距联动、样式预设与紧凑排字工作流参考。

精确状态见 `docs/INTEGRATION_STATUS.md`，不要把“参考”与“直接调用”混为一谈。

---

## macOS Studio GUI v0.6（KCC 风格淡蓝精简版 + 批量/MPS）

本地 ZIP 版继续使用 `Manga HD Transfer Studio` PySide6 GUI：低饱和淡蓝、白色卡片、细边框、紧凑布局，并吸收 KCC-Kindle-CHS 的后台 QThread、进度/取消与资源清理方式：

- 页面管理：成对导入旧中文/高清日文，自动配对、批量状态、处理路线、断点续跑与安全取消；
- 配准/OCR：Apple Vision OCR、PP-OCRv5、cheap-first/SIFT/LightGlue/LoFTR、MangaLens 与 Sidecar；模型中心只做无副作用浅探测；
- 替换工作台：页面配准 → 中文气泡/文本框 → 实例匹配 → 局部精对齐 → 可选 MPS Torch 超分 → 蒙版替换/高清重排 → 出版 QA；
- Publication Builder：整册构建、输出统计、QA 状态。

Mac 推荐先双击 `启动_Manga_HD_Transfer.command`。需要 Apple Silicon AI 加速时再运行 `首次安装_MPS_AI加速.command`；该脚本安装运行库，但不捆绑漫画模型/超分权重。


### v0.6 降本增效

Auto 配准改成 `同源快速验证 → OpenCV SIFT/ORB → LightGlue/MPS → LoFTR/MPS`，不再默认先支付深度匹配成本；Auto 且 `allow_model_downloads=false` 时不会为了自动升级而隐藏下载模型权重。LightGlue、LoFTR、MangaLens、Torch 超分整册常驻复用。完成页可通过 job fingerprint 断点跳过，配准/OCR/气泡结构另有阶段缓存。

用户真实同源测试页上，v0.6 快速配准 5 次中位约 `0.016 s`，强制 SIFT 中位约 `1.083 s`，单配准约 `67.5×` 加速；最终图与 v0.5 逐像素一致，仍为 5/5 替换、QA 0/0。详见 `docs/PERFORMANCE_V06.md` 与 `docs/SELF_ACCEPTANCE_V06.md`。

### v0.5 真实成对页面验收

使用一组同页“日文原图 / 已正确嵌字中文图”进行端到端回归：SIFT 配准置信度约 `0.9998`，自动找到正好 5 个真正发生翻译变化的区域（4 个对白气泡 + 1 个说明文本框），顶部未翻译小气泡与画面拟声词不会被误选。最终替换蒙版内与中文参考逐像素一致，蒙版外与日文母版逐像素一致，QA 为 0 error / 0 warning。详见 `docs/REAL_PAIR_ACCEPTANCE_V05.md`。

### 蒙版替换 GUI

工作台右侧可直接切换 `高清重排 / 蒙版替换 / 智能混合`。v0.5 新增同源页面 `paired-diff` 快速路径：先通过页面配准定位真正发生翻译变化的气泡/文本框，同源页面可进行像素级精确覆盖；跨版本页面仍使用气泡实例匹配、局部 ECC、可选超分、Mask IoU 与目标覆盖率安全门槛。


## v0.8 手机翻拍旧版：光照归一化 + OCR 完整性保护

使用真实 `2400×3650` 手机翻拍中文版与 `850×1200` 日文母版做 5 页回归后，摄影版处理被单独拆成 `photo_pair` 路线：配准仍使用高清日文页作为几何真值，但不再依赖原始像素差；中文照片先做透视/尺寸映射，再在白色气泡内进行**确定性光照归一化**，去除反光、灰底和色温漂移，同时保留旧译文字形的真实抗锯齿。归一化后仍偏软才进入墨迹重建；极小字、开放爆炸框、覆盖不足或直接层漏检区域自动交给 OCR 高清重排。

Mac Studio 默认优先 Apple Vision OCR；摄影页固定 `safe_to_skip_ocr=False`，所以即使直接蒙版层已经成功若干气泡，也不会因此跳过 OCR。零候选/无 OCR 证据会产生 blocking QA error，避免把残缺页静默当成成品。5 页真实压力测试的配准 confidence 为 `0.867–0.890`；关闭 OCR 仅测直接层时 27 个安全候选中 21 个直接应用，其余明确转 OCR/几何拒绝。详见 `docs/PHOTO_PAIR_V08.md`。

## v0.8.1 合并版：保留 v0.8 结构差分回退

本合并包以 v0.8.1 为主体，不回退 `photo_pair`、摄影页 OCR 完整性保护、源分辨率保留和出版阻断 QA；同时恢复 v0.8 的结构墨迹差分、低频 DIS 局部光流、enclosed barrier 气泡检测、free-text/SFX 检测和 target-driven transfer。默认只在 v0.8.1 的高噪声摄影回退未能找到安全候选时启用 `structural_v08`，避免与新版 `photo_pair` 争抢同一区域。

兼容开关为 `mask_replace.paired_diff_structural_fallback_enabled`；旧版细粒度阈值字段也继续接受。结构回退始终保持 OCR，不会绕过 v0.8.1 的摄影页完整性保护。

## v0.7 低清中文文字保护

针对手机拍摄、反光、失焦和旧扫描版，蒙版替换新增 **Pixels → Ink reconstruction → OCR re-letter** 自动清晰度门控。模糊源文字不再被静默贴到高清母版；生成式图像模型不参与中文文字绘制。配准可靠时，旧中文版会先按高清页几何矫正后再做 Apple Vision / PP-OCRv5 OCR，并对低置信度块尝试 CLAHE、锐化和自适应二值版本。详见 `docs/TEXT_FIDELITY_V07.md`。

## v0.8.2：摄影版高清文字重建与小气泡完整替换

本版继续以 v0.8.1 + v0.8 合并版为基础，重点解决“替换后文字发糊”和摄影版小气泡被直接拒绝的问题。新增 `photo-crisp-ink` 路径：不再把手机照片里的灰底、眩光和模糊文字块原样贴进高清页，而是从已配准的源图中提取真实中文暗部细节，生成抗锯齿中性墨迹，再覆盖到高清日文页的干净纸面。该路径不依赖 OCR，也不会重新生成字符。

主要变化：

- 摄影版默认启用 `photo_pair_crisp_text_enabled`，清除相机灰雾、色偏和采样模糊；
- 通过 `photo_pair_crisp_border_guard_px` 排除源气泡边框，避免替换后出现“双重气泡轮廓”；
- 取消小于 88px 的摄影气泡硬拒绝，优先执行高清墨迹恢复；
- 对轻微欠分割的摄影气泡加入 1–3px 源掩膜扩张 salvage；
- 摄影页目标几何门槛调整为 target-driven 安全策略：IoU 0.74、coverage 0.84、spill 0.27；最终写入仍受高清目标掩膜约束；
- 当初始页面配对分数较低、但 SIFT/仿射配准高置信且所有摄影候选均成功替换时，页面配对 QA 降为 warning，不再误阻断；
- 无 OCR 时仍保留“可能漏掉开放式气泡/SFX”的 QA warning，但完整成功的闭合气泡替换不再返回失败退出码；
- 本版回归测试为 **29/29 通过**。

实图 `009` 复测：8 个摄影候选全部替换，`8/8 applied`，QA `0 error / 2 warning`；文字均走 `photo-crisp-ink`，不再复制摄影灰底或模糊像素。

## v0.8.3：源照片裁字保护，禁止“残缺中文假成功”

真实 `009` 的右上气泡暴露了一个重要完整性问题：中文版手机照片在画面右边缘已经把气泡和部分中文字符物理裁掉，但 v0.8.2 仍以约 `0.864` 的 target coverage 接受该区域，随后清空整块高清日文气泡，造成“只贴进去一部分中文，却显示替换成功”。

v0.8.3 新增摄影源边缘完整性门控：如果源气泡掩膜触碰照片边缘，就不能只使用普通 `coverage=0.84` 门槛，而必须达到 `photo_pair_edge_clip_min_target_coverage=0.94`。达不到时程序保持高清目标气泡原样，并输出 blocking QA `mask_replace_source_translation_clipped`，同时在 `mask_transfer.json` 写入 `source_edge_clipped` 和 `source_edge_sides`。这条规则只针对实际不完整的边缘源；本页右下同样触碰照片右边缘的「哼！」覆盖率约 `0.9889`，仍会正常替换，不会被误杀。

真实回归：`009` 现在为 7 个安全区域直接替换 + 1 个明确边缘裁切拒绝，不再出现残缺中文假成功；`007` 保持 4/4 成功。自动测试 **33/33** 通过。

## v0.8.5：摄影主路线 + 结构差分补漏 + GUI 待补字

v0.8.5 继续以 `photo_pair` 作为摄影旧版的主路线，同时把 v0.8 的结构差分限制为**非重叠补漏器**：只补主路线没有覆盖的开放爆炸框 / free text，并过滤邻近已确认气泡、页脚、低亮度和变化不足区域。补漏区域同样使用 `photo-crisp-ink`，不会直接发布模糊摄影像素。

普通对白在 OCR 可用时优先进入高清 reletter；排版方向会参考源 OCR 文字块几何，圆形气泡中的竖排中文不再仅由气泡长宽比决定。

对于源照片边缘物理裁掉的译文，替换工作台新增“待补文字”：选择待补气泡、输入完整中文、选择自动/竖排/横排后，可直接生成高清复核图，不必手工编辑 JSON。裁切源仍然不会被旧结构差分绕过。

彩色拟声词形状迁移保留为实验能力，默认关闭；真实摄影页上如果字形阴影重、源图本身模糊，强行迁移可能降低质量，因此正式默认策略优先保证成品清晰和可复核。

## v0.8.6：先给中文候选 + 模糊标注 + 可编辑/可还原

v0.8.6 将摄影旧版的失败策略改成 **review-first**：只要源中文版仍有可恢复中文字形，就先输出中文候选，而不是直接把高清日文留在最终页。候选不会被当作出版通过结果，会在 `review_preview.png` 和替换工作台中标注为“可能不完整/不准确”或“可能模糊/扭曲”，并支持接受、重新编辑高清排字或一键还原日文。

同时，摄影页局部几何不再使用可能压扁中文字形的 X/Y 独立 bbox 拉伸；默认使用等比例局部拟合。普通 OCR 重排则使用 4× 超采样字体渲染后一次缩回目标尺寸，改善小字号清晰度。

## v0.8.7：黑白中文版 → 彩色母版跨版本替换

v0.8.7 增加跨色彩版本路由。程序检测到低饱和/黑白中文版与彩色目标母版时，会优先信任高清目标页的气泡几何，不再因为两个版本的气泡尺寸、留白或排字位置不同而直接拒绝。对于目标页中可靠的白色文字容器，只要已配准源图和目标图都存在文字墨迹且差异足够，就可以先生成可复核的中文结果。

小气泡会从目标气泡附近恢复完整的中文字形块，过滤气泡边线和长画面线条后，以单一等比例缩放重新居中；同时清除目标 mask 边缘遗漏的旧日文字形和抗锯齿灰边。普通彩色摄影页仍保持 v0.8.6 的路线，不会因为跨版本增强而被强制切换。

本版继续按“**替换效果优先**”推进，重点吸收 `manga-translator-ui / BallonsTranslator / comic-text-detector / AutoScanlate-AI` 一类思路中**最容易直接提升出版级迁移效果**的部分：

- 新增 `replace_translation/` 语义层导出：每页会导出 `source_ocr.json`、`target_ocr.json`、`unit_matches.json`、`summary.json`，便于与 Replace Translation 工作流互通。
- 单元匹配吸收更多 soft prior：除中心距离和 overlap 外，新增 **projected IoU** 与 **text length ratio**，更接近“重叠率 + 几何成本”的混合匹配。
- 轻量多 SOURCE 证据：如果 `page.replace_sources.json` 提供候选的已汉化高清页，Direct Patch 在主 SOURCE 不安全时会尝试这些替代 SOURCE 做二次 Direct 候选。
- mask 路线进一步强化：pixel text mask 侧边优先、边线保护带二次校验、以及对白底气泡优先使用 **threshold clear** 清暗像素，而不是默认把整个区域交给重型 inpaint。
- 配置与 sidecar 契约同步：新增 `replace_translation` 命名空间，`config.example.json` 已从真实 `PipelineConfig` 重新导出。

## v0.9.0-alpha.1 出版级基建

- 新增 `docs/PUBLICATION_GATE.md` 与可执行 `scripts/publication_gate.py`：支持私有 3–5 部作品 / 100–300 页真实配对基准，不在仓库内打包版权页。
- Gate 统计页面配对准确率、可选 golden UnitMatch 准确率、自动通过页 TARGET-only residual、transfer-mask 外边线损伤、safe-area spill、Review 率与每页耗时。
- 硬失败页自动归档 source / target / final / registration / matching / mask / project / QA，并生成 `failure_evidence.json`。
- Dual SOURCE 不再“有 secondary 就偏向 secondary”：新增 publication arbitration，综合 same-page、registration、reprojection、Direct coverage、border alignment、SOURCE sharpness、residual/review risk。
- secondary 配准差或 Direct plan 不安全会被硬拒；分数接近时优先 primary，保证主 SOURCE 译文权威。
- 新增 `benchmarks/README.md`，规范私有真实成对基准的数据目录和 `labels.json`。

## v1.3.6 推荐工作流：自动尽量净，TARGET 擦除收尾

1. 默认使用 `transfer.mode=auto`、`registration.backend=auto`，开启 cache + resume。macOS 优先 Apple Live Text；默认气泡后端保持 `seeded_white`。
2. 白色气泡优先 TARGET 纸色/threshold clear；复杂彩色区域才使用 Telea/外部 LaMa。不要为了速度关闭 residual cleanup。
3. 自动结果仍有日文、黑点、短线或标点时，**优先使用「仅擦 TARGET 日文层…」**。它在最终结果上工作，并硬保护已迁移/重排/人工补漏的中文。
4. 「编辑自动清除蒙版…」只用于改变自动清字管线，不作为默认最终收尾画笔。
5. 白气泡漏检用人工「白色气泡 · 文字迁移」；彩色开放字用「擦除显字」。
6. 实验整页对齐挖洞必须看到绿色 armed 状态，并重新处理当前页/整册才会生成结果。
7. 批量速度优先从缓存、resume、registration auto、模型常驻和关闭默认 SR/LaMa 获得；不要关闭白底残笔清理、边线保护或中文保护。

同一输入和同一影响像素的配置二次运行应命中配准/OCR/气泡缓存；CPU 线程、MPS 内存比例、resume、缓存开关以及导出层设置不会改变已完成页的像素 fingerprint。

## v1.3.12：pages 空间与漏框补全

默认输出现在采用精简工作区：`final.png`、SOURCE/TARGET lossless 原页、核心 clear/transfer/text layer、`project.json`、QA 与人工 Review 状态仍保留；Debug 图、逐组件 mask、自动 ORA/PSD、自动 `inpainted.png` 改为按需输出。GUI 的“出版输出”页可重新开启这些诊断项。

已有项目可点击 **“清理 pages 冗余诊断文件…”**，或命令行执行：

```bash
mhd-transfer cleanup-workspace /path/to/output
```

清理不会删除 final、SOURCE/TARGET 原页、中文迁移图层、Review override、人工清除或 TARGET 日文层擦除数据。

白色气泡/旁白框识别新增 OCR-free completion 可视化与短文本/细长矩形救援。OCR 漏掉文字块时，只要 SOURCE/TARGET 已验证为同页且注册后的墨迹确实发生翻译变化，容器仍可进入自动迁移与 debug/project 列表；不会仅凭“白色矩形”直接写入。


## v1.3.13：彩色人工补漏与大画布工作区

- 红/黄/紫等高饱和气泡的人工“擦除显字”会先锁定 TARGET 彩色容器内部，再提取 SOURCE 中文与 TARGET 日文；平坦彩底直接恢复 TARGET 局部中位色，避免 Telea 把黄色/红色背景修糊。
- 替换工作区中间画布扩大，右侧参数栏可隐藏；人工补漏/Reveal/Mask 编辑器扩大并自动聚焦选区。
- CPU 模式对可选 Torch 更稳；未自动迁移的 passthrough 页仍保留进入 GUI 人工修复所需材料。
- “扩大 Direct 候选范围（难页）”默认关闭，仅在特殊难页手动启用。
