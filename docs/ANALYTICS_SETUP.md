# Folirina Web — 匿名使用统计接入指南

## 当前状态

代码已实现，**但默认关闭**。`analytics-config.js` 的 `UMAMI_WEBSITE_ID` 为空时，**不会加载第三方脚本或上报数据**。绑定网站 ID 后才开始统计，不能回溯此前的访问。

## 为什么使用 Umami 而不是 Cloudflare Web Analytics？

Cloudflare Web Analytics 当前主要提供页面访问统计，官方 FAQ 明确说明自定义事件尚未支持。Umami 的免费 Hobby 方案支持页面访问、独立访客和自定义事件，因此可以查看“漫画导入成功”和“成品导出成功”。

## 一次性启用步骤

1. 打开 <https://cloud.umami.is/signup>，注册 Umami Cloud 账户（选择免费 **Hobby** 方案）。
2. 在 Umami 仪表盘 **Websites → Add website**，填写 `Folirina Web`，域名使用 `amster-ilvil.github.io`（只写主机名，不要 `https://` 和 `/Folirina/`）。
3. 进入新网站的 **Edit → Tracking code**，复制 `data-website-id` 的 UUID（公开网站标识，不是密码或 API Key）。
4. 在本仓库 `docs/analytics-config.js` 中填写：`export const UMAMI_WEBSITE_ID = '粘贴网站ID';`；提交到 `main` 后等待 GitHub Pages 部署成功。
5. 从公开站点 <https://amster-ilvil.github.io/Folirina/> 测试，并从 Umami 仪表盘查看统计。可打开浏览器 Network 面板核对 `cloud.umami.is`，广告拦截器和浏览器隐私设置可能阻止统计。

## 指标和口径

| 统计 | Umami 仪表盘 | 触发条件 |
| --- | --- | --- |
| 网页访问量、访客数 | Views / Visitors | Umami 脚本自动统计，单页打开自动上报 |
| `source_import_success` | Events | SOURCE 文件解析完毕且成功更新编辑工作区 |
| `target_import_success` | Events | TARGET 文件解析完毕且成功更新编辑工作区 |
| `workspace_open_success` | Events | 已保存的工作区 ZIP 被正确加载 |
| `review_apply_success` | Events | 人工复核掩码确认应用 |
| `page_export_success` | Events | 当前页图片编码成功且发起本地下载 |
| `book_export_success` | Events | 整册 ZIP 创建成功且发起本地下载 |
| `mask_export_success` | Events | 差异掩码 ZIP 创建成功且发起本地下载 |
| `workspace_save_success` | Events | 工作区 ZIP 创建成功且发起本地下载 |

这些计数不是唯一用户数，也无法证明用户实际保存或阅读了下载内容。浏览器拦截、无网络、Do Not Track/Global Privacy Control 等可能使数据少于真实使用量。

## 隐私与安全

- **不会上传漫画图像、文件名、工作区、原始文字、编辑内容或用户 ID**。应用层只发送固定事件名，不附加事件数据或页数。
- 页面访问由 Umami 统计；设置 `data-exclude-search=true` 和 `data-exclude-hash=true`，不收集 URL 查询参数和片段；限制 `data-domains=amster-ilvil.github.io`。
- 尊重浏览器 Do Not Track / Global Privacy Control，统计失败不影响编辑、预览和导出。
- 不向公共源码提交 Umami 密码、API token 等凭证；`UMAMI_WEBSITE_ID` 是公开网站标识。
- 仅从站点正式启用后开始收集，无法获得历史未记录的访问次数。

官方参考：<https://docs.umami.is/docs/track-events>、<https://docs.umami.is/docs/tracker-configuration>。
