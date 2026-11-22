# 调研日志

## 2026-11-15
- 读取 web-access 与 planning-with-files 技能。
- 检查仓库、当前变更与项目级指令；工作区初始为空。
- web-access 附带 CDP 脚本缺失，采用内置 computer use。
- CUA 首次 getBrowser 连接超时，继续定位可用浏览器。
- 应用更新后 CUA 工具被移除；使用仍可用的 Playwright 浏览器交互工具恢复访问，成功打开官方 Overview。
- planning-with-files session-catchup 正常退出，提示不支持 Codex 原生历史解析。
- 已读 Overview、Data、Rules（专属与基础规则），记录了数据行数与获奖许可证表述差异。
- Discussion 列表已打开，识别官方置顶入门帖，继续阅读。
- 已读官方入门/Discord 说明与 Code 列表；打开置顶 Starter Notebook（新标签）。未运行代码。
- 已完整阅读官方 Starter、提交 FAQ、调试说明、获奖文档指南；已浏览 Models、Leaderboard。
- Discussion 通过 Admin 过滤发现含官方回复的规则集中答疑与输入读取性能帖，继续核对关键限制。
- 已阅读全部 Admin 过滤结果中的讨论正文与可见回复；区分 staff 澄清和普通参赛者自报。已读一个社区高分 Notebook 的方法/部分代码/归属、一个关联 GitHub README、一个原始论文摘要与 Open Pool v15 数据卡。
- 新增 README.md、docs/competition-brief.md、docs/engineering-plan.md；新增 .gitignore，仅排除自动生成的 .playwright-mcp/ 原始浏览器 traces。
- 文档审查：运行 python3 一次性脚本，检查 6 份 Markdown 的本地链接、18 标签与常见凭据/临时签名 URL 模式，全部检查满足；19 个不同外部链接来自已访问源页面（含锚点），未做额外 HTTP 链接爬取。此扫描仅覆盖选定模式，不等于全面安全审计。
- 运行 git diff --check：退出 0。初始空仓库，新文件尚未 tracked，另由 Python 脚本检查文档内容；无原文件被替换。
- 运行 git check-ignore .playwright-mcp/page-2026-11-15T15-51-11-293Z.yml：确认浏览器原始 traces 被忽略。扫描到的 35 份 traces 中，选定临时签名 URL 模式命中 0；不发布 traces。
- 最终回读 Overview 的 DOM 锚点：timeline、evaluation、code-requirements 均存在，文档章节链接正确。
- 无源代码行为变更，未运行模型/训练测试，也没有可声称的本项目分数。未安装依赖、未调用付费模型 API、未接受规则、未下载比赛文件、未提交或推送；公开浏览产生的网络请求属于本次授权阅读范围。
- 关闭本轮 Playwright 创建的 11 个研究标签，回读 tab list 仅剩初始 about:blank。没有操作用户原有 Chrome 标签。
- 最终命令：git diff --check；shasum -a 256 .gitignore README.md docs/competition-brief.md docs/engineering-plan.md findings.md progress.md task_plan.md > research-manifest-2026-11-15.sha256；shasum -a 256 -c research-manifest-2026-11-15.sha256。manifest 是本轮新增文件的首次摘要，不替换既有历史哈希；不包括原始浏览器 traces。
- 上述首次摘要检查 7/7 OK。随后 planning-with-files/check-complete.sh 报 0/0：脚本只识别 `### Phase` 与 `**Status:**`，不识别原中文列表。已将计划状态改为其支持的格式，未改变实际完成条件。保留首次 manifest 作为先前快照；当前文件摘要另存 research-manifest-2026-11-15-v2.sha256，首次 manifest 不作为当前工作区校验入口。最终重新运行 check-complete.sh、git diff --check 和 v2 shasum 校验。
