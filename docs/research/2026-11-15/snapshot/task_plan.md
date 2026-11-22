# 比赛调研计划

日期：2026-11-15（Asia/Shanghai）

## 目标与范围
通过 computer use 阅读 Name That LLM 的官方比赛信息：任务、数据、评分、规则、期限、代码/运行限制、提交流程、官方补充说明及有价值的公开基线；形成带来源与待核实项的中文说明。当前不报名、不接受规则、不下载比赛数据、不运行训练、不提交。

## 阶段
### Phase 1 仓库与访问能力检查
**Status:** complete
空仓库，无项目级 AGENTS.md。

### Phase 2 官方公开内容阅读
**Status:** complete
Overview / Evaluation / Data / Rules / Code / Models / Discussion / Leaderboard，以及 Admin 过滤结果；不声称穷尽全部社区内容。

### Phase 3 形成约束和工程路线
**Status:** complete
比赛说明、方案比较、工程规范和后续验收已形成。

### Phase 4 核对来源和交付
**Status:** complete
文档校验通过；数据/硬件/模型复现属于后续阶段，明确未执行。

## 错误与处理
- web-access 的 scripts/check-deps.mjs 与 site-patterns 不存在：改用内置 CUA。
- CUA getBrowser 首次连接超时并重置：改用表面清单定位可用浏览器。
- 完成检查脚本首次显示 0/0，因为原阶段为中文列表格式：仅调整本计划的状态格式以匹配脚本，保留原 manifest，并为最终状态新增 v2 manifest；未改验收标准。

## 完成条件
说明已观察的官方事实、来源与日期；将建议和事实分开；无法访问内容明确列出，不声称全部核验。
