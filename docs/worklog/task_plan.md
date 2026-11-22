# 官方基线复现计划

日期：2026-11-15（Asia/Shanghai）

## 目标与范围
用户已授权获取官方数据、搭建项目、训练、验证和生成提交文件，并整理文件树与 README。本轮不向 Kaggle 正式提交、不增加外部训练数据或模型、不使用付费 API。接受比赛规则由用户在个人浏览器完成。

上一轮调研与原始 manifest 已复制归档到 docs/research/2026-11-15/snapshot，v2 的 7 个文件摘要检查通过。原始哈希未改写。

## 阶段
### Phase 1 当前状态与设计
**Status:** complete
空代码仓库、尚无 Git commit；已读取用户准则和既有设计。使用 Python 包 + 薄 Notebook；不增加框架，复用官方 TF-IDF + LogisticRegression；先比较兼容依赖与现代版本，再安装本地隔离环境。

### Phase 2 数据获取与环境
**Status:** complete
当前正常浏览器会话已登录且比赛下载可用；代理直接从官方 Download All → ZIP 获取，不代为接受规则、不提取凭据。ZIP CRC、三文件 schema、18 类与源哈希审计通过；2,699 train / 150 prompts / 3 example test。环境已锁定、安装兼容检查通过。

### Phase 3 测试驱动实现
**Status:** complete
按 red/green 实现；46 个合成需求测试、ruff、mypy、Notebook 源码检查通过。CLI、直接参照与离线 Notebook 在明确合成输入上完整实跑，输出一致；未把合成分数计为官方数据成绩。

### Phase 4 官方数据端到端复现
**Status:** complete
真实数据审计、5-fold CV、全量训练、独立预测、Parquet读回、直接官方参照与真实数据离线Notebook通过。折分数逐项一致，CLI/Notebook OOF/folds/submission字节一致；原始摘要未变。本地CV0.299655，与官方历史0.3145差异原因未定位，明确披露，不改标准。

### Phase 5 文档与质量核验
**Status:** complete
README/文件树/数据说明/依赖选择/归属/项目AGENTS/复现报告已与实际状态对齐；46 tests、ruff、format、mypy、依赖兼容、Notebook源码与空输出、仓库链接/选定凭据模式、冻结7摘要、真实产物摘要核验通过。当前源码摘要与真实执行完全一致，无后续源码改写；无Kaggle正式提交、远端发布或付费API。

## 错误与处理
- web-access 的 scripts/check-deps.mjs 与 site-patterns 不存在：改用内置 CUA。
- CUA getBrowser 首次连接超时并重置：改用表面清单定位可用浏览器。
- 完成检查脚本首次显示 0/0，因为原阶段为中文列表格式：仅调整本计划的状态格式以匹配脚本，保留原 manifest，并为最终状态新增 v2 manifest；未改验收标准。
- 首次归档命令错误地指定尚未创建的目标目录为 cwd，进程未执行；改从仓库根创建归档并校验成功。

## 完成条件
真实官方数据上的训练/验证/Parquet 生成与读回有新鲜证据；官方参数一致；记录与同环境参照的差异及公开分数未复现的边界；README 命令与文件树符合实际。若数据未到，继续完成独立工作并明确待办，不宣称基线已复现。
