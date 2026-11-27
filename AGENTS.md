# 项目工程规则

## 学习协作

用户以能力训练和个人作品集为主要目的。先解释目标、代码路径与验收，再给一个小练习；默认由用户操作和表达结论，代理负责提示、检查和代码审查。不要主动实现学习练习、参数改进或下一阶段功能；用户明确要求代写/代跑时再执行。区分官方方法、代理搭建的工程与用户本人完成的实验，不把代理代做记录成用户原创成果。项目介绍入口是 README.md，导读是 docs/learning-guide.md，执行命令在 docs/running.md，过程记录放 docs/worklog/。

遵循用户提供的全局工程准则。当前目标是复现官方 TF-IDF + LogisticRegression，不混入新模型、外部训练数据或参数搜索。2026-11-16 用户新增授权私有 CPU 离线 Notebook 正式提交 baseline，并在失败时定位修复；此前不正式提交的阶段边界已被替代。保持失败历史和每日额度记录。

- 环境：Python 3.11，`uv sync --locked`；已有环境的离线命令用 `uv run --offline --locked`。版本与依赖变更需说明原因、核对一手资料和许可证，再更新锁文件。
- 行为变更先写失败测试再实现；使用 `uv run --offline --locked pytest -q`。合成测试不等于比赛数据验证。
- 静态检查：`uv run --offline --locked ruff check src tests scripts`、`uv run --offline --locked ruff format --check src tests scripts`、`uv run --offline --locked mypy src/ntl_baseline`。
- Notebook 由源码生成：`uv run --offline --locked python scripts/build_notebook.py`；用 `--check` 核对。不要手工维护第二套训练逻辑；提交到 Git 的 Notebook 必须没有执行输出。
- 训练折内部拟合所有特征，GroupKFold 按 prompt_id；保存 OOF、fold 映射、每类指标、版本、资源与哈希。不根据公榜或验证失败放宽检查。
- `data/raw/` 原始文件只读；`artifacts/` 私有且忽略。禁止写入或公开凭据、比赛文本、原始浏览器 traces。模型只加载本项目自己训练的可信本地 joblib；哈希不是任意 pickle 的安全保证。
- 每次运行使用新的产物目录，拒绝覆盖历史结果。失败目录没有 completed.json；不得把部分结果当完成。
- `docs/research/2026-11-15/snapshot/` 是冻结的前置调研，禁止编辑或重新计算其 manifest。当前工作改根文档，发现旧结论变化时新增有来源的说明。
- 完成前执行 `uv run --offline --locked python scripts/check_repository.py`、Notebook `--check`、项目测试与静态检查、`git diff --check`。Kaggle 镜像/隐藏测试未经实际运行时不能声称通过。

2026-11-28 用户确认 A1 阶段。官方 baseline 继续独立维护。允许在独立的 ntl_member 路径上使用 Qwen3-8B-Base 与经审计的既有公开数据训练 q8pool，首个候选只替换完整 V31 的 q8v5。禁止模仿标注与自生成训练文本。按 prompt_id 评估，公开报告披露上游数据选择偏差。参数及门槛预先登记。Opus review 和修订后推送，每次正式提交前核对额度并取得用户确认。
