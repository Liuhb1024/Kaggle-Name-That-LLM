# 官方基线复现状态

2026-11-16 当前状态：官方数据获取、审计、真实 5 折验证、全量训练、独立模型加载、Parquet 生成与读回、独立官方代码对照和离线 Notebook 均已完成。本项目实现与同环境官方代码参照的每折分数和测试预测完全一致；不把官方历史发布值或合成分数写成本项目成绩。

## 真实数据与首次运行

原件保留 data/raw/downloads/name-that-llm.zip（874,800 字节），ZIP CRC 通过；train.csv 2,699 条，150 prompts、18 labels；test.csv 3 条。claude-fable-5 为149条，其余各150；149个prompt有18条，1个17条。词数36–274，无空文本、训练重复全文或 train/test 全文重叠。

| 项目 | 实测 |
| --- | --- |
| 5 折 balanced accuracy 平均 | 0.2996551724 |
| fold 标准差（非置信区间） | 0.0167636672 |
| pooled OOF balanced accuracy | 0.2996718867 |
| CV 耗时 | 52.77s |
| 全量训练与模型保存 | 15.51s |
| 样例预测与 Parquet 读回 | 0.044s |
| 全流程耗时 | 69.11s |
| 进程峰值 RSS | 620,019,712 bytes，约591.3MiB；不是独立阶段峰值或进程树 |
| 环境 | macOS15.7.2 arm64，8 logical CPU，Python3.11.15；BLAS/OpenMP 单线程 |
| 产物目录 | artifacts/official-baseline-v1 |
| 提交文件 | artifacts/official-baseline-v1/prediction/submission.parquet，3行、id/model、读回通过 |

官方页面发布 CV 0.3145，与本机相差约0.01484。本项目没有为了追该值改变数据、折分或参数；原因需与独立参照、原官方环境/划分证据区分核验，不能直接归因于某个库或硬件。

原始摘要与来源见 data/raw/name-that-llm/source.json，运行记录 completed.json/validation/metrics.json。原始文件、OOF 和模型不公开提交到 Git。

### 独立对照与 Notebook 证据

直接官方参照文件：artifacts/official-reference-v1.json。它独立使用官方默认 read_csv、FeatureUnion、LogisticRegression 和 cross_val_score；没有调用本项目 build_pipeline。

| Fold | 项目与独立参照 balanced accuracy |
| --- | --- |
| 0 | 0.2833333333 |
| 1 | 0.3277777778 |
| 2 | 0.3092592593 |
| 3 | 0.2925925926 |
| 4 | 0.2853128991 |

每折最大绝对分差为0；3条 test 的预测完全一致；原始数据摘要未变。同环境直接参照也得到0.2996551724，因此目前没有证据表明本地方法实现偏离官方代码；它不证明已复现官方历史运行的0.3145。原官方环境、原折分映射与完整运行状态未取得，差异原因不作无证据归因。

离线 Notebook：artifacts/official-notebook-v1/executed.ipynb，最终文件为该目录下 submission.parquet。全流程110.15s，进程峰值RSS600,195,072 bytes。Notebook 在临时独立 kernel 执行，Python 网络连接与 DNS 被禁用；其 OOF、fold 映射和最终 Parquet 与 CLI 字节完全相同。第三次独立 CLI predict 输出也与首次 Parquet 字节相同。

上述执行部分并发，耗时是本机实际观察，不能当作隔离性能对比。没有使用GPU、云端训练或付费模型API；本地 .venv 约478MiB，原始数据与 ZIP 约2.9MiB。无正式 Kaggle 评分提交。

关键 SHA-256：

```text
ZIP              7c375db7dfeea6b36cefc7b48a290dd48c54c7fe03bfd402390728a6a2bb9386
train.csv        40a22ce8d88c5f46feb4637a099db815bef5f713ffe53e89dbe1b7b21de77707
test.csv         4545aa3a3459f4c0724f97ad5bd06c4caa3506bdfea5fab905e8203184fc0c57
sample parquet   ccf86796b80fd7f83d8463664603fd1208e17559abcf416f198776b819646188
submission       f6677b37d0d4bea1b751c6c8ac508203952583279fba368d4985b761c8cea36a
```

这是首次记录上述实际原件/产物摘要，不修改前置调研历史 manifest。原始文件、source.json 和 ZIP 在当前工作区设为0444，私有父目录0700；后续运行必须使用新目录。

## 已实现流程

1. ZIP 安全导入、原始数据摘要、schema/标签/id 审计。
2. 5 折 GroupKFold 按 prompt_id 切分，每折新建完整 TF-IDF Pipeline，仅在该折训练部分拟合。
3. 保存 OOF 概率、标签顺序、fold 映射、每类 recall、混淆矩阵、fold mean/std 与 pooled OOF balanced accuracy。
4. 全量训练，保存 pipeline.joblib 与环境/输入/模型摘要；独立进程加载时核对包版本和模型 hash。
5. 动态读取 test 行数，生成真正 Parquet，按 id 对齐，读回核验。新的运行目录防止覆盖历史结果；完成标记只在全流程成功后创建。
6. 从同一包源码生成无执行输出的独立 Notebook，使用临时目录加载嵌入代码；其最终 working/submission.parquet 符合比赛格式。

## 当前验证证据

- 初始 TDD：32 个需求测试因 API 未实现而失败；测试夹具问题的修正和后续 red/green 记录见[执行日志](worklog/progress.md)。原始 red 日志位于 Git 忽略的 artifacts/checks/tdd-red-2026-11-16/pytest.log。
- 当前完整测试：`uv run --offline --locked pytest -q --tb=short`，46 passed（7.86s，含独立内核、禁用 Python 网络连接的 Notebook 端到端测试）。测试输入均为合成文本。
- `ruff check src tests scripts`、`ruff format --check src tests scripts`、`mypy src/ntl_baseline` 均通过。仓库链接/选定凭据模式检查与 7 份冻结摘要核验通过；依赖兼容检查通过。
- Notebook 的源码映射和 hash 可用 `python scripts/build_notebook.py --check` 检查；共享 Notebook 没有执行输出。

## 官方参数及复现差异边界

| 项目 | 实现 |
| --- | --- |
| word TF-IDF | ngram=(1,2)，max_features=25000，sublinear_tf=True |
| char TF-IDF | analyzer=char，ngram=(3,5)，max_features=35000，sublinear_tf=True |
| LogisticRegression | C=2.0，max_iter=1000，solver=lbfgs，不加入 class_weight |
| 验证 | GroupKFold(5, shuffle=False)，balanced_accuracy |
| 原始文本 | 不删除标点、不额外归一化、不裁剪；缺失文本填空串 |
| 本地资源 | 串行 folds，BLAS/OpenMP threads=1，CPU；资源调度与官方 n_jobs=-1 不同 |
| CSV schema | 显式字符串 dtype，保留前导零 id；将在真实数据对照脚本中检查与官方默认 read_csv 的差异 |

官方 Starter 发布 CV 0.3145±0.0153、公榜 0.322，仅是参考。我们的 Python3.11.15/sklearn1.6.1 是固定的本地环境，并未证明它与官方运行镜像完全一致。

## 本轮实际执行命令

```bash
uv run --offline --locked python scripts/run_offline.py run \
  --data-dir data/raw/name-that-llm \
  --config configs/baseline.json \
  --run-dir artifacts/official-baseline-v1

uv run --offline --locked python scripts/compare_official_reference.py \
  --data-dir data/raw/name-that-llm \
  --run-dir artifacts/official-baseline-v1 \
  --output artifacts/official-reference-v1.json

uv run --offline --locked python scripts/execute_notebook_offline.py \
  --data-dir data/raw/name-that-llm \
  --work-dir artifacts/official-notebook-v1
```

直接参照脚本不调用本项目 build_pipeline，独立按官方参数构建、默认 read_csv、cross_val_score，再比较每折分数与全量模型 test 预测。它不检查公榜目标是否恰好相等，也不掩盖差异。Notebook 执行在新的临时 kernel，禁用 Python sockets/DNS；这不是面向恶意代码的 OS 沙箱。

以上命令本轮均已执行成功；原始摘要、真实行数、CV、直接参照、原始数据未变、独立预测/Notebook一致性与资源记录已获得。重复执行需将 v1 改成新的目录名，避免覆盖。任何后续失败都需保留原目录并记录，不覆盖或降低门槛。

Kaggle 账号的 GPU 周配额、RAM/磁盘/镜像、隐藏约 4,000 条重跑和公榜评分尚未验证。本轮用户只要求生成提交文件，未执行正式 Kaggle 提交或发布。

## 2026-11-16 至 17 日：Kaggle 平台验证补记

用户新增授权私有 Notebook 正式提交，上一阶段未提交的记录保留为历史。平台版本1（scriptVersionId=370339758）已于2026-11-16运行成功，耗时127.8秒，CPU/Internet off。平台CV为0.3144955300±0.0153335442，与官方历史展示值四舍五入一致；本地CV0.2996551724及独立参照一致的记录不变。平台Python3.13.18/Linux x86_64与本地Python3.11.15/macOS arm64及部分库版本不同，尚未用控制实验确认具体差异原因。

首次提交版本2（370341317）隐藏重跑异常，没有评分。检查发现完整流程把可见sample的id当成隐藏test的标准；新增stale/absent样例测试先复现失败，再以显式Notebook模式只审计train/test，保留最终输出合同和默认CLI/导入严格检查。修复版本已重试，最终状态见[本次提交摘要](worklog/baseline-submission-summary.md)。平台普通Notebook日志反映可见3行test，不应当作隐藏数据日志。

修复版本3（370342837）正式提交已显示Succeeded，公榜 **0.322**；确认于上海时间2026-11-17。普通运行137.8秒；隐藏重跑独立耗时未公开。共2次提交（1失败、1成功），无调参。官方历史公榜展示值已匹配，最终私榜尚未产生。
