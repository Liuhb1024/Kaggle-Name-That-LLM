# 运行与复现手册

学习项目先看[项目导读](learning-guide.md)。本页保留可执行命令，已有v1结果不要覆盖；不用一次运行全部步骤。

## 快速开始

需要已安装 uv 和 Python 3.11（本机使用 3.11.15）。首次安装会从官方包分发渠道下载依赖，局限在本项目 .venv；此后命令使用离线锁定环境。

```bash
uv sync --locked
```

Git 不包含原始数据。下面展示干净工作区的导入与完整运行；已有 artifacts/official-baseline-v1 时使用新名称，例如 v2，程序拒绝覆盖。

新的工作区需在本人 Kaggle 账号接受比赛规则后，从[官方 Data 页](https://www.kaggle.com/competitions/name-that-llm/data)下载 ZIP，保留原件。

```bash
uv run --offline --locked ntl-baseline import-data \
  --archive "$HOME/Downloads/name-that-llm.zip" \
  --data-dir data/raw/name-that-llm

uv run --offline --locked ntl-baseline audit \
  --data-dir data/raw/name-that-llm

uv run --offline --locked python scripts/run_offline.py run \
  --data-dir data/raw/name-that-llm \
  --config configs/baseline.json \
  --run-dir artifacts/official-baseline-v1
```

每次运行使用新的目录，存在即拒绝覆盖。run_offline.py 禁止当前 Python 进程的网络连接和 DNS，不是运行恶意程序的 OS 沙箱。程序不下载模型、不调用外部推理服务，也不修改原始输入。

成功后关键产物为：

```text
artifacts/official-baseline-v1/       # Git ignored；当前已生成
├── started.json                     # 环境、配置、源码摘要
├── data_manifest.json               # 原始文件摘要与数据审计
├── validation/
│   ├── metrics.json                 # folds、balanced accuracy、每类 recall、混淆矩阵
│   ├── oof.parquet                  # OOF 标签/概率、id、prompt、fold
│   └── folds.parquet
├── model/
│   ├── pipeline.joblib
│   └── metadata.json                # 训练输入、包版本、模型摘要
├── prediction/
│   ├── submission.parquet           # id、model，已读回校验
│   └── report.json
└── completed.json                   # 全流程成功后创建，含产物摘要
```

## 单独训练和预测

独立入口便于对新 test.csv 推理；推理不依赖 train.csv 或样例提交的行数。

```bash
uv run --offline --locked ntl-baseline validate \
  --data-dir data/raw/name-that-llm --output-dir artifacts/manual-validation-v1

uv run --offline --locked ntl-baseline train \
  --data-dir data/raw/name-that-llm --output-dir artifacts/manual-model-v1

uv run --offline --locked ntl-baseline predict \
  --data-dir data/raw/name-that-llm --model-dir artifacts/manual-model-v1 \
  --output-dir artifacts/manual-prediction-v1

uv run --offline --locked ntl-baseline check-submission \
  --data-dir data/raw/name-that-llm \
  --submission artifacts/manual-prediction-v1/submission.parquet
```

只加载自己训练的可信本地 joblib。它使用 pickle，哈希检查不能使未知来源模型安全；加载时还要求相同的包版本。失败退出码为 2，具体输入合同见[数据说明](../data/README.md)。

## 官方对照与 Kaggle Notebook

直接官方参照和真实数据 Notebook 的命令见[复现记录](baseline-reproduction.md)。参照脚本独立按官方示例重建方法，比较每折分数与 test 预测；它不会把官方发布成绩当成本项目实测值。

```bash
uv run --offline --locked python scripts/build_notebook.py
uv run --offline --locked python scripts/build_notebook.py --check
```

[official_baseline.ipynb](../notebooks/official_baseline.ipynb) 嵌入同一包源码，无需挂接代码仓库或联网安装。Kaggle 中挂接比赛数据，选择 CPU 并关闭 Internet，Save & Run All 后输出 working/submission.parquet。其余日志、OOF、模型保存在 working/baseline-run。Notebook 记录实际运行版本；Kaggle 镜像一致性、隐藏约 4,000 条重跑和正式评分需另行实测，本地运行不能证明公榜分数。

## 检查命令

```bash
uv run --offline --locked pytest -q
uv run --offline --locked ruff check src tests scripts
uv run --offline --locked ruff format --check src tests scripts
uv run --offline --locked mypy src/ntl_baseline
uv run --offline --locked python scripts/build_notebook.py --check
uv run --offline --locked python scripts/check_repository.py
git diff --check
```

测试覆盖：fold 内拟合、防 prompt 泄漏、id/标签/schema、缺失文本、动态 test 条数、Parquet 对齐、模型篡改、历史不覆盖、ZIP 路径/链接、独立 CLI 和离线 Notebook。合成测试不包含比赛文本，也不代表真实数据指标。

原始数据、模型、OOF、已执行 Notebook、凭据和浏览器 traces 不进入 Git。公开代码与获奖交付仍需遵守[官方 Rules](https://www.kaggle.com/competitions/name-that-llm/rules)。服务器首次拉取后按[服务器工作流](server-workflow.md)启动。

## Kaggle 运行与提交补记

仓库Notebook从同一包源码生成；挂接Name That LLM competition input，选CPU（Accelerator None）、关闭Internet、保持Private，再Save & Run All。平台提交会再次保存版本并在隐藏test上重跑，不能将本地3行预测直接上传代替。Notebook显式使用`require_sample=False`，只依赖train/test；公开样例不能限定隐藏test的id。最终submission.parquet仍按实际test完整校验。默认CLI及归档导入仍要求三个文件且严格校验样例。

首次平台运行、失败和修复重试均由代理执行，见[提交摘要](worklog/baseline-submission-summary.md)。不自动重复提交，也不根据分数改参数。
