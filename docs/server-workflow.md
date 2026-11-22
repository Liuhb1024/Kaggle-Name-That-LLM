# 服务器首跑与 Kaggle 交付

CPU 基线留作固定对照。3090 Ti 的首轮实验直接使用独立 PyTorch 环境运行 V31 的 Qwen3-8B 单分支。模型与适配器由服务器从授权来源准备。 [单卡运行与锚点验收](v31-single-gpu.md)

## 可选的 CPU 基线参考

在已有的 clone 执行以下命令。将 `REPLACE_WITH_COMMIT_SHA` 换成已推送的完整 commit SHA，确保运行的是审过的版本。

```bash
cd ~/kaggle-name-that-llm-lhb/Kaggle-Name-That-LLM
git fetch origin main
git switch --detach REPLACE_WITH_COMMIT_SHA
git rev-parse HEAD
uv sync --locked
uv run --offline --locked pytest -q
```

测试使用合成数据。若要运行完整基线，在自己的 Kaggle 账号下从[官方 Data 页](https://www.kaggle.com/competitions/name-that-llm/data)取得原始 ZIP，按[运行手册](running.md)导入，并用一个新的产物目录运行：

```bash
uv run --offline --locked python scripts/run_offline.py run \
  --data-dir data/raw/name-that-llm \
  --config configs/baseline.json \
  --run-dir artifacts/server-baseline-v1
```

全流程成功后，`artifacts/server-baseline-v1/completed.json` 会出现。先读 `validation/metrics.json`，核对按 prompt 分组的五折结果；复现报告记录过环境差异，服务器的结果应如实保留。[基线复现记录](baseline-reproduction.md)

## 代码与产物目录

| 位置 | 内容 |
| --- | --- |
| `src/ntl_baseline/`、`configs/baseline.json` | 官方基线代码与参数，可作为固定对照。 |
| `src/ntl_repro/`、`configs/v31_single_gpu.toml` | V31 plain LoRA 成员的单卡入口、参数与资产指纹。 |
| `notebooks/` | 从源码生成无执行输出的交付 Notebook，推理逻辑只维护一份。 |
| `data/raw/` | 本地只读原始数据，每台机器从授权来源取得，Git 忽略。 |
| `artifacts/<run-id>/` | 本地权重、适配器、OOF、日志和产物清单，Git 忽略；每次运行用新目录。 |
| `docs/worklog/` | 记录结论、分数、复现条件和失败原因，避免写入比赛原文和私密信息。 |

GitHub 负责代码和版本；服务器负责实验；Kaggle 负责隐藏测试推理和评分。训练在服务器完成后，只在核对规则、许可和来源后向 Kaggle 提供推理必需的资产，并记录 Git commit、资产版本及哈希、Notebook 版本。Kaggle Notebook 关闭网络，读取当次挂载的 `test.csv`，生成 `submission.parquet`；服务器对示例 test 的预测只用于检查格式。[比赛要求](https://www.kaggle.com/competitions/name-that-llm/overview/abstract)

检查无误再提交：跑 `scripts/check_repository.py`，并审查 Git 暂存区，只含源码、锁文件、配置、测试及允许分享的文档。原始数据、运行产物、权重、凭据和浏览器痕迹不随代码同步。比赛代码如需公开，遵守[本赛代码分享规则](competition-brief.md#参赛规则和官方澄清)。
