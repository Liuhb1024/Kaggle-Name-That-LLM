# A1 单卡训练与 V31 交接

A1 训练 24 层 Qwen3-8B 的 LoRA 分类成员 q8pool，C1 在固定 V31 中替换 q8v5。第一步评审依据是 [Opus 实施规格](specs/member-q8pool-v1.md)。0.750 是公榜目标。约 10–14 GPU 小时是规格估计，机器实测在阶段 01 产生。

## 评审与启动

Opus 审查本地 diff、先失败的回归测试、实际输出和资源假设。修订通过后推送审过的 commit。开跑前核对主办方规则答疑、完整 V31 复跑记录及当前可复现候选。核对人记录日期和版本。[规则答疑](https://www.kaggle.com/competitions/name-that-llm/discussion/752084) · [团队提交页](https://www.kaggle.com/competitions/name-that-llm/submissions)

下面是服务器命令模板。发布时 Codex 将 SHA 替换成已推送的完整哈希。服务器需要已参加比赛和可下载公开资产的 Kaggle 身份。当前试验使用已验过的 train.csv 和底座缓存，首跑自动准备锁定的 Python 3.11 CPU 环境；GPU 环境复用同级的 .venv-gpu-check。安装、公开数据下载和取得 V31 固定源码需要联网，模型微调及 Kaggle 评分在本地资产上执行。

```bash
cd ~/kaggle-name-that-llm-lhb/Kaggle-Name-That-LLM && (
  set -e
  commit='<review 通过并推送的 40 位 SHA>'
  git cat-file -e "$commit^{commit}" 2>/dev/null || git -c http.version=HTTP/1.1 fetch origin main
  git switch --detach "$commit"
  bash scripts/run_member_server.sh --commit "$commit" --python ../.venv-gpu-check/bin/python --run-root artifacts/member-q8pool-v1
)
```

独立终端运行 `watch -n 1 nvidia-smi` 看 GPU，或 `tail -f artifacts/member-q8pool-v1/pipeline.log` 看当前阶段和详细日志路径。计划预览加 `--dry-run`，仅需系统 Python 标准库。故障处理先读对应 attempt 的 stage.log。在同一个 commit 和目录上加 `--resume`，已完成阶段先验摘要，失败的阶段新建尝试。阶段 04 从最近完整检查点恢复，阶段 06、07 复用有完整摘要的折。修代码后用新 commit、新目录；可加 `--reuse-stage artifacts/member-q8pool-v1:04` 引用旧的下载、审计和第一阶段结果，程序会检查配置、代码、来源锁、抽样摘要、环境和 token 预算。

## 实验与验收

| 阶段 | 得到的证据 |
|---|---|
| 00–01 | 核对 GPU、磁盘、底座分片、train.csv、五折映射和 V31 源码摘要。小 Qwen 训练恢复测试与真实 24 层合成样本的显存、速度、读出对齐分别验收 |
| 02–03 | 固定公开数据、审核来源许可和版本、抽查原文与真实模型标签、分层和去重。数据不合格时保留缺口报告 |
| 04–05 | 仅在公开池训练 S1，用官方样本测 balanced accuracy。低于 0.45 时停止，0.45 至 0.50 之间标记警告 |
| 06–08 | 同一 prompt_id 五折，分别取得两阶段 S2 和官方数据对照 C0 的折外预测。每个验证文件核对所属折的 id，按 prompt 做配对 bootstrap |
| 09 | S2 达 0.56、S2−C0 达 0.05 且区间下界大于 0.02 时，打包五折和全量的六个适配器。核对 V31 原程序与本地路径、fp16 与 bf16 的对齐 |
| 10 | 导出聚合指标、配置和来源 SHA、成员文件清单、资源数据。根 completed.json 给出私有成员包及公开报告路径 |

A1 的三个发布原件和抽查文件已在 [来源锁](../configs/member_q8pool_sources_v1.json) 固定。数据入口和许可口径见[数据说明](member-public-data.md)。上游发布新版时继续使用已锁定的文件；实际贡献来源变为私有或许可不再获准时停止。

F9、F10 用官方文本及题组做跨数据集重合检查和隔离。F11 排版特征只作审计。Open Pool 发布者曾按官方样本筛选公开语料，因此 S1 和 OOF 均须披露上游选择偏差。C0 同折对照衡量整套两阶段配方的差异；两个训练臂的训练步数和学习率也有区别。官方 prompt_id 下的 centered 准确率与每题抽两条实验只作诊断。

## 报告与权重回传

阶段 10 只向 `reports/member-q8pool-v1/` 写白名单聚合报告。训练语料、OOF、逐行预测、原始日志和成员包保存在被 Git 忽略的 artifacts 目录。报告回传前在仓库内运行以下命令，核对暂存文件名并向新的结果分支推送；分支名中的短 SHA 对应正式运行的代码。

```bash
PYTHONPATH=src .venv/bin/python scripts/check_repository.py &&
git add -- reports/member-q8pool-v1/ &&
git diff --cached --stat
```

审阅后提交报告，再将该提交推送至单独的结果分支。Codex 按 summary.json 中的训练 commit、source_lock 和 package_manifest 摘要核对。模型上传到有比赛权限的私有 Kaggle 数据集。上传前核对目标账户；若重用现有句柄，先核对其可见性。新建时及上传后均须核实 Private。只上传根 completed.json 中 package 所指的 kaggle_dataset 子目录。

```bash
../.venv-gpu-check/bin/python - <<'PY'
import json
from pathlib import Path
import kagglehub

record = json.loads(Path("artifacts/member-q8pool-v1/completed.json").read_text())
kagglehub.dataset_upload(
    "huabi1024/ntl-q8pool-member-v1", record["package"],
    version_notes="q8pool-v1 " + record["commit"],
)
PY
```

记录句柄、确切的正整数版本号及 package_manifest.json 的 SHA-256 至 kaggle_dataset.json，一起返还报告。发布方实际挂接该版本，Notebook 启动时逐文件校验。Notebook 由固定源码与真实训练后导出的清单生成。

```bash
PYTHONPATH=src .venv/bin/python scripts/build_v31_member_notebook.py \
  --source artifacts/v31-asset-cache/v31_372363760.json \
  --manifest reports/member-q8pool-v1/package_manifest.json \
  --dataset huabi1024/ntl-q8pool-member-v1 --version "$DATASET_VERSION" --candidate C1
```

Kaggle 私有副本沿用完整 V31 的原挂接版本，再挂接锁定的成员资产。设置 T4 ×2、Internet 关闭，Save & Run All，核对锚点及 submission.parquet。确认当日额度后正式提交一次，把状态和实际公榜成绩写入[提交记录](worklog/submissions.md)。如果原版复跑失败或低于 0.705，先按规格 K0 排查。C2 留作后续预登记候选。

阶段 00 的 `stage.log` 记录 GPU Python 版本、V31 源码来自下载还是缓存，以及通过校验的 SHA-256。五个底座权重分片的预期摘要来自开发机 11 月 23 日的 `v31-claude-local-20261123-121213/completed.json`，阶段 00 会逐个重算。服务器 CPU 依赖安装在 `artifacts/member-cpu-venv`。

阶段 02 按仓库中的快照复现筛选和抽查。实时来源状态保存在本次运行的 `source_status_live.json`；复用阶段时看运行目录下新建的 `compliance-*/source_status_live.json`。报告中的 data 和 s1 会保留门槛警告；M5 可用 `audit_summary.json` 的 `excluded_sources` 核对整源排除的许可和行数。
