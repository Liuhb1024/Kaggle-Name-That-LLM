# SPEC-Q8POOL-V1：在完整 V31 里换掉 q8v5，换成我们用公开数据自训的成员

起草：2026-11-28，评审方 Kiro。执行：Codex。

本文是交给 Codex 的实现规格。数字分三类标注：作者或社区自报、我们实测、我的估算；估算都未经验证。Codex 按第 16 节的顺序实现，门槛见第 8、10.5、12 节，需要用户手动完成的步骤集中在第 15 节。

## 0. 唯一首选方案

**主线 A1**：完整 V31（scriptVersionId 372363760，我们的私有复跑副本 374825819）保持不变，只换一个分支。

- 用核验过的公开数据（Open Pool 第 19 版、Extra API Passages 第 2 版）加官方训练集，分两阶段训练我们自己的成员 `q8pool`：第一阶段只用池数据，第二阶段只用官方数据。
- 底座是 Qwen3-8B-Base，截到 24 层，用 LoRA 微调。
- 按 V31 `run_l24` 的接口打包，替换主混合里的 `q8v5`。两者底座相同、结构相同；q8v5 有 6 个适配器，我们也给 6 个；权重和开关都不改。这个候选记为 **C1**。保留 q8v5、再追加 q8pool 的备选记为 **C2**（第 9.3 节）。

**为什么选它**

1. V31 离 0.750 只差 0.033。轻量方案在社区里的同类结果是 0.592–0.624，差距超过 0.13。
2. q8v5 和我们的成员走同一条代码路径，替换只改一个因素，Kaggle 运行时间也不变。V31 的 GLM 阶段已经按 8.8 小时截止自动缩放覆盖，放不下新分支。
3. 第一阶段完全不接触官方行，所以成员能在全部 2,699 条官方行上做无泄漏评估。V31 里其余社区成员都做不到这一点。
4. 将来切换到轻量方案时，这个成员可以直接复用。

**预期，说实话**

- 单步 C1 达到 0.750 的概率我估计低于 5%，更可能落在 0.715–0.730。
- 这一步主要回答一个问题：更干净、更大的公开数据，能不能训练出明显强于社区的成员。
- 如果能，就沿 A2（同配方训练 14B 成员，替换权重 3.0 的 q14v5）和 A3（家族内专用头）继续推进。两三周内累计达到 0.750 的机会我估计约 15–25%。这些都是主观估计。

**最早能看到的结果**（从服务器开跑算起）

| 时间 | 结果 |
|---|---|
| 约 1 小时 | 冒烟测试（显存、耗时、产物格式）和数据审计 |
| 约 8 小时 | 只用公开数据训练的成员在 2,699 条官方行上的 BA，无泄漏 |
| 约 14 小时 | 五折折外预测（OOF），以及它与只用官方数据的对照之间的配对差值 |
| 打包上传当天 | 可以提交 C1 |

**依据和核查范围**：本规格基于 2026-11-28 的只读核查，包括：

- 仓库 `a37facc`（本地 HEAD 与 GitHub main 一致）加本地未提交的 v31_plain 改动；
- 重新下载的 V31 固定源码，摘要与仓库记录一致；
- 相关 Kaggle 数据集的公开元数据，以及 kagglehub 1.0.0 的源码。

Kaggle 的规则、讨论区、榜单、提交页都需要登录才能渲染，本轮读不到。规则沿用仓库 2026-11-15 的记录。

## 1. 分数依据

| 项目 | 作者或社区自报（未复现） | 我们实测 | 我的估算（待验证） |
|---|---|---|---|
| 完整 V31 | 372363760 标题公榜 0.717；同页后续版本标题 0.724，版本号未核 | 374825819 仍无分数 | 复跑 0.710–0.722 |
| V31 里的 lora24_claude 成员 | 第 0 折 539 行 BA 0.4838，加 Claude 门控 0.4932 | 单分支公榜 0.465；3090 Ti 推理峰值预留 10.19 GiB | — |
| 两阶段 LoRA（Nishant Dahal，池 34,940 段） | 8B、14B 单独本地 CV 都是 0.569；所在 Notebook 公榜 0.592 | — | — |
| 两阶段编码器（Nuri Dudak，池 122,425 段） | ModernBERT OOF 0.569，只用池 0.542；三成员融合 OOF 0.592；加入后公榜 0.620→0.624 | — | — |
| Open Pool 数据实验（Qwen3-1.7B，4.5 万段，只用池） | 清洗 +0.027；Gemini 3.7 只留 Vertex 路由 +0.009；DeBERTa 全池公榜 0.585/0.587 | — | — |
| 社区"加一个组件"带来的公榜增益 | +0.002 到 +0.012 | — | C1 相对复跑 −0.005 到 +0.012 |
| 官方 TF-IDF 基线 | CV 0.3145，公榜 0.322 | 本地 OOF 0.2997（95% 区间 0.285–0.315），公榜 0.322 | — |
| q8pool 第一阶段（S1，只用池） | — | — | 2,699 条官方行 BA 0.53–0.58 |
| q8pool 两阶段 OOF（S2） / 只用官方数据的对照（C0） | — | — | 0.56–0.61 / 0.45–0.50 |
| 轻量方案 B | 同类 Notebook 0.592–0.624 | — | 公榜 0.60–0.66 |
| 公榜前沿 | 2026-11-15 快照，榜首 0.774 | — | 当前值需要登录查看 |

下文"复跑"指 374825819 的公榜分数。

## 2. 两条路线比较

| 维度 | A1：V31，换掉 q8v5 | B：公开数据轻量方案 |
|---|---|---|
| 起点 | 0.717（作者自报） | 无；同类结果 0.592–0.624 |
| 到 0.750 的差距 | +0.033 | +0.13 以上 |
| 离线能验证什么 | 成员能在官方 OOF 上干净评估；整套 V31 只能看公榜 | 全部都能在官方 OOF 上评估 |
| Kaggle 运行时长 | 约 8.8 小时，替换不增加时长 | 约 1–2 小时 |
| 3090 Ti 成本 | 10–14 GPU 小时 | 同一个成员加编码器，15–20 小时 |
| 主要风险 | 社区成员没有干净 OOF；单次公榜差值在 0.010 以内无法解读；依赖 V31 的 39 个挂接别名 | 分数天花板低 |

结论：选 A1，B 作为切换后备（见第 12 节）。

## 3. 范围

**本规格要做的**

- 公开数据的下载、核验、清洗和审计。
- q8pool 两阶段训练、只用官方数据的对照、OOF 评估、打包。
- 服务器一键入口，可以续跑。
- V31 补丁 Notebook 生成器（C1 替换、C2 追加）和 Kaggle 交接材料。

**本规格不做的**

- 自生成数据。
- 任何"让一个模型模仿另一个目标模型、再按目标模型打标"的数据。
- 14B 成员和家族内专用头。
- 改动 V31 的其他成员或混合权重。
- 用公榜选参数。
- 第 5.3 节两项之外的新依赖。

## 4. 固定输入

| 资产 | 固定版本 | 怎么校验 |
|---|---|---|
| 比赛数据 | 服务器 `data/raw/name-that-llm/` | 用 `source.json` 中的 train.csv 摘要复核 |
| 折映射 | `ntl_baseline.baseline.make_folds`（GroupKFold 5 折，shuffle=False） | 按 id 排序后 `id,fold\n` 文本的 SHA-256 等于 `ff3453131989daf2379816eef5ed02086ec0a6488e20815d0bd2ef35c1c40193` |
| 底座 | Kaggle `qwen-lm/qwen-3/transformers/8b-base/1`；服务器缓存 `artifacts/v31-asset-cache/models/qwen-lm/qwen-3/Transformers/8b-base/1` | config 和 index 按 `configs/v31_single_gpu.toml` 中的摘要；分片按已记录的摘要 |
| Open Pool | `crtproduction/name-that-llm-open-pool` 第 19 版（2026-11-25；`ntl_open_pool.parquet` 119,227,255 字节） | 首次下载时写入 `source_lock.json`，之后严格比对。数据集描述里有 238,750 和 223,840 两个行数，以实际文件为准并记录 |
| Extra API Passages | `seyitkaangunes/ntl-extra-api-passages` 第 2 版（`ntl_extra_passages_claude.csv` 4,387,162 字节） | 同上 |
| V31 源码 | `https://www.kaggle.com/kernels/scriptcontent/372363760/download` | SHA-256 `1556c5110b52c917144cd961610f2fc6fd803d4c3982b4e6d2e977273eb9038d` |
| Kaggle 复跑副本 | `huabi1024/name-that-llm-full-v31-reproduction` 版本 374825819 | 逐项记录挂接输入的版本，补丁版不得更新这些输入 |
| GPU 环境 | `requirements-v31-gpu.txt`（torch 2.12.1+cu126、transformers 5.14.1、peft 0.20.0）。transformers 版本与 V31 离线 wheel 一致 | `pip check`；版本写入每个阶段的 `started.json` |
| CPU 环境 | `uv.lock`（Python 3.11、scikit-learn 1.6.1） | `uv run --offline --locked` |

`configs/member_q8pool_v1.toml` 按下列内容建立。字段名可以调整，数值预先登记，不调：

```toml
run_name = "member-q8pool-v1"
seed = 0

[pins]
v31_json_sha256 = "1556c5110b52c917144cd961610f2fc6fd803d4c3982b4e6d2e977273eb9038d"
base_handle = "qwen-lm/qwen-3/Transformers/8b-base/1"
open_pool = "crtproduction/name-that-llm-open-pool/versions/19"
extra_api = "seyitkaangunes/ntl-extra-api-passages/versions/2"
folds_digest = "ff3453131989daf2379816eef5ed02086ec0a6488e20815d0bd2ef35c1c40193"
source_lock = ""        # 首跑后填 reports/member-q8pool-v1/source_lock.json

[filters]
min_words = 36
max_words = 275
gemini_drift_cutoff = "2026-11-01"
kimi_reasoning = ["high"]
near_dup_char_cosine = 0.80
prompt_centroid_cosine = 0.50
typography_never_max_count = 1
spot_check_rows = 200
spot_check_max_mismatch = 0.02
allowed_licenses = ["CC BY 4.0", "CC BY-SA 4.0", "CC0 1.0"]
approved_readme_license_sources = []

[sampling]
per_class_cap = 6000
fallback_per_class_cap = 4000

[model]
keep_layers = 24
pool_layers = [2, 4, 6, 8, 12, 16, 20, 24]
max_len = 384
append_eos = false
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

[train]
token_budgets = [8192, 6144, 4096]
min_rows_per_step = 64
weight_decay = 0.01
warmup_ratio = 0.05
grad_clip = 1.0
stage1 = { epochs = 1, lr_lora = 1e-4, lr_head = 1e-3, checkpoint_every = 500 }
stage2 = { epochs = 2, lr_lora = 5e-5, lr_head = 5e-4, prompts_per_step = 2 }
control = { epochs = 3, lr_lora = 1e-4, lr_head = 1e-3, prompts_per_step = 2 }

[gates]
smoke_peak_reserved_gib = 21.0
smoke_max_projected_stage1_hours = 10.0
data_min_rows_per_class = 1000
data_stop_rows_per_class = 500
data_stop_class_count = 3
s1_continue = 0.50
s1_stop = 0.45
s2_min = 0.56
s2_minus_control_min = 0.05
s2_minus_control_ci_low = 0.02
parity_argmax_agreement = 0.995
parity_ba_diff = 0.003
parity_mean_abs_logp = 0.02
precision_ba_diff = 0.005
anchor_abs_logp = 0.05

[evaluation]
bootstrap_resamples = 10000
pair_replicates = 20
length_bins = [36, 65, 102, 158, 275]

[kaggle]
dataset_handle = "huabi1024/ntl-q8pool-member-v1"
member_name = "q8pool"
weight = 2.0
twin = 0.35
gauss = 0.44
batch_size = 16
```

## 5. 代码改动

### 5.1 复用，不改行为

- `ntl_baseline.data`：`validate_train`、`read_csv_input`、`file_sha256`、`validate_submission`。
- `ntl_baseline.constants`：`LABELS`、`ContractError`。
- `ntl_baseline.baseline.make_folds`、`cross_validate`（用来在服务器重算 E0）。
- `ntl_baseline.artifacts`：`write_json_new`、`write_parquet_new`、`runtime_info`、`directory_hashes`。
- `ntl_repro.branch`：`POOL_LAYERS`、`HIDDEN_SIZE`。
- `ntl_repro.gpu`：截断与加载方式作为参考实现。
- `ntl_repro.run`：run-dir 规则和 CUDA 统计字段。
- `scripts/run_v31_server_smoke.sh`：环境检查和 `KAGGLEHUB_CACHE` 约定。

### 5.2 新增

| 文件 | 职责 | 运行环境 |
|---|---|---|
| `src/ntl_member/config.py` | 读取并校验 TOML | CPU |
| `src/ntl_member/sources.py` | 用 kagglehub 下载固定版本；文件摘要锁；用公开元数据接口查询来源当前的许可和可见性（带 User-Agent，不用凭据） | 下载用 GPU 环境，其余 CPU |
| `src/ntl_member/pool.py` | 统一格式、过滤 F1–F12、分层、去重、泄漏检查、第一阶段抽样、审计 | CPU（需要 sklearn） |
| `src/ntl_member/metrics.py` | BA、每类召回、混淆矩阵、家族指标、按 prompt 的 bootstrap、每题抽 2 条的诊断 | CPU，只用 numpy |
| `src/ntl_member/batching.py` | 按题组装批次、控制 token 预算、结果确定 | CPU |
| `src/ntl_member/features.py` | 截断主干、分词、池化，语义逐条对齐 V31 `l24_predict` | GPU |
| `src/ntl_member/train.py` | 两阶段训练和对照训练、检查点、续训 | GPU |
| `src/ntl_member/package.py` | 导出 V31 格式的成员包、检查键名映射、写 MANIFEST 和锚点 | 导出用 GPU，格式检查用 CPU |
| `src/ntl_member/v31_bridge.py` | 从固定 V31 JSON 提取 `pipe.py`、`gauss.py`、`infer.py` 到运行目录；生成 C1/C2 补丁 Notebook | CPU |
| `src/ntl_member/pipeline.py` | 阶段编排、尝试目录、续跑、复用、门槛、资源采样 | 两种环境 |
| `src/ntl_member/report.py` | 用白名单导出可公开报告 | CPU |
| `src/ntl_member/selftest.py` | 服务器自检：随机初始化的 2 层小 Qwen3 上的训练步、检查点续训、safetensors 头格式、键名映射。GPU 环境不装 pytest，所以写成 `python -m ntl_member.selftest` | GPU |
| `scripts/run_member_server.sh` | 服务器一键入口 | — |
| `scripts/build_v31_member_notebook.py` | 补丁 Notebook 生成器 | CPU |
| `configs/member_q8pool_v1.toml` | 第 4 节配置 | — |
| `docs/member-q8pool.md` | 运行手册：命令、人工步骤、门槛 | — |
| `tests/test_member_*.py` | 合成数据测试 | CPU |

### 5.3 修改现有文件

- `requirements-v31-gpu.txt`：加入 `scikit-learn==1.6.1` 和 `scipy==1.15.3`。V31 的 `infer.py` 在导入时就需要这两个包，服务器做对齐测试必须执行它。两者版本与 `uv.lock` 相同，许可为 BSD。在 `docs/dependency-decisions.md` 写明原因。
- `pyproject.toml`：mypy 检查范围加上 `src/ntl_member`。
- `scripts/check_repository.py`：增加一条检查，`reports/**/*.json` 中不得出现 `text`、`id`、`prompt_id`、`prompt_canon` 键，也不得出现绝对路径。
- `docs/running.md` 加链接到 `docs/member-q8pool.md`；同步更新 README 的文件树、`directory_structure.txt` 和 `docs/worklog/`。本规格文件本身也要登记进这两处。
- `AGENTS.md`：阶段边界由用户确认文字后再改（M0）。

### 5.4 不得改动

- `data/raw/`、`docs/research/2026-11-15/snapshot/`、已有的 `artifacts/*` 历史目录。
- `ntl_baseline` 和 `ntl_repro` 的现有行为。如果确需修复，先写失败测试，并单独说明。
- V31 源码和补丁后的 Notebook 不进 Git，只写到 `artifacts/notebooks/`。V31 源码的再发布许可没有核实。
- 用户本地尚未提交的 v31_plain 改动（M1）不得删除或覆盖。

## 6. 数据规格

### 6.1 核验

- **比赛规则**：仓库 2026-11-15 的记录，加上 Open Pool 引用的 752084 号帖（2026-10-28），说法一致：已公开的社区数据可以训练，包括通过 Benchmarks proxy 生成的数据；proxy 配额不应再用于生成新数据。开跑前由用户复查讨论区（M2）。
- **许可和可见性**：对每个来源调用公开元数据接口，查询当前的许可字段和可见性。
  - 只接受 CC BY 4.0、CC BY-SA 4.0、CC0 1.0。
  - 许可只写在 README 里的来源（例如 kilupy 的两个数据集）默认排除，列入待用户批准的清单。
  - 已经不公开的来源排除。
- **生成日期**：用行级 `gen_date`。缺失时用来源的 `lastUpdated` 作为上界，并加标记。
- **实际模型 ID**：用行级 `gen_served_model`。与标签不一致的行排除。
- **路由**：用 `gen_route`。不读取 `gemini37_unverified.parquet`。
- **禁止模仿数据**：如果 `source_protocol.csv` 声明某来源做过改写、风格迁移、润色，或由非标签模型生成，整源排除。字段映射以实际文件为准，并在审计中列出。流水线本身不生成任何文本。缺少可靠样本只记录缺口，不补造。
- **来源抽查**：取保留行数前 5 的来源，加上所有提供 A 层行的来源，最多 10 个。下载它们当前的版本，各随机抽 200 行，比对文本（空白规范化后）和标签。不一致率超过 2% 的来源整源排除。

### 6.2 过滤（顺序执行，逐条计数）

- **F1**：标签必须属于 18 类。
- **F2**：词数 36–275。
- **F3**：单段文本，不含 Markdown 标题、列表、代码块，不含 HTTP 或 JSON 残留。
- **F4**：返回模型 ID 与标签不符，排除。
- **F5**：Gemini 四个标签中，`gen_date ≥ 2026-11-01` 的行排除。Open Pool 记录到 11 月生成的 Gemini 有 20–70% 的段落含 em dash，训练集只有 0–1%。
- **F6**：kimi-k3 只保留 `gen_reasoning == "high"` 的行。
- **F7**：许可或可见性不合格，排除。
- **F8**：同一段规范化文本带多个标签，全部排除。
- **F9**：与任一官方段落的字符 3–5 gram TF-IDF 余弦 ≥ 0.80，排除。向量化器只在官方行上拟合。
- **F10**：与任一官方 prompt 质心的词级 TF-IDF 余弦 ≥ 0.50，排除。质心是该题 18 段向量的均值，去掉英文停用词。
- **F11**：如果某个标签的官方行里某排版标记出现不超过 1 次（em dash、空格 em dash、弯撇号、弯双引号），池里同标签带该标记的行排除。这条规则用到了全部官方行的统计，记为已知的轻微乐观偏差。
- **F12**：来源抽查不合格，整源排除。

另外两点：

- 保留带 `flag_kb_proxy` 或 `flag_kb_proxy_after_ruling` 标记的行，但单独计数。
- 不使用 `match_weight` 和 `style_score`。这两个字段是用全部官方行拟合的，会污染折外评估。

### 6.3 标签可信度分层

- **A 层**：满足以下任一条件。
  - 行级返回模型 ID 与标签完全一致，且记录了路由；
  - 来源是第一方 API，协议声明逐次核对返回模型，且 0 不一致。Extra API Passages 属于这种情况。
- **B 层**：不属于 A 层，但记录了路由，或协议声明做过返回模型核对。
- **C 层**：其余行，标签只有发布者自己的声明。

带快照后缀的模型 ID（例如 `qwen3.8-max-0902`）不算完全一致，最多归入 B 层，并加 `served_model_variant` 标记。

### 6.4 去重与统一格式

- **去重**：规范化键为 NFKC、合并空白、小写。Open Pool 和 Extra API 合并后按键去重，A 层优先保留，同层内按来源名排序，保证结果确定。
- **私有文件 `pool_rows.parquet` 的列**：`row_key`、`text`、`label`、`family`、`source_dataset`、`source_version`、`source_file`、`source_license`、`license_basis`、`prompt_group`（`prompt_canon` 或来源内 id）、`tier`、`gen_route`、`gen_served_model`、`gen_date`、`gen_reasoning`、`flag_kb_proxy`、`flag_kb_proxy_after_ruling`、`n_words`、`text_sha256`、`kept`、`excluded_by`。

### 6.5 第一阶段抽样

- 每类最多 6,000 行，按 A→B→C 的顺序填满。层内先按 `text_sha256` 排序，再用种子 0 抽样。
- 损失按类别频率的倒数加权。
- 写出 `stage1_sample.parquet` 和每类的构成。

### 6.6 审计产物

私有，全部写在运行目录：

- `audit/per_class.json`：各层、各路由、各日期段（10 月及以前 / 11 月 / 未知）、各推理设置、proxy 标记的计数；词数分位；排版标记出现率，以及官方同类对照。
- `audit/per_source.json`：许可及其依据、可见性、版本、各条规则剔除的行数、协议字段、抽查结果。
- `audit/filters.json`、`audit/leakage.json`、`audit/gaps.json`、`source_lock.json`。

公开部分汇总进 `audit_summary.json`（第 11.1 节）。

### 6.7 数据门槛

- **通过**：固定输入全部校验一致；每类保留 ≥ 1,000 行；泄漏检查后剩余违规为 0。
- **记缺口后继续**：有类别只有 500–999 行。
- **停止**：3 个及以上类别少于 500 行，或者许可和可见性核验无法完成。

## 7. 模型与训练

### 7.1 结构（逐项对齐 V31 q8v5 的推理语义）

- **主干**：`AutoModel`（不带 LM 头），`num_hidden_layers=24`，`use_cache=False`，`layer_types` 同步截断。
- **分词**：用底座目录里的 tokenizer，调用 `tok(texts, truncation=True, max_length=384)`，默认特殊 token，不追加 EOS；padding 用 eos，右侧填充。
- **池化**：取 hidden_states 索引 [2,4,6,8,12,16,20,24]；屏蔽第一个 token，在 fp32 下求均值；每层各接一个 LayerNorm(4096)，拼成 32,768 维。
- **分类头**：`plain` 和 `centered` 两个 Linear(32768→18)，都用 fp32。`centered` 的输入是 LayerNorm 之后、减去同组均值的特征。
- **LoRA**：PEFT `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none")`，作用在 7 个线性投影上。
  - 必须挂在 `AutoModel` 上，保证权重键名形如 `base_model.model.layers.N.…`。V31 合并 LoRA 时会去掉这个前缀，再按模块名查找。
  - 禁用 DoRA 和 rsLoRA。V31 按 `lora_alpha / r` 计算缩放。

### 7.2 训练（预先登记，不调参）

| | 第一阶段 | 第二阶段（5 折 + 全量） | 只用官方数据的对照（5 折） |
|---|---|---|---|
| 初始化 | 底座 | 第一阶段权重 | 底座 |
| 数据 | `stage1_sample` | 训练折，或全部 2,699 行 | 训练折 |
| epoch | 1 | 2 | 3 |
| 学习率（LoRA / 头与 LN） | 1e-4 / 1e-3 | 5e-5 / 5e-4 | 1e-4 / 1e-3 |
| 批次 | 在 token 预算内按题打包，累积到每步 ≥ 64 行 | 每个微批 1 个完整 prompt，每步累积 2 题 | 同第二阶段 |

**共同设置**

- AdamW，β 取 0.9/0.999，weight decay 0.01，LayerNorm 和 bias 不衰减。
- 5% 预热后余弦衰减到 0；梯度裁剪 1.0。
- bf16 autocast；梯度检查点开启，`use_reentrant=False`；种子 0。

**损失**：加权 CE(plain)，加上加权 CE(centered)。centered 那一项只在 ≥ 2 行的组上计算。池数据按 `prompt_canon` 分组；没有组的行只参与 plain 损失。

**token 预算**：在冒烟阶段从 {8192, 6144, 4096} 中选出能通过的最大值，之后全程固定。

**打包用的 6 个适配器**：第二阶段的 5 个折模型加 1 个全量模型，顺序为 f0–f4、最后全量，和 q8v5 相同。

### 7.3 检查点与续训

- 第一阶段每 500 步保存一次，结束时再存一次。内容包括 LoRA、头、LN、优化器、调度器、RNG 状态、数据游标，以及记录各文件 SHA-256 和 global_step 的 `checkpoint.json`。
- 续训只能从通过校验的最新检查点开始，并且写到新的尝试目录里。
- 第二阶段和对照每折结束时保存。某一折失败，就在新尝试里从头重训这一折。

## 8. 评估

**对比臂**（都在同一张折映射、同一批 2,699 行上）

| 臂 | 内容 |
|---|---|
| S1 | 第一阶段模型，对全部行预测 |
| S2 | 两阶段 5 折 OOF |
| C0 | 只用官方数据的对照，5 折 OOF |
| E0 | 参考：官方 TF-IDF 基线，在服务器按同一折映射重算 |

**每个臂报告的指标**

- OOF BA（主指标）和 5 折各自的 BA。
- 每类召回，附按 prompt 重采样的 95% 区间。
- 18×18 混淆矩阵，给计数和按行归一化两种。
- 四个多成员家族各自的家族内混淆矩阵。
- FamAcc（判对家族的比例）、SibAcc（已知真实家族时判对子类的比例，给总体和分家族）。
- BA 损失拆成跨家族和同家族两部分。
- 按词数四分位（36–64 / 65–101 / 102–157 / 158–274）分别算 BA。

**配对比较**

- 150 个 prompt 有放回重采样 10,000 次，两臂共用同一批重采样。
- 报告差值 Δ、95% 区间、Δ>0 的比例。
- 主比较是 S2−C0；次比较是 S2−S1 和 S1−C0。

**诊断项，不进门槛**

- centered 头在真 prompt_id 分组下的 BA。
- 每题抽 2 条、在这两条内做中心化后的 BA，重复 20 次，和同样抽样下的 plain 头配对比较。

**门槛**

- **S1**：BA ≥ 0.50 继续；0.45–0.50 继续，并在报告里标警告；低于 0.45 停止排查。
- **S2**：S2 ≥ 0.56，且 S2−C0 ≥ +0.05、区间下界 > +0.02，才打包。达不到就停止，不进 Kaggle。

## 9. 打包与接入 V31

### 9.1 成员包

私有 Kaggle 数据集 `huabi1024/ntl-q8pool-member-v1`，结构如下：

```text
q8pool_f0/ … q8pool_f4/  q8pool/
  adapter/adapter_config.json
  adapter/adapter_model.safetensors
  heads.safetensors   # norms.{0..7}.weight/bias, plain.weight/bias, centered.weight/bias（float32）
  meta.json           # labels, cfg{keep:24, layers, max_len:384, eos_id:null, no_final_norm:false}, git_commit, config_sha256, data_lock_sha256
anchors/test.csv                # 3 条合成文本，不是比赛文本
anchors/expected_logits.json    # 服务器在 fp16 下经 V31 infer.py 算出的 18 维对数概率
MANIFEST.json                   # 每个文件的字节数和 SHA-256
NOTICE.md                       # Qwen3-8B-Base 用 Apache 2.0；训练数据来源及许可；比赛数据 CC BY-SA 4.0
```

打包时检查键名：去掉 `base_model.model.` 之后的每个模块名，都必须能在 24 层 `Qwen3Model.named_modules()` 里找到。

### 9.2 服务器上的对齐与锚点

- **锚点**：用只含 q8pool 的模型列表，参数 `w=1.0, gauss=0, twin=0, dp=False`，运行：

  ```bash
  infer.py --tfidf_w 0 --distort_w 0 --svc_w 0 --adapt_w 0 --save_final_logits …
  ```

  输入是 3 条锚点文本，结果写入 `expected_logits.json`。`--data` 指向一个临时目录，其中 `train.csv` 是官方训练集的符号链接，`test.csv` 是锚点。
- **官方行对齐**：用第 0 折的验证行，只挂 `q8pool_f0`，比较 `infer.py` 和我们自己的评估路径。要求：
  - argmax 一致率 ≥ 99.5%；
  - BA 差 ≤ 0.003；
  - 平均 |Δlog p| ≤ 0.02。

  transformers 5.14.1 中第 24 层是否经过 final norm 没有核实，靠这项测试兜底。
- **精度**：第 0 折分别用 fp16 和 bf16 跑一遍。要求 BA 差 ≤ 0.005，argmax 一致率 ≥ 99%。

### 9.3 V31 补丁（`scripts/build_v31_member_notebook.py`）

- **输入**：固定的 V31 JSON（先校验 SHA-256）、`package_manifest.json`、数据集的句柄和版本、候选类型（C1 或 C2）。
- **输出**：`artifacts/notebooks/v31-q8pool-<c>.ipynb` 和一份改动清单 `.cells.md`，Notebook 不含执行输出。
- 生成器按内容标记定位格子，不只靠下标：
  - 资产设置格含 `AQ8V5 = [adapter_dir(`；
  - comp_stack 写出格以 `%%writefile comp_stack.py` 开头；
  - 运行格以 `def run(cmd, env=None):` 开头；
  - 成员格含 `MODELS = [` 和 `"name": "q8v5"`。

只允许改动三处：

1. **资产设置格之后插入新格 A**：
   - 解析数据集路径，兼容 `/kaggle/input/ntl-q8pool-member-v1` 和 `/kaggle/input/datasets/huabi1024/ntl-q8pool-member-v1` 两种挂载方式；
   - 按 MANIFEST 校验全部文件的 SHA-256；
   - 定义 `AQ8POOL`（f0–f4 加全量）和从 `meta.json` 读出的 `Q8POOL_CFG`。
2. **comp_stack 写出格之后、运行格之前插入新格 B**（放在 comp_stack 运行之前，失败可以尽早暴露）：
   - 定义 `run_logged()`，和原来的 `run()` 相同，但把每一行输出也存进 `RUN_LOG`；
   - 执行锚点预检：用 9.2 的同一条命令，与 `expected_logits.json` 比较。要求 |Δ| ≤ 0.05 且 argmax 相同，否则报错。
   - 预检在保存运行（FAST_COMMIT）和评分重跑里都执行。新格自己导入所需的模块。
3. **成员格**：
   - C1 把 q8v5 那一行换成下面这一行；C2 保留 q8v5，并在它后面追加同一行：

     ```python
     {"name": "q8pool", "type": "l24", "path": B8, "adapters": AQ8POOL, "keep": 24, "layers": LAYERS, "w": 2.0, "bs": 16, "max_len": 384, "twin": 0.35, "gauss": 0.44, "dp": True}
     ```

   - 把 `code = run(cmd, ...)` 改成调用 `run_logged`。
   - 只在非 FAST_COMMIT 分支里加断言：6 条 `l24 q8pool: adapter … done` 都出现了，并且没有出现 `MEMBER FAILED q8pool`。原因是 V31 在成员失败时只记日志、继续运行，失败会是静默的。

**生成器必须断言**

- 原成员格恰好有一行 q8v5 的定义，恰好有一处 `code = run(cmd`；
- 其余所有格的源码哈希与原版相同；
- 输出的 Notebook 没有执行输出。

### 9.4 Kaggle T4 运行要求

- **挂接**：374825819 的全部原始输入，版本不变；再加我们数据集的固定版本。
- **环境**：GPU T4 ×2，关闭 Internet，不安装新包（只用 V31 自带的离线 wheel）。
- **显存**：成员用 fp16 推理，每张卡放一份，约 9.8 GiB，和 q8v5 相同。
- **时长**：C1 与原版相同，另加锚点预检约 3–5 分钟。C2 再多约 20 分钟，这部分由 GLM 阶段按截止时间自动缩小覆盖来吸收。
- **保存运行**（3 行公开样例）必须成功，并且日志显示锚点预检通过。评分重跑由 9.3 第 3 条保证成员没有静默失败。

## 10. 服务器入口与运行

### 10.1 命令

```bash
cd ~/kaggle-name-that-llm-lhb/Kaggle-Name-That-LLM
git fetch origin
bash scripts/run_member_server.sh --commit <40位SHA> \
  --python ../.venv-gpu-check/bin/python \
  --run-root artifacts/member-q8pool-v1
# 续跑：同一命令加 --resume；只看阶段计划：加 --dry-run
```

脚本按以下步骤执行：

1. 确认工作区干净（被忽略的文件除外），切到 `git switch --detach <SHA>`，再用 `git rev-parse HEAD` 核对。
2. GPU 环境检查沿用 `scripts/run_v31_server_smoke.sh`：torch 2.12.1+cu126、恰好 1 张卡。只在缺包时按 `requirements-v31-gpu.txt` 安装一次，然后跑 `pip check`。
3. CPU 阶段用 `uv run --offline --locked`，GPU 阶段用指定的 Python；`KAGGLEHUB_CACHE=artifacts/v31-asset-cache`，复用已有的底座缓存。
4. 依次执行各阶段。冒烟通过后自动推进到报告阶段。任何门槛失败都停在该阶段，以非零退出，并打印续跑命令。
5. 结束时打印完成报告和公开报告的路径。

### 10.2 阶段

| 编号 | 阶段 | 环境 | 主要产物 | 估时 |
|---|---|---|---|---|
| 00 | preflight | 两者 | 各库版本、GPU、磁盘（≥ 60 GB 空闲）、底座与 train.csv 摘要、折摘要 | 5–10 分钟 |
| 01 | smoke | GPU | 合成数据上的峰值显存、tokens/s、选定的 token 预算、检查点续训、打包格式、V31 锚点路径；`python -m ntl_member.selftest` | ≤ 45 分钟 |
| 02 | download | GPU 环境（kagglehub） | 固定版本文件、`source_lock.json`、来源状态与许可、抽查用的来源文件 | 10–20 分钟 |
| 03 | data | CPU | `pool_rows.parquet`、`stage1_sample.parquet`、`audit/*`、数据门槛 | 20–40 分钟 |
| 04 | stage1 | GPU | 第一阶段权重、训练曲线、检查点 | 5–7 小时 |
| 05 | eval_s1 | GPU 推理，CPU 算指标 | S1 概率（私有）、指标、S1 门槛 | 约 10 分钟 |
| 06 | stage2 | GPU | 5 个折模型、OOF 概率、全量模型 | 1.5–2.5 小时 |
| 07 | control | GPU | C0 的 OOF 概率 | 1.5–2.5 小时 |
| 08 | eval_s2 | CPU | E0 重算、全部指标、配对比较、S2 门槛 | 约 10 分钟 |
| 09 | package | GPU | 成员包、锚点、三项对齐结果、MANIFEST | 30–40 分钟 |
| 10 | report | CPU | 私有 `REPORT.md`、公开 `reports/member-q8pool-v1/` | 约 2 分钟 |

合计约 10–14 小时，以冒烟阶段的实测外推为准。

### 10.3 资源估算（待冒烟实测）

| 场景 | 估计峰值显存 | 依据 |
|---|---|---|
| 第一阶段训练，微批 16 条 × 384 token | 14–18 GiB | 24 层 8B 的 16 位权重约 9.8 GiB；LoRA 约 2,900 万参数，连同 AdamW 状态约 0.5 GiB；梯度检查点下的激活 2–4 GiB；CUDA 上下文与碎片 1–2 GiB |
| 评估推理 | 10–11 GiB | lora24_claude 推理实测峰值预留 10.19 GiB |
| Kaggle T4 推理 | 每卡权重约 9.8 GiB | 与 q8v5 相同 |

第一阶段耗时的估算方法：约 10 万段，每段约 140 token，批内填充约 1.3 倍；8B 截 24 层每 token 前向约 9.3 GFLOPs，带检查点的 LoRA 训练约为前向的 4 倍；3090 Ti 有效算力按 30–40 TFLOPS 计，得 5–7 小时。

### 10.4 目录、日志与续跑

```text
artifacts/member-q8pool-v1/
  logs/pipeline.log                    # 追加写：阶段起止、门槛结果、续跑记录
  00-preflight/attempt-1/
    started.json  stage.log  resources.json  completed.json
  04-stage1/attempt-1/
    started.json  stage.log  gpu_samples.csv  checkpoints/step-000500/  failed.json
  04-stage1/attempt-2/                 # 续跑时新建；started.json 记录 resumed_from
    ...  completed.json
  completed.json                       # 全部阶段通过后最后写入
  REPORT.md                            # 私有完整报告
```

- 尝试目录只有成功后才写 `completed.json`。失败时写 `failed.json`，只含异常类型、阶段和不带敏感信息的消息；完整 traceback 留在 `stage.log`。
- `--resume`：
  - 跳过已完成的阶段，跳过前先核对其产物摘要和上游引用；
  - 为第一个未完成的阶段新建尝试目录；
  - 第一阶段可以从上一次尝试最新的有效检查点继续。
- 门槛失败不能靠续跑绕过。修好代码后用新 commit 和新 run-root。
- `--reuse-stage <旧 run-root>:<阶段>`：只用于 02–04。要求该阶段的配置段、`source_lock`、相关模块的源码哈希与当前完全一致。新 run-root 只记录引用，不复制、不修改旧目录。
- 资源记录：
  - GPU 阶段在后台运行 `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv -lms 1000`，写入 `gpu_samples.csv`。
  - 每个阶段还用 `torch.cuda.reset_peak_memory_stats()`、`max_memory_allocated()`、`max_memory_reserved()` 和墙钟时间记录资源。
  - 两种显存数字都要报，因为 PyTorch 的统计不含 CUDA 上下文。

### 10.5 首跑验收（冒烟门槛）

全部满足才自动进入下载和数据阶段：

- 在选定的 token 预算下没有 OOM，峰值预留 ≤ 21 GiB。
- 按实测 tokens/s 外推第一阶段 ≤ 10 小时；否则改用预先登记的每类 4,000 行上限，并记录。
- 检查点续训：从第 10 步续训后，学习率、global_step、数据游标与不中断的运行完全一致，第 11–20 步损失的相对差 ≤ 2%。
- 用合成数据训练出的成员包能被 V31 `infer.py` 读取，输出全是有限值；我们的特征路径与 `infer.py` 的相对误差 ≤ 0.02。
- `python -m ntl_member.selftest` 全部通过。
- 冒烟总时长 ≤ 45 分钟。

## 11. 回传与交接

### 11.1 公开报告

`reports/member-q8pool-v1/` 下只放 `.json` 和 `.md`。`check_repository.py` 会拒绝 Git 可见的 csv、parquet 和权重文件。

- `summary.json`、`summary.md`：commit、配置摘要、输入版本与摘要、折摘要、各臂的聚合指标与区间、门槛结果、资源摘要、环境版本。
- `source_lock.json`：下载文件的版本、字节数、SHA-256。
- `audit_summary.json`：每类、每个来源的计数，缺口，待批准的来源。
- `package_manifest.json`：成员包每个文件的字节数和 SHA-256。
- `kaggle_dataset.json`：上传后由用户填写句柄和版本号。

禁止出现：文本、id、prompt_id、逐行数组、绝对路径、主机名、日志正文、凭据。`report.py` 按白名单生成并扫描敏感模式，`check_repository.py` 再查一遍。

### 11.2 权重交接

用户在服务器上运行下面的命令（Codex 在 `docs/member-q8pool.md` 里写出最终版本）：

```bash
../.venv-gpu-check/bin/python -c "import kagglehub; kagglehub.dataset_upload('huabi1024/ntl-q8pool-member-v1', 'artifacts/member-q8pool-v1/09-package/attempt-N/kaggle_dataset', version_notes='q8pool-v1 <commit>')"
```

- kagglehub 1.0.0 的源码在新建数据集时会设置 `is_private = True`。上传后仍要在网页上确认可见性是 Private，并记下版本号。
- 私有阶段的许可字段可以填 Other，在 `NOTICE.md` 写明各组成部分的许可。以后如果要公开，需要重新评估 CC BY-SA 的相同方式共享要求。
- 训练检查点、池数据、逐行预测、私有日志只留在服务器的 `artifacts/`。

### 11.3 Kaggle 操作（Codex 在用户授权的会话中进行）

1. 拉取 GitHub，核对 `summary.json` 里的 commit 与 `package_manifest.json`，确认所有门槛都已通过。
2. 在本地生成 C1 Notebook（第 9.3 节），把 `.cells.md` 交给 Kiro 审查。
3. 在私有副本上新建版本：
   - 导入 C1 Notebook，添加我们数据集的固定版本；
   - 逐项核对其余输入的版本与 374825819 相同，Kaggle 提示更新输入时一律不更新；
   - 选 GPU T4 ×2，关闭 Internet。
4. Save & Run All：确认日志里锚点预检通过，`submission.parquet` 通过 V31 自带的检查。
5. 用户确认当天额度后再提交。

### 11.4 提交记录

`docs/worklog/submissions.md` 每次记录：UTC 时间、Notebook 版本号、候选（复跑、C1、C2）、我们数据集的版本、当日额度、状态、公榜分数、备注。失败的提交也要记。

## 12. 继续、停止与切换

| 编号 | 条件 | 动作 |
|---|---|---|
| K0 | 374825819 报错，或分数低于 0.705 | 不提交 C1；先排查复跑（挂接版本、超时）；2 天内恢复不了，就把主线切到 B |
| K1 | 数据门槛判定停止 | 不训练；报告缺口 |
| K2 | S1 低于 0.45 | 停止，排查数据或实现 |
| K3 | S2 低于 0.56，或 S2−C0 低于 +0.05 | 不进 Kaggle；记录：按本配方，公开数据路线无效 |
| K4 | C1 比复跑高 0.010 或更多 | 写 A2 规格：同配方训练 Qwen3-14B 成员替换 q14v5，先做显存可行性检查 |
| K5 | C1 与复跑之差的绝对值小于 0.010 | 只提交一次预先登记的 C2；仍然不显著，就停止换成员，转为评估 A3（家族内专用头） |
| K6 | C1 比复跑低 0.010 或更多 | 回到复跑版，不做 C2；先对比 q8pool 和社区成员的每类召回，再决定下一步 |
| K7 | 出现资产可核验、公榜 ≥ 0.735 的公开 Notebook | 先单独复跑它，再把同一个换成员方案移植过去 |
| K8 | 任何泄漏检查或对齐检查失败 | 暂停解读结果，修复后用新 run-root |

阈值 0.010 的来由：公榜约 1,600 行，C1 只会改变少数预测，我估算两次提交配对差值的标准误约 0.005（未验证）。差值在 ±0.010 以内只能算不确定。公榜不用来在候选之间选参数。

## 13. 测试与本地验证

按 AGENTS.md 先写失败测试再实现，保留先失败的输出。

**Mac 本地**（`uv run --offline --locked pytest -q`）覆盖：

- 配置校验；来源锁比对（伪造文件）；元数据查询（伪造响应，确认带 User-Agent、不读凭据）。
- F1–F12 每条规则的正例和反例；分层；去重结果确定；泄漏阈值；抽样上限和确定性；审计计数自洽。
- 按题打包与 token 预算。
- 指标在已知混淆矩阵上的数值；两臂相同时 bootstrap 区间宽度为 0；每题抽 2 条的抽样确定。
- 尝试目录、completed/failed 语义、续跑跳过、门槛停止、`--reuse-stage` 哈希校验、`--dry-run`。
- 报告白名单拒绝 text、id 和绝对路径；`check_repository.py` 的新规则。
- 补丁生成器：用合成 Notebook 夹具，确认只改三处、无执行输出；定位标记缺失或重复时报错。
- 键名映射：用按 Qwen3 命名规则生成的模块名集合验证。

**服务器**：`python -m ntl_member.selftest`，在冒烟阶段自动执行。

**静态检查**：

- `uv run --offline --locked ruff check src tests scripts`
- `uv run --offline --locked ruff format --check src tests scripts`
- `uv run --offline --locked mypy`（范围是 `src/ntl_baseline` 加 `src/ntl_member`）
- 两个现有 Notebook 的 `scripts/build_notebook.py --check`
- `uv run --offline --locked python scripts/check_repository.py`
- `git diff --check`

测试和夹具都不得包含比赛原文。

## 14. 风险与已知缺口

- 社区成员的训练数据未知，C1 的增益只能在公榜上看到；单次差值在 0.010 以内无法解读。
- F11 和 Open Pool 自身的排版过滤用到了全部官方行，OOF 会轻微偏乐观。S1 的"无泄漏"只是指训练没见过官方文本。
- transformers 5.14.1 中第 24 层 hidden state 是否经过 final norm 未核实，由第 9.2 节的对齐测试兜底。
- Kaggle 评分重跑的日志看不到，只能靠成员强制检查发现失败。
- V31 依赖 39 个挂接别名，其中任何一个被作者改为私有，副本都可能失效（K0）。
- 只在 README 声明许可的来源默认排除，Claude 的第一方数据可能因此变少。
- 用 CC BY-SA 数据训练出的权重如何再发布还没有定论，本轮只建私有数据集。获奖交付前需要另行确认。
- Kaggle 的规则、讨论、榜单页本轮读不到，规则沿用 2026-11-15 的记录。
- 第 1 节和第 10.3 节的全部估算都未经验证。

## 15. 需要用户手动完成的步骤

- **M0**：确认 AGENTS.md 的新阶段边界，由 Codex 修改。建议文字："2026-11-28 起：在完整 V31 内只替换 q8v5 一个成员；允许使用 Qwen3-8B-Base 和 SPEC-Q8POOL-V1 第 6 节核验后的公开数据；禁止自生成数据和模仿标注；超参与门槛按规格预先登记，不按公榜调参；每次正式提交前由用户确认当日额度。"
- **M1**：把本地未提交的 v31_plain 改动（6 个文件）提交并推送；或者明确让 Codex 把它们并入首个 commit。
- **M2**：登录 Kaggle 查三件事：
  - 752084 号帖等规则答疑有没有更新；
  - 374825819 的状态和分数；
  - Code 页有没有资产可核验、公榜 ≥ 0.735 的公开 Notebook（K7）。
- **M3**：在服务器配置 Kaggle 凭据：`~/.kaggle/kaggle.json`（权限 600）或环境变量 `KAGGLE_API_TOKEN`。不要写进仓库、日志或聊天。
- **M4**：用 review 通过并推送的 commit 运行第 10.1 节的命令。失败时按提示续跑，把终端最后 40 行发回（先去掉凭据）。
- **M5**：审阅 `audit_summary.json` 里待批准的来源。默认排除就能继续；如果要纳入，作为下一版（v2）处理。
- **M6**：全部阶段通过后，审阅 `git diff reports/member-q8pool-v1/`，提交并推送。
- **M7**：按第 11.2 节上传成员包，在网页确认 Private，把句柄和版本号写进 `kaggle_dataset.json` 并推送。
- **M8**：授权 Codex 的 Kaggle 会话，或者自己按第 11.3 节操作。每次正式提交前确认当天额度。
- **M9**：公榜出分后，在 `submissions.md` 补记，或让 Codex 补记。

## 16. 实施清单（Codex）

1. 从包含 v31_plain 改动的 commit 开始。先跑一遍全部现有检查，保存基线输出。
2. 写 `configs/member_q8pool_v1.toml` 和 `config.py`，先写测试。
3. 写 `metrics.py`，先写测试。
4. 写 `sources.py`，先写测试。
5. 写 `pool.py`，先写测试，F1–F12 每条都要有正例和反例。
6. 写 `batching.py`，先写测试。
7. 写 `pipeline.py`：阶段、尝试目录、续跑、复用、门槛、资源采样、`--dry-run`，先写测试。
8. 写 `report.py` 和 `check_repository.py` 的新规则，先写测试。
9. 写 `features.py`、`train.py`、`package.py`、`selftest.py`。键名映射和 heads 键名与形状的检查放在 CPU 测试里，其余由服务器自检覆盖。
10. 写 `v31_bridge.py` 和 `scripts/build_v31_member_notebook.py`，先用合成夹具写测试。
11. 写 `scripts/run_member_server.sh`：参数校验、工作区不干净时拒绝运行、`--dry-run` 打印阶段计划。
12. `requirements-v31-gpu.txt` 增加两项依赖，并更新 `docs/dependency-decisions.md`。
13. 文档：`docs/member-q8pool.md` 运行手册、`docs/running.md` 里的链接、README 文件树、`directory_structure.txt`（含本规格文件）、`docs/worklog/` 决策记录和 `submissions.md` 模板。
14. 跑全部检查命令，按第 17 节整理 review 材料。

## 17. 需要看到的 review 证据

**第一轮：代码，推送前**

- 改动文件清单和完整 diff，逐条对应本规格的章节。
- 先失败的测试输出，以及实现后的完整测试输出。
- ruff、mypy、两个 Notebook `--check`、`check_repository.py`、`git diff --check` 的输出。
- 在合成数据上跑 CPU 阶段（data、eval、report）的目录树、`completed.json` 和公开报告样例，证明报告里没有文本和 id。
- `run_member_server.sh --dry-run` 的输出。
- 补丁生成器的两份输出：合成夹具上的结果；用真实 V31 JSON 在本地生成的 `.cells.md`（私下交给 Kiro）。需要显示其余格的哈希不变。
- 依赖变更说明。
- 已知缺口清单：哪些只能在服务器或 Kaggle 上验证。

**第二轮：服务器结果**

- `completed.json`、各阶段的门槛结果、公开报告。
- 每个阶段的墙钟时间、峰值显存、tokens/s。
- 三项对齐结果和 MANIFEST 摘要。
- 失败过的尝试及其原因。

**第三轮：Kaggle**

- 新版本号和输入版本对照表。
- 保存运行日志里锚点预检和提交检查的摘录。
- 提交编号、当日额度、公榜分数。

## 18. 服务器与 Kaggle 交接清单

【用户】【Codex】【Kiro】【服务器】标明每一步由谁完成。

- [ ] 【用户】完成 M0、M1，告诉 Codex 起始 commit。
- [ ] 【Codex】完成实现，提交第 17 节第一轮证据。
- [ ] 【Kiro】给出 review 结论：通过、需修复或证据不足。
- [ ] 【用户】推送 review 通过的 commit，记下 40 位 SHA。
- [ ] 【用户】完成 M2、M3。
- [ ] 【服务器】运行第 10.1 节命令；冒烟通过后自动推进到报告阶段。
- [ ] 【服务器】中断时用 `--resume` 续跑；门槛失败时停下，把 `failed.json` 和门槛结果交给 Codex。
- [ ] 【用户】完成 M5、M6：审阅并推送 `reports/member-q8pool-v1/`。
- [ ] 【Codex】拉取报告，核对 commit 和成员包摘要；【Kiro】做第二轮 review。
- [ ] 【用户】完成 M7：上传私有数据集，推送 `kaggle_dataset.json`。
- [ ] 【Codex】生成 C1 Notebook，交付 `.cells.md`；【Kiro】审查补丁。
- [ ] 【Codex】在 Kaggle 新建版本，核对输入，完成保存运行，确认锚点预检通过。
- [ ] 【用户】确认当日额度，批准提交 C1。
- [ ] 【Codex】记录提交和分数，按第 12 节给出下一步；【Kiro】做第三轮 review。
