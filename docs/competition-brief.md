# Name That LLM 比赛与规则调研

核对日期：2026-11-15（Asia/Shanghai）。使用真实浏览器交互阅读公开页面；浏览器未登录。已覆盖官方七个导航页、完整规则正文、Admin 过滤下的三个讨论主题、官方基线和三份关联平台文档。社区帖子、模型与外部数据是研究线索，未逐项复现或审计，不能据此保证分数或合规性。

本文件记录前置比赛调研时点；2026-11-16 起的项目实现、依赖安装和实际复现进展见[复现记录](baseline-reproduction.md)。原时点文件与哈希原样保存在 research/2026-11-15/snapshot，后续实施不改写该历史证据。

## 比赛任务

输入一段模型生成文本，输出其来源模型的精确名称，属于 18 分类作者归因。任务不是区分人类与 AI。官方描述文本约 40–250 个英文词，涉及哲学、物理、数学、生物、计算机与创意写作；六家机构的模型风格和同一家族不同版本都需要区分。[Overview](https://www.kaggle.com/competitions/name-that-llm/overview)

## 数据合同

| 文件 | 官方说明 | 列 |
| --- | --- | --- |
| train.csv | 2,699 条，150 个 prompt | id、prompt_id、text、model |
| test.csv | 交互示例很小；评分重跑替换为约 4,000 条隐藏文本 | id、text |
| sample_submission.parquet | 提交格式样例 | id、model |

`id` 是匿名唯一标识；`prompt_id` 是训练文本所回答问题的匿名标识。**隐藏测试的 prompt 不出现在训练中，且 test 没有 prompt_id。** 官方仅做轻微格式清理，原始标点、大小写与空白可能是有用信号。三个文件合计页面显示 2.14 MB，数据许可标为 CC BY-SA 4.0。[Data](https://www.kaggle.com/competitions/name-that-llm/data)

不要固定样例行数：Overview 和 Starter 注释写 1 行，Data 与 Starter V4 实际发布输出写 3 行。后续程序应只依赖实际读入数据，完整覆盖 test id。官方 40–250 词是任务描述，不应未经检查就作为删行门槛；社区统计存在超出该范围的文本。

本轮没有下载文件、检查完整行内容或计算原始数据哈希。公开 Starter 的少量发布预览仅用于理解结构，未复制比赛样本进仓库。

## 标签必须精确匹配

| Provider | model 合法值 |
| --- | --- |
| OpenAI | `gpt-5.5`、`gpt-5.6-luna`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-6-astra` |
| Google | `gemini-3.1-pro-preview`、`gemini-3.6-flash`、`gemini-3.7-flash`、`gemini-3.8-flash` |
| Anthropic | `claude-fable-5`、`claude-opus-4-6`、`claude-opus-4-8`、`claude-opus-5`、`claude-sonnet-5` |
| Qwen | `qwen3.8-max` |
| ZAI | `glm-5.3`、`glm-5.3-flash` |
| Moonshot AI | `kimi-k3` |

标签来自本比赛 Evaluation，仅作为任务类别，不表示本项目使用了这些模型或已核验其 API 可用性。[Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview#evaluation)

## 评分与验证

Balanced accuracy 是 18 个类别 recall 的平均值：每类对最终分数权重相同。隐藏测试类别数量不等，公榜和私榜的模型比例也不同。因此普通 accuracy、整体 loss、公榜排名都不能代替主指标。Leaderboard 页面说明公榜约用 40% 测试数据，最终私榜用另外 60%。[Evaluation](https://www.kaggle.com/competitions/name-that-llm/overview#evaluation)、[Leaderboard](https://www.kaggle.com/competitions/name-that-llm/leaderboard)

工程推论：以 `prompt_id` 分组做验证，与 unseen-prompt 测试条件对齐；同时报告每类 recall、混淆矩阵与 fold 波动。随机切分造成的偏差方向依赖具体数据和算法，不能只凭它高或低就判断模型好坏。外部近重复文本和 prompt 也需跨来源分组去重。

访问时公榜最高展示 0.774；官方基线展示 0.322，部分社区 Notebook 展示约 0.69–0.70。这些是动态页面记录，均非本机复现或最终私榜结果，不作为本项目成绩承诺。

## 时间和奖金

| 事项 | 官方日期与时间 UTC | 北京时间 Asia/Shanghai |
| --- | --- | --- |
| 开始 | 2026-10-14（页面未单列开赛时刻） | 不推算具体开赛时刻 |
| 报名并接受规则截止 | 2027-01-06 23:59 | 2027-01-07 07:59 |
| 合队截止 | 2027-01-06 23:59 | 2027-01-07 07:59 |
| 最终提交截止 | 2027-01-13 23:59 | 2027-01-14 07:59 |

主办方可修改期限，正式提交前需复查。总奖金 $50,000，第 1–10 名各 $5,000，获奖仍需资格与合规审查。[Timeline 与 Prizes](https://www.kaggle.com/competitions/name-that-llm/overview#timeline)

## Notebook 运行与输出

本比赛只接受 Kaggle Notebook 提交；CPU 或 GPU Notebook 运行时间均不超过 9 小时，提交环境关闭 Internet。允许免费、公开的外部数据及预训练模型；自生成数据另见下文官方澄清。具体账户 GPU 配额、RAM、磁盘、可选硬件、镜像版本需登录编辑器后核验，不能把社区使用 T4×2 的经验当作本账户资源保证。[Code Requirements](https://www.kaggle.com/competitions/name-that-llm/overview#code-requirements)

输入为 `/kaggle/input/competitions/name-that-llm/test.csv`。Notebook 在 working directory 生成真正的 Parquet 文件 `submission.parquet`，含 `id` 与 `model` 列。不得仅把 CSV 改扩展名。每个 test id 必须出现一次，无缺失、重复或空值，标签属于上述 18 类。提交文件不包含行索引。

标准过程是挂接比赛数据和必要公开资源，生成输出，`Save Version → Save & Run All`，在 Notebook Viewer 的 Output 提交。评分时会端到端重跑 Notebook，替换隐藏测试集；交互运行成功不等于隐藏重跑成功。[提交说明](https://www.kaggle.com/docs/competitions#notebooks-only-FAQ)

本赛未要求 Evaluation API 推理服务器；通用 debugging 文档列出的 server 错误不构成本赛新增要求。错误提交也占每日额度。需预检超时、内存/磁盘、路径、离线依赖、文件格式和测试规模变化。[官方调试文档](https://www.kaggle.com/code-competition-debugging)

## 参赛规则和官方澄清

以下为操作摘要，不能替代接受规则前阅读全文。基础规则明确优先于冲突的专属条款。[完整 Rules](https://www.kaggle.com/competitions/name-that-llm/rules)

| 范围 | 对参赛的影响 |
| --- | --- |
| 账号和团队 | 每人一个账号，只能参加一个队伍；最多 5 人；合队需在截止前且累计提交次数满足额度要求 |
| 提交次数 | 每队每天最多 5 次，可选最多 2 个最终版本；最终按私榜，规则写同分较早提交优先 |
| 人工预测 | 禁止把对 validation/test 记录的人工标注、人工预测用于提交；这不同于正常使用训练标签构造离线验证折 |
| 分享 | 禁止队外私下分享比赛代码/数据；公开比赛代码需放在本比赛 Kaggle 论坛或 Notebook，供全体参与者使用，并符合 OSI 商业使用许可要求 |
| 数据安全 | 规则禁止向未正式接受规则者提供比赛数据；不能只因页面写 CC BY-SA 就把原始文件放入 GitHub |
| 外部资源 | 审查可得性、成本、许可、来源和复现要求；能公开下载不等于标签真实、权属已审计或值得采用 |
| 资格 | 注册账号，满足成年/监护人同意、居住地、制裁及当地规则要求；代表单位参赛须符合单位政策；不推断本人的参赛资格 |
| 获奖 | 提供训练/推理代码、模型与环境、方法和资源说明；响应通知、签署交付与税务文件；税费由获奖者承担 |

主办方 MartynaPlomecka（页面标注 Kaggle Staff）在[集中规则答疑](https://www.kaggle.com/competitions/name-that-llm/discussion/752084)中进一步说明：

- 现有 Kaggle 公开社区数据可以训练，包括通过 Benchmarks proxy 生成的公开数据；向全体开放的 Kaggle 模型权重、LoRA 可用。
- 合理成本内的自生成训练数据不要求公开，也无相应的公开截止日。主办方把 token 支出超过单个可能奖金 $5,000 举为不合理；这不是批准花费 $5,000，也不是低于该值就自动合规。本轮没有生成数据或支出预算。
- Benchmarks proxy 配额用于 benchmark evaluation，不应继续用于生成比赛训练数据。主办方表示不追罚此前使用；针对后续“封闭验证生成”具体问题，当前未见明确回答，不推定获准。
- 对 AI 编程助手工作流，主办方给出肯定回复；这不构成外发密钥、私密文件或任意比赛数据的授权。
- 对全自动利用测试文本之间关系的问题，主办方允许探索；依然不能人工标注测试、恢复隐藏答案或外传隐藏数据。

许可证表头写获奖许可 CC-BY 4.0，正文要求 OSI 许可；管理员答复认可 OSI 方案，但关于代码和 write-up 组合的措辞仍不完全清晰。可按 OSI 开源交付准备，最终交付前向主办方确认具体许可证组合。**本轮没有替用户接受任何条款。**

官方[入门与 Discord 帖](https://www.kaggle.com/competitions/name-that-llm/discussion/749382)强调，重要规则问题应留在 Kaggle 论坛；Discord 频道不由 staff 持续答疑，代码共享仍须公开在 Kaggle。

## 代码与数据研究入口

| 来源 | 已看到的内容 | 使用边界 |
| --- | --- | --- |
| [官方 Starter](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition) | Apache 2.0；词和字符 TF-IDF + 逻辑回归；5 折 prompt GroupKFold；发布 CV 0.3145±0.0153，页面公榜 0.322 | 三个代码单元已读，尚未独立运行；适合作为首个复现目标 |
| [社区风格与集成 Notebook](https://www.kaggle.com/code/dhananjayk2609/name-that-llm-style-fingerprints-llm-ensemble/notebook) | Apache 2.0；多种 LoRA/embedding/encoder/sparse/likelihood；约 40 个 input；best 0.693 属于 V8，所见 V11 是 measurement run | 已读方法、部分代码与归属，未审计全部代码/输入；不能把 V11 54 秒当隐藏推理耗时 |
| [Open Pool 数据卡](https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool) | 访问时 v15，自述 226,191 条、65 来源、CC BY-SA 4.0；带来源/许可/路由/推理设置字段 | 标签是发布者声明；未下载；应固定版本核验。`match_weight` 用全量官方训练 prompt 拟合，不可直接用于无偏 CV |
| [社区实验原帖](https://www.kaggle.com/competitions/name-that-llm/discussion/757855)与[关联 GitHub](https://github.com/pedrohbenetton/kaggle-name-that-llm) | README 标 MIT；9 commits；0 stars/0 forks；记录 CV 与公榜不一致、强制 prompt 分配失败等经验 | 小型参赛项目，不称为热门成熟框架；测试数和指标是作者报告，未执行或复制代码 |
| [ATLAS 原始论文摘要](https://arxiv.org/abs/2610.19583) | 2026-10-14 v1；source-conditioned likelihood、token clipping、跨领域/版本评估 | 本轮只读摘要，作研究候选；论文任务的 accuracy 不能替代本赛指标 |

Open Pool 标题仍写 224k，文件说明部分写 223,840；社区文章也有旧版数量。只能以固定版本的实际文件审计确定条数。数据卡还明确有未知生成参数、标签真实性和路由风险，不能把“clean”当本项目已核验结论。

[官方 I/O 答复](https://www.kaggle.com/competitions/name-that-llm/discussion/750924)承认曾有 dataset cache 延迟，建议模型采用 Kaggle Model，必要时在磁盘预算允许下把随机读取文件复制到 `/tmp`；后续回复称已稳定。当前性能仍需实测，载入与复制都应算入 9 小时预算。

## 尚待后续处理

- 参赛者登录、核对资格并明确接受规则，再获得数据；本轮未报名、未下载、未提交。
- 下载后记录原始文件哈希、schema、行数与标签、prompt 覆盖，保护原件。
- 核对本账户编辑器硬件、周配额、RAM/磁盘与镜像，完成真正离线运行。
- 逐项审核拟采用的外部数据与权重版本、真实许可、来源与泄漏，不能直接导入整套社区依赖链。
- 进入具体模型选型前阅读全文/官方仓库与许可证，补足维护、成本、复现性比较。
- 正式提交前复核官方更新；获奖前确认许可组合及交付通知。

这次调研没有支付费用、安装依赖、训练模型、调用付费模型 API、读取凭据、修改远端资源或推送代码。已授权联网仅用于浏览公开资料。浏览器自动产生的原始 traces 可能含临时 URL，保留在本地并通过 `.gitignore` 排除，不作为可公开资料。
