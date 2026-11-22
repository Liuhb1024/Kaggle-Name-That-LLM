# 调研发现

## 仓库现状
- 2026-11-15：仓库只有 .git；git status --short 无输出。尚无代码、测试、依赖或项目规范。
- 本轮采用用户提供的全局 AGENTS.md 约束。

## 官方比赛
访问日期：2026-11-15（Asia/Shanghai）。访问方式：更新前尝试 CUA，更新后使用 Playwright 真实浏览器页面交互。未登录、未接受规则。

### Overview
来源：https://www.kaggle.com/competitions/name-that-llm/overview
- 18 分类的模型作者归因；6 家机构；文本 40–250 词；隐藏测试约 4,000 条。
- 指标 balanced accuracy = 各类 recall 的宏平均；测试类频率不均，公榜/私榜模型比例不同。
- 2026-10-14 开始；2027-01-06 报名/组队截止；2027-01-13 最终提交截止；均 23:59 UTC。
- Notebook-only；CPU/GPU 各最多 9 小时；Internet disabled；允许免费且公开的外部数据/预训练模型。
- 输入 /kaggle/input/competitions/name-that-llm/test.csv；输出 working directory 的 submission.parquet，列 id、model；id 完整唯一、无空值、标签必须属于官方 18 类。
- Overview 称交互测试只有 1 行，而 Data 页称 3 行：官方文本不一致，后续应动态读入，不硬编码条数。
- 总奖金 $50,000；第 1–10 名每名 $5,000。

### Data
来源：https://www.kaggle.com/competitions/name-that-llm/data
- train.csv：2,699 段，150 个 prompt；列 id/prompt_id/text/model。
- test.csv：列 id/text；hidden test 的 prompt 不在训练集出现。
- sample_submission.parquet：id/model。
- 共 3 文件、2.14 MB；页面标示 CC BY-SA 4.0。
- 仅轻微格式清理；完整文件预览/下载要求登录并接受规则，未执行。
- 项目推论：需要按 prompt_id 分组验证，保留标点与风格信号，不把 prompt_id 当推理特征。

### Rules（已读专属条款，继续核对基础条款）
来源：https://www.kaggle.com/competitions/name-that-llm/rules
- 队伍最多 5 人；每天最多 5 次提交；最多选择 2 个最终提交。
- 单人不得使用多个账号；合队需要满足累计提交次数上限。
- 数据标示 CC BY-SA 4.0，同时专属数据安全条款禁止向未正式接受规则者提供比赛数据；不能仅凭许可证标题公开比赛文件。
- 外部数据需公开、平等可得，免费或满足主办方 Reasonableness（最小成本、合理可得）审查；Overview 更明确写免费公开，当前采用更保守边界，不默认允许付费 API。
- AutoML 允许，但需有足够授权以符合规则。
- 获奖条款标题写 CC-BY 4.0，正文要求 winning submission/source 使用 OSI 批准、不得限制商业用途的许可证；二者关系需获奖前向主办方核实，不自行宣称一致。
- 获奖需要交训练/推理代码、计算环境、资源需求、复现文档；可能需说明架构、预处理、loss、超参和录制交流。
- 点击 Rules 后页面仍在加载，首次 DOM main 读取空：改为等待页面快照后读取；未修改页面。
- 基础规则优先于冲突的专属规则；禁止人工标注/人工预测 validation/test 记录用于提交；禁止队外私下分享比赛代码/数据；公开代码应在本比赛 Kaggle 论坛/Notebook 向全体分享并满足 OSI 商业使用要求。
- 最终以私榜定名次；同分较早提交优先。参赛需成年/符合资格与当地要求；获奖通知需一周内响应、获奖文件通常两周内返回，税费由获奖者承担。
- 读取时发现页面无 main 标签，AX main 实为语义角色；改读 document.body.innerText 成功。局部快照会使旧 ref 失效，后续按新快照/观察过的控件定位。

### 官方置顶与 Code 列表
- 官方入门：https://www.kaggle.com/competitions/name-that-llm/discussion/749382 。署名 Kaggle Staff Ashley Oldacre。Discord 非官方答疑/规则更新的可靠入口；重要信息保留在 Kaggle 论坛；代码共享仍须 Kaggle 公开论坛/Notebook。
- Code 列表出现官方置顶 Starter Notebook（页面展示 score 0.322）；其他公开条目约 0.69–0.70，但均只是页面展示分数，未复现、未审计许可证，不作为已验证能力。
- 已完整阅读官方 Starter 的 3 个代码单元与输出：word TF-IDF (1,2), max 25k；char (3,5), max 35k；LogisticRegression(C=2,max_iter=1000)；5-fold GroupKFold(prompt_id)，balanced_accuracy。Notebook 输出 CV 0.3145±0.0153，公榜展示 0.322，运行记录 106.3 秒；这些是作者已发布运行记录，并非本机复现。
- 官方 Notebook V4 输出 Test shape (3,2)，佐证 Data 页 3 行；代码注释仍写 1 行。它读取实际行数，不会受文字差异影响。
- 官方代码许可证 Apache 2.0；来源：https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition 。未复制代码到仓库、未执行。
- Models 页已查看，仅为社区挂接模型，页面 Best public score 不代表单模型能力，更不代表许可证已审查。
- Leaderboard 页：https://www.kaggle.com/competitions/name-that-llm/leaderboard 。公榜约 40%，私榜另 60%；访问时榜首 0.774，50 名前沿约 0.719（动态快照，非最终排名）；不采集队伍身份。

### 官方提交文档
- https://www.kaggle.com/docs/competitions#notebooks-only-FAQ ：代码赛不能直接上传预测文件；挂接比赛数据，生成 working 输出，Save Version → Save & Run All，再到 Output Submit。团队共用每天额度。具体 RAM/CPU 配额需在编辑器查看，当前未登录，未核验账户硬件/额度。
- https://www.kaggle.com/code-competition-debugging ：隐藏数据替换后端到端重跑；报错提交也占每日额度；超时、内存/磁盘限制、异常、缺输出、格式错误等均应本地预检；示例的 submission.csv 是通用说明，本比赛必须 parquet。避免用分数/时间编码隐藏数据或探测答案。
- https://www.kaggle.com/WinningModelDocumentationGuidelines ：获奖复现应含 README、精确版本依赖/镜像、硬件 OS、目录、集中路径配置、训练/预测入口、序列化模型、运行时间、方法/特征/外部来源等；英文 Word/PDF summary 与 presentation 为一般指导，最终以本赛通知为准。此轮不创建获奖材料或读取/填写个人资料。

### 主办方规则澄清（修正前述保守边界）
来源：https://www.kaggle.com/competitions/name-that-llm/discussion/752084 。页面显示回复者 MartynaPlomecka 为 Kaggle Staff；访问时回复标注 18/17/12 天前。
- 现有 Kaggle 公开社区数据可用于训练，包括历史通过 proxy 生成的公开数据；公开可得的 Kaggle 模型权重/LoRA 可用。
- 合理成本内自生成训练数据不必公开，无此类数据的公开截止日；token 花费超过单项潜在奖金 $5,000 被明确举为不合理。并非 $5,000 以下自动获准或当前预算授权。
- Benchmarks proxy quota 用于 benchmark evaluation，不应新用于生成比赛训练数据；主办方不追罚此前已用者。针对仅作私有封闭验证的后续具体提问，未看到回复，不将其推定获准。
- AI 编程助手工作流获肯定答复；不等于获得上传凭据或任意原始比赛数据的授权。
- 自动探索数据关系（包括问题中提及的测试聚类/伪标注）获肯定答复；仍不能人工预测/标注测试记录、恢复隐藏标签或外传数据。
- 获奖许可答复认可 OSI 许可，但问题问 code MIT/Apache + write-up CC BY，回复措辞却写 write-up/docs 用 OSI；可按 OSI 方案准备，具体组合在交付前再次确认。
- 官方 I/O 性能答复：https://www.kaggle.com/competitions/name-that-llm/discussion/750924 。Dustin（Kaggle Staff）承认曾有 dataset cache/延迟问题，建议模型用 Kaggle Model，随机读文件可在磁盘容量允许时复制到 /tmp；后续回复称已稳定延迟。这是历史情况，不认定当前必然慢；后续端到端计时需覆盖载入/复制与推理。

### 社区实验（未经独立复现）
- https://www.kaggle.com/competitions/name-that-llm/discussion/757855 ：作者报告 random CV 0.104 vs prompt-group CV 0.315，并报告 prompt 内 Hungarian 强制分配的 CV 提升几乎不转化到公榜；标点不变量可能有价值。这些是作者自报的经验，只作假设来源，不作为数据事实或硬规则。
- 已读上述作者直接关联 GitHub 项目 README：https://github.com/pedrohbenetton/kaggle-name-that-llm 。页面显示 MIT、9 commits、latest ebd3111、0 stars/0 forks；属于小型近期参赛实现，不能称为热门成熟框架。作者自报 TF-IDF+风格、embedding、LoRA、分组失败实验；README 自述 53 tests，但本轮未执行/独立证实。未下载、安装或复用代码。局限包括单次实验、同 folds 调参、标点规则出域风险。
- 原始论文摘要已读：https://arxiv.org/abs/2610.19583 （Who's Behind the Text? Robust Model Authorship Attribution，2026-10-14 v1）。作者提出 source-conditioned likelihood 与 token evidence clipping，评估领域/模型版本偏移。只读摘要，未复现，也不把论文其他任务的 accuracy 当成本赛 balanced accuracy；仅形成似然特征与跨 prompt 稳健性研究候选。
- 已阅读社区高分 Notebook 的方法说明、开头代码和末尾归属/输入检查，未全量审计 37 万字符代码：https://www.kaggle.com/code/dhananjayk2609/name-that-llm-style-fingerprints-llm-ensemble/notebook 。Apache 2.0；V11 为 measurement run，GLM stage off；best score 0.693 来自 V8，不能把 V11 的 54 秒当全量隐藏推理成本。
- 该 Notebook 引入多家作者的 embedding LoRA、LLM/encoder/sparse ensemble、ATLAS、GLM likelihood、多份外部权重和 40 个 input；强依赖链和许可证仍待逐项审查。作者提到镜像未固定导致评分失败、推理时间预算导致 fallback；页面分数不等于可复现保证。
- 社区 Notebook 输出每 prompt 18 条×149、17 条×1（作者记录未复现），所以不能写死每 prompt 恰好 18；其 Open Pool 文案 212,623 和输出 226,191 数量也不一致，使用外部数据前必须固定版本重新审计。
- 已读 Open Pool 数据卡：https://www.kaggle.com/datasets/crtproduction/name-that-llm-open-pool 。访问时 v15、226,191 passages、65 sources、CC BY-SA 4.0；标题仍写 224k，文件说明仍写 223,840。作者明确标签来自发布者声明，未逐条重新验证实际生成模型；reasoning/route/provenance 部分未知。match_weight 用全部官方训练 prompt 拟合，直接用于训练集 CV 存在泄漏/乐观偏差；后续必须折内重算或禁用。只读数据卡，未下载文件。
