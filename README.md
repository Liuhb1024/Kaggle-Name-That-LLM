# Name That LLM 模型来源识别

这是一个用Kaggle比赛学习机器学习工程的项目：给定一段AI生成的文字，预测它由18个候选模型中的哪一个生成。当前是Python实验与命令行工具，没有产品网页。

已有的官方基线工程用于学习和复现；后续按小任务由你亲自阅读、操作和解释，我负责提示和审查，不提前代做后续实验。

## 从这里开始

1. **[项目导读与代码地图](docs/learning-guide.md)**：先理解任务、数据、模型和验证，再按模块阅读。
2. [比赛介绍](https://www.kaggle.com/competitions/name-that-llm/overview)、[数据说明](https://www.kaggle.com/competitions/name-that-llm/data)、[官方基线](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition)：原始题目与方法来源。
3. [运行手册](docs/running.md)：需要亲手操作时再查命令，不用一次执行全部步骤。
4. [实测复现报告](docs/baseline-reproduction.md)：查看结果、资源和未解决的问题。
5. [服务器与 Kaggle 工作流](docs/server-workflow.md)：在新机器上跑首版，并了解下一阶段 GPU 实验的目录边界。

## 项目在做什么

训练时，CSV提供文字和来源模型标签。程序用词与字符片段的TF-IDF将文字变成数字，再用逻辑回归学习区分类别。推理时读入没有标签的test.csv，生成id/model两列的submission.parquet。

我们没有训练18个大模型，也没有调用它们逐条判题。18是候选来源类别数。为了模拟新问题，验证按prompt_id分组，所有特征只在训练折拟合。

## 文件怎样分类

```text
Kaggle-Name-That-LLM/
├── README.md                       # 项目入口
├── AGENTS.md                       # 工程规则与学习协作约定
├── NOTICE.md                       # 官方方法归属
├── pyproject.toml                  # 包与检查配置
├── uv.lock / requirements.txt      # 锁定依赖
├── .python-version / .gitignore    # Python版本与忽略规则
├── configs/baseline.json           # 当前基线参数
├── src/ntl_baseline/               # 主代码
│   ├── constants.py / config.py    # 标签与参数
│   ├── data.py                     # 数据读取、检查、导入与提交合同
│   ├── baseline.py                 # 核心：特征、模型、分组验证
│   ├── workflow.py                 # 串联训练、保存与预测
│   ├── cli.py / __main__.py         # 终端命令入口
│   ├── artifacts.py                # 结果保存与哈希
│   ├── notebook.py                 # Notebook生成逻辑
│   └── __init__.py                 # 包标识
├── tests/                          # 合成数据测试，不含比赛原文
├── scripts/                        # 生成、离线执行、官方对照与检查
├── notebooks/official_baseline.ipynb # 生成的Kaggle交付文件，最后再读
├── data/
│   ├── README.md                   # 获取方式与数据合同
│   └── raw/                        # 本地只读原件，Git忽略
├── artifacts/                      # 本地模型、OOF、运行结果，Git忽略
├── docs/
│   ├── learning-guide.md           # 学习入口与模块地图
│   ├── running.md                  # 命令手册
│   ├── server-workflow.md           # 服务器首跑和模型交付边界
│   ├── baseline-reproduction.md    # 真实复现证据
│   ├── competition-brief.md        # 比赛规则调研
│   ├── engineering-plan.md         # 设计和未来边界
│   ├── dependency-decisions.md     # 依赖选择依据
│   ├── worklog/                    # 项目过程记录，不是初学入口
│   └── research/                   # 必须保留的冻结调研证据
└── directory_structure.txt         # Git可见文件的完整索引
```

根目录不再散放代理的计划、发现和进度文件；它们原样归入docs/worklog。浏览器自动traces已移出仓库，未删除证据。环境和检查缓存仅本地使用，Git忽略；它们不是作品集内容。冻结调研属于本项目来源和证据，保留但不混入主代码。

## 已实现到哪里

官方数据2,699条、150个prompt；本机训练、5折验证、独立预测和Parquet生成已跑通。CV balanced accuracy为 **0.29966 ± 0.01676**；与同环境独立官方代码一致，CLI与离线Notebook产物一致。本地值与官方历史0.3145有差异；相同源码在Kaggle镜像CV为0.31450±0.01533。2026-11-16至17日由代理按用户授权完成私有CPU离线Notebook提交：首次隐藏重跑异常，修复输入审计后成功，公榜 **0.322**。这不是最终私榜成绩，详见[提交摘要](docs/worklog/baseline-submission-summary.md)。

当前框架是AI辅助搭建的官方基线复现，不应写成你已经独立完成的原创算法。你接下来的亲手复现、分析、实验和解释，将逐项成为可展示的个人贡献；[学习路线](docs/learning-guide.md)列出任务和验收方式，尚未替你完成这些任务。

## 日常使用

```bash
uv sync --locked
uv run --offline --locked ntl-baseline audit --data-dir data/raw/name-that-llm
```

先解释audit输出，再决定是否运行训练。完整训练、独立预测、Notebook生成与质量检查命令见[运行手册](docs/running.md)。已有v1产物不得覆盖，新的实验需使用新目录。

本次提交保留官方算法与参数；修复Notebook对公开样例id的依赖，未覆盖既有本地结果。敏感数据和运行产物不进入Git；公开分享遵守[比赛规则](https://www.kaggle.com/competitions/name-that-llm/rules)。
