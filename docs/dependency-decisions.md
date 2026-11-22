# 基线依赖和方案选择

核验日期：2026-11-15 至 2026-11-16。目标是固定复现环境，不宣称下列历史版本是最新或已完成漏洞数据库审计。新增依赖用于用户已授权的官方基线、Parquet 输出、测试与 Notebook 执行；不引入外部模型、AutoML、服务或付费 API。

## 选择依据

| 问题 | 可行方案 | 本次选择及原因 | 代价和限制 |
| --- | --- | --- | --- |
| 分类器与 CV | 官方 sklearn 1.6 API 版本族；当前 stable 1.9.1 | 固定 sklearn 1.6.1，保持首次复现参数与默认行为可追溯 | 官方原 Notebook 精确环境未公开核验；1.6.1 是本地选择，不是证明官方用了该 patch |
| Parquet | pyarrow；fastparquet | 显式 pyarrow，pandas 文档优先使用且当前文档已弃用 fastparquet 路径 | 增加本地二进制 wheel；Mac 与 Kaggle Linux 的 wheel 不可混用 |
| 模型保存 | joblib/pickle；skops/ONNX | joblib 已是 sklearn 依赖，适合自己训练的可信 Pipeline | 不加载外来 pickle，检查版本和摘要；哈希不能消除任意 pickle 的执行风险 |
| 交付形态 | 可维护 Python 包 + 生成 Notebook；手写两套独立逻辑 | Notebook 嵌入本包源码，独立运行且保留源码摘要 | 生成后必须 --check；Kaggle 镜像版本仍需实际核验 |
| 环境 | 系统环境；项目 .venv + uv.lock | 已有 Python 3.11.15 与 uv，安装到 .venv，锁传递依赖与分发文件哈希 | 首次安装需联网；离线运行需先有完整环境；只针对 macOS/Linux 验证 |

算法路线 A–D 的比较见[工程设计](engineering-plan.md)。本轮只实现路线 A。社区 GitHub 实现与 ATLAS 原始论文属于前置研究输入，不复制其训练代码或增加生产依赖。

## 固定直接依赖

| 用途 | 包与版本 | 官方 PyPI 声明的核心许可 |
| --- | --- | --- |
| 数组与稀疏矩阵 | numpy 2.2.6、scipy 1.15.3 | BSD |
| 表格 | pandas 2.2.3 | BSD |
| 特征、分类、分组、指标 | scikit-learn 1.6.1 | BSD |
| Parquet | pyarrow 19.0.1 | Apache |
| 本地模型保存 | joblib 1.4.2 | BSD |
| 限制 BLAS/OpenMP 资源 | threadpoolctl 3.6.0 | BSD |
| 构建 | hatchling 1.27.0 | MIT |
| 开发检查 | pytest 8.3.5、ruff 0.11.13、mypy 1.15.0 | MIT |
| Notebook 测试 | nbclient 0.10.2、nbformat 5.10.4、ipykernel 6.29.5 | BSD |

核验了上述固定 release 的 Python 要求、wheel 存在和许可分类声明，来源为官方 PyPI JSON：`https://pypi.org/pypi/<name>/<version>/json`。核心许可分类不替代 wheel 中随附的二进制组件、归属与详细许可文件。上游许可文件随安装分发保留，不把声明写成全面法律审计。

uv.lock 固定生产和开发解析；requirements.txt 从锁文件导出生产依赖（含传递依赖和允许的分发哈希）。使用库时不需要把本地数据发给任何外部提供商；安装联网只传递包解析/下载请求。新版本升级、外部模型和数据将分别比较维护、许可、成本和实测收益后再决定。

## 一手资料

- [官方 Starter Notebook](https://www.kaggle.com/code/martynaplomecka/starter-notebook-for-name-that-llm-competition)：基线参数、验证方式和 Apache 2.0 归属。
- [GroupKFold 1.6.1](https://scikit-learn.org/1.6/modules/generated/sklearn.model_selection.GroupKFold.html)：组隔离、默认不 shuffle。
- [LogisticRegression 当前文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)：1.8 起 penalty/n_jobs 弃用、数值环境差异说明。
- [pandas Parquet 文档](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html)：引擎与 index=False。
- [sklearn 模型保存](https://scikit-learn.org/1.6/model_persistence.html)：格式比较、可信来源与同版本要求。

没有新增 Git hook、后台服务、Kaggle 凭据、训练 API 或云端写入。
