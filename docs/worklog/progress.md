# 调研日志

## 基线实现阶段 2026-11-15 起

- 用户授权数据获取、项目搭建、官方基线复现和文档整理；不包含正式评分提交。
- 现有代码为空，Git main 尚无 commit；本机 Python 3.13/3.12/3.11 可用，uv 可用；全局未安装本项目库。
- 本机 Kaggle 凭据环境变量/默认凭据文件均未配置（只检查存在性，未读取秘密）。浏览器数据页明确要求登录和接受规则；用户表示会自行完成并下载到 Downloads，当前等待实际文件。
- 已原样复制上一轮文件与两份 manifest 到 docs/research/2026-11-15/snapshot；在该目录校验 v2 摘要 7/7 OK。根目录旧 manifest 已迁入归档，未改写其内容。
- 已扩展 .gitignore 排除原始数据、运行产物、环境、缓存和凭据；未下载或外发数据。
- 已读 scikit-learn 当前 stable 官方 LogisticRegression 文档（1.9.1）；penalty/n_jobs 在 1.8 发生弃用，拟优先评估官方基线文档版本族 1.6，固定复现环境而非盲目升级。
- 已比较 sklearn 固定基线版本与当前 stable，以及 Parquet 两类引擎；官方 PyPI 元数据的兼容/wheel/许可声明已核验，决定 Python3.11 + sklearn1.6.1/pandas2.2.3/pyarrow19.0.1，在 .venv 内 uv sync。
- 首批需求测试已写，数据/基线函数仅为 NotImplementedError API 桩，准备运行 red phase；尚未实现算法行为。
- 公共 localhost CDP Proxy 连接被拒绝，未读取任何浏览器 Cookie；继续可用浏览器与用户下载流程。
- uv sync 已完成（53 个安装包，隔离于 .venv，下载约 1m42s）；uv.lock 固定完整解析。安装输出有历史元数据版本规范修复警告，无安装失败。
- TDD 首次命令 uv run --offline --locked pytest -q --tb=short：32 failed，退出 1，原始日志保留 artifacts/checks/tdd-red-2026-11-16/pytest.log。主要失败是尚未实现的 API 桩；其中一个测试夹具用 .loc 给单格赋 dict 触发了 pandas 自身对齐错误，改 .at 保持原需求，单独确认 red 后再实现；未放宽断言。
- .at 修正后的单测确实在 API 桩处 NotImplementedError（1 failed）。首次实现后 31 passed/1 failed：组合测试把未知标签的输入污染带到“缺少类别”用例，改为复制未知标签输入，两项原断言保持；随后 32 passed。mypy 首次检查 9 个源文件无问题；ruff 指出格式/import/行长等，已按工具修正，待最终复查。
- 新增 CSV 纯数字/前导零 id 与数字文本回归测试，先得到 2 failed，原因是 pandas 推断为数值。最小修复为按官方 schema 显式指定字符串 dtype，空文本仍沿用官方 fillna('')；这属于输入合同加固，后续与直接官方参照比较实际数据预测，确认是否影响复现。
- CSV 加固后 safety 定向测试 8 passed。新增 Notebook 输出清洁、源码哈希与无网络独立执行测试，先得 2 failed（NotImplementedError），随后实现可移植源码包嵌入与 Notebook 生成器；与 CLI 复用相同模块，不复制另一套训练逻辑。
- Notebook 定向 2 passed；随后完整 42 passed（6.69s）。mypy 10 个源文件通过；Notebook --check 与 ruff check 通过。新增直接官方对照脚本、离线 Notebook 执行器和项目级 AGENTS.md。
- 仓库安全检查增加合成回归测试：初次 2 failed/2 passed，发现 artifacts 路径层级判定漏报和正则定义匹配自身；修正实现后 4 passed，未放宽测试或检查范围。
- 更新 README 的首次补丁引用了不匹配的旧行，整批补丁未执行；重新读取实际文件后准确应用，无历史文件被覆盖。
- 用户澄清不自行下载，要求全部由代理负责。立即重新读取浏览器实时状态：当前会话已登录，比赛页已显示 Submit Prediction 与 Team/Submissions，规则访问门槛已解除。未替用户点击接受任何条款；直接使用官方 Download All → Download as zip 获取文件。
- ZIP 点击工具等待导航超时，随后浏览器显示 GCS 响应超时，同时存在 Downloading 事件。未盲目重下：检查发现 .playwright-mcp/name-that-llm.zip 已保存 874,800 字节，继续 CRC/条目/数据合同校验以判断实际结果。原始包保留；未读取或导出浏览器 Cookie。系统已有本地代理，此轮没有修改代理或安全设置。
- 官方下载 ZIP CRC 和三文件条目检查通过；通过 ntl-baseline import-data 安全导入 data/raw/name-that-llm，source.json 保存原件摘要。archive SHA-256 为 7c375db7dfeea6b36cefc7b48a290dd48c54c7fe03bfd402390728a6a2bb9386。
- 实际审计：2,699 train，150 prompts，18 labels，3 test；claude-fable-5 有149条、其余150；149个prompt18条、1个17条；词数36–274；无空文本、训练重复全文或 train/test 全文重叠。仅汇总，未输出/公开原始文本。
- 已启动真实命令 uv run --offline --locked python scripts/run_offline.py run --data-dir data/raw/name-that-llm --config configs/baseline.json --run-dir artifacts/official-baseline-v1。
- 官方 ZIP 另按相同摘要保存至 data/raw/downloads/name-that-llm.zip；原下载包不变。data/raw、artifacts、.playwright-mcp 父目录权限收紧至0700，不改变系统代理/浏览器安全设置。
- 真实 baseline-v1 完成：5-fold mean 0.2996551724、std 0.0167636672；pooled OOF 0.2996718867；验证52.77s、全量训练15.51s、预测0.044s、总69.11s；进程峰值RSS约620.0MB。原始文件摘要未变，生成3行Parquet并读回。与官方发布0.3145存在差异，未改数据、测试或门槛去迎合；启动独立官方代码参照确认方法是否一致。
- 真实离线 Notebook 独立kernel运行中（Python网络连接/DNS禁用），单独目录 artifacts/official-notebook-v1。
- 独立真实 predict 与 check-submission 成功，独立进程生成的Parquet与主运行字节相同；三份原始输入哈希仍与source.json完全相同。
- 直接官方参照完成：五折依次0.2833333333/0.3277777778/0.3092592593/0.2925925926/0.2853128991；与主实现最大分差0，3条test预测相同，原件未变。直接参照耗时140.16s；官方历史0.3145仍不能声称一致，原环境/折分证据不足，不猜测具体原因。
- 真实离线Notebook完成：CV与主CLI相同，总110.15s，峰值RSS600,195,072bytes。实际比较OOF、folds和最终submission字节完全一致；脚本输出/已执行Notebook仅保留于忽略目录。
- 最终提交文件SHA-256：f6677b37d0d4bea1b751c6c8ac508203952583279fba368d4985b761c8cea36a。原件/来源manifest/ZIP设为0444；不改变或覆盖其历史摘要。
- 本轮创建的5个浏览器研究标签已关闭，回读仅剩原始about:blank；未提取Cookie/令牌、未代点接受规则、未创建远端Notebook、未正式提交或推送。用户最终明确下载也由代理完成，本轮实际采用该方式。
- 最终产物核验通过：主run completed.json中全部产物摘要与文件相符；真实运行的package源码摘要与当前10个源文件相符；独立参照差异0；CLI/Notebook OOF、folds、submission字节相同；原ZIP/三个raw摘要不变；共享Notebook源码匹配且输出为空。结果保存 artifacts/checks/final-verification-2026-11-16.json；测试46 passed来自本轮最后一次完整测试，后续没有源码改写，不重复训练或改写结果。
- 最终仓库检查命令 uv run --offline --locked python scripts/check_repository.py：53个Git可见文件、18份Markdown，链接/选定凭据模式/7个冻结摘要均通过；Notebook --check通过；git diff --check退出0。ruff check/format、mypy10文件、uv pip check（53包兼容）先前已通过，之后仅更新文档和私有结果。
- 工作区原先无commit/实现，本轮新增Python包、测试、脚本、锁文件和生成Notebook，整理根README/文档/文件树，归档前置资料。所有修改仍本地未提交；raw/model/OOF/执行输出均被Git忽略，没有未经授权的远端写入、数据外发、真实API推理或付费操作。

## 2026-11-15
- 读取 web-access 与 planning-with-files 技能。
- 检查仓库、当前变更与项目级指令；工作区初始为空。
- web-access 附带 CDP 脚本缺失，采用内置 computer use。
- CUA 首次 getBrowser 连接超时，继续定位可用浏览器。
- 应用更新后 CUA 工具被移除；使用仍可用的 Playwright 浏览器交互工具恢复访问，成功打开官方 Overview。
- planning-with-files session-catchup 正常退出，提示不支持 Codex 原生历史解析。
- 已读 Overview、Data、Rules（专属与基础规则），记录了数据行数与获奖许可证表述差异。
- Discussion 列表已打开，识别官方置顶入门帖，继续阅读。
- 已读官方入门/Discord 说明与 Code 列表；打开置顶 Starter Notebook（新标签）。未运行代码。
- 已完整阅读官方 Starter、提交 FAQ、调试说明、获奖文档指南；已浏览 Models、Leaderboard。
- Discussion 通过 Admin 过滤发现含官方回复的规则集中答疑与输入读取性能帖，继续核对关键限制。
- 已阅读全部 Admin 过滤结果中的讨论正文与可见回复；区分 staff 澄清和普通参赛者自报。已读一个社区高分 Notebook 的方法/部分代码/归属、一个关联 GitHub README、一个原始论文摘要与 Open Pool v15 数据卡。
- 新增 README.md、docs/competition-brief.md、docs/engineering-plan.md；新增 .gitignore，仅排除自动生成的 .playwright-mcp/ 原始浏览器 traces。
- 文档审查：运行 python3 一次性脚本，检查 6 份 Markdown 的本地链接、18 标签与常见凭据/临时签名 URL 模式，全部检查满足；19 个不同外部链接来自已访问源页面（含锚点），未做额外 HTTP 链接爬取。此扫描仅覆盖选定模式，不等于全面安全审计。
- 运行 git diff --check：退出 0。初始空仓库，新文件尚未 tracked，另由 Python 脚本检查文档内容；无原文件被替换。
- 运行 git check-ignore .playwright-mcp/page-2026-11-15T15-51-11-293Z.yml：确认浏览器原始 traces 被忽略。扫描到的 35 份 traces 中，选定临时签名 URL 模式命中 0；不发布 traces。
- 最终回读 Overview 的 DOM 锚点：timeline、evaluation、code-requirements 均存在，文档章节链接正确。
- 无源代码行为变更，未运行模型/训练测试，也没有可声称的本项目分数。未安装依赖、未调用付费模型 API、未接受规则、未下载比赛文件、未提交或推送；公开浏览产生的网络请求属于本次授权阅读范围。
- 关闭本轮 Playwright 创建的 11 个研究标签，回读 tab list 仅剩初始 about:blank。没有操作用户原有 Chrome 标签。
- 最终命令：git diff --check；shasum -a 256 .gitignore README.md docs/competition-brief.md docs/engineering-plan.md findings.md progress.md task_plan.md > research-manifest-2026-11-15.sha256；shasum -a 256 -c research-manifest-2026-11-15.sha256。manifest 是本轮新增文件的首次摘要，不替换既有历史哈希；不包括原始浏览器 traces。
- 上述首次摘要检查 7/7 OK。随后 planning-with-files/check-complete.sh 报 0/0：脚本只识别 `### Phase` 与 `**Status:**`，不识别原中文列表。已将计划状态改为其支持的格式，未改变实际完成条件。保留首次 manifest 作为先前快照；当前文件摘要另存 research-manifest-2026-11-15-v2.sha256，首次 manifest 不作为当前工作区校验入口。最终重新运行 check-complete.sh、git diff --check 和 v2 shasum 校验。
