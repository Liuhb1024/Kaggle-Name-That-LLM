# 数据目录

真实比赛数据属于本地私有输入，`data/raw/` 已由 Git 忽略。测试夹具由代码生成的合成文本构成，不包含比赛段落。

2026-11-16 已由代理通过登录浏览器的官方 Download All → ZIP 获取并导入。原 ZIP 在 `data/raw/downloads/name-that-llm.zip`，三份原始文件在 `data/raw/name-that-llm/`；两者均忽略且父目录权限为 0700。下面的下载/导入命令用于新的干净工作区，现有非空数据目录不应重复导入。

在本人 Kaggle 账号登录并接受比赛规则后，从[官方 Data 页](https://www.kaggle.com/competitions/name-that-llm/data)点击 Download All。保留下载原件，不从第三方镜像补齐受限数据。导入命令：

```bash
uv run --offline --locked ntl-baseline import-data \
  --archive "$HOME/Downloads/name-that-llm.zip" \
  --data-dir data/raw/name-that-llm
```

导入器仅接受 ZIP 根部的 train.csv、test.csv、sample_submission.parquet 三个文件，检查路径、链接、体积、schema、18 类标签和样例 id，验证成功后移入数据目录。拒绝覆盖非空目标目录。压缩包原件不变；`source.json` 记录来源、导入时间、压缩包与文件 SHA-256。

```text
data/
├── README.md
└── raw/                         # Git ignored
    └── name-that-llm/
        ├── train.csv
        ├── test.csv
        ├── sample_submission.parquet
        └── source.json          # 本地导入记录
```

以上 raw 子树已在当前工作区生成。审计命令和[当前复现状态](../docs/baseline-reproduction.md)提供实际证据，不能仅凭目录示意推断其他工作区也有数据。

```bash
uv run --offline --locked ntl-baseline audit --data-dir data/raw/name-that-llm
```

train 需要字符串列 id/prompt_id/text/model，test 需要 id/text。程序按字符串读取，保留前导零 id；缺失文本按官方基线填为空串。未知标签、空 id、重复 id、缺列、多余列会明确失败。sample_submission 只用于审计，不作为推理时行数模板；隐藏测试行数动态处理。

不把原始 CSV、OOF、模型、数据包、凭据、已执行 Notebook 或浏览器临时 URL 提交到 Git。公开分享代码需遵守本比赛 Kaggle 分享要求，数据再分发不能只看页面许可证标题。
