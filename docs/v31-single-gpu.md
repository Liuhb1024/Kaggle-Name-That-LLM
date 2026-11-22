# V31 的 Qwen3-8B 单卡推理

首轮交付选择 `lora24_claude` 的 plain 18 类成员。主入口是 `python -m ntl_repro.run`。它从本地加载模型和资产，输出每行预测及概率，并在校验输出后记录耗时和 CUDA 显存。复现目标来自 Vladislav Lavrentev，LoRA 权重与分类头来自 Zbigniew Kirchner。[V31 固定 Notebook](https://www.kaggle.com/code/lightningv08/0-717-user-prefix-glm-centering-qwen3-8-27b?scriptVersionId=372363760) · [适配器 v1](https://www.kaggle.com/datasets/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/versions/1)

## 来源与选择

2026-11-23 从 [Kaggle 固定版本下载端点](https://www.kaggle.com/kernels/scriptcontent/372363760/download) 取得 Notebook JSON，SHA-256 为 `1556c5110b52c917144cd961610f2fc6fd803d4c3982b4e6d2e977273eb9038d`。核查了其中的 `pinned_assets.py` 和 `comp_stack.py`，尤其是 `LORA_RUNS`、`find_checkpoint`、`score_lora_checkpoints` 和 `add_member`。Notebook 页面标题中的 0.717 是作者报告的集成成绩。固定下载件没有可用于确认这条分支实际加载路径的执行输出，Notebook 源码的再发布许可也未核实。本仓库依据推理合同独立实现。

| 对象 | 核查结果 |
| --- | --- |
| LoRA 权重 | 选择 `lora24_claude`。适配器专门发布，元数据标注 Apache 2.0，`final/` 是发布方指定的推理资产。v1 文件清单列出 safetensors 权重 116,434,760 字节。适配器配置记录 PEFT 0.20.0、rank 16、alpha 32，以及 attention 和 MLP 的 7 个投影。 [发布元数据](https://www.kaggle.com/api/v1/datasets/view/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) · [文件清单](https://www.kaggle.com/api/v1/datasets/list/zbigniewkirchner/ntl-qwen3-8b-lora-claude-head) |
| 底座 | 首跑固定为适配器作者声明的 `qwen-lm/qwen-3/Transformers/8b-base/1`。已从 Kaggle 校验配置和五分片索引。Qwen 官方为 Qwen3-8B-Base 标注 Apache 2.0。 [Kaggle 底座 v1](https://www.kaggle.com/models/qwen-lm/qwen-3/transformers/8b-base/1) · [Kaggle 配置](https://www.kaggle.com/api/v1/models/qwen-lm/qwen-3/transformers/8b-base/1/download/config.json) · [Kaggle 分片索引](https://www.kaggle.com/api/v1/models/qwen-lm/qwen-3/transformers/8b-base/1/download/model.safetensors.index.json) · [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-8B-Base) |
| V31 底座歧义 | `comp_stack.py` 运行时取按路径排序后第一个 `qwen3`、宽度 4096 的配置。固定 Notebook 的其他路径表指向 Qwen3-Embedding-8B，且附件清单含多个符合该筛选条件的底座。Embedding 8B 也采用 Apache 2.0，但资产字节和配置不同。优先采用发布方指定的底座和锚点来验证成员，固定版实际命中的路径仍需执行记录确认。 [Embedding Kaggle 配置](https://www.kaggle.com/api/v1/models/qwen-lm/qwen-3-embedding/transformers/8b/1/download/config.json) · [Qwen 官方 Embedding 模型卡](https://huggingface.co/Qwen/Qwen3-Embedding-8B) |
| 其他适配器 | V31 同一逻辑下还有 `lora24_public` 和 `lora24_seed7`。前者从作者的多资产镜像定位，镜像许可按组件声明。后者引用训练 Notebook 输出。此次选用有独立许可、明确的发布目录、底座声明和三行锚点的 Claude 适配器，以减少镜像依赖和来源歧义。 [V31 源码](https://www.kaggle.com/kernels/scriptcontent/372363760/download) · [多资产镜像](https://www.kaggle.com/datasets/lightningv08/ntl-olmo2-native24-final-inference-assets) · [seed7 训练 Notebook](https://www.kaggle.com/code/lucasboesen/name-that-llm-train-lora24-s7) |

当前配置固定了检查点控制文件及底座配置、索引的发布哈希。LoRA 和底座大权重的发布方 SHA-256 尚未核验。实际运行会记录全部读取的 safetensors 权重 SHA-256，便于比较实验资产。

## 推理口径

1. 读取 `id,text`，保留唯一字符串 ID 和原行序。用 `final/tokenizer` 编码原文，不加模板或特殊 token，最多保留 447 token 后追加一个 EOS。批内右填充，按长度稳定排序后还原输入顺序。
2. 底座截断至 24 层并用 FP16 在一张 CUDA 卡上执行，关闭 KV cache。取 hidden states 索引 2、4、6、8、12、16、20、24。屏蔽 padding 和第一个 token，EOS 仍参与均值。每层在 FP32 下做发布方提供的 LayerNorm，再拼接成 32,768 维。若原文未产生 token，屏蔽后无法求均值，运行明确失败。
3. 用 `readouts.npz` 的 `plain_coef`（18 × 32,768）和 `plain_intercept` 加权并逐行 softmax。资产中 18 类顺序已经同赛事标签和 V31 排序核对。输出列依次是 `id`、`claude-fable-5`、`claude-opus-4-6`、`claude-opus-4-8`、`claude-opus-5`、`claude-sonnet-5`、`gemini-3.1-pro-preview`、`gemini-3.6-flash`、`gemini-3.7-flash`、`gemini-3.8-flash`、`glm-5.3`、`glm-5.3-flash`、`gpt-5.5`、`gpt-5.6-luna`、`gpt-5.6-sol`、`gpt-5.6-terra`、`gpt-6-astra`、`kimi-k3`、`qwen3.8-max`。V31 后续会把这个成员的概率裁剪、取对数并加权融合。组中心化 twin、Claude gate 和其余模型属于集成阶段。
4. `branch.py` 管理输入、分类头和输出合同，`gpu.py` 提供 `CudaFeatures`，`run.py` 管理离线运行和产物。未来 Kaggle Notebook 可以将 `src/` 放到导入路径并调用相同的 `CudaFeatures`、`load_readout`、`branch_outputs`，传入已挂载的目录。

## 在 3090 Ti 上验证

若服务器已经通过 PyTorch `2.12.1+cu126` 的 CUDA 实测，从仓库根目录执行：

```bash
bash scripts/run_v31_server_smoke.sh ../.venv-gpu-check/bin/python
```

脚本先检查已有虚拟环境和 GPU，然后安装 [GPU 直接依赖](../requirements-v31-gpu.txt)，保留已验证的 PyTorch。用 [KaggleHub](https://github.com/Kaggle/kagglehub/tree/v1.0.0) 下载两个固定版公开资产并缓存到 `artifacts/v31-asset-cache/`，然后在离线模式跑三条合成输入。权重首次下载约 16.4 GB；依赖和下载需要联网。若平台要求登录或同意许可，请在自己的账号完成，再重试；不要把凭据写入仓库或日志。推理报告路径会打印出来。直接依赖固定了版本，报告记录实测包版本；不修改 baseline 的锁文件。

如果资产已准备好，下面的命令可跳过下载和安装，直接验证。

通过授权渠道将底座 v1 文件解压到本地独立目录。底座索引统计约 16.38 GB 张量数据。适配器下载选择数据集 v1，使用其中的 `final/` 目录。首次实验 batch size 为 1，便于测量资源。以 36 层 8B 模型裁成 24 层估算，FP16 底座显存约 11 GiB，另有 LoRA、运行时和激活占用。实际峰值以运行为准。

从仓库根目录，在已有 Linux GPU Python 环境运行。环境需要 CUDA 版 PyTorch、Transformers、PEFT、Accelerate、safetensors、tokenizers、NumPy、pandas 和 PyArrow。V31 源码列出的离线 wheel 有 Transformers 5.14.1、Accelerate 1.15.0、safetensors 0.8.0、tokenizers 0.22.2。适配器元数据标记 PEFT 0.20.0。可用这些版本作为 Linux 验证起点，实际版本由完成记录保存。项目的 baseline 锁文件继续用于本地测试。 [Transformers](https://github.com/huggingface/transformers) · [PEFT 0.20.0](https://github.com/huggingface/peft/releases/tag/v0.20.0) · [Accelerate](https://github.com/huggingface/accelerate) · [safetensors](https://github.com/huggingface/safetensors) · [PyTorch](https://github.com/pytorch/pytorch)

```bash
BASE=/absolute/path/to/qwen-3-8b-base-v1
CHECKPOINT=/absolute/path/to/ntl-qwen3-8b-lora-claude-head-v1/final

CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=src \
  python -m ntl_repro.run --synthetic \
  --base-dir "$BASE" --checkpoint-dir "$CHECKPOINT" \
  --run-dir artifacts/v31-claude-smoke-001
```

命令用三条合成文本测试加载、截断和推理。目录成功时包含 `probabilities.parquet`、`predictions.parquet` 和最后写入的 `completed.json`。报告记录版本、哈希、加载和预测耗时以及 PyTorch 的峰值已分配与已保留显存。前处理资产校验会读完权重以计算哈希，加载和预测计时从权重哈希结束后开始。重跑使用新的目录。

在自己有权限的服务器上复核发布方锚点时，保持原版 `train.csv` 行序，向同一命令加 `--anchor-data-dir data/raw/name-that-llm` 并改用新产物目录。程序用头三行复算 pooled features 并要求相对平均绝对误差小于 0.02。若只跑合成样本，报告的 `anchor_relative_error` 为 `null`。要测少量赛事输入，可把 `--synthetic` 换成 `--data-dir data/raw/name-that-llm --limit 3`，命令只读 `test.csv`，锚点参数独立可选。保留完成记录中的行数、锚点误差、预测耗时和显存，作为自己的服务器实验记录。

本次在本地验证合成合同。服务器 CUDA 加载、原始锚点、Kaggle 隐藏测试和集成效果均待实测。单分支合成运行和锚点通过后可以分别记录对应状态。公榜成绩需要日后实际提交取得。
