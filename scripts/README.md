# 量化分析脚本使用指南

本文档介绍 `scripts/` 目录下两个常用量化分析脚本：

- [`analyze_perchannel_vs_perblock.py`](/data/home/cgxu2/sglang/scripts/analyze_perchannel_vs_perblock.py)
- [`analyze_sglang_bf16_fp8_layers.py`](/data/home/cgxu2/sglang/scripts/analyze_sglang_bf16_fp8_layers.py)

它们的定位不一样：

- `analyze_perchannel_vs_perblock.py`
  用于量化前分析。它不依赖真实量化 checkpoint，而是对 FP 模型做一层层的 `FP8 E4M3 fake quant`，帮助判断一层更偏向 `per-channel` 还是 `per-block`。
- `analyze_sglang_bf16_fp8_layers.py`
  用于量化后验证。它会分别通过 SGLang 的真实加载路径跑 BF16 和 FP8 checkpoint，对比每层输出和最终 logits 的差异。

推荐把它们串成一套实验流程：

1. 先用 `analyze_perchannel_vs_perblock.py` 做策略预分析。
2. 再做真实量化导出。
3. 最后用 `analyze_sglang_bf16_fp8_layers.py` 验证真实 BF16/FP8 checkpoint 的层级误差和 logits 差异。

## 1. 环境准备

建议在已经能正常运行 SGLang 和 Transformers 的环境里使用。

常见依赖：

- `torch`
- `transformers`
- `sglang`
- 模型对应 tokenizer

如果要跑第二个脚本，还需要：

- 能正常通过 SGLang 加载 BF16 模型和 FP8 量化模型
- CUDA/ROCm 环境与量化模型格式兼容

## 2. 校准数据格式

这两个脚本在当前实现里，`--prompts-file` 都按 `JSONL` 逐行读取。

推荐使用下面三种行格式之一：

```jsonl
{"text": "Explain quantization briefly."}
{"prompt": "What is per-channel quantization?"}
{"messages": [{"role": "user", "content": "Why can block quantization hurt accuracy?"}]}
```

说明：

- 第一种、第二种会直接取字符串作为 prompt。
- 第三种会把 `messages` 转成最终输入文本。
- 如果你的文件是普通纯文本而不是 JSONL，这两个脚本当前都不适用。

## 3. 脚本一：量化前分析 per-channel vs per-block

脚本路径：

- [`analyze_perchannel_vs_perblock.py`](/data/home/cgxu2/sglang/scripts/analyze_perchannel_vs_perblock.py)

### 3.1 它做什么

对每个 `Linear` 层，脚本会做三件事：

1. 看权重分布。
2. 抓一小批真实输入激活样本。
3. 分别对权重做：
   `per-channel FP8 E4M3 fake quant`
   `per-block FP8 E4M3 fake quant`

然后比较：

- 权重误差
- 层输出误差

最后给每层一个结论：

- `leans per-channel`
- `leans per-block`
- `close`

### 3.2 它不做什么

它不是在跑真实 SGLang FP8 checkpoint。

它的结果适合回答：

- 哪些层更敏感
- 哪些层更像是 `per-channel` 候选
- 哪些层更像是 `per-block` 候选

但它不能替代真实量化 checkpoint 的端到端验证。

### 3.3 最常用命令

```bash
python scripts/analyze_perchannel_vs_perblock.py \
  --model-path /path/to/hf-model \
  --prompts-file /path/to/calib.jsonl \
  --block-size 128 128 \
  --output-json perchannel_vs_perblock.json \
  --output-csv perchannel_vs_perblock.csv
```

### 3.4 常见参数

- `--model-path`
  HF 模型路径。

- `--tokenizer-path`
  tokenizer 路径，默认等于 `--model-path`。

- `--prompts-file`
  校准集，当前按 `JSONL` 逐行读取。

- `--max-prompts`
  最多读取多少条 prompt。

- `--max-input-length`
  每条 prompt 最多截断到多少 token。

- `--samples-per-layer`
  每层最多保留多少条激活样本，默认 `96`。

- `--block-size ROWS COLS`
  `per-block` 的块大小，默认是 `128 128`。

- `--include`
  只分析匹配到的层名，支持重复传入。

- `--ignore`
  跳过某些层名。

- `--skip-activations`
  只看权重，不抓激活。

- `--top-k`
  每类结果在终端最多显示多少行。

### 3.5 推荐的起手命令

如果你只想先看 attention 和 MLP 主干层：

```bash
python scripts/analyze_perchannel_vs_perblock.py \
  --model-path /path/to/hf-model \
  --prompts-file /path/to/calib.jsonl \
  --block-size 128 128 \
  --include "q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj" \
  --output-json perchannel_vs_perblock.json
```

如果你只想做静态权重预分析：

```bash
python scripts/analyze_perchannel_vs_perblock.py \
  --model-path /path/to/hf-model \
  --skip-activations \
  --block-size 128 128
```

### 3.6 它的输出怎么读

这个脚本会给出三类层：

- `Top layers that lean per-channel`
- `Top layers that lean per-block`
- `Representative close layers`

表格里最重要的列：

- `out_mse_pc`
  `per-channel` 下的层输出相对误差。越小越好。

- `out_mse_pb`
  `per-block` 下的层输出相对误差。越小越好。

- `w_mse_pc`
  `per-channel` 下的权重相对误差。越小越好。

- `w_mse_pb`
  `per-block` 下的权重相对误差。越小越好。

- `ch_spread`
  通道间尺度离散度。越大说明不同输出通道差异越大。

- `blk_spread`
  block 间尺度离散度。越大说明不同局部 block 差异越大。

### 3.7 判定规则

当抓到了激活样本时，优先看输出误差：

- 如果 `out_mse_pb / out_mse_pc >= 1.35`
  判为 `leans per-channel`

- 如果 `out_mse_pb / out_mse_pc <= 0.85`
  判为 `leans per-block`

- 其他情况判为 `close`

如果没抓激活，只能退化成权重统计判断。

### 3.8 什么时候用它

适合这些场景：

- 你还没导出真实量化 checkpoint
- 你想先判断哪些层值得单独处理
- 你想决定默认用 `per-channel` 还是 `per-block`

不适合这些场景：

- 你想回答“真实 SGLang FP8 checkpoint 到底误差多大”
- 你想直接替代端到端验证

## 4. 脚本二：用 SGLang 真实加载路径对比 BF16 和 FP8 checkpoint

脚本路径：

- [`analyze_sglang_bf16_fp8_layers.py`](/data/home/cgxu2/sglang/scripts/analyze_sglang_bf16_fp8_layers.py)

### 4.1 它做什么

这个脚本不是 Hugging Face 直接 forward hook，而是走 SGLang 的低层加载与 prefill 路径。

这样做的意义是：

- FP8 checkpoint 会通过 SGLang 实际 serving 时使用的量化代码加载
- 更接近真实部署链路

它会对比：

- 每一层 decoder layer 的输出状态
- 最终 logits

### 4.2 最常用命令

```bash
python scripts/analyze_sglang_bf16_fp8_layers.py \
  --bf16-model-path /path/to/bf16-model \
  --fp8-model-path /path/to/fp8-model \
  --fp8-quantization compressed-tensors \
  --prompts-file /path/to/calib.jsonl \
  --output-json bf16_vs_fp8_layers.json
```

### 4.3 常见参数

- `--bf16-model-path`
  BF16 参考模型路径。

- `--fp8-model-path`
  FP8 或 mixed precision checkpoint 路径。

- `--fp8-quantization`
  SGLang 加载 FP8 checkpoint 时使用的量化名称，默认是 `compressed-tensors`。

- `--bf16-quantization`
  一般不用传，只有参考模型也需要特殊量化加载时才用。

- `--prompts-file`
  当前实际按 `JSONL` 逐行读取。

- `--max-prompts`
  最多读取多少条 prompt。

- `--max-input-tokens`
  输入 token 上限。

- `--dtype`
  SGLang 运行 dtype，默认 `bfloat16`。

- `--device`
  默认 `cuda`。

- `--tp-size`
  当前脚本只支持 `1`。

- `--bf16-kv-cache-dtype`
  BF16 模型 KV cache dtype，默认 `bf16`。

- `--fp8-kv-cache-dtype`
  FP8 模型 KV cache dtype，默认 `bf16`。

- `--attention-backend`
  如果你需要固定 backend，可以显式指定。

### 4.4 输出指标怎么读

脚本会打印 `Layer comparison` 和 `Logits comparison`。

每层指标：

- `cos`
  量化前后层输出余弦相似度。越接近 `1` 越好。

- `rel_l2`
  相对 L2 误差：
  `||a - b|| / ||a||`
  越小越好。

- `mae`
  平均绝对误差。越小越好。

- `max_abs`
  最大绝对误差。越小越好。

logits 指标：

- `cos`
  BF16 与 FP8 logits 的余弦相似度。

- `rel_l2`
  logits 的相对 L2 误差。

- `mae`
  logits 平均绝对误差。

- `max_abs`
  logits 最大绝对误差。

- `top1_match`
  top1 token 一致率。

- `top{k}_overlap_mean`
  top-k 候选重叠率。

### 4.5 什么时候用它

适合这些场景：

- 你已经有 BF16 checkpoint 和 FP8 checkpoint
- 你想知道真实 SGLang 加载路径下，误差主要集中在哪些层
- 你想对比不同量化配置的真实效果

### 4.6 这个脚本的限制

- 目前只支持 `--tp-size 1`
- 会真正加载两套模型，显存和加载时间都明显高于脚本一
- 需要你的 FP8 checkpoint 能被当前 SGLang 分支正确加载

## 5. 推荐实验流程

如果你要做一套比较完整的量化实验，推荐顺序如下：

### 步骤 1：先做量化前粒度分析

```bash
python scripts/analyze_perchannel_vs_perblock.py \
  --model-path /path/to/hf-model \
  --prompts-file /path/to/calib.jsonl \
  --block-size 128 128 \
  --output-json step1_perchannel_vs_perblock.json
```

看什么：

- 哪些层明显偏 `per-channel`
- 哪些层偏 `per-block`
- 哪些层是 `close`

### 步骤 2：导出真实量化 checkpoint

这一步取决于你的量化工具链，比如 `llmcompressor`、`compressed-tensors`、其他离线量化流程。

重点是记录：

- 量化格式
- `per-channel` / `per-block` 配置
- block size
- 是否跳过敏感层

### 步骤 3：对比真实 BF16 / FP8 checkpoint

```bash
python scripts/analyze_sglang_bf16_fp8_layers.py \
  --bf16-model-path /path/to/bf16-model \
  --fp8-model-path /path/to/fp8-model \
  --fp8-quantization compressed-tensors \
  --prompts-file /path/to/calib.jsonl \
  --output-json step3_bf16_vs_fp8.json
```

看什么：

- 哪些 layer 的 `cos` 最差
- 哪些 layer 的 `rel_l2` 最大
- logits 的 `top1_match` 和 `topk_overlap_mean`

### 步骤 4：回到策略层做修正

如果真实 BF16/FP8 差异集中在少数层，可以回到量化配置里做这些调整：

- 对敏感层改用 `per-channel`
- 缩小 `per-block` 的 block size
- 跳过极少数高风险层
- 调整校准集

## 6. 常见建议

### 6.1 校准集不要太随便

这两个脚本都很依赖输入 prompt 分布。

如果你的线上负载主要是：

- 长上下文对话
- 代码补全
- 数学推理

那校准集最好跟真实分布接近，否则层敏感度判断会偏。

### 6.2 先看输出误差，再看解释性指标

对脚本一来说：

- `out_mse_pc`
- `out_mse_pb`

优先级高于：

- `ch_spread`
- `blk_spread`

因为前者更接近“这一层实际算出来偏了多少”。

### 6.3 `close` 很多是正常的

如果最后很多层都是 `close`，通常不是脚本坏了，而是这些层对两种粒度都没有特别强的偏好。

这时候更该关注：

- 极少数特别敏感层
- 最终真实 checkpoint 的层输出和 logits 对比

### 6.4 脚本一和脚本二不要混着解释

脚本一回答的是：

- “量化前，按层看，这层更像适合哪种粒度”

脚本二回答的是：

- “真实量化后，SGLang 跑起来，这层和 logits 到底偏了多少”

两者互补，但不是同一个问题。

## 7. 常见问题

### Q1：为什么脚本一和脚本二结果不完全一致？

因为脚本一做的是 `FP8 fake quant` 的局部分析，脚本二比的是实际量化 checkpoint 加载后的真实行为。

差异可能来自：

- 真正的量化导出方式
- kernel / packing
- scale 存储格式
- 运行时实现

### Q2：`--prompts-file` 能不能直接传纯文本？

当前这两个脚本都建议传 `JSONL`。不要假设纯文本一定可用。

### Q3：为什么第二个脚本显存压力更大？

因为它要分别加载 BF16 和 FP8 checkpoint，并且跑真实 SGLang 路径。

### Q4：应该先信哪个脚本？

建议是：

- 做策略判断时先看脚本一
- 做最终验证时以脚本二为准
