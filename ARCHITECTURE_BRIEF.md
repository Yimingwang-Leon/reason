# huikang 架构 brief(基于代码 read,故意 skip 了 cryptarithm/equation_numeric)

## 五步 pipeline

```
reasoning.py        # 把 reasoners/*.py 每类跑一遍,产 reasoning/<id>.txt(CoT)
augmentation.py     # 跑 augmenters/{spelling,concat,split,match,lstrip},产 augmentations/<id>.txt
corpus.py           # tokenize CoT + augmentation → corpus/<id>/synthetic.jsonl + corpus.jsonl(索引)
train_sft.py        # 拉 Tinker 训 LoRA,记 per-epoch logprob
upload_adapter.py   # Tinker checkpoint → Modal volume → Kaggle Model instance(.zip)
```

跑命令:`uv run python reasoning.py && augmentation.py && corpus.py && train_sft.py && uv run modal run upload_adapter.py`

## 每个 reasoner 做什么(简单 4 类已读)

**numeral.py(58 行)**: Arabic→Roman 标准算法,贪心拆 (1000,M)/(900,CM)/.../(1,I)。每步打印 `n >= val -> sym, remainder`,最后 `\boxed{result}`。**没有学习例子里的 char_set**——直接套标准罗马表,所以对训练集 20 种 char-set 变体应该有失败 case(估计他靠 SFT 让模型从例子里学习覆盖)。

**unit_conversion.py(75 行)**: 对每个 example 算 `factor = output / input`,显示**长除法**逐步(`long_division_lines`),取 median factor,再用 query × factor(**长乘法**逐步)。所有算术都展开成 token-level computation——这就是"教模型按算法走 token"的精髓。

**gravity.py(87 行)**: 同 unit_conversion 模板,但算 `k = d / t²`,先算 `t²` 长乘法,再除得 k,median 选取,query 走 `d = k * t²`。

**cipher.py(381 行,最复杂的"简单"类)**: 三阶段——(1) 从 example 累积 a→z partial mapping;(2) 对 query 每个词,已映射字符直接译,未映射字符遍历整个 wonderland 字典(`wonderland.txt`)按字长 + pattern + 已有 mapping 一致性筛候选;(3) 选第一个候选词,扩充 mapping,继续下一个词。**关键依赖**:`wonderland.txt` 是题目专用词表(估计是从训练 prompt 反推出来的英文小词汇),没有这个词表 cipher 就废了。

## 训练设计(`train_sft.py` + `loss_config.py`)

- **base**:`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` + LoRA rank 32(train_mlp+attn+unembed)
- **超参**:batch=64, max_len=8192, 1 epoch, LR=2e-4 **step linear decay**(随 step 线性降到 0), Adam β=(0.9,0.95) wd=0 clip=1e9
- **micro_batch**:Tinker 自动决定(`micro_batch_size=None`),Modal 时 16
- **loss 选项**(`loss_config.py` 5 种):`cross_entropy`(默认)、`importance_sampling`、`ppo`、`cispo`、`dro`
  - **min-logprob 实现**:不是直接 min,而是 `CrossEntropyWithWeightingLossConfig` 的 `branch_logprob`(≈0.01)阈值:当某 token 的 logprob 绝对值 ≤ 0.01,full weight=1;反之 weight 按 `|lp|/0.01` 上升。**直观**:模型已经会的 token(高 logprob)训练权重低,差的(低 logprob)权重高——这是 mean loss 的"min-logprob 倾斜近似"
- **stratified batching**:`_stratified_batches` 按 category 均匀分散到 batch,每个 batch 各类样本都有
- **logprob 跟踪**:epoch 0 收集 ref_logprobs;后续 epoch 比对 ref vs current,统计 `logprob_decreased/increased` + 百分位 + diff2p 阈值分布
- **filter_training_examples 警示**:当前他这版代码硬编码 `category in ("spelling",)`——**只用 spelling augmentation 训**,显然是某次特定实验的临时改动;**我们要改回 `return examples`**(用全部)

## 训练数据流(`corpus.py`)

- 每个 `reasoning/<id>.txt` 拼成 `{reasoning_text}\n</think>\n\boxed{ans}<|im_end|>` 作为 completion
- prompt 用 `tokenizer.apply_chat_template(messages, enable_thinking=True)` —— **必须加 `\nPlease put your final answer inside \boxed{}...`** suffix(matches metric_reference)
- mask:prompt 部分 mask=0(不算 loss),completion 部分 mask=1(算 loss)
- 截到 8192 token,超的直接 truncate(可能丢答案,需要警惕)
- augmentation 走另一条路径:直接 prompt+completion 形式,**不**走 reasoning 模板

## 上传机制(`upload_adapter.py`)

- 用 Modal 起 image(`kaggle>=1.6, tinker>=0.5.1`)
- `download_adapter()`:从 Tinker 拉 checkpoint tar → 解到 `/adapter/weights`(Modal volume)
- `upload_to_kaggle()`:用 Kaggle API 创建 / 更新 model instance `huikang/nemotron-adapter/Transformers/default`
- 我们必须改成自己的 instance 名(比如 `<你的 username>/nemotron-adapter/Transformers/default`)
- 需要 `env.json` 存 `KAGGLE_API_TOKEN` 和 `TINKER_API_KEY`

## 依赖

- Python ≥ 3.11(他用 uv,我们也可以 pip)
- **关键**:`transformers==4.57.6`(他锁的版本)——但我们 memory 写了"必须 ≥ 5.3 去 trust_remote_code 修 KV cache bug"。这是 **SFT 不受影响**(他只做 SFT),如果我们将来加 RL 必须升级。**Phase 1 暂时跟他用 4.57.6**,避免引入未知差异
- `tinker>=0.16.1`(我们已 install 5.3,可能与他的不同步,先观察)
- `modal>=1.4.1`
- `jinja2`, `pydantic`

## 我们要 fork 什么

| 文件 | 动作 | 理由 |
|---|---|---|
| `reasoners/{numeral,cipher,gravity,unit_conversion}.py` | **直接 copy** | 简单 4 类没有 IP,他写得也对 |
| `reasoners/bit_manipulation.py` | **借鉴算法,自己重写** | 主战场;他算法已公开发表,我们重写为 Phase 3 残余 15% 留扩展位 |
| `reasoners/{cryptarithm,equation_numeric}.py` | **完全不看,从零写** | 思维独立性纪律(等 Phase 5 写完才能比对) |
| `train_sft.py` + `train_common.py` + `loss_config.py` + `lr_schedule.py` | **直接复用** | 训练框架,工程问题无创意 |
| `corpus.py` + `augmentation.py` | **复用** | tokenize + 数据装配,标准做法 |
| `augmenters/*` | 可选保留 | spelling/concat/split/match/lstrip 5 种数据增强,他认为有用就先留着 |
| `upload_adapter.py` | **改 instance 名** | 必须用我们自己的 Kaggle account |
| `train_sft.py::filter_training_examples` | **改回 `return examples`** | 当前硬编码只用 spelling,**bug 或临时改动** |

## 我们要 SKIP 的(纪律)

- `reasoners/cryptarithm.py`(164 行)
- `reasoners/equation_numeric.py`(603 行)
- 这两份在 Phase 5 写完我们自己版本之前**不读**

## 5 个潜在的坑

1. **`filter_training_examples` 当前过滤为 spelling**——按现状直接跑,会**只用 augmentation 数据**,不会跑出 0.85
2. **`transformers==4.57.6`** 与我们 `requirements.txt`(5.9)冲突——他用旧版是因为没加 RL;Phase 1 可以两套环境共存
3. **`wonderland.txt` 是英文词表**——cipher 强依赖,我们可能要重新生成(从训练集英文 plaintext 抽词)
4. **`upload_adapter.py` 硬编码他的 Kaggle instance**——必须改成我们的
5. **`corpus.py` truncate 到 8192 token**——超长样本被截可能丢答案,Phase 2 需要监控

## Phase 1 下一步

1. 你注册 Tinker + Modal,给 API key
2. 我把"我们要 fork"那一列文件 copy 到 `src/`(项目自己的代码目录),改 instance 名 + filter 那行 bug
3. 跑 mini Tinker 验证 pipeline
