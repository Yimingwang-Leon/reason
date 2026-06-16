# Nemotron Model Reasoning Challenge

NVIDIA Nemotron Model Reasoning Challenge 参赛方案。

**成绩:Private LB 0.86,第 51 名(🥈 银牌)。**

---

## 方案概述

竞赛要求在固定 base 模型 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` 上训练一个 LoRA(rank ≤ 32)适配器,对七类推理题(bit_manipulation / cryptarithm / equation / cipher / numeral / gravity / unit_conversion)在 greedy 解码下输出 `\boxed{}` 答案。

得分可分解为:

```
LB = oracle × R
  oracle = 确定性解题器能解对的题目比例
  R      = 模型在贪心解码下复现解题器解题过程的可靠度
```

本方案的核心:**为每类题目编写确定性解题器,把完整解题过程展开成逐步推理文本(CoT),用这些文本做 LoRA 监督微调,让模型学会照同样的步骤推导并给出答案。** 训练标签中的答案使用解题器自洽算出的值,模型学习的是推理过程而非记忆答案。

## 算法

### 1. 确定性解题器(`src/reasoners/`)

每个题型一个解题器,输出 token 级逐步推导:

| 题型 | 解法 |
|---|---|
| `bit_manipulation` | 逐位列出 `out_i = f(输入位)` 的计算,一位一行 |
| `equation_numeric_deduce` | 穷举所有候选运算符,逐个用示例验证,选唯一一致的 |
| `cryptarithm_deduce` | 列式进位推导,反解字符→数字映射 |
| `cipher` | 从示例累积字母映射,对查询逐词解密(依赖 `wonderland.txt` 词表) |
| `numeral` | 罗马数字标准算法逐步拆解 |
| `gravity` / `unit_conversion` | 比例/因子估计,长乘除法展开成逐步计算 |

### 2. 语料构建(`src/corpus.py` + `src/augmenters/`)

- 解题器 CoT 拼接为 `{推理过程}\n</think>\n\boxed{答案}<|im_end|>` 作为训练 completion;
- prompt 段 mask=0(不计损失),completion 段 mask=1;
- 数据增强(`augmenters/`):spelling / concatenation / splitting / matching 等;
- 过滤错误答案行、超长(>7680 token)行。

### 3. 训练(`src/train_tinker.py`)

- **LoRA rank-32**,目标模块覆盖全栈 `q/k/v/o + in/out + up/down_proj`(含 MoE 专家层的 up/down_proj);
- **min-logprob curriculum**:对模型已掌握的 token 降权、未掌握的 token 升权,把训练算力集中到难点;
- math-replay 配平数据 + 2 epoch + 恒定学习率;
- 训练在 Tinker(远端 GPU LoRA)上进行。

### 4. 提交(`src/build_submission.py`)

Tinker checkpoint → 后处理为 bf16 adapter → 打包 zip → 上传 Kaggle。

## 仓库结构

```
src/
  reasoners/          # 七类题型的确定性解题器
  augmenters/         # 数据增强
  corpus.py           # CoT + 增强 → tokenized 训练语料
  train_tinker.py     # 训练入口(Tinker LoRA)
  build_submission.py # checkpoint → adapter zip → 提交
  eval_gate.py / r_harness.py / problems.py / reasoning.py  # 判分与评测
autoresearch/         # 实验日志与账本
rl/                   # 进阶实验(RFT / 模型融合 / GRPO / 诊断)
data/                 # 训练/评测数据(大文件权重不入库)
```

## 复现

```bash
cp env.json.template env.json          # 填入 TINKER_API_KEY / KAGGLE 凭证
pip install -r requirements.txt

python -m src.corpus                    # 生成解题器 CoT + 增强语料
python -m src.train_tinker \
    --run-name run-012 --num-epochs 2 --lr 3.5e-4 --curriculum
python -m src.build_submission --run-name run-012 --submit -m "run-012"
```

银牌提交对应的代码快照已打 tag:

```bash
git checkout silver-medal-run012
```

## 竞赛约束

- base 模型固定;LoRA rank ≤ 32;提交为 adapter
- 目标模块须含 MoE up/down_proj(缺失会显著掉分);不含 lm_head
- 评测:greedy 解码、max_tokens 7680、`\boxed{}` 答案提取、判分为 string-exact 或相对容差 1e-2

## 依赖

见 `requirements.txt`(本地数据/语料处理 + HF 训练栈;GPU 训练在远端环境)。
