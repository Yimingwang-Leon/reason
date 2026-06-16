# Nemotron Reasoning Challenge — 🥈 银牌方案 (51/榜)

NVIDIA Nemotron Model Reasoning Challenge 的参赛代码与完整研究记录。
**最终成绩:private LB 0.86,第 51 名,银牌。**

---

## TL;DR — 这块牌是怎么拿到的

LB = **oracle**(解题器能解对的比例) × **R**(模型贪心复现解题过程的可靠度)。
核心打法**不是教模型推理,是教模型"照抄一个确定性解题器的逐步解题过程"**:

1. 给每个题型写 Python 解题器,把解题的**每一步**打印成文字(不是只给答案);
2. 用这些"标准解题过程"做 LoRA SFT,让模型学会照套路一步步写;
3. 答案 = 解题器自洽算出的值(不重写),模型学的是**过程**而非记忆。

**真正产生银牌模型的配方 = `run-012`**(git tag `silver-medal-run012`):

```
run-011 基础 (private 0.85):
  crypt 22.2% solver trace + equation 全枚举验证 trace
  + bit_manipulation 逐位推导 + champion drills
  + LoRA rank-32 全模块栈(含 MoE up/down_proj) + min-logprob curriculum + 2 epoch
        ↓ run-012 关键增益 (private 0.85 → 0.86):
  + math-replay(配平) + LR 3.5e-4 + train_split-only(剔除 holdout 防过拟合)
```

> ⚠️ **公榜会骗人**:run-012 公榜仅 0.84,private 才是 0.86。本赛公榜系统性误导——
> 公榜 0.84 的(run-012/rft-r1)private 全是 0.86;公榜最高 0.86 的"复制票"private 反而 0.84。
> **绝不可用公榜分做终选或判生死。**

### 与公开解法的区别

整体框架与 huikang(Progress Prize 得主)谱系**同源**,无算法突破。真正的差异在**工程纪律**:
- **自己重新生成全部语料**(自洽答案),不原样复制他人 adapter 权重;
- **equation 全枚举 trace** 自己从零写(穷举验证运算符,run-011 +1pp 主源);
- **train_split-only**:训练剔除 holdout → private(未见分布)泛化更稳。
  这是"我们 private 0.86 vs 公开复制 private 0.84"差 2pp 的最可能原因。

### 终选决策(同样决定成败)

终选 2 份取 private 较高者。规则:**选权重最独立的两份,不选公榜最高分**。
- 第一签 `run-011`(private 0.85),第二签 `rft-r1`(ΔW cos=0.237,最独立);
- **剔除公榜最高的复制票**(它是 Kaggle 默认自动选项,private 掉到 0.84,自动选就丢牌)。

---

## 仓库结构

```
src/
  reasoners/            # 各题型确定性解题器(核心:把解题过程展开成 token 级 CoT)
    bit_manipulation.py     # 逐位推导(注:当前为赛后新规则族版,银牌用版见 tag)
    cryptarithm_deduce.py   # 列式推导(oracle ~22%,硬上限)
    equation_numeric_deduce.py  # 全枚举验证(自研,+1pp 主源)
    cipher.py / numeral.py / gravity.py / unit_conversion.py  # 确定性算法展开
  augmenters/           # 数据增强(spelling/concat/split/match 等)
  corpus.py             # CoT + 增强 → tokenized 训练语料
  train_tinker.py       # 训练入口(Tinker LoRA;支持 --init-from-state/--constant-lr/--jsonl-corpus)
  build_submission.py   # Tinker checkpoint → adapter zip → Kaggle 提交
  eval_gate.py / r_harness.py / problems.py / reasoning.py   # 判分与评测

autoresearch/           # 研究日志、实验账本、路径生死总表(PATHS.md = 单一事实源)
rl/                     # 赛后 RL 探索 + 诊断(见下;大多为"已证伪"的诚实记录)
data/                   # 小数据已入库;adapter/replay/rollout 等大文件已 gitignore
```

### `rl/` — 赛后 RL 攻坚与复盘(诚实的失败记录)

银牌敲定后为冲 0.87 做的探索,**绝大多数被实测证伪**,作为方法论沉淀保留:
- `bit_prior.py` / `bit_synth.py` — 逆向出题器规则族 + 合成增广(离线 oracle 110/110,但五轮训练 LB 未兑现)
- `rft_harvest.py` / `rft_filter.py` — 拒绝采样微调(RFT;净增 -1,无贡献)
- `merge_lora.py` — LoRA ΔW 空间融合 + 五重诊断闸(跨 init 合并实测伤分,全族死)
- `grpo_loop.py` / `pair_builder.py` — GRPO / DPO 工具(未发车,成本/时钟不划算)
- `GAP_DIAGNOSIS.md` + `AUDIT_VERDICT.md` — **离线↔在线 gap 诊断 + 独立红队复核**:
  证明"离线 vs 在线系统性差异"是等权外推制造的幻觉(真权离线 86.6% > LB,是高估自己);
  0.85→0.87 是**结构墙**(数学上界自洽口径 87.4–88.3%,P≈0.10)。

---

## 复现银牌模型

```bash
git checkout silver-medal-run012      # run-012 配方的精确代码快照
cp env.json.template env.json         # 填入 TINKER_API_KEY / KAGGLE 凭证
python -m src.corpus                  # 生成 reasoner CoT + 增强语料
python -m src.train_tinker --run-name run-012 --num-epochs 2 \
    --lr 3.5e-4 --curriculum          # + replay 配平、train_split-only(见该 commit 配置)
python -m src.build_submission --run-name run-012 --submit -m "run-012"
```

> 训练在 Tinker(LoRA,远端 GPU);本地无 GPU,只做语料生成与脚本编写。

## 硬约束(竞赛固定,不可改)

- base 模型固定 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`(MoE);LoRA rank ≤ 32
- 提交 = adapter;目标模块须全栈 q/k/v/o + in/out + up/down(**MoE up/down 必需**,缺则崩到 0.56);无 lm_head
- 评测:greedy(temp=0)、max_tokens 7680、`\boxed{}` 提取、判分 = string-exact 或相对容差 1e-2

## 关键教训(可迁移到下次)

1. **公榜 ≠ private**:用公榜判生死会扔掉真正的好配方(run-012 差点被误判)。终选靠权重独立性 + private 方差,不靠公榜分。
2. **复制公开 adapter 必死**:private 同分大簇 + 提交时间排尾 + 原创性验证不过。
3. **oracle 天花板锁死上限**:crypt 解题器硬上限 ~22%,模型再强也学不会解题器都解不出的题。
4. **离线漂亮 ≠ 在线兑现**:多次"离线 +X 题"是 holdout 过拟合假象(bit 五轮、RFT 均如此)。
5. **单线推理会自我欺骗**:最终可信结论来自多智能体对抗红队(独立复算),而非自查。

---

*完整逐次提交账本见 `autoresearch/EXPERIMENTS.md` 与 `autoresearch/PATHS.md`(路径生死单一事实源)。*
