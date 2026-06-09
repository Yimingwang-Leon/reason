# 实验台账 (Experiment Ledger) — Nemotron Reasoning

**纪律:每个实验先登记(假设 / 单变量改动 / 精确配置 / 预测),再跑,再补结果+结论(留/弃,对比噪声底)。一行一总结,可追溯、可复现。混杂(多变量)实验明确标注。**

## 干净实验协议(红队 wf_1d741661 硬化中,待填最终版)
- **训练 = `train_split.csv`(7601);评估 = `holdout.csv`(1899,与训练不相交)用 Tinker greedy sampling → 每类准确率 = oracle × 泛化R(不用 LB、可分类目)。**
- **噪声底**:每类 holdout 二项 SE = √(p(1-p)/N)(crypt N=165→SE≈3%);+ 同配置换 seed 跑 2 次的 run-to-run 方差。**Δ < 噪声底 = 噪声,不算信号。**
- **单变量**:对比两次只动目标类的 CoT/数据,其余字节不变;seed/epoch/LR/curriculum/batch 全相同。未改的确定性 4 类 = 对照组,Δ 应≈0(否则有混淆)。
- **决策**:只有 Δ > 噪声底 的改进,才放进**全量 train.csv** 模型 → LB。
- 地基 = run-005 的 0.84 配方(legacy reasoner + curriculum),不从头来;train_split 只是干净测量台。

## 每实验登记模板
```
### <exp-id>  [PLANNED|DONE]  日期
- 假设:
- 单变量改动(vs 哪个 baseline):
- 训练数据 / 配置:seed= epochs= LR= curriculum= batch= train_unembed=
- 预测(跑前写,防事后合理化):
- 结果:LB= / holdout 每类准确率=
- Δ vs 噪声底:
- 结论(留/弃 + 为什么):
- checkpoint:tinker://...
```

## 台账(历史 005–010 如实补录)
| ID | 改动 | 数据 | 关键配置 | LB | 干净? | 结论 |
|---|---|---|---|---|---|---|
| run-005 | legacy reasoner(局部逐列匹配)基线 | train.csv 全量 ⚠含holdout | curr, 3ep, 2e-4, no-lm_head | **0.84** | 单点(无对照) | **自训最优基线**;ckpt `de3a5482` |
| run-006 | 结构化 bit_manip + lm_head | 全量 | 2ep | 0.71 | ❌混杂(2变量) | R 崩(结构化全局规则) |
| run-007 | 结构化 terse | 全量 | | 0.73 | ❌ | R 崩,确认非"应用步"问题 |
| run-008 | 结构化+逐列脚手架 | 全量 | | 0.72 | ❌ | ≈007,确认全局规则断言是 R 杀手 |
| run-009 | bit_manip瘦身 + 子技能augmenter + 2M replay + lm_head | 全量 | 2ep | 0.82 | ❌**4变量混杂** | 不可归因;"加数据"未帮上 |
| run-010 | 5类 locality-hardening CoT | 全量 ⚠含holdout | 2ep(005是3ep) | 0.82 | ❌**5类+epoch混杂,无噪声底** | **不可定论**(我曾误判为"改CoT死路"——证据不足) |
| run-010-filtered | run-010 删 MoE LoRA | 同上 | 提交侧改动 | 0.56 | 提交侧单变量 | ✅ 证实 MoE LoRA 必需(删→残废) |

## 关键教训(已进记忆)
- oracle↑ ≠ LB↑;全局规则断言→R 崩(run-006/7/8 实证)。
- "加现有数据"无效(run-009);模型不缺数据(nll 0.003),缺的是"解题器多解题/合成新技能数据"。
- **从未量过噪声底 + holdout 泄漏进训练 → 过去所有"结论"都可能被噪声/污染污染。干净协议从此强制。**

## 待跑(干净)实验队列 — 等红队定最省方案后填
- E1: 噪声底(同配置 train_split 跑 2 seed)
- E2: equation 解题器改进(oracle 77%→?,局部 CoT)
- E3: crypt 合成数据 + 解题器(oracle 17.6%→?)
