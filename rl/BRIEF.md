# RL 突围任务书(单一事实源,所有专家先读这份)

**目标:LB 0.85 → 0.865+(+1.5pp ≈ 28 题/1899)。deadline 2026-06-15(今天 06-11,剩 4 天),5 提交/天。**

## 为什么开新路线
SFT-模仿路线已四面碰壁(证据 A 级,见 `autoresearch/PATHS.md` 死亡名单):
- run-011 = **0.85**(自训最佳,语料配方保真);run-012 = 0.84(replay+LR3.5e-4 证伪);CoT 格式重设计两个方向都 0.82。
- 探针(2026-06-11):SFT 教"搜索过程"(枚举-验证式 CoT)在**未见难题**上贪心执行 ≈0%(crypt 5%、bit-EV 0%);模型连背过的 crypt 演绎 trace 都只复现 3%。
- **这正是 RL 的经典动机:模仿学不会的"过程泛化",用结果奖励直接优化贪心答对率;on-policy 数据天然 R-安全(模型自己的分布)。**

## 比赛硬约束
- base 固定 `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`(MoE);LoRA rank ≤ 32;提交=adapter;目标模块全栈 q/k/v/o+in/out+up/down(~1.38GB,MoE up/down 必需),无 lm_head。
- 评测 harness 固定:**greedy temp=0**、max_tokens=7680、`\boxed{}` 提取、判分 = string-exact OR 相对容差 1e-2。我们只能通过**权重**影响结果。
- 训练平台 Tinker(SDK 0.22.3,API key 在 env.json);本地无 GPU;Modal 可用。

## 现状账本(2026-06-11 晚修订)
- LB = oracle × R。离线 oracle ~0.90(cipher/gravity/numeral/unit 100%,bit 92-94%,eq 77-80%,crypt 22.2%);R≈0.96-0.97。
- **run-011 = 0.85 自训最佳**(53534096);run-012 = 0.84(53556034,replay+LR3.5e-4 主菜证伪,成分混杂 B 级);彩票 alpha36/ep0/SWA50/SWA75 = 0.84/0.84/0.85/0.85 全开奖清零。
- **final-selection 地板 = 0.86 在账**:53414820(06-06)"public 0.86 adapter (kienngx) reproduce" = 0.86 COMPLETE。挑战者打 0.85/0.86 对名次贡献 = 0,只有显示 0.87 改变结局(合规审计与终选预登记见 `rl/rules_audit.md`)。
- **run-012 holdout 解码已全量落账(770/770 行):总 615/770 ≈ 0.799 题面、外推 ≈0.866 vs LB 0.84(世界 B)。明细:四健康类 440/440 全对,bit 91/110,eq 84/110,crypt 0/110。** 丢分全在 bit/eq/crypt 三类 → 猎物池即由此圈定;holdout 口径收益一律打 0.7 折扣。
- run-011 (0.85) 为最佳 checkpoint,Tinker 上有 sampler_weights 可采样、weights+optimizer state 可续训;run-011 资产物理不动、永不覆盖。
- holdout 1899 题(从 9500 train 切出);run-012 在 train_split 7601 上训。
- 实测成本:训练 ~$0.04-0.077/步(batch 64, 8k ctx);采样 ~$0.0025/rollout(死区挖矿 ~2000 rollouts ≈ $5)。
- 关键采样证据:**死区**(solver 也不会的题)pass@8 ≈ 1-2%(eq 1.8%、crypt ~1%)→ RL 翻不动 oracle 死区;**营救臂**(solver 会/贪心错的 R-loss 题)采样+splice 实测 17% 可救 → RL/RFT 的主猎物是 R-loss 与 marginal 题。
- 反遗忘教训:60 步小补丁 FT(LR 8e-5)就把已对锚题打到 80% → 任何 RL/FT 必须带 KL 锚/replay/低 LR 设计。

## 死亡名单(不许再碰,除非明确论证"为什么这次不同")
权威表:`autoresearch/PATHS.md`。要点:CoT 格式重设计、crypt synth SFT、crypt closed-form >22.5%、equation tiebreak、bit 断言式格式、lm_head、滤 MoE、死区采样捡漏。

## 必读文件
- `autoresearch/PATHS.md`(死亡名单)/ `autoresearch/research-log.md`(run-011/012 全史,文末)/ `autoresearch/findings.md`
- `src/train_tinker.py`(我们的训练入口)/ `external/nemotron-huikang/loss_config.py`(含 importance_sampling/ppo/cispo)/ `external/nemotron-huikang/train_sft.py`
- `src/r_harness.py`、`src/eval_gate.py`、`src/problems.py`、`split_holdout.py`
- `data/run012_holdout_decode.jsonl`(770 行 holdout 贪心解码,已全量判分:615/770,明细见上)
- `baseline.json`、`corpus.jsonl`

## 纪律(违反即提案作废)
1. **本阶段只设计不花钱**:禁止任何 tinker/modal API 调用、禁止 kaggle 提交;只读代码/数据/联网检索。
2. 每条提案必须有:数字账(+X 题从哪来)、成本上限、4 天内的时刻表、预登记预测与证伪线、反遗忘设计。
3. 引用死亡名单路径必须写"为什么这次不同",否则红队直接毙。
4. 产出写到 `rl/` 下指定文件,中文。
