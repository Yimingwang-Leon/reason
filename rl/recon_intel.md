# RL 突围 · 文献与社区情报侦察报告

**日期:2026-06-11 | 侦察员:recon | 任务:回答 BRIEF 六问,每条带出处。本阶段零花费(只读本地 + 联网检索 + kaggle 只读 CLI)。**

---

## TL;DR(给红队/规划的 60 秒版)

1. **量级**:在"已 SFT 收敛"的模型上,RLVR 的可信增益是 **+1~5pp pass@1**(Tülu 3:RLVR 终段 +3.3 GSM8K / +1.7 MATH;DeepSeekMath:GRPO 压在 SFT-Instruct 上 +5.3 GSM8K / +4.9 MATH)。机制共识:**RL 不教新能力,只把 pass@k 里已有的正确路径压进 pass@1**(Yue et al. "Limit of RLVR")——这与我们"营救臂 17% 可救、死区 pass@8≈1-2% 翻不动"的探针完全互相印证。**RL 的猎物 = R-loss + marginal 题,死区不是猎物。**
2. **LoRA rank≤32 做 RL 绰绰有余**:Thinking Machines 实测 **rank=1 的 LoRA 在 policy-gradient RL 上就能打平 full fine-tuning**(信息论:每 episode 只学 ~1 bit)。LoRA LR ≈ full-FT 的 **10×**;必须覆盖 MLP/MoE 层(我们提交本来就全栈,合规)。
3. **小数据 GRPO 的修正取舍**(greedy 评测视角):最相关的是 **DAPO 的 dynamic sampling**(扔掉全对/全错的零方差组——省钱且去噪)与 **Dr.GRPO 的去长度偏置**;熵塌缩对 greedy 提交本身无害(我们交的就是 argmax),但会提前杀死训练期探索 → clip-higher/温度≥1 维持。KL 锚在文献里普遍被砍(DAPO/Dr.GRPO β=0),但**我们必须留**——不是为了 RL 稳定,而是为了反遗忘锚题(60 步 8e-5 打掉锚题 80% 的教训);ProRL(NVIDIA 自家)就是"KL+定期 reset 参考策略"的范本。
4. **DPO 系做"自采样对 vs 贪心错"有成熟先例(Iterative RPO)但有一个致命坑**:同题正/误 trace 编辑距离极小 → **likelihood displacement**(chosen 概率反降)。必须加 **NLL 项(RPO)或 DPOP 罚**;而 greedy 评测下真正抬 argmax 的就是 NLL 项 → **RFT(拒绝采样 SFT)+ replay 是该族里 greedy-对齐度最高、风险最低的成员**。
5. **比赛社区**:LB 实况(06-11 拉取)**0.89×1 / 0.88×3 / 0.87×30 / 0.86×1495 / 0.85×510**。公开披露全部是 SFT 路线:0.86 大包 = 共享 adapter 热启;公开"0.87 攻略"= huikang adapter 热启 + failure mining + 1.2 万合成题 + 课程 SFT。**公开 GRPO notebook 全是玩具规模(220 题×4 rollout),无任何 RL 做出 >0.85 的公开证据**——RL 在本比赛是无人区,既是风险也是差异化機会。
6. **4 天最小可行 RLVR**:从 run-011 续(weights+optimizer 在 Tinker);题集 = R-loss+marginal(~1-2k 题);**G=8~16 rollouts/题、64~128 题/批、loss=importance_sampling(不稳再切 cispo,tinker 原生支持)、LR 5e-6~2e-5(两点小扫)、KL 锚或重 replay、100~300 迭代**。账面成本 ~$1.4/迭代(P=64×G=8),全程 $150~450,4 天排得下。社区经验:**前 100~150 步 reward 可能纹丝不动,≥300 步才下结论**(unsloth 指南)——4 天窗口必须预登记"何时止损"。

---

## Q1 · RLVR/GRPO 在已 SFT 收敛的模型上还能提多少?"SFT 背了但 RL 才会执行"有无证据?

### 1.1 量级证据(都是"SFT/DPO 收敛后再上 RL"的设定)

| 工作 | 设定 | RL 增益(pass@1/greedy 同类指标) | 出处 |
|---|---|---|---|
| DeepSeekMath 7B | GRPO 压在 **SFT 收敛的 Instruct** 上 | GSM8K 82.9→88.2(+5.3),MATH 46.8→51.7(+4.9) | [arXiv:2402.03300](https://arxiv.org/abs/2402.03300) |
| Tülu 3 (Ai2, COLM'25) | SFT→DPO→**RLVR 终段** | DPO ckpt 之上 +3.3 GSM8K、+1.7 MATH、+1.3 IFEval | [arXiv:2411.15124](https://arxiv.org/pdf/2411.15124) / [Ai2 blog](https://allenai.org/blog/tulu-3-technical) |
| 1-shot RLVR | 单题 RLVR(Qwen2.5-Math-1.5B base) | MATH500 36.0→73.6(但 base 弱、潜能大,不是收敛 SFT) | [arXiv:2504.20571](https://arxiv.org/abs/2504.20571) |
| ProRL (NVIDIA) | 长程 RLVR(含逻辑 puzzle 域) | 数千步后还在涨,且声称能扩展边界(争议见下) | [arXiv:2505.24864](https://arxiv.org/abs/2505.24864) |

**对我们的换算**:文献"收敛后 +1~5pp"是全 benchmark 口径;我们 LB=oracle×R、oracle≈0.90,RL 可触达的池子 = R-loss(~3-4pp)+ marginal。营救臂实测 17% 可救 → 文献量级与我们 +1~1.5pp 的目标**数量级吻合,不属于奢望**。

### 1.2 机制:RL 提的是"执行率",不是"会不会"

- **Limit of RLVR**(Yue et al.):RLVR 提 pass@1 / 小 k,但大 k 的 pass@k **不超过甚至低于 base**——所有正确路径本来就在 base 支撑集里,RL 只是**采样效率压缩**(把 pass@k 压进 pass@1)。[arXiv:2504.13837](https://arxiv.org/abs/2504.13837) / [项目页](https://limit-of-rlvr.github.io/)
- DeepSeekMath 原文同款结论:RL "并不必然教新能力,而是**提高正确推理路径被生成的概率**、增强已有能力的可靠性"(K=1 提升、K 大不提)。[arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- **推论(对照我们的探针)**:
  - 死区(pass@8≈1-2%,eq 1.8%/crypt ~1%)→ 支撑集里就没有 → **RLVR 翻不动,文献站在我们 PATHS.md 死刑判决这边**;
  - 营救臂(solver 会/贪心错/采样能对)→ 这恰是"pass@k>0 待压缩"人群 → **RL 的主战场,理论与我们 17% 实测一致**。

### 1.3 "SFT 记住了但 RL 才能泛化执行"的直接证据

- **SFT Memorizes, RL Generalizes**(Chu et al., ICLR'25):规则型任务(GeneralPoints/V-IRL)上,outcome-reward RL 在规则/视觉 OOD 变体上泛化,SFT 记训练分布、OOD 崩;但 **SFT 仍是 RL 的前置稳定器**(先把格式/基本能力焊住)。[arXiv:2501.17161](https://arxiv.org/pdf/2501.17161) / [OpenReview](https://openreview.net/forum?id=dYur3yabMj)
- **RL Fine-Tuning Heals OOD Forgetting in SFT**:SFT 期间丢的 OOD 能力,后接 RL 能"治愈"——机制是**旋转回奇异向量方向而非学新知识**;"SFT 记忆、RL 泛化"被该文修正为"SFT 过度专注、RL 恢复"。[arXiv:2509.12235](https://arxiv.org/html/2509.12235v2)
- **Quagmires in SFT-RL Post-Training**:警告"SFT 分数高"可能误导后续 RL 的起点选择(高 SFT 分 ≠ 好 RL 起点)。[arXiv:2510.01624](https://arxiv.org/pdf/2510.01624)
- ⚠️ **反面警告——Spurious Rewards**:Qwen2.5-Math 上**随机奖励也能 +21pp**(真奖励 +29pp),因为 RL 只是把 base 里的"代码推理"潜在模式放大;该效应**强烈依赖模型家族**(Llama/OLMo 上无效)。对我们的含义:RL 实验必须设"对照臂"(如 format-only reward)才能归因,且不能指望 RL 凭空造出 crypt 演绎能力。[arXiv:2506.10947](https://arxiv.org/abs/2506.10947)

**结论 Q1**:文献支持"背过≠贪心会执行,RL 通过结果奖励直接优化执行率"这个动机,**但只对支撑集内(可采样出正确)的题成立**。预期收益应按"营救臂×迁移率"记账,不按死区记账。

---

## Q2 · LoRA(rank≤32)做 RL 够不够?TML 结论 + tinker cookbook 默认配置

### 2.1 LoRA Without Regret(Thinking Machines, Schulman 等)

出处:[thinkingmachines.ai/blog/lora](https://thinkingmachines.ai/blog/lora/)(及 [TRL 复现文档](https://huggingface.co/docs/trl/en/lora_without_regret))

- **RL 容量需求极低**:"LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for RL, **even with ranks as low as 1**"。信息论:policy gradient 每 episode 只提取 ~1 bit;rank-1 Llama-8B LoRA 已有 3M 参数 ≈ 需求的 10 倍。**→ rank 32 对我们的 RL 不构成任何瓶颈。**
- **LR 规则**:LoRA 最优 LR ≈ full-FT 的 **10×**,SL 和 RL 都成立;最优 LR 对 rank 基本不变。
- **层覆盖**:attention-only LoRA 显著差于 MLP-only;结论"apply LoRA to **all weight matrices, especially MLP and MoE layers**" → 与我们提交约束(全栈 q/k/v/o+in/out+up/down,MoE up/down 必带)完全一致,无需改动。
- 大 batch 下 LoRA 罚比 full-FT 大,但 RL 的有效 batch(token 级)不大,影响小。

### 2.2 tinker cookbook RL recipe 默认配置(本地源码实测,版本 0.4.1)

来源:`/opt/anaconda3/envs/nvdia_kaggle/lib/python3.12/site-packages/tinker_cookbook/`

| 参数 | `recipes/rl_loop.py`(GRPO-style 最小循环) | `recipes/math_rl/train.py`(正式 recipe) |
|---|---|---|
| batch(题/批) | 128 | groups_per_batch=100 |
| group_size(rollout/题) | 16 | 4 |
| LR | 4e-5(LoRA rank 32) | 1e-5 |
| loss | importance_sampling | importance_sampling(可切 ppo/cispo) |
| KL | 无 | kl_penalty_coef=0.0(默认关) |
| temperature | (默认) | 1.0,注释:"T=1 near-optimal;改温度不推荐" |
| lora_rank | 32 | 32 |
| 优化器 | Adam β=(0.9,0.95), eps 1e-8 | 同 |
| 其它 | — | num_substeps=1;max_steps_off_policy 可设(异步 off-policy 上限) |

- `rl/train.py` 的 `Config` 默认:`kl_penalty_coef=0.0`、`loss_fn="importance_sampling"`、`lora_rank=32`;KL>0 时需配 `kl_reference_config`(即**官方支持挂参考策略 KL 锚**,我们反遗忘设计可直接用)。
- `hyperparam_utils.py`:`get_lora_lr_over_full_finetune_lr()` **固定返回 10.0**(注释直接引 LoRA Without Regret);`get_lr("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16")` **抛 NotImplementedError("not yet calibrated, specify manually")** → 官方没给我们模型的标定 LR,必须自己定(建议从 SFT 已验稳的 LR 向下取,见 Q6)。
- Tinker 官方 loss 指南:"**Start with importance_sampling for RL, and switch to ppo or cispo if you see training instability from large policy updates**";tinker 原生支持 `importance_sampling/ppo/cispo`(本地 `tinker/types/loss_fn_type.py` 三者都在;huikang 的 `loss_config.py` 也已封装三者)。出处:[Tinker docs: Loss Functions](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/loss-functions/)、[CISPO 页](https://tinker-docs.thinkingmachines.ai/tinker/losses/cispo/)

**结论 Q2**:rank≤32 完全够;真正要拍板的只有 LR(无官方标定)与 KL/replay 反遗忘设计。MoE 模型上 LoRA-RL 的公开先例少(MiniMax/雀类都是 full-FT),这是我们要用小步 dry-run 自己验的残余风险。

---

## Q3 · GRPO 小数据稳定性:各修正解决什么?greedy 评测下哪些最相关?

### 3.1 病灶 → 修正对照表

| 病灶 | 表现 | 修正 | 出处 |
|---|---|---|---|
| **长度偏置**(GRPO 的 1/\|o\| 归一) | 错误回答越来越长、"长而错"被纵容 | **Dr.GRPO**:去掉响应长度归一与组内 std 归一(回到无偏 policy gradient) | [arXiv:2503.20783](https://arxiv.org/abs/2503.20783) |
| **难度偏置**(std 归一) | 全对/全错附近的题方差小 → 被 std 除法放大权重 | Dr.GRPO 同上 | 同上 |
| **熵塌缩** | 训练早期熵速降、探索死、上限封死(R=-a·e^H+b,可预测) | **DAPO clip-higher**(ε_high=0.28/0.4 解耦,让低概率探索 token 涨得动);**Clip-Cov/KL-Cov**(对高协方差 token 施约束);温度≥1 | DAPO [arXiv:2503.14476](https://arxiv.org/abs/2503.14476);熵机制 [arXiv:2505.22617](https://arxiv.org/abs/2505.22617)([verl recipe](https://verl.readthedocs.io/en/latest/algo/entropy.html)) |
| **零方差组浪费/噪声**(小数据致命) | 全对或全错的组 advantage=0,白花采样费还稀释梯度 | **DAPO dynamic sampling**:过滤 acc∈{0,1} 的题,只训有梯度的组 | DAPO 同上 |
| **clip 杀掉关键低频 token**("However/Wait/Recheck") | PPO/GRPO 的 token 级 clip 把反思 token 的梯度整段丢掉 | **CISPO**:clip 重要性权重而非 clip 目标,**所有 token 保留梯度**;MiniMax 实测同性能只要 DAPO 50% 步数 | MiniMax-M1 [arXiv:2506.13585](https://arxiv.org/abs/2506.13585) |
| **截断奖励噪声** | 超长被截断的回答记 0 分,惩罚了"还没写完"的好推理 | DAPO overlong reward shaping / mask 截断样本 | DAPO 同上 |
| **长程漂移/灾难遗忘** | 几百步后崩盘或忘旧 | **ProRL:KL 罚 + 定期 reset 参考策略**(KL 起飞或性能停滞时把 ref 换成当前快照);lr 2e-6、batch 256、n=16、T=1.2、ε=(0.2,0.4) | ProRL [arXiv:2505.24864](https://arxiv.org/html/2505.24864v1)(超参见原文;β 正文未给,社区复现常取 1e-3 量级,[解读](https://ritvik19.medium.com/papers-explained-386-prorl-261c9ac00bc7)) |
| **reward hacking** | 模型钻 reward 缝(格式分、长度分、随机分也能涨) | 用与 LB 完全同构的 verifier(string-exact OR 1e-2 容差);别加手写"质量分" | Spurious Rewards [arXiv:2506.10947](https://arxiv.org/abs/2506.10947) |

KL 锚的两派:DAPO 与 Dr.GRPO 都**砍掉 KL(β=0)**,理由是推理 RL 本来就要分布漂移、规则 verifier 不怕漂移;ProRL 反向**保 KL+reset** 以撑数千步。综述对比见 [Medium: GRPO/DAPO/Dr.GRPO 演化](https://medium.com/@jenwei0312/the-evolution-of-policy-optimization-understanding-grpo-dapo-and-dr-3e758c54b2c6)。

### 3.2 greedy 评测下的优先级(我们的特殊性)

1. **dynamic sampling(最相关)**:我们题池小(~7.6k train,RL 目标池 1-2k)且 pass 率两极分化(锚题全对、死区全错)——不过滤零方差组等于把大半采样费扔水里。**它同时是省钱手段**。
2. **去长度偏置(Dr.GRPO 项)**:max_tokens=7680 截断是我们已知丢分项,绝不能让 RL 再学会写长。token-level loss + 不按响应长度归一。
3. **CISPO(tinker 原生)**:作为 instability 时的一键替代,优先级高于自己调 clip(官方推荐路径:importance_sampling→不稳→cispo)。
4. **熵塌缩(降级关注)**:我们交付 greedy argmax,**分布变尖本身就是目标**(pass@k→pass@1 压缩);只需保证训练期熵别在前 100 步就死(T=1、clip-higher 兜底),不必上 Clip-Cov 这类重武器。
5. **KL 锚(用途换位)**:文献砍 KL 是为了让能力漂移;**我们留 KL/replay 不是为 RL 稳定、是为锚题反遗忘**(60 步 LR 8e-5 把锚题打到 80% 的 A 级教训)。形式可二选一:`kl_penalty_coef>0` 挂 run-011 参考(cookbook 原生支持),或 batch 内混 replay/SFT 损失。
6. **reward 设计**:只用比赛同构 verifier(exact OR 1e-2)+ 可选 boxed-format 小分;**拒绝**公开 notebook 里的"reasoning_quality_reward"(正则打分,纯 hacking 靶子)。

---

## Q4 · DPO/KTO/SimPO 用"自采样正确 vs 贪心错误"构造 pair:先例与坑

### 4.1 正面先例

- **Iterative RPO**(Pang et al., NeurIPS'24):与我们设想**完全同构**——每轮对训练题采样多条 CoT,赢家=答案正确、输家=错误,组 pair 训 **DPO+NLL(chosen 上加 NLL 项)**,迭代 3-4 轮重新采样。Llama-2-70B GSM8K 55.6→81.6、MATH 12.5→20.8、ARC 77.8→86.7。**关键发现:不加 NLL 项时 chosen 的 logprob 随训练下降;加了才上升**。[arXiv:2404.19733](https://arxiv.org/abs/2404.19733)
- **KTO**:只需逐条二元标签(可取/不可取),**天然适配"贪心错误"这种无配对负例**;不需要同题配对。[arXiv:2402.01306](https://arxiv.org/abs/2402.01306)

### 4.2 坑(按对我们的杀伤力排序)

1. **Likelihood displacement(头号坑)**:chosen/rejected **编辑距离小**时(同题的对/错 trace 正是如此),DPO 会把两者概率一起压低、只保住"相对差"——数学/推理域实测掉点,被定性为"catastrophic"。[Unintentional Unalignment, ICLR'25, arXiv:2410.08847](https://arxiv.org/pdf/2410.08847);修法 **DPOP/Smaug**(罚 chosen 掉到 ref 之下)[arXiv:2402.13228](https://arxiv.org/abs/2402.13228)、或 RPO 的 NLL 项(同上)。
2. **greedy 错位**:DPO 族优化的是 margin(相对排序),**不直接抬 argmax**;贪心评测要的是 chosen 成为众数。真正抬众数的是 NLL/SFT 分量 → 推论:**该族对我们的价值排序 = RFT(只用自采样正确样本做 SFT+replay)≥ DPO+NLL > 纯 DPO/SimPO**。我们的"营救臂 splice 17%"本质已是 RFT 数据管线。
3. **长度偏置**:DPO 隐式 reward 与长度正相关 → 越训越长(撞 7680 截断);SimPO 用长度归一 reward + margin γ 专治此病,且与"平均 logprob ≈ 生成时打分"的口径一致。[SimPO arXiv:2405.14734](https://arxiv.org/html/2405.14734v1)
4. **off-policy 漂移**:pair 是 run-011 采的,训几步后政策已变,pair 变 off-policy → 一轮别训太久,**迭代重采**(Iterative RPO 每轮重新生成)或退回 on-policy RL。
5. **SimPO 无参考模型** → 没有 KL 锚兜底,反遗忘风险比 DPO 更高,与我们纪律冲突,不推荐主用。

---

## Q5 · 本比赛公开情报:有没有人用 RL?>0.86 路线披露?

### 5.1 LB 实况(2026-06-11 kaggle CLI 拉取,4170 队)

```
0.89 ×1(NullSira) | 0.88 ×3 | 0.87 ×30 | 0.86 ×1495 | 0.85 ×510 | 0.84 ×250
```
- 0.86 是"公开共享 adapter 大包"(1495 队);**0.87 只有 30 队,0.865+ 即进入真实差异区**。队名彩蛋:0.87 区有一队叫 "**Lora is all you need**"。
- 讨论区流传说法:奖牌线约 0.877(搜索摘要,未能直读原帖)。

### 5.2 公开披露的方法(全部 SFT,无 RL 成功案例)

- **huikang(Progress Prize 得主,LB 0.85 公开)**:程序化 CoT + min-logprob 监控的 SFT;[写法 writeup](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge/discussion/689915)、[GitHub tonghuikang/nemotron](https://github.com/tonghuikang/nemotron)、[E2E notebook "End-to-end finetuning for LB 0.85"](https://www.kaggle.com/code/huikang/end-to-end-finetuning-for-lb-0-85)。本地镜像 `external/nemotron-huikang/` grep 证实:**repo 无 GRPO/DPO 训练管线**(`loss_config.py` 里封装了 importance_sampling/ppo/cispo 三个 tinker loss,说明他搭过 RL 插头,但语料/训练全是 SFT)。
- **0.86 大包成因**(拉了 5 个高票 notebook 源码验证):共享 adapter 热启 + 微调,如 [NemotronCOMP best0.86+ (under 5min)](https://www.kaggle.com/code/debatreyabiswas/nemotroncomp-best0-86-solution-nvidia-under-5min)(直接装 kienngx 训好的 adapter)、[Nemotron+Replay_Data 0.86](https://www.kaggle.com/code/mohamedamr992/nemotron-replay-data-0-86)(huikang 语料快照 + 1GB 数学 replay 续训)。
- **公开"0.87 攻略"**:[💗AGI FOR MEDAL, 0.87 IS POSSIBLE?💨](https://www.kaggle.com/code/johnjanson/agi-for-medal-0-87-is-possible)(118 票,源码已拉):huikang adapter v20 **热启** + train.csv **failure mining** 分桶 + **12,000 条合成题**(桶定向)+ 课程 SFT(LR 2e-4、240 步、max_len 6144)。仍是 SFT+合成,无 RL。
- **公开 GRPO notebook 均为玩具规模、无 LB 增益声明**(源码已拉验证):
  - [NVIDIA Nemotron - SFT → GRPO - Colab faster](https://www.kaggle.com/code/johnnyhyland/nvidia-nemotron-sft-grpo-colab-faster)(153 票):TRL GRPOTrainer,**220 题 × 4 rollouts、GRPO_LR=5e-6、T=0.7、beta=0(无 KL)、max_grad_norm=0.1**,reward=cosine 正确分+boxed 格式分+长度罚;
  - [nemotron-ultimate-sft-grpo-v3](https://www.kaggle.com/code/amanatar/nemotron-ultimate-sft-grpo-v3):同款再加正则"reasoning_quality_reward"(典型可 hack 奖励)。
- **结论**:0.87~0.89 的 33 队没有任何人公开过超出"热启+合成+课程 SFT"的配方;**RL 路线在本比赛公开域是空白**。顶部 0.89 的存在证明 ≥0.89 可达(对照我们 oracle×R≈0.90×0.97≈0.873 的账,说明头部在 oracle 上比我们多挖了东西,大概率是 crypt/equation 的合成或更强 CoT)。

---

## Q6 · 4 天窗口内 RLVR 一轮迭代的最小可行配置(业界口径 → 我们的换算)

### 6.1 业界参考配置

| 来源 | rollouts/题 | 题/批 | LR | 步数口径 | 备注 |
|---|---|---|---|---|---|
| tinker cookbook `rl_loop.py` | 16 | 128 | 4e-5(LoRA r32) | — | 官方最小 GRPO-style 循环 |
| tinker cookbook `math_rl` | 4 | 100 | 1e-5 | — | KL=0,importance_sampling |
| [unsloth GRPO 指南](https://unsloth.ai/docs/get-started/reinforcement-learning-rl-guide) | 6~16(小数据建议拉到 8~16) | 1×ga4 | 5e-6 | **≥300 步才见 reward 动;前 100~150 步常无信号** | TRL/全模型口径 |
| ProRL(NVIDIA) | 16 | 256(mini 64) | 2e-6 | 2000+ 步 | T=1.2、ε=(0.2,0.4)、KL+ref reset;多周级,4 天学不动 |
| Tülu 3 RLVR | 中等 | — | — | 终段单独一轮 | 收敛管线上 +1~3pp 的实证锚点 |

### 6.2 对我们的最小可行落地(设计稿,待红队)

- **起点**:run-011(0.85)weights+optimizer 在 Tinker 直接续;rank 32 不变(Q2 已证容量无虞)。
- **题池**:R-loss + marginal(死区剔除,Q1 论证)≈1-2k 题;**dynamic sampling:零方差组(全对/全错)当步丢弃**,既去噪又把有效采样费集中在可学题上。
- **核心参数(预算版)**:P=64 题/批 × G=8 rollouts(温度 1.0,max_tokens 压到 4096 控费,truncation 不给负分只 mask);loss=importance_sampling,出现 ratio 爆炸/熵跳水即切 **cispo**;LR 两点小扫 {5e-6, 2e-5}(上界 = SFT 已验安全区的零头;官方对本模型无标定,Q2);`kl_penalty_coef` 小值挂 run-011 参考 **或** 每批混 10-20% replay(反遗忘,二选一,dry-run 决定)。
- **账(按 BRIEF 实测单价)**:采样 512 rollouts/迭代 ≈ $1.28 + 训练 1-2 步 ≈ $0.08-0.15 → **~$1.4/迭代**;150~300 迭代 = **$210~430** + holdout 评测若干刀,4 天预算内。
- **时刻表**:D1 dry-run 10-20 迭代(闸门:format 率不崩、锚题 greedy 不掉、reward 方差>0 的组占比);D2-3 主跑 150~300 迭代,每 20 迭代 greedy holdout 闸门;D4 选点提交(每天 5 配额)。
- **预登记证伪线(吸收 unsloth 经验)**:300 迭代内 holdout greedy 无 +0.3pp 趋势 → 停;锚题回撤 >1pp → 立即回滚加重 replay。

---

## 附:本报告的证据缺口(诚实声明)

1. Kaggle 讨论区正文是 JS 渲染,WebFetch 读不到帖子内文;0.87+ 选手"无 RL 披露"的判断基于:全部高票 notebook 源码实拉 + 搜索摘要 + huikang repo 本地 grep,**不能排除有未公开的 RL 私货**。
2. ProRL 的 KL 系数 β 原文未给数值(已 fetch 原文核实),"1e-3 量级"来自第三方解读,引用时已标注。
3. LoRA-RL 在 **MoE** 大模型上的公开复现稀少(TML 实验以 dense + MoE 兼有,但非 Nemotron);rank32-LoRA-RL 在我们模型上的实际表现需 D1 dry-run 自证。
4. "RL 修复'背过但贪心不会执行'"没有逐字同构的论文;最接近的是 sharpening/压缩机制链(Q1.2)+ Chu et al. 的 OOD 执行证据,推断等级 B。
