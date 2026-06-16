# 提案 01 · GRPO/RLVR:用结果奖励把"采得出"压成"贪心对"

**提案人:GRPO/RLVR 专家 | 日期 2026-06-11 | 状态:设计稿,零花费,待红队/用户解锁**
**一句话:从 run-011(0.85)起步,在 train_split 的 R-loss/marginal 猎物池上跑 on-policy GRPO(组内中心化 advantage、动态丢零方差组、锚题内置巡逻),把模型自己采样能对(pass@8≈17-20%)但贪心执行错的题压进 argmax;预登记中心预测 +1.0pp(0.86),触 0.865 概率 ~25%,总成本中心 $280 / 上限 $330,4 天排得下。**

---

## 0. 预登记卡(先亮底牌)

| 项 | 值 |
|---|---|
| 初始 checkpoint | run-011 final(`tinker://cad6ab5c-…:train:0/weights/state_ep1`,执行第 0 步 list_checkpoints 确认) |
| 中心预测 ΔLB | **+1.0pp(0.85→0.86)**;区间 +0.3 ~ +1.6pp |
| P(LB≥0.865) | **~0.25**(单路;若与他路叠加另算) |
| 成本 | 中心 **$280**(两腿全跑),lean 路径 $180,**硬上限 $330** |
| 止损线 | 三档:Phase-0 失败 ≤$30 / step-30 闸门失败 ≤$100 / 首提 LB 无增益 ≤$200 |
| 证伪线 | ①Phase-0:猎物池 G=16 下混合奖励组占比 <5% → 弹药不存在,弃。②Leg-1 step-60:holdout 猎物贪心营救 <+3 题 → 停。③首次 RL 提交 LB ≤0.85 → 不开 Leg-2 |
| 反遗忘 | 五件套:锚题内置(梯度绊线)+ 每 10 步锚探针熔断 + LR 2e-5 恒定 + 每 10 步 checkpoint 可回滚 + 备用 KL 子采样锚(详见 §5) |

---

## 1. 核心论点:SFT 模仿买不到的"过程泛化",结果奖励为什么买得到

### 1.1 先把死因说清楚:SFT 失败的不是"教推理",是"教别人的分布"

SFT 路线的三连尸检(全 A/B 级实证):
- 模型连**背过的** crypt 演绎 trace 都只能贪心复现 3%(PATHS.md 探针死刑);
- 60 步小补丁 FT 教新格式:留出 crypt 5%、bitEV 0%,还把锚题打到 80%(EXPERIMENTS E4);
- CoT 格式重设计两个相反方向都 0.82 < 0.84(run-009/010)。

共同病灶:**SFT 的梯度方向是"提高外生 trace(reasoner 写的 CoT)的逐 token 似然"**。这条 trace 不在模型自己的自回归舒适区里——token-level 模仿误差在 7680 步长程生成里复利放大,T=0 一步走岔全盘崩。这就是 R-loss 的本质:答案知识在,**执行分布不在**。

### 1.2 GRPO 在结构上避开了这三个坑

| | SFT 模仿 | on-policy GRPO |
|---|---|---|
| 训练数据来源 | 外生 reasoner trace | **模型自己刚生成的 rollout**——被强化的轨迹定义上是模型能一步步走完的(它刚走过) |
| 优化目标 | 逐 token 似然(代理) | **整条轨迹末端 boxed 答案对错**——与 LB 判分函数逐字同构(复用 `src/reasoning.py:38/46`) |
| 梯度方向 | 把模型拽向别人的分布 | 在模型自有支撑集内做**质量再分配**:正 advantage 抬自己的成功模式,负 advantage 压自己的失败模式 |

### 1.3 greedy 评测下 RL 提 pass@1 的机制:锐化到"自己能执行的解法"

greedy = 逐 token argmax。一道 R-loss 题贪心错,意味着在若干分叉 token 处,失败模式的概率 > 成功模式。policy gradient 用正/负 advantage 在这些分叉处做概率质量转移;**当成功模式概率在每个分叉处超过竞争失败模式,argmax 整条翻转**。三条实证说明这个翻转门槛对我们的猎物不高:

1. **"会就稳"结构**:挖矿命中的 eq 题组内 4-5/8 全对(`/tmp/mine_results.jsonl`)——正确模式一旦在支撑集内,质量是**集中的**而非弥散的,小幅转移即可过 argmax 门槛;
2. **bit R-loss 全是 1-2 bit 近失**(`/tmp/oracle_vs_model.json`,11/110 实测):失败是轨迹末段的薄薄一层失误,正负 advantage 的差分恰好打在分歧 token 上;
3. **crypt R-loss 是"刹不住车"**:run-012 holdout 解码 107 题 0 对、102 撞 7680 cap;而 T=0.9/2000-token 采样下 17% 能出短而对的解(`/tmp/rescue_test.log`)。截断 rollout 在我们的 reward 下=0 分、在混合组里吃负 advantage → RL 同时学"做对"与"在预算内收口"——**这一项 SFT 根本表达不了,是 RL 独有的杠杆**。

熵塌缩在文献里是病,在我们这里是药:**我们交付的就是 argmax,分布变尖正是目标**(recon_intel Q3.2)。

### 1.4 诚实边界(不许自己骗自己)

- **RL 翻不动支撑集外**(Limit of RLVR, arXiv:2504.13837;与我们死区 pass@8≈1-2% 实测互证)。本提案的收益账**只按 pass@k>0 的猎物记**,死区一题不进池。
- Spurious Rewards 警告(arXiv:2506.10947):增益归因不能只看 LB 总分。我们不烧钱做 format-only 对照臂,改用更强的归因:**预登记 64 题 holdout 猎物名单,逐题跟踪翻转**——涨分必须落在预登记猎物上,否则按"未归因涨分"处理,不外推。
- LoRA rank32 容量对 RL 绰绰有余(TML:rank-1 即打平 full-FT;每 episode ~1 bit),容量不是风险;**MoE 30B 上 LoRA-RL 无公开先例**才是,Phase-0 dry-run 自证 + 盯免费 MoE 路由 metrics。

---

## 2. 数字账(a):+X 题从哪来

### 2.1 猎物池(holdout 口径,= LB 镜像;每题 0.0527pp)

| 块 | 题数 | 证据 | pass@8(采样可达性) |
|---|---|---|---|
| bit R-loss(oracle 对/贪心错,1-2 bit 近失) | **≈32/320**(11/110 实测外推,`data/run012_holdout_decode.jsonl`) | +1.69pp | 未直测;近失结构 + bit_tail 6%@K4 → 估 15-35%,中心 ~20% |
| crypt solver-ok(R-loss 全集,贪心 0 对、95% 撞 cap) | **26/165**(id 清单在 recon_data §3.3,truth 均可 box) | +1.37pp | **17% 直测**(2/12 @K8/2000tok,CI 2-48%) |
| eq 签名/marginal | **≈6**(R≈0.95 假设;今晚解码完成后钉死) | +0.3pp | 签名类(62 vs -62)估不低,未直测 |
| **合计** | **≈64 题** | **+3.4pp 理论上限** | |

### 2.2 转化链(pass@k → greedy 的假设,每环给依据)

```
60 步 Leg-1 = 猎物池 ~6 个 pass × G16 = 累计 ~96 rollouts/题
①累计覆盖率(训练期内至少采出过 1 条对):30-45%
   依据:pass@8≈17-20% 起步,policy 改进使命中率逐 pass 上升(复利);per-题命中
   分布"会就稳"(4-5/8)的题第一 pass 即覆盖
②覆盖→贪心翻转转化:60-80%
   依据:GRPO 正 advantage 直接抬已采出模式;"会就稳"题质量集中,锐化即翻;
   1/8 命中题难翻(留给 Leg-2 / RFT 兜底)
③train 营救率 = ①×② ≈ 20-35%
④train→holdout 迁移折扣:×0.5-0.8
   依据:修的是行为模式(近失失误/收口),非题目知识,理应迁移;但 run-012
   "holdout 外推 0.86-0.87 vs LB 0.84"的 2-3pp 未归因矛盾要求打折并用 LB 实测校准
⑤holdout 营救 = 64 × 12-25% ≈ 8-16 题 ≈ +0.4-0.85pp(Leg-1)
⑥Leg-2(再 50 步,扩池/加 K/RFT 混合)≈ +0.2-0.5pp
→ 中心 +1.0pp,区间 +0.3~+1.6pp;吃满 0.865 需 ~28 题=猎物池的 44%,
   故 P(≥0.865)≈0.25 —— 不吹"必达",这条路的角色是把 0.85→0.86 坐实并买一张 0.865 彩票
```

### 2.3 训练池(全部取 train_split,holdout 一题不进训练)

| 层 | 内容 | 题数 | 角色 |
|---|---|---|---|
| 核心猎物 | bit train R-loss(Phase-0 贪心筛 hard-tail ~400 → 取贪心错的 100-150)+ crypt solver-ok 116-146(**剔 `}`-truth:boxed 提取下永久零 reward,纯烧钱**)+ eq 签名类 30-50 | **~280-350** | G=16,主梯度来源 |
| 巩固 | bit/eq 已对题分层抽样 | ~300 | G=8;多数组全对被丢(只花采样费);**一旦 policy 开始破坏它们,组变混合 → 自动产生纠正梯度** |
| 锚 | cipher/gravity/numeral/unit 已对题各 ~75 | ~300 | G=8;同上,梯度绊线 + 漂移金丝雀 |

reward 函数:`metric_correct(truth, extract_answer(decoded))`,0/1,truth 来自 train.csv(完美离线 verifier,与 grader 同构,无 judge、无格式分、无长度分——拒绝一切可 hack 的附加奖励项)。截断 rollout 记 0 分**不 mask**:DAPO 的 overlong-shaping 是因为它的评测允许更长,我们的 grader cap 就是 7680,在 cap 内收不了口就是真错,罚之与 LB 目标严格对齐。

---

## 3. 训练设计:K/温度/advantage/loss/LR

| 参数 | 取值 | 依据 |
|---|---|---|
| 批结构 | 每步 64 题 = 32 猎物×G16 + 16 巩固×G8 + 16 锚×G8 = **768 rollouts/步** | 猎物 pass@8≈17-20% → G=8 只有 ~1/5 组有信号;G=16 把有信号组占比近似翻倍(recon_data §4.4 建议 K=16-32) |
| 温度 / top_p | **1.0 / 1.0** | cookbook math_rl 注释"T=1 near-optimal";挖矿 0.9 先例兼容 |
| max_tokens | **7680 全类统一** | 与 grader 同构;计费按实际生成 token,短答不浪费;crypt 撞 cap 的 rollout 吃 0 分负 advantage = 收口压力(§1.3) |
| advantage | **A = r − mean(组)**,不除 std,不按长度归一 | cookbook `rl/data_processing.py:23` 原样;Dr.GRPO 去长度/难度偏置——绝不能让 RL 学会写长(7680 截断是已知丢分项) |
| 零方差组 | **当步丢弃,不进 forward_backward** | rl_loop.py 原生模式 = DAPO dynamic sampling;省训练费(锚/巩固组大多在此免费蒸发) |
| loss | **importance_sampling 起步;ratio 爆/熵跳水即切 cispo(clip 0.0/4.0 默认)** | tinker 官方推荐路径;每步刷新采样权重 → 近 on-policy,ratio≈1,IS 足够;**不用 huikang PPOLossConfig(0.2/0.2 是 ratio-界约定 bug,recon_tinker §2.2)** |
| LR | **2e-5 恒定**,grad_clip_norm=1.0;熔断后预定降档 7e-6 | 官方对 Nemotron 无标定(NotImplementedError);处于 recon 推荐带 1e-5~4e-5 内、是伤锚的 8e-5 的 1/4、是 cookbook 8B-rank32 4e-5 的一半。不开双臂 LR 扫(预算/时窗不够),用"单臂+预定降档+回滚"替代 |
| 优化器 | Adam β=(0.9,0.95) eps=1e-8,**fresh Adam**(`create_training_client_from_state`,不带 optimizer) | RL 梯度统计与 SFT 末期 Adam 矩无关;rank32/全模块栈自动从 weights_info 继承,提交约束不会漂 |
| 权重刷新 | 每步 `save_weights_and_get_sampling_client`(~$0.05/步) | off-policy 漂移最小化;loss 的 logprobs 永远填采样那一刻的值 |
| epoch | 同批 rollouts 只训 1 个 epoch(E=1) | 多 epoch 重训同批 = off-policy 化,得不偿失 |

**何时切 greedy 验证(三级)**:
1. **每 10 步**:40 题锚探针贪心(4 健康类各 10,固定名单)≈ $0.10 —— 熔断输入;
2. **每 30 步**:holdout 猎物集贪心(64 题:11 bit 已知 + 26 crypt + eq 名单今晚定 + 预留)+ 100 题分层 ≈ $0.45 —— 营救曲线 = 主验证信号;
3. **提交前一次**:全 holdout 1899 贪心 ≈ $4.75 —— 出完整 R 报告,与 LB 互证(注意 run-011 对 holdout 有训练污染,猎物营救指标受污染最小——这些题污染了也没做对;最终仲裁是 LB)。

---

## 4. 成本表(b):分项、上限、止损

单价锚(recon_tinker §4,A 级):采样 $0.0025/rollout(均 5.5k 生成 tok;crypt 撞 cap ~$0.0037),训练 $0.218/M token,RL 步更新远贵于 SFT 步——**预算守卫 usd_per_step 按 $2.4 重标,不沿用 0.04/0.077**。

| 项 | 计算 | 金额 |
|---|---|---|
| Phase-0:管线试点 10 步(P=32×G8)+ pass@16 测量(80 猎物)+ run-011 猎物基线贪心 + train 猎物筛选贪心(~450 题) | $13 + $3.2 + $0.3 + $1.2 + 杂 | **~$20(上限 $30)** |
| Leg-1:60 步 × [采样 $2.1(768 rollouts 混合长度)+ 更新 $0.27(f_keep≈0.3,~175 datum)+ 刷新 $0.05] | 60 × $2.4 | **~$145** |
| 验证闸门:6×锚探针 + 2×猎物集 + 1×全 holdout | $0.6+$0.9+$4.75 | **~$7** |
| Leg-2(条件触发):50 步,扩池/加 K 或 RFT 混合 | 50 × $2.4 | **~$120** |
| 提交杂项:checkpoint 下载/上传、存档 | | **~$5** |
| **合计** | lean(不开 Leg-2)**$180** / 中心 **$280** / **硬上限 $330** | |

止损三档(预登记,触线即停、不许讨价还价):
- **$30**:Phase-0 闸门失败(f_signal<5% 或管线/MoE 异常)→ 全案废弃,转 RFT 或弃 RL;
- **$100**:Leg-1 step-30 中检猎物营救 0 题且 reward 曲线平 → 停腿,剩余预算还给其他路;
- **$200**:首次 RL 提交 LB ≤0.85 → 不开 Leg-2,已采 rollouts 里的正确 trace 免费转 RFT 语料兜底。

发车前查余额 ≥ 1.3×$330 ≈ $430(402 事故纪律)。

---

## 5. 反遗忘设计(e):正面回应"60 步 8e-5 把锚打到 80%"

先定性那次事故:**off-policy SFT 补丁**(外生新格式语料)@ **8e-5**,纯拉扯式遗忘。本案三点结构性不同 + 五件套硬防:

结构性不同:① on-policy——梯度只在模型自有分布邻域内动,遗忘压力天然小于 off-policy 拉扯;② LR 2e-5 = 事故 LR 的 1/4;③ 零方差丢弃使锚题在健康状态下**根本不产生梯度**(不像 SFT replay 持续消耗梯度预算)。

五件套(全部进训练循环,不是口头承诺):
1. **锚题内置 = 梯度绊线**:300 锚题常驻池。policy 健康时组全对→丢弃,零训练费;**一旦开始遗忘,锚组出现错样本→组内方差>0→立刻产生指向恢复的负 advantage 梯度**。这是 on-policy RL 独有的自愈机制:防遗忘梯度恰好在需要时自动开火。
2. **每 10 步锚探针熔断**:40 固定锚题贪心($0.10);**≥4 题(10%)回撤 → 自动 halt + 回滚上一 checkpoint + LR 降档 7e-6**;连续两次触发 → 加挂第 5 件。
3. **LR 2e-5 恒定 + grad_clip 1.0**:不退火不冲高。
4. **每 10 步 save_state(ttl 7 天)**:回滚粒度 10 步 ≈ $24 沉没成本上限;最坏情况丢弃整腿,提交底座 run-011 0.85 毫发无损——**本案 LB 下行风险被提交闸门封死在 0**(不达标就不提交)。
5. **备用 KL 锚(默认关)**:`incorporate_kl_penalty(data, run-011_sampler, coef≈0.05)` 只对 15% 子采样 datum 计算(全量 prefill ~$40/腿太贵,recon_tinker §4 公式);仅在熔断二次触发后开。
6. (明确不用)SFT-replay 混训:往 RL 批里掺 cross_entropy reasoner-trace = 把死掉的 off-policy 模仿压力请回来,拒绝。

监控免费送:每步 fwd_bwd metrics 记录 `e_frac_with_tokens:mean / e_max_violation:max`(MoE 路由崩塌前兆),趋势恶化 = 与锚熔断同级处置。

---

## 6. 时刻表(c):06-11 起逐日,含提交配额与出分延迟

| 日 | 事项 | 花费 | 闸门 |
|---|---|---|---|
| **06-11 晚(今天)** | 零花费准备:①`/tmp` 抢救清单 cp 进 repo(recon_data §6,重启即灭);②等 holdout 解码跑完,钉死 eq/gravity/numeral/unit 的 R 与 eq 猎物名单(**顺手解 2-3pp 未归因矛盾的第一优先情报**);③离线写好 `rl/grpo_loop.py` + rollout 落盘 schema + 猎物候选清单(bit hard-tail、crypt solver-ok 剔 `}`、eq 签名);④红队过案 | $0 | 用户解锁花钱 |
| **06-12(D1)** | 上午:第 0 步 `list_checkpoints` 确认 state_ep1(没有→降级 state_ep0 起跳或 $10 重放 epoch1);run-011 猎物基线贪心;train 猎物筛选;80 题 pass@16 测量。下午:**Phase-0 试点 10 步**(验 f_signal、f_keep、MoE metrics、**每步 wall-clock 实测**、账单核 prefill 计价)。晚:**闸门 A** 过 → 发车 Leg-1,通宵跑 | ~$20-90 | 闸门 A:f_signal≥10%(<5% 废案);锚探针无回撤;wall-clock ≤20 min/步(超 → 砍 P 或 G) |
| **06-13(D2)** | Leg-1 跑完 60 步(step-30 中检);晚:猎物集+锚+100 分层贪心;**过线(营救≥+6 题)→ build adapter、提交 #1**(留出出分延迟,5 配额用 1) | ~$150 累计 ~$240 内 | 闸门 B:step-60 营救 <+3 题 → 证伪停案 |
| **06-14(D3)** | 早:读 LB(校准 holdout→LB 迁移率)。LB>0.85 → **Leg-2** 50 步(选项按 D2 数据定:扩猎物池 / 猎物 G=24 / 掺自采正确 trace 的 RFT 混合腿);晚提交 #2/#3(Leg-1 final、Leg-2 mid)。LB≤0.85 → 停,转 RFT 兜底($10,用已采 rollouts 免费语料) | ~$120 | 闸门 C:LB≤0.85 不开 Leg-2 |
| **06-15(D4)** | 上午:Leg-2 final 全 holdout 贪心($4.75)→ 提交 #4/#5;下午:LB 择优定终稿。**当日 12:00 后不再发起任何新训练**(出分延迟缓冲) | ~$10 | 终选 = max(LB) |

wall-clock 是头号执行风险(768 rollouts/步,挖矿吞吐外推 5-20 min/步,band 宽):Phase-0 实测后,若 >20 min/步则 Leg-1 砍成 P=48 或 G=12,步数预算不变、总钱变少、覆盖率假设同步下修(预登记表同步改,不许事后挪标准)。

---

## 7. 与死亡名单的关系(f):逐条声明

| 死亡名单条目 | 本案关系 |
|---|---|
| crypt 死区采样捡漏(☠️A 级) | **不踩**。池内 crypt 全部是 solver-ok(26/165 holdout、116-146 train),即 22.5% 可恢复集内、模型 pass@8=17% 直测的题;solver-fail 死区(0.75%)一题不进。Phase-0 的 pass@16 测量对象是 R-loss 猎物,不是死区——种群不同,不是"再挖一次矿" |
| crypt synth 语料教 meta-skill(☠️) | **不踩**。零外生 trace 注入,只强化模型自采轨迹 |
| crypt closed-form >22.5%(☠️) | **不踩**。只榨 solver-ok 题的执行率,不碰 solver 上限 |
| bit 断言式格式 / CoT 格式重设计(☠️) | **不踩**。不改任何 CoT 格式,reward 只看 boxed 终答 |
| equation tiebreak / qop_unseen(☠️) | **不踩**。eq 只取签名类 marginal(模型采样可达),qop_unseen 量值死区不进池 |
| 死区 pass@8≈1-2% → RL 翻不动 | **本案立论基石而非违背**:Limit of RLVR + 我们实测双重证据,所以收益账只按支撑集内猎物记 |
| 60 步小补丁伤锚(⚰️教训) | 正面设计回应,§5 五件套 |
| "近似 SFT 模仿失败路"质疑 | **为什么这次不同**:数据源(自采 vs 外生)、目标(终答对错 vs token 似然)、梯度语义(支撑集内质量再分配 vs 分布拉扯)三者全换;这正是 BRIEF 开新路线的动机本身 |

---

## 8. 可执行伪代码(g):基于 tinker 0.22.3 实际 API

骨架 = cookbook `recipes/rl_loop.py`(本地源码逐行核过)+ 我们的 reward/落盘/熔断。新文件 `rl/grpo_loop.py`(单版本纪律,不开 v2/v3)。

```python
# rl/grpo_loop.py —— 伪代码级设计(关键 API 全部实名)
import tinker, numpy as np, torch
from tinker import types
from tinker.types.tensor_data import TensorData
from src.corpus import tokenize_prompt, BASE_MODEL
from src.reasoning import extract_answer, metric_correct
from src.train_tinker import _load_env   # env.json 鉴权

STATE = "tinker://cad6ab5c-...:train:0/weights/state_ep1"   # 第0步 list_checkpoints 确认
LR, CLIP, G_PREY, G_OTHER = 2e-5, 1.0, 16, 8
sc = tinker.ServiceClient()
tc = sc.create_training_client_from_state(STATE)            # weights-only, fresh Adam
adam = types.AdamParams(learning_rate=LR, beta1=0.9, beta2=0.95,
                        eps=1e-8, grad_clip_norm=CLIP)
anchor_ref = None   # 备用KL锚: 熔断二次触发后 = sc.create_sampling_client(model_path=run011_sampler)

for step in range(60):
    budget_guard(step, usd_per_step=2.4, cap=330)            # 重标!不沿用0.04
    sampler = tc.save_weights_and_get_sampling_client()      # 每步刷新, 近on-policy
    batch = draw_batch(prey=32, reinforce=16, anchor=16)     # 分层抽样, 题来自train_split
    futs = [(p, sampler.sample(                              # 全部future先发后收
                prompt=types.ModelInput.from_ints(tokenize_prompt(p.prompt, tok)),
                num_samples=(G_PREY if p.layer=="prey" else G_OTHER),
                sampling_params=tinker.SamplingParams(max_tokens=7680, temperature=1.0)))
            for p in batch]
    datums, journal = [], open(f"rollouts/step{step:03d}.jsonl", "a")
    for p, fut in futs:
        seqs = fut.result().sequences                        # SDK内置retry
        rewards = np.array([float(metric_correct(p.answer,
                      extract_answer(tok.decode(s.tokens)))) for s in seqs])
        for s, r in zip(seqs, rewards):                      # ★边采边落盘(402教训)
            journal.write(jsonl(p.id, s.tokens, s.logprobs, r, s.stop_reason))
        adv = rewards - rewards.mean()                       # 组内中心化, 不除std(Dr.GRPO)
        if np.all(adv == 0.0):
            continue                                         # 动态丢零方差组(省训练费)
        for s, a in zip(seqs, adv):
            ob = prompt_len(p) - 1                           # rl_loop.py:203-227 拼法
            datums.append(types.Datum(
                model_input=prompt_input(p).append(types.EncodedTextChunk(tokens=s.tokens[:-1])),
                loss_fn_inputs={                             # 白名单键, 无"mask"
                  "target_tokens": TensorData.from_torch(torch.tensor([0]*ob + s.tokens)),
                  "logprobs":   TensorData.from_torch(torch.tensor([0.]*ob + s.logprobs)),
                  "advantages": TensorData.from_torch(torch.tensor([0.]*ob + [a]*len(s.tokens))),
                }))
    if anchor_ref: incorporate_kl_subsample(datums, anchor_ref, coef=0.05, frac=0.15)
    if datums:
        fb = tc.forward_backward(datums, loss_fn=LOSS)       # "importance_sampling"
        op = tc.optim_step(adam)                             #  →不稳切"cispo"(0.0/4.0)
        m = fb.result().metrics                              # 流水线: 两future先后收
        watch_moe(m["e_frac_with_tokens:mean"], m["e_max_violation:max"])  # 免费MoE哨兵
        watch_ratio_entropy(fb, LOSS)                        # ratio爆/熵跳水→LOSS="cispo"
    if step % 10 == 0:
        tc.save_state(ttl_seconds=604800)                    # 回滚粒度10步
        if anchor_probe_greedy(n=40) < BASELINE - 4:         # ★熔断
            rollback_halve_lr(); maybe_enable_kl()
    if step % 30 == 29:
        prey_rescued = holdout_prey_greedy(64)               # 主验证信号(预登记名单)
# 异常纪律: 宽接tinker.TinkerError子类(SidecarDied/InternalServer/RequestFailed)→恢复;
# Auth/BadRequest/NotFound→fail-fast; 绝不把tinker.Timeout塞进except元组(TypeError坑);
# reward/advantage全numpy(心跳饿死教训); rollouts已落盘, session死亡采样钱不丢。
```

---

## 9. 残余风险与未决(诚实清单)

1. **2-3pp 未归因矛盾**(holdout 外推 0.86-0.87 vs run-012 LB 0.84):若根因是 holdout↛LB 分布差,迁移折扣 0.5-0.8 可能仍乐观 → 今晚解码完成 + D3 早的首个 LB 读数是两道校准闸,Leg-2 决策完全押在 LB 实测上。
2. **bit pass@8 无直测**(猎物最大块 32 题只有间接估计 15-35%)→ Phase-0 的 80 题 pass@16 测量($3.2)在花大钱前直接钉死,f_signal<5% 即废案,损失封顶 $30。
3. **MoE-LoRA-RL 无公开先例** → Phase-0 10 步 dry-run + 免费路由 metrics 哨兵。
4. **wall-clock 不确定 4 倍 band** → Phase-0 实测,超线砍 P/G(预登记降档规则)。
5. **run-011 holdout 污染** → 训练池零 holdout 题;主指标用"污染最小"的猎物翻转;最终仲裁 LB。
6. **unsloth"300 步才见信号"folklore** → 我们的设定(近失猎物 + G16 + LoRA 10×LR + 混合组浓度高)信号应早于通用情形;但仍预登记:60 步营救 <+3 题即认输,不追加"再跑 240 步就好了"的沉没成本。

---

## 10. 与兄弟提案的接口

- **RFT 是本案的内置兜底而非竞争者**:所有 rollouts 落盘,任何止损点触发后,已采正确 trace(+ `/tmp/mine_results.jsonl` 12 条 + `/tmp/crypt_traces.jsonl` 161 条)免费转 RFT 语料,$10 续训一腿。
- **DPO 腿不进本案**:likelihood displacement 坑(同题对错 trace 编辑距离小,arXiv:2410.08847)+ greedy 错位(margin 不抬 argmax),若做必须 DPO+NLL,优先级低于 GRPO/RFT。
- 若其他路(如 solver port +0.4pp)同期上车,本案预测与其叠加计账,提交配额按日协调。
