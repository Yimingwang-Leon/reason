# 提案 03 · 偏好优化(DPO 族)— 判决:不单独成路,定位为 RFT 的附加损失(RPO 式混合)

**作者:偏好优化专家 | 日期 2026-06-11 | 状态:待红队**
**一句话:纯 DPO/SimPO/KTO 在本赛制(greedy 评测 + 同题小编辑距离 pair)下证据不支持单独成路;本提案给出的是 RPO 式 "NLL 主 + DPO margin 辅" 的混合配方,与 RFT 路(02)共享同一批采样产物,边际成本 ~$6-9,独立跑全程 ~$22(上限 $40)。**

---

## 0. 先亮判决与依据(为什么不单独成路)

三条证据(全部出自 recon_intel Q4,已交叉验证):

1. **greedy 错位**:DPO 优化的是 chosen/rejected 的 margin(相对排序),不直接抬 argmax;比赛评测是 temp=0 贪心,要的是正确 trace 成为**众数**。Iterative RPO(arXiv:2404.19733)实测:不加 NLL 项时 chosen 的绝对 logprob 随训练**下降**——margin 在涨、众数在跌。真正抬 argmax 的是 NLL/SFT 分量。
2. **likelihood displacement(头号坑)**:我们的 pair 天然是"同题对/错 trace",编辑距离极小(bit R-loss 全是 1-2 bit 近失,前缀几乎全同),正是 arXiv:2410.08847 定性为 catastrophic 的场景——chosen/rejected 概率被一起压低,只保相对差。
3. **off-policy**:pair 采自 run-011,训几步后策略已漂,纯 DPO 无 on-policy 修正(不像 GRPO 有 ratio/clip)。

但 DPO margin 项有一个**纯 RFT 没有的机制价值**:RFT 只把正确 trace 推上去,不保证把**当前贪心错误 argmax** 压下来(贪心错解就是现任众数);DPO 的 rejected 梯度恰好定向压制这条现任错误路径。所以正确用法是 **RPO 式组合:L = L_DPO + α·NLL(chosen)**——NLL 抬正确众数,margin 压现任错误众数,两者打的是同一道"argmax 翻转"题的两面。

**SimPO:否决**——无参考模型 = 无 KL 锚兜底,与反遗忘纪律正面冲突(recon_intel Q4.2-5)。
**KTO:否决**——能吃无配对负例是优点,但同样不抬 argmax,且我们 pair 本来就配得齐,没必要换更弱的信号形式。

---

## a) 数字账:+X 题从哪来

### 猎物池(holdout 外推口径,recon_data §3.1)
| 来源 | 题数 | pp | 依据 |
|---|---|---|---|
| bit R-loss(oracle 对/贪心错,1-2 bit 近失) | ≈32/320 | +1.69 | 110 题解码实测 11 题外推 |
| crypt solver-ok(贪心 0%,102/107 撞 cap) | ≈26/165 | +1.37 | 解码已 107 题全错;26 个 solver-ok id 在 recon_data §3.3 |
| eq 签名类 marginal | ≈6 | +0.30 | R≈0.95 假设 |
| **合计(理论上限)** | **≈64** | **+3.4** | 吃 45% 即达 +1.5pp |

### 训练 pair 从哪来(全部 train_split,**holdout 一题不进训练**)
⚠️ **正面推翻任务书的一条建议**:`data/run012_holdout_decode.jsonl` 里的贪心错误解是 **holdout 题**,拿来构 pair = 污染唯一干净评测集 + 直接泄漏到 LB 镜像分布,**禁止**。它只用于(1)证明 rejected 形态(crypt=截断式枚举死循环、bit=近失)、(2)holdout 评测基线。训练 pair 全部重新从 train_split 采。

| 层 | 候选题 | 筛法 | 预期 pair 产出 |
|---|---|---|---|
| bit train R-loss | ~300 候选(oracle-ok 难尾)→ 贪心筛出 ~100-150 贪心错 | 贪心 1 遍 = rejected 顺手落盘 | pass@16≈30-50%(已训题,probe80 难尾贪心 85% 的补集)→ 40-60 题 ×≤2 pair = **80-120 对** |
| crypt solver-ok | 116-146 题,**剔 `}` truth**(boxed 提取永零分,recon_data §2)→ ~120 | 同上(贪心几乎全错,直接全采) | pass@16≈20-30%(营救臂 17%@K8 上推)→ 25-40 题 = **50-80 对** |
| eq 签名类 | 30-50 题(`62 vs -62` 型) | 同上 | ~12-20 题 = **15-30 对** |
| 免费种子 | /tmp/mine_results.jsonl 12 条正确 trace(run-011 采样,同分布) | rejected = 贪心解码补 12 条 | **+12 对** |
| **合计** | 筛查 ~500 题 → 采样 ~250 题 | | **150-240 对(中心 ~190)** |

**明确排除**:死区题(eq 1.8%/crypt 0.75% pass@8,PATHS A 级死刑,采了全是零产出);`/tmp/crypt_traces.jsonl` 161 条 solver 生成 trace(**非模型自身分布**,作 chosen 违反同分布原则且探针已证模型连背都背不动 3%——那是 crypt-synth 死路的素材,不碰)。

### pass@k → greedy 的转化率假设
机制依据 = Limit of RLVR(arXiv:2504.13837):RL/偏好优化做的是把支撑集内已有路径压进 pass@1;我们 pair 题全部满足"支撑集内有正确路径"(chosen 就是采出来的)。训练题本身不在 LB 上,**全部收益走类内迁移**:训练"在 R-loss 题上把 argmax 从错误路径翻到正确路径"的行为,迁移到 holdout/LB 同类 R-loss 题。
- 中心假设:holdout 64 题池翻转 15-25% → **+10~16 题 ≈ +0.5~0.85pp,中心 +0.6pp**(bit +8、crypt +3、eq +1)。
- 乐观:30% → +1.0pp;悲观:≤5%(displacement 抵消)→ +0.1pp,熔断兜底不为负。
- 其中 **DPO margin 相对纯 RFT 的边际**估 +0.2~0.3pp(机制:压制现任贪心错误路径;无直接文献量化,Iterative RPO 的 DPO+NLL > 纯 SFT 迭代差距约 1-3pp 是最近的锚)。
- ⚠️ 置信度折扣:run-012 holdout 外推 0.86-0.87 vs LB 实测 0.84 的 2-3pp 矛盾未归因(recon_data §3.1)——所有 holdout 记账都要打这个折扣,**LB 提交是唯一终审**。

---

## b) 成本表(单价全部引 recon_tinker §4 实测锚)

| 项 | 量 | 单价 | 小计 |
|---|---|---|---|
| ① 贪心筛查(=rejected 落盘) | ~500 题 ×1 | $0.0025/rollout | $1.3 |
| ② chosen 采样 | ~250 贪心错题 × K16 @T1.0(bit 7680 / crypt 4096 / eq 4096 cap) | $0.0025/rollout(crypt/eq 短 cap 打 8 折) | $9-10 |
| ③ ref logprob | ~190 对 ×2 序列 × ~6k tok ≈ 2.3M tok prefill | prefill 未实测,~$0.1-0.2/M 量级(recon_tinker §6.5) | $0.3-0.5(**预算顶 $2**) |
| ④ RPO 训练 | 190 对 ×2 序列 × ~5k tok ≈ 1.9M tok/ep × 2ep;custom = 1 fwd + 1 bwd ≈ 2× 训练价($0.218/M ×2) | | $1.7;+replay CE 混入 30% → **$2.5** |
| ⑤ 对照臂(RFT-only,同 chosen 数据纯 CE) | 同上单边 | | $1.5 |
| ⑥ 锚题探针熔断 | 40 题贪心 × 6 次 | $0.1/次 | $0.6 |
| ⑦ holdout 子集评测 | 385 题(每类 55)× 3 次 | ~$1/次 | $3 |
| ⑧ 存档/杂项 | save_state×3 + sampler 权重 ×2 | ~$0.05/次 | $0.3 |
| **独立全程合计** | | | **~$19-22,硬上限 $40** |
| **作为 RFT 附加(①②⑤⑥⑦ 与 02 号提案共享)** | 只付 ③④+增量评测 | | **~$6-9** |

**止损线**:累计花费 $40 或任一证伪线触发即停;第二轮迭代重采(+$12)只有在第一轮 LB 验证 ≥+0.3pp 后才解锁。

---

## c) 时刻表(06-11 → 06-15,提交配额 5/天,出分延迟按 2-4h 计)

| 日 | 内容 | 花费 |
|---|---|---|
| **06-11 今晚(D0,免费)** | 写完三件脚本:`rl/pair_builder.py`(筛查+采样+构 pair,边采边落盘 jsonl)、`rl/train_rpo.py`(下文伪代码)、`rl/anchor_probe.py`(40 题熔断);把 /tmp 抢救清单 cp 进 repo;等解冻批准 | $0 |
| **06-12(D1)** | 上午:①筛查 + ②K16 采样(挖矿先例 2000 rollouts 小时级,~4500 rollouts 预计 2-4h);下午:构 pair + ③ref logprob + ④训练(<20 optim 步,1h 内)+ ⑤对照臂;傍晚:锚题探针 + holdout 子集;过闸 → **当晚提交 RPO 臂 + RFT-only 臂两个 LB**(用 2/5 配额) | ~$17 |
| **06-13(D2)** | 早:读 LB。RPO ≥ RFT 且 ≥+0.3pp → 第二轮迭代重采(Iterative RPO 一轮,重采已翻转题的邻域 + 未翻转题,$12);RPO < RFT → **本路终止**,余预算与 chosen 数据移交 02/01 号提案 | $0-12 |
| **06-14(D3)** | 第二轮训练+评测+提交;与其它路线(GRPO/RFT)合流选最佳底座 | ~$4 |
| **06-15(D4)** | 只留提交保底窗口,不排任何新训练(出分延迟风险日) | $0 |

核心一轮 **2 天内闭环**(D1 训完、D2 早 LB 终审);全程含迭代 3 天,留 1 天裕量。

---

## d) 预登记预测 + 证伪线

| # | 预登记预测 | 证伪线(≤Y 即停/降级) |
|---|---|---|
| P1 | 筛查后贪心错猎物 ≥150 题;pair-able(≥1 条正确采样)≥50 题 | **<30 题** → pair 太薄,DPO 不开训,数据移交 RFT,本路花费止于 ~$12 |
| P2 | 训练全程 chosen completion 平均 logprob(policy−ref)≥0 且不降 | **连续 3 个 step 为负**(displacement 显形)→ 关 DPO 项,只留 NLL(退化为 RFT),不另花钱 |
| P3 | 锚题探针 ≥38/40(95%) | **<36/40** → 立即回滚上一 save_state,LR 减半重跑一次;再触 → 本路废弃 |
| P4 | holdout 子集相对 run-011 基线 ≥+2 题;**LB ≥0.853(+0.3pp)** | **LB ≤0.850** → 停止本路一切花钱 |
| P5 | RPO 臂 > RFT-only 对照臂(归因 DPO margin;Spurious Rewards 警告 arXiv:2506.10947 要求对照) | RPO ≤ RFT → DPO margin 无边际价值,后续只跑 RFT |

---

## e) 反遗忘设计(正面回应"60 步 @8e-5 把锚打到 80%")

那次事故的三要素:**off-policy 新格式语料 × LR 8e-5 × 60 步**。本提案逐项对冲,且叠 5 层:

1. **LR 5e-6**(= 事故 LR 的 1/16,SFT 峰值 2e-4 的 1/40;unsloth/cookbook RL 区间下沿),grad_clip_norm=1.0(不再用 1e9)。
2. **总步数 <20 optim 步**(190 对 / batch 32 对 ×2ep ≈ 13 步 + replay 步)——只有事故步数的 1/3。
3. **数据 on-policy**:chosen/rejected 都是模型自身分布的 trace(事故语料是外造新格式,遗忘压力本质不同)。
4. **DPO 自带 β-KL 锚**:loss 里的 ref(run-011)就是锚策略,β=0.1 对偏离 ref 的更新内生惩罚——这是 DPO 族相对 RFT 在反遗忘上的固有优势。
5. **CE-replay 混合 + 熔断**:每 2 个 RPO 步插 1 个 replay CE 步(32 条:cipher/gravity/numeral/unit 已对锚题 + `data/replay/` 池抽样,约占总梯度 30%);每 5 步跑 40 题锚探针($0.1),触 P3 线即回滚(save_state 每 5 步存,ttl 7 天)。

---

## f) 与死亡名单的关系(逐条)

| 死路(PATHS.md) | 本提案是否沾边 | 论证 |
|---|---|---|
| 死区采样捡漏(A) | **不踩** | 题集显式剔除 solver-fail 死区(eq 1.8%/crypt 0.75%),只取 R-loss/marginal——与死刑判决同向 |
| crypt synth 语料教 meta-skill(A) | **不踩** | `/tmp/crypt_traces.jsonl` 161 条 solver trace 显式排除出 chosen;chosen 只收模型自采样正确解 |
| CoT 格式重设计 / bit 断言式格式(A) | **不踩** | 不造任何新格式;chosen 是模型自己写的 trace,R-安全性由分布同一性保证 |
| crypt closed-form >22.5% / equation tiebreak(A) | 无关 | 不动 solver |
| lm_head / 滤 MoE(A) | **不踩** | `create_training_client_from_state` 从 run-011 weights_info 自动继承 rank32 全模块栈(recon_tinker §3.2) |
| 小补丁 FT 伤锚(B+,探针处死) | **近似但不同** | 见 e):on-policy 数据 ×1/16 LR ×1/3 步数 × 内生 β-KL × replay × 熔断,五项全是该事故的直接对冲;且这是"必须正面回应"项而非"绕开"项 |

---

## g) 可执行性:伪代码(基于 tinker 0.22.3 实际 API,逐行有出处)

```python
# ========= 0. 初始化(出处:recon_tinker §3;cookbook preference/train_dpo.py:142-196) =========
sc = tinker.ServiceClient()
# 第 0 步免费确认 state_ep1 存在(rest_client.list_checkpoints),否则按 recon_tinker §3.1 降级
training_client = sc.create_training_client_from_state(          # weights-only,fresh Adam
    "tinker://cad6ab5c-...:train:0/weights/state_ep1")
# ref 不必 save_weights_and_get_sampling_client():run-011 final 的 sampler_weights 已存在,零成本
ref_client = sc.create_sampling_client(base_model=BASE_MODEL,
    model_path="tinker://cad6ab5c-...:train:0/sampler_weights/final")

# ========= 1. 筛查 + 采样(出处:src/r_harness.py:84-110;rl_loop.py:149-172 future 批发) =========
# 贪心 1 遍 → 贪心错题的 decode 即 rejected(边采边落盘 jsonl: id/tokens/logprobs/reward)
# K=16, T=1.0 采 chosen;reward = metric_correct(truth, extract_answer(text))  # src/reasoning.py:38,46

# ========= 2. 构 pair =========
for q in greedy_wrong:
    chosen_list = dedup([s for s in samples[q] if s.reward == 1])[:2]   # 每题 ≤2 对,防单题霸权
    rejected = greedy_trace[q]
    if cat == "crypt": rejected = rejected[:3072]   # 截断式死循环只训"开局策略压制",省一半训练费
    # weights:completion-only;再把 chosen/rejected 的公共前缀 token 置 0
    #(公共前缀在 DPO margin 中数学上抵消,置 0 只去噪、专治近失 pair 的 displacement)

# ========= 3. ref logprob(出处:train_dpo.py:374 gather + compute_logprobs_async,prefill 计价) =========
ref_lps = await asyncio.gather(*[ref_client.compute_logprobs_async(seq) for seq in full_seqs])

# ========= 4. RPO 损失(出处:forward_backward_custom training_client.py:393;DPO 公式 train_dpo.py:199) =========
def rpo_loss_fn(data, logprobs_list):                      # logprobs_list = policy 逐 token logprob
    lc, lr = split_chosen_rejected(logprobs_list)          # 偶=chosen 奇=rejected(cookbook 同款拼法)
    d_c = wsum(lc - ref_c); d_r = wsum(lr - ref_r)         # weights 已含 completion mask+前缀置 0
    dpo  = -F.logsigmoid(BETA * (d_c - d_r)).mean()        # BETA=0.1(cookbook 默认;备选 0.3)
    nll  = -(wsum(lc) / wlen_c).mean()                     # 长度归一 NLL,ALPHA=1.0(Iterative RPO)
    return dpo + ALPHA * nll, {"d_chosen": d_c.mean().item(), ...}   # d_chosen 即 P2 熔断监控量,免费

# ========= 5. 训练循环(<20 步) =========
for step, batch in enumerate(pair_batches):                # batch=32 对=64 序列
    fb = training_client.forward_backward_custom(batch, rpo_loss_fn)   # ≈2× 训练价
    training_client.optim_step(tinker.AdamParams(learning_rate=5e-6, grad_clip_norm=1.0))
    if step % 2 == 1:                                      # 交替步混 replay(不赌梯度跨调用累积语义)
        training_client.forward_backward(replay_batch_32, "cross_entropy")
        training_client.optim_step(tinker.AdamParams(learning_rate=5e-6, grad_clip_norm=1.0))
    if step % 5 == 0:
        save_state(ttl_seconds=604800); anchor_probe_40()   # P2/P3 熔断;另盯 fwd_bwd metrics 的
        # e_frac_with_tokens / e_max_violation(MoE 路由免费监控,recon_tinker §5.9)
# 工程坑全套照搬 recon_tinker §5:except 元组剥 tinker.Timeout、心跳饿死 numpy 化、
# rollout 先落盘再训练、余额 ≥1.3× 预算再发车。
```

**初始化拍板**:主臂直接在 run-011(0.85,提交底座)上做——pair 全部来自 train_split,无 holdout 泄漏;holdout 评测对 run-011 偏乐观(它训过 holdout),所以 P4 以 **LB 为终审**、holdout 只做相对闸门。不另跑 run-012 干净复刻(双倍成本,4 天窗口不值;归因靠 RFT-only 对照臂)。

---

## 结论卡

| 项 | 值 |
|---|---|
| 定位 | **RFT 的附加损失(RPO 式)**,不单独成路;与 02 号 RFT 共享采样,边际 $6-9 |
| ΔLB 中心 | +0.6pp(RFT+DPO 合计;DPO margin 边际 +0.2~0.3pp) |
| 单独推到 ≥0.865 概率 | ~0.12(需 +1.5pp,本路中心只给 0.6) |
| 成本 | 独立 $19-22,上限 $40;附加模式 $6-9 |
| 天数 | 核心闭环 2 天(D1 训完、D2 LB 终审),含迭代 3 天 |
| 最大风险 | likelihood displacement(P2 熔断)+ holdout↔LB 2-3pp 未归因矛盾(P4 以 LB 终审) |
