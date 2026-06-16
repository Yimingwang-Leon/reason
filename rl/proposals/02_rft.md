# 提案 02:RFT / Expert Iteration(迭代拒绝采样 SFT,STaR/ReST 系)

**作者:RFT 专家 | 日期 2026-06-11 | 状态:设计稿(零花费),待红队**

**一句话:把"营救臂 17%"产品化——从 run-011 (0.85) 用 temperature 采样 R-loss 题,只留模型自己写出的 boxed-正确 trace,混 on-policy 锚做 20 步级小 LR SFT,2-3 轮迭代,每轮 greedy 闸 + LB 验证。中心预期 +0.8pp(0.858),上限 +1.3pp,封顶花费 $80。**

---

## 0. 为什么 RFT 是本族里 greedy-对齐度最高、风险最低的武器

1. **评测是 greedy argmax**。DPO 优化 margin 不直接抬众数(recon_intel Q4:Iterative RPO 证明真正抬 chosen logprob 的是 NLL 项);GRPO 在 K=8、R-loss 题 pass@8≈17-20% 下只有 ~1/5 的组有梯度。RFT = 纯 NLL on 自采正确 trace,**每一分钱训练费都直接花在"抬 argmax"上**。
2. **on-policy 数据天然 R-安全**。run-012 的惨案(synthetic crypt trace 背了也只复现 3%)是 off-policy 模仿失败;RFT 的 trace 是模型自己的写法,p_model(trace) 本来就高,SFT 只是把它从"采样可达"推成"贪心首选"——这正是营救臂 2/12=17% 实测验证过的机制(/tmp/rescue_test.log:d0b1e41a、524cb5c6 被救)。
3. **最便宜**。训练费只付正样本(recon_tinker §2.5:RFT 更新费 ≈ 采样费的 5-15%);采样产物(tokens+logprobs+reward 落盘)可被 GRPO/DPO 提案 100% 复用,本提案即使止损,弹药不浪费。
4. **代码 100% 复用**:loss=cross_entropy + completion mask,就是 `src/train_tinker.py` 现有路径(`_build_datum`,:128-154);采样就是 `src/r_harness.py:84-110` 的 K>1、T>0 版。新代码只有"采样落盘 + 过滤"两个脚本和 train_tinker 的两个小 flag。

文献锚(recon_intel):RFT/STaR 在已收敛模型上 +2~5pp(Yuan et al. RFT、Iterative RPO 每轮重采);Limit-of-RLVR 机制(把 pass@k 压进 pass@1)对 RFT 同样成立且更直接。

---

## a) 数字账:+X 题从哪来

### a.1 猎物池(引用 recon_data §3,2026-06-11 解码已更新到 329 行)

| 池 | holdout 实测/外推 | train_split 对应训练 cohort | 价值 |
|---|---|---|---|
| bit R-loss(oracle 对/贪心错,全是 1-2 bit 近失) | 11/110 实测 → 外推 **~32/320** | oracle-ok 1282 题中估 **100-150 题**(贪心识别 pass 圈定) | +1.69pp 上限 |
| crypt solver-ok(R-loss 全集:贪心 0/110,105 撞 7680 cap) | **26/165**(id 清单在 recon_data §3.3) | production~improved solver-ok **116-146 题**,剔 `}` truth 后 ~135 | +1.37pp 上限 |
| eq 签名/marginal(实测 10 题解码 7 对;`07aef27f` 62 vs -62 即签名类) | 估 **6-12 题** | solver-ok ~450 题贪心识别后取错题 ~30-50 | +0.3~0.6pp 上限 |
| **合计** | **~64-70 题 = +3.4pp 理论上限** | RFT 采样 cohort ≈ **300-335 题** | |

**明确不打**(死区,见 §f):eq solver-fail(pass@8=1.8%)、crypt solver-fail(0.75%)、`}` truth crypt(boxed 提取下永远零 reward)、4 个 100% 健康类(只做锚)。

### a.2 转化链与每环假设依据

```
LB 增量 = Σ_类 [holdout R-loss 池] × [pass@16 可采率] × [RFT 训后贪心转化率(同类泛化)]
```

| 环 | 假设 | 依据 |
|---|---|---|
| crypt pass@16 可采率 ~25%(15-35%) | pass@8=17%(2/12 直接实测,且当时 max_tokens 仅 2000;K 翻倍 + cap 放开到 7680 取 1.5× 折半保守) | /tmp/rescue_test.log |
| bit R-loss pass@16 可采率 ~35%(20-50%) | 近失结构(1-2 bit)+ bit_tail 6%@K4(半死区都有 6%,R-loss 题理应更高);recon_data §4.4 中心 20%@K8 | /tmp/mine_results.jsonl |
| 已采题训后**自身**贪心转化 60-80% | on-policy 正确 trace 小 LR SFT ≈ 把已高概率路径推成众数;E4 探针:已训锚题贪心复现 80-100%(off-policy synthetic 才是 3%) | /tmp/probe80_results.json |
| **holdout 同类泛化率 50-60%**(本提案最大不确定) | 营救机制是"执行巩固"不是背题;bit 近失=进位/边界执行错,模式同构;文献 RFT 增益全部来自 unseen test(Yuan et al. +2~5pp 即 test 口径) | recon_intel Q1/Q4,B 级 |

### a.3 三档账(holdout/LB 口径,题 ×0.0527pp)

| 档 | bit | crypt | eq | 合计题 | ΔLB |
|---|---|---|---|---|---|
| 悲观(可采率取下限、泛化 30%) | +3 | +1 | +1 | +5 | **+0.26pp** |
| **中心**(中心可采率、泛化 50%) | +7~9 | +3~4 | +2 | **+13~15** | **+0.7~0.8pp** |
| 乐观(上限可采率、泛化 60% + 第 2/3 轮迭代增量 + self-distill 锚顺带巩固 flaky 题) | +12 | +7 | +4 | +23~25 | **+1.2~1.3pp** |

**坦白:RFT 单兵中心值到不了 +1.5pp(0.865)**。要单独够线需要乐观档 + "holdout 外推 0.86-0.87 vs LB 0.84 的未归因 2-3pp"里有一部分对 RFT 有利(recon_data §3.1 矛盾)。单兵 P(≥0.865) ≈ 0.15。但它是各提案中 **$/题 最低、与 GRPO/DPO 完全共享弹药、且每轮都有独立提交点**的一条路,适合作主干或与 GRPO 第二棒串联(round-1 的 rollouts 直接是 GRPO 的预筛 f_keep 数据)。

### a.4 免费种子(第 0 轮就有,$0)

- `/tmp/mine_results.jsonl`:12 条 on-policy 正确 trace(3 eq + 3 crypt + 6 bit,含 pass 数);
- `/tmp/rescue_test.log` 的 2 条 crypt 营救 trace(d0b1e41a、524cb5c6,913-1628 字符——**短**,正是治 crypt 105/110 撞 cap 的解药形状);
- ⚠️ `/tmp/crypt_traces.jsonl` 的 170 条 synthetic 正确 trace **不算弹药**(off-policy,run-012 已证伪其可迁移性),不入训练集。
- 第 0 步必须把 /tmp 清单(recon_data §6)cp 进 repo,机器重启即灭失。

---

## b) 成本表(全部按 recon_tinker §4 实测锚:采样 $0.0025/rollout、训练 $0.218/M tok)

| 项 | 量 | $ |
|---|---|---|
| **B. 识别 pass(贪心 T=0,1 次/题)** | bit oracle-ok 1282 + eq solver-ok ~450 + 4 健康类锚 harvest 400 ≈ 2130 条 | **$5.3** |
| **C. round-1 harvest(T=1.0, K=16, cap 7680)** | bit R-loss ~130×16 + crypt ~135×16 + eq marginal ~40×16 ≈ 4880 条(crypt 长 rollout 溢价 ~20%) | **$13-15** |
| **D. round-1 训练(cross_entropy)** | ~1400 例 × 均 3.5k tok ≈ 4.9M tok + save_state/sampler ×2 | **$1.3** |
| **E. 每轮闸** | 锚题 40 + holdout R-loss 探针 68 + holdout 分层抽样 200 ≈ 308 条贪心 | **$0.8/轮** |
| **F. round-2**(剩余未破题 ~170×16 + 训 + 闸) | ≈ 3000 rollouts | **$9** |
| **G. round-3(条件触发)** | 同上略减 | **$8** |
| **应急:state_ep1 缺失时确定性重放 epoch1** | recon_tinker §3.1 方案 (b) | **$10** |
| **中心合计** | 两轮全跑 + 闸 | **≈ $40** |
| **硬上限(止损线)** | 三轮 + 重放应急 + 20% 意外重跑 | **$80,触线即停** |

省钱设计:识别 pass 一鱼三吃(R-loss 圈定 + 85-90% 正确贪心 trace = 免费 on-policy 锚 + train 侧真 R 测量);crypt 不做识别(已知贪心≈0,直接 harvest 省 $0.4);正样本落盘带 tokens+logprobs,GRPO/DPO 提案零成本复用。

---

## c) 时刻表(今天 06-11 起;5 提交/天,出分按 ~2-4h 延迟留量;06-15 截止)

| 时间 | 动作 | 花费 | 闸/产出 |
|---|---|---|---|
| **D0 = 06-11 晚(解冻后)** | ① cp /tmp 抢救清单进 repo;② `rest_client.list_checkpoints` 确认 `cad6ab5c.../weights/state_ep1`(免费);③ 识别 pass(B);④ round-1 harvest(C) | $19-21 | P1/P2 预测当场判;P2 证伪 → 全停($21 止) |
| **06-12 上午** | round-1 过滤+训练(D)+ 三重闸(E) | $2.1 | P3 证伪 → 停($25 止) |
| **06-12 下午** | **LB 提交 #1(rft_r1)**;并行 round-2 harvest(从 r1 新策略重采未破题——expert iteration 的"迭代"就在这步:策略变强后新题变可采) | $7 | 留 4 个提交配额给其它臂 |
| **06-13 上午** | 看 LB#1;round-2 训练 + 闸 | $2.1 | P4 判:LB ≤0.85 且 r2 闸无增量 → 全停 |
| **06-13 下午** | **LB 提交 #2(rft_r2)**;视闸决定 round-3 | $0-8 | |
| **06-14** | round-3 收尾 + **LB 提交 #3**;**12:00 后不再开新训练轮**,只做选点 | $0-2 | 留足 06-15 最终选点余量 |
| **06-15** | 选最优 ckpt 终提交 | $0 | deadline |

关键依赖:训练发车需用户解锁花钱(纪律);**提交本身已获自动授权**(memory 2026-05-24)。每轮 wall-clock:harvest ~5k rollouts 在挖矿当晚同量级(小时级),训练 20-25 步 <1h,单日一轮完整闭环可行。

---

## d) 预登记预测 + 证伪线(写死,不许事后挪)

| # | 预测(中心) | 区间 | 证伪线(≤Y 即停) |
|---|---|---|---|
| P1 识别 | train bit R-loss 数 ~130 | 80-200 | <40 → bit 臂砍掉,弹药并入 crypt/eq(不停整体) |
| P2 harvest | bit ≥1 条正确的题占比 35%;crypt 25% | bit 20-50 / crypt 15-35 | **bit <10% 且 crypt <8% → RFT 弹药不足,全停($21 止),rollouts 移交 GRPO 提案** |
| P3 round-1 闸 | holdout R-loss 探针 68 题贪心转化 ≥10 题;锚 40/40 | 6-18 | **转化 <4 题 或 锚 <36/40 → 停训并回滚($25 止)** |
| P4 LB#1 | 0.855 | 0.851-0.860 | **≤0.850 且 round-2 闸无增量 → 全线停($50 止)** |
| P5 总闸 | — | — | **累计 >$80 即停;06-14 12:00 后冻结训练** |

诚实声明:P3 的"泛化率 50%"是 B 级推断(无直接实验),它是整条链最可能断的环——所以 P3 闸放在第一次提交**之前**,$25 就能证伪,不用等 LB。

---

## e) 反遗忘设计(正面回应"60 步 LR 8e-5 把锚打到 80%")

E4 事故三要素:**off-policy 数据(synthetic 新格式)× LR 8e-5 × 60 步**。本设计逐项对冲:

1. **数据反转为 on-policy 为主**:每 batch 64 = 24 正样本(37%)+ 24 **self-distill 锚**(37%,识别 pass 里模型自己贪心答对的 trace——KL(p_new‖p_old)≈0 的完美锚,顺带巩固 flaky 题)+ 16 replay(25%,`data/replay/` 17570 行池 + champion corpus 抽样,维持 SFT 原配方记忆)。锚占比 62%。
2. **LR 砍到 1e-5~1.5e-5 恒定**(事故值的 1/5~1/8;tinker cookbook math_rl 对 rank32 用 1e-5,recon_intel Q2),`grad_clip_norm=1.0`(不再 1e9)。
3. **步数 ~22 步/轮**(1400 例 ÷ 64,1 epoch,不重复 epoch;事故是 60 步)。
4. **熔断闸**:每轮训后 40 锚题贪心($0.1),<36/40 立即回滚上一 state 并把 replay 比例提到 50% 重训;另盯 fwd_bwd metrics 的 `e_frac_with_tokens / e_max_violation`(MoE 路由健康,免费,recon_tinker §5.9)。
5. **正样本卫生即反遗忘**:只收 `stop_reason=='stop'`(不收截断)、bit/eq gen≤6500 tok、**crypt gen≤3000 tok**(营救成功 trace 实测 0.9-1.6k 字符;短 trace 正面治疗 crypt 105/110 撞 cap 的死法,也防 RFT 学会写长)、每题最多 2 条(取最短+次短去重,防 8/16 全对的易题统治梯度)、退化检测(重复 n-gram 比 >0.3 弃)。
6. 不用 KL 锚项(那是 GRPO 的事;RFT 的 NLL+replay 等价且免 prefill 费 ~$40,recon_tinker §4.2)。

---

## f) 与死亡名单逐条对表(PATHS.md)

| 死路 | 本提案是否踩 | 为什么这次不同 |
|---|---|---|
| crypt synth 语料教 meta-skill(A) | **不踩** | 死的是 synthetic reasoner trace(p_model≈0,背了只复现 3%);RFT 只用模型**自采**正确 trace(p_model 高,营救臂 17% 实测同机制成立)。`/tmp/crypt_traces.jsonl` 明确排除出训练集 |
| crypt 模型先验捡漏/死区采样(A) | **不踩** | 题集硬性剔除 solver-fail 死区(eq 1.8%/crypt 0.75% 已 A 级证死);只打 solver-ok 且贪心错的 R-loss 池 |
| crypt closed-form >22.5%(A) | 不踩 | 不改 solver,只用其 22.2% 圈题 |
| CoT 格式重设计 / bit 断言式格式(A) | 不踩 | 零新格式,trace 全是模型自产分布 |
| equation tiebheak(A) | 不踩 | eq 只取签名/marginal 题采样,不动 solver |
| lm_head / 滤 MoE(A) | 不踩 | `create_training_client_from_state` 自动继承 run-011 的 rank32 全栈配置(recon_tinker §3.2) |
| 小补丁 FT 伤锚(B+) | **近似,已正面设计** | 见 §e:LR 1/6、步数 1/3、62% 锚、熔断闸 |
| 0.16.1 SDK(A) | 不踩 | 0.22.3 已装 |

---

## g) 可执行性:伪代码(基于 tinker 0.22.3 实际 API)

### g.1 采集(新脚本 `src/rft_harvest.py`,~120 行,参照 r_harness.py:84)

```python
sc = tinker.ServiceClient()                       # env.json 鉴权,_load_env()
sampler = sc.create_sampling_client(
    base_model=BASE_MODEL,
    model_path="tinker://cad6ab5c-...:train:0/sampler_weights/final")  # run-011 0.85
sp_id  = tinker.SamplingParams(max_tokens=7680, temperature=0.0, top_p=1.0)   # 识别 pass
sp_hvt = tinker.SamplingParams(max_tokens=7680, temperature=1.0, top_p=1.0, seed=...)

futs = {}                                          # 教科书写法:全发 future 再收割(rl_loop.py:149)
for pid, prompt_ids in cohort:                     # cohort 来自离线 solver-ok ∩ 目标层
    futs[pid] = sampler.sample(prompt=tinker.ModelInput.from_ints(prompt_ids),
                               num_samples=K, sampling_params=sp_hvt)
with open(out_jsonl, "a") as f:                    # 边采边落盘(402 教训,recon_tinker §5.1)
    for pid, fut in futs.items():
        for seq in fut.result().sequences:
            text = chat_tok.decode(seq.tokens)
            rec = dict(pid=pid, tokens=seq.tokens, logprobs=seq.logprobs,  # GRPO/DPO 可复用
                       stop=seq.stop_reason, gen=len(seq.tokens),
                       reward=int(metric_correct(truth[pid], extract_answer(text))))
            f.write(json.dumps(rec) + "\n")
```

### g.2 过滤(`src/rft_filter.py`,纯本地 $0)

```python
keep = (r.reward == 1 and r.stop == "stop"
        and r.gen <= (3000 if cat=="crypt" else 6500)
        and ngram_repeat_ratio(text) < 0.3)
per_pid: sort by gen, keep 最短 2 条(去重);写成 corpus.jsonl 兼容格式
batch 配比:24 正样本 + 24 self-distill 锚(识别 pass 贪心正确)+ 16 replay
```

### g.3 训练(改 `src/train_tinker.py` 两个 flag,原地改,不开新文件)

```python
# 新增 --init-from-state(weights-only,fresh Adam)与 --constant-lr
tc = await sc.create_training_client_from_state_async(    # 不带 _with_optimizer
    "tinker://cad6ab5c-...:train:0/weights/state_ep1")    # D0 用 list_checkpoints 确认;
                                                          # 缺失 → $10 确定性重放 epoch1(§b 应急行)
for step, batch in enumerate(batches):                    # ~22 步/轮
    data = [_build_datum(ex["tokens"], ex["mask"]) for ex in batch]   # 现有函数原样复用
    fb = await tc.forward_backward_async(data, loss_fn="cross_entropy")
    await tc.optim_step_async(tinker.AdamParams(learning_rate=1.2e-5,
        beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=1.0))
    watch(fb.metrics["e_frac_with_tokens:mean"], ...)      # MoE 路由熔断
await tc.save_state_async(name=f"rft_r{n}", ttl_seconds=604800)
sampler_path = (await tc.save_weights_for_sampler_async(name=f"rft_r{n}")).path
```

### g.4 轮间闸 + 迭代

```python
# 闸:40 锚 + 68 holdout R-loss 探针 + 200 分层 holdout,全贪心,r_harness.score_model_outputs 打分
# 注意:run-011 系 holdout 有污染(recon_data §5.4),闸只看「同系前后 delta」,绝对值以 LB 为准
# round-(n+1):用 rft_rn 的 sampler 对【仍未破的题】重采 K=16 —— 策略变强后新题进入支撑集(STaR 迭代核心)
# 历轮正样本累积进训练集(旧轮权重 ×0.5),避免灾难性来回摆
```

### 衔接说明
- 与 GRPO 提案(01)互为犄角:本提案 round-1 的 rollouts(含 logprobs)= GRPO 的预筛数据(f_keep 实测免费拿到);若 P2/P3 证伪,移交弹药转 GRPO/DPO。
- 与提交底座一致:全程 run-011 血统,任何一轮的 sampler ckpt 都可直接走 `src/build_submission.py` 上传。

---

## 风险登记(红队请打这里)

1. **泛化率 50% 是 B 级假设**,无直接实验;P3 闸($25 处)是它的第一个实测点。
2. run-011 holdout 污染 → 闸的绝对值偏乐观;已用"delta + LB 仲裁"对冲,但烧 1-2 个提交配额。
3. holdout外推 vs LB 的未归因 2-3pp(recon_data §3.1)可能整体压低一切 holdout 口径收益;解码进程跑完(免费)前不应发车 round-2。
4. bit R-loss 的 pass@k 从未直接测——P2 在 harvest 当场就能证伪,损失封顶 $21。
