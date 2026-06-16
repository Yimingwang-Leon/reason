# 05 · 总参谋部:点数预算与组合作战手册(0.85 → 0.865)

**角色:账本与组合策略。不提新训练方法;给出 +28 题的逐题账、各路线题集交并、4 天日历、预算分层、决策树与兜底。**
**撰写 2026-06-11 15:00。所有数字引用 recon_data/recon_tinker/recon_intel 与本日新增解码数据(`data/run012_holdout_decode.jsonl` 已 329 行,crypt 107/110 解码完成)。**

---

## 0. 一页结论

- **目标换算**:显示分 0.87 = 真分 ≥0.865。run-011 显示 0.85 = 真分 ∈ [0.845, 0.855],中心取 0.850 → **需 +1.0~2.0pp,中心 +1.5pp ≈ 28.5 题(1 题 = 0.0527pp)**。
- **可触达猎物总池(holdout 口径)≈ 64 题 ≈ +3.4pp**:bit R-loss 32 + crypt solver-ok 26 + eq R-loss ~6。达标需吃下 ~45%(若 holdout→LB 迁移打 0.7-0.85 折,需 ~53-64%)——**紧但不是奢望**,与文献 RLVR +1~5pp 量级吻合(recon_intel Q1)。
- **结构性锁死 ≈ 181 题 ≈ 9.5pp,任何路线都碰不到**:crypt oracle-死区 139(pass@8=0.75%)+ eq qop_unseen ~20(1.8%)+ bit oracle-dead 22(信息论上限)。这是 0.89 头部与我们的差距所在(他们 oracle 更高),4 天内不追。
- **路线交并核心结论:RL/RFT/DPO 吃的是同一池肉(高度重叠,不可加),融合/方差吃的是另一池(可加)**。因此组合 = 串行升级(RFT→DPO/GRPO 按 gate 升级),而非并行烧钱;融合腿全程免费并行。
- **中心预测:组合拳 +0.6~1.2pp(中心 +0.8pp)→ 真分 0.858,显示 0.86;P(真分 ≥0.865)≈ 25-30%。**
- **兜底铁约束:run-011 final(0.85)永不覆盖,最终 2 份提交锁定 {run-011, 最佳挑战者} → 全部失败仍保 0.85。**
- 预算三档:**$40 保守(RFT-only)/ $80 标准(+DPO 或二轮)/ $150 激进(+mini-GRPO)**;全局证伪线 = 06-13 晚 holdout 猎物净转化 <+5 题即停花钱。
- ⚠️ 执行前提:run-012 证伪线触发的"停止花钱"仍在生效,本手册同时是解锁申请书;解锁档位由用户拍板。

---

## 1. 点数预算:+28 题从哪来(硬性要求 a)

### 1.1 账本基础(全部可溯源)

| 量 | 值 | 出处 |
|---|---|---|
| holdout 总题 | 1899,1 题 = 0.0527pp | recon_data §1.2 |
| holdout oracle | 1708/1899 = 89.9% | recon_data §1.3(2026-06-11 实测) |
| run-012 holdout R(干净测量) | bit 91/110=82.7%、cipher 110/110、crypt 0/107(102 撞 cap)、eq/gravity/numeral/unit 待解码 | `data/run012_holdout_decode.jsonl` 329 行(本日 15:00 复核,解码进程仍活) |
| run-011 LB | 0.85(53534096);run-012 LB 0.84(53556034) | EXPERIMENTS.md |

### 1.2 猎物清单(可触达池,共 ~64 题 ≈ +3.4pp 理论上限)

| # | 题集 | 题数(holdout) | 证据 | pass@k 证据 | 转化率假设(→贪心) | 中心增益 |
|---|---|---|---|---|---|---|
| P1 | **bit R-loss**(oracle 对/贪心错,全部 1-2 bit 近失) | **~32**(11/110 实测外推;11 题 id 在 `/tmp/oracle_vs_model.json`) | recon_data §3.1 | 未直测;近失结构 + bit_tail 6%@K4 → pass@8 估 15-35% 中心 20%(recon_data §4.4) | 池 × 覆盖(K16 命中题占比 ~30-50%)× RFT 翻转 ~60% → **吃 30-40%** | **+10~13 题** |
| P2 | **crypt solver-ok**(R-loss 全集,模型现 0%、102/107 撞 cap) | **26**(id 全列在 recon_data §3.3) | 本日解码坐实 0/107 | 训练侧同构题 pass@8 = **17%**(2/12,K8@2000tok,`/tmp/rescue_test.log`)| 覆盖 ~25%(K16-32)× 翻转 ~50% → **吃 12-20%** | **+3~5 题** |
| P3 | **eq R-loss**(待解码;假设 R≈0.95) | **~6**(4-8) | recon_data §3.1 外推 | 已训 eq 贪心复现 100%(probe80)→ R-loss 少而软 | 吃 30-50% | **+2~3 题** |
| P4 | eq 签名类 marginal(oracle-fail 但 `62 vs -62` 型) | ~8-10 | recon_data §5.1 | 无直测,估 pass@8 20-40% | 吃 10-25% | **+1~2 题** |
| P5 | **融合/方差彩票**(SWA×2、alpha、epoch0、checkpoint 选点;吃贪心边际翻转方差,**与 P1-P4 不重叠、可加**) | ±5-8 题摆动 | PATHS"存活"表(已在上传/评分中) | n/a(max-of-draws,有 run-011 兜底则非负) | 5 submit/天 × 选点 = 取最大值 | **+2~4 题** |

**转化率假设的依据**(硬性要求 a 第二句):pass@k→greedy 的机制 = "RL/RFT 只把支撑集内已有路径压进 pass@1"(Limit of RLVR / DeepSeekMath,recon_intel Q1.2);我们的直接证据 = 营救臂 17%@K8(P2)与 bit 近失结构(P1,1-2 bit 之差意味着支撑集内正确路径密度高,pass@16 覆盖应显著高于 17%)。**"覆盖×翻转"两段式记账,M0 测量轮($4)直接钉死覆盖项,翻转项由 R1 实测替换**——这两个数是本手册全部预测的可调参数,先验区间已预登记(§5)。

### 1.3 合计与达标判定

```
中心:P1 11 + P2 4 + P3 2.5 + P4 1.5 + P5 3 ≈ +22 题 ≈ +1.16pp(holdout 口径)
迁移折扣 0.7-0.85(见 1.5 矛盾项)→ LB 真分 +0.8~1.0pp → 0.858~0.860,显示 0.86
乐观(覆盖/翻转取上界 + 彩票上界):+32~38 题 → 真分 ≥0.865,显示 0.87 ✓
悲观(M0 覆盖测崩):+3~6 题 → 0.853,显示 0.85,损失 = 已花预算
```
**P(真分 ≥0.865)≈ 25-30%(标准档);保守档 ~20%,激进档 ~30-35%。** 这是诚实数,不是动员数:单靠本池满打满算 +3.4pp,达标要求高位执行 + 彩票配合 + run-011 真分恰好在桶内偏高位。

### 1.4 结构性不可达(明示,不许任何提案再碰)

| 块 | 题数 | pp | 证据(全 A 级) |
|---|---|---|---|
| crypt oracle-死区 | 139/165 | 7.32 | 挖矿 pass@8=0.75%(3/400);含 `}` truth 永远零 reward |
| eq qop_unseen 死区 | ~20 | 1.05 | 0/115 规则可恢复 + 挖矿 1.8% |
| bit oracle-dead | 22/23 | 1.16 | `/tmp/bit_headroom.json`:18 无候选、4 歧义集 |
| **合计** | **~181** | **~9.5** | LB 天花板 ≈ 0.899×R;头部 0.89 的差距在 oracle,4 天追不动 |

### 1.5 账本的最大不确定项(预登记为 G0 闸门)

run-012 holdout 外推 ≈ 0.866 vs LB 真分 ∈[0.835,0.845]:**2.1-3.1pp 未归因**。两种世界:
- **世界 A**:待解码的 eq/gravity/numeral/unit 在 holdout 上 R<1 → 缺口在 holdout 侧,**猎物池变大**(每发现 10 题 R-loss = 池 +0.5pp),计划不变、期望上调;
- **世界 B**:四类解码回来 ~100% → 缺口在 LB 侧(分布/难度差异)→ **holdout 增益迁移 LB 要打 ~0.7 折**,期望下调,预算档位锁保守。
解码进程已活(免费),**今晚出答案,作为 G0 输入**。本手册中心预测已按 0.7-0.85 折扣计入。

---

## 2. 路线题集交并:谁吃谁的肉(组合的逻辑根基)

| 路线 | 吃的题集 | 与其它路线关系 | 定位 |
|---|---|---|---|
| **RFT(拒绝采样 SFT + replay)** | P1+P2+P3+P4(全部 R-loss/marginal) | — | **主攻**。greedy-对齐度最高(NLL 直接抬 argmax,recon_intel Q4.2)、最便宜、与现有 train_tinker.py 100% 复用 |
| **GRPO** | 与 RFT **同一池**(重叠 >80%) | RFT 的替代而非补充;额外价值仅在"覆盖高但 RFT 翻转失败"的残余 | 条件升级项(激进档);K=8 下仅 ~1/5 猎物有梯度(recon_data §4.4),起步即需 K16+ |
| **DPO+NLL/RPO** | 同一池;机制差异 = 压错模态 vs 抬对模态 | 复用 R1 rollouts,$8-15 边际成本;likelihood-displacement 坑(recon_intel Q4) | 条件升级项(标准档) |
| **融合/方差**(SWA、alpha、ckpt 选点、提交组合) | 贪心边际翻转(**与上面正交可加**) | 免费并行;5 submit/天 = max-of-draws | 全程开着的彩票腿 |
| ~~死区采样/合成语料~~ | §1.4 的 181 题 | — | **不碰**(死亡名单) |

**结论:正确组合 = 一条主攻线(RFT)按 gate 串行升级(→DPO→GRPO),融合线免费并行,而不是三条训练线并行**——并行训练线互相吃同一池肉,采样费(成本大头 60-85%,recon_tinker §4)却要付三遍。

---

## 3. 成本表(硬性要求 b;单价全部引 recon_tinker §4 实测锚)

单价锚:采样 $0.0025/rollout(生成均值 5.5k tok)、训练 $0.218/M token、RL 步(512 datum)~$0.75/步、锚题探针(40 题贪心)~$0.1/次。

### 3.1 分项报价

| 项 | 内容 | 采样$ | 训练$ | 小计 |
|---|---|---|---|---|
| **M0 测量轮** | holdout 猎物 58-64 题(32 bit R-loss 外推补完 + 26 crypt)× K8 @7680 + 锚探针基线 | $3.5 | 0 | **~$4** |
| **R1 RFT 弹药** | train_split 核心猎物 ~350 题(bit 100-150 + crypt 116-146 剔 `}` truth + eq 30-50,recon_data §5.2)× K16 | $14 | — | $14 |
| **R1 RFT 训练** | 正样本 ~150-250 条(去重 ≤2 条/题)+ 巩固 300 + 锚 replay ≥50%,共 ~3-5k 行 × 1ep ≈ 25-35M tok | — | $6-8 | $6-8 |
| **R1 评测** | holdout 猎物 64 + 锚 200 + 各类 spot 300 贪心解码 + ckpt 存/取 | $1.5 | $1 | ~$3 |
| **R2a DPO+NLL** | pair 复用 R1 rollouts;ref logprob prefill + forward_backward_custom(1.5-2× 训练价) | $0 | $8-15 | $8-15 |
| **R2b 补采+RFT-v2** | bit 未破题 ~100 × K16 增量 + 重训 | $4 | $6 | ~$10 |
| **R3 mini-GRPO** | cohort 500-1000 × G8-16,dynamic sampling 丢全同组,importance_sampling→cispo,2-3 个 pass | $25-50 | $10-25 | $35-75 |
| **D3 consolidation** | 赢家配方并入全 train(含 holdout 猎物补采 $2.4)重训一遍(run-011 先例) | $3 | $10-12 | ~$15 |

### 3.2 三档预算(含上限与止损)

| 档 | 内容 | 预期实花 | **硬上限** | 止损线 |
|---|---|---|---|---|
| **$40 保守** | M0 + R1 + 融合腿 + 余量 | $26-32 | **$40** | M0 覆盖闸 FAIL → 已花 ≤$5 即停;R1 净转化 <+3 题 → 停 |
| **$80 标准** | + R2(a 或 b,按 G2 分支二选一)+ 二轮评测 + D3 consolidation | $55-72 | **$80** | 06-13 晚全局证伪线(<+5 题)→ 剩余预算冻结 |
| **$150 激进** | + R3 mini-GRPO(仅当 G2 显示"覆盖高、RFT 翻转不足") | $115-140 | **$150** | GRPO 50 步内猎物 reward 均值无上行 → 停(吸收 unsloth"前 100-150 步无信号"警告,我们只打预筛猎物池,50 步 ≈ 池过 3 遍,再无信号即死) |

工程红线(每笔发车前):余额 ≥1.3× 当笔预算;rollout 边采边落盘(id/tokens/logprobs/reward);`--usd-per-step` 按 RL 步 $0.75 重标;MoE `e_frac_with_tokens` 趋势监控(recon_tinker §5)。

---

## 4. 时刻表(硬性要求 c;今天 06-11,截止 06-15,5 submit/天)

实测节奏锚:run-012 全链"发车训练→出分"= 6h(07:12→13:21);纯"提交→出分"≈ 1-3h;2000 rollouts ≈ 小时级;RFT 训练(~5k 行 1ep,~80 步)< 1h。

| 日 | 动作(花费) | 提交配额用法 | 闸门 |
|---|---|---|---|
| **D0 今晚 06-11** | $0:解码跑完→账本重算(世界 A/B 判定);`/tmp` 抢救清单 cp 进 repo(§6 recon_data,重启即灭失);红队过各提案;用户拍板解锁与档位;收 SWA/alpha/epoch0 免费彩票分 | 0-2(彩票已在评分中) | **G0**(免费):世界 A/B + 彩票是否已白捡 ≥0.86 |
| **D1 06-12** | 上午 M0($4)→ G1;下午 R1 弹药采样($14);傍晚 RFT-v1 训练($7)+ holdout 评测($3) | 晚间 3 发:RFT-v1 + 融合变体 ×2 | **G1**(M0):见 §5 |
| **D2 06-13** | 上午读 LB+holdout 复盘 → G2 分支;执行 R2a/R2b(标准档)或 R3 首段(激进档);晚间评测 | 3 发:R2 产物 + 保险 | **G2** + **当晚 22:00 = 全局证伪检查点** |
| **D3 06-14** | 赢家 consolidation(可选,$15:配方并全 train 重训,run-011 先例)+ 融合(RFT-ckpt × run-011 SWA/alpha) | 4 发:consolidation + 融合 ×2 + 重交最佳 | **G3**(LB):≥0.86 → 进决赛圈 |
| **D4 06-15** | **不开新训练**;只做选点与补交;预留出分延迟缓冲(最后一发不晚于截止前 4h) | 3 发 + 2 缓冲 | 最终锁定 2 份:**run-011(0.85)+ 最佳挑战者** |

配额总账:25 个 slot,计划用 13-15,留 ≥10 缓冲——出分延迟、彩票加注、意外重交都有余地。

---

## 5. 预登记预测与证伪线(硬性要求 d)

| 闸 | 预登记预测(写在花钱前) | 证伪线(≤Y 即停/降级) |
|---|---|---|
| **G0**(免费) | 解码四类回来:中心预测 = 世界 A/B 五五开;eq R-loss 4-8 题 | 非闸,只更新账本与折扣系数 |
| **G1 = M0 覆盖**($4 后) | bit:≥1 hit@8 的题占比 **中心 35%(区间 15-55%)**;crypt:**中心 15%(5-30%)** | bit <20%(<6/32)→ P1 砍半、档位锁保守;**bit 与 crypt 双双 <10% → RL/RFT 全线弃,只剩融合腿,总损失 ≤$5** |
| **G2 = R1 转化**(D2 上午) | holdout 猎物净翻转 **中心 +8 题(区间 +3~+15)**,锚题 ≥98%,全类 spot 不低于 run-012 水平 | <+3 题 → 保守档终局;+3~5 → 只做 D3 consolidation 不升级;锚 <97% → 回滚,replay 加倍重训一次(预算内) |
| **G2 分支判据** | 覆盖高(M0 ≥30%)且翻转低(<40% 的覆盖题翻成贪心对)→ R2a DPO / R3 GRPO 有肉;覆盖低 → R2b 补 K 或砍腿 | — |
| **全局**(06-13 22:00) | 累计 holdout 猎物净转化 **≥+5 题** | **<+5 → RL/RFT 路线对本比赛判死,停花钱,D3/D4 只跑融合与选点;保 0.85** |
| **LB 终判** | 中心:显示 0.86;P(0.87)≈25-30% | 挑战者 ≤0.84 → 杀;=0.85 → 与 run-011 融合后再试一发 |

---

## 6. 反遗忘设计(硬性要求 e;直面"60 步 LR8e-5 把锚打到 80%")

那次事故的三要素:**off-policy 异格式语料 + 8e-5 + 从退火态续训无 replay**(E4,probe_ft)。本计划逐项反着来:

1. **数据 on-policy**:RFT 正样本全部是模型自己采的 trace(自身分布,遗忘压力天然小,recon_tinker §3.3);**严禁混入离线 solver 写的 trace**(那是死亡名单上的 crypt-synth/forward-crypt)。
2. **replay ≥50%**:语料里巩固 300 + 锚四类各 ~100 + `data/replay/` 抽样,正样本占比 <50%——不是 60 步纯补丁,是"补丁泡在锚汤里"。
3. **LR ≤4e-5,fresh Adam**:`create_training_client_from_state`(weights-only)从 state_ep1 起;LR 取 2-4e-5(< 事故的 8e-5,= cookbook rank32 RL 标定区间);grad_clip_norm=1.0(不再 1e9)。
4. **锚题探针熔断**:每 20 步 40 锚题贪心($0.1/次),<95% 即停训回滚——比 KL 全量 prefill(~$40)便宜 400 倍,M0 先建基线。
5. **GRPO 腿(若启用)**:`incorporate_kl_penalty` 挂 run-011 参考策略,只对 10-20% 子样本计 KL 控费(recon_tinker §2.3),或直接沿用锚题熔断。
6. **结构性兜底**:一切训练产物是新 checkpoint,run-011 final 物理不动;最终提交锁双份。

---

## 7. 与死亡名单的关系(硬性要求 f,逐条)

| 死路(PATHS.md) | 本计划是否碰 | 说明 |
|---|---|---|
| crypt closed-form >22.5% / 出题器复刻 | ❌ 不碰 | 不动 solver |
| **crypt 死区采样捡漏** | ❌ 不碰 | 猎物集 = solver-ok 26 题(R-loss),死区 139 题明示锁死(§1.4);`}` truth 剔除 |
| **crypt synth 语料 / forward-crypt 格式** | ⚠️ 近似但不同 | 死的是"离线写的演绎 trace 教 meta-skill"(贪心复现 3%/5%)。本计划 crypt 腿只用**模型自己采出的正确 trace**(on-policy,机制 = 把自身支撑集内路径压进 argmax,有营救臂 17% 直接证据)。**若 M0 测得 crypt 覆盖 ≈0,crypt 腿当场砍掉**——预登记,不恋战 |
| equation tiebreak / qop_unseen | ❌ 不碰 | eq 腿只打 R-loss(P3)与签名 marginal(P4)的自采样,不动 solver 语料 |
| bit 断言式格式 / bit >92.8% | ❌ 不碰 | 不改格式;只打 oracle-ok 的 R-loss 32 题 |
| CoT 格式重设计(瘦身/硬化) | ❌ 不碰 | RFT 保持模型自己的格式 |
| lm_head / 滤 MoE | ❌ 不碰 | 提交配置原样(全栈 + MoE up/down) |
| tinker 0.16.1 | ❌ 不碰 | 0.22.3 |
| (B+)小补丁 FT 伤锚 | ⚠️ 正面应对 | §6 全套;"为什么这次不同"= on-policy + replay ≥50% + LR 减半 + 熔断 |

---

## 8. 执行骨架(硬性要求 g;API 全部来自 recon_tinker 实测源码)

```python
# ============ 第 0 步(免费):资产确认 ============
sc   = tinker.ServiceClient()                                  # env.json 鉴权,r_harness._load_env 先例
rest = sc.create_rest_client()
ok   = "weights/state_ep1" in list_ckpts(rest, "cad6ab5c-...") # run-011 训练态;无则降级 state_ep0 或 $10 重放
SAMP = "tinker://cad6ab5c-...:train:0/sampler_weights/final"   # 0.85 底座 = rollout 源 + 参考策略

# ============ M0:覆盖测量(~$4)============
sampler = sc.create_sampling_client(base_model=BASE, model_path=SAMP)
futs = {q.id: sampler.sample(prompt=ModelInput.from_ints(tokenize_prompt(q)),   # corpus.tokenize_prompt 制式
                             num_samples=8,
                             sampling_params=SamplingParams(max_tokens=7680, temperature=1.0))
        for q in prey_holdout_64}                              # 32 bit R-loss + 26 crypt + eq(G0 后补)
for qid, f in futs.items():                                    # 先发全部 future 再收割(rl_loop.py 范式)
    for s in f.result().sequences:                             # 边收边落盘:session 死了钱不白花
        r = metric_correct(truth[qid], extract_answer(decode(s.tokens)))   # reasoning.py:38/46,与 grader 同构
        append_jsonl(ROLLOUTS, dict(id=qid, tokens=s.tokens, logprobs=s.logprobs, reward=r))
gate_G1(coverage_by_class(ROLLOUTS))                           # §5 阈值,FAIL 即止损

# ============ R1:RFT(拒绝采样 SFT,$25 内)============
# 弹药:train_split 猎物 ~350 × K16(同上);留 reward==1、去重 ≤2 条/题 → 150-250 正样本
corpus = positives + consolidation_300 + anchors_replay        # 正样本占比 <50%(§6.2)
tc = sc.create_training_client_from_state("tinker://cad6ab5c-...:train:0/weights/state_ep1")  # weights-only, fresh Adam
for step, batch in batches(corpus, bs=64, epochs=1):
    fb = tc.forward_backward(to_datums(batch), loss_fn="cross_entropy")    # weights = completion mask
    tc.optim_step(AdamParams(learning_rate=3e-5, grad_clip_norm=1.0))
    moe_watch(fb.result().metrics)                             # e_frac_with_tokens / e_max_violation 趋势
    if step % 20 == 0 and anchor_probe_greedy(tc) < 0.95:      # 40 锚题 ~$0.1,熔断
        rollback(); break
evaluate_holdout(prey_64 + anchors_200 + spot_300); gate_G2()

# ============ R2/R3(条件分支,G2 判据)============
# R2a DPO+NLL:pair 取自 R1 rollouts(同题正/负);ref = SAMP 的 compute_logprobs_async;
#              forward_backward_custom(cookbook train_dpo.py);必须带 NLL 项(Iterative RPO 教训)
# R2b 补采:bit 未破题 ×K16 增量 → RFT-v2
# R3 GRPO(激进档):cohort 500-1000;A = r - mean(组)(data_processing.compute_advantages);
#    全同奖励组丢弃(dynamic sampling,免训练费);loss="importance_sampling" 不稳切 "cispo";
#    LR 1e-5;每 2-4 步 save_weights_and_get_sampling_client 刷新;KL 锚 10% 子样本或锚题熔断
```

---

## 9. 决策树(全图)

```
D0 G0 ──世界A(四类有R-loss)→ 池+N题,期望上调┐
      └─世界B(四类干净)──→ 迁移折扣0.7,锁保守┘→ 用户解锁档位
D1 G1 ──bit≥20% 或 crypt≥10% ──→ R1 RFT
      └─双双<10% ──────────────→ 弃 RL/RFT(损失≤$5),融合-only,保0.85
D2 G2 ──净转化≥+8 ──→ 提交 + (标准档)R2 + (激进档)R3 ──→ D3 consolidation
      ├─+3~+8 ──────→ 提交 + 只做 D3 consolidation
      └─<+3 ────────→ 停;若覆盖高翻转低 且激进档解锁 → R3 试50步,否则融合-only
06-13 22:00 全局证伪:累计<+5题 → 全线停花钱
D3 G3 ──LB≥0.86 → 决赛圈:consolidation/融合再加注
      └─LB≤0.85 → 融合一发,不行即收
D4 锁定2份:run-011(0.85) + 最佳挑战者    ← 任何分支都到这,0.85 永在
```

---

## 10. 给红队的自首清单(已知薄弱点)

1. **P1 的 pass@k 无直接测量**(最大单块 +10~13 题建立在 15-35% 估计上)——M0 就是为它设计的,$4 钉死,FAIL 即按 G1 降级,不赖账。
2. **"覆盖×翻转"的翻转项(RFT 训 train 猎物 → holdout 兄弟题贪心翻转)无先例直接证据**,是本计划最大的假设;Iterative RPO/Tülu 的量级证据是间接的。R1 一轮 $25 就是它的测谎仪。
3. 2.6pp 未归因缺口若是世界 B,达标概率从 ~30% 掉到 ~20%;已用折扣系数计入中心预测,但不排除更糟(LB 分布性差异)。
4. crypt 腿与死亡名单距离最近(§7),靠"on-policy vs 离线 trace"的机制区分立足;M0 若测出 crypt 覆盖 ≈0 立即砍,不进入任何后续轮。
5. 4 天内 GRPO 只够 50-100 步量级,远低于文献"≥300 步见信号"的口径——所以它被放在激进档最末位、且只打预筛猎物池;主攻必须是 RFT。
