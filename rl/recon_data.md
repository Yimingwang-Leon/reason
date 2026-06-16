# 数据与评测资产侦察(RL 弹药账)

侦察时间 2026-06-11 14:1x。所有路径为绝对路径;⚠️ 注意 `/tmp` 下多个关键原始数据(重启即失,见 §6 抢救清单)。
**注意:run-012 holdout 解码进程仍在跑**(`/tmp/decode012_supervise.sh` → 写入 `data/run012_holdout_decode.jsonl`,本文撰写时 269/770 行),数字会继续更新。

---

## 1. 题量账

### 1.1 数据集与切分
| 文件 | 行数 | 说明 |
|---|---|---|
| `data/train_cat.csv` | 9500 | 全量 train(带 cat 列) |
| `data/train_split.csv` | 7601 | 训练侧(run-012 只用这份) |
| `data/holdout.csv` | 1899 | 分层 20%(`split_holdout.py`,seed=42,锁定不许重切) |
| `data/test.csv` | 3 | 仅示例,**LB 真实分布未知** |

### 1.2 各类题量(train 全量 / train_split / holdout)
| 类 | train 9500 | split 7601 | holdout 1899 | 占比 |
|---|---|---|---|---|
| bit_manipulation | 1602 | 1282 | 320 | 16.9% |
| cipher | 1576 | 1261 | 315 | 16.6% |
| cryptarithm_deduce | 823 | 658 | 165 | 8.7% |
| equation_numeric_deduce | 732 | 586 | 146 | 7.7% |
| gravity | 1597 | 1278 | 319 | 16.8% |
| numeral | 1576 | 1261 | 315 | 16.6% |
| unit_conversion | 1594 | 1275 | 319 | 16.8% |

LB 按 BRIEF 口径以 1899 题计(holdout 同规模镜像);真实 LB 分布假设与 train 同比例。
**每题价值 = 1/1899 = 0.0527pp;目标 +1.5pp ≈ 28.5 题。**

### 1.3 holdout oracle(当前 repo solver,2026-06-11 离线实测;脚本 `/tmp/oracle_fast.py`、`/tmp/oracle_crypt.py`)
| 类 | oracle | 失败明细 |
|---|---|---|
| bit | **297/320 = 92.8%** | 23 失败 id 在 `/tmp/oracle_fast_fails.json` |
| cipher / gravity / numeral / unit | **100%**(315/319/315/319) | 无失败 |
| equation | **117/146 = 80.1%** | 29 失败(含 diff)同上文件 |
| crypt | **26/165 = 15.8%** | 139 失败在 `/tmp/crypt_holdout_fails.json`;26 个 solver-ok id 见 §3.3 |
| **合计** | **1708/1899 = 89.9%** | LB 上限 = 0.899 × R |

⚠️ `baseline.json` 是旧 freeze(crypt 6.9%、eq 20.4% 旧 solver),**不要用它当现状**;train 侧现状:crypt production 17.6% / improved 22.2%,eq ~77-80%,bit ~92-94%。

### 1.4 run-012 语料覆盖(`corpus.jsonl`,19401 行,与 holdout 0 重叠)
主语料 6940:bit 1281、gravity 1278、unit 1275、cipher 1261、numeral 1261、eq 449、crypt 135(forward 格式);drills/replay 12461(champion 7502 + eq drills 900 + replay 4059)。replay 原料 `data/replay/nemotron_math_1gb.jsonl` 共 17570 行可继续抽。

---

## 2. reward 函数:直接复用现成判分路径

**RL reward = `metric_correct(problem.answer, extract_answer(decoded_text))`,一行即得 0/1 reward,与 grader 同构。**

- `src/reasoning.py:38 extract_answer()`:取**最后一个非空** `\boxed{...}`(正则 `\\boxed\{([^}]*)(?:\}|$)`,容忍末尾截断)。
- `src/reasoning.py:46 metric_correct()`:
  - truth 全为 `[01]+` → 串等(bit);
  - 否则尝试 float → `math.isclose(rel_tol=1e-2, abs_tol=0.0)`(已硬化为与 grader 相对容差严格同构);
  - 失败则 lowercase 串等(cipher/crypt 文本)。
- 包装好的批量打分:`src/r_harness.py:61 score_model_outputs()`;贪心解码参考实现:`src/r_harness.py:84 sample_adapter()` 与 `/tmp/decode012.py`(tokenize 用 `src/corpus.tokenize_prompt`,chat template + enable_thinking,max_tokens=7680 temp=0)。
- ground truth 全部 9500 题在 train.csv 中 → **RL 有完美离线 reward,无需 judge**。
- ⚠️ crypt 真答案含 `}` 的题在 `\boxed{}` 提取下**永远无 reward**(~76% 格式上限来源)——这类题必须排除出 RL 题集(零信号纯烧钱)。holdout 26 个 solver-ok crypt 均可 box。

---

## 3. R-loss 清单(RL 主猎物)

### 3.1 run-012 holdout 贪心解码(唯一没见过 holdout 的干净模型)
- 文件:`data/run012_holdout_decode.jsonl`(jsonl:id/cat/ok/pred/truth/gen/cap_hit/text);ckpt `tinker://cbbf1624-…/sampler_weights/final`;计划每类 110 题、按 id 排序取前 110,贪心@7680。
- 截至撰写(269 行):

| 类 | 解码 | 模型对 | 模型 acc | oracle 对(同批) | **R-loss(oracle对/贪心错)** |
|---|---|---|---|---|---|
| bit | 110(完) | 91 | 82.7% | 102 | **11 题(10%)** |
| cipher | 110(完) | 110 | 100% | 110 | **0** |
| crypt | 47(进行中) | **0** | 0%(45/47 撞 7680 cap) | ~7-8(按 15.8% 折算) | **≈全部 solver-ok 题** |
| eq / gravity / numeral / unit | 各 1(canary) | 1/1/—/— | 未知 | — | **待解码** |

- bit 的 11 题 R-loss(id、pred/truth 见 `/tmp/oracle_vs_model.json`):全部近失(多为 1-2 bit 之差),无 cap_hit,平均 6646 tok;其中 5 题 oracle 靠 EV 格式、6 题 legacy 格式(`/tmp/bit_details.json`)。
- **外推全 holdout:bit R-loss ≈ 32/320 题 ≈ +1.69pp;crypt R-loss ≈ 26 题 ≈ +1.37pp;eq 假设 R≈0.95 → ≈6 题 ≈ +0.3pp。主猎物总池 ≈ 64 题 ≈ +3.4pp 理论上限;吃下 45% 即达 +1.5pp 目标。**
- ⚠️ 矛盾待解:按已解码数据外推 LB ≈ 0.86-0.87,但 run-012 实测 LB=0.84,~2-3pp 差距未归因(可能藏在未解码的 eq/gravity/numeral/unit 的 R<1,或 LB 分布差异)→ 解码跑完是第一优先情报。

### 3.2 训练侧(RL 训练题应来自 train_split,holdout 留作干净评测)
- bit:train_split oracle-ok ≈ 1183;微探针显示已训难尾贪心 85%(17/20)→ train 侧 R-loss 候选 ≈ 100-150 题。
- crypt:solver-ok ≈ 116(production 17.6%)~146(improved 22.2%)题;run-012 语料只进了 135。
- eq:已训题贪心复现 100%(微探针 30/30)→ train 侧 R-loss 少,收益主要靠泛化。

### 3.3 holdout 26 个 solver-ok crypt id(穷 R-loss 池,贪心全错中)
`0babcba2 0e2d6796 23b0eb54 2c017f70 2d89386e 39a1f5e9 41a8a9f0 524cb5c6 56ac76c6 58eadc55 64d775e5 6fcbf5fd 7681df4d 7d8f22b1 99d6a3b5 bc83b0a1 bca230fd c9780577 cef50c24 dac75343 db4f43ef e2361a00 ea04ed35 eabc719f f624f57c f8bbabab`

---

## 4. pass@k 证据(原始数据定位)

全部采样自 **run-011 final**(`tinker://cad6ab5c-35bc-516c-90bd-804b0a6166f5:train:0/sampler_weights/final`),**temperature=0.9, top_p=1.0**。

### 4.1 死区挖矿(2026-06-11 01:03;`/tmp/mine.py` + `/tmp/mine.log` + `/tmp/mine_results.jsonl`)
| 池 | 定义 | K / max_tokens | 收率 | pass@K 分布 |
|---|---|---|---|---|
| eq_dead | solver 失败(train) | 8 / 1600 | **3/166 = 1.8%** | 命中题 4-5/8(会就稳) |
| crypt_dead | solver 失败(train) | 8 / 1600 | **3/400 = 0.75%** | 命中题 1-2/8 |
| bit_tail | 有 trace 但错(难尾) | 4 / 7680 | **6/100 = 6%** | {1:2, 3:2, 4:2} |

⚠️ eq/crypt 挖矿 max_tokens=1600 < 7680,长 trace 被截断,死区收率可能轻微低估;但量级结论(≈1-2%)稳。
**12 条挖到的正确 trace 就存在 `mine_results.jsonl` 里(3 eq + 3 crypt + 6 bit)= 免费 RFT 种子。**

### 4.2 营救臂(2026-06-10 23:54;`/tmp/rescue_test.py` + `/tmp/rescue_test.log`)
- crypt_fail 池(**已训 + solver 会 + 贪心错 = 标准 R-loss 题**):**2/12 = 17%** @K8, max_tokens=2000。这就是 BRIEF 里"营救臂 17%"的原始出处(splice 是会诊方案里的拼接补丁设计,未单独实测)。
- eq_unseen(solver 不会):3/10 = 30% —— 小样本运气,已被全量 1.8% 修正(PATHS 误杀注册表在案)。

### 4.3 配套探针
- `/tmp/probe80_results.json` + `probe80.log`(已训题贪心):eq 30/30、bm 难尾 17/20、crypt 1/30。
- `/tmp/probe_ft.py` + `probe_ft.log`(E4,60 步 LR8e-5 续 run-011 state_ep1):crypt 新格式留出 2/40=5%、bitEV 0/30、锚 32/40=80% → SFT 教不会搜索泛化,且小 FT 伤锚。
- `/tmp/crypt_traces.jsonl`:161 条 forward 格式正确 crypt trace(离线 oracle 为真,贪心驮不动)。

### 4.4 R-loss 题 pass@8 估计
- 唯一直接证据 = crypt R-loss 17%(2/12,95%CI 约 2-48%)。
- bit R-loss **未直接测**,但其近失结构(1-2 bit 差)+ 半死区 bit_tail 6%@K4 → 估 **pass@8 ≈ 15-35%,中心 ~20%**。
- 设计含义:GRPO group K=8 时,R-loss 题中约 1/5 有非零梯度信号 → 要么 K=16-32,要么先 RFT(采样→筛正确→SFT)。
- **解冻后第一笔便宜测量(~$3-4):对 11+21(bit R-loss 外推补完)+26(crypt solver-ok holdout)题做 K=8 采样,直接得 RL 可达上限**;~60 题 × 8 × ~7k tok ≈ 3.4M tok。

---

## 5. marginal 题与 RL 题集圈定建议

### 5.1 各类边际结构
- **bit(92.8% oracle 上限)**:23 个 holdout oracle-dead 经扩展假设空间穷举(`/tmp/bit_headroom.py/.json`):18 无候选、4 truth 在歧义集中、1 唯一可恢复 → oracle-dead 几乎无肉;**肉全在 R-loss 32 题**。
- **eq(80.1%)**:29 失败 = 签名/格式类(`62 vs -62`、`-11 vs 11` 等约 8 题,pass@k 可能不低,是 marginal)+ 怪符号 truth(`/13`、`}51` 等 ~7 题,多数不可 box)+ qop_unseen 量值类(剩余,死区 1.8%)。**marginal ≈ 8-10 题签名类可进 RL 题集试**。
- **crypt(15.8%)**:26 solver-ok = R-loss 全集(模型现实现 0%);139 失败 = 死区(0.75%),其中含 `}` truth 永远零 reward,严禁入集。
- **cipher/gravity/numeral/unit**:R=100%(cipher 已实测 110/110),无 RL 增益空间,只做锚。

### 5.2 三层题集(训练全部取自 train_split,holdout 只做评测)
| 层 | 内容 | 题数 | 角色 |
|---|---|---|---|
| **核心猎物** | bit train R-loss 候选(~100-150)+ crypt solver-ok(116-146,剔 `}` truth)+ eq 签名类 marginal(~30-50) | **~300-350** | K=8-16 rollout,期望 hit 率 6-20%,产生主要梯度 |
| **巩固** | bit/eq 已对题分层抽样(~300) | ~300 | 维持高 hit 率题稳定 group baseline、防策略漂移 |
| **锚** | cipher/gravity/numeral/unit 各 ~100 已对题 + replay 抽样(`data/replay/` 17570 行池) | ~400-600 | KL 锚/replay,防遗忘(E4 教训:60 步 LR8e-5 锚掉到 80%) |

### 5.3 成本锚点
采样 ~$0.0025/rollout(挖矿 2000 rollouts ≈ $5);bit/crypt 单 rollout ~7k tok,cipher ~3k;训练 $0.04-0.077/步(batch 64)。

### 5.4 初始化 checkpoint 取舍
- `run-011 final`(LB 0.85,采样质量最好)——**但训练集含 holdout,holdout 评测对它被污染**;
- `run-012 final`(LB 0.84,train_split only)——**唯一可做真 R 测量的干净底座**;
- 折中:RL 从 run-011 起(它是要提交的底座),评测改用"LB 提交"或接受 holdout 偏乐观;或从 run-012 起、用 holdout 干净闭环,最后把配方搬回 run-011。该取舍需 RL 设计组拍板。

---

## 6. ⚠️ /tmp 抢救清单(机器重启即灭失,建议尽快 cp 进 repo)
`/tmp/mine.py /tmp/mine.log /tmp/mine_results.jsonl /tmp/rescue_test.py /tmp/rescue_test.log /tmp/probe80*.{py,log,json} /tmp/probe_ft.{py,log} /tmp/crypt_traces.jsonl /tmp/oracle_fast.py /tmp/oracle_fast_fails.json /tmp/oracle_crypt.py /tmp/crypt_holdout_fails.json /tmp/bit_headroom.{py,json} /tmp/bit_details.json /tmp/oracle_vs_model.json /tmp/decode012.py`
