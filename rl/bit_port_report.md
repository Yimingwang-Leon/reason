# bit 扩展规则族移植报告(M0,$0 本地)
**日期:2026-06-12 | 任务:把 rl/bit_prior.py 已验证规则族+先验裁决移植进 src/reasoners/bit_manipulation.py 并重生成 train_split bit 语料 | 全程未触 tinker/modal/kaggle**

## 1. 移植内容(单版本原地改)

- `src/reasoners/bit_manipulation.py`:EV 死格式段(whole-register 扫描,探针 0/30,PATHS.md 死亡名单)整段删除,替换为 `_ExtEngine`(bit_prior.py 1:1 移植:24 基原子 + depth≤2 复合 + 8 布尔二元 + nested op2(op1(a,b),c) + MAJ/MUX 及取反 + gf3 fallback,同成本、同 dedupe、同评分)+ `_reasoning_prior_family` 逐位发射器。
- 先验:`data/train_split.csv` 1282 道 bit 上在线学习 canonical 规则分布(687 签名 / 1281 题),**无 holdout 泄漏**(holdout id 不在 train_split)。裁决 = 先验分数排序后**优先取真值一致候选**(真值-box 语料哲学);族不覆盖时取先验首选(self-consistent,沿 run-011 bit 收录政策)。
- 格式安全:winner 先投影成逐位局部形式 `out_i = f(I/NOT/C 叶子)`(8 布尔 op 经 De Morgan 折叠到 {XOR,AND,OR}+叶子取反;NMAJ/NMUX 折叠进叶子),按 legacy 视觉几何发射:Output/Input 逐例逐位块 → 列表(含新增 Input bit columns)→ 逐位 `Matching output` 行(每行用打印列重算一条输出列,代码强制校验列相等才发射)→ Selected → 逐位 Applying → `\boxed{}`。**全文无任何 whole-register 规则断言**(rule=X 先行的 run-006/7/8 死格式被结构性排除)。gf3 额外有 Cells 表(置于 Matching 之后,先证据后表)。

## 2. 回归(R 安全核心保证)

| 检查 | 结果 |
|---|---|
| 抽样 20 道老族题新旧输出字节对比 | **20/20 identical** |
| 全量 1086 道 legacy-correct 题字节对比 | **1086/1086 identical,0 改动** |
| 移植前磁盘 trace 与 HEAD 代码同步抽检 | 31/31 一致(基线干净) |

legacy 代码路径与 `legacy 正确 → 原样返回` 闸门均未动,字节一致是结构保证 + 全量实证。

## 3. 重生成验收(train_split 1282 道 bit;注:任务书写 ~1492 是 train_cat 去 holdout 口径,train_split 实际 1282)

| 指标 | 实测 | 验收线 |
|---|---|---|
| boxed==truth | **1281/1282 = 99.92%** | ~99.8% ✓ |
| completion token p95(全量) | **7259** | <7400 ✓ |
| completion token p50/p99/max | 6609 / 7387 / 7684 | max 为 1 道既有 legacy 超长 trace(corpus.py 会按 GEN_LIMIT 7680 丢弃,移植前已存在) |
| 新族 trace token p50/p95/max | 1878 / 2429 / 2562 | 远短于 legacy,无截断风险 |

- 路径构成:legacy 1086 + 新族 196(= 旧解题器 95 道 EV 题 + 101 道 legacy 错题,全部落入预计 150-250 区间)。
- 新族 196 题规则层级:nested 49、MUX 44、MAJ 42、gf3 36、pair(深复合)24、unary 1。
- 唯一错题 `ccf02dee`:族不覆盖(dig_bit.md 已知 3 残差之一),发射先验首选、self-consistent 错 box(政策内)。
- holdout 题的 reasoning/*.txt 未触碰(R 测量面干净)。

## 4. 语料产出

**a. `data/rft/bit_new_traces.jsonl`** — 196 行,只含新族题。行格式与 rl/rft_filter.py 输出一致:`{problem_id, category, sample_idx, tokens, mask, text, n_prompt_tok, n_gen_tok}`;tokens = `src.corpus.tokenize_prompt`(chat template,enable_thinking=True)+ completion `{trace}\n</think>\n\boxed{ans}<|im_end|>`(tokenizer.json, add_special_tokens=False),prompt mask=0 / completion mask=1;总长 max 2859 ≤7680。已 decode 抽检:prompt 段以 `<think>\n` 收尾、completion 段以 `</think>\n\boxed{...}<|im_end|>` 收尾、mask 边界正确。

**b. `data/rft/bit_corpus_mix.jsonl`** — 500 行 = 196 新族 + **304 镇纸(60.8% ≥60%)**,seed=0 确定性抽样后打乱:
| 镇纸成分 | 行数 | 说明 |
|---|---|---|
| corpus_r1 crypt | 59 | 全收 |
| corpus_r1 eq | 119 | 全收 |
| corpus_r1 bit | 66 | 排除与新族重叠 11 行、超长(>7680)2 行后抽样 |
| legacy 老族新生成 trace | 60 | 从 1086 legacy-correct 抽样,corpus.py 同配方 tokenize |

全 500 行通过校验:总长 ≤7680(max 7620,p95 7375)、len(tokens)==len(mask)、mask 含监督段;trainable tokens 合计 1,387,324。

## 5. 判定

**可以开训** —— 回归零破坏(legacy 字节不变)、新族 oracle 兑现(99.92% 真值-box)、格式沿用 R≈0.95 的逐位局部几何且结构性禁断言、长度全部入闸。但按 HOW_OTHERS.md 纪律:**先跑 G1 免费 30 题本地 R-probe(16 族外 npred=1 + 5 多候选 + 9 旧族对照),R≥0.8 才进入付费 D1;任何 tinker/Modal 训练前需用户显式授权。**

## 文件清单
- 改:`src/reasoners/bit_manipulation.py`(EV 段 → 扩展族引擎+逐位发射器,docstring 同步)
- 重生成:`reasoning/<id>.txt` ×1282(其中 196 道换新族 trace,1086 道字节不变;改动前全量备份 /tmp/bit_backup)
- 新增:`data/rft/bit_new_traces.jsonl`(196 行)、`data/rft/bit_corpus_mix.jsonl`(500 行)、本报告

---

# 第二版:全长枚举(Option A,2026-06-12,$0 本地)

**背景(run-009 验尸,data/probes/r3_bit_exam.jsonl 可复核):** v1 短格式新族 trace(completion p50 1878 / p95 2429 tok)与模型已学的 legacy bit trace(中位 ~6600)长度/结构分布割裂 = 统计上的"格式瘦身"信号,实训打崩模型(110 道判卷 94→51,老题也开始写 1.5k 残缺枚举、box 32 位乱码)。本版按 Option A 重做:新族 trace 复刻 legacy 完整候选扫描结构,全长发射。

## 1. v2 trace 结构(单版本原地改 src/reasoners/bit_manipulation.py)

- **纯重构:**`_reasoning_legacy` 拆出共享前缀 `_legacy_prefix`(到 "Selecting" 摘要为止);legacy 路径输出零字节变化(§2 全量实证)。
- **新族 trace = legacy 前缀逐字节复刻 + 同循环格式扫描延续:**
  1. Output 例块 → Output bit columns → Input 例块 → 9 个 per-column 候选段(Identity→XOR-NOT,每段 raw records + per-output-column match/否决 + Left/Right runs + Best)——与老 trace 同一份代码生成,视觉逐块同构;
  2. 在 legacy 原 "Selecting" 分叉点改接 `Continuing the scan with whole-byte rules.` + 2 行原子/op 图例,继续扫 whole-byte 层(**Unary→Pairs→Nested→MAJ→MUX→F**):每个展示候选 = label 行 + 逐 example 真实重算验证行(操作数寄存器值 → 结果 vs 实际 Output k),首个 mismatch 标 `x` 否决;gf3 否决展示冲突 cell;
  3. **winner 完整全例验证**(7-10 行全 `ok` + `match`;gf3 先列 Cells 表再逐例验证);
  4. per-bit Applying 推导(每行自带局部规则名,query 逐位算出)+ `The answer is \boxed{}`。不再发独立 Selected 块(Applying 行已重复每位规则,省 ~90 tok 用于对齐长度带)。
- **长度控制:**否决候选数逐题按 token 预算定(每层至少 1 个、round-robin 填充到 ≥6000);发射上限 7300(legacy 语料自身 p95=7259,band 尾 7200-7300 即 legacy 自己的尾部形状;p95<7400 硬线由 `_EV_MAX_TOKENS=7400` 兜底);超预算瘦身阶梯:去图例 → winner 行去操作数列(验证仍全例)→ 去否决(否决证据最后才砍)。

## 2. 回归(R 安全核心保证)

| 检查 | 结果 |
|---|---|
| 全量 1086 道 legacy-correct 题重生成字节对比 | **1086/1086 identical**(重构前另抽 40+150 道两轮独立验证均 0 mismatch) |
| holdout reasoning/*.txt | 未触碰 |

## 3. 重生成验收(196 道新族,真值全部已知)

| 指标 | 实测 | 验收线 |
|---|---|---|
| boxed==truth | **195/196**(唯一例外 ccf02dee,见下) | 100% |
| completion tok p50 | **6772** | 5500-6800 ✓ |
| completion tok p95 | **7274** | <7400 硬线 ✓(legacy 自身 p95 7259) |
| min / max | 6017 / 7349 | ≥5000 / ≤7400 ✓(29 道落 7200-7349,与 legacy 尾部对齐) |
| winner tier 构成 | nested 49 / MUX 44 / MAJ 42 / gf3 36 / pair 24 / unary 1 | 与 v1 裁决逐题一致 ✓ |

- **ccf02dee:**扩展族不覆盖(dig_bit.md 已知 3 残差之一),任何一致候选都推不出真值;按 run-011 政策发射先验首选 self-consistent 错 box 的 v2 全长 trace(留在 reasoning/,**不进训练语料**)。语料内新族行 boxed==truth = 100%。
- **独立行校验:**196/196 道 trace 的全部扫描行(label、操作数值、结果、vs 列、ok/x 标记、match 行)用独立重写的原子/op 语义(不复用引擎表)复核,**0 错误**。
- **人工抽检 3 道**(0245b9bb nested / 000b53cf MUX / 0ec17d2e gf3):前缀与 legacy 老 trace 逐块同构(同代码),扫描循环 label行+逐例验证行 的视觉几何与 legacy 候选段一致,winner 全例 ok+match,Applying 逐位推导收尾。
- **预算瘦身残量(最大前缀题):**47 道无图例行、28 道 winner 行为值-only(验证仍全例)、4 道零否决;核心骨架(legacy 前缀→pivot→层扫描→winner match→Applying→boxed)全 196 道一致。

## 4. 语料产出(单版本覆盖)

**a. `data/rft/bit_new_traces.jsonl`** — 196 行,v1 同 schema(`{problem_id, category, sample_idx, tokens, mask, text, n_prompt_tok, n_gen_tok}`),tokens = chat-template prompt(mask=0)+ `{trace}\n</think>\n\boxed{ans}<|im_end|>`(mask=1),decode 抽检 prompt 段 `<think>\n` 收尾、completion 段 `</think>\n\boxed{...}<|im_end|>` 收尾。

**b. `data/rft/bit_corpus_mix.jsonl`** — **723 行**,seed=0 打乱:
| 成分 | 行数 | 说明 |
|---|---|---|
| 新族全长 trace(boxed==truth) | 195 | ccf02dee 剔除 |
| legacy 老族新生成 trace | 350 | 从 1086 legacy-correct 随机(seed 0),boxed 全对、≤7680 |
| 非 bit 镇纸(corpus_r1) | 178 | crypt 59 + eq 119 全收;**corpus_r1 非 bit 行只有 178,凑不满任务书的 200**(差额 22) |

全 723 行校验:len(tokens)==len(mask)、mask 为 0…0/1…1 连续段、completion ≤7680、总长 ≤8192;trainable tokens 合计 **3,837,790**(v1 为 1.39M,全长化 ×2.8)。

## 5. 判定

**可以开训**:legacy 零字节破坏、新族 195/196 真值 box(语料内 100%)、长度分布与 legacy 对齐(p50 6772 vs legacy 6609,p95 7274 vs 7259)、扫描数据 196/196 独立复核无误。结构上是"扫描-否决-命中-全例验证",非 run-006/7/8 的"先断言后验证"死格式;但这是该格式首次实训,纪律不变:先免费本地 R-probe(新族族外题 + 旧族对照),任何 tinker/Modal 付费训练前需用户显式授权。

## 文件清单(v2)
- 改:`src/reasoners/bit_manipulation.py`(_legacy_prefix 重构 + scan_pool 候选池 + v2 全长发射器,docstring 同步)
- 重生成:`reasoning/<id>.txt` ×196(v1 短 trace 备份 /tmp/bit_v1_196_backup;1086 道字节不变)
- 覆盖:`data/rft/bit_new_traces.jsonl`(196 行)、`data/rft/bit_corpus_mix.jsonl`(723 行)、本报告追加

---

# 第三版:同构混入(2026-06-13,$0 本地)

**改版依据(两轮实训验尸,data/probes/r3_flips.json / r3b_flips.json):** v1 短 trace 长度塌缩(110 判卷 94→51,43 跌 0 涨);v2 全长+第二阶段标记,模型 0/110 执行第二阶段,反在 5 道 ~7k 长老题上过度扫描撞截断(94→88,2 涨 8 跌)。教训:**任何"额外阶段/标记"都学不进贪心,"何时扩展"的判别教不会。** v3 把判别问题消灭:删除第二阶段概念,扩展族候选作为普通候选段裸混进唯一的扫描循环。

## 1. v3 trace 结构(单版本原地改 src/reasoners/bit_manipulation.py,只动 _reasoning_prior_family 装配层)

- **删除:**"Continuing the scan with whole-byte rules." 行、2 行 Atoms/Ops 图例、6 个 tier 头(Unary/Pairs/Nested/MAJ/MUX/F)——v2 引入的全部新结构标记清零。
- **扩展族候选 = 普通候选段:**紧跟第 9 个 legacy 段(XOR-NOT 的 Best 行+空行)直接续流,每段 = label 行 + 逐 example 真实重算验证行(`k [操作数寄存器…] -> 结果 vs 实际输出 k ok/x`),首个 mismatch 标 `x` 否决即止;段间无空行、无任何头部,与 legacy 候选检查同一视觉几何。顺序 cheap-first(unary→pair→nested→MAJ→MUX→gf3),**命中 winner(全例 ok + `match` 行)即停止扫描**,随后逐位 Applying 推导 + `The answer is \boxed{}`,与 legacy 收尾几何逐字节同构。gf3 winner 段内保留 Cells 表(label 后、验证行前;表内容即 label 的 F[tt],为 Applying 的 F 规则提供推导根据)。
- **长度纪律(用否决候选数调节):**_EXT_CAP 7300→7250、FILL_MIN 6000→6250、FILL_MAX 7100→7000、硬顶 _EV_MAX_TOKENS 7400→7349;瘦身阶梯只剩两级(winner 行去操作数列→去否决候选,否决证据最后才砍)。

## 2. 回归 + 重生成验收(train_split 1282 bit)

| 检查 | 实测 | 验收线 |
|---|---|---|
| 老族全量字节回归 | **1086/1086 identical,0 diff** | 字节不动 ✓ |
| 新族 boxed==truth | **195/196**(唯一例外 ccf02dee,已知族外残差,self-consistent 错 box,不进语料) | ✓ |
| completion tok p50 | **6670** | 6200-6700 ✓ |
| completion tok p95 | **7221** | ≤7250 ✓ |
| min / max | 6251 / **7337** | max ≤7349 ✓ |
| winner tier 构成 | nested 49 / MUX 44 / MAJ 42 / gf3 36 / pair 24 / unary 1 | 与 v1/v2 裁决逐题一致 ✓ |
| 残留 v2 标记(全 reasoning/*.txt grep) | **0** | ✓ |

- **独立行校验:**196 道 trace 扫描区共 **1075 个候选段 / 2524 条展示行**,用独立重写的原子/op/MAJ/MUX/gf3 语义(不复用引擎表)逐行复核 label、操作数寄存器值、中间值链、结果、vs 列、ok/x 标记、否决止于首 mismatch、winner 全例 ok+match、gf3 Cells 表与 cell 冲突,**0 错误**。
- **结构机器证明:**解析器强制扫描区每一行 ∈ {候选 label, 验证行, gf3 Cells 行, match},无空行、无其它行形——"无任何新结构标记"不止人工抽检,是全 196 道的解析级保证。
- 预算瘦身残量:27 道 winner 行值-only(验证仍全例)、5 道零否决(最大前缀题);holdout reasoning/*.txt 未触碰。

## 3. 人工抽检 3 道(与 legacy 老 trace 00066667 并排逐块对齐)

块图对齐:两者共享 `header→Output 例块→Output bit columns→Input 例块→When matching output→Identity→…→XOR-NOT`(同代码字节同构);分叉点 legacy 走 `Selecting→Lefts→Rights→Matching→Perfect match→Matched→Selected→Applying`,v3 在同一位置直接续扫候选段到 winner,`Applying→Input→Output→The answer is \boxed{}` 收尾两者几何一致。

- **0245b9bb(nested):**接缝 `Best: none` 空行后直接 `id` / `AND(id,nt)` / … 候选段;winner `XOR(ORN(sr3,rl2),sl4)` 8 例验证行全 ok(`0 ORN 00010011 01111110 -> 10010011 XOR 10010011 11110000 -> 01100011 vs 01100011 ok`)+ match。
- **000b53cf(MUX):**接缝同构;winner `MUX(sl1;sr1,rl2)` 9 例全 ok + match;Applying 逐位 `2 MUX(I3;I1,I4) = MUX(0;1,0) = 0`。
- **0ec17d2e(gf3):**winner `F[…]` 段内 Cells 表后 7 例验证全 ok + match;Applying 逐位 `0 F(I2,I4,C0) = F(0,1,0) = 0`。
- 三道扫描区均无 "Continuing"/图例/tier 头等任何新标记行。

## 4. 语料产出(覆盖)

**a. `data/rft/bit_new_traces.jsonl`** — 196 行,schema 同前(`{problem_id, category, sample_idx, tokens, mask, text, n_prompt_tok, n_gen_tok}`),n_gen_tok max 7337。

**b. `data/rft/bit_corpus_mix.jsonl`** — **1073 行**,seed=0 打乱,行 schema `{problem_id, category, tokens, mask}`:
| 成分 | 行数 | 说明 |
|---|---|---|
| v3 新族(truth-box) | 195 | ccf02dee 剔除 |
| legacy 老族新生成 | **700** | seed=0:**350 道从 completion 最长 1/3(6905-7633 tok)抽**——用长老题重新锚定"命中即停"(v2 的 5 道截断死全是 ~7k 长老题);350 道从其余 2/3 抽;a5749dc0(7684>7680)按 GEN_LIMIT 剔除 |
| 非 bit 镇纸(corpus_r1 非 bit 全收) | 178 | crypt 59 + eq 119 |

全 1073 行校验:len(tokens)==len(mask)、mask 连续 0…0/1…1、completion ≤7680、总长 ≤8192;decode 抽检 prompt 段 `<think>\n` 收尾、completion 段 `</think>\n\boxed{…}<|im_end|>` 收尾;trainable tokens 合计 **6,237,495**;mix completion p50 6684 / p95 7265 / max 7633(尾巴是合法长 legacy)。

## 5. 判定

**可以开训** —— 判别问题已结构性消灭(新族 trace 里不存在任何"第二阶段"可拒绝执行,扫描区行形与 legacy 同一词汇表);legacy 零字节破坏;新族语料 100% truth-box;长度三指标全部入闸(p50 6670 / p95 7221 / max 7337);2524 条展示行独立复核零错。纪律不变:先免费本地 R-probe,任何 tinker/Modal 付费训练前需用户显式授权。

## 文件清单(v3)
- 改:`src/reasoners/bit_manipulation.py`(装配层删标记/图例/tier 头 + 长度常量收紧,docstring 同步;legacy 路径未动)
- 重生成:`reasoning/<id>.txt` ×196(v2 备份 /tmp/bit_v2_196_backup;1086 道字节不变)
- 覆盖:`data/rft/bit_new_traces.jsonl`(196 行)、`data/rft/bit_corpus_mix.jsonl`(1073 行)、本报告追加
