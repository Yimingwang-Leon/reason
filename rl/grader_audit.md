# C1 官方判分语义审计报告(G0 闸输入)

日期 2026-06-11 · 脚本 `rl/grader_audit.py` · 数据 `data/run012_holdout_decode.jsonl`(770 行,110/类,run-012 final 贪心@7680)

## 0. 一句话结论

**G0 闸 PASS,维持计划:bit 19 道错题 100% 存活(19/19),11 题 R-loss 猎物 100% 存活(11/11),外推 32 题分母原样成立(11/110×320≈32)。官方语义 vs 我们的判分在全部 770 行上 0 行差异。** 唯一实质风险点是二进制守卫的真伪(§3),证据强烈支持守卫存在,但若不存在则 bit 猎物缩水至 7/11(仍 >50% 存活,G0 也不触红线)。

## 1. 官方语义钉死(对照源码)

| 环节 | 官方实现 | 我们(src/reasoning.py) | 差异 |
|---|---|---|---|
| 提取 | `notebook_tinker.py:434-481 extract_final_answer`:boxed(同正则,容忍末尾未闭合)→ "final answer is:" 文案 → 文中**最后一个数** → 最后一行 | `extract_answer`:仅 boxed,无 boxed 返回 `""` | 官方多 3 级 fallback |
| 数值容差 | `notebook_tinker.py:484-502 verify`:`isclose(rel_tol=1e-2, abs_tol=1e-5)` | 同 rel_tol,`abs_tol=0.0` | abs_tol 仅影响近零 truth,本 770 行 0 行受影响 |
| 二进制守卫 | notebook 版 verify **没有**;`reasoning.py:91-93 compare_answer` **有**(`re.fullmatch(r"[01]+", stored)` → 严格串等) | 有(与 reasoning.py 同) | 见 §3 证据 |
| 串比较 | 大小写不敏感 | 同 | 无 |

审计采用:**官方 fallback 链 + 守卫 ON**(= PLAN §4 条目2 指定口径),同时输出守卫 OFF 敏感性。

## 2. 770 行重打结果

| cat | 我们 ok | 官方(守卫 ON) | 官方(守卫 OFF) | n |
|---|---|---|---|---|
| bit_manipulation | 91 | **91** | 100 | 110 |
| cipher | 110 | 110 | 110 | 110 |
| cryptarithm_deduce | 0 | **0** | 0 | 110 |
| equation_numeric_deduce | 84 | **84** | 84 | 110 |
| gravity / numeral / unit | 110×3 | 110×3 | 110×3 | 330 |
| **TOTAL** | 615 | **615** | 624 | 770 |

**逐行差异(official_ok ≠ 原 ok):0 / 770。**

为什么 fallback 链一行都救不回:全 770 行中**只有 104 行完全没有 boxed,且全部是 crypt cap-hit**(105 cap-hit 中的 104)。这些行官方 fallback 提取到的是 CoT 中最后一个数字(如 '3'、'68'),而 crypt truth 是符号串('\\^?'、':::')——串等永不命中。其余 666 行全有 boxed,两套提取逐字相同。结论:**我们的判分器与钉死后的官方语义在本语料上严格同构;此前所有 R/oracle 记账无需重算。**

## 3. 二进制守卫:证据与敏感性(唯一残余风险)

**证据(守卫存在,采信):**
1. `external/nemotron-huikang/reasoning.py:91-93` 显式实现守卫,且 docstring 给出 `verify("10011000","10011001") → False`、`verify("11011","00011011") → False`。这两例在无守卫的 float 路径下都会判 **True**(相对差 1e-7 / 前导零等值)——docstring 断言 False,即 huikang 对官方 grader 行为的镜像证词。
2. 旁证:run-012 实测 LB=0.84,已低于我们按守卫 ON 口径的外推(0.86-0.87)。若 LB 真无守卫,bit 类还要白送 +9/110(≈+0.47pp 全局),缺口进一步扩大到 ~3.5pp,与所有已知 LB 读数更不自洽。

**敏感性(若守卫不存在):** 19 道 bit 官方错题中 **9 道**会被 float 容差判对(1-2 bit 近失恰落在 rel_tol=1e-2 内,如 '11011111' vs '11111111' 相对差 0.9%)。其中落在 11 题 R-loss 猎物内的是 4 题(053f87d3 / 0e70c867 / 0f7ddd75 / 30ba0cf4),猎物变 7/11(64% 存活)——仍未触 G0 红线(缩水>50% 才砍半),但 RFT 期望收益按比例下修。**该世界线下这 9 题在 LB 上本来就是"对的",猎物消失=分已到手,不是损失。**

## 4. G0 闸答案(PLAN §2.1)

| 问题 | 答案 |
|---|---|
| bit 19 道 holdout 错题经官方重打存活? | **19/19 = 100%** |
| 11 题 R-loss(错∧oracle-ok)存活? | **11/11 = 100%**(id 与 `data/probes/oracle_vs_model.json` 快照逐一吻合) |
| 32 题猎物分母? | **维持 32**(11/110 × 320 ≈ 32,外推系数未变) |
| 判据 ≥24/32 存活? | **PASS(32/32)→ 名单修正后发 M0,维持计划** |

修正后 holdout 猎物池(`data/prey_holdout.json`):

| 层 | 实测(已解码样本内) | 外推全 holdout | 备注 |
|---|---|---|---|
| bit_rloss | 11(/110) | ≈32(/320) | 全部近失、无 cap_hit |
| crypt_solver_ok | 26(全集) | 26 | 15 已解码全错;11 未解码(decoded=false,贪心未测);truth 均可 box |
| eq_rloss | 8(/110) | ≈10.6(/146) | 26 错题中 18 题 oracle-dead(签名/怪符号/qop_unseen) |
| **合计** | **45 实测** | **≈68.6** | 与 PLAN "≈68 题 ≈ +3.6pp 理论上限" 口径一致,无需改 §1.2 |

## 5. train_split 侧候选池(`data/prey_train_candidates.json`,全部"待 M0 直测")

repo production solver 直跑 train_split(本机,$0):

| 层 | oracle-ok 母池 | recon §3.2 预期 | 备注 |
|---|---|---|---|
| bit_rloss | 1185/1282 | R-loss 子集 100-150 | 贪心错子集待 M0 K8 直测 |
| crypt_solver_ok | 121/658(剔 '}'truth 后 113) | ~116 | 与 17.6% production 口径吻合(121/658=18.4%) |
| eq_rloss | 460/586 | R-loss 少(微探针 30/30) | 签名类默认不入池,M0 ≥25% 覆盖才许进 |

## 6. 产出物

- `rl/grader_audit.py` — 审计脚本(可重跑;`--train` 重建 train 候选池)
- `data/prey_holdout.json` — 三层猎物名单(官方语义修正后;M0/R1/G2 闸统一用这份)
- `data/prey_train_candidates.json` — train_split 候选母池(待 M0 直测)
