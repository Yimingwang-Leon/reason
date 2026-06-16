# eq 运算符先验反学习(监督版 E2 复盘)— 判决:无杠杆

日期:2026-06-12 | 代码:`rl/eq_prior.py` | 铁律 $0,纯本地实测

## 任务与方法

E2 用手工排序改 tiebreak 得 train +2 / holdout +0(过拟合)。本实验改为监督学习:
用 train_cat eq 真值统计出题器在歧义(decisive)题上实际选的规则,学 P(rule) 先验,
按后验 argmax 重排 survivors。隔离:先验只在 train586(= 732 eq − 146 holdout)上学,
holdout110(run012 decode 口径,模型基线 84/110)+ holdout146 纯评测;另做 train 内
5-fold CV 估真泛化。

数据结构(train586 / ho110):decisive(多答案歧义,tiebreak 唯一起作用的地方)
127 / 28 道;single_answer 346 / 65(怎么选都对);qop_unseen 113 / 17(硬上限,0 可救)。

## 主结果表(oracle 道数)

| 先验变体 | train586 (base 434) | CV/586 (base 434) | ho110 (base 88) | ho146 (base 113) |
|---|---|---|---|---|
| P1 全局规则频率 | 443 (+9) | 443 (+9) | 87 (−1) | 113 (0) |
| P2 family+mode 粗桶 | 435 (+1) | — | 87 (−1) | 113 (0) |
| P3 精度先验 wins/survives | 444 (+10) | 438 (+4) | 86 (−2) | 112 (−1) |
| P4 按 n_ex 分桶 | 441 (+7) | — | 86 (−2) | 113 (0) |
| op 类型桶 (arith/symbol) | 445 (+11) | 444 (+10) | 87 (−1) | 114 (+1) |
| 数值范围桶 (<100/≥100) | 443 (+9) | 443 (+9) | 87 (−1) | 113 (0) |
| 符号约定桶 (fmt) | 440 (+6) | 440 (+6) | 87 (−1) | 111 (−2) |
| **P5 精确 op 字符桶 (n≥15)** | **444 (+10)** | **442 (+8)** | **89 (+1)** | **114 (+1)** |
| 完美 tiebreak 天花板 | 469 | — | 93 | 121 |

- **E2 教训系统性重演**:8 个变体在 train 全正向(+1~+11),holdout110 上 7 个 0~−2。
- **P5 是唯一双 holdout 正向且零 BREAK 的变体**:唯一翻转 = FIX `16699d43`
  (q=67+57,'+' 字符桶学到"'+' 出题时几乎不会是列式带符号减法",把 sub(T,T) 压给
  absdiff(F,F)),恰好是模型 26 道错题之一;bootstrap(200 次 train 重采样)同向
  175/200 = 87.5%。**但 +1/110 = 单题,统计上在噪声水平。**

## 三个结构性发现

### 1) 基线污染:现有 tiebreak 本身就是 holdout 调出来的
`equation_numeric_deduce.py` L130 注释自认 `_MODE_ORDER` 是 "learned from the holdout"
(commit 0f0d3cf "equation 74->80%")。实测 decisive 子集基线命中:**train 69.3%
(88/127) vs holdout 82.1% (23/28),+12.8pp** —— holdout 上的 88/110 solver oracle 是
向 holdout 过拟合的乐观数。鲜数据(LB 测试集)上 decisive 命中应接近 train 的 69-77%,
此时 CV 口径 P5 ≈ +8/586 ≈ eq-cohort oracle +1.4pp 是真实的,但量级太小(见下)。

### 2) headroom 5 道解剖:4 道被 train 统计反向压制,频率先验原理上不可恢复
base 88 → ceiling 93 的 5 道(全部 base 选了 sub(T,T),train 胜场 58 = 最高):

| 题 | 真规则 | 真规则 train 胜场 | 可学? |
|---|---|---|---|
| 16699d43 67+57 | sub/absdiff plain → 10 | 29-30 | **是(op字符条件,P5 已修)** |
| 7ac90433 48-25 sym_prefix | negabsdiff(T,T) → −23 | 29 | 否(条件统计 50:4 反向) |
| 891942ba 85-88 | absdiff(T,T) → 03 | 27 | 否(58 vs 27) |
| 55f19327 77:38 | maxmodmin/mod plain → 1 | 2-17 | 否(58 vs 17) |
| 4cb5e927 69-49 sym_prefix | negabsdiff(F,F) → −20 | 16 | 否(条件统计反向) |

### 3) E2 的 sign-convention 方向被监督数据正式否决
train 上「fmt≠num 且 survivors 顶部 = sub/revsub」时真规则 signed:abs = **50:4**,
先验必须选 signed(基线已如此);ho110 的 2 道 abs 例外正是出题器随机尾部。
E2 的手工 abs 偏置在 train 上就该输,holdout +0 不是运气差,是方向错。

## 对模型分 (84/110) 的含义

- 模型 26 道错题中 solver 能给对答案的:base/P1 = 8 道,P5 = 9 道 —— 换先验几乎不
  改变营救集。把 solver oracle 变成模型分需要付费重训;eq cohort(732/8228 ≈ 8.9%)
  × oracle +1.4pp(CV 口径)≈ **LB ≤ +0.1pp**,远低于重训成本噪声。
- eq 完美 tiebreak 天花板 = 93/110 solver-oracle;模型 84 vs solver base 88 的缺口
  主要在执行(R)而非 oracle。**别人的 0.87 不在 eq 先验里。**

## 判决

- `lever_real = false`。监督先验在 train 内可证泛化(CV +8~10/586),但 holdout110
  实测净增 +1(单题、噪声水平),headroom 的 5 道中 4 道被 train 统计反向压制,
  任何按频率学的先验都注定选错 —— 这是出题器随机性,不是先验缺失。
- 免费顺手项:P5(op 字符桶,n≥15 回退全局)是零回归微改进,**下次任何已付费重训
  时可顺带并入语料 solver**,单独为它花钱不值。
- 主攻力应回到 bit / crypt(oracle 洞在那边)。
