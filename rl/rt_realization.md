# 红队复查 — R(实现率)角度的免费漏点排查

**复查员视角:LB = oracle × R。团队两天猛攻 oracle(RFT/融合全死),把 R≈0.96-0.97 当局部最优。本报告专查"不改 CoT 内容、靠 R 拿分"的免费手段,并对团队的"R 已封顶"判决逐条上诉。**

数据源:`data/run012_holdout_decode.jsonl`(770 行 run-012 holdout 贪心解码,带 `gen`=token 数、`cap_hit`、全文 `text`)、`src/corpus.py`、`src/eval_gate.py`、`src/reasoning.py`、`baseline.json`、记忆 `project_run006_collapse_diagnosis.md`。全部免费、可复算。

---

## 一、直接结论(先给赔率)

**R 这一侧没有被埋没的真杠杆。** 团队"R≈0.96 是局部最优"的判决,在我能免费验证的范围内**基本成立(B+ 级)**,但有一处**真未测的缺口**值得记账(不值花钱)。把 LB 推到 0.865+ 的概率,单靠 R 侧 ≈ **0.05-0.08**。

三个被追问的具体假设,实测裁决:
1. **"过程对但 box 被 `}` 截断"导致 R 损失** → **证伪**。balanced-box 重提取在 45 个"出框但判错"行上**恢复 0 题**。非 crypt 类**没有**一道正确答案被 `}` 边界悄悄吃掉。
2. **去噪/截断修复有免费分** → **基本证伪**,见下 family A-F。
3. **"CoT 格式重设计伤 R"被过度外推成"任何 R 提升都死"** → **部分成立**:确有一处(bit +238 脏样本)从未被干净 A/B,是真盲区;但"压缩 trace 清 cap"这条具体路其实**已被 run-009 间接测过且失败**。

---

## 二、R-loss 全账(770 行 holdout,按families拆,这是关键证据)

| family | 题数 | 性质 | 是否 R-可救(免费) |
|---|---|---|---|
| A. crypt cap_hit 跑飞循环 | 105/110 crypt | 烧满 7680 token 枚举假设永不收敛,1/105 才到 `\boxed` | **否**:truth 串只在 7/105 循环里出现(且多是短符号巧合),early-stop+box 救≈0 |
| B. crypt 非cap判错 | 5/110 crypt | truth 含 `}`/`)` 外部符号(`%}|`、`!}`、`}}?)`)| **否**:brace/external-symbol 结构 cap |
| C. eq cap_hit 跑飞 | 5 | 枚举 (a-b)%90/%91/%99… 永不收敛 | **否**:qop_unseen 欠定,无 clean rule |
| D. eq 短促自信错 | 20 | 中位 312 token,秒答错 | **否**:oracle miss(不认得该运算)|
| E. eq 符号近错 | 6 | `62` vs `-62`、`46-` vs `46`、`/13` vs `13` | **否**:truth **本身就带**那个怪符号 → 模型漏了"读法/符号变换",是 reasoner 正确性问题,不是 box 提取问题 |
| F. bit 汉明1-2 | 19(12 个汉明1)| 长且自信(6647 token,与对的题 6717 几乎一样),确定性执行出错 | **否**:bit 信息论上限 ~92.8% |

**免费 box 提取层 R 恢复(balanced-box / 重提取)= 0/45。** 这条死得很干净:`extract_answer` 用 `r"\boxed\{([^}]*)(?:\}|$)"` 在第一个 `}` 截断,与 grader 同构;corpus.py 已主动 `if "}" in final_answer: skip`(第 301 行),math_replay 还专门用 balanced_boxed 渲染(第 137-159 行)。截断边界**该堵的洞团队都堵了**。

---

## 三、被追问的三点,逐条上诉裁决

### 追问① "CoT 格式伤 R" 是否被过度外推?
**裁决:方向成立,但一处归因不干净。**
- "压缩 bit trace 清 token cap"这条**具体**路 ≠ 未测。`project_run006_collapse_diagnosis.md` 的 **run-009 行 = "legacy slimmed + augmenters + 2M replay = 0.82"**——slim 过了,掉 2pp。机制:legacy 是逐列局部 copy 脚手架(6700 token),slim 掉脚手架 → 模型失去机械 replay 的依据 → R 掉。run-010 locality-hardening 反方向也 0.82。**两个相反方向都负**,这是 A 级"改 bit 格式=死"。
- **但 run-009 混了 3 个变量**(slim+augmenters+replay),slim 单独归因是 B 级。这是团队判决里唯一不干净的一处。

### 追问② run-011 语料里有没有"过程对/box微瑕"丢分题?
**裁决:没有。** 45 个出框判错行 balanced-box 全恢复失败;6 个 eq 近错的 truth 本身带怪符号(模型读法错,不是 box 微瑕)。corpus 的 self-consistent box 策略(box reasoner 自己的答案、不重写)已经把"CoT↔box 矛盾"这个 R<1 主嫌堵掉了(corpus.py 第 280-296 行注释)。

### 追问③ corpus.py / eval_gate.py 判分与截断逻辑找漏点
**裁决:逻辑干净,无免费漏点。** 三处 GEN_LIMIT(7680)硬门(corpus.py 310/316,replay 203/214)是 **skip-not-truncate**,正确(截断会 strand box)。eval_gate 的 set-based 回归门也对。唯一可议:augmenter 路径(第 386-389 行)用的是 `truncate`(`tokens[:TOKEN_LIMIT]`)而非 skip——但 augmenter 是 drill、不进推理评测,且 p95 长度远低于 cap(matching 248、equation_* 80),**实际 0 行受影响**,非漏点。

---

## 四、唯一真盲区(记账,不值花钱)

**bit_manipulation +238 "faithful-but-wrong" 脏样本从未干净 A/B。**
- 事实:corpus 把 238 个"忠实执行但 box 错"的 bit trace 主动放进训练集(reasoning.py 第 85-96 行 per-category 策略;corpus 1601 bit 行 = 1364 对 + ~238 错)。holdout 上 bit 错的 19 题中 12 个是**汉明1**、且**和对的题一样长一样自信**(6647 vs 6717 token)——这正是"学了一个确定性但错的 bit 规则"的指纹。
- 缺口:run-011(引入 +238)是单 commit f41de29 一次性混入 crypt-22.2%-port + eq 枚举 trace + legacy 复原 + champion drills + 门加固 **5+ 项**,run-011=0.85 vs run-005=0.84 **无法归因到 +238**。"+238 提 R"是 dual-audit 的**裁定**,非实测。
- **但这不是花钱的理由**:(a)bit 上限信息论 92.8%,+238 即便全错也只摸 ~1-2 题边际;(b)纯免费验证不了(R 只有 LB 才显形,团队 BRIEF 反复强调);(c)要测必须烧一次提交配额——而眼下配额该留给差异化终选签。**记为"未测缺口",不升级为路线。**

---

## 五、给真实赔率

- **R 侧把 LB 推到 0.865+ 的概率 ≈ 0.05-0.08。** 理由:免费可恢复 R = 0;唯一未测缺口(bit +238)边际 ≤1-2 题(<0.1pp),且摸不到 +1.5pp 缺口。
- **团队判决"R 是局部最优"——我支持,但把证据等级从隐含的 A 降到 B+**:其中"压缩 trace 清 cap"是 A 级死(run-009/010),"+238 提 R"是 C 级裁定(从没单测)。两者都不构成花钱的理由。
- **真正的丢分仍在 oracle 洞**:crypt 105 题跑飞(oracle-dead,不是 R)、eq 20 题 qop_unseen(oracle-dead)。这两块团队已在 PATHS 判 A 级死(crypt closed-form ≤22.5%、eq qop_unseen 0/115)。**R 角度无法绕过这两个 oracle 硬顶。**

**一句话给团队:别再往 R 上找钱花。截断/去噪/box 边界全部干净,免费恢复=0。唯一没测的是 bit +238 的去留,但它够不到 0.865,留着配额做差异化终选签更值。**
