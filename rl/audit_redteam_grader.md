# 红队2 审查报告 — 判分对齐(grader_audit_full.py / GAP_DIAGNOSIS 断言2)

**独立审查人 · 2026-06-13 · 铁律 $0(只读代码/数据/已落盘解码,本机复跑脚本,无 tinker/modal/提交)**
被审对象:`rl/GAP_DIAGNOSIS.md` 断言2「判分口径虚高=0pp(770 行×4 官方判分变体重打,我们对∧官方错=0/770)」+ 支撑脚本 `rl/grader_audit_full.py`。
攻击面:独立复跑脚本,不信 0/770;核对"官方判分语义"是否真来自公开 notebook/官方 harness;攻击语料覆盖盲点(科学计数法/前导零/分数/负号/单位)。

---

## 0. 一句话裁决

**断言2「判分=0pp」SOUND(成立)。** 脚本提取/判分逻辑逐字符核对均**真来自公开文件**(huikang notebook_tinker.py / reasoning.py、agi087.ipynb、ultimate-sft-grpo-v3.ipynb),非作者脑补;0/770 我本机复跑复现,且用一个脚本里**没用到**的、最权威的官方复刻(huikang `reasoning.compare_answer`,自称 "matching metric_reference",带 binary 守卫+数值容差)独立重打,离线虚高**仍是 0/770**。健康类/eq 在所有官方口径下都不掉。**但脚本设计有 3 处不致命瑕疵 + 2 个真实盲点未在诊断里点明**(见 §4),它们不推翻 0pp 主结论,但让"0pp"的适用边界比诊断写得更窄。

---

## 1. 官方判分语义来源核对(逐字符,默认怀疑→证实)

脚本声称 4 个判分变体的语义"逐字搬运"自公开文件。我逐一比对源文件:

| 脚本函数 | 声称来源 | 核对结果 |
|---|---|---|
| `extract_official` (行52-76) | huikang `notebook_tinker.py:434-481 extract_final_answer` | ✅ **逐字一致**(boxed正则 `\\boxed\{([^}]*)(?:\}\|$)` → "final answer"四 patterns → 末数 `-?\d+(?:\.\d+)?` → 末行)。源文件实测 434-481 行 |
| `v2_nb_noguard` (行91-97) | huikang `notebook_tinker.py:484-502 verify` | ✅ **逐字一致**(float isclose rel_tol=1e-2 abs_tol=1e-5,否则 lower() 串等,**无 binary 守卫**)。源文件实测 484-502 行 |
| `v4_ult_norm`+`_normalize_answer` (行111-135) | ultimate-sft-grpo-v3 `_answers_match`/`_normalize_answer` | ✅ **逐字一致**。源 ipynb cell 9 实测:int-cast normalize → 串等 → lower 串等 → rel_tol 1e-2、g==0 时 abs<1e-6 |
| `extract_ours`/`v1_ours` (行41-88) | 本仓 `src/reasoning.py:38/46` | ✅ 一致(boxed-only;binary 守卫;abs_tol=0.0) |
| `v3_rs_guard` (行100-108) | 本仓 `reasoning.py:91-93` | ✅ 守卫+容差(abs_tol=1e-5) |

**额外的独立佐证(脚本没用,我自己加的第 5 个判分器)**:huikang `reasoning.py:69 compare_answer` 自带 docstring 注释 "Extract … matching metric_reference",且 doctest 直接给出官方语义示例:`verify("24.64","24.6401")=True`(数值容差)、`verify("XLVII","xlvii")=True`(文本大小写不敏感)、`verify("11011","00011011")=False`(binary string-exact 守卫)。这是第三方对**官方 metric_reference** 的复刻,带 binary 守卫+rel_tol=1e-2 容差,与脚本 v3 语义一致。

**官方判分权威出处**(MEMORY `project_grader_tolerance` 查证):`kaggle.com/code/metric/nvidia-nemotron-metric`,规则 = **完全字符串匹配 OR 相对容差 1e-2**(OR,任一满足即对)。数值容差分支**确实存在**,不是作者编的。

**裁决:脚本判分语义来源真实、核对逐字符通过,不是"脑补的一套"。**

---

## 2. 复跑结果(不信 0/770)

`python3 rl/grader_audit_full.py` 本机复现:

```
[完整性] decode.truth vs holdout.answer 不一致: 0/770
[完整性] decode.ok vs 重算 v1_ours(extract_ours(text)) 不一致: 0/770
表1 TOTAL  ours=615  v1=615  v2_noguard=624  v3_guard=615  v4_ult=624 / 770
表2 离线虚高(我们对∧官方错): 全 0
表3 官方更宽(我们错∧官方对): bit +9 (仅 v2_noguard / v4_ult,即无守卫口径)
crypt 官方变体救活: 0
离线虚高总行数: 0/770
```

**0/770 复现成立。** 健康类(cipher/gravity/numeral/unit)与 eq 在表2全部 0,即官方任何口径都不会把它们判成虚高。

**独立交叉验证(我自写,脚本里没有这条):** 用 huikang `reasoning.compare_answer`(官方 metric_reference 复刻,带 binary 守卫)+ huikang 官方提取器重打 770 行 → 逐类 615 vs 615 完全一致,**离线虚高仍 0/770**。
- cipher 文本答案:pred 与 truth **string-exact 110/110**(大小写、空格无隐患)。
- 用 agi087 的**真实**提取器(脚本错配成 huikang 版,见 §4.1)重打:8 行提取不同但判分结果 100% 一致,**agi087 真口径下虚高仍 0/770**。

**裁决:断言2「判分=0pp」CONFIRMED。三套独立官方复刻(huikang notebook verify、huikang reasoning.compare_answer、agi087/ult)全部给出 0 虚高。**

---

## 3. 关键反驳:我先怀疑的 string-exact 攻击不成立

我最初的攻击假设:gravity/unit 的 truth 是 `'19.00'/'27.90'` 等带尾随零浮点,而模型 pred 是 `18.987/27.873`,**string-exact 命中率 = 0/110**(实测,gravity 0/110、unit 0/110)。**若官方判分是 string-exact-only,健康类会从 100% 崩到 0%。**

**这个攻击被证据反驳:** 官方规则是 **string-exact OR rel_tol=1e-2(OR)**,容差分支存在(三处独立证据:kaggle metric 页、huikang compare_answer doctest `verify("24.64","24.6401")=True`、agi087 verify)。gravity/unit 全 110 道靠 `abs(18.987-19.00)/19=7e-4 ≤ 1e-2` 判对,**合规**。诊断在"健康类靠容差判对"这点上不是漏洞。**我找不到反例。**

---

## 4. 盲点与瑕疵(不推翻 0pp,但缩小其适用边界)——诊断该补的话

### 4.1 [瑕疵·不致命] 脚本的 `extract_official` 不是 agi087 的真实提取器
脚本注释说 v2/v4 代表 "agi087",但 `extract_official` 用的是 **huikang** 版(带 "final answer" 四 patterns)。agi087 真实 `extract_final_answer`(ipynb cell 2)是:`re.findall(r'\\boxed\{([^}]*)', ...)`(**无 `(?:\}\|$)`**)→ `matches[-1].strip()`(取最后一个,**不过滤空串**,也**无 "final answer" patterns 层**)→ 末数 → 末行。两者在 770 行上 8 处提取不同。**但**:我用 agi087 真版重打,8 处判分结果全一致,虚高仍 0/770。→ 误标来源,但不影响结论。

### 4.2 [瑕疵·不致命] v4 是 GRPO 训练 reward,提取器与判分器被混搭
ult 的 `_answers_match` 原生配对的提取器是 `_extract_boxed`(`\\boxed\{([^}]*)\}`,**要求闭合 `}`**),脚本却喂 huikang 的 `extract_official`。即脚本测的是"判分器语义"而非 ult 整条 pipeline。且这些 kernel 全是各队**自己的训练 reward / 推理 verify**,**没有一个是竞赛官方 harness 本体**——官方 harness 未在本仓落盘,4 变体是对官方语义的间接复刻。0pp 的最强支撑其实是 huikang `reasoning.compare_answer`(自称 matching metric_reference),不是这 4 个 kernel reward。诊断把它们统称"官方判分变体"略过强。

### 4.3 [真盲点·诊断未点明] binary 守卫对 eq 的 collateral —— `truth='001'`
全 1899 holdout 中**有 7 道 eq 的 truth 命中 `[01]+` 守卫**(`'10','1','001','11'…`)。其中 **`truth='001'` 是唯一一处两口径分歧点**:
- 带守卫(我们 v1/v3、huikang compare_answer):模型必须逐字符 box `001`,box `1` 判**错**。
- 无守卫(huikang notebook verify、ult):`float('001')=1`,box `1` 判**对**。

实际 770 解码里,模型恰好 box 了 `001`(逐字符对),两口径都对 → **本语料没触发损失**。但这意味着:**(a)** 0pp 是"我们语料碰巧 box 对了 001"的产物,不是结构上不可能;**(b)** 若官方真实 metric 是**无守卫**版,我们的 binary 守卫让自己**更严**(对 `001` 类),LB 反而可能比离线更高 —— 与诊断"bit +9 方向相反、LB 更高"是**同一机制**,诊断应把"eq binary 守卫 collateral"并入"我们更严→LB 更高"的同侧证据,而不是只列 bit。

### 4.4 [真盲点·诊断未点明] eq 分数 truth `'/13' '/14'`
全 holdout 有 2 道 eq truth 是 `'/13'`(cap-hit 截断,像 crypt 的非数值符号串)。`float('/13')` 抛异常 → 两口径都退到 string-exact，模型不可能逐字符匹配 → 必丢分。这不是判分口径分歧(两口径一致丢),但是诊断"eq oracle 88 乐观"之外**另一个 eq 结构性丢分源**,且不在 770 解码的 eq 子集里被覆盖。

### 4.5 [已澄清·非盲点] 科学计数法/负号
- 科学计数法 truth:全 1899 holdout = **0 例**。该攻击点不成立。
- 负号:crypt 9 例 + eq 6 例 truth 带负号,但 crypt 是符号串截断(本就 0 分),eq 负号走正常 float,无口径分歧。
- 前导零整数:182 例(bit 170 走 binary 守卫正确;eq 12 例,其中 `001` 即 §4.3)。

---

## 5. 对断言2 的最终裁决

| 子断言 | 裁决 | 依据 |
|---|---|---|
| 脚本判分语义"逐字搬运公开文件"非脑补 | **确认** | §1 逐字符核对 3 源文件通过 |
| 770 行×官方变体「我们对∧官方错」=0 | **确认** | §2 复跑 + huikang compare_answer 独立重打仍 0;agi087 真提取器仍 0 |
| 健康类(cipher/gravity/numeral/unit)官方口径不掉 | **确认** | §2/§3,容差分支合规,cipher string-exact 110/110 |
| eq 官方口径不掉(在 770 子集上) | **确认** | §2,84 vs 84 全口径一致 |
| 判分口径虚高 = 0pp | **确认(主结论 SOUND)** | 三套独立官方复刻全 0 虚高 |
| "判分不是缺口来源、不可白捡" | **确认** | 唯一分歧(bit +9、eq 001)方向都相反 = 我们更严,LB 只会更高不会更低 |

**残留保留意见(不改裁决,只缩边界):** 0pp 是在「run-012 已生成的 770 行文本 + holdout 这批 truth 形态」上证的。诊断把它表述为绝对"判分=0pp"略满;更精确的表述是:**在现有解码语料的覆盖范围内,判分口径对 LB 的净影响 ≤ 0(方向偏向我们更严→LB 更高)**。§4.3/§4.4 两个格式点(eq `001` binary 守卫、eq `/13` 分数)是诊断未显式覆盖的真实格式敏感点,但量级极小(eq 全体仅占真权 7.7%,这两类合计 <10 题 / 1899),且方向要么对我们有利、要么两口径一致丢,**不构成"被低估的隐藏增益"也不构成"虚高来源"**,因此不动 0pp 结论。

**与诊断其它断言的关系:** 我只审判分对齐(断言2)。断言1(等权记账错→真权 86.6%)的权重表我顺手交叉验证:holdout.csv 1899 行类分布 = bit 320/unit 319/gravity 319/cipher 315/numeral 315/crypt 165/eq 146,归一后与诊断引用的 train_cat 真权(bit .1686/.../crypt .0866/eq .0771)逐位吻合 → 真权来源独立可查,**等权 79.87% 确系记账幻觉**这点的权重基础成立。断言3 的 eq 过拟合自供(`equation_numeric_deduce.py` 注释 "Mode preference, learned from the holdout" + "account for EVERY mode that ever produces a correct answer on the holdout")**源码实测存在**,B 级证据坐实。这些不在我攻击面内,仅作旁证记录。
