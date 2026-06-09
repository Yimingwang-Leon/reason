# Cryptarithm 攻坚(open leads — NOT dead)

**定位(lead 校正):** crypt 当前 oracle 仅 ~17.6%,但格式上限 ~76%(20% 答案被 `}` 截断),是**最大杠杆**(占全题失分 7.1% ≈ 最大单项)。下方所有 "封死/搁置/give up" 历史判词已删除——它们均**未被一次 paid run 证伪**,仅是 closed-form 符号求解器的局限,不代表任务不可学。

## 数据事实(verified)
- 823 题(train_cat.csv, cat=cryptarithm_deduce)+ holdout。每题 3-5 个 `LHS = RHS` 例 + 1 query;.answer = query 的真 RHS。
- ~26 符号标点字母表;LHS ~4-5 符号,RHS ~1-4 符号。
- operator 永远在 pos2,取 `* + -`,从不出现在操作数位。
- ~20% 答案含 `}`(被 \boxed 截断)⇒ 纯格式上限 ~76-80%。

## 已验证的结构机制(open leads — 可作解题/先验)
- `+` = **字符串拼接**(每题定序 O1O2 或 O2O1,reverse-concat)。`+` query 100% 命中(26/26);AB-concat 48 + BA-concat 11 共 ~7.2% 覆盖、96.7% 精确。这是目前唯一干净、可复现的机制。
- operator → **输出长度**先验:`*`→4,`-`→1-2,`+`→2-3。局部可复现(定长度不定符号),可作长度 prior。
- `-` 带负号前缀。
- 32% 的 LHS **不是** `dd OP dd`(pos2 是普通符号)= 存在另一类非算术变换,尚未刻画。

## Open leads(未解,值得继续挖)
- `*` / `-` 的字符串操作是什么(concat 之外)?线索:操作数可能是 4 个独立值的非位置组合,或 base-23 仅作用于非算子符号。
- 每题 cipher 严格 per-problem(同 LHS 跨题→不同 RHS;9 冲突),且 3-5 例欠定(~2.8 例 vs ~5.8 符号)⇒ closed-form 逐题求解难;但**任务是为 LLM pattern-match 设计的**,模型路径(SFT 学 pattern,不解 closed-form)未被探过,可能远高于符号求解器上限。
- ~12% query 答案含 prompt 内从未出现的符号 ⇒ 这部分 closed-form 不可解,但属少数,不否定其余 88%。

## Closed-form 符号求解器的边界(narrowing,非"死刑")
以下角度作为**符号闭式求解器**均低产,记录以免重复劳动(均非 paid run,仅 agent 探索):
- per-problem digit-map + 列式算术(add/sub/mul):Z3 base 10-16 unsat。
- 位置式 base-N 算术:query 值唯一确定的 175 题解码 0/175 ⇒ crypt 非数字算术,方向应为"字符串代数"。
- enumerate-and-vote ~1.5%;现有 string-rule 求解器(agentB)6.9%;全局码本 / 全局算子码本 / 位置 cipher / 2×2 叉积 / 集合运算 / Caesar 均 falsified。
- 全局非位置值映射:要求 23 glyph 中 ≥2 取不同值即全局 UNSAT(唯一解=平凡常数)。

→ 闭式符号求解器约 ~7%。**这是符号方法的局限,不是任务上限。** 模型/SFT pattern-match 路线与"非算术 32%变换 + `*`/`-`字符串操作"是真正的 open frontier。
