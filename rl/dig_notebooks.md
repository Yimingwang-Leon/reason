# 六份公开 notebook 逐份解剖(2026-06-12)

任务背景:我们 run-011 自训 0.85,榜上 ~30 队 0.87。已确认我们的"bit 92.8% 信息论上限"是用**自己解题器的规则族**测的歧义(循环论证)。本文逐份解剖盘上缓存的 6 份公开 notebook,找别人的真实路径。

数据锚点(run012_holdout_decode.jsonl,110/类):
- bit 91/110,**19 道错里 12 道 Hamming=1、5 道 Hamming=2**(纯平局裁决输掉,不是不会做)
- eq 84/110(26 道错);crypt 0/110;cipher/gravity/numeral/unit 全 110/110

---

## 1. agi087/agi-for-medal-0-87-is-possible.ipynb

**一句话:标题党。没有 0.87 的证据,没跑出输出,实质是"huikang 公开 adapter 热启动 + 玩具合成数据再训"。**

- 热启动:`WARMSTART_ADAPTER_DIR = /kaggle/input/models/huikang/nemotron-adapter/transformers/default/20`(huikang adapter 第 20 版),QLoRA 4bit(nf4 双量化)继续训 240 步,LR 2e-4 cosine,batch 1×16,max_len 6144。
- 合成数据是**玩具级**,远弱于我们:bit 只有 invert/reverse/rotate-1,2 三种规则;cipher 只有 reverse/caesar/atbash;eq 只有线性方程 ax+b;与真题分布完全不符。12000 条按"failure_weight"分桶配比。
- 真题部分两种 target:**answer-only**(`\boxed{ans}`,难题 ×4 过采样)和**短模板 trace**("I identify one rule... Final answer")。无任何真 CoT。
- 打包时从 huikang reference submission.zip 抄 adapter_config 字段。
- 桶分类器是糙正则(bit_like/unit_like/numeral_like/equation_like/cipher_like),连 crypt 都没单列。
- **结论:无解题器创新、无 0.87 实证。唯一可取:failure-mining 配比合成额度 + 难题答案-only 过采样的思路。"0.87 is possible" 是愿望句。**

## 2. nemotroncomp-best0-86-solution-nvidia-under-5min.ipynb

**一句话:零技术,纯复制提交 kienngx 的预训 adapter(version_14 = `triton/tinker-adapter`),再用 kaggle CLI 直接提交。**

- 价值在**泄露了 kienngx 的 14 个模型版本名 = 他的完整配方日志**:
  `600-samples-packing-false`、`1800s-lora-rank32-false`、`2400-1e-4_lr-all_linear-packingfalse`、`9500s-batch1-lr1e-4`、`cot-labels-3000samples`、`1200samples-cot-5e-5 / 1e-5 / -`、triton 系 `1e-5(-batch2-workers8)`、`grad-accum1-workers8-maxlen-4096-lr1e-5`、`tinker-adapter`。
- 读出来的 kienngx 路线:**小语料(600–3000 条 CoT 标注样本)、LR 1e-5~1e-4、all-linear、packing=False、batch 1**,最后最强的一版仍是 **tinker 训的 adapter**。0.86 簇 = 这个 adapter 的复制品(与我们 rules_audit 结论一致:复制票私榜必死)。

## 3. end-to-end-finetuning-for-lb-0-86-custom-repo.ipynb(huikang 本人管线)

**一句话:把 huikang 当年 Tinker SFT 跑分 0.86 的那次训练在 Kaggle/Modal 上逐 token 复刻("replay"),全部工程细节公开。**

- 语料:`tahaalam2009/nemotron-data/nemotron/tokens/{problem_id}/synthetic.json`(预 token 化 tokens+mask),**训练顺序按 `logprobs/index.jsonl` epoch-0 原序重放**,不 shuffle。
- 超参(= huikang 原始 Tinker run):LoRA r32/α32/drop0,targets = q/k/v/o + up/down + in/out + **lm_head**;LR **2e-4 线性衰减到 0**(`lr = LR*(1-step/num_steps)`);batch 32、micro 4;AdamW β=(0.9,0.95)、wd=0、eps 1e-8;max_seq 8192;~1 epoch(NUM_STEPS=1000 被语料量截断);不 clip(max_norm=1e9)。
- 工程三件套:
  1) **Tinker 式 MoE expert LoRA 绑定**(`MOE_TIE_WEIGHTS=True`):up/gate 侧绑 lora_A、down 侧绑 lora_B,128 expert 切片保持相同;梯度按 expert 维 **sum**(不是 mean)再 step;保存时自然展开成 128 份 → 提交 zip 与 tinker 兼容。
  2) lm_head LoRA 手动补(Unsloth 对 MoE 默认丢),保存后 key 重命名 `lm_head → backbone.lm_head`。
  3) Cut Cross-Entropy 免物化 logits;LoRA 参数强制 fp32、底模 bf16、MoE router fp32。
- **与我们 run-013("replay+LR3.5e-4"=0.84)的核心差异:他重放的是 huikang 语料本体(他那套程序化 CoT + investigation 痕迹),我们重放的是自己 run-011 语料。配方相同、语料不同、结果差 2pp → 增量在语料内容,不在 LR/replay 机制。**

## 4. nemotron-replay-data-0-86.ipynb

**一句话:#3 的 fork + 一个我们从没用过的成分:掺 NVIDIA 官方数学后训练数据当"防遗忘锚"。**

- 新增:`mohamedamr992/replay-math/nemotron_math_1gb.jsonl`(Nemotron 官方后训练 math 数据,messages 带 `reasoning_content`),用 chat template 重新 token 化,**截到 2M unmasked answer tokens**,按 `replay_every = len(target)//len(replay)` 均匀插空进目标语料。
- LR 改 **3.5e-4**(注释里原值 2e-4),其余同 #3(MoE 绑定、CCE、原序重放、RESET_WEIGHTS=True 从零 LoRA)。
- **解读:这是公开线里唯一的"通用能力锚"实现——掺官方分布数据防 LoRA 把底模推理能力训歪。我们 run-012/013 的 replay 是重放自家题目语料,概念不同。该法声称 0.86(题目自带,无输出 cell 留存)。**

## 5. nemotron-ultimate-sft-grpo-v3.ipynb(信息量最大)

**一句话:在 konbu17 的"验证正确自生成 CoT"数据集上做 SFT;GRPO 写了但禁用("还没帮上忙")。这是 0.86 线里唯一与我们语料构造哲学正面冲突的一份。**

- **konbu17 数据集**(`konbu17/nemotron-sft-lora-cot-selection/train_split_with_cot.csv`):**6558 条"verified-correct"样本**,字段含 `generated_cot`——即**模型自己生成、对答案验证通过的 CoT(拒绝采样)**。配比:Numeral 1491 / Gravity 1511 / Unit 1342 / Cipher 1407 / **Bit 607** / **Equation 200**。konbu17 本人只用了 2907 条拿到基线,本 notebook 全量 6558。
- 训练格式:把 CoT 里的 `\boxed{}` 删掉,接 `'\n</think>\n\boxed{answer}'`;SFT **max_len=7680 与 eval cap 对齐**(明示"4096 会截断 CoT");LR 1e-4("konbu17 验证过")、batch 1×8、cosine、warmup 5%、**NEFTune α=5**、1 epoch、packing=False。
- LoRA targets:正则 `.*\.(in_proj|out_proj|up_proj|down_proj)$` —— **不带 q/k/v/o、不带 lm_head**(与我们提交规范冲突,照抄会踩我们已验证的 MoE 覆盖坑,仅作记录)。
- **解题器层(fallback 路径)的关键差异:**
  - bit:per-output-bit 函数族 = direct / NOT / **const0,const1** / 2-bit(XOR,XNOR,AND,NAND,OR,NOR)/ 3-bit(majority,minority,**choice/mux**)。无平局裁决;
  - **"scaffold with answer"**:解题器搞不定或预测≠真值时,**照样把真值写进 \boxed,前面配诚实的部分分析**("Identified k/8 bit functions... Applying to query: {真值}")。即**全部训练样本都 box 真值**;
  - eq:按运算符字符分组,假设族 = 加/减(双向)/绝对差/乘/拼接(双向)/整除(双向)/模(双向)/XOR/AND/OR/max/min/**digit sum/digit product/power/平均floor**;固定 5 字符 `AA?BB` 解析。
- GRPO:cosine 长度调制 reward + format + reasoning-quality,但 `USE_GRPO=False`,注释:"GRPO not helping yet — model needs more SFT first"。

## 6. nvidia-nemotron-sft-grpo-colab-faster.ipynb

**一句话:LoRA 层选择实验——按 Nemotron 3 Nano 论文 §4.2 量化敏感性只挂 12/52 层;小规模(600 样本),无超 0.85 证据。**

- 层选择:`hybrid_override_pattern` 里 `*`=attention 的 6 层 + 各自前一层 Mamba 共 12 层 + **shared experts(常活跃)**;router 冻结、routable experts 排除。r32/α32/drop0.05。
- SFT:600 样本、LR 5e-5、packing=True、max_len 2048(会截断长 CoT,自伤);解题器同 #5 的早期版(bit 无 const,有 choice_inv;注释自报 bit 全解率 **~42%**,远低于我们)。
- GRPO:beta=0(省参考模型)、temp 0.7、cosine/format/length 三 reward、max_grad_norm 0.1。
- **价值:层敏感性框架(6 GQA + 6 pre-attn Mamba + shared experts)是唯一引用官方论文的定位依据,可作为低预算消融的先验;其余无料。**

---

# 横向结论

## ① bit/eq 解题器对比:**公开 6 份里没有任何一个 bit 解题器强于我们**

- 公开线 bit 族 = per-output-bit 独立函数(上限自报 ~42% 全解);我们/huikang = whole-byte 规则族(unary/pair/const + left/right runs + 位置外推),覆盖更广。
- **"别人 bit 98-99%"不是来自更强的 closed-form 解题器**,这 6 份里不存在那个解题器。
- 真正的机制差异在**语料策略**:公开线(#5 双路径都是)**永远把 ground truth 写进 \boxed**——要么 CoT 是模型自生成+验证通过(konbu17,607 条 bit),要么是"诚实部分分析 + 真值"scaffold。**而我们 corpus 用 reasoner 自洽答案、从不改写为真值** → 歧义题上我们从未教过模型"出题器的真实规则先验"。我们 19 道 bit 错里 17 道 Hamming≤2,正是模型在平局点按错误先验掰的——这与"歧义可由正确先验消解"的情报完全自洽。
- 风险对照:run-006/7/8 崩盘的是"断言全局规则"的**不可学 CoT 格式**;konbu17 路径的 CoT 是**模型自己写的**(可学性由构造保证),scaffold 路径是"部分分析+断答案"(不伪造完整规则)。两者都绕开了我们已验证的坑。

## ② 0.87 那份(agi087)的声称与证据

- **无证据**。无输出、无分数记录,机制 = huikang adapter 热启动 + 玩具合成。0.87 在公开盘面上仍然没有公开复现物;0.86 簇全部追溯到两个源头:kienngx tinker-adapter(复制)与 huikang 语料 replay。

## ③ 我们没用过的数据构造/训练技巧清单

| 技巧 | 出处 | 我们的状态 |
|---|---|---|
| **真值 box + 诚实部分 scaffold(歧义题教先验)** | #5 | 未用(我们 box 自洽答案)→ **头号杠杆候选** |
| **自生成 CoT 拒绝采样语料(konbu17 6558 条,bit 607/eq 200)** | #5 | 未用;数据集公开可下载比对 |
| 掺官方 math 后训练数据 2M tokens 防遗忘 | #4 | 未用 |
| SFT max_len 与 eval 7680 对齐 | #5 | 我们部分语料 6144/8192,需核对 bit 长 CoT 是否被截 |
| NEFTune α=5 | #5 | 未用(小料) |
| MoE expert LoRA 绑定 sum-grad 语义 + lm_head key 重命名 | #3/#4 | tinker 原生已含,本地复刻时才需要 |
| 12 敏感层 + shared experts 定位(论文 §4.2) | #6 | 未用(消融先验,非直接增分) |
| failure-weight 配比合成 + 难题 answer-only ×4 过采样 | #1 | 部分类似(我们有难例倾斜) |

## ④ 语料规模/配比/超参 vs 我们 run-011

- huikang 0.86 线:全题量程序化 CoT 语料、**LR 2e-4 线性衰减到 0**、batch 32、~1 epoch、r32/α32、全 targets 含 lm_head、不 clip。我们 run-011:自有语料、(tinker)LR 更低多 epoch。run-013 已证明 **replay 机制+3.5e-4 在我们语料上 = 0.84**,即配方不是变量,**语料内容才是**。
- konbu17 0.86 线:仅 2907~6558 条**验证过的自生成 CoT**、LR 1e-4、1 epoch、7680 长度。**量级比我们小一个数量级也能 0.86** → 语料"可学性/正确性"比体量重要。

## ⑤ 建议次序(全免费)

1. 下载 konbu17 `train_split_with_cot.csv`,把它的 607 条 bit/200 条 eq 与我们 holdout 19+26 道错题对账:它的自生成 CoT 在歧义题上 box 的是真值时,模型学到的先验长什么样(逐题看它怎么"掰")。
2. 离线量化"先验杠杆":对我们 19 道 bit 近失题,枚举我们规则族的全部平局候选,统计真值落在哪个固定先验下(例如"最小位移/最简规则/出题器偏好"),若存在一致先验 → 用真值-box scaffold 语料重训该 cohort 即是 +19/770≈+2.5pp cohort 的实垫。
3. eq 同理(26 道):用 #5 的 digit-sum/digit-product/power 扩族先跑覆盖率,再决定是否进语料。

*bit/eq_recovered_holdout 本次填 -1:本任务只挖情报,未跑重训实验。*
