# 实验台账 (Experiment Ledger) — Nemotron Reasoning

**纪律:每个实验先登记(假设 / 单变量改动 / 精确配置 / 预测),再跑,再补结果+结论(留/弃,对比噪声底)。一行一总结,可追溯、可复现。混杂(多变量)实验明确标注。**

## 干净实验协议(红队 wf_1d741661 硬化中,待填最终版)
- **训练 = `train_split.csv`(7601);评估 = `holdout.csv`(1899,与训练不相交)用 Tinker greedy sampling → 每类准确率 = oracle × 泛化R(不用 LB、可分类目)。**
- **噪声底**:每类 holdout 二项 SE = √(p(1-p)/N)(crypt N=165→SE≈3%);+ 同配置换 seed 跑 2 次的 run-to-run 方差。**Δ < 噪声底 = 噪声,不算信号。**
- **单变量**:对比两次只动目标类的 CoT/数据,其余字节不变;seed/epoch/LR/curriculum/batch 全相同。未改的确定性 4 类 = 对照组,Δ 应≈0(否则有混淆)。
- **决策**:只有 Δ > 噪声底 的改进,才放进**全量 train.csv** 模型 → LB。
- 地基 = run-005 的 0.84 配方(legacy reasoner + curriculum),不从头来;train_split 只是干净测量台。

## 每实验登记模板
```
### <exp-id>  [PLANNED|DONE]  日期
- 假设:
- 单变量改动(vs 哪个 baseline):
- 训练数据 / 配置:seed= epochs= LR= curriculum= batch= train_unembed=
- 预测(跑前写,防事后合理化):
- 结果:LB= / holdout 每类准确率=
- Δ vs 噪声底:
- 结论(留/弃 + 为什么):
- checkpoint:tinker://...
```

## 台账(历史 005–010 如实补录)
| ID | 改动 | 数据 | 关键配置 | LB | 干净? | 结论 |
|---|---|---|---|---|---|---|
| run-005 | legacy reasoner(局部逐列匹配)基线 | train.csv 全量 ⚠含holdout | curr, 3ep, 2e-4, no-lm_head | **0.84** | 单点(无对照) | **自训最优基线**;ckpt `de3a5482` |
| run-006 | 结构化 bit_manip + lm_head | 全量 | 2ep | 0.71 | ❌混杂(2变量) | R 崩(结构化全局规则) |
| run-007 | 结构化 terse | 全量 | | 0.73 | ❌ | R 崩,确认非"应用步"问题 |
| run-008 | 结构化+逐列脚手架 | 全量 | | 0.72 | ❌ | ≈007,确认全局规则断言是 R 杀手 |
| run-009 | bit_manip瘦身 + 子技能augmenter + 2M replay + lm_head | 全量 | 2ep | 0.82 | ❌**4变量混杂** | 不可归因;"加数据"未帮上 |
| run-010 | 5类 locality-hardening CoT | 全量 ⚠含holdout | 2ep(005是3ep) | 0.82 | ❌**5类+epoch混杂,无噪声底** | **不可定论**(我曾误判为"改CoT死路"——证据不足) |
| run-010-filtered | run-010 删 MoE LoRA | 同上 | 提交侧改动 | 0.56 | 提交侧单变量 | ✅ 证实 MoE LoRA 必需(删→残废) |

## 关键教训(已进记忆)
- oracle↑ ≠ LB↑;全局规则断言→R 崩(run-006/7/8 实证)。
- "加现有数据"无效(run-009);模型不缺数据(nll 0.003),缺的是"解题器多解题/合成新技能数据"。
- **从未量过噪声底 + holdout 泄漏进训练 → 过去所有"结论"都可能被噪声/污染污染。干净协议从此强制。**

## 待跑(干净)实验队列 — 等红队定最省方案后填
- E1: 噪声底(同配置 train_split 跑 2 seed)
- E2: equation 解题器改进(oracle 77%→?,局部 CoT)
- E3: crypt 合成数据 + 解题器(oracle 17.6%→?)

## E2 equation 解题器改进 [DONE, 免费, 不花钱]  2026-06-09
- 假设:equation oracle 77% 有免费可提空间 → 提 oracle 涨 LB。
- 诊断(扎实):23% 漏题 = 68% qop_unseen(运算符未演示,**不可解硬上限**,0/115 可救)+ 28% tiebreak 先验(_ORDER 把 signed-sub 排在 absdiff 前)+ 4% 噪声/sign-marker。perfect-tiebreak 天花板 = 83.5% train/86.3% holdout。
- 改动:_ORDER tiebreak 先验(单例 grp=1、signed-top、sign-convention 才动;局部 CoT、无全局规则、R-安全)。
- **结果(实测):train 77.0%→77.3%(+2),holdout 80.1%→80.1%(+0)。** "headroom"大多被 `}` 提取器丢 / sign-ambiguous(强解 overfit)。
- **结论:弃。安全增量 ≈ 0(LB 零),过不了 oracle 审 → 不花钱。** equation 不是杠杆。

### run-011  [PLANNED]  2026-06-10
- 假设:R 是配方属性而非模型上限(双路会诊裁决)。修复"买 R"的语料政策 + 交付已验 oracle 增量,LB 应从 0.84 → ≥0.85,冲 0.86。
- 改动(vs run-005 0.84 锚,全部过离线验收):
  1. crypt solver 17.6%→22.2% 移植(box 与已验版 0 差异;局部验证式 trace;1 行收尾)
  2. equation trace 重写:隐藏断言 stub → 全枚举 transcript(box 732/732 字节不变;每行机械可推;max 3588 tok)
  3. bit_manip 回退 run-005 字节原版 89ea8f4(median 6687/max 7668,oracle 1364/1602)+ 纳入 238 条过程忠实 wrong traces(其余类 correct-only,crypt concat 兜底/equation 错断言 = 毒类不收)
  4. champion drills 8463 导入(数据文件解析,enable_thinking=True 推理 regime + </think> 收尾)+ 我们 equation drills 900
  5. 加固:metric_correct abs_tol=0、brace 门只丢 '}'(+15 crypt)、收尾统一 run-005 1 行式
- 配置:curriculum ON、2ep、LR 2e-4、batch 64、rank 32、全模块栈、无 lm_head、无 replay(留 run-012)
- 预测(跑前写):中心 0.85(0.848-0.855);显示 0.86 概率 ~30-40%;≤0.84 = 配方保真假说证伪即止损
- 预算:~$26-28 / 上限 $35;三审(隔离/oracle/R-安全)全票通过才开训

#### run-011 三审闸第 1 轮(2026-06-10)
- Lane2 oracle PASS(全量 8671 解码验证、469 条 crypt 验证行重算)。措辞修正:crypt 161 条含 11 条 lucky-correct unseen-op concat-fallback(诚实声明无规则可验→默认规则→box 恰好正确,correct-only 门放行;"验证失败仍box"毒类 = 0 条)。
- Lane3 R-safety PASS(238 难尾两级机器验证 procedure-faithful;0 locality 违规;cap 余量 47 tok 提示:不得再加任何后缀)。
- Lane1 isolation FAIL(blocker):预算守卫单价 $0.0694/步按 run-004 重 token 批次校准,对本 drill 语料高估 ~2× → $33 默认上限在 564 步中第 ~474 步触发 → 截断存残废 final。
- 修复:--usd-per-step CLI 旗标 + token 比例重校 0.0351(46.4M total tok / 282 批 = 164.6k tok/步 vs 锚 326k)。对账:预计实花 ~$20,钱包上限 $35,守卫余量 941 步。
- **预登记发车命令**:`python -m src.train_tinker --run-name run-011 --num-epochs 2 --curriculum --budget-usd 33 --usd-per-step 0.0351`

#### run-011 训练事故与恢复(2026-06-10 15:42)
- 三审第2轮全票 PASS 后按预登记命令发车。训练健康推进至 **543/564 步(96%,LR 已退火至 7.4e-6,nll ~0.002,实质收敛)** 时,**Tinker 账户余额耗尽(402 billing blocked)**,进程死亡。
- 教训:闸审核了"花费≤上限"却没核"账户实际余额≥预计花费"。下次发车前必查余额。
- 连带 bug:RESUME_ERRORS 含非异常类 tinker.Timeout → except 时 TypeError,402 没走到优雅存状态(已修)。
- 可恢复资产:epoch0 ckpt(`sampler_weights/epoch0`,ep0 末 nll 0.0037 已基本收敛,可作保底)+ `weights/state_ep0`(含优化器,可续 epoch 1)。
- 已装自动完成链(finish011.sh):每 5 分钟探测 billing,解锁后自动 `--start-epoch 1 --resume-state state_ep0` 续跑 epoch 1(282 步,~$10)→ 下载→构建→提交→出分。注:跨进程恢复时 epoch1 的 curriculum branch-weight 因 prev_lp 缺失退化为 plain-CE(可接受:curriculum 价值本未定,huikang 全程 plain-CE)。
- **待用户:给 Tinker 充值(建议 ~$15:续训 ~$10 + 余量)。充值后无需任何操作,链条自动跑完。**

#### run-011 结果(2026-06-10)
- **LB = 0.85**(53534096)。跑前预测中心 0.85(0.848-0.855)→ **命中中心**。
- 意义:7 次付费以来首次超过 run-005(0.84→0.85,+1pp);"配方保真买 R"假说部分验证;"0.84 天花板"正式证伪。
- 实际成本:~$60(预估失误 + 余额中断重跑,单价实测 ~$0.077/步;教训已记)。
- 预承诺分支 =0.85 触发:剩余 ~1pp 缺口的候选杠杆 = ① math-replay + LR 3.5e-4(mohamedamr 公开 0.85→0.86 的 delta,刻意留作 run-012 单变量)② 分类 R 诊断 ③ crypt synth。任何花钱动作需用户逐项批准(auto-spend 已收回)。

#### 微探针结果 + 白纸0.87会诊(2026-06-10 夜)
- 微探针($~1,评审3/3放行,80道已训题 greedy):**crypt 复现 1/30=3%(决定性死刑,synth 正式毙,省$45)**;equation 枚举 30/30=100%(run-011 +1pp 主源);bm 难尾 17/20=85%(无毒)。推论:可复现类目上 R≈0.97,已持平 pack。
- 白纸 0.87 会诊(5 从零设计+对抗):**全部 weak(15-30%)**;综合裁决 = 最优打法是 HYBRID:**run-012(=run-011语料字节不动 + 修正replay + LR3.5e-4,但只训 train_split 7601 剔除 holdout)作 M0 → holdout 真R测量 → on-policy 营救采样+splice 补丁FT(GreedyPatch层)**。
- 预登记:run-012 LB 中心 0.86(0.855-0.865),≤0.85 = replay-delta 证伪。全链 P(true≥0.87)≈25%(20-30%),中位=挤进0.86堆;下限=run-011 0.85 保底。增量成本 ~$65-95,$100 硬停,逐门降级。
- 铁律:发车前 Tinker 余额 ≥1.3×预计;不碰 CoT 重设计/crypt synth/equation 调参(全被实测封死)。

### E4 死区挖矿 + crypt-R 探针 [PLANNED] 2026-06-11(独立议会方案,$15 分段)
- 预登记预测:挖矿收率 eq~30%@K8 / crypt~17% / bit 未知(>0 即新信息);探针门 = 43 留出 crypt 贪心 ≥50% 且 30 题非 crypt 无遗忘(≥95% box 不变)。
- 自检修正:①crypt 重写禁止"假设先行"(run-011 假设+验证式实测 3%,必须前向推导/fail-fast 枚举);②剔除议会的 eq-tiebreak 块(E2 实测安全增量≈0);③bit 采样限 100×K4 防超算。
- 止损:挖矿 ≤$6、探针 ≤$7、总 ≤$14;任一门不过即停,余款留给用户决定。
