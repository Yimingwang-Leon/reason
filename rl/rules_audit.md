# 规则审计:他人公开 adapter 衍生物的合规性(2026-06-11)

**问题:终选(final selection)能否包含"0.86 地板票"(53414820,kienngx 公开 adapter 复现)及其融合衍生物(T-A 系)?**

## 结论(分三级)

| 用法 | 判定 | 依据 |
|---|---|---|
| 公开 adapter + **我们的实质性改动**(T-A 融合注入 / warm-start 续训) | **合规(绿)** | External Data 条款:"publicly available and equally accessible at no cost"满足;Kaggle 惯例明确允许公开 notebook/预训练模型复用;论坛 #703896 社区将 huikang adapter warm-start 视为正当路径 |
| 原样复制他人 adapter 直接提交(53414820 即此类) | **灰色偏风险** | 技术上不违反 External Data;但 "Submission is your own original work" 条款若被严格解读有暴露面;社区有争议帖("Why are most public notebooks just copying Tong's adapter?");若进获奖区,winner verification 要交付训练代码,纯复制票无法交付 |
| 第三方组件 license 交付 | 风险极低 | Kaggle 规则明确豁免:"For third-party components you are not required to grant a license, provided clearly identified";huikang repo 无 LICENSE 文件,需在 writeup 中明确标注出处 |

注:比赛规则页需登录,以上条款来自 Kaggle 官方标准模板(多个 2025-2026 比赛逐字一致),host(NVIDIA/Kaggle staff)对 adapter 复用无专门表态。

## 预登记终选名单(2026-06-12 用户裁定修订;06-15 执行)

**⚠️ 纯复制票(53414820)从终选剔除——用户判断正确,理由三杀:**
1. 私榜上所有同权重复制品分数完全相同 → 巨型同分簇,Kaggle 同分按提交时间排,我们 06-06 提交排簇尾;
2. 复制品无独立私榜方差,零上行;自有模型是唯一权重,私榜抽独立签;
3. 进奖区过不了原创性验证(无法交付训练代码)。

- **终选 = {两个互相差异最大的自有/差异化模型}**:默认 {run-011 0.85, 最佳挑战者(RFT/GRPO 产物或 T-A 融合票,权重唯一即有独立私榜签)}。
- 同为 0.85 的 SWA 票与 run-011 几乎同权重(同轨迹),不算差异签,不选。
- T-A 融合票虽宿主主导,但权重唯一、含我方实质改动 → 合格挑战者;若 T-A 公榜 ≥0.86,与 run-011 配对优于任何复制票。

## 执行纪律
- T-A 融合票的描述与最终 writeup 必须标注 "host = public kienngx adapter (leegongman/0-86-adapter) + our run-011 ΔW injection",透明化第三方来源。
- 进获奖区的预案:能交付的训练代码 = run-011 全管线 + merge_lora.py;纯复制票(53414820)**不得**作为获奖主张票,只作排名保底。

来源:Kaggle 标准规则模板(gen-ai-intensive-course-capstone-2025q1 / ai-cup-2026-performance 等)、discussion #703896、kaggle general #434205/#215878、github.com/tonghuikang/nemotron(无 LICENSE)。
