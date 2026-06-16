# kienngx-0.86 宿主包验明正身(2026-06-11 23:30)

来源:`leegongman/0-86-adapter`(kaggle 公开 dataset,3.55GB)。本包即 06-06 我们原样提交拿到 0.86 的那份(53414820)。

| 项 | kienngx | 我们(run-011) | 判定 |
|---|---|---|---|
| r / lora_alpha | 32 / 32(含 rank_pattern/alpha_pattern in_proj=32) | 32 / 32 | ✅ 同尺度,s=1.0 |
| target_modules | q/k/v/o + in/out + up/down + **lm_head** | 同,无 lm_head | lm_head 走 passthrough 原样保留(0.86 过审形态,不动) |
| 键数 | 12010(=6005 对) | 12008(=6004 对) | 差 = lm_head 对 |
| 键命名空间 | `base_model.model.model.layers.*` | `base_model.model.backbone.layers.*` | ⚠️ 不同 PEFT 包装;`--inject-key-map "base_model.model.backbone.=base_model.model.model."` 映射覆盖 **12008/12010**,仅 lm_head host-only |
| MoE up/down 键 | 11776 | 11776 | ✅ 全对齐 |
| dtype | F32(3.55GB) | BF16(1.77GB) | 输出逐张量跟宿主(F32) |
| base_model | NVIDIA-Nemotron-3-Nano-30B-A3B(-BF16) | 同 | ✅ |

**结论:T-A 可行,无需降级。** 配方:adapter_config 用宿主原件字节不动;宿主独有 lm_head 对 passthrough;6004 对在 ΔW 空间注入(s1=s2=1.0)。

- T-A1:λ=0.25 直加,QR-SVD 低秩截 32(范数膨胀理论 √(1+λ²·‖r‖²/‖k‖²) 量级,范数闸 ±5% 把守)
- T-A2:λ=0.5 + DARE p=0.9(置零 90%、×10 rescale),稠密路径 svd_lowrank 截 32

预登记(承 04_merge §3,字节不改):T-A1 中心 0.86,P(0.87)≈10%;T-A2 中心 0.85-0.86,P(0.87)≈6%。证伪线:两票均 ≤0.85 → 注入伤宿主,本族死。
