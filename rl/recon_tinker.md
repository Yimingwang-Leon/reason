# Tinker RL 侦察报告 — Nemotron-3-Nano-30B-A3B (MoE) LoRA rank32 跑 GRPO/RFT/DPO 的工程路径与成本模型

**日期 2026-06-11。本报告纯侦察,零 API 调用、零花费。证据来源:本地已装 `tinker==0.22.3` SDK 源码(`/opt/anaconda3/envs/nvdia_kaggle/lib/python3.12/site-packages/tinker/`)、`tinker_cookbook==0.4.1` 源码(同目录 `tinker_cookbook/`)、我们的 `src/train_tinker.py`、huikang 的 `external/nemotron-huikang/{loss_config.py,train_sft.py,trainer/client.py}`、`autoresearch/EXPERIMENTS.md` 实测账、官方 docs(tinker-docs.thinkingmachines.ai 损失函数页)。**

---

## TL;DR

1. **采样**:`sc.create_sampling_client(base_model=..., model_path=<run-011 sampler_weights>)` + `sampler.sample(prompt, num_samples=K, sampling_params=SamplingParams(max_tokens=…, temperature>0))`,一次调用拿 K 条 rollout,**每条自带 per-token `logprobs`(RL 必需,挖矿时没存,这次必须存)**。先例代码 `src/r_harness.py:84-110`。
2. **RL 更新**:`forward_backward(data, loss_fn=...)` 原生支持 `cross_entropy / importance_sampling / ppo / cispo / dro` 五种 loss;RL 类 loss 的 `loss_fn_inputs = {target_tokens, logprobs(采样时), advantages}`。**GRPO 可以完整实现**(组内 baseline 在客户端算,cookbook `recipes/rl_loop.py` 就是教科书实现);**DPO 可以实现**(`forward_backward_custom` + 参考策略 sampling client 的 `compute_logprobs`,cookbook `preference/train_dpo.py` 现成)。
3. **续训**:`sampler_weights/*` 路径**只能采样不能续训**;训练初始化要用 `weights/state_*` 路径。run-011 final(0.85)对应的训练态 `state_ep1` 按代码逻辑应存在于 session `cad6ab5c`(执行阶段第一步用 `rest_client.list_checkpoints` 确认)。**RL 推荐 weights-only 加载(fresh Adam)+ 手动恒定低 LR**;LR 完全是客户端逐步传入的,没有服务端退火态。
4. **成本**:实测锚点换算 **训练 ≈ $0.21/M token、采样 ≈ $0.0025/条 rollout(均长 ~5.5k 生成 token)**。一轮"7601 题 × K8 + 1 epoch RL 更新"全价 ≈ **$240**(采样 $152 + 更新 $88);利用 GRPO 全同奖励组丢弃(死区组占多数)实际 ≈ **$170-190**。缩放公式见 §4,**1000-2000 题营救 cohort 一轮 $25-65,4 天预算内可跑 2-3 轮**。
5. **头号工程坑**:402 余额(发车前查余额 ≥1.3×预算)、`tinker.Timeout` 不是异常类(except 元组里会 TypeError)、download future hang(超时重试 + 并行 curl)、RL rollouts 必须先落盘再训练(session 死了采样钱才不白花)、MoE 路由崩塌用免费的 `e_frac_with_tokens` 等 metrics 监控。

---

## 1. 采样:从 run-011 sampler_weights 以 temperature>0 批量采 K 条

### 1.1 先例代码在哪

- **在库里的唯一活体**:`src/r_harness.py:84-110` `sample_adapter()` —— 完整展示了 鉴权(`_load_env`)→ `tinker.ServiceClient()` → `create_sampling_client(base_model=BASE_MODEL, model_path=adapter_path)` → `SamplingParams(max_tokens=7680, temperature=0.0, top_p=1.0)` → `sampler.sample(prompt=ModelInput.from_ints(ids), num_samples=1, ...).result()` → `chat_tok.decode(resp.sequences[0].tokens)`。prompt 用 `src/corpus.tokenize_prompt`(chat template + boxed 后缀 + enable_thinking),与训练/评测制式一致。
- **死区挖矿/微探针**(2026-06-11,~2000 rollouts ≈ $5)用的就是同一套调用,只是 `temperature>0`、`num_samples=K(8 或 4)`;**那段脚本是 scratch 一次性产物,未提交 git,已删**(单版本纪律)。设计参数留档在 `autoresearch/EXPERIMENTS.md` E4 段(eq 死区 166 题×K8、bit 限 100×K4)。
- **批量 K + 并发的教科书写法**:cookbook `recipes/rl_loop.py:149-172` —— 对每题发一个 `sampling_client.sample(prompt, num_samples=group_size, sampling_params)` **future(不立即 .result())**,全部发完再逐个收割。SDK 内置重试(`retry_handler`)、连接池上限 1000(`tinker/_constants.py`),并发由服务端排队消化。

### 1.2 API 签名(0.22.3 实测源码)

```python
# tinker/lib/public_interfaces/sampling_client.py:292
sample(prompt: ModelInput, num_samples: int, sampling_params: SamplingParams,
       include_prompt_logprobs=False, topk_prompt_logprobs=0) -> Future[SampleResponse]

# tinker/types/_pydantic_types/sampling_params.py
SamplingParams(max_tokens=None, seed=None, stop=None|str|seq, temperature=1.0, top_k=-1, top_p=1.0)

# 返回:resp.sequences: list[SampledSequence],每条有
#   .tokens (list[int]) / .logprobs (list[float], 每生成 token 的采样 logprob) / .stop_reason
```

**关键点:`SampledSequence.logprobs` 默认就返回**(`rl_loop.py` 直接 assert 非 None)。RL 的 importance ratio 必须用它,所以 rollout 落盘 schema 必须存 `tokens + logprobs + reward`,不能只存 decode 文本。

另有 `sampling_client.compute_logprobs(prompt)`(`sampling_client.py:379`):给完整序列算逐 token logprob,按 prefill 计价,是 DPO 参考 logprob / KL 锚的取数通道。

### 1.3 采样参数与计价

- `max_tokens`:评测制式 7680;rollout 可按类目下调(run-012 holdout 解码 281 条实测:生成均值 5492 tok、中位 6504、22.5% 撞 7680 顶)。把营救 cohort 的 cap 收到 ~6k 可省 ~20% 采样钱,但会截掉部分长尾正样本——建议首轮保持 7680,拿到长度分布再收。
- `temperature`:挖矿先例 K8 用 ~1.0(死区 pass@8 实测 1-2%);RL 常规 0.7-1.0。`seed` 可设保证可复现。
- **计价(实测锚)**:~**$0.0025/rollout**(BRIEF;挖矿 2000 条 ≈ $5),对应我们 prompt ~1-1.5k + 生成 ~5.5k 的混合。换算隐含采样价 ≈ **$0.4-0.5/M 生成 token**(官方按 prefill/sample/train 三价目分开计,MoE 按激活参数计价;官方表是 JS 渲染抓不到,以实测锚为准)。

---

## 2. RL 更新:forward_backward 的 loss 家族与 GRPO/DPO 可行性

### 2.1 支持的 loss(SDK 硬编码,`tinker/types/loss_fn_type.py`)

`LossFnType = "cross_entropy" | "importance_sampling" | "ppo" | "cispo" | "dro"`

调用签名(`training_client.py:259`):

```python
forward_backward(data: list[Datum], loss_fn: LossFnType,
                 loss_fn_config: dict[str, float] | None = None) -> APIFuture[ForwardBackwardOutput]
# 随后照常 optim_step(AdamParams(learning_rate=..., beta1=0.9, beta2=0.95, eps=1e-8, grad_clip_norm=...))
# 两个 future 先后 await(流水线),与 SFT 完全一样
```

### 2.2 每种 loss 的输入(`tinker/types/datum.py` `_KEY_TO_TYPE` 白名单)

`loss_fn_inputs` 合法键只有:`target_tokens(int64) / weights / advantages / logprobs / clip_low_threshold / clip_high_threshold(均 float32, 长度 = len(tokens)-1)`。**多余键(如 cookbook 内部用的 "mask")发给后端会出错,cookbook 在 `rl/train.py:265 _remove_mask` 里先剥掉——我们也要剥。**

| loss | loss_fn_inputs | loss_fn_config | 公式(官方 docs) |
|---|---|---|---|
| cross_entropy | target_tokens, weights | — | `L = Σ -logp·w`(SFT;也是 RFT 的 loss) |
| importance_sampling | target_tokens, **logprobs**(采样策略), **advantages** | 无 | `L_IS = -E[ ratio · A ]`,`ratio = exp(logp_θ - logprobs)` |
| ppo | 同上 | `clip_low_threshold`(默认 0.8)、`clip_high_threshold`(默认 1.2)——**是 ratio 的绝对界,不是 ε** | `-E[min(ratio·A, clip(ratio, lo, hi)·A)]` |
| cispo | 同上 | `clip_low_threshold`(默认 0.0)、`clip_high_threshold`(默认 4.0) | `-E[ sg(clip(ratio,lo,hi)) · logp_θ · A ]`(截断系数 stop-grad,被 clip 的 token 仍有梯度,适合难探索) |
| dro | target_tokens, advantages(=reward), logprobs(=ref) | `beta`(如 0.05;huikang `DROLossConfig` epoch0 置 0) | direct reward optimization,惩罚项 `β·½·(logp-ref)²`(huikang `loss_config.py:345-376` 的 metrics 复现了它) |

注意两个易错点:
- **huikang `loss_config.py:330-335` 的 `PPOLossConfig(clip_low=0.2, clip_high=0.2)` 与官方约定冲突**(官方阈值=ratio 界 0.8/1.2;他的 0.2/0.2 直接传 `clip_low_threshold=0.2` 会把 ratio 钳到 0.2)。他的 CISPO(0.8/1.2)倒是 ratio-界约定。**照官方 docs 用 ratio-界**。
- `clip_*_threshold` 也允许作为 **per-datum 的 loss_fn_inputs 张量**传(datum.py 白名单里有),可做逐 token 阈值,一般用不上。

`ForwardBackwardOutput.loss_fn_outputs[i]["logprobs"]` 返回每 datum 的训练时逐 token logprob(与 SFT 相同),可用来算 sample-train KL 漂移(cookbook `rl/metrics.py:18 compute_kl_sample_train`)。**`metrics` 里免费附送 MoE 健康度:`e_frac_with_tokens:mean / e_frac_oversubscribed:mean / e_max_violation:mean|max`(`tinker/types/forward_backward_output.py` 文档注释:下降/上升趋势 = 路由崩塌前兆)——RL 阶段必须盯。**

### 2.3 GRPO:可以,且有官方参考实现

**组内 baseline 完全在客户端算,后端无感**。cookbook 两处:
- `rl/data_processing.py:23 compute_advantages`:`A_i = r_i - mean(r_group)`(只中心化、不除 std,即 Dr.GRPO 风格);
- `recipes/rl_loop.py`(整文件,~270 行)即一个完整同步 GRPO 循环:
  1. 每个 batch 先 `training_client.save_weights_and_get_sampling_client()` 刷新采样权重;
  2. 每题 `sample(num_samples=G)` → 打分(boxed 提取 + grade)→ `A = r - mean`;
  3. **全组同奖励(A 全 0)的题直接丢弃,不进 forward_backward**(死区题大多在这一步被免费过滤,只花了采样钱不花训练钱);
  4. datum 拼法:`model_input = prompt + sampled[:-1]`,`target_tokens = [0]*ob_len + sampled`,`logprobs/advantages` 在 prompt 段补 0(`rl_loop.py:203-227`);
  5. `forward_backward(datums, loss_fn="importance_sampling")` + `optim_step`。
- 工业版(异步、minibatch 流、KL 锚、多 substep)在 `rl/train.py`(`train_step` 把 batch 切 `num_substeps`,每段一对 fwd_bwd+optim_step 流水线)。
- **KL 锚(反遗忘,BRIEF 硬要求)**:`rl/metrics.py:124 incorporate_kl_penalty(data, base_sampling_client, coef, discount)` —— 对每条 rollout 用锚策略(= run-011 sampler client)`compute_logprobs_async`,把 `coef·(avg_kl - per_token_kl)` 加进 advantages,原地改。代价 = 全部 rollout 再过一遍 prefill(见 §4,非零,可只对子样本/低 coef 用)。

### 2.4 DPO:可以,参考 logprob 这样拿

cookbook `preference/train_dpo.py` 全套:
- **参考策略 client**:`create_dpo_clients()`(:142-196)在训练开始时 `reference_client = training_client.save_weights_and_get_sampling_client()` —— 把**初始权重**(对我们=run-011)存成 sampler 权重,开个冻结的 sampling client;
- **ref logprob**:`do_update()`(:374)`await asyncio.gather(*[reference_client.compute_logprobs_async(seq) for seq in full_sequences])`(prefill 计价);
- **loss**:`compute_dpo_loss()`(:199-257)`L = -logsigmoid(β·((logp_c - ref_c) - (logp_r - ref_r)))`,β 默认 0.1;
- **反传通道**:`forward_backward_custom(data, custom_loss)`(`training_client.py:393-505`)——先 forward 拿 logprobs,客户端 torch 求 `dC/dlogp`,再把 `weights = -grad` 塞回 cross_entropy 后端做 backward。**即一次 DPO step = 1 次 forward + 1 次 forward_backward ≈ 1.5-2× 训练价**,且 chosen/rejected 成对 → 数据量×2。
- 我们的 DPO 对儿可直接从 GRPO 组里取(同题正/负 rollout),无需另采。

### 2.5 RFT(最便宜的兜底)

采 K 条 → 留 reward=1 的 → `loss_fn="cross_entropy"`(weights=completion mask)小 LR 续训。和现有 `src/train_tinker.py` 代码 100% 复用(只换语料),训练费只付正样本(死区/营救 cohort 正样本占比低 → 更新费近乎忽略)。**这是"营救臂 17% 可救"探针的直接产品化路径。**

---

## 3. 续训/初始化:从哪个 checkpoint 开、LR/退火态

### 3.1 run-011 在 Tinker 上的资产盘点(本地 `training/run-011/`)

| 资产 | 路径 | 能干什么 |
|---|---|---|
| epoch0 采样权重 | `tinker://6b434b58-ff28-5694-bb42-8bec46470302:train:0/sampler_weights/epoch0` | 只能采样 |
| epoch0 训练态(含优化器) | `tinker://6b434b58-...:train:0/weights/state_ep0` | 可恢复训练(但这是 ep0 末,非 0.85 权重) |
| **final 采样权重(LB 0.85)** | `tinker://cad6ab5c-35bc-516c-90bd-804b0a6166f5:train:0/sampler_weights/final` | **RL 的 rollout 来源 + KL/DPO 参考策略** |
| final 训练态 | **推断为 `tinker://cad6ab5c-...:train:0/weights/state_ep1`**(`train_tinker.py:431-439` 每个完成 epoch 必存 state;路径未落盘到本地文件,console log 没留) | RL 的训练初始化 |

**硬约束(官方 docs save-load 页 + SDK 注释):`sampler_weights/...` 只能喂 sampling client;训练初始化必须 `weights/...` 路径。** 所以执行阶段第 0 步:`sc.create_rest_client().list_checkpoints(...)`(查询类,基本免费)确认 `cad6ab5c` 下 `state_ep1` 存在。若不存在,降级方案:(a) 从 `state_ep0` 起 RL(权重≈收敛,nll 0.0037,但比 final 少一个 epoch,LB 未测);(b) 花 ~$10 用 `--start-epoch 1 --resume-state state_ep0` 重放 epoch1 再存 state(确定性重放,种子按 epoch 固定)。我们 `save_state` 没传 `ttl_seconds` → 永不过期,不用担心被回收。

### 3.2 推荐初始化方式

```python
# 权重要 run-011 final,优化器要全新(RL 梯度统计与 SFT 末期 Adam 矩无关,且 SFT 末 LR 已退火到 3.5e-7)
training_client = await sc.create_training_client_from_state_async(   # 注意:不带 _with_optimizer
    "tinker://cad6ab5c-...:train:0/weights/state_ep1")
# rank/train_mlp/train_attn/train_unembed 自动从 checkpoint 的 weights_info 继承(service_client.py:278-289),
# 即 rank32 + 全模块栈 + 无 lm_head,与提交约束一致,不会漂
```

`create_training_client_from_state_with_optimizer` 只在"RL 进程中途断了、续同一场 RL"时用(此时要还原 RL 自己的 Adam 矩)。

### 3.3 LR / 退火态注意

- **没有服务端 LR 调度**:LR 是每次 `optim_step(AdamParams(learning_rate=...))` 现传的(`train_tinker.py:350-358` 我们自己做线性退火)。**所谓"退火态"不存在于 checkpoint 里**,从 state 恢复只还原权重+Adam 矩;LR 想给多少给多少。
- cookbook 对 Nemotron-3-Nano **明确未校准 LR**(`hyperparam_utils.py:264-278` 直接 `raise NotImplementedError`);其通用规则:LoRA LR = full-FT LR × 10(`get_lora_lr_over_full_finetune_lr`,"LoRA Without Regret" 结论);`rl_loop.py` 对 Llama-8B rank32 用 **4e-5 恒定**。
- 我们自己的标定:SFT 用 2e-4(峰值);**探针教训:60 步 @ 8e-5 离线小补丁就把锚题打到 80%**(EXPERIMENTS.md E4)。RL 是 on-policy(自身分布,遗忘压力天然小于 off-policy 补丁 FT),但仍建议:**恒定 1e-5 ~ 4e-5 起步、grad_clip_norm 设实数(别再用 1e9)、KL 锚或每 N 步锚题探针(30-40 题贪心,$0.1)做熔断**。
- batch:64 题 × K8 = 512 datum/步;`forward_backward` 内部自动按字节分块(`_chunked_requests`),也可学 cookbook 切 `num_substeps`。

---

## 4. 成本模型(全部从我们的实测锚校准)

### 4.1 锚点(A 级,付费实测)

| 量 | 实测 | 换算 |
|---|---|---|
| SFT 训练单价 run-004 | $10/144 步 = $0.0694/步 @ ~326k tok/步 | **$0.213/M train token** |
| SFT 训练单价 run-012 | ~$22/608 步 ≈ $0.036/步 @ ~165k tok/步 | **$0.218/M train token**(两锚互证 ✔) |
| run-011 含事故单价 | $0.077/步(含 402 重跑摊销) | 预算守卫用这个保守值 |
| 采样 | 挖矿 ~2000 rollouts ≈ $5 | **$0.0025/rollout**(prompt ~1-1.5k + 生成均值 ~5.5k,cap 7680) |
| 生成长度 | run-012 holdout 解码 281 条:均值 5492 tok,22.5% 撞 cap | rollout 长度基准 |

### 4.2 公式

设 N=题数、K=每题 rollout 数、L̄≈6.8k(prompt+生成总 token)、f_keep=进训练的 rollout 占比(GRPO 丢全同组后)、E=对同批 rollout 的更新 epoch 数:

```
C_采样   ≈ N·K·$0.0025                  (max_tokens 收紧到 ~6k 时 ×0.8)
C_更新   ≈ f_keep·N·K·L̄·$0.218e-6·E    ≈ f_keep·N·K·$0.00148·E
C_存档   ≈ n_policy_refresh·$0.05       (save_weights_for_sampler/save_state 每次≈一小步的钱)
C_KL锚   ≈ f_keep·N·K·L̄·p_prefill      (可选;prefill 价未实测,~$0.1/M 量级 → 全量约 +$40,建议只对 10-20% 子样本算或干脆用锚题探针代替)
```

f_keep 经验预估:死区题组几乎全灭(pass@8 1-2% → 组内全 0 全同 → 丢);营救/边缘题组才有混合奖励。全量 7601 上预估 f_keep ≈ 0.25-0.4;预筛 cohort 上 ≈ 0.4-0.6。

### 4.3 关键场景报价(E=1)

| 场景 | 采样 | 更新 | 合计/轮 |
|---|---|---|---|
| **全量 7601 × K8**(题面问的) | 60,808 × $0.0025 = **$152** | f_keep=1 上界 $90;f_keep≈0.3 → **$27** | **上界 ~$245,实际 ~$170-190** |
| 营救+边缘 cohort 2000 × K8 | $40 | $7-14 | **$47-55** |
| 1000 × K8 | $20 | $4-7 | **$25-28** |
| 微试点 200 × K8(预登记校准轮) | $4 | ~$1.5 | **~$6** |
| RFT 版(同采样,只训正样本) | 同上 | ≈采样的 5-15% | cohort 2000 ≈ **$42-46** |
| DPO 版(组内取对,1k 对) | 0(复用 rollouts) | 2k 序列×2 pass + ref prefill | **$8-15** |

每步 RL fwd_bwd 单价参考:512 datum × 6.8k tok ≈ 3.5M tok → **~$0.75/步**(比 SFT 步贵 10-20 倍,因为序列长且每步 datum 多;预算守卫的 `--usd-per-step` 必须按这个重标,别沿用 0.04/0.077)。

**结论:采样钱是大头(60-85%),省钱杠杆排序 = 选题(N)> K > max_tokens > f_keep。** 4 天窗口、~$100-150 预算下的合理编排:$6 微试点(验 reward 管道+f_keep 实测)→ 2000-cohort 一轮 $50 → 看 holdout 增益决定第二轮或转 RFT/DPO。

---

## 5. 工程坑清单(全部踩过或源码确认)

1. **402 余额(run-011 事故)**:训练推进 96% 时余额耗尽进程死。**发车前必查余额 ≥ 1.3×预计**;RL 加一条:**rollout 边采边落盘(jsonl: id/tokens/logprobs/reward),任何 session 死亡后采样钱不丢、可断点续训**。
2. **`tinker.Timeout` 不是异常类**(是 httpx 超时配置的 re-export,`tinker/__init__.py:4`)。塞进 `except (...)` 元组会在 catch 时 TypeError——这 bug 当时正好掩埋了 402 的优雅存档。`train_tinker.py:285-288` 的防御性过滤(`issubclass(c, BaseException)`)必须带进 RL 循环。
3. **session 死亡类**:`SidecarDiedError / InternalServerError / RequestFailedError / 卡死` 都是 `tinker.TinkerError` 子类,宽接 + 从 `save_state` 恢复;`AuthenticationError / BadRequestError / NotFoundError / PermissionDenied / UnprocessableEntity / Conflict` 是永久错,fail-fast(`train_tinker.py:290-291`)。
4. **心跳饿死**(run-005 v1 死因):客户端循环里长时间持 GIL 的纯 Python 逐 token 处理会饿死 SDK 心跳线程 → session 被杀。RL 的 reward 打分/advantage 计算用 numpy/向量化,float32 存 logprobs(`train_tinker.py:156-186` 注释有完整事故记录)。
5. **download hang(0.22 已知)**:`get_checkpoint_archive_url(...).result()` 可能挂死 → 超时+重试包裹;1.4GB checkpoint 用 16 连接 curl range 并行(~12MB/s)。上传 Kaggle 用 2MB 块断点续传,**8MB 块必撞 90s write-timeout**(memory `project_tinker_sdk_pin.md`)。`src/build_submission.py:52` 的下载入口可直接复用(其中 0.16 的 WARN 注释已过时,0.22.3 已修)。
6. **预算守卫单价要按语料重标**(run-011 Lane1 blocker:run-004 锚高估 2× 差点中途截断)。RL 步 ~$0.75 ≫ SFT 步,`usd_per_step` 不重标守卫就是摆设。
7. **`loss_fn_inputs` 白名单**:只许 `target_tokens/weights/advantages/logprobs/clip_*_threshold`;"mask" 等额外键要剥掉(cookbook `_remove_mask`);ragged 列表会被 datum 构造直接拒(`datum.py:67-82`),prompt 段的 logprobs/advantages 用 0 填平。
8. **每个 RL step 都要刷新采样权重**(`save_weights_and_get_sampling_client`),每次≈一小步的钱;想省可每 2-4 步刷一次,off-policy 漂移交给 ppo/cispo 的 clip 兜底(loss 里的 `logprobs` 永远填**采样那一刻**的值,不要回填)。
9. **MoE 路由监控**:fwd_bwd `metrics` 免费带 `e_frac_with_tokens:mean / e_max_violation:*`;RL+较高 LR 下路由崩塌是 0.85→残废的静默路径,逐步记录、趋势恶化即停。
10. **checkpoint 卫生**:`save_state(ttl_seconds=604800)` 给 RL 中间态(7 天自动清),final 不带 ttl;`delete-tinker-checkpoint.sh`(huikang 库里有)清理旧的,避免存储费。
11. **0.16.1 已死**:服务端 400 拒绝,环境必须 0.22.3(已装,`pip show tinker` 验证过)。

---

## 6. 残留不确定点(执行阶段第 0 步要验证,均为低价/免费查询)

1. `tinker://cad6ab5c-...:train:0/weights/state_ep1` 是否存在(`rest_client.list_checkpoints`)。不存在 → §3.1 降级方案。
2. `num_samples=K` 的 prefill 是收 1 次还是 K 次(官方价表 JS 渲染抓不到;实测锚 $0.0025/rollout 已含,微试点轮顺手用账单核一次)。
3. f_keep 真值(微试点 200×K8 直接量)。
4. RL 步 wall-clock(采样并发吞吐随集群负载浮动;挖矿当晚 2000 rollouts 在小时级完成,够 4 天窗口,但首轮要计时)。
5. KL 锚 prefill 单价(若太贵就换"锚题贪心探针 + 熔断"作反遗忘机制,$0.1/次)。
