# 第 2 名路线迁移：多时间尺度生产序列 + 极简任务 PPO

源方案见 [第 2 名摘要](../sources/02_MODERNBERT_PPO_SOURCE_DIGEST_ZH.md)。

## 迁移结论

第 2 名最适合迁移的不是 10B steps，而是“每个实体先做未来序列编码，再让小 Transformer 做全局选择”。Kaggriculture 的植物/动物周期比 Orbit ETA 更长，必须把固定 20-step horizon 改成跨小时和跨日的多尺度 horizon。

## Future sequence

对每个己方 plot 在“继续当前计划、对手不影响私有农场”的假设下，构造未来采样点：

`t+1, +6, +12, +23(日终前), +24, +48, +72, +120, +168, +240, 终局`。

每点包含：tile type、crop/animal age、yield、water/fed/care debt、施肥、下一 production、衰减/逃跑、若无人操作的状态、预计可收获价值。对 opponent plot 只使用公开事实；对 market token 构造 town demand 和已知供给下的价格轨迹，不假装知道对手未来订单。

plot sequence 用 3–4 个 depthwise/residual Conv1D block 编码成 128–192 维，再与 unit、global、market embeddings 一起进入 4–6 层 ModernBERT-style Transformer。位置用 farm side + 2D coordinates embedding；不要因为序列已有时间就删除地块空间编码。

## 极简动作的两档设计

Orbit 的 all-in/noop 对应 Kaggriculture 的“保持任务/执行一个完整模板”，而不是“卖出全部”。

### V0 窄动作

- `CONTINUE/NOOP`；
- `MAINTAIN_URGENT`：自动去最紧急 WATER/FEED；
- `HARVEST_AND_BANK`；
- `PLANT_TEMPLATE(crop)`；
- `START_ANIMAL_TEMPLATE(animal)`；
- `SELL_TEMPLATE(item, fraction)`；
- `HIRE/BUY_LAND`。

模板内路径、PICKUP、设施、PLACE 和 DROP 均确定性。它能很快形成可玩策略，适合作为纯 RL 起点。

### V1 扩展动作

V0 稳定后，把模板拆成 task type + target plot/product + quantity，使模型可以并行维护多种作物/动物。不要一开始就开放全部原子动作。

## 训练与评估

- 先用 replay/现有强 agent 做 BC，仅比较 sequence horizon、Conv 宽度和动作模板；
- 最终是否保留 BC 由两臂决定：BC→PPO 与 from-scratch PPO，同模型、同 100M budget；
- terminal W/D/L；对 `40 turns no action` 不可照搬，Kaggriculture 的等待可能是正确策略；改为检测“没有任何任务进展且错过维护/终局现金化”的非法停滞诊断，不自动截断。
- 固定 market/animal/ongoing-crop 专项 seeds，避免窄模板只在 wheat/carrot 上看起来强。

## 覆盖风险

这个方案最容易重演用户在 PTCG BC 的问题：示范数据里某卡组/某农场流派少，模型就不会。缓解必须写进数据管线：

- 按 crop/animal/land/hands/market timing 分层统计 actor decisions；
- rare bucket 过采样，但保留原始分布评估；
- 用规则模板生成 animal、melon、ongoing-crop 专项 demonstrations；
- BC 只做 warm start，不用永久强 KL 锁死；
- self-play 对手池放入这些专项策略。

## 本机效率

1D-CNN + 2–4M Transformer 对 3090 可行，但 future sequence 构造应在 JAX 内批量完成。目标是端到端 10k–40k transitions/s；100M 约 0.7–2.8 小时热循环。若 sequence builder 低于 10k/s，先减少 horizon/通道，而不是扩大模型。

## 适用位置

推荐作为快速强 baseline 和 representation 实验路线，不建议把极简模板直接当最终上限。只要 V0 稳定，应向第 3 名的语义任务张量或第 8 名的自回归联合任务升级。
