# 第 5 名路线迁移：Coverage-aware BC + IMPALA/V-trace

源方案见 [第 5 名摘要](../sources/05_IMPALA_BC_SOURCE_DIGEST_ZH.md)。

## 迁移结论

可做，但不应作为第一条训练线。Kaggriculture 官方 replay 和本地强规则 agent 能提供 BC 起点；然而数据覆盖不足会像 PTCG 卡组覆盖不足一样，让动物、长周期作物或罕见市场策略失效。必须先做覆盖账本，再用 self-play 和 specialists 纠偏。

## BC 数据 schema

每个 actor-visible decision 保存：

- episode/seed/player/step/day/hour；
- observation hash 和 schema version；
- 当前 job state；
- 候选任务列表、legal mask、被选 candidate/mode；
- 编译出的 farmer/hands/market raw action；
- 终局 W/D/L 和 banked cash；
- strategy tags：crop mix、animal、land count、hands peak、market sell timing、winner/loser。

如果 replay 只有 raw action而没有 job label，用确定性逆向标注器把连续移动和最终操作合并成 task；无法唯一解释的样本只监督 raw/low-level head，不强造高层标签。

## 覆盖门槛

训练前输出每类任务的 actor-side decisions、unique episodes、win/loss 状态数。建议最低覆盖：每个 crop、animal、market family、land/hire 阶段至少 100 episodes；不足项用规则 specialist 自生成。任何“Top player broad mix”都要按风格聚类，避免互相冲突的标签平均。

BC loss：task CE + target CE + mode/quantity CE，全部 mask；可加 value/未来状态辅助。rare family 用 capped inverse-frequency weight，避免 1 个罕见样本权重无限放大。

## IMPALA 系统

Kaggriculture 已有 end-to-end JAX PPO，单 3090 下另建 CPU actor farm 不一定更快。若采用 IMPALA，建议：

- 12900K 上 12–18 actor processes，只负责官方/规则对照或多策略 rollout；
- GPU inference server 批量 512–2048 requests；
- rollout 32–64；
- learner 使用 V-trace 修正 policy lag；
- shared-memory ring buffer，记录 behavior logits/version；
- lag 超阈值的 rollout 丢弃，不能只靠 V-trace 吞下无限陈旧数据。

更务实的变体是“JAX actors + learner 同 GPU，保留 V-trace/teacher 思想”，先测 PPO 与 off-policy learner 的真实 SPS 再决定。

## Teacher 与 opponent pool

delayed teacher 从 `N` transitions 前 checkpoint 加载，KL 只约束合法候选分布。BC 初期 KL 大，逐步转向 delayed RL teacher；不能永久拉向 BC，否则覆盖盲区无法突破。

对手池：live/frozen history/v16/animal specialist/market specialist/terminal specialist。每 10M transitions 固定 arena，若某策略族胜率跌幅>5pp，阻止 checkpoint 替换。

## 训练阶段

1. BC：只到 validation task accuracy 和 fixed arena 不再提升；保留 best，不追求拟合全部 replay。
2. value warmup：policy frozen，20–50M transitions。
3. RL：100M terminal W/D/L，teacher KL 逐步从强到弱；与 from-scratch PPO 同预算对照。
4. 若 BC→RL 在 100M 内明显领先且无 coverage blind spot，再扩到 300M–1B。

## 本机速度与复杂度

模型可控制在 2–5M，但 actor/inference/learner 调度复杂度高。预期 end-to-end 5k–25k transitions/s，低于现有简单 PPO 并不奇怪；优点是能复用 replay、actor 与 learner 异步。如果 100M wall time >6 小时且强度不优于 JAX PPO，应停止 IMPALA 工程，保留 BC checkpoint 给第 3/9 名路线。

## 成功/失败判据

成功不是 BC action accuracy 高，而是：BC→RL 比相同模型 from-scratch 更快达到固定 arena 强度；rare strategy panel 不弱；移除 BC KL 后不 collapse；对历史 checkpoint 不遗忘。任一不满足，就说明数据起点没有转化成可泛化策略。
