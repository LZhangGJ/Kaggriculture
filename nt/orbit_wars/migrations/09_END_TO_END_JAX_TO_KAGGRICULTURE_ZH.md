# 第 9 名路线迁移：全 JAX Future-State PPO 主线

源方案见 [第 9 名摘要](../sources/09_END_TO_END_JAX_SOURCE_DIGEST_ZH.md)。

## 迁移结论

技术栈与本项目现状最匹配：JAX 规则核心、features、masks、rollout、optimizer 全在 RTX 3090。建议把它作为长期主训练框架，但模型先控制在 1–3M，并借用第 3/7 名的语义任务和 top-k，避免一上来复刻数十亿步。

## 时间信号迁移

Orbit 的 fleets calendar/planet future 对应四类 Kaggriculture fixed-shape signals：

1. **Plot Maintenance Calendar**：未来 1/6/12/24/48/72 turns 的 WATER/FEED/CARE、production、decay、逃跑风险。
2. **Plot Future**：在当前 jobs 继续且不新增任务时，逐日模拟 tile type、age、yield、fertilization、animal pending bonus。
3. **Logistics Calendar**：units 当前路径、到 target/depot 的 ETA、背包交付和 shed 占用。
4. **Market Future**：已知 town demand、自己的计划供给、无对手新订单情景下的库存/价格；再附 opponent public-history belief 区间，不能用隐藏真值。

Calendar 是紧凑统计，Future 是因果 rollout；二者可同时保留，再做单独/联合消融。

## Token 与网络

- 72 plot tokens；
- 1 global；
- fixed masked unit slots；
- 9 market tokens；
- 每类 time signal 用小 Conv1D 编码后 residual 加入对应 token；
- 4 层、d128 起步约 1M，晋级后 d192/6层约 3–5M；
- 2D farm coordinates + side embedding；
- 可选 unit-task edge adapter，只有在消融为正后保留。

动作建议不是第 9 名的 source/destination/fraction 原样，而是第 3 名候选任务 + 第 8 名轻量自回归。最多 2 task edits/turn + 2 market intents，未编辑任务继续。

## PPO/PBT

- 1024–4096 batched env；rollout 64–128；
- BF16 trunk、FP32 logprob/value/optimizer sensitive math；
- 1 PPO epoch 起步；
- terminal W/D/L，gamma 0.999/1.0 对照；
- per-head target entropy，维护/market/投资分开；
- history/PBT pool 20–40%，但 fixed diverse panel 决定晋级。

不要复制 Orbit “2P self-play 足够”的结论：Kaggriculture 虽是 2P，却有非平稳市场和多策略经济循环，必须保留历史/专家对手。

## 规模路线

1. JAX feature builder parity：对同 state 的 Python reference feature 实现逐字段比较。
2. 0.6M policy 30M transitions，验证合法率、吞吐和能完成基本种植/卖出。
3. 1–3M 100M，固定 arena。
4. 只对胜出的 future/edge/action ablation 做 300M。
5. 若 300M 曲线仍明显上升，1B final；否则转搜索/league，不盲目堆步数。

## 本机吞吐目标

已有实测：sim-only 1.56M/s、608K policy+sim 188K/s、full PPO 57.3K/s。新 token Transformer 目标端到端 10k–40k/s；100M 0.7–2.8 小时，1B 7–28 小时热循环。若低于 8k/s，优先 profile future Conv、candidate generation、autoregressive slots，不能仅以 GPU 利用率判断。

## 提交边界

JAX 首次 JIT 可能超过 1 秒动作预算。最终包必须：固定 shape、启动 warmup、删除 value/aux heads、测试最坏 units/tasks 数量、准备纯规则 fallback。训练 JAX graph 与提交 graph 分离，但 feature/action parity 必须相同。
