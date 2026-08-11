# 第 2 名：ModernBERT + 1D-CNN + PPO

- 作者：simjeg（Hober Malloc）
- 官方文章：[2nd Place Solution for Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars)
- 开源代码：[SimJeg/orbit-wars](https://github.com/SimJeg/orbit-wars)
- 发布时间：2026-07-08

## 一句话

把复杂动作极端压缩为“每个 body 不动或 all-in 发向 ETA<20 的目标”，用未来 20 步序列的 1D-CNN + 4.3M ModernBERT，最后从零 PPO self-play 10B steps；早期 BC 只用于找对状态/动作设计。

## 状态

每个 planet/comet 取 10 个特征：time、production、极坐标、自己/中立/各对手 ships、capture 所需额外 ships。作者不只取当前值，而是假设之后没有新发射，向前模拟 `T=19`，得到每个 body 的 20-step time series。它显式表示时间/几何，也把已在途 fleet 的未来影响折进序列。

每个 body 序列经过 4 个 residual Conv1D block（kernel 5、GELU、LayerNorm）、global average pool 和 128→256 投影。最多 44 bodies，再送进 ModernBERT XXS：7 层、4 heads、hidden 256、全局 attention、无 token embedding/positional encoding。

## 动作和 mask

每个 body 两阶段：

1. launch head：是否发射；
2. target head：对其他 bodies 做 attention 选目标。

只允许全量发射，并 mask 非己方 source 和无法在 20 步内命中的 target。推理阈值约 53%/56%。2P 用 4 种旋转 TTA；3/4P 还做对手槽 permutation，共 8 views。

## BC 阶段

作者从 189K episode 中过滤约 20K，得到约 5M samples；保留高分玩家或击败高分玩家的轨迹，并偏向几乎全程 all-in 的示范。masked BCE 训练 launch，categorical CE 训练 target。BC 在比赛中先达到 Top10，也帮助确认：额外 fraction head、T>20 和不同模型大小没有带来足够收益。

最终提交没有使用 BC checkpoint。作者在截止前 5 天发现从零 RL 很快超过 IL 初始化，说明 BC 在此方案里主要是低成本表示搜索工具，而非最终能力上限。

## PPO 与工程

- PufferLib 异步实现；feature/mask 从 Rust 移到 C，模型从 Torch 移到 CUDA；
- 8×H100 约 40K steps/s；
- 1024 agents/GPU、rollout 128、minibatch 4096；
- gamma 0.995、λ 0.97、clip 0.2；launch entropy 0.01、target entropy 0.002；
- 40% 4P；旋转和对手 permutation augmentation；
- 超过 40 steps 无动作就截断；500-step 才赢奖励降为 0.5；
- 3 个 24 小时阶段：3B + 3.5B + 3.5B，学习率 1e-3→3e-4→1e-4。

最终是一个 4.3M 模型同时玩 2P/4P，约 10B self-play steps。作者早期 4P 曾用 frozen pool，但最终 run 放弃。局部评估使用 Rust arena，可做固定 matchup 和 OpenSkill 本地榜。

## 关键启示

- 极强动作约束能换取巨大的样本效率，但会形成盲区：看不到 ETA>20，不能分批发射。
- BC 很适合低成本筛特征/动作，但不保证最终 RL 仍需要 BC。
- 本方案训练算力远超单 3090；可迁移的是“小模型 + 强时间投影 + 极简动作”，而不是 10B steps。
