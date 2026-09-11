# 第 7 名：Structured Experiments + JAX PPO

- 作者：Audun Ljone Henriksen、Eirik Torp
- 官方文章：[How structured experiments saved my sanity](https://www.kaggle.com/competitions/orbit-wars/writeups/7th-place-solution-how-structured-experiments-sa)
- DOI：[10.34740/KAGGLE/W/104983](https://doi.org/10.34740/kaggle/w/104983)
- 发布时间：2026-07-23

## 一句话

JAX 环境 + 约 9M Transformer + analytic planner + terminal-only PPO；真正特别之处是近 200 个单因素实验，每个 100M steps，再用 512 局/matchup round-robin 决定保留或丢弃。

## 表示与动作

- 每个 planet 一个 token；fleet 折进其目标 planet；
- source-target pair 特征作为 attention bias；
- global features；总计约 60 个概念特征、138 channels；
- 训练模型约 9M，部分 auxiliary heads 在推理删除。

每个 source 先选 target/noop，再选 send bucket：20/40/60/80/100% 或“到达时恰好夺取”。analytic planner 给每个候选求 angle/ETA 并 mask 不可命中。为了避免 48×48×6 全部求解，先由模型 logits 选 top4 target，再加 4 个随机 target，只对这 8 个算 planner；作者逐步从 16+8 缩到 4+4，未见明显回退。

planner 用 Newton-style lead pursuit 和 closest-approach collision check。以 3600 angles 暴力 oracle 检验：文章报告建议 shot 约 99% 命中，能找出真实可行 shot 的约 93–95%。

## 训练与实验治理

- 2P 2.2B steps，约 5 天；4P 1.6B，约 3.5 天；两张 GPU；
- terminal-only reward，无 shaping；
- 4P winner +1、loser -1/3；
- 每个候选改动独立实现、固定 100M 预算；
- 与 baseline 和近期 runs 做 round-robin，约 512 games/matchup；
- 观察特征是最大杠杆，其次模型规模和 planner。

作者公开的负结果包括：double batch 41.4%、dominance shaping 34.6%、short rollout 33.4%、训练对 live self 20.7%、8-model league 17.6%（均为相对当时 baseline 的局部实验结果，不能泛化为算法定理）。

## 过拟合与辅助任务

2P 只对最多落后 1M steps 的 frozen self、末期降 entropy/LR，并且评估也只看自家模型，导致对 Top10 多数对手很差。4P 加入全历史 opponent pool 后从坏局部最优明显恢复。

辅助头预测未来 2/8/32/64 turns 的 owner、garrison、production 等，target 直接来自 rollout，推理丢弃。early reset 在玩家失去最后 planet 时结束，不再等无望 fleet 飞完，提高有效样本密度。

## 迁移价值

文章最可迁移的是实验纪律：单因素、固定预算、固定 tournament、保留负结果；其次是“模型先粗排 top-k + 少量随机探索 + 精确候选求值”。它特别适合 Kaggriculture 任务候选很多、精确日程计算昂贵的场景。
