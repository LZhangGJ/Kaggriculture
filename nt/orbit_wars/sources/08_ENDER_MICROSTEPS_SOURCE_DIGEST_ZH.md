# 第 8 名：Ender 的自回归 Micro-steps

- 作者：Billy Bradley（sinkingpoint）
- 官方文章：[How I Made Ender for <$200](https://www.kaggle.com/competitions/orbit-wars/writeups/8th-place-how-i-made-ender)
- 开源代码：[sinking-point/ender](https://github.com/sinking-point/ender)
- 发布时间：2026-07-08

## 一句话

在 3080 10GB + 11 天租用 4090、约 170 美元预算下，把一个游戏 turn 拆成最多 16 个可变 micro-step，自回归构造多次发射；JAX PPO + 历史 league，再加轻量 test-time search。

## Micro-step 动作

每个 turn 反复：

1. 全局选 launch/halt；
2. 若 launch，联合选择 origin 和 20/40/60/80/100% fraction；
3. 只对该 origin-fraction 计算 reachable targets/ETA；
4. 选 target 或 abort；
5. 把新 fleet 临时写回 observation；
6. 最多 16 次，直到 halt。

PPO trajectory 以 micro-step 为一步，而非游戏 turn。因先选 origin-fraction，再只对 43 个 targets 精算，把原本约 44×43×5 个候选缩到 43；允许同 source 多次发射。

## 特征与模型

planet token 包含 owner、production、garrison、velocity、radius、turn、incoming、未来 ceasefire 投影等；incoming fleet 先按规则抵消，编码成未来 24 turns 的净到达 bins。future feature 同样给未来 24 turns 的 owner/garrison，解决“早一回合不够、晚一回合丢先手”。坐标用 2D RoPE；所有 seat canonicalize 到 player0 视角。

核心为 4 层、d_model 192 Transformer。launch/halt、origin-fraction、abort、value 都是简单 linear；target MLP 接 target hidden、fleet size、ETA、能否超过防守、source/target ceasefire future。

## 训练

- PPO + GAE，2P/4P 分开；
- 2P：2048 env、rollout 256 microsteps；4P：384 env、512 microsteps；
- gamma 0.998、λ 0.95、2 PPO epochs、minibatch 2048；
- 2P 约 1B env steps/3.1B samples；4P 约 413M env/1.5B samples；
- 4090 报告约 15K microsteps/s、峰值约 4K env steps/s（含 PPO）；
- final 2P run 3.4 天、约 51 美元；4P 全在 3080。

为避免小舰队 spam，作者没有简单使用 uniform entropy，而是对 launch/halt 和 fraction 使用相对先验的 KL：halt 初始 0.9，fraction prior `1:1:1:1:10` 偏向 all-in。后期降低系数。4P 一半 env 在 50 turns 截断并 value bootstrap，以提高 early-game 样本占比。

league 用历史 checkpoint 和少量旧强 run，按近期 winrate 优先难对手。

## 推理搜索

- sampled action search：采样 10 个整回合动作，再用小型蒸馏 opponent policy 抽几个对手动作，推进一步后按 mean value 选；
- launch/halt search：只在首个非 abort micro-step 分 halt/launch 两长枝，greedy 推演到 fleet 命中后 3 turns。

GPU 训练用 256 directions 并行试射；CPU 提交改用碰撞窗口/切线求解和 occlusion sweep。

## 迁移价值

这是“联合动作不是独立 heads”的强案例。Kaggriculture 同回合 farmer、hands、market 有资源依赖，micro-step 临时账本可以保证种子、现金、仓容和订单顺序一致；但多次重跑 trunk 会显著降低吞吐，应先做 policy-head-only 自回归。
