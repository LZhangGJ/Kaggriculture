# 第 6 名：Edge Attention + League + 2-step Search

- 作者：flg（Ferdinand Limburg）
- 官方文章：[[6th] RL + league + search](https://www.kaggle.com/competitions/orbit-wars/writeups/6th-rl-league-search-with-a-custom-edge-atte)
- 开源代码：[LeFiz/OrbitWars_PPO](https://github.com/LeFiz/OrbitWars_PPO)
- 发布时间：2026-07-24

## 一句话

在本地 RTX 3090 起步的资源效率路线：2.5M edge-attention Transformer、C++ forward prediction、async PPO/self-play，4P 加手工历史 league，2P 推理再做窄 2-step rollout search。

## 特征与网络

作者不输入绝对 x/y 或绝对 player_id；位置关系全部成为 A→B edge 特征，玩家按起始位置定义相对槽。fleet 不作为 token，而被 forward prediction 折入 23 个 arrival buckets。

每条 source-target edge 描述 travel time、不同 send fraction 是否可到、到达后能否夺取/守住。网络主体为 4 层 Transformer、hidden 256、8 heads、FF expansion 2，总约 2.5M。custom attention 显式读取 edge；policy head 对 source-target 和 25/50/75/100% fraction 评分，另有 noop。value 用 51-bin Gaussian histogram 预测胜率，作者报告它比 MSE 稳定。

## 训练

- 2P/4P 分开模型；2P self-play，4P self-play + 简单历史 league；
- async PPO + GAE，terminal game outcome；
- batch 1024、single PPO pass、clip 0.2；
- gamma 2P 0.999、4P 1.0；λ 随阶段 0.9–0.98；
- joint action ratio 做 PPO clipping；
- EMA advantage normalization 支持较小 batch。

作者先训练 2-layer/hidden128 小模型，用 production-delta dense capture reward 快速学会；一周后通过 teacher KL 把知识转给大模型。最终 checkpoint 用 5 个时间相近权重做 SWA，候选由大规模 all-vs-all tournament 选择。

开发前半在本地 3090，后半租 1×A100。尽管 C++ 重写和优化，actor 仍 CPU-bound，约 800–2000 SPS。这个数据对本项目很重要：GPU 并非天然等于环境在 GPU；该方案的主环境/forward features 仍由 CPU C++ 供给。

## 推理搜索

4P 直接 argmax。2P 先试一层 restricted matrix game/CFR，收益有限；最终用更简单的 2-step greedy rollout：抽 5 个己方动作，对手取 argmax，推进后双方再各走一步，以 value 选动作。作者报告约 +30–40 leaderboard points，主要避免“刚夺下就被反夺”的短视错误。

## 有效与无效

有效：相对特征、edge attention、forward prediction、Gaussian histogram value、早停已决定对局、4P league、2P 小搜索。

无明显收益：小于 100% 的 fraction（模型几乎总 all-in）；4P 搜索局部变强但榜上变弱；过窄历史池仍会让 4P 害怕弱对手/特殊风格。

## 迁移价值

这是最符合单 3090 条件的 Orbit 金牌范式之一。可直接迁移的是“小模型 + 重特征 + edge + 窄搜索”，不是其 Orbit 的 C++ 几何。Kaggriculture 已有全 GPU JAX 环境，因此可把 CPU feature bottleneck 改成 GPU schedule-feasibility kernel。
