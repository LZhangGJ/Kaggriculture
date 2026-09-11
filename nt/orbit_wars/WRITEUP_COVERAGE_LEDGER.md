# Orbit Wars 金牌 writeup 覆盖台账

核验日期：2026-08-11。

核验方法：以 Orbit Wars 官方最终 leaderboard 为金牌范围和名次真值；读取每个金牌行是否绑定 `description` writeup 链接，再逐页核对标题、作者和名次。最终第 1–19 名显示金牌，第 20 名开始不再显示金牌。

## 覆盖结论

- 金牌队伍：19 支。
- leaderboard 绑定公开 writeup：11 篇。
- 没有绑定公开 writeup：8 支（第 4、12、13、14、15、16、17、18 名）。
- 本目录逐篇摘要和迁移文档：11 + 11，覆盖所有可核实的公开金牌 writeup。

| 名次 | 队伍 | 官方 writeup | 本地来源摘要 | 本地迁移文档 |
|---:|---|---|---|---|
| 1 | Isaiah @ Tufa Labs | [Scaling Reinforcement Learning to the Stars](https://www.kaggle.com/competitions/orbit-wars/writeups/1st-place-solution-scaling-reinforcement-learnin) | [01](sources/01_SCALING_RL_SOURCE_DIGEST_ZH.md) | [01](migrations/01_SCALING_RL_TO_KAGGRICULTURE_ZH.md) |
| 2 | Hober Malloc | [2nd Place Solution for Orbit Wars](https://www.kaggle.com/competitions/orbit-wars/writeups/2nd-place-solution-for-orbit-wars) | [02](sources/02_MODERNBERT_PPO_SOURCE_DIGEST_ZH.md) | [02](migrations/02_MODERNBERT_PPO_TO_KAGGRICULTURE_ZH.md) |
| 3 | Felix M Neumann | [[3rd Place] Ab in den Orbit](https://www.kaggle.com/competitions/orbit-wars/writeups/3rd-place-ab-in-den-orbit) | [03](sources/03_REACHABILITY_PFSP_SOURCE_DIGEST_ZH.md) | [03](migrations/03_REACHABILITY_PFSP_TO_KAGGRICULTURE_ZH.md) |
| 4 | Jake Will | 未在官方榜单绑定公开 writeup | — | — |
| 5 | TonyK | [Orbit Wars 5th place solution](https://www.kaggle.com/competitions/orbit-wars/writeups/orbit-wars-5th-place-solution) | [05](sources/05_IMPALA_BC_SOURCE_DIGEST_ZH.md) | [05](migrations/05_IMPALA_BC_TO_KAGGRICULTURE_ZH.md) |
| 6 | flg | [[6th] RL + league + search](https://www.kaggle.com/competitions/orbit-wars/writeups/6th-rl-league-search-with-a-custom-edge-atte) | [06](sources/06_EDGE_ATTENTION_SEARCH_SOURCE_DIGEST_ZH.md) | [06](migrations/06_EDGE_ATTENTION_SEARCH_TO_KAGGRICULTURE_ZH.md) |
| 7 | Audun Ljone Henriksen | [How structured experiments saved my sanity](https://www.kaggle.com/competitions/orbit-wars/writeups/7th-place-solution-how-structured-experiments-sa) | [07](sources/07_STRUCTURED_JAX_PPO_SOURCE_DIGEST_ZH.md) | [07](migrations/07_STRUCTURED_JAX_PPO_TO_KAGGRICULTURE_ZH.md) |
| 8 | Ender | [How I Made Ender for <$200](https://www.kaggle.com/competitions/orbit-wars/writeups/8th-place-how-i-made-ender) | [08](sources/08_ENDER_MICROSTEPS_SOURCE_DIGEST_ZH.md) | [08](migrations/08_ENDER_MICROSTEPS_TO_KAGGRICULTURE_ZH.md) |
| 9 | Boey | [End-to-End JAX PPO](https://www.kaggle.com/competitions/orbit-wars/writeups/9th-place-solution-end-to-end-jax-ppo) | [09](sources/09_END_TO_END_JAX_SOURCE_DIGEST_ZH.md) | [09](migrations/09_END_TO_END_JAX_TO_KAGGRICULTURE_ZH.md) |
| 10 | Xiangyu Liu | [10th Place Solution](https://www.kaggle.com/competitions/orbit-wars/writeups/10th-place-solution) | [10](sources/10_STRATEGIC_ENV_SOURCE_DIGEST_ZH.md) | [10](migrations/10_STRATEGIC_ENV_TO_KAGGRICULTURE_ZH.md) |
| 11 | moriiiiiiiiim | [Standing on the Shoulders of Supergiants](https://www.kaggle.com/competitions/orbit-wars/writeups/11th-place-solution-standing-on-the-shoulders-of) | [11](sources/11_IL_LEAGUE_SOURCE_DIGEST_ZH.md) | [11](migrations/11_IL_LEAGUE_TO_KAGGRICULTURE_ZH.md) |
| 12 | One Man Wrecking Machine | 未在官方榜单绑定公开 writeup | — | — |
| 13 | Luca | 未在官方榜单绑定公开 writeup | — | — |
| 14 | M & J & M.ver2 | 未在官方榜单绑定公开 writeup | — | — |
| 15 | dragon warrior | 未在官方榜单绑定公开 writeup | — | — |
| 16 | Azat Akhtyamov | 未在官方榜单绑定公开 writeup | — | — |
| 17 | Vadasz & Ascalon | 未在官方榜单绑定公开 writeup | — | — |
| 18 | Slawek Biel | 未在官方榜单绑定公开 writeup | — | — |
| 19 | Golden Orbit Goblins | [19th Place Gold Writeup](https://www.kaggle.com/competitions/orbit-wars/writeups/19th-place-gold-writeup) | [19](sources/19_GPU_SIM_BC_PPO_SOURCE_DIGEST_ZH.md) | [19](migrations/19_GPU_SIM_BC_PPO_TO_KAGGRICULTURE_ZH.md) |

## 边界说明

“没有绑定公开 writeup”只表示在本次官方榜单核验中没有文章链接，不表示这些选手从未在评论、代码、聊天或其他平台分享思路。本研究不把二手描述扩写成一篇不存在的官方金牌文章。

第 11 名文章包含一张对若干前排方案的二手比较表，其中提到第 4 名的一些训练规模；本目录只在第 11 名来源摘要中把它标为“第 11 名作者的比较”，不会据此伪造第 4 名独立 writeup。
