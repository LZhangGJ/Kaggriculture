# 第 9 名：End-to-End JAX PPO

- 作者：Boey
- 官方文章：[9th Place Solution - End-to-End JAX PPO](https://www.kaggle.com/competitions/orbit-wars/writeups/9th-place-solution-end-to-end-jax-ppo)
- 开源代码：[CJBoey/kaggle-orbit-wars](https://github.com/CJBoey/kaggle-orbit-wars)
- 发布时间：2026-07-20

## 一句话

environment、feature、mask、rollout、optimizer 全部 JIT/vmap 到单张 4090/5090；用丰富 planet future/calendar 和自回归 pointer heads，2P/4P 分开 PPO + league/PBT。

## 全 GPU 热循环

作者先建立逐 state/action 的 reference-vs-JAX parity suite，然后把模拟、observation、mask、rollout 和优化全部放进 JAX hot loop，没有 Python 逐局模拟。最多 256 fleets，但作者观察 fleet 发射后不再决策，所以把全部 fleet 折成目标 planet 的时间信号，避免 44 planet + 256 fleet tokens。

## 状态

- planet token：位置、radius、garrison、production、active、relative ownership、comet；
- first-flip forecast：若无人再发射，第一次 owner flip 的 ETA/next owner/garrison；
- owner×ETA Calendar：11 个不等宽时间桶，每格 sum/count/max ships 和 fractional ETA；
- Planet Future：未来 29 个精确 turn + 1 个 tail，模拟 production/arrival/combat 后 owner/garrison/flip；
- comet path 多时距 lookahead；
- global token：step、angular velocity、comet timing、各相对玩家 garrison/in-flight/eliminated ledger。

Calendar 和 Planet Future 信息重叠，但作者消融发现一起保留更好；前者是便于读取的压力统计，后者保留因果顺序。显式 edge/reachability 在其实现中反而不稳，作者明确把它归因于实现可能不对，而非结论“edge 无用”。

## 动作与网络

最多 8 launches/turn，自回归 source/noop→destination→10 fraction，GRU context 串起 head 和 launch slots，同一 source 可多次使用。mask 与实际 angle 用同一 intercept solver，提交侧再 engine-exact refinement。

- 2P 最终约 5M，embed 256；4P 约 1M，embed 128；
- 4 层 self-attention；time signals 各自 Conv1D 后加回 planet token；
- 4P 额外 LSTM horizon 16，2P 无 recurrent；
- 2P 大模型从已训练 1M warm start。

## 训练规模

- PPO+GAE、single GPU、terminal-only；2P +1/-1，4P winner +1/losers -1；
- 2P 260 env×400 rollout；4P 220×480；1 update epoch；
- gamma 0.9999、λ0.95；
- dynamic per-head entropy target；末期降 LR 和 target entropy；
- 2P lineage 约 6.35B env steps，3.5K–9K SPS；4P 约 3.95B，3K–5K SPS。

作者报告 2P self-play 几乎与 80/20 league 一样强，本地到榜相关好；4P 评估极不稳定，1v1v1v1 与 2v2 对同 checkpoint 判断相反。

## 迁移价值

此方案与本项目现有 JAX simulator 技术方向最接近。最重要的是把可变、不可再决策的过程压成固定时间信号；在 Kaggriculture 中可把“已排定任务、植物/动物生产、城镇需求、市场订单后果”折进 plot/market future，而不是创建大量事件 token。
