# 第 10 名：MapCache + StrategicEnv + JAX PPO

- 作者：Xiangyu Liu
- 官方文章：[10th Place Solution](https://www.kaggle.com/competitions/orbit-wars/writeups/10th-place-solution)
- 开源代码：[xiangyu-liu-dev/orbit-wars](https://github.com/xiangyu-liu-dev/orbit-wars)
- 发布时间：2026-07-21

## 一句话

把真实几何预计算进 MapCache，训练只跑 ownership/garrison/production/arrival/combat 的 StrategicEnv；JAX PPO 学 source→target→语义 ship bin，并用 autoregressive slots 构造多动作。

## 系统拆分

训练链：`real geometry → MapCache → StrategicEnv → JAX PPO`。MapCache 预计算昂贵的几何，StrategicEnv 的动作只是 `(source, target, ships)`；提交侧 C++ intercept solver 再转成 `[source, angle, ships]`。代价是 map pool 不够大时可能过拟合 cache，而且 parity 要同时覆盖 cache 和真实引擎。

## 状态与网络

44 planet/comet entity；fleet 折入：

- relative ownership、ships、production、position；
- 4 个 incoming pressure windows：0–5、6–10、11–15、16–20；
- no-op future 在 +5/+10/+15/+20 的 active/owner/ships/position。

每实体 MLP→self-attention→entity token；global/grouped summary 给 value。最终约 d_model384、2 attention blocks、8 heads、约 4M 参数。

动作按 source→target→fraction/noop。ship bins 同时包含 source-relative 25/50/75/100% 和 target-relative 0.5×/1×/1.5×/2× target ships +1。fraction head 直接得到 arrival time、selected/source/remaining/target ships、production、capture/reinforce margin、over-target ratio 等算术特征。

## 同回合 autoregression

每个 slot 重新选择 source、target、fraction；发射后立即 patch 临时 strategic state，减少 source ships，更新 target arithmetic，再给后续 slot。2P 最多 4 actions，4P 最多 3。作者比较“每 slot 重跑全 trunk”和“只重跑 policy head”，后者 2P 更快且榜上更好；全 autoregression 的收益不足以抵消成本。

## 训练与评估

- JAX rollout/update，BF16 网络、FP32 PPO-sensitive math；
- terminal reward：unique win +1、tie 0、loss -1；
- 先训练一动作/turn 的最小版本，再逐步增加 slots；
- 2P/4P 分开；末期两台 RTX 5090 分别训练；
- arena watcher 在 held-out maps 评估、维护排名，并把 anchor checkpoints 放回 self-play。

文章没有给出统一步数和 SPS，不能从名次反推其效率。

## 迁移价值

Kaggriculture 没有 Orbit 连续拦截，但有大量静态距离、生产日历和确定性订单规则。可预计算 `unit-position→tile` 距离、tile→仓库距离、作物/动物生产模板、市场价格表，形成 ScheduleCache；RL 只处理任务级经济状态。作者关于“policy-head-only autoregression 往往够用”的负结果对本项目尤其重要。
