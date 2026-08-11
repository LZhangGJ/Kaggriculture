# 第 1 名路线迁移：缩小规模，保留 Scaling 纪律

源方案见 [第 1 名摘要](../sources/01_SCALING_RL_SOURCE_DIGEST_ZH.md)，共用状态/动作底座见 [迁移共用规格](../MIGRATION_COMMON_SPEC_ZH.md)。

## 迁移结论

不建议复刻 `200M + 15B + B200 集群`。建议复刻它的实验逻辑：先证明小模型 PPO 能学，再逐级加参数/steps；保持较少、通用的 entity features；把确定性物流和 legality 放到编译器；用历史 checkpoint 和严格晋级赛防止“只会打自己”。

## Kaggriculture 状态

把 Orbit 的 entity set 换成：

- 72 个 plot tokens（双方农场）；
- 1 个 global/economy token；
- farmer + hand tokens；
- 9 个 market-product tokens；
- 4 个 plan/scratch tokens，让 Transformer 有共享计划空间。

相对第 3/6/9 名，本路线减少手工 ROI 特征，只保留规则事实：时间、位置、库存、生命周期、价格、任务和确定性 deadline。允许模型自己组合“卖价高但来不及收获”“动物维护挤占 melon 浇水”等关系。

## 动作

冠军从 raw angle 改成 target planet 才变强；Kaggriculture 对应改动是从 raw NORTH/WATER/SELL 切到 task target：

1. 对每个 unit 选 `continue/noop/new-task`；
2. 若 new-task，选 task type；
3. 选 target plot/depot/product；
4. 选 quantity/budget bucket；
5. 确定性 scheduler 生成逐回合路径和原始动作。

第一版每回合最多改 2 个 unit tasks + 2 个 market intents。对明显非法动作做 mask；与冠军文章不同，本项目没有算力去赌“让网络自己学会所有 legality”。

## 模型与训练规格

建议递进而非一次定大：

| 阶段 | 参数 | transitions | 目的 |
|---|---:|---:|---|
| S0 | 0.6–1M | 10–30M | PPO/动作编译器闭环、能击败 random |
| S1 | 2–3M | 100M | 与规则 agent、v16 做固定赛程 |
| S2 | 5–8M | 300M | 只在 S1 明确随规模变强时启动 |
| S3 | 10–15M | 1B | 仅在端到端速度和 scaling curve 均支持时 |

网络使用 6–10 层、d128–256 的 Pre-LN entity Transformer；actor/value 同 trunk，推理删除 value/aux heads。一次 forward 同时输出双方动作只适合 self-play 采样，部署仍只能基于己方可见 private observation，必须确保没有把对手私有状态混进共享计算。

PPO terminal W/D/L，rollout 64–128；训练对手至少包含 frozen best、历史 checkpoint、规则 specialists。候选对 previous best 胜率达到阈值才能进入池，但不能只做 head-to-head；还要过固定多样 panel。

## Kaggriculture 特有修正

- 只有 2P，避免了 Orbit 4P general-sum，但动态市场让 self-play 仍可能形成“双方都只种一种作物”的共谋式循环。
- 不能用 raw terminal cash 当 value 的唯一尺度；actor 目标是 W/D/L，value 建议预测 win probability，同时 auxiliary 预测终局 cash 供表示学习。
- 719 步固定长局无法像消灭星球那样自然 early terminate。可在胜率 critic 极端且现金/剩余可实现上界严格证明无法翻盘时截断；在没有保守证明前只做 value bootstrap，不能改规则终局。

## 本机可行性

S0/S1 可行；S2 需 benchmark；S3 风险高。按共用规格的未实测区间，2–3M 结构化模型 100M 热循环约 0.6–1.9 小时，实际连同 arena 可在半天内形成一个实验结论。200M/15B 不在本机合理范围。

## 晋级判据

只有同时满足才扩模：相同 100M 数据预算下大模型显著胜小模型；吞吐下降后的 wall-clock 仍有净收益；对至少 4 个不同策略族无回退。否则停止 scaling，转向第 3/6/7 名的结构设计。
