# 第 3 名：Reachability Tensor + PPO/PFSP

- 作者：Felix M Neumann
- 官方文章：[[3rd Place] Ab in den Orbit](https://www.kaggle.com/competitions/orbit-wars/writeups/3rd-place-ab-in-den-orbit)
- 发布时间：2026-07-08

## 一句话

把策略决策改写为“有语义的发射意图”，用 JAX 计算所有 source-target-action 的可达性/到达时间/所需舰数，把它同时用于特征、edge attention、mask 和动作落地；6.2M Transformer 以 PPO + PFSP 纯 self-play。

## Reachability tensor

核心张量形状约 `(B, P, P, S, 3)`，对每个 source、target、语义动作保存 ships、angle、arrival time。四种语义动作：

- send all；
- sortie：尽量发走但仍能守住 source；
- hold：恰好夺取并在未来 8 回合守住 target；
- kill-at-arrival：考虑已在途 fleets 后，恰好在到达时夺取。

它不只是 legality mask，而是整个模型的关系数据库。作者用 JAX 向量化 collision、Newton/fixed-point 求角度；以 24 seeds 的 forward simulation 测命中率和可用路线数量。作者也指出 JAX 大 kernel 的 XLA 编译时间在 Kaggle 提交侧危险，未来会考虑 Rust/C++。

## 状态与网络

- 48 tokens：4 player + 44 planet/comet；
- 8 层 actor trunk，critic 再加 2 层，hidden 192、expansion 4；
- 总参数约 6.2M；
- garrison/fleet size 用离散 embedding，精确到 384 后再分桶；
- incoming fleet 做 arrival calendar，2P horizon 24、4P horizon 16；
- global scalars 既广播进 token，也用 FiLM 调节每个 block；
- reachability edge features 投影成 Graphormer-style attention bias。

每个 owned source 输出 `{noop} ∪ {44 targets × 4 intents}`，共 177 类。action logit 同时包含 source-target bilinear、只在 reachable edges 上运行的小 edge MLP、语义 action bias。edge MLP 用 gather/scatter 避免对全部稠密边计算。

## 训练与评估

- PPO + GAE，纯 self-play；2P/4P 分开模型；
- 2P 约 8.4B steps，4P 约 2.7B；
- 早期用本地 RTX 3090，后换 5090 和两张 RTX 6000 Pro；
- 两张 6000 Pro 报告端到端 2P 19K SPS、4P 15K SPS；
- 1024 env、rollout 256、1 PPO epoch、minibatch 8192、clip 0.2；
- terminal reward；2P gamma 0.993，4P gamma 0.99；entropy schedule 是作者认为最关键的 knob。

PFSP 按历史 checkpoint 对当前模型的难度优先采样。为了免费从 rollout 得到 win rate，作者让同一对手固定跨 2 个 PPO update（512 steps），之后放弃未结束游戏再换对手。评估时 1024 局并行；4P 使用 1v3 clone 的评估事后被认为缺少多样性。

## 关键启示

- 最强的不是“大模型”，而是把动作意图和未来约束设计对；语义动作显著加快达到参考强度。
- entropy schedule、历史对手选择和严格 checkpoint pipeline 是 RL 主体，不是附属功能。
- `reachability tensor` 在 Kaggriculture 中不能照搬为距离矩阵，应迁移成“单位—任务—目标—deadline 的可行性/成本/收益张量”。
