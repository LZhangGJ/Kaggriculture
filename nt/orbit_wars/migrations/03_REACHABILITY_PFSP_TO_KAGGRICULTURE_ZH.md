# 第 3 名路线迁移：Schedule-Feasibility Tensor + PFSP

源方案见 [第 3 名摘要](../sources/03_REACHABILITY_PFSP_SOURCE_DIGEST_ZH.md)。

## 迁移结论

这是本项目最值得优先实现的建模主线。Orbit 的 reachability tensor 在 Kaggriculture 中改为 `schedule feasibility tensor`：对每个单位、候选任务、目标、执行模式，给出完成时间、资源成本和结果 margin；它同时驱动特征、attention bias、action mask 和动作编译。

## 张量定义

建议稀疏逻辑形状：

`F[B, unit, candidate_task, mode, fields]`

每条候选 fields：

- `travel_steps`、`finish_step`、`deadline_slack`；
- `cash_cost`、`seed/item/backpack cost`、`shed capacity delta`；
- `expected_yield`、`bankable_cash_low/base/high`；
- `maintenance_risk`、`terminal_liquidation_slack`；
- `legal_now`、`feasible_before_deadline`；
- `opportunity_cost`：占用该单位后错过的紧急任务。

mode 不是 25/50/100%，而是有经济语义的意图：

- `MINIMUM`：完成任务所需最少资源；
- `SAFE`：完成并留出一次维护/价格下跌 buffer；
- `MAX_ROI`：在当前预算内最高保守收益；
- `LIQUIDATE`：优先在终局前入仓并卖出。

例如 `HARVEST_AND_SELL(melon,tile,SAFE)` 会计算到 tile、HARVEST、回仓 DROP、市场 SELL 的最早完成 step，而非只看作物当前售价。

## 网络

- plot、unit、market、global tokens；d192，6–8 layers，约 3–6M；
- edge bias 由 unit→task 的距离、slack、cash margin、maintenance urgency 投影；
- action head对每个 unit 输出 `{continue/noop} ∪ {candidate × mode}`；
- 对稀疏合法 candidates gather→edge MLP→scatter，避免对所有单位×36格×任务×商品稠密计算；
- global FiLM 注入 day/hour、终局距离、土地/现金阶段和市场 regime；
- value 预测 win probability；辅助预测下一日维护违约、未来现金和价格。

候选生成器先覆盖硬任务（今日必须 WATER/FEED、即将满产/衰减、终局必须 DROP/SELL），再加入 ROI top-k 和少量随机合法候选。这样不会让模型把“救活植物”丢在 top-k 外。

## PFSP 和 self-play

每 5–20M transitions 保存 checkpoint。PFSP 权重可按 `1 - winrate(current, opponent)` 的单调函数分配，但要设均匀探索下限。一个 rollout segment 内固定对手；不要中途替换导致 terminal attribution 错乱。

对手池同时含：历史 current lineage、规则 v16、纯作物、纯动物、市场压价、终局清仓 specialists。PFSP 只负责采样难度，不能把低频风格永久饿死；每个 specialist 设最低 5% 配额。

## 训练计划

1. 只实现 8–12 类任务和 `MINIMUM/SAFE` 两 modes，0.8–1.5M 模型，30M transitions。
2. parity 检查任务编译后的 raw action，尤其同回合 PLANT 原子规则和 BUY 后不能同回合使用。
3. 100M PPO，对固定 512-game panel；记录每类任务选择率/mask率/完成率。
4. 加 `MAX_ROI/LIQUIDATE`、market quantity 和 auxiliary future heads，单因素各跑 100M。
5. 只有 tournament 通过才合并成 3–6M final。

## 奖励与 entropy

主 reward 为 terminal W/D/L。可在前 10M curriculum 观察 `task completion`/`maintenance violation`，但默认不进入 policy return。按 action family 单独 entropy target：维护任务低 entropy，战略投资/市场高 entropy；末期不允许把所有 heads 一起降到近零。

## 本机效率与难点

主要工程难度是 JAX 稀疏候选和可行性 kernel，不是 PPO。建议目标 8k–30k transitions/s；100M 约 0.9–3.5 小时热循环。与第 1 名相比参数小、样本效率高；与第 9 名相比需要更多规则工程，但更符合 3090。

## 成功判据

除胜率外，必须检查：硬维护候选召回=100%；合法 action rate=100%；任务预计 finish 与模拟实际 finish 一致；按任务族无系统性 blind spot。只要 tensor 不可靠，模型分数没有解释意义。
