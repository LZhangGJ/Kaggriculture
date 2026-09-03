# Round2 语义拳法重测结果

## 结论

训练门失败：当前 25 套语义拳法在 12 条训练对手路线中没有覆盖任何一条，因此无法形成“由同一拳法克制”的有效 response cluster。按预注册规则，没有运行 validation 或 holdout。

这不表示 NT 无法生成克制拳法；它只说明此前在单一 discovery seed/座位/对手状态上找到的这批候选，换到新 seed 和双座位后不稳健。

## 这次实际测试的拳法

- 使用当前 v3 G003（`103928643:1`）作为固定开局，执行到 step 143。
- 在 day 6、9、12、18、24 保存并调用五段语义 `PlanDelta` 意图，而不是保存动作带或候选 rank。
- 每个检查点都根据当局实时盘面、库存和商店重新生成可行候选，再按稳定意图字段精确匹配，并由 Candidate8 规划当局动作。
- 所有 24,000 个 rank telemetry 均为 `-1`，确认没有按 rank 重放。
- 未匹配的意图不污染状态，交回普通实时规划器；其 telemetry 哈希已改为空值。

稳定意图由 `family_id`、8 维目标变化、延迟、日程/市场/恢复 profile、后缀项目、市场物品和恢复问题组成。容量、估值、预计现金成本、预计动作负载和 signature 会随实时状态重新计算，不用于跨局身份匹配。

## Discovery 等价门

- 48/48 个旧搜索结果完整复现：reward、719-step 联合轨迹、family、signature、PlanDelta、动作哈希和固定开局前缀均一致。
- 48 个搜索选择去重为 25 套语义拳法。
- 79/79 个 `family + signature` 均能唯一反解为稳定意图。

## 新 seed、双座位训练结果

共运行 4,992 局：`25 fists × 12 opponents × 8 seeds × 2 seats = 4,800` 个 treatment 对局，另加 192 个 Opening A baseline 对局。

| 指标 | 结果 |
|---|---:|
| PlanDelta 阶段匹配 | 18,083 / 24,000 = 75.3458% |
| 五段全部匹配的对局 | 26.0833% |
| 匹配后实时 delta 相对 discovery 发生变化 | 16,293 / 18,083 = 90.1012% |
| 匹配后 signature 发生变化 | 40.3030% |
| 300 个拳法–对手单元中通过 score 门槛 | 1 |
| 通过相对 Opening A 的 uplift 门槛 | 0 |
| 通过正 mean-margin 门槛 | 0 |
| 同时通过三项、可用于聚类 | 0 |

逐检查点匹配率：day 6 为 91.3333%，day 9 为 62.4167%，day 12 为 62.3125%，day 18 为 88.3125%，day 24 为 72.3542%。

最接近门槛的是拳法 `0450d0a3cfa9...` 对 P02：score `0.625`，但 Opening A baseline 为 `1.0`，所以 uplift `-0.375`，mean margin `-5,866.5625`。全部单元中的最高 mean margin 仍只有 `-2,774`。

另有 14 个拳法–对手单元在 16 局中五段全部匹配；其中最好 score 仅 `0.25`、最好 mean margin `-10,641.125`。因此失败不能主要归因于意图未匹配，而是这些候选本身无法在新随机条件下保持克制效果。

## 下一步

不要降低门槛，也不要打开 validation。应对当前 12 条 residual 路线重新运行 NT，但把搜索目标改成多 seed、双座位的稳健目标，直接优化 score、相对 Opening A 的 uplift 和 reward margin。每得到新拳法就重算 response coverage，把能被同一拳法稳定击败的路线合并；只有训练集得到非空 portfolio 后才进入 validation。

本轮没有重新运行昂贵的 NT 搜索，只进行了语义化复现、状态自适应执行和冻结训练矩阵重测。
