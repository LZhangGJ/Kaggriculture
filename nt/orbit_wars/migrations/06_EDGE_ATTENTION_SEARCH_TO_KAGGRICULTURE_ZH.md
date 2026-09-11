# 第 6 名路线迁移：Unit–Task Edge Attention + 窄规划搜索

源方案见 [第 6 名摘要](../sources/06_EDGE_ATTENTION_SEARCH_SOURCE_DIGEST_ZH.md)。

## 迁移结论

这是本机最优先的“快速上分工程路线”之一：小模型、重规则特征、历史 league，推理用窄 2-step/日程 rollout 修正短视错误。与第 3 名可共享 schedule tensor；区别是第 6 名更强调小网络和 test-time search。

## Edge 设计

节点：units、plots、market products、global。关键 unit→task/plot edge：

- shortest-path distance 和预计 finish step；
- 背包是否已有材料，若无则 depot 往返成本；
- 是否赶得上 day-end WATER/FEED、production、decay、terminal sell；
- 任务后的 inventory/shed/cash margin；
- 与其他 unit task 的冲突或协同。

plot→market edge：预计 harvest step、入仓 step、该时点基础/压力情景价格、终局剩余 slack。market↔market/global edge 表示同回合订单顺序和双方共享价格冲击。

网络建议 4 层、d192–256、4–8 heads、2–4M。edge feature 作为 attention bias；policy 对每个 unit 的 top-k candidate task 评分。value 用 51-bin distributional head 预测 win-probability/logit 或终局 cash advantage bins，再映射胜率；必须与普通 scalar value 做消融，不能直接照抄 Gaussian histogram。

## 小模型→大模型

先用 0.5–1M 小模型和 curriculum dense signals（完成维护、成功入仓）快速确认行为，再用 teacher KL 转到 2–4M 主模型。转移后主 PPO reward 只保留终局 W/D/L。dense 版本不能直接晋级，避免学会刷代理指标。

## League

Kaggriculture 只有 2P，但市场导致策略循环：一种 policy 可能专门压某商品、却被多元生产克制。历史池每类至少保留：早期便宜策略、动物、长周期作物、market timing、当前 champion。采样不能只按全局 Elo，额外按“当前模型最弱 matchup”优先。

## 窄搜索

不要搜索 719 步。只在高影响节点触发：

- day/hour 边界；
- 是否 BUY_LAND/HIRE/BUY_ANIMAL；
- 高价商品批量 SELL；
- plant/animal 投资；
- 终局 48 turns 内。

每个触发点取 policy top-4 ego macro actions，对手取 argmax + 2 个高概率动作，JAX 并行 rollout 6–24 turns，使用 value + 已 bank cash 选行动。对完全私有的对手未来订单只能按 opponent model 分布抽样，不能读取隐藏库存。

优先实现 2-step decision lookahead：走一个宏动作，双方执行若干回合到下一个决策点，再各走一次。它主要应避免“买动物后没有时间/小麦维护”“收获后回仓太晚”“卖出导致价格崩但下一批更大”等短视。

## 性能预算

无搜索训练目标 10k–40k transitions/s；2–4M 模型 100M 约 0.7–2.8 小时热循环。搜索只用于 arena/提交，按 trigger 稀疏调用；目标平均每 turn 额外模拟 <64 branches×steps，并实测提交 CPU 时间。若搜索让本地胜率只涨 <2pp 或提交超时风险增加，则保留纯 policy。

## 验证

- edge finish/cost 对 JAX 实际 rollout 的误差；
- search 使用的 opponent-visible state 合规；
- 搜索开/关至少 2048 座位平衡 games；
- value calibration 按 day、cash regime、strategy family 分桶；
- 搜索收益若只来自固定 opponents，不晋级。
