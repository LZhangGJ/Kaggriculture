# 架构与状态合同

## 离线链路

```text
609 原始 replay
  → 按参赛方提取完整经营意图与逐步动作
  → 经营/布局/日程/土地劳工距离聚类
  → 411 家族中筛出 245 个可执行代表
  → 245×245、64 seeds、双座路线互打
  → 5 opening × 3 检查点 × 26 target × 245 对手 × 128 seeds × 双座
  → 147 维公开状态特征
  → seed 分组 + 对手家族分组 + 阈值扰动的鲁棒浅树
```

路线互打共 3,825,920 局，原生核记录约 16,207 局/秒。精细切换搜索共 24,460,800 局、940,800 个状态向量、147 个特征，记录约 35,849 局/秒。大规模 rollout 数据已经固化为 `data/artifacts/*.npz`，训练不同树深时应直接复用，不应重跑仿真。

`scripts/` 保留成熟实现：

- `analyze_macro_route_library.py`：从 replay 建路线意图与距离缓存。
- `audit_macro_execution_quality.py`：审计真实动作效果和执行质量。
- `cluster_intended_macro_routes.py`：按显式距离阈值聚类。
- `export_intent_route_carriers.py`：导出完整路线动作载体。
- `run_native_intent_round_robin.py`：全 C++ 路线互打。
- `run_native_intent_switch_search.py`：全 C++ 反事实切换 rollout。
- `train_robust_search_route_trees.py`：分组验证、扰动阈值和置信下界选树。
- `search_native_route_tree_sequences.py`：研究多节点/多次路线切换序列。

## 在线链路

`agent/main.py` 构造两个模块：

1. `SearchRoutedTeammateAgent` 根据真实观察执行 replay 路线并在浅树节点选择目标路线。动作磁带之外仍执行成熟的杂草、交易、现金和喂养保护。
2. `policy/r1/Agent` 是无开局模板的 JointAFS R1。前期每帧调用 `observe_external(observation, actual_action)`；达到接管步后调用原生 R1 `act()`。

外部观察桥只维护 R1 本来就从公开历史可恢复的状态：

- `live.joint.observe/record`；
- `ObservedCropClock` 及对手收获权重；
- `PublicTradeLedger` 与 `SaleClock`；
- 接管帧的 `previous_step` 和 `sale_memory`。

桥不在接管前调用另一个分支的 `plan()` 或 `choose()`。接管帧由 R1 原始 `SearchController::act()` 按自身顺序观察、选择、规划和行动。这个选择避免把带 opening-template 分支的初始化顺序再次混入 R1，但目前并未证明足以恢复所有内部状态。

## R1 组合目标与实验边界

`Planner::value` 按天统一评价作物、动物、库存和既有资产：已开放商店形成确定需求，未开放商店形成
折扣后的期望需求；双方公开地块形成未来产量，市场库存经过需求与供给后沿官方价格曲线变化；目标再
扣除工资、动作成本、资本约束和加权对手收入。对手预计供给默认一半先于我方成交、一半晚于我方成交。
小麦和肥料不作为独立投资项目，但作为动物饲料、施肥投入和现金流进入同一 Asset 流。

默认组合构造仍是逐地块边际贪心，并由多组 settings proposal 的一日公开状态 rollout 二次选择。
`R1_CONFIG_OVERRIDES={"portfolio_swaps":1}` 可启用单地块局部替换，值 2 启用受限双地块联动替换；
二者都只处理本轮空地现金项目，不改存量资产、库存承诺或 joint bundle。正式部署值为 0。单换两批
16-seed 净胜 0，双换 8-seed 为净 `-1`，所以当前证据否定“局部组合搜索深度是主要瓶颈”。

## 清晰边界

- 原始 replay、来源、对手身份只允许离线使用；在线不按 replay ID 或对手 ID 查表。
- 公开对手仅用于官方环境验证，不进入在线特征。
- `route_policy.json` 当前最多发生一次 replay→replay 切换；多次切换是已有搜索代码支持的研究方向，尚未接入生产控制器。
- 三块地后延迟 1 天、最晚 step288 是当前部署接管条件，不是游戏规则或普适最优时机。
- `trees-deep-d8-16.json` 已保存，但未通过强对手门禁，不能因训练集指标更高直接替换浅树。
