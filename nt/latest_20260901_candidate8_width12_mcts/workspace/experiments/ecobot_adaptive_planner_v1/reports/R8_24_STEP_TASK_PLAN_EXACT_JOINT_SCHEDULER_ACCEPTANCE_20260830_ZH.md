# R8：24步任务计划 + 4–8步精确多人调度验收报告

日期：2026-08-30  
规则版本：官方 `kaggle-environments==1.32.7`  
结论：**R8 原生执行内核通过验收。**

## 1. 本阶段完成了什么

R8 没有替换 R6/R7 的经济候选和经营项目选择。职责划分为：

```text
R6/R7 经济计划：决定做哪些项目、购买多少、目标规模和市场事务
        ↓
R8 24步任务账本：记录当前日内义务、截止时间、价值、负责人和状态变化
        ↓
R8 4–8步候选内精确DP：比较每个工人的合法短任务顺序
        ↓
精确多人匹配：在本回合选定任务集合内决定谁做哪个任务
        ↓
执行一个真实步骤
        ↓
读取真实新状态，取消失效任务、释放预留并重新规划
```

关键实现：

1. 每天建立一个24步任务级计划，整局正好重建30次。
2. 每个任务节点保存首次/最后出现时间、截止时间、价值、延迟损失、语义产业和当前负责人。
3. 每一步都用真实状态更新账本，整局更新719次；消失的任务立即释放，不播放固定动作表。
4. 普通自主计划使用8步窗口；已有完整高密度日历时使用4步窗口，避免重复估值大量可互换任务。
5. 每名工人最多保留8个当前合法候选，用 bitmask DP 在该候选集内穷尽所有满足移动、动作时长、deadline和物品前置条件的顺序。
6. 当前回合的多人任务分配使用最大权重精确匹配；同一任务和占用型地块只能被一个单位预留。
7. R8 不能擅自改变 R6 选中的任务集合。只有在即时 R6 评分不下降、总移动不增加、短程收益至少提高一个 `task_stickiness` 单位时才允许换人。
8. 自主计划每天最后8步、高密度计划最后12步回到确定性收尾，保证浇水、喂养和回仓链不被前瞻换人破坏。
9. 没有作者名、Replay ID、固定坐标、固定日期或对手身份路由进入 R8 热路径。

主要源码：

- `research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/src/native_adaptive.hpp`
- `research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/src/native_adaptive.cpp`
- `research/team_mate/Kaggriculture_main_512631c/agents/route_clustering_switch_agent/fast_kaggriculture/src/bindings.cpp`

验收工具：

- `experiments/ecobot_adaptive_planner_v1/tools/run_r8_executor_ablation.py`
- `experiments/ecobot_adaptive_planner_v1/tools/verify_r8_state_isolation.py`
- `experiments/ecobot_adaptive_planner_v1/tools/verify_r8_official_replay.py`

## 2. 冻结配置

| 参数 | 最终值 | 含义 |
|---|---:|---|
| 日内计划窗口 | 24步 | 覆盖完整一天 |
| 自主计划短程窗口 | 8步 | 稀疏任务图允许更深前瞻 |
| 高密度日历短程窗口 | 4步 | 防止重复估值可互换任务 |
| 每单位候选上限 | 8 | bitmask DP 的精确候选集 |
| 前瞻权重 | 0.01 | 只作保守增益信号 |
| 精确接管最低增益 | `1.0 × task_stickiness` | 避免边际换人 |
| 自主计划确定性收尾 | 最后8步 | 保护当日义务 |
| 高密度计划确定性收尾 | 最后12步 | 为取饲料和喂养留足时间 |

## 3. 最终盲测结果

下表均为全新固定 seed、双座位、R6 与 R8 使用完全相同的高层经济计划。

| 测试 | 局数 | R6平均现金 | R8平均现金 | 现金变化 | 分差变化 | P10变化 | 胜局变化 | 硬错误 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PASSIVE | 512 | 129,928 | 130,589 | **+660** | **+660** | **+1,688** | 512→512 | 0 |
| G001 | 512 | 64,205 | 64,372 | **+167** | **+370** | **+967** | 12→16 | 0 |
| G001语义计划 | 256 | 64,303 | 66,592 | **+2,289** | -159 | **+3,190** | 1→0 | 0 |

语义计划另一个独立256局批次中，R8现金 **+1,332**、分差 **+178**、胜局 `0→1`。两批合计512局：

- 平均自身现金约 **+1,810**；
- 平均分差约 **+9**，基本持平；
- 总胜局 `1→1`；
- 枯死、动物逃跑、容量溢出均为0。

因此，R8 已证明能在不改变经济计划的条件下提高计划兑现现金，但它**没有解决 G001 的高层经济路线差距**。对 G001 的绝对胜率仍低，不能把低层执行提升解释为已经击败强对手。

最终性能收据：

- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_exactdp_blind_passive_seed2867001_n256x2_v1.json`
- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_exactdp_blind_g001_seed2867001_n256x2_v1.json`
- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_exactdp_blind_g001_semantic_seed2867001_n128x2_v1.json`
- `experiments/ecobot_adaptive_planner_v1/receipts/r8_exactdp_threshold1_g001_semantic_seed2866001_n128x2_v1.json`

## 4. 任务与资源诊断

最终诊断批次为128局 PASSIVE：

| 指标 | 平均每局/结果 |
|---|---:|
| 日计划重建 | 30次 |
| 滚动更新 | 719次 |
| 动态任务节点 | 2,798 |
| 已解决任务节点 | 2,664 |
| 移动动作 | 3,868 |
| 空闲单位动作 | 734 |
| 任务解决延迟 P50 | 10.9步 |
| 任务解决延迟 P95 | 21.4步 |
| 实际未兑现硬任务 | 0 |
| 重复任务/占用地块预留 | 0 |
| 作物损失 | 0 |
| 动物损失 | 0 |
| 仓库终局溢出 | 0 |

“未兑现硬任务”只统计状态转移后真正造成的枯死或动物逃跑。仍然可见的候选任务记录、可替代的 DROP 路线和日终自动回仓不算硬错误。

诊断收据：

- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_diagnostics_passive_seed2870001_n64x2_v1.json`

## 5. 跨局污染、吞吐和线上时间门

最终版连续运行1,000局，然后将完全相同任务逆序再运行1,000局：

| 项目 | 结果 |
|---|---:|
| 奖励值不一致 | 0 |
| 诊断值不一致 | 0 |
| 硬错误 | 0 |
| 正序吞吐 | 24.56局/秒 |
| 逆序吞吐 | 24.48局/秒 |
| 单局719步 P50 | 459.36ms |
| 单局719步 P95 | 477.13ms |
| 折合每环境步 P50 | 638.9μs |
| 折合每环境步 P95 | 663.6μs |

精确DP明显慢于旧 beam 版，但仍满足原生内核的1秒单步门，余量超过三个数量级。

重要边界：以上是**原生 C++ 完整对局调用**，不包含线上 Python observation 解析、跨语言调用和 submission 打包开销。R8 内核已满足时间门；完整 Python submission 仍需单独接入并计时，不能直接用本数字宣称线上包已验收。

污染/延迟收据：

- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_diagnostic_state_isolation_passive_seed2871001_n1000_v1.json`

## 6. 官方规则一致性

使用最终精确DP代码生成 R8 联合动作流，再在官方 `kaggle-environments==1.32.7` 重放：

- 对手：PASSIVE、G001；
- 4个全新 seed；
- 双座位；
- 合计16局；
- 16/16 完成719步；
- 16/16 官方奖励与原生模拟器完全一致；
- 硬错误0。

官方收据：

- `experiments/ecobot_adaptive_planner_v1/receipts/r8_final_diagnostic_official_1327_seed2872001_n4x2x2_v1.json`

现有差分规则测试：`10 passed in 6.33s`。

## 7. 验收门判定

| 验收门 | 判定 | 证据 |
|---|---|---|
| 24步任务级计划 | PASS | 每局30次重建 |
| 每步滚动更新 | PASS | 每局719次更新 |
| 4–8步候选内精确搜索 | PASS | bitmask DP，无 beam 截断 |
| 多人精确首步分配 | PASS | 最大权重匹配 |
| 资源/地块唯一预留 | PASS | 128局重复违规0 |
| PASSIVE无明显退化 | PASS | 512局均值+660、P10+1,688 |
| G001同经济计划改善 | PASS | 512局分差+370、胜局12→16 |
| G001语义计划兑现改善 | PASS | 两批512局现金约+1,810，分差总体持平 |
| 枯死/逃跑/溢出 | PASS | 最终验收全部为0 |
| 跨局状态污染 | PASS | 1,000局正逆序全部逐值一致 |
| 官方1.32.7一致 | PASS | 16/16奖励完全一致 |
| 原生1秒单步门 | PASS | P95约0.664ms/步 |
| 完整线上Python包 | **未在本Goal验收** | 需要单独接入和计时 |

## 8. 最终结论与下一步

R8 已完成“经济计划不变时，提高任务兑现和多人协同”的目标，可以作为 R6/R7 后面的新低层执行器继续使用。

它当前解决的是：

- 日内任务不再每步完全失忆；
- 多人不会只做局部贪心分配；
- 4–8步内的任务顺序在最多8个合法候选中精确求解；
- 每步观察实际结果后只提交一步并重新规划；
- 硬义务和资源预留有明确安全门。

它没有解决的是：

- R6/R7 高层经济候选仍然偏窄；
- 对 G001 的绝对胜率仍很低；
- Python submission 尚未集成 R8；
- 精确DP吞吐约25局/秒，不适合直接承担几十万局的全量路线搜索。

推荐下一工作包：保留 R8 作为高质量决赛执行器；离线海量候选初筛继续使用更快的 R6/轻量调度，最后只对入围路线使用 R8 复评，并另外实现/验收线上 Python 调用链。
