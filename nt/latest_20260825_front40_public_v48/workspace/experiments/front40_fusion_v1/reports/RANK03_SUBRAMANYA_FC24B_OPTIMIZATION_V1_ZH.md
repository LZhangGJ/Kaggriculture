# 第 3 名 Subramanya Replay 家族对 FC24B 优化结论

日期：2026-08-24

## 结论

- 已复原 157 条完整获胜 Replay，全部来源、Episode、终局资金和 SHA-256 均已保存。
- 该 Replay 集合具有很高的事后路线覆盖能力，但目前没有找到能因果部署、并在独立随机库稳定击败 FC24B 的路线选择器。
- 本阶段不通过 `>50%` 入选门，不进入最终融合池；所有失败版本保留，后续只有在实现“任务承诺兼容的局部重规划”后才值得重开。
- 最好的独立留出结果是首商店路线图 `81/256 = 31.64%`；不是合格强拳。

## 来源与完整性

- 队伍：Subramanya N。
- 当时榜单名次：第 3 名。
- Submission：`55616096`。
- 完整获胜 Replay：157 局。
- 路线库：`artifacts/rank03_subramanya_trace_bank_v1.npz`。
- 来源回执：`receipts/rank03_subramanya_trace_bank_v1.json`。
- 157 条路线在两套各 128 seed、双座位屏幕中均完成 719 步，`hard_counter_total = 0`：
  - `receipts/rank03_subramanya_all157_vs_fc24b_screen_seed252001_n128x2_v1.json`
  - `receipts/rank03_subramanya_all157_vs_fc24b_screen_seed272001_n128x2_v17.json`

## 关键发现

### 1. 事后 oracle 很强，但不能直接部署

- 第一套 256 个座位上下文中，157 路线 oracle 为 `243/256 = 94.92%`。
- 第二套独立上下文中，157 路线 oracle 为 `93.36%`。
- 这只表示：如果提前知道整局结果，几乎总能在 Replay 库里找到一条赢家。
- 它不表示第 3 天的 Agent 能知道应该选择哪一条，因为路线收益高度依赖尚未公开的未来商店和后续随机事件。

### 2. 固定路线不稳定

- 第一套随机库的最佳固定路线是 route 40，胜率 `29.30%`。
- 第二套随机库的最佳固定路线变成 route 117，胜率 `28.13%`。
- 第三套留出候选集中最佳固定路线是 route 61，胜率 `23.44%`。
- 最佳路线随随机库大幅变化，不能把某条 Replay 当作稳定拳法。

### 3. 首商店和座位路由不足

- 两套开发库上，最好的 unrestricted Wilson 首商店图达到 `210/512 = 41.02%`。
- 完全独立 seed 280001–280128 上跌至 `81/256 = 31.64%`，平均分差 `-10,525`。
- 座位感知版本在同一留出更低，说明加入先后手并未解决未来信息缺失。
- 回执：
  - `receipts/rank03_subramanya_shop_route_maps_dev_eval_n256x2_v18.json`
  - `receipts/rank03_subramanya_shop_route_maps_holdout_seed280001_n128x2_v20.json`

### 4. 简单可见状态 LGBM 仍然失败

训练特征只使用第 72 步合法可见状态和我方私有库存，包括：

- 双方公开农场、人员位置和资金；
- 我方库存、种子、携带物；
- 市场价格和库存；
- 已公开商店；
- 实际杂草分布和当前路线偏差。

训练种子从 256 扩大到 768 后：

- 256-seed 模型独立留出：`76/256 = 29.69%`；
- 768-seed 模型独立留出：`75/256 = 29.30%`；
- 同一留出 20 路线 oracle：`78.91%`。

增加三倍数据没有改善，说明问题不是简单的“小样本再多一点”即可解决。标签包含未来商店带来的事后优势，而第 72 步状态不包含这些未来信息。

主要回执：

- `receipts/rank03_subramanya_route_ranker_dev20_holdout_seed280001_n128x2_v27.json`
- `receipts/rank03_subramanya_route_ranker_dev20_dev768_holdout_seed280001_n128x2_v32.json`
- `receipts/rank03_subramanya_dev20_vs_fc24b_screen_seed288001_n512x2_v28.json`

### 5. 日级最近状态切换破坏任务连续性

- 每日根据公开状态重新选择最近 Replay，只得到 `33/128 = 25.78%`。
- 每局平均切换约 2.29 次，非法意图/恢复次数明显增加。
- 权重网格的最好结果也只有 `18.75%`。
- 原因是“状态看起来接近”不代表当前工人、携带物、未完成任务和市场事务承诺可以安全接上另一条 Replay。

回执：

- `receipts/rank03_subramanya_day_public_state_vs_fc24b_seed260001_n64x2_v12.json`
- `receipts/rank03_subramanya_day_router_weight_screen_seed264001_n16x6x2_v13.json`

### 6. 精确前缀兼容的第二商店切换没有稳定增益

- 只允许在第 144 步切换到 72–143 步原始动作完全相同的路线，避免破坏既有承诺。
- 两个开发库互换验证：一边净增 7 胜，另一边净减 4 胜。
- 用两库共同训练后，在第三独立库由基础 `72/256 = 28.13%` 降至 `70/256 = 27.34%`。
- 因此这组二阶段补丁不能保留。

回执：

- `receipts/rank03_subramanya_secondshop_train252_eval272_v33.json`
- `receipts/rank03_subramanya_secondshop_train272_eval252_v33.json`
- `receipts/rank03_subramanya_secondshop_dev256_holdout_seed280001_n128x2_v36.json`

## 已排除的方法

- 单条高源收益 Replay。
- 同源首商店静态图。
- 跨首商店静态图。
- 座位感知静态图。
- 每日最近状态路由。
- 路由距离权重调参。
- Replay 逐动作众数合成；源路线逐步动作一致率均值只有约 0.51，不能安全投票。
- 第 72 步可见状态 LGBM margin/win/rank 混合排序。
- 精确动作前缀约束的第二商店切换。
- 在同一个 JAX 图中同时计算 Replay 后缀与 FC24B 回退；该工程结构编译过重，而且 FC24B 镜像回退会产生大量平局，本身无法让严格胜率过 50%。

## 为什么暂时停止该家族

继续把完整 719 步 Replay 当作一个候选，会遇到不可解的信息边界：早期必须选择整条未来路线，但未来商店尚不可见。继续增加首商店 if/else 或 LGBM 样本，只会学习随机库相关性。

未来若重开，必须先具备：

1. 将完整 Replay 分解为带前置条件、资源承诺和完成条件的任务段；
2. 商店公开后只替换尚未启动的任务，不替换已承诺任务；
3. 对工人携带、地块、动物维护、待收获和市场事务进行显式状态迁移；
4. 每次局部编辑都从同一真实状态做反事实续跑，而不是拼接两条原始动作表。

这属于新的“语义任务重规划器”，不是继续调当前 Replay 路由器的阈值。
