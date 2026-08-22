# Kaggriculture 分层任务与预约规划 RL V4

V4 在 V3 的任务匹配、十订单市场和双棋盘网络上增加四项能力：持久任务状态机、
真实 ETA、时空预约 A*、可变现净资产奖励，以及每日 KEEP/SWITCH 战略门控。

## 任务状态机与真实 ETA

每个工人拥有独立的 `UnitTaskState`，状态为：

```text
IDLE → ASSIGNED → ACQUIRE_RESOURCE → NAVIGATE → EXECUTE
                                   ↘ WAITING
                         任意活动状态 → BLOCKED / CANCELLED / COMPLETED
```

状态记录任务、开始/更新时间、当前 ETA、预约路线和累计失败数。所有状态变化进入
有界日志，并按 `任务类型:阶段` 聚合。

`estimate_task_eta` 不再只计算曼哈顿距离。它显式加入：

- 去仓库的移动时间；
- `PICKUP` 操作；
- 仓库到目标的第二段路线；
- 目标操作；
- 已浇水但尚未成熟等必要等待；
- 资源是否存在，以及完成时间是否超过任务 deadline。

不可达或无法按期完成的 worker-task 边直接进入 `pair_mask=False`。ETA 矩阵保存在
`EncodedTaskSet.eta_steps`，并作为 pair feature 输入网络。

## BFS/A* 与多工人预约

静态距离使用 BFS；实际执行使用带 WAIT 动作的时空 A*。多工人按以下顺序规划：

1. mandatory 任务优先；
2. deadline 更早者优先；
3. ETA 更短者优先；
4. 工人编号用于稳定打破平局。

预约表同时阻止同一时刻占用同一顶点和同一时刻对向交换边。低优先级工人可等待或
绕路；预约等待属于正常 `WAITING`，不会误报为 `PATH_BLOCKED`。确实找不到路径才进入
`BLOCKED` 并写执行器失败日志。

## 可变现净资产奖励

势能由现金和当前可执行出售的产品构成：

- 仓库产品按 0.98 执行折扣；
- 工人携带产品按 0.90 折扣；
- 棋盘上已经生成的作物/动物产品按 0.75 折扣；
- 根据剩余步数自动把来不及收获、入库或出售的部分降为 0；
- 按真实市场库存逐件重算报价，包含批量清算造成的价格冲击；
- 种子、土地、工人、建筑和动物没有出售动作，因此不伪造残值。

同一环境的两个私有观察先分别计算绝对净资产，再形成严格反对称的 log 势能。逐步
奖励使用势能差；终局胜负 bonus 仍只依据官方现金结果，避免改变竞赛最终目标。
V1/V2/V4 的 RL 脚本都使用同一个净资产势能实现。

## 每日 KEEP/SWITCH

网络保留 8 类模式 head，并增加二分类 `switch_head`：

- 第一次决策直接选择初始模式；
- 同一天内模式固定，不产生额外 policy log-prob；
- 每个新日第一次决策采样 `KEEP` 或 `SWITCH`；
- `SWITCH` 后再从排除当前模式的 7 个模式中选择新模式；
- 所有选择、日期和切换次数写入 episode memory。

BC 只在开局监督初始模式，并在专家轨迹的每日边界监督 KEEP。SWITCH 的策略及新模式
主要由 RL 学习，避免用整局固定的人工标签伪造切换真值。V3 checkpoint 可以用
`strict=False` 加载，新增 switch head 随机初始化。

## 运行

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe scripts\train_hierarchical_bc.py --device cuda
.venv\Scripts\python.exe scripts\train_hierarchical_rl.py `
  --init-checkpoint artifacts\hierarchical_bc_v4.pt --device cuda
```

训练输出的 `.diagnostics.json` 包括 executor failures、任务状态转换、当前路线、ETA、
每日模式决策和切换历史。
