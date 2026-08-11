# Kaggriculture 迁移共用建模底座 v0

逐篇迁移方案都基于本页。它解决的是 Kaggriculture 的共性，不把 Orbit Wars 的“行星—航线—舰队”变量硬套到农场。

完整规则真值见 [竞赛概要与完整游戏规则](../docs/KAGGRICULTURE_COMPETITION_AND_GAME_RULES_ZH.md)。

## 1. 决策分层

Kaggriculture 的原始一步动作同时包含主农夫、可变数量 hands 和最多 10 条有序市场订单。让 PPO 直接枚举所有组合会造成极大的联合动作空间。共用底座采用三层：

1. **管理策略（学习）**：选择少量高层任务、任务优先级、预算和市场意图。
2. **任务调度器（确定性）**：把任务分配给具体单位，处理截止期、背包、仓库容量、种子和现金约束。
3. **动作编译器（确定性）**：把当前任务变成当回合的移动、PLANT/WATER/HARVEST/FEED/CARE、PICKUP/DROP 和有序市场动作，并做 legality/mask。

策略每回合可以修改任务，但未修改的任务自动持续执行。这样“走 5 步去喂牛”是一个任务，而不是 5 次独立探索。

## 2. 状态 token

### 2.1 全局 token

- step/day/hour、距日终和终局的回合数；
- 自己与对手的银行余额、土地解锁数、单位数、公开生产资产；
- shed 使用量、种子、自己各单位背包汇总；
- 9 种商品的市场库存、当前价、局部价格斜率；
- 商店/镇中心未来确定性消费时间；
- 下一名 hand 价格、下一块土地价格；
- 当前任务的预计完成率和终局可现金化资产。

对手私有 shed、种子和背包不可作为 actor 输入；只能使用公开状态和由公开历史构造的 belief 特征。

### 2.2 地块 token

建议对自己 36 格和对手 36 格都编码，附 `side` 与坐标：

- locked/empty/weed/facility/plant/animal；
- crop/animal 类型、age、yield、watered/fed/cared、连续缺水/缺粮；
- 下一次生产、施肥失效、衰减、逃跑风险；
- 到仓库入口和各单位的距离；
- 在多个未来窗口的 no-new-action 投影：所有权不适用，但保留植物/动物状态、yield、维护债务、预计现金价值。

### 2.3 单位 token

- farmer/hand 类型、位置、背包及剩余容量；
- 当前任务、目标格、剩余路径、deadline；
- 当日剩余可行动小时；
- PICKUP→运输→执行→DROP 所需完整成本。

### 2.4 商品/市场 token

每种商品一个 token：当前库存和价格、供给斜率、双方可见产能、自己 shed 数量、未来城镇需求、终局前可出售窗口。种子、动物、土地、HIRE 可作为固定价格的购买 token。

## 3. 候选任务和动作头

先由确定性候选生成器建立合法、可行任务，再由模型排序，避免全组合。

### 3.1 单位任务族

- `WAIT/PASS`；
- `GO_TO(target)`；
- `PLANT(crop, tile)`、`WATER(tile)`、`HARVEST(tile)`、`FERTILIZE(tile)`、`DIG(tile)`；
- `BUILD_COOP/PASTURE(tile)`、`PLACE(animal,tile)`、`FEED/CARE/COLLECT_FERTILIZER(tile)`；
- `PICKUP(item,qty)`、`DROP`。

候选必须携带：最早完成时间、所需移动步数、材料/现金、对日终和终局的 slack、失败原因 mask、保守收益区间。

### 3.2 市场意图

V0 每回合最多输出 2 个市场宏动作槽：

- BUY_SEED/BUY_ANIMAL/BUY_PRODUCT/SELL/HIRE/BUY_LAND/NOOP；
- item；
- 数量 bucket；
- order slot。

动作编译器负责不超过 10 单、资金/仓库容量检查，以及避免把同回合市场购买错误地用于更早发生的单位动作。

### 3.3 自回归顺序

若需要联合动作，按 `farmer → hands → market slot 0 → market slot 1` 自回归；每采样一步立刻更新临时资源账本。这样可正确处理同回合种子原子 PLANT、共享背包/仓库容量和现金占用。

## 4. 推荐网络基线

- 1 个 global token；
- 72 个 plot token；
- 固定上限的 unit token（多余 slots mask）；
- 9–15 个 market/purchase token；
- 4–6 层 Pre-LN Transformer，`d_model=128–192`，约 1–5M 参数；
- unit↔task 或 unit↔plot 的 edge bias：路径距离、完整物流成本、deadline slack、维护紧急度、预计现金 margin；
- policy heads：候选任务、目标/商品、数量、市场动作；
- value head：预测胜率，而不是直接预测金币差；
- 可丢弃辅助头：未来 1/6/24/72 回合资产状态、维护违约、市场价格、终局现金。

## 5. RL、对手和奖励

### 5.1 主奖励

- 胜 `+1`、负 `-1`、平 `0`；依据双方终局 banked cash。
- gamma 是否为 1 必须消融。Kaggriculture 固定 719 步且不存在 Orbit 式提前消灭，建议初始 `gamma=0.999–1.0`。
- dense reward 只用于 curriculum 或辅助 loss，不能替代终局胜负；必须单独做无 shaping 对照。

### 5.2 对手池

- 当前冻结版本：40–60%；
- 历史强 checkpoint：20–40%；
- 确定性/规则 agent 与当前公开方案：10–20%；
- 专项 exploiters（只种田、抢市场、动物、早卖/晚卖）：10–20%。

不要仅与 live self 训练；每次对局固定一个对手 checkpoint，避免 rollout 中途换策略。

## 6. 评估与晋级

候选不能凭 training reward 或单一 Public LB 晋级。每个候选至少：

- 固定未训练 seeds；
- 双方交换座位；
- 对基线、历史冠军、规则专家和近期候选做 round-robin；
- 报告总胜率、按 seed/座位/对手族分层胜率和 Wilson 区间；
- 对市场冲击、终局清仓、动物、持续作物、hands 密集场景做专项面板；
- 默认至少 512 局/关键 matchup；小差异扩大到 2,048+。

晋级建议：对当前 champion 的座位平衡胜率下界 >50%，并且在固定多样对手面板上无明显回退；禁止在看到评估结果后修改同一实验的晋级公式。

## 7. 本机性能基线与预算解释

本地已实测的 608K MLP 基准完整 PPO 为 57,291 transitions/s。更复杂模型的下列区间只是预算用估算：

| 配置 | 方向性端到端区间 | 100M transitions 热循环时间 | 用途 |
|---|---:|---:|---|
| 0.6–1.5M 轻量 MLP/小 Transformer | 30k–80k/s | 21–56 分钟 | 代码/特征消融 |
| 1–3M 结构化 Transformer | 15k–50k/s | 0.6–1.9 小时 | 主迭代路线 |
| 3–8M + 丰富 edge/未来投影 | 5k–25k/s | 1.1–5.6 小时 | 晋级后扩模 |
| 自回归多次重跑 trunk/搜索 | 2k–12k/s | 2.3–13.9 小时 | 最后增强，不宜起步 |

实际 wall time 还包括评估、checkpoint、JIT、数据准备和失败实验。每次新增 token、候选数量、自回归槽或搜索深度后都必须重新 benchmark；不能把本表当作承诺。

## 8. Public 提交的 CPU 部署边界

本地训练使用 RTX 3090，不代表提交端也有 GPU。按当前竞赛资源表，agent
需要在 `1.6 vCPU / 6.5 GiB RAM / 1 秒基础动作时间 / 100 MiB 提交包`
内工作。因此所有迁移路线共同遵守：

- 主提交模型默认 `1–3M` 参数，优先直接训练可部署规模；
- `5M` 是软上限，只有通过等价 CPU 压测才能晋级；
- 每回合主 trunk 最多运行一次，联合动作只重复小型 policy head；
- 训练侧可以 JAX/JIT，提交侧不得依赖首次 JIT，需导出轻量 CPU 推理；
- 训练用 value/auxiliary heads 不随提交打包；
- 本地提交安全线：包 `<80 MiB`、峰值内存 `<2 GiB`、最慢回合
  `<0.8 s`、p95 `<0.25 s`，并完整跑完 719 次决策；
- 搜索默认仅用于本地评估。若要随提交启用，必须单独通过最坏状态 CPU
  压测，并保留超时前切换规则策略的 fallback。

以上安全线是项目晋级标准，不是官方额外规定。详细通俗说明见
[全部方案比较](COMPARATIVE_MIGRATION_ANALYSIS_ZH.md)。
