# Kaggriculture 竞赛概要与完整游戏规则

> 用途：作为后续状态表示、动作抽象、规划器、self-play、RL 奖励和评估设计的共同规则基准。
>
> 规则版本：`kaggle-environments==1.32.6`。
>
> 竞赛页面快照：2026-08-11（会随时间变化）。
>
> 本文描述的是官方默认配置；配置覆盖项另行列出。

## 0. 证据等级与冲突处理

规则事实按以下优先级处理：

1. 冻结的官方 Python 解释器 `kaggriculture.py`；
2. 冻结的官方环境规格 `kaggriculture.json`；
3. 官方 How to Play/README；
4. 由源码直接推导的时间表与边界说明。

如果页面说明和实际解释器行为不一致，建模、回放解析和本地模拟器均以
解释器行为为准。

本地冻结证据：

- [官方解释器](../gpu_sim/reference/kaggle_environments_1_32_6/kaggriculture/kaggriculture.py)，SHA256：`fb9215c5e21a25243e2d13e75b3d70a79cf7d78fff150a90f1bb5eacf9ba2bcf`
- [官方环境配置](../gpu_sim/reference/kaggle_environments_1_32_6/kaggriculture/kaggriculture.json)，SHA256：`7f196f4aaf9b4e482fbd61ba4bf19f5cec4da0f81938ece756954035a65c0bb5`
- [官方规则 README](../gpu_sim/reference/kaggle_environments_1_32_6/kaggriculture/README.md)，SHA256：`63d0497ca655ea1857a96a3226bed6664a2fa98506d76d1fa14ce9101c5f1d34`
- [冻结文件校验回执](../gpu_sim/receipts/reference_verification.json)
- [JAX 与官方 100 种子逐状态 parity 回执](../gpu_sim/receipts/heldout_parity.json)

官方竞赛页面：[Kaggriculture](https://www.kaggle.com/competitions/kaggriculture)。

## 1. 竞赛概要

Kaggriculture 是一个两人回合制农业经营对战环境。两位玩家拥有彼此独立
的农场，但共享同一个动态市场和城镇需求系统。玩家需要管理：

- 农地购买和空间配置；
- 作物种植、浇水、施肥、成熟、收获和衰减；
- 动物设施、购买、搬运、放置、喂养、照料、产物和肥料；
- 主农夫和每日临时雇工的移动与任务调度；
- 私有仓库、种子和单位随身库存；
- 共享市场的供需、价格、买卖时机和订单顺序；
- 城镇商店的随机解锁和持续消费；
- 赛季末之前的现金化。

### 1.1 胜负目标

- 唯一官方胜负依据是终局银行余额 `money`。
- 银行余额较高者获胜，相等则平局。
- 仓库、种子、单位背包、地上未收获产物、动物、设施和土地本身均不折现。
- 官方环境的原始 `reward` 是终局银行余额，不是 `+1/0/-1`。
- 竞赛评级只看胜/负/平，不看金币差距；本地 RL 可以把终局余额比较转换为
  `+1/0/-1`，但这是训练层变换，不是官方原始 reward。

### 1.2 排名机制（2026-08-11 页面快照）

- 每队每天最多提交 5 个 agent。
- 只有最近 2 个提交持续被跟踪，并用于最终评估。
- 新提交先与自身进行 Validation Episode；失败则标记为 `Error`。
- 天梯按相近评级匹配对手。
- 最终使用已完成对局进行 Bradley-Terry 评估。
- 金币差不影响评级变化，只有胜/负/平结果影响。

### 1.3 时间、奖励与资源（会变化）

| 项目 | 2026-08-11 核验值 |
|---|---:|
| 开赛 | 2026-07-29 |
| 报名截止 | 2026-09-23 23:59 UTC |
| 合队截止 | 2026-09-23 23:59 UTC |
| 最终提交截止 | 2026-09-30 23:59 UTC |
| 继续跑局/收敛期 | 2026-10-01 至约 2026-10-15 |
| 总奖金 | 50,000 USD，前 10 名各 5,000 USD |
| 提交包大小 | 100 MiB |
| Agent 磁盘 | 8 GiB |
| Agent 内存 | 6.5 GiB |
| Agent CPU | 1.6 vCPU |
| 提交目录 | `/kaggle_simulations/agent/` |
| 单次动作基础超时 | 配置 `actTimeout=1` 秒 |
| 初始 overage time | observation 规格默认 60 秒 |

### 1.4 提交接口

- 提交包根目录必须有 `main.py`。
- `main.py` 必须暴露 `agent(obs)`，返回第 7 节定义的 action dict。
- 多文件提交可以打包，但导入路径应以 `/kaggle_simulations/agent/` 为基准。
- 官方本地环境提供 `pass`、`random`、`starter` 三个内置基线。

## 2. 默认配置变量

| 配置 | 默认值 | 精确含义 |
|---|---:|---|
| `episodeSteps` | 720 | 环境记录帧上限；实际调用次数见第 3 节 |
| `actTimeout` | 1 | 每次 agent 调用的基础时间额度 |
| `boardSize` | 10 | 每位玩家农场为 10×10 |
| `startingMoney` | 3000 | 每位玩家初始银行余额 |
| `maxMarketOrdersPerTurn` | 10 | 每位玩家每回合最多处理的市场订单数，超出部分静默丢弃 |
| `turnsPerDay` | 24 | 每游戏日的状态推进单位 |
| `shedCapacity` | 100 | 仓库非种子物品总容量 |
| `weedSpawnChance` | 0.005 | 日终时每个空闲已解锁格子的杂草概率 |
| `townShopUnlockInterval` | 3 | 每 3 天解锁一个商店实例 |
| `townShopSellInterval` | 4 | 商店每 4 个 step 消耗一次商品 |
| `townCenterSellInterval` | 24 | 镇中心每 24 个 step 消耗一次商品 |
| `seed` | `null` | 可选对局随机种子；解析后从 agent 配置中移除 |
| `farmHandCostMult` | 1 | 当日第 n 次雇工的 Fibonacci 费用乘数 |
| `marketParams` | `{}` | 可按商品稀疏覆盖市场曲线参数 |

比赛默认配置不改变作物种子价格、动物价格和各商品默认市场参数。

## 3. 时间、step、day、hour 与终止边界

### 3.1 页面叙述

官方页面用“30 天 × 24 回合 = 720 回合”描述赛季。

### 3.2 解释器的实际行为

在 `episodeSteps=720` 下，本地直接运行官方 `1.32.6` 得到：

- 720 个记录帧：`step=0..719`；
- 每位 agent 实际调用 719 次；
- 被处理的动作对应旧状态 `step=0..718`；
- 终止帧为 `step=719, day=29, hour=23, status=DONE`；
- `day=29, hour=23` 是终止观察，不再给 agent 一次动作；
- 最后一次正常日终刷新发生在进入 `day=29` 时；不会进入 `day=30`，因此
  最后一日末尾没有自动背包入仓、动物/植物日刷新或雇工清理。

建模和 JAX rollout 必须使用 719 次状态推进，而不能按 720 次动作推进。

### 3.3 时钟公式

处理动作前，解释器使用：

```text
day = step // turnsPerDay
```

动作处理完成后：

```text
next_step = step + 1
observation.day  = next_step // turnsPerDay
observation.hour = next_step % turnsPerDay
```

日终条件：

```text
(step + 1) % turnsPerDay == 0
```

### 3.4 终局现金化影响

- 最后一次 agent 决策发生在 `day=29, hour=22` 的旧状态。
- 该回合仍会执行单位动作、市场订单、城镇消费和植物逐 step 衰减。
- 单位若已在仓库入口，可用 `DROP` 把随身物品入仓；由于单位动作先于市场，
  同一回合可以随后用 `SELL` 卖出刚 DROP 的商品。
- `HARVEST` 后商品进入单位背包；同一单位不能再执行 DROP，因此通常无法在
  最后一个动作中“收获后立刻卖出”。
- 终局仍在仓库或背包中的商品完全不计分。

## 4. 每回合精确处理顺序

对旧状态 `step=t`，源码执行顺序如下：

1. 读取并规范化双方 action；缺失字段使用空动作/`PASS`。
2. 对每位玩家做同作物的原子 PLANT 需求检查。
3. 依次处理玩家 0、玩家 1 的单位动作。
4. 每位玩家内部按“主农夫 → hands 列表顺序”处理单位动作。
5. 处理双方有序市场队列。
6. 城镇商店和镇中心从共享市场库存中消费商品。
7. 按当前 step 执行植物逐 step 衰减。
8. 若为日终，依次刷新植物、动物、杂草、单位库存、位置、雇工和商店解锁。
9. 更新 `step/day/hour` 对应的共享观察。
10. 若达到终止条件，将两位玩家状态设为 `DONE`，reward 设为各自银行余额。

需要注意：页面把双方单位动作称为“同时发生”，但源码中同一玩家的单位动作
按固定顺序落地。两个单位可以站在同一格；第一个单位修改格子后，第二个单位
看到的是修改后的格子。例如，同格两次 HARVEST 通常只有先执行者取得产物。

除最终 day29 外，hour23 的单位动作仍发生在日终刷新之前，因此最后一小时 WATER
或 FEED 可以挽救即将在该日终达到连续两天缺水/缺粮的植物或动物。

## 5. 地图、坐标、土地和仓库

### 5.1 坐标

- `tiles[y][x]`，`x` 向右增加，`y` 向下增加。
- 单位位置为 `[x, y]`。
- 默认地图 10×10，分为 4 个 5×5 象限。

### 5.2 象限

| 象限 | 初始状态 | 解锁次序 | 价格 |
|---|---|---:|---:|
| NW | 已解锁 | 0 | 0 |
| NE | 锁定 | 1 | 1000 |
| SW | 锁定 | 2 | 2000 |
| SE | 锁定 | 3 | 4000 |

`BUY_LAND` 不能指定象限，总是按 NE → SW → SE 解锁。一次市场列表中可放多个
`BUY_LAND`，只要余额足够即可同回合连续解锁。

### 5.3 锁定格子

- `tiles[y][x] == "LOCKED"`。
- 单位可以进入和穿过锁定格子。
- PLANT/WATER/HARVEST/BUILD/DIG 等格子动作在锁定格上静默 no-op。
- 仓库相关 PICKUP、DROP、PLACE-to-shed 是例外；只要站在仓库入口，即使该格
  仍锁定，也可使用仓库。

### 5.4 中央仓库

- 仓库不是 `tiles` 中的格子。
- 默认四个入口为 `(4,4)、(5,4)、(4,5)、(5,5)`，分别位于四个象限内角。
- 主农夫初始及每天重置到 NW 入口 `(4,4)`。
- 仓库容量 100，统计所有非种子商品、肥料和未放置动物。
- 种子在独立 `private.seeds` 中，不占仓库容量。

## 6. Observation：公开信息与私有信息

### 6.1 顶层字段

| 字段 | 可见性 | 类型/含义 |
|---|---|---|
| `player` | 本人 | 0 或 1 |
| `step` | 本人 | 框架提供的当前记录帧编号 |
| `day` | 共享 | 当前游戏日，0-based |
| `hour` | 共享 | 当日小时/回合，0-based |
| `farms` | 共享 | 两位玩家的公开农场状态 |
| `market` | 共享 | 市场库存和当前价格 |
| `town` | 共享 | 已解锁商店实例列表 |
| `private` | 私有 | 当前玩家自己的仓库、种子和单位背包 |
| `remainingOverageTime` | 本人 | 框架剩余额外时间 |

对手的仓库、种子和单位背包不在 observation 中；对手身份、未来随机事件和
对局 seed 也不在 agent observation 中。

### 6.2 公开 farm 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `money` | float | 银行余额；默认规则下数值实际为整数金额 |
| `tiles` | 10×10 | 全部土地、植物、设施、动物、杂草和锁定状态 |
| `farmer` | `[x,y]` | 主农夫位置 |
| `hands` | `[[x,y],...]` | 当日所有雇工位置，长度可变 |
| `unlocked_quadrants` | list | 已解锁象限 |
| `hires_today` | int | 当日已雇工数，决定下一次 HIRE 价格 |

### 6.3 私有 private 字段

| 字段 | 类型 | 含义 |
|---|---|---|
| `shed` | `{item: count}` | 商品、肥料和未放置动物的仓库库存 |
| `seeds` | `{crop: count}` | 5 种作物种子，独立且不占仓库容量 |
| `inventories` | list[dict] | `[主农夫背包, hand1背包, ...]`，长度随雇工变化 |

单位随身背包没有显式容量上限，但日终自动入仓时仍受仓库容量限制。

### 6.4 Tile 联合类型

一个格子可能是：

- `None`：空闲、已解锁；
- `"LOCKED"`：未购买土地；
- `{"kind":"WEED"}`：杂草；
- 植物字典；
- 空设施 `{"kind":"COOP"}` 或 `{"kind":"PASTURE"}`；
- 带动物的 COOP/PASTURE 字典。

源码中空设施通常根本没有 `animal` 键，而不是 `animal=None`。解析器应使用
`"animal" in tile` 判断是否已有动物。

### 6.5 植物 tile 字段

| 字段 | 含义 |
|---|---|
| `kind="PLANT"` | 植物标识 |
| `crop` | WHEAT/CARROT/TOMATO/STRAWBERRY/MELON |
| `planted_day` | 种植日 |
| `watered_today` | 当天是否已浇水；日终重置 |
| `consecutive_unwatered` | 连续未浇水日计数；达到 2 变杂草 |
| `yield_units` | 当前格子上可收获数量 |
| `max_lifespan_step` | 开始逐 step 衰减的 step；未确定时为 -1 |
| `fertilized_until_day` | 肥料有效的最后一个 day，含当日；未施肥为 -1 |

### 6.6 动物 tile 字段

| 字段 | 含义 |
|---|---|
| `kind` | COOP 或 PASTURE |
| `animal` | GOOSE/COW/SHEEP；仅带动物设施存在该键 |
| `placed_day` | 动物放置日 |
| `yield_units` | 格子上尚未收取的蛋/奶/羊毛 |
| `fed_today` | 当天是否喂过小麦 |
| `consecutive_unfed` | 连续未喂日数；达到 2 时动物逃走 |
| `cared_today` | 当天是否 CARE |
| `fertilizer_available` | 是否有 1 份肥料可收集 |
| `pending_care_bonus` | 已积累、等待下一次生产结算的 CARE 奖励 |

## 7. Action 格式与通用语义

```python
{
    "farmer": [op, ...args],
    "hands":  [[op, ...args], ...],
    "market": [[op, ...args], ...],
}
```

- 每个现存单位每回合最多一个单位动作。
- `hands[i]` 对应公开 `farm.hands[i]`。
- 少提供 hand 动作时，未列出的雇工不行动。
- 多提供的 hand 动作通常找不到单位并 no-op，但仍可能参与原子 PLANT 需求统计。
- 非 list、空 list、未知 op、参数不足、资源不足或位置不合法均静默 no-op。
- 单位可重叠，不存在碰撞阻挡。

### 7.1 同回合 PLANT 原子规则

解释器先统计同一玩家 action 中某作物的全部 PLANT 请求。如果请求数超过该作物
种子数，则该作物的所有 PLANT 请求都改成 PASS，而不是只成功前几个。

统计发生在检查单位是否存在、格子是否空闲之前，因此无效/多余 hand 的 PLANT
请求也可能让整个作物批次被阻止。动作编译器必须只输出真实单位、真实空格上的
PLANT，并在同回合做种子资源保守分配。

## 8. 单位动作完整规则

| 动作 | 必要条件 | 效果与边界 |
|---|---|---|
| `NORTH/SOUTH/EAST/WEST` | 目标仍在地图内 | 移动一格；可进入 LOCKED；越界 no-op |
| `PASS` | 无 | 不做事 |
| `PICKUP item [n]` | 站在仓库入口、n>0、仓库有货 | 从 shed 搬至当前单位背包；默认 n=1；不能拿种子 |
| `DROP` | 站在仓库入口 | 整个背包按物品全部清空入仓；超出仓库容量部分直接丢失 |
| `PLACE item [n]` | 见下 | 动物放置或部分物品入仓 |
| `PLANT crop` | 已解锁空格、有种子、通过原子检查 | 消耗 1 种子，创建植物 |
| `WATER` | 当前格为植物且当天未浇 | 设置 watered；一次性作物在奖励窗口增加 yield |
| `HARVEST` | 植物/动物 yield_units>0 且植物达到首收日 | 全部产物进入当前单位背包；一次性作物随后清空格子 |
| `FERTILIZE` | 当前格为植物、背包有肥料 | 消耗 1 肥料，有效至 `max(旧值, day+2)` |
| `DIG` | 已解锁且格子非空、没有动物 | 删除植物、杂草或空设施；不返还资源 |
| `BUILD_COOP` | 已解锁空格 | 免费建空鹅舍，但消耗本回合单位动作 |
| `BUILD_PASTURE` | 已解锁空格 | 免费建空牧场，但消耗本回合单位动作 |
| `FEED` | 格上有动物、当天未喂、背包有 WHEAT | 消耗 1 小麦，设置 fed_today |
| `CARE` | 格上有动物、当天未 CARE | 设置 cared_today；当天是否入账在日终判断 |
| `COLLECT_FERTILIZER` | 格上有动物且 fertilizer_available | 取 1 肥料到背包并清空标志 |

`PLACE` 的两个分支：

1. 若当前格是匹配的空设施，且背包有对应动物，则忽略 n，放置恰好 1 只动物；
2. 否则若站在仓库入口，把最多 n 个指定物品放入仓库；仓库满时剩余物品留在背包，
   不像 DROP 那样丢弃。

动物设施匹配：GOOSE→COOP，COW/SHEEP→PASTURE。

## 9. 仓库、背包、种子与资源流

### 9.1 三种库存位置

1. `private.seeds`：种子，所有单位可直接用于 PLANT；无搬运、无容量。
2. `private.shed`：仓库，市场只能从这里 SELL，BUY_PRODUCT/BUY_ANIMAL 也进入这里。
3. `private.inventories[i]`：单位背包，HARVEST/PICKUP/COLLECT 的产物先到这里。

### 9.2 关键限制

- 市场 SELL 不能直接卖单位背包物品，只能卖 shed。
- FEED/FERTILIZE/PLACE animal 只能消耗当前执行单位的背包，不能直接消耗 shed。
- BUY_PRODUCT 或 BUY_ANIMAL 在 shed 满时失败，不扣钱。
- BUY_SEED 不占 shed，因此不受 shedCapacity 限制。
- 每日结束时，按 `主农夫 → hands` 的背包顺序自动入仓；能装多少装多少，超出部分丢弃。
- 自动入仓之后所有雇工消失，背包列表重置为仅主农夫空背包。

### 9.3 同回合生产与市场的时间差

- 单位动作先执行，市场后执行。
- 已在背包中的商品可以先 DROP，再在同回合 SELL。
- 当回合 HARVEST 得到的商品仍在该单位背包，除非另一个机制已把它入仓，否则不能卖。
- 当回合买到的种子、动物、小麦或肥料不能被更早执行的单位动作使用，要等下一回合。
- 当回合 HIRE 的 hand 在市场阶段出现，因此不能在同一回合获得单位动作。

## 10. 植物规则

### 10.1 作物参数与有效时间表

`age = current_day - planted_day`。

| 作物 | 种子价 | 基准售价 | 类型 | 最早收获/首次生产 age | 奖励/生产时间表 | 格上 yield 上限 | 衰减开始 age | 官方无肥参考 yield/格/日 |
|---|---:|---:|---|---:|---|---:|---:|---:|
| WHEAT | 10 | 25 | 一次性 | 2 | 浇水奖励 age 2–4 | 6；无肥最高 4 | 5 | 0.80 |
| CARROT | 20 | 35 | 一次性 | 2 | 浇水奖励 age 2–3 | 4；无肥最高 3 | 4 | 0.75 |
| TOMATO | 50 | 60 | 持续型、有限 4 次 tick | 8 | age 8,9,10,11 | 4 | 12 | 0.33 |
| STRAWBERRY | 100 | 120 | 持续型、有限 4 次 tick | 10 | age 10,12,14,16 | 4 | 17 | 0.24 |
| MELON | 80 | 250 | 一次性 | 10 | 浇水奖励 age 6–12；无肥 age10 达到 cap | 6 | 13 | 0.55 |

说明：

- 页面把 Melon “Time to Max Yield”写为 10，这是每日浇水时达到 cap=6 的有效年龄；
  源码参数 `max_yield_day=12`，因此奖励窗口和寿命计算仍使用 12。
- Tomato/Strawberry 的“4”是生产 tick 数和格上未收取库存上限。每次 tick 可产 1 或 2，
  及时收获并施肥时，生命周期累计收获可超过 4，理论最多 8。
- 表中“基准售价”只是市场库存为 I0 时的价格，不是保证售价。
- 官方参考 yield/格/日按每日浇水、无肥、峰值收获计算；它不是利润率，也没有计入
  种子、土地、雇工、移动、市场冲击和终局剩余资产。

### 10.2 新种植物

新植物字段：

```text
watered_today = false
consecutive_unwatered = 1
yield_units = 1（一次性）或 0（持续型）
fertilized_until_day = -1
```

种植日已经被记作第一次未浇水。若种下后当天没有 WATER，日终时计数从 1 变 2，
植物当夜直接变成杂草；新植物没有一天宽限期。

因此无论是否进入产量奖励窗口，所有新种植物都必须在种植当天浇水才能存活。

### 10.3 缺水与变杂草

日终时：

```text
若 watered_today：consecutive_unwatered = 0
否则：consecutive_unwatered += 1
然后 watered_today = false
若 consecutive_unwatered >= 2：变为 WEED
```

连续两次日终未浇水即变杂草。只漏一天会存活，但第二天必须浇水。

### 10.4 一次性作物产量公式

一次性作物创建时 `yield_units=1`。WATER 时若 age 位于奖励窗口：

```text
bonus = 2  当 fertilized_until_day >= current_day
bonus = 1  否则
yield_units = min(max_yield, yield_units + bonus)
```

- 窗口外浇水只负责存活，不增加产量。
- HARVEST 必须达到 `first_yield_day`。
- HARVEST 一次取走全部 yield，并立即把植物格清空。
- 可以在最早收获日提前收获较低产量，也可以等待更高产量，但要承担维护、衰减和机会成本。

逐作物解释：

- WHEAT：初始 1；无肥在 age2/3/4 三次奖励后为 4；施肥可达 6。
- CARROT：初始 1；无肥在 age2/3 后为 3；施肥可达 4。
- MELON：初始 1；无肥从 age6 起每天 +1，在 age10 达 6；age11/12 因 cap 不再增加；
  若 age6–8 均处于施肥且浇水状态，可在 age8 达 cap。

### 10.5 持续型作物生产

持续型作物不因 WATER 立即增加 yield。日终从 current_day 进入 next_day 时，根据
next_day 的作物年龄检查生产 tick：

```text
days_since_first = next_day - planted_day - first_yield_day
days_since_first >= 0 且 days_since_first % interval == 0
```

每个作物最多 4 个生产 tick：

- 基础产量 1；
- 若“刚结束的 current_day”同时浇水且在施肥有效期，tick 产量为 2；
- `yield_units` 仍被 cap=4 截断，满仓时新产量会损失；
- 第 4 个 tick 完成后，把衰减起点设为再过 1 天。

持续型作物即使在某个生产日前一天只漏浇一次、尚未变杂草，也可产生基础 1；
但没有施肥翻倍。若已经连续两次漏浇，则先变杂草，不再生产。

### 10.6 施肥

- FERTILIZE 消耗执行单位背包中的 1 份 FERTILIZER。
- 一次施肥覆盖当前 day、day+1、day+2，共 3 个游戏日。
- 重复施肥只延长 `fertilized_until_day`，不会把倍率叠加到 3×、4×。
- 一次性作物在奖励窗口内、当天 WATER 时从 +1 变 +2。
- 持续型作物仅在生产 tick 对应的护理日同时 WATER 时从 1 变 2。
- 肥料不代替浇水。

### 10.7 收获与衰减

从 `max_lifespan_step` 开始，每隔一个 step：

```text
yield_units -= 1
若 yield_units <= 0：植物变为 WEED
```

- 一次性作物的衰减起点：`(planted_day + max_yield_day + 1) * turnsPerDay`。
- 持续型作物：第 4 个生产 tick 后，再过 1 天开始衰减。
- 衰减是每两个 step 减 1，不是每天减 1。
- 若衰减开始时 `yield_units=0`，第一次衰减立即把植物变为杂草。
- HARVEST 持续型作物不会移除植物；HARVEST 一次性作物会移除植物。
- 在同一个衰减 step，单位动作早于衰减，所以若单位已就位，可以先 HARVEST，
  再由解释器处理衰减。

## 11. 动物规则

### 11.1 动物参数

动物年龄同样为 `age = day - placed_day`。

| 动物 | 固定购买价 | 设施 | 产物 | 产物基准价 | 首次生产 age | 间隔 | 格上未收取上限 | 稳态基础 yield/格/日 |
|---|---:|---|---|---:|---:|---:|---:|---:|
| GOOSE | 300 | COOP | EGG | 50 | 4 | 每天 | 4 | 1.00 |
| COW | 400 | PASTURE | MILK | 160 | 8 | 每 2 天 | 6 | 0.50 |
| SHEEP | 500 | PASTURE | WOOL | 200 | 6 | 每 3 天 | 6 | 0.33 |

默认生产时间表：

- Goose：age 4,5,6,...；
- Cow：age 8,10,12,...；
- Sheep：age 6,9,12,...。

动物没有成熟后死亡或固定寿命；只要不逃走即可无限生产。`max_held` 只是格上未收取
产物上限，不是生命周期总产量。

官方对象表中的动物“Action Cost 1+1”仅指放置动物和建造设施各占一个单位动作，
不是完整物流成本；实际还需要购买、PICKUP、移动、每日 FEED、可选 CARE 和 HARVEST。

### 11.2 建造、购买、搬运、放置

1. 在空地执行 BUILD_COOP 或 BUILD_PASTURE；建造不扣金币，只消耗单位动作。
2. 用市场 BUY_ANIMAL，动物进入 shed；仓库满则购买失败。
3. 单位在仓库入口 PICKUP 动物到自己背包。
4. 单位站到匹配空设施上执行 PLACE；恰好放置 1 只。

BUY_ANIMAL 在市场阶段，晚于单位阶段，因此不能在购买同回合 PICKUP/PLACE。

### 11.3 新放置动物和缺粮

新动物：

```text
consecutive_unfed = 0
fed_today = false
cared_today = false
yield_units = 0
pending_care_bonus = 0
fertilizer_available = false
```

日终时：

```text
若 fed_today：consecutive_unfed = 0
否则：consecutive_unfed += 1
若 consecutive_unfed >= 2：动物逃走，空设施保留
```

- 新动物放置当天不喂，日终变为 1，仍存活。
- 连续第二个日终仍未喂，则先逃走，不生产、也不生成肥料。
- FEED 每天最多成功一次，消耗执行单位背包中的 1 WHEAT。
- 小麦在 shed 中不能直接喂，必须先 PICKUP。

### 11.4 正常生产与未喂时生产

在计划生产日，只要动物没有因连续两天未喂而先逃走：

- 基础产量总是 1；
- 当天只漏喂一次时，仍产生基础 1；
- 但未喂时不能获得 CARE 奖励，已有 pending bonus 会被清零；
- 产物加到格子的 `yield_units`，并受 `max_held` 截断；
- 满格时多余产量和 CARE bonus 均会浪费。

### 11.5 CARE 精确结算顺序

CARE 本身只设置 `cared_today=true`。日终顺序是：

1. 检查动物是否逃走；
2. 若当日进入生产 tick，结算“此前已积累”的 pending bonus；
3. 生产 tick 后把 pending bonus 清零；
4. 若刚结束的当天同时 `fed_today && cared_today`，再为下一次生产积累 +1；
5. 重置 fed_today/cared_today。

因此生产日当天的 CARE 不增加当前 tick，而是进入下一次生产的 bonus。过去积累的
bonus 只有在生产日也喂食时才兑现；生产日漏喂会永久丢失该 bonus。

### 11.6 肥料生产

- 每只仍存活的动物每天日终将 `fertilizer_available=true`。
- 与是否 FEED、CARE、是否产出商品无关。
- 未收集肥料不累计；标志连续保持 true，仍只有 1 份。
- COLLECT_FERTILIZER 成功后标志变 false，1 份肥料进入执行单位背包。

### 11.7 动物收获与设施

- HARVEST 取走格上全部 EGG/MILK/WOOL 到单位背包，并将格上 yield 清零。
- 收获不会移除动物。
- DIG 不能删除有动物的设施。
- 动物逃走后设施保留为空设施，可再次放置匹配动物，也可 DIG。

## 12. 农夫、雇工和每日重置

### 12.1 主农夫

- 每位玩家始终有 1 名主农夫。
- 每天开始位于 `(4,4)`。
- 日终背包自动入仓，然后位置重置。

### 12.2 雇工费用

当日已雇 n 人时，下一次 HIRE：

```text
cost = farmHandCostMult * fib(n)
fib(0..)=1,1,2,3,5,8,13,21,...
```

默认连续雇工价格为 1、1、2、3、5、8、13、21……。日终 `hires_today` 清零，
次日重新从 1 开始。官方 Python 解释器没有显式雇工数量上限，实际受资金制约。

### 12.3 雇工出生位置

在四个仓库入口中选择当前占用人数最少的位置；相同时按 NW、NE、SW、SE 顺序。
单位允许重叠。主农夫默认占 NW，因此当天第一个 hand 通常出生在 NE 入口 `(5,4)`，
即使 NE 尚未购买也不受影响，因为锁定格可行走。

### 12.4 日终清理

- 所有单位背包依序尝试入仓，溢出丢弃；
- 主农夫回到默认入口；
- 所有 hands 消失；
- `hires_today=0`；
- `private.inventories` 重置为一个空的主农夫背包。

最后一个终止帧之前不会执行 day29 的日终清理，详见第 3 节。

## 13. 市场规则

### 13.1 市场中的商品

共享动态库存和价格覆盖 9 种商品：

```text
WHEAT, CARROT, TOMATO, STRAWBERRY, MELON,
EGG, MILK, WOOL, FERTILIZER
```

- 种子供应无限、价格固定。
- 动物供应无限、价格固定。
- BUY_PRODUCT 只允许 WHEAT 和 FERTILIZER，价格动态。
- SELL 允许所有 9 种商品。
- 市场 SELL 只能消耗玩家 shed。

### 13.2 市场动作

| 动作 | 价格 | 进入/离开位置 | 失败条件 |
|---|---|---|---|
| `BUY_SEED crop n` | 固定种子价 | 增加 private.seeds | 参数错、钱不足 |
| `BUY_ANIMAL animal n` | 固定动物价 | 进入 shed | 参数错、钱不足、shed 满 |
| `BUY_PRODUCT item n` | 动态买价 | 市场库存→shed | 非 WHEAT/FERTILIZER、钱不足、shed 满 |
| `SELL item n` | 动态卖价 | shed→市场/现金 | shed 无货、非法商品 |
| `HIRE` | 当日 Fibonacci | 增加 hand | 钱不足 |
| `BUY_LAND` | 1000/2000/4000 | 解锁下一象限 | 钱不足或已全解锁 |

数量 n 会转换为整数，必须大于 0。一个大数量订单按单位逐个执行，直到完成或第一次
因钱/仓库/库存不足而停止。每回合每位玩家只取市场列表前 10 单。

### 13.3 双方锁步成交

市场按订单索引 i 处理双方第 i 单：

1. 先处理该索引上的 HIRE/BUY_LAND（一次性原子动作）；
2. 对 SELL/BUY_*，双方每次各处理 1 单位；
3. 双方在本单位成交前使用同一个市场库存报价；
4. 然后提交双方成交，更新库存；
5. 重复到该索引两边订单结束；
6. 刷新观察中的市场价格，再进入下一订单索引。

双方同卖一个商品时，同一锁步单位获得相同的卖价；提交后库存可一次增加 2，下一
单位价格可能变化。订单顺序、与对手订单的索引对齐以及数量拆分都会影响成交路径。

### 13.4 动态价格公式

每种商品独立计算：

```text
price(inv) = base + sign * amp * f(abs(inv-I0))

sign = +1，inv < I0（稀缺）
sign = -1，inv >= I0（过剩侧；inv=I0 时差值为0）
amp = target * base / f(T)
f ∈ {linear, sq, sqrt, log, log10}
log(x) 实际使用 ln(1+x)，log10 使用 log10(1+x)
最终价格 = max(1, int(round(price)))
```

这里 `round` 是 Python 对 binary64 float 的银行家舍入。价格下限为 1。

### 13.5 默认价格曲线

| 商品 | base | I0 | T | 稀缺函数 | 稀缺 target | 过剩函数 | 过剩 target | P(I0−T) | P(I0+T) | P(I0+2T) |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|
| WHEAT | 25 | 10000 | 400 | sqrt | 0.80 | log | 0.20 | 45 | 20 | 19 |
| CARROT | 35 | 10000 | 450 | log | 0.20 | sqrt | 0.70 | 42 | 10 | 1 |
| TOMATO | 60 | 10000 | 200 | linear | 0.40 | sqrt | 0.60 | 84 | 24 | 9 |
| STRAWBERRY | 120 | 10000 | 100 | sqrt | 0.70 | linear | 1.60 | 204 | 1 | 1 |
| MELON | 250 | 10000 | 300 | log | 0.20 | sq | 3.60 | 300 | 1 | 1 |
| EGG | 50 | 10000 | 332 | linear | 0.40 | log | 0.20 | 70 | 40 | 39 |
| MILK | 160 | 10000 | 122 | sqrt | 0.60 | linear | 1.60 | 256 | 1 | 1 |
| WOOL | 200 | 10000 | 105 | log | 0.20 | sq | 3.20 | 240 | 1 | 1 |
| FERTILIZER | 100 | 10000 | 200 | linear | 0.40 | linear | 0.40 | 140 | 60 | 20 |

### 13.6 买卖报价细节

- SELL 使用成交前库存报价。
- BUY_PRODUCT 使用 `inventory-1` 的成交后库存报价。
- 因此在没有其他变化时，先买 1 再立刻卖 1 的净收益为 0。
- SELL 成交价若为 1，商品从玩家 shed 删除、玩家仍得到 1 元，但市场库存不增加；
  防止继续向过剩侧无限推高库存。
- BUY_PRODUCT 会减少共享库存；种子和动物购买不改变 9 种商品市场库存。
- 城镇消费也减少市场库存，但不会向任何玩家支付金币。

## 14. 城镇需求系统

### 14.1 镇中心

- 在 `step % 24 == 0` 时消费一次。
- 默认从 WHEAT/CARROT/TOMATO/STRAWBERRY/MELON/EGG/MILK/WOOL 各减 1。
- 不消费 FERTILIZER。
- `step=0` 也满足条件，因此第一回合动作之后就发生第一次镇中心消费。
- 默认完整执行路径中 step=0,24,...,696 共 30 次。

### 14.2 商店解锁

- 默认每进入 day 3、6、9、12、15、18、21、24 时解锁一个实例。
- 最多 8 个实例，因此默认 day27 不再新增。
- 从 8 种商店中均匀、有放回抽取；同名商店可以重复。
- 已解锁商店永久存在，每个实例独立消费。

### 14.3 商店商品表

| 商店 | 每个消费 tick 的商品 |
|---|---|
| BAKERY | EGG 1、WHEAT 1 |
| PIZZA_SHOP | MILK 1、TOMATO 1、WHEAT 1 |
| BRUNCH_SPOT | EGG 1、WHEAT 1、STRAWBERRY 1 |
| YARN_STORE | WOOL 2 |
| ICE_CREAM_SHOP | STRAWBERRY 1、MILK 1、WHEAT 1 |
| PET_CAFE | CARROT 2 |
| SMOOTHIE_SHOP | STRAWBERRY 1、MILK 1 |
| FARMERS_MARKET | WHEAT 1、CARROT 1、TOMATO 1、STRAWBERRY 1 |

每个实例在 `step % 4 == 0` 时消费。默认每天 6 个 tick；重复商店按实例数倍增。
消费直接降低共享市场库存，随后刷新价格，因此商店是持续的外生需求来源。

## 15. 杂草与随机性

### 15.1 杂草来源

1. 植物连续两天未浇水；
2. 植物寿命后逐 step 衰减到 0；
3. 日终在空闲已解锁格子上随机生成。

DIG 可以清理杂草，但不产生物品。

### 15.2 日终随机数

每个 day 创建独立 Python RNG：

```python
random.Random((episode_seed * 1_000_003) ^ day)
```

然后严格按以下顺序消耗随机数：

1. 玩家 0 农场按 y、x 扫描；仅 `tile is None` 时调用一次 `rng.random()`；
2. 玩家 1 农场同样扫描；
3. 若 next_day 需要解锁商店，再用同一个 RNG 从 `sorted(SHOPS)` 中 choice。

由此产生一个重要但容易忽略的规则：商店抽取并非只由 seed/day 决定，还受双方
当日日终空格数量影响，因为不同空格数会使 RNG 在 choice 前消耗不同数量的 draw。
空地规划可以改变未来随机流；模型不能把商店序列当成预先固定、与行为无关的外生表。

对局 seed 从 agent observation 中移除，但保存在 replay 的 `env.info['seed']` 中。

## 16. 日终完整顺序

当 `(step+1)%24==0` 时，对玩家 0 后玩家 1 依次执行：

1. 植物缺水计数、杂草转换和持续型作物生产；
2. 动物缺粮计数、逃走、生产、CARE 入账、肥料生成和日标志重置；
3. 在空闲已解锁格生成随机杂草；
4. 主农夫和雇工背包按顺序自动入仓，溢出丢弃；
5. 主农夫回到默认仓库入口；
6. 删除全部雇工；
7. `hires_today=0`；
8. 背包列表重置。

两位玩家完成后，若进入商店解锁日，再使用同一 day RNG 抽取商店。

## 17. 非法动作与边界清单

- 非法动作不报错、不扣动作外资源，通常静默 no-op。
- 移动越界 no-op；移动进入锁定土地合法。
- 格子动作在 LOCKED 上 no-op；仓库动作例外。
- 重复 WATER/FEED/CARE 同日 no-op。
- HARVEST 总是拿走格上全部 yield，不能指定数量。
- 一次性植物未到 first_yield_day，即使内部有初始 yield=1，也不能收获。
- 同作物 PLANT 请求超过种子数时全部失败。
- 空设施没有 animal 键；有动物设施不能 DIG。
- DROP 溢出会丢失；PLACE-to-shed 溢出会留在背包。
- 市场只看 shed，不看单位背包。
- 超过 10 个市场订单的尾部订单直接忽略。
- HIRE 在单位阶段之后发生，新 hand 同回合不能行动。
- 当日最后一小时才 HIRE 的 hand 可能在随后的日终立刻消失，完全没有行动机会。
- 动物生产日只漏喂一次仍产基础 1，但 CARE bonus 丢失。
- 动物第二次连续漏喂时先逃走，不再产物或肥料。
- ongoing crop 只有 4 个生产 tick，但施肥后每 tick 可为 2；及时收获可累计超过 4。
- 终局只看 money，所有未售资产价值为 0。

## 18. 建模时必须保留的事实变量

本节不是固定模型方案，而是从规则直接推出的最低信息需求。

### 18.1 时间与截止

- `step/day/hour`；
- 距离下一个日终、生产 tick、施肥失效、衰减、商店消费和终局的距离；
- 最后一次可执行动作是 day29/hour22 的实现边界。

### 18.2 每块植物地

- crop、age、watered_today、连续缺水；
- 当前 yield、下一次/剩余生产 tick；
- 施肥剩余天数；
- 距衰减开始的 step；
- 维护、收获、清地的动作和路径成本。

### 18.3 每只动物

- animal、age、下一次生产日；
- fed/cared、连续缺粮；
- pending CARE bonus；
- 格上库存与 max_held 空间；
- fertilizer_available；
- 喂养所需小麦是否位于正确单位背包。

### 18.4 物流与行动能力

- 所有单位位置、背包和到任务的距离；
- shed 剩余容量；
- 当日雇工数量和下一名雇工价格；
- 任务是否能在日终前完成；
- 同回合 PLANT、DROP→SELL 等资源约束。

### 18.5 经济与对手

- 当前银行余额与必须保留的现金；
- 每种商品 market inventory、当前价、局部价格斜率；
- 已解锁商店实例和每种商品未来确定性需求；
- 自己可见的生产管线与预计供给；
- 对手公开土地、作物、动物、单位、银行余额和市场变化；
- 对手私有库存只能从可见历史推断，不能直接作为 actor 输入。

### 18.6 训练目标边界

- 官方 raw reward 是终局 money；
- 竞赛目标是 W/D/L 与 Bradley-Terry，而不是最大金币差；
- 任何 dense reward、潜在价值、库存估值或辅助预测都属于训练设计，必须与官方
  终局目标分开记录和消融。

## 19. 官方模拟器与本地 JAX 模拟器的角色

- 官方 Python 模拟器：规则真值、对局回放、提交验证和 parity 参考。
- 本地 JAX 模拟器：固定形状 GPU 批处理、self-play、Arena 和 PPO 训练。
- JAX 版本已经对 100 个未见 seed × 720 帧（72,000 帧）逐状态验证，整数状态、
  市场价格、status 和 reward 零误差；随机/非法动作另有 11,520 帧差分验证。
- 修改规则核心后必须重新生成 parity 回执；特征、网络、动作抽象和 League 可以在
  规则核心外迭代。

详见 [JAX GPU 模拟器说明](../gpu_sim/README.md) 和
[最终验收报告](../gpu_sim/FINAL_ACCEPTANCE.md)。

## 20. 规则核验速查

在冻结建模规格或实现新动作编译器前，至少确认：

- [ ] 使用 719 次实际动作推进，而非盲目执行 720 次；
- [ ] day29/hour23 是终止观察，不再行动；
- [ ] actor 没有对手 private、seed 或未来商店信息；
- [ ] PLANT 同作物请求先做原子种子检查；
- [ ] 单位动作早于市场动作；
- [ ] 市场 SELL 只从 shed 出货；
- [ ] BUY/HIRE 结果不能被同回合更早的单位阶段使用；
- [ ] 新植物种植当天必须浇水；
- [ ] ongoing crop 生产 tick、产量 cap 和衰减分别建模；
- [ ] 动物 CARE 结算顺序正确；
- [ ] 日终背包溢出会丢失；
- [ ] 商店 RNG 受双方空格数影响；
- [ ] 价格使用 Python round 和 $1 下限；
- [ ] official raw reward 与训练 WDL reward 明确区分；
- [ ] 所有未售资产终局价值为 0。
