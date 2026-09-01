# W4 自主计划记忆与产能门验收报告

日期：2026-08-28  
规则引擎：队友 `fast_kaggriculture`，官方 1.32.7 语义  
模式：无 backbone、无 Replay 日历、无对手身份输入

## 1. 验收结论

**有条件失败，不冻结 W4。**

本阶段已经证明跨日承诺记忆和按产能触发土地投资具有真实正向价值，也找到了一个几乎不影响现金却能消除稀有动物逃跑的通用调度安全能力。但当前自主规划器在独立 1,024 局静止对手下平均终局现金只有 `112,458`，尚未达到开发文档规定的 `130k` 收益门；面对 G001 的胜率也只有约 `1.76%`。因此不能进入高手日历调参、对手身份路由或正式 Arena。

## 2. 已验证能力

### 2.1 跨日承诺记忆

`autonomous_commitment_persistence` 会保留上一日仍未兑现的动物、作物和土地承诺，避免规划器每天从零开始后让有价值项目凭空消失。

在 256 seed、双座位的受控网格中，`0.75` 相比 `0` 将静止对手平均终局现金从约 `107,938` 提升到约 `114,027`，增益约 `6,088`。方向成立，但固定比例仍不是最终的 KEEP/SCALE/DEFER/SWITCH/CANCEL 决策器。

证据：`receipts/w4_autonomous_commitment_persistence_grid_passive_seed2385001_n256x2_v1.json`

### 2.2 按产能触发土地投资

`land_capacity_trigger_fraction` 不使用固定日期，而是在已选生产项目接近当前地块产能时，把下一块土地成本计入边际项目价值。

`0.84` 在对应独立网格中取得约 `114,624` 平均现金，优于该组 `0` 回滚臂约 `3,252`；但对 G001 只带来小幅现金改善，没有解决竞争强度问题。

证据：

- `receipts/w4_autonomous_land_capacity_trigger_grid_passive_seed2388001_n256x2_v1.json`
- `receipts/w4_autonomous_land_capacity_trigger_grid_vs_g001_seed2388001_n256x2_v1.json`

### 2.3 hard latest-start reservation

当前 1,024 局基准发现一例可避免动物逃跑：第 23 天最后一步，一头位于远端的牛已连续一天未喂，场上有足够小麦和 12 个单位，但没有单位提前占用可达路径。

启用 `hard_latest_start_reservation=1` 后：

| 指标 | 关闭 | 开启 |
|---|---:|---:|
| 静止对手平均现金 | 112,463 | 112,458 |
| 可避免动物损失 | 1 | 0 |
| 终局溢出 | 0 | 0 |
| 对 G001 胜率 | 1.76% | 1.76% |

现金差约 `-5`，属于统计和轨迹级中性；硬损失从 1 降到 0。该能力作为通用安全规则保留。

证据：

- `receipts/w4_autonomous_hard_latest_start_grid_passive_seed2392001_n512x2_v1.json`
- `receipts/w4_autonomous_hard_latest_start_grid_vs_g001_seed2392001_n256x2_v1.json`
- `artifacts/w4_autonomous_current_baseline_passive_seed2392444_seat1_animal_loss_trace_v1.json`

## 3. 被拒绝的方向

### 3.1 盲目提前买地

`proactive_land_investment` 在没有项目产能证据时提前买土地，使静止对手平均收益降到约 `95k–96k`。拒绝，不再搜索固定提前天数。

### 3.2 重复作物生命周期估值

仅把解析估值扩展成多个种植周期，会让规划目标增加，但当前执行与现金化能力无法同步兑现，平均收益下降。拒绝直接晋级。

### 3.3 按作物成熟期延长晚期种植

在 1,024 局受控 A/B 中：

- 静止对手平均现金只增加约 `163`；
- 非 PASS 操作平均增加约 `573`；
- 对 G001 平均现金增加约 `2,851`，但出现累计 `96` 单位日终仓储溢出。

这说明作物“理论上来得及成熟”不等于“人员有能力收获、回仓并在终局前卖成现金”。该开关默认保持关闭，直到项目估值同时纳入剩余调度容量和现金化窗口。

证据：

- `receipts/w4_autonomous_crop_specific_horizon_grid_passive_seed2390001_n256x2_v1.json`
- `receipts/w4_autonomous_crop_specific_horizon_grid_vs_g001_seed2390001_n256x2_v1.json`

## 4. 当前冻结基准

无-backbone当前回滚组合：

- `autonomous_commitment_persistence = 0.75`
- `land_capacity_trigger_fraction = 0.84`
- `hard_latest_start_reservation = 1`
- `crop_specific_terminal_horizon = 0`

独立静止对手验收：`512 seed × 双座位 = 1,024` 局。

| 指标 | 结果 |
|---|---:|
| 平均终局现金 | 112,458 |
| 中位数 | 113,852 |
| P10 | 89,906 |
| P90 | 132,450 |
| 最小值 | 71,878 |
| 最大值 | 153,420 |
| 可避免作物损失 | 0 |
| 可避免动物损失 | 0 |
| 日终溢出 | 0 |

## 5. 下一步唯一主线

不再调高手日历。下一步实现真正的项目级滚动状态：

1. 每个作物/动物/土地项目保存 `KEEP / SCALE / DEFER / SWITCH / CANCEL`；
2. 项目价值必须同时受剩余人员动作容量、可用地块、物流距离、仓储容量和终局现金化窗口约束；
3. 只修改发生显著价值变化或执行偏差的项目，其余项目保持；
4. 先把独立 1,024 局静止对手平均现金从 `112.5k` 提升到 `130k`，同时保持硬错误和溢出为 0；
5. 达标后再验证三种不同产业结构，随后才加入市场和未知对手适应。

