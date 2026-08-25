# Rank7 ActiveMusyoku 对 FC24B 优化与冻结报告

## 1. 结论

天梯第 7 名 `ActiveMusyoku` 家族已经冻结为 `rank07_active_v32`。

- 两个最终 512-seed 双座位独立面板合计 `1840/2048 = 89.84%`；
- 座位 0 和座位 1 均为 `920/1024 = 89.84%`；
- 两个最终面板分别为 `91.31% / 88.38%`；
- 所有对局完成 719 步，`hard_counter_total = 0`；
- 在线 one-step-JIT Agent 另测 256 局为 `220/256 = 85.94%`；
- 在线与离线的我方现金、FC24B 现金和最终路线 mismatch 全部为 0。

该家族第一商店版已经很强，但没有在刚超过 50% 后停止。我们继续筛选第二商店安全后缀，并在三个开发/确认随机库和两个最终留出库上逐级验证，最终将独立胜率稳定在约 90%。

## 2. 经营风格与来源

ActiveMusyoku 是重牛、大规模草莓家族：

- 典型规模约 11 牛、4 羊、2 块额外土地；
- 草莓约 41 株，是主要作物现金流；
- 同时保留甜瓜、胡萝卜与大量小麦，支撑前期现金和动物饲料；
- 相比 Rank5 MiMi 的重牛混作，它进一步提高牛与草莓规模；
- 相比 Rank6 KANTA 的 6 牛/6 羊均衡结构，它明显偏向牛奶长期回报。

来源：

- 天梯第 7 名 `ActiveMusyoku`；
- submission `55691574`；
- 84 条完整获胜 Replay，编译拒绝数为 0；
- 路线银行：`artifacts/rank07_active_trace_bank_v1.npz`；
- 来源 Episode、奖励、座位、商店序列和 SHA256：`receipts/rank07_active_trace_bank_v1.json`。

## 3. 行为去重与第一商店路由

84 条来源 Replay 只有 10 套真正不同的运行时行为。这里的“相同”不只检查八组原始单位/市场动作，还检查恢复逻辑会读取的：

- `expected_unit_pos`；
- `expected_unit_active`。

总计完成 840 次逐字段字节一致性检查。来源奖励、Episode ID 与商店历史只作为证据和路由元数据，不作为强制路线的行为输入。

在 `seed=456001..456128` 上对全部 84 条路线完成 21,504 局双座位筛选：

- `all_done = true`；
- `hard_counter_total = 0`；
- 最强固定行为为 `211/256 = 82.42%`；
- 经第一商店路由后，开发地图收敛为 `[19,19,19,42,42,42,42,52]`。

该第一商店地图在独立 `seed=472001..472512` 上达到 `926/1024 = 90.43%`。

## 4. 第二商店安全后缀

后缀候选必须在 step 72–143 的八组原始动作逐字节相同，确保第二商店出现时不会把已有投资接到不兼容的生产链。共完成 528 次精确前缀检查。

开发阶段使用 seed456 与 seed464 构建候选；seed472 与 seed480 比较保守 5 条和扩展 7 条切换：

| 面板 | 第一商店版 | 保守5条 | 扩展7条 |
|---|---:|---:|---:|
| seed472001 | 926/1024 | 933/1024 | 939/1024 |
| seed480001 | 903/1024 | 913/1024 | 919/1024 |

扩展 7 条在两个确认面板均最强，因此在最终留出测试前锁定：

| 第一商店 → 第二商店 | 基线 route | 后缀 route |
|---|---:|---:|
| Bakery → Ice Cream | 19 | 2 |
| Bakery → Pizza | 19 | 2 |
| Brunch → Brunch | 19 | 2 |
| PET → PET | 42 | 2 |
| PET → Pizza | 42 | 2 |
| Smoothie → Smoothie | 42 | 2 |
| Smoothie → Yarn | 42 | 19 |

这些条件只读取公开出现的第一、第二商店，不使用未来事件、Replay ID、对手身份或隐藏状态。

## 5. 最终独立验证

扩展 7 条确定后不再修改，随后运行两个新随机面板：

| 面板 | 第一商店版 | Rank7 V32 | V32胜率 | 座位0 | 座位1 | 净增胜场 | 硬错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed488001 | 926/1024 | 935/1024 | 91.31% | 91.80% | 90.82% | +9 | 0 |
| seed496001 | 903/1024 | 905/1024 | 88.38% | 87.89% | 88.87% | +2 | 0 |
| 合计 | 1829/2048 | 1840/2048 | 89.84% | 89.84% | 89.84% | +11 | 0 |

## 6. 在线语义验收

`seed=504001..504128` 双座位 256 局：

- `220/256 = 85.94%`；
- 座位 0 为 `108/128 = 84.38%`；
- 座位 1 为 `112/128 = 87.50%`；
- 全部完成 719 步；
- `hard_counter_total = 0`。

同 seed 离线路线矩阵逐局核对：

- candidate cash mismatch = 0；
- FC24B cash mismatch = 0；
- final route mismatch = 0。

`invalid_intent_total` 是来源 Replay 原始意图经合法守卫修复的诊断量，不是环境收到非法动作；实际环境动作没有硬错误或卡死。

## 7. 冻结交付物

- 冻结配置：`configs/rank07_active_v32_frozen_best.json`；
- 第一商店地图：`artifacts/rank07_active_stable_win_map_dev640_v6.json`；
- 第二商店地图：`artifacts/rank07_active_secondshop_risk7_canonical_v22.json`；
- 全量筛选：`receipts/rank07_active_all84_vs_fc24b_screen_seed456001_n128x2_v1.json`；
- 独立最终面板：`receipts/rank07_active_final_eval_seed488_v25.json`、`rank07_active_final_eval_seed496_v27.json`；
- 在线回执：`receipts/rank07_active_online_risk7_vs_fc24b_seed504001_n128x2_v30.json`；
- 在线/离线一致性：`receipts/rank07_active_online_screen_parity_seed504_v31.json`。

Rank7 V32 进入 FC24B 克星池。它补充了重牛、大规模草莓骨架，与 Rank4 羊优先、Rank5 重牛混作和 Rank6 牛羊均衡形成明显互补。
