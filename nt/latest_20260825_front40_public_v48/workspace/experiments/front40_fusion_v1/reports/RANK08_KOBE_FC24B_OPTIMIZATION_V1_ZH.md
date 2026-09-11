# Rank8 Kobe BRYANT 对 FC24B 优化与冻结报告

## 1. 结论

天梯第 8 名 `Kobe BRYANT` 家族已经冻结为 `rank08_kobe_v46`。

- 两个最终 512-seed 双座位独立面板合计 `1436/2048 = 70.12%`；
- 座位 0 为 `717/1024 = 70.02%`，座位 1 为 `719/1024 = 70.21%`；
- 两个最终面板分别为 `69.34% / 70.90%`；
- 所有对局完成 719 步，`hard_counter_total = 0`；
- 在线 one-step-JIT Agent 另测 256 局为 `180/256 = 70.31%`；
- 在线与离线的我方现金、FC24B 现金和最终路线 mismatch 全部为 0。

固定路线最高只有 44.53%。我们继续完成两轮全量路线筛选、第一商店跨来源路由、第二商店逐条件消融和三轮确认，最终独立胜率稳定在约 70%，没有在刚超过 50% 时停止。

## 2. 经营风格与来源

Kobe 是中等规模牛羊、较轻作物家族：

- 典型终局规模约 6 牛、4 羊；
- 保留草莓和小麦，但不像 Rank7 那样把草莓与牛扩到极端规模；
- 现金和市场响应更敏感，固定一条完整路线不够稳定；
- 它与 Rank1、Rank4、Rank5、Rank6、Rank7 的完整运行时行为哈希均无重合。

来源：

- 天梯第 8 名 `Kobe BRYANT`；
- submission `55668682`；
- 89 条完整获胜 Replay，编译拒绝数为 0；
- 路线银行：`artifacts/rank08_kobe_trace_bank_v1.npz`；
- 来源 Episode、奖励、座位、商店序列和 SHA256：`receipts/rank08_kobe_trace_bank_v1.json`。

## 3. 固定路线和第一商店路由

在 `seed=512001..512128` 对全部 89 条路线完成 22,784 局双座位筛选：

- `all_done = true`；
- `hard_counter_total = 0`；
- 最强固定 route88 只有 `114/256 = 44.53%`；
- 每局事后 oracle 为 `254/256 = 99.22%`，但该数字使用终局结果，只是诊断，绝不进入运行时。

同来源商店约束的第一商店路由为 58.20%。允许在看到第一家公开商店后，从所有经过实际模拟的路线中选择，提升到 67.58%。

第二个全量库 `seed=520001..520512` 又完成 91,136 局；合并两个开发库后，三种稳健选择准则一致收敛到：

`[88, 88, 88, 47, 88, 47, 47, 57]`

该地图在 seed520 为 `716/1024 = 69.92%`。

## 4. 第二商店消融

候选后缀必须在 step 72–143 的八组原始动作逐字节相同，共完成 497 次精确前缀检查。

直接把开发库上正向的16或21条切换全部加入会在 seed528 退化，证明小样本条件不能整包照收。随后逐条件拆解并继续验证：

- seed528：安全7条为 `727/1024 = 71.00%`；
- seed536：加入 `Farmers→Smoothie` 的8条版优于其余变体；
- seed544：8条版 `722/1024 = 70.51%`，4条低方差版 `707/1024 = 69.04%`。

三个确认库上，最终8条版合计 `2159/3072 = 70.28%`，因此在最终留出前锁定：

| 第一商店 → 第二商店 | 基线 route | 后缀 route |
|---|---:|---:|
| Bakery → Yarn | 88 | 31 |
| Brunch → Pizza | 88 | 47 |
| Farmers → Ice Cream | 88 | 52 |
| Farmers → Smoothie | 88 | 41 |
| Ice Cream → Yarn | 47 | 31 |
| PET → Pizza | 88 | 47 |
| PET → Yarn | 88 | 10 |
| Pizza → Yarn | 47 | 31 |

运行时只读取公开出现的第一、第二商店，不使用未来事件、Replay ID、对手身份或隐藏状态。

## 5. 最终独立验证

| 面板 | 第一商店版 | Rank8 V46 | V46胜率 | 座位0 | 座位1 | 净增胜场 | 硬错误 |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed552001 | 704/1024 | 710/1024 | 69.34% | 68.95% | 69.73% | +6 | 0 |
| seed560001 | 713/1024 | 726/1024 | 70.90% | 71.09% | 70.70% | +13 | 0 |
| 合计 | 1417/2048 | 1436/2048 | 70.12% | 70.02% | 70.21% | +19 | 0 |

## 6. 在线语义验收

`seed=568001..568128` 双座位 256 局：

- `180/256 = 70.31%`；
- 座位 0 为 `89/128 = 69.53%`；
- 座位 1 为 `91/128 = 71.09%`；
- 全部完成 719 步；
- `hard_counter_total = 0`。

同 seed 离线路线矩阵逐局核对：

- candidate cash mismatch = 0；
- FC24B cash mismatch = 0；
- final route mismatch = 0。

`invalid_intent_total` 是来源 Replay 意图经合法守卫修复的诊断量，不是环境收到非法动作；实际环境动作没有硬错误或卡死。

## 7. 冻结交付物

- 冻结配置：`configs/rank08_kobe_v46_frozen_best.json`；
- 第一商店地图：`artifacts/rank08_kobe_all89_stable_win_map_dev640_v16.json`；
- 第二商店地图：`artifacts/rank08_kobe_secondshop_safe8_fs_dev1152_v28.json`；
- 两轮全量筛选：`receipts/rank08_kobe_all89_vs_fc24b_screen_seed512001_n128x2_v1.json`、`rank08_kobe_all89_merged_seed520001_n512x2_v15.json`；
- 最终独立面板：`receipts/rank08_kobe_final_eval_seed552_v39.json`、`rank08_kobe_final_eval_seed560_v41.json`；
- 在线回执：`receipts/rank08_kobe_online_safe8fs_vs_fc24b_seed568001_n128x2_v44.json`；
- 在线/离线一致性：`receipts/rank08_kobe_online_screen_parity_seed568_v45.json`。

Rank8 V46 进入 FC24B 克星池。它提供了一个比 Rank7 更轻、更市场敏感的 6牛4羊骨架，可作为后续融合中的中等投入分支。
