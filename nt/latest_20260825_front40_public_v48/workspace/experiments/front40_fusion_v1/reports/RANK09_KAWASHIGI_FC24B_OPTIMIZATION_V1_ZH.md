# Rank9 KAWASHIGI 对 FC24B 优化与冻结报告

## 1. 结论

天梯第 9 名 `KAWASHIGI` 家族冻结为 `rank09_kawashigi_v115`。

- 最终全新随机库双座位 `551/1024 = 53.81%`；
- 座位 0 为 `279/512 = 54.49%`，座位 1 为 `272/512 = 53.13%`；
- 所有对局完成 719 步，`hard_counter_total = 0`；
- Agent 只依据当前已经公开的前三家城镇商店切换路线；
- 不使用玩家身份、Replay ID、seed 或未来事件。

该家族已经做过 179 条完整获胜路线筛选、多随机库复核、第一至第三商店动态路由、公开状态 LGBM 排序器和 Pizza 弱分支消融。最终胜率虽明显低于 Rank4/Rank7，但已在全新随机库和两个座位上严格超过 50%。现有 KAWASHIGI Replay 在 Pizza 开局下普遍弱，继续在同一家族内特调的预期收益很低，因此按潜力门冻结并转向下一家族。

## 2. 来源与路线银行

- 天梯排名：9；
- 玩家：`KAWASHIGI`；
- submission：`55540317`；
- 完整获胜 Replay：179 条；
- 行为唯一完整路线：179 条；
- 编译拒绝：0；
- 路线银行：`artifacts/rank09_kawashigi_trace_bank_v1.npz`；
- Episode、奖励、座位、商店序列与 SHA256：`receipts/rank09_kawashigi_trace_bank_v1.json`。

最终运行时实际使用 15 条路线：

`17, 31, 35, 38, 54, 63, 65, 81, 94, 100, 115, 116, 118, 169, 175`

## 3. 已完成的潜力搜索

### 3.1 全量路线筛选

首先在 seed576 与 seed600 两个随机库上对全部 179 条路线进行双座位筛选。随后保留跨库稳健的 24 条路线，在 seed608、616、624、632、640、648 上继续评估。

这一步说明：KAWASHIGI 不是靠一条固定路线稳定克制 FC24B，收益主要来自观察公开商店后选择不同完整经营路线。

### 3.2 动态路由

依次验证了：

1. 第一商店路线图；
2. 第二商店兼容后缀；
3. 第三商店兼容后缀；
4. 只看当前公开状态的 LGBM 路线排序器；
5. 按第一商店在 pooled、stable、minimax 三棵树中选择。

公开状态 LGBM 在开发库有收益，但在全新 seed640 只有 45.70%，说明路线之间相当一部分事后差异来自尚未发生的随机事件，不能安全地通过当前公开状态预测。因此没有把它放进最终 Agent。

最终父树 V107 在 seed648 在线逐步运行达到 `524/1024 = 51.17%`，且与离线路线矩阵的我方现金、对手现金、最终路线逐局 mismatch 均为 0。

## 4. Pizza 弱点与最后修正

V107 在 Pizza 第一商店条件下只有：

- seed640：`31/162 = 19.14%`；
- seed648：`23/134 = 17.16%`。

复核当前24条路线后，固定 route100 分别达到：

- seed640：`35/162 = 21.60%`；
- seed648：`32/134 = 23.88%`。

因此 V115 在第一商店公开为 Pizza 后固定 route100，不再根据第二、第三商店切换。该修正使 seed648 整体从 `51.17%` 提高到 `52.05%`。seed640 为 `50.49%`，说明增益不大但方向稳健。

全179路线的早期筛选和后续24路线大样本均表明 Pizza 是该来源家族的结构性弱点，而不是遗漏一条明显强路线。继续扩大同来源 if/else 的收益低于转向新家族。

## 5. 最终独立验收

最终模型锁定后，使用未参与 V115 选择的 `seed=656001..656512`：

| 座位 | 胜场/局数 | 胜率 | 719步完成 | 硬错误 |
|---|---:|---:|---:|---:|
| 0 | 279/512 | 54.49% | 是 | 0 |
| 1 | 272/512 | 53.13% | 是 | 0 |
| 合计 | 551/1024 | 53.81% | 是 | 0 |

最终回执：`receipts/rank09_kawashigi_v115_online_vs_fc24b_seed656001_n512x2_independent_v118.json`。

`invalid_intent_total` 统计的是来源 Replay 中已不适合当前随机局面的原始意图。合法守卫会在送入环境前将其改成可执行动作或 PASS，因此它不代表向官方环境发送了非法动作。实际环境 `hard_counter_total = 0`，全部对局正常终止。

## 6. 冻结交付物

- 冻结配置：`configs/rank09_kawashigi_v115_frozen_best.json`；
- 路线银行：`artifacts/rank09_kawashigi_trace_bank_v1.npz`；
- 最终三商店树：`artifacts/rank09_kawashigi_v115_pizza_fixed_route100.json`；
- 在线运行器：`tools/run_three_shop_trace_tree_vs_fc24b.py`；
- 全量筛选回执：`receipts/rank09_kawashigi_all179_vs_fc24b_screen_seed576001_n128x2_v1.json`、`rank09_kawashigi_all179_vs_fc24b_screen_seed600001_n128x2_v35.json`；
- 最终独立回执：`receipts/rank09_kawashigi_v115_online_vs_fc24b_seed656001_n512x2_independent_v118.json`。

Rank9 V115 进入 FC24B 克星池。它的价值主要是补充一个依赖前三个公开商店进行完整路线切换的混合经营家族，而不是作为最终融合的 Pizza 分支。
