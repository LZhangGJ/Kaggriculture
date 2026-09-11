# FC6：早期现金 Replay 路线互补性审计

日期：2026-08-21  
结论：**拒绝把该路线接入 FC2B。它不是当前短板的有效互补分支。**

## 1. 候选来源与真实性边界

- 来源 Replay：Episode `95981440`，James Holland（seat 0）。
- 原始终局现金：`146,145 : 141,632`。
- 观察到的资本结构为 `10 牛 + 4 羊`，约第 93 步开始出售草莓、第 102 步开始出售牛奶，明显早于 FC2B 的常见现金循环。
- 已将双方原始动作编译成独立 JAX 路线，并在原始 seed `399672617` 上逐步复现到完全相同的终局现金，证明 JAX 转录本身没有篡改收益。
- 该验证只证明“原 Replay 动作在原局面下可复现”，不证明它能自由滚动或成为可部署策略。

相关证据：

- `experiments/fusion_champion_v1/receipts/james_efe_95981440_source_reproduction_v1.json`
- `experiments/fusion_champion_v1/artifacts/james_95981440_early_cash_route_recovery_probe_v1.npz`

## 2. 独立事件自由滚动

使用独立事件 `seed_start=551501`、64 个 seed、双方换座，共 128 局/对手。Replay 路线加入通用的状态重同步、非法动作修复、市场订单裁剪和终局清仓，但不读取未来。

| 对手 | FC2B 胜局 | Replay 恢复路线胜局 | 两者都赢 | 仅 FC2B 赢 | 仅 Replay 路线赢 | 两者都输 | 事后理想二选一得分率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| K320 | 110/128 | 85/128 | 80 | 30 | 5 | 13 | 89.84% |
| Rank12 | 107/128 | 88/128 | 87 | 20 | 1 | 20 | 84.38% |
| PRT V6 | 116/128 | 91/128 | 90 | 26 | 1 | 11 | 91.41% |

证据：

- `experiments/fusion_champion_v1/receipts/fc2b_targets_seed551501_n64x2_v1.json`
- `experiments/fusion_champion_v1/receipts/james_95981440_early_cash_route_recovery_seed551501_n64x2_v2.json`

## 3. 判定

1. 该 Replay 的高现金依赖原局面的生产状态、市场成交和日内任务链；通用恢复后仍有大量市场订单被裁剪，说明它不是一个稳定的条件策略。
2. 它只分别救回 FC2B 对 K320、Rank12、PRT 的 `5、1、1` 局，却分别制造 `30、20、26` 个 FC2B 原本能赢的新败局。
3. 即使假设有一个能够事后预知结果的完美选择器，K320 与 Rank12 仍未达到 90%。因此继续训练路线选择器也无法从这两条路线中满足验收门。
4. 当前最优配置保持 FC2B，不修改 `CURRENT_BEST_RULE_CONFIG.json`。

## 4. 下一步

继续单线程下载 2700+ 全量 Replay。下一批候选必须先满足：

- 不是单局动作回放，而是能从多个 Replay 中归纳出的“公开状态条件 → 日内任务链/事务顺序”；
- 在独立事件、双方换座下，至少显著救回 FC2B 对 K320 或 Rank12 的共同败局；
- 先做逐局 oracle 上限审计。若两路线的理想组合仍低于 90%，直接淘汰，不进入规则路由或 LGBM。
