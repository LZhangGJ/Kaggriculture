# Rank11 peikopon 对 FC24B 优化与冻结报告

## 1. 结论

Rank11 `peikopon` 冻结为 `rank11_peikopon_v7`。

- 最终全新随机库 `593/1024 = 57.91%`；
- 座位0为58.01%，座位1为57.81%；
- 全部719步完成，硬错误0；
- 只读取已经公开的第一家城镇商店；
- 不使用身份、Replay ID、seed或未来事件。

该家族胜率过线，但平均分差为 `-2,001`，说明部分败局损失较大。三轮随机库已证明进一步同家族修改收益有限，因此冻结为互补拳法，不继续特调。

## 2. 来源与去重

- submission `55635759`；
- 113条完整获胜Replay，拒绝0；
- 113条完整动作哈希全部唯一；
- 与已冻结 Rank1、Rank4–Rank10 的动作哈希重合0；
- 来源收益中位数88,916，最高152,117。

证据：`receipts/rank11_peikopon_trace_bank_v1.json`。

## 3. 搜索过程

seed696 全量筛选113条路线、双座位共28,928局：

- 最强固定 route43：56.25%；
- 第一商店开发路由：70.31%；
- 事后 oracle：89.84%，只作诊断。

保留各商店前8名和固定前15名，去重为33条，在 seed704 复筛。跨两库 pooled 地图为69.14%/63.28%，minimax 为66.41%/64.45%。

在 seed712 的1024局大样本中，旧 pooled/minimax 分别回落到53.32%/51.66%。把三个库共同纳入稳健选择后，最终地图为：

`[43, 43, 43, 50, 42, 93, 50, 103]`

该地图在 seed696/704/712 分别为69.14%、60.94%、56.35%。

## 4. 最终独立验收

地图锁定后使用 `seed=720001..720512`：

| 座位 | 胜场/局数 | 胜率 | 平均分差 | 完成719步 | 硬错误 |
|---|---:|---:|---:|---:|---:|
| 0 | 297/512 | 58.01% | -1,983.64 | 是 | 0 |
| 1 | 296/512 | 57.81% | -2,018.85 | 是 | 0 |
| 合计 | 593/1024 | 57.91% | -2,001.24 | 是 | 0 |

最终回执：`receipts/rank11_peikopon_v4_online_vs_fc24b_seed720001_n512x2_independent_v6.json`。

## 5. 冻结交付物

- 配置：`configs/rank11_peikopon_v7_frozen_best.json`；
- 路线银行：`artifacts/rank11_peikopon_trace_bank_v1.npz`；
- 首店地图：`artifacts/rank11_peikopon_threepanel_minimax_firstshop_v4.json`；
- 全量筛选：`receipts/rank11_peikopon_all113_vs_fc24b_screen_seed696001_n128x2_v1.json`；
- 最终验收：`receipts/rank11_peikopon_v4_online_vs_fc24b_seed720001_n512x2_independent_v6.json`。

Rank11 V7 进入克星池，但后续融合应把它视为高方差补充分支，而不是主骨架。
