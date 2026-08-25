# Rank12 fufufukakaka 对 FC24B 潜力筛选与拒绝报告

## 结论

Rank12 `fufufukakaka` 不进入 FC24B 克星池。

- submission：`55692247`；
- 75条完整获胜Replay，拒绝0；
- 75条完整动作哈希全部唯一；
- 与 Rank1、Rank4–Rank11 的动作哈希重合0；
- seed728 双座位全量筛选19,200局，全部719步完成，硬错误0。

潜力结果：

- 最佳固定 route61：`4/256 = 1.56%`；
- 第一商店路由：`5/256 = 1.95%`；
- 使用每局终局结果选择路线的不可部署 oracle：`5/256 = 1.95%`；
- 第一商店路由平均分差：`-61,822`。

oracle 与可部署路由完全相同，说明不是缺少更复杂的公开状态路由，而是现有鹅路线离开来源 Replay 的随机条件后整体无法与 FC24B 竞争。按“潜力有限且反复修改预期收益低时可跳过”的门控，停止 Rank12 特调。

保留证据：

- 路线银行：`artifacts/rank12_fufufukakaka_trace_bank_v1.npz`；
- 来源回执：`receipts/rank12_fufufukakaka_trace_bank_v1.json`；
- 全量筛选：`receipts/rank12_fufufukakaka_all75_vs_fc24b_screen_seed728001_n128x2_v1.json`。
