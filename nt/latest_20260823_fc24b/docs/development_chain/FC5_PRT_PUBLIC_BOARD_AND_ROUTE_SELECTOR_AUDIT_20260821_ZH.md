# FC5：FC2B 对 PRT 的公开盘面与路线选择审计

日期：2026-08-21

## 结论

本轮没有产生可晋级候选，当前最佳仍为 FC2B。

用户报告的 Public 快照为：

- FC2B：2705 分；
- 当时名次：第 35 名。

这条榜单信息证明修复后的提交具有实战强度，但它是用户提供的时点快照，不替代本地固定事件库、双方换座和官方 Python 验收。

## 1. 新独立事件库结果

事件库：seed `550001..550128`，每个 seed 双方换座。

| 对手 | 胜局/总局 | 得分率 |
|---|---:|---:|
| Rank12 | 223/256 | 87.11% |
| PRT V6 | 228/256 | 89.06% |

来源：

- `receipts/fc5b_fc2b_rank12_prt_public_board_seed550001_n128x2_v1.json`

## 2. PRT 败局的公开形态

PRT 败局不是随机均匀发生。第 12～20 天，对手公开盘面上的羊数量明显更高：

- 第 14 天，败局中的对手平均约 8.86 羊；
- 胜局中的对手平均约 5.04 羊；
- 第 18～20 天仍保持约 9.93 羊对 5.14 羊。

因此，“对手形成重羊产能”是可在线观察的败局形态。但败局形态只说明问题发生在哪里，不能直接证明某个动作能够修复。

来源：

- `receipts/fc5c_fc2b_rank12_prt_public_board_divergence_seed550001_n128x2_v1.json`

## 3. 出售时机消融：拒绝

测试了：

- 羊毛单品提前出售；
- 草莓、奶、羊毛组合提前出售；
- 提前出售数量上限 12、20、32、64；
- 根据公开重羊形态和 PRT 两类出售时间表提前 1～2 步出售。

核心结果：所有有效候选均为 113/128，没有比当前四步提前出售多救一局。部分候选仅提高平均利润，未改变胜负。

因此，PRT 剩余短板的主因不是“卖得不够早”。

来源：

- `receipts/fc5d_k320_prt_preempt_capmask_a_seed550001_n64x2_train_v1.json`
- `receipts/fc5d_k320_prt_preempt_capmask_b_seed550001_n64x2_train_v1.json`
- `receipts/fc5h_k320_prt_heavy_sheep_counter_a_seed550001_n64x2_train_v1.json`

## 4. 路线选择 Oracle：有上限，但暂不可部署

在第 72/120 步保留当前 FC2B，或锁定 K320 的现有合法路线：

- 训练段保守路线 Oracle：123/128，96.09%；
- 独立留出段保守路线 Oracle：125/128，97.66%。

这证明现有路线库本身能够覆盖大部分 PRT 败局。真正缺失的是稳定的在线选择信号。

### 4.1 单规则

直观规则“第一家商店为毛线店时改走双毛线路线”：

- 在一个分段救 6 局、伤 1 局；
- 在另一个分段救 5 局、伤 6 局。

29 个有经济含义的单阈值特征，经 5 折 seed 分组后，最优规则是永不切换。

### 4.2 决策树与 LGBM

- 小决策树：训练 OOF 可到 90.63%，留出只有 89.06%；
- LGBM：训练 OOF 90.63%，留出 A 从 90.63% 降到 89.06%，留出 B 保持 89.06%。

两者都存在“救一部分败局，同时破坏相近数量原胜局”的问题，不能接入正式 Agent。

来源：

- `receipts/fc5j_fc2b_prt_route_step72_seed550001_n64x2_train_merged_v1.json`
- `receipts/fc5i_fc2b_prt_route_step72_seed550065_n64x2_holdout2_v1.json`
- `receipts/fc5k_fc2b_prt_route_rescue_tree_step72_n64x2_v1.json`
- `receipts/fc5n_fc2b_prt_binary_route4_public_threshold_step120_n64x2_v1.json`
- `receipts/fc5o_fc2b_prt_binary_route4_score_delta_lgbm_step120_n64x2_v1.json`

## 5. 工程结论

1. FC2B 的 Shadow-PRT 卡死修复有效，Public 已获得用户报告的 2705 分；
2. PRT 已接近 90%，但不能用不稳定路线分类器强行跨线；
3. 出售时机方向已经充分消融，应停止继续微调；
4. 下一阶段优先处理 Rank12。公开盘面诊断显示，其败局更稳定地伴随小麦产能落后和草莓市场库存过高，应分别验证：
   - 小麦/草莓出售窗口；
   - 中期小麦生产兑现；
   - 只改变受影响商品的局部项目，不切换整条路线。

