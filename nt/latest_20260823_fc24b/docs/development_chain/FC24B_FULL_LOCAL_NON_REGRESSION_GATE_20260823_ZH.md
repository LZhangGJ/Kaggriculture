# FC24B 全本地 Agent 无退化门检报告（2026-08-23）

## 1. 结论

FC24B 通过本地无退化门检，可以替代 FC22 进入下一阶段的 Python/JAX 一致性实现。

- 官方规则版本：`1.32.7`
- 后端：RTX 3090 / JAX GPU
- 事件协议：每个对手 128 个固定随机事件，双方换座，共 256 局
- 对手总数：34（旧 roster 28 + 2026-08-22 最新公开强 Agent 6）
- 总对局数：8,704
- 所有对手单项胜率：均不低于 90%
- 相对 FC22：4 个 `loss -> win`，0 个 `win -> loss`
- 相对 FC22：1,313 局候选现金增加，0 局候选现金下降

因此，FC24B 的改动不是靠牺牲其他对局换取 PRT 的一局胜利。

## 2. 改动内容

FC24B 以 FC22 为基座，只增加一个终局补收能力：

1. 仅在第 30 天；
2. 单位当前站在已经成熟且有产量的作物上；
3. 原策略准备移动离开；
4. 当前正好是“收获 -> 最短回仓 -> DROP”仍能在第 719 步前完成的最后时刻；
5. 待收作物按当前市场价计算的价值，至少是单位随身货物价值的 2 倍；
6. 才插入一次补收、回仓、入库和终局出售。

不使用对手名字、路线 ID、随机 seed、固定坐标或未来事件。

## 3. 被拒绝的 FC24 初版

FC24 初版没有“作物价值 >= 2 × 随身货物价值”的机会成本门。

它虽然在最新 6 个 Agent 上没有新增败局，但造成了真实经济退化：

- Soil V26-H：平均候选现金约下降 127；
- Kaito V39：平均候选现金约下降 129；
- 大量对局的高价牛奶或羊毛被推迟到最后一个市场步骤，订单槽拥挤时无法及时出售。

因此不能只看胜率汇总。FC24 初版已判定为不合格，不应继续使用。

## 4. 最新 6 个公开强 Agent

| 对手 | FC22 胜场 | FC24B 胜场 | FC24B 胜率 | cash up | cash down |
|---|---:|---:|---:|---:|---:|
| Boatlee V21 | 248 | 248 | 96.88% | 58 | 0 |
| Soil V26-H | 235 | 235 | 91.80% | 0 | 0 |
| Moon V92 | 245 | 245 | 95.70% | 65 | 0 |
| Kaito V39 | 253 | 253 | 98.83% | 0 | 0 |
| Steven E284 | 240 | 240 | 93.75% | 62 | 0 |
| Salem HarvestForge X | 246 | 246 | 96.09% | 58 | 0 |

合计 1,536 局：

- `loss -> win = 0`
- `win -> loss = 0`
- `cash up = 243`
- `cash down = 0`

原始凭证：

- `experiments/fusion_champion_v1/receipts/fc22_feed_value_vs_latest6_seed839001_n128x2_v1.json`
- `experiments/fusion_champion_v1/receipts/fc24b_value_guard_vs_latest6_seed839001_n128x2_v1.json`

## 5. 旧 28-Agent full roster

全部 28 个对手均达到至少 90%。最低和关键对手如下：

| 对手 | FC22 胜场 | FC24B 胜场 | FC24B 胜率 | 结论 |
|---|---:|---:|---:|---|
| Local PRT V6 | 230 | 231 | 90.23% | 跨过 90% 硬门 |
| Public Soil G04 | 235 | 235 | 91.80% | 持平 |
| Flex V59 | 235 | 235 | 91.80% | 持平 |
| Gold Proxy Rank12 | 235 | 236 | 92.19% | +1 胜 |
| x562 Latest | 239 | 239 | 93.36% | 持平 |
| Gold Proxy Rank07 | 241 | 241 | 94.14% | 持平 |
| Gold Proxy Rank14 | 245 | 245 | 95.70% | 持平 |
| RayK K320 Adaptive | 251 | 253 | 98.83% | +2 胜 |

全 28-Agent 合计 7,168 局：

- `loss -> win = 4`
- `win -> loss = 0`
- `cash up = 1,070`
- `cash down = 0`
- 单项最低胜率：FC22 `89.84%` -> FC24B `90.23%`

原始凭证：

- `experiments/fusion_champion_v1/receipts/fc22_vs_old_other24_seed594001_n128x2_v1.json`
- `experiments/fusion_champion_v1/receipts/fc22_vs_old_low4_seed594001_n128x2_v1.json`
- `experiments/fusion_champion_v1/receipts/fc24b_value_guard_vs_old_full_roster_seed594001_n128x2_v1.json`

## 6. PRT 临界败局证据

FC22 在 `seed=594122, candidate_seat=0` 的终局现金为 154,468，对手为 154,609，落后 141。

第 709 步时，一个单位站在成熟草莓上：

- 草莓产量：2
- 单位随身小麦：2
- 原动作：离开作物并返回仓库
- 剩余动作数恰好够完成：`HARVEST -> 8 次移动 -> DROP`

FC24B 执行补收后：

- 候选现金：154,776
- 对手现金：154,609
- 分差：+167

该局从失败变为获胜，且 2 倍价值门仍允许触发。

相关凭证：

- `experiments/fusion_champion_v1/receipts/fc22_prt_seed594122_seat0_step_trace_v1.json`
- `experiments/fusion_champion_v1/receipts/fc24b_value_guard_prt_seed594097_n32x2_q4_probe_v1.json`

## 7. 门控决定

通过：

- JAX 语义与本地竞技场无退化门；
- 34 个 Agent 单项胜率均达到至少 90%；
- 没有任何逐局现金下降；
- 没有任何原胜局被改成败局。

尚未完成：

- FC24B Python/官方执行版；
- Python 与 JAX 的动作、状态、终局现金逐步一致性；
- 官方 1.32.7 双座位复验；
- submission 打包检查。

因此当前只能将 FC24B 认定为“JAX 候选冠军”，不能直接提交。
