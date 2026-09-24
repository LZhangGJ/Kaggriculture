# 第二轮融合开发与选择

V1 正式验收：内战 1243/1300（95.62%），外战 861/1000（86.10%），未达双 90%。完整失败证据保存在 `versions/v1` 与 `runs/holdout_v1`。

下表是同一批 16 个开发种子、双座位，2 内部难对手 + 3 外部代表的筛选结果。它刻意突出薄弱对手，不能把该表外战胜率当成公开十对手平均。所有参数选择都发生在 V2 新验收种子开跑前。

| 方案 | 内部胜场 / 64 | 外部胜场 / 96 | 平均现金差 |
|---|---:|---:|---:|
| v2_animal06 | 53 | 62 | +4,513.6 |
| v2_animal06_pressure | 51 | 60 | +4,560.5 |
| v2_animal055 | 55 | 58 | +4,875.8 |
| v2_animal08 | 50 | 56 | +4,703.6 |
| v2_animal08_pressure | 50 | 56 | +4,553.1 |
| v2_animal065 | 50 | 55 | +4,248.7 |
| v2_animal05 | 52 | 53 | +4,153.7 |
| cf_liq_h12_nointraday | 52 | 50 | +3,523.0 |
| v2_q8 | 52 | 50 | +3,523.0 |
| v2_animal07 | 41 | 50 | +3,290.3 |
| cf_liq_nointraday | 48 | 49 | +3,020.3 |
| v2_pressure | 51 | 48 | +3,347.0 |
| v2_h10 | 50 | 48 | +2,736.4 |
| v2_q15 | 52 | 32 | +704.7 |
| v2_q5 | 52 | 32 | -183.2 |
| v2_animal04 | 10 | 10 | -17,061.4 |

重复对局检查：V1 完整开发与 V2 控制组之间有 160 局重叠，双方现金、胜负和步数 160/160 完全一致。开发与验收之间的落差是样本和策略泛化问题，未观察到这组重放的不确定性。

机制结论：开局买卖量 8 与 10 在筛选池终局完全相同，5 和 15 明显变差；增加运输压力触发未带来净胜场收益。调整新动物候选的排序权重有效，但 0.4 过度削弱动物竞争力，开局样例转成 20 甜瓜种子并导致整体崩溃。保持原生动态规划，用适中权重调整后续资产分配更可靠。

## 完整开发面板

16 开发种子 × 双座 × 23 对手。仍不是独立验收。

| 候选 | 内战 | 外战 |
|---|---:|---:|
| v2_animal06 | 392/416 = 94.23% | 284/320 = 88.75% |

选入 V2 验收的候选是 `v2_animal06`。Best balanced candidate in the 16-seed five-opponent rules ablations: internal 53/64 and external 62/96. Full development regression: internal 392/416 (94.23%), external 284/320 (88.75%); target not yet met. Animal ranking weight 0.6 improves the two hard public opponents while retaining the Cashflow core, opening overlay, intraday-off and 12-hand cap. Freeze before a fresh independent test; do not select using its outcomes.

全新 50 个种子记录在 `SEEDS_V2.json`，候选源文件哈希记录在 `FINAL_FREEZE_V2.json`。V1 用过的种子被排除。正式结论仅以新验收回执为准。

## 最终没有晋级

V2 独立验收为内战 1242/1300（95.54%），外战 832/1000（83.20%）。同一批种子外战，V1 为 863/1000，V2 为 832/1000。保留 V1 作为基线；开发改善没有提供足够的晋级依据。最终完整说明见 FINAL_REPORT_ZH.md。
