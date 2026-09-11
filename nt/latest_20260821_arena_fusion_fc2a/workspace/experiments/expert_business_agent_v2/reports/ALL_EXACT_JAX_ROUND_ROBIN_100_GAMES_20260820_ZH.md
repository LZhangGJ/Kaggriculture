# 全部严格验收 JAX Agent 两两 100 局结果

生成时间：2026-08-20T16:43:25.693522+00:00
结论：**PASS**

## 验收口径

- 参赛：28 个去重后的完整 JAX Agent。
- 对局：378 组 × 100 局 = 37,800 局；每局 719 步。
- 每组使用相同的 50 个独立事件种子，并以 50/50 交换座位。
- 八个 GPU 分片并行运行；最终排名只在完整 378 组无缺失、无重复后计算。
- 仅纳入官方 1.32.7 逐步一致性验收通过的完整 JAX 策略。
- Bradley-Terry 分数只表示本次冻结本地对手池内的相对强度。

## 综合排名

| 排名 | Agent | BT/Elo | 得分率 | 胜-平-负 | 平均现金 | 平均分差 | 最差对手（得分率） |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | rayk_k320_adaptive_rank1 | +643.9 | 86.7% | 2342-0-358 | 94565 | +9356 | gold_proxy_rank14_recursion (24.0%) |
| 2 | gold_proxy_rank14_recursion | +527.6 | 78.4% | 2118-0-582 | 94952 | +11332 | x562_latest (28.0%) |
| 3 | tetsutani_adaptive_premium_queue | +521.1 | 77.9% | 2096-16-588 | 93684 | +8176 | rayk_k320_adaptive_rank1 (6.0%) |
| 4 | gold_proxy_rank19_manu_nicholas_jacob | +510.5 | 77.1% | 2081-0-619 | 91709 | +5271 | rayk_k320_adaptive_rank1 (12.0%) |
| 5 | tetsutani_adaptive_latest | +477.7 | 74.4% | 1972-73-655 | 93652 | +8112 | rayk_k320_adaptive_rank1 (5.0%) |
| 6 | boatlee_v20_multi_route | +477.7 | 74.4% | 1972-73-655 | 93652 | +8112 | rayk_k320_adaptive_rank1 (5.0%) |
| 7 | gold_proxy_rank07_junichiro_morita | +474.4 | 74.1% | 2001-0-699 | 94366 | +6909 | rayk_k320_adaptive_rank1 (12.0%) |
| 8 | x562_latest | +469.1 | 73.7% | 1989-0-711 | 95020 | +8007 | rayk_k320_adaptive_rank1 (14.0%) |
| 9 | flexonafft_v59_multi_route | +449.8 | 72.0% | 1943-4-753 | 93757 | +9441 | gold_proxy_rank07_junichiro_morita (10.0%) |
| 10 | kaito_v36_latest | +447.1 | 71.8% | 1939-0-761 | 95294 | +10737 | gold_proxy_rank19_manu_nicholas_jacob (4.0%) |
| 11 | public_g04_soil_rain | +424.1 | 69.9% | 1884-4-812 | 93757 | +9443 | gold_proxy_rank07_junichiro_morita (10.0%) |
| 12 | local_prt_v6 | +402.8 | 68.0% | 1836-2-862 | 94564 | +5237 | rayk_k320_adaptive_rank1 (20.0%) |
| 13 | gold_proxy_rank12_ai_b2b67_saas | +401.9 | 68.0% | 1835-0-865 | 93636 | +7113 | boatlee_v20_multi_route (23.0%) |
| 14 | public_g02_rc5_c166 | +305.7 | 59.9% | 1617-0-1083 | 93336 | +7335 | gold_proxy_rank07_junichiro_morita (2.0%) |
| 15 | deniz_v111_8c4s_latest | +229.9 | 54.0% | 1457-0-1243 | 93245 | +6884 | public_g04_soil_rain (0.0%) |
| 16 | public_g07_c95 | +165.8 | 49.4% | 1333-0-1367 | 89241 | +4866 | public_g04_soil_rain (0.0%) |
| 17 | public_g01_boatlee_v16 | +154.3 | 48.6% | 1312-0-1388 | 90946 | +1994 | public_g02_rc5_c166 (0.0%) |
| 18 | kaito_v27_midgame_reset | +42.9 | 41.7% | 1126-0-1574 | 89591 | +510 | public_g02_rc5_c166 (0.0%) |
| 19 | public_g10_four_hire | -99.4 | 34.4% | 928-0-1772 | 83390 | -5142 | public_g01_boatlee_v16 (0.0%) |
| 20 | public_g06_v25 | -227.1 | 28.8% | 778-0-1922 | 81384 | -9779 | public_g01_boatlee_v16 (0.0%) |
| 21 | public_g09_c68_thunder | -233.6 | 28.6% | 771-0-1929 | 81830 | -9258 | public_g01_boatlee_v16 (0.0%) |
| 22 | public_g08_v14 | -267.7 | 27.2% | 735-0-1965 | 83239 | -9153 | public_g01_boatlee_v16 (0.0%) |
| 23 | public_g12_v13_r3 | -468.5 | 20.5% | 554-1-2145 | 76646 | -13318 | public_g01_boatlee_v16 (0.0%) |
| 24 | public_g11_v21 | -606.8 | 17.0% | 460-0-2240 | 77743 | -14125 | public_g01_boatlee_v16 (0.0%) |
| 25 | public_g14_v19_control | -950.4 | 10.8% | 292-0-2408 | 73903 | -16252 | public_g01_boatlee_v16 (0.0%) |
| 26 | public_g13_bruce_route1 | -1144.4 | 7.8% | 211-0-2489 | 73640 | -16675 | public_g01_boatlee_v16 (0.0%) |
| 27 | public_g15_v18_closed_loop | -1400.7 | 4.2% | 114-0-2586 | 73348 | -17395 | public_g01_boatlee_v16 (0.0%) |
| 28 | public_g16_tran_cashflow | -1727.8 | 0.6% | 17-1-2682 | 73333 | -17735 | public_g01_boatlee_v16 (0.0%) |

## 运行与完整性

- GPU：cuda:0
- 并行墙钟时间：6,220.3 秒
- 有效总吞吐：4,369 transitions/s
- 全部终局完成：True
- hand cap / market loop cap / price LUT 越界：0 / 0 / 0
- 运行中外部观测峰值：GPU 97%，显存 7808 MiB。

## 去重别名

- `flex_multi_route_latest` → `public_g06_v25`（source/action semantics identical）
- `boatlee_v20_latest` → `boatlee_v20_multi_route`（source/action semantics identical）
- `kunal_2026_v1_latest` → `boatlee_v20_multi_route`（source/action semantics identical）
- `rayk_rank_agent_latest` → `rayk_k320_adaptive_rank1`（source/action semantics identical）

## 近重复审计

- `boatlee_v20_multi_route` 与 `tetsutani_adaptive_latest` 在本次冻结事件库的汇总战绩相同，但不按别名合并。
- 两者官方验收源文件 SHA 不同，JAX 分别使用 `MODE_BOATLEE` 与 `MODE_TETSUTANI_LATEST`；直接对战为 18 胜 / 64 平 / 18 负（按 Boatlee 视角）。
- 本轮只说明聚合表现相同，不把它扩大解释成全状态动作完全一致。

## 排除范围与解释边界

- 46 个路线骨架、未严格对齐代理、实验性 BC/PPO 模型未纳入。
- `gold_proxy_*` 是通过本地逐步一致性验收的模仿 Agent，不是原金牌选手源码。
- 本结果是固定事件库下的本地 JAX 对手池排名，不等于当前 Public 榜分。
