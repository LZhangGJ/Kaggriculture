# 全部严格验收 JAX Agent 两两 100 局结果

生成时间：2026-08-21T10:59:11.386932+00:00
结论：**SMOKE_PASS**

## 口径

- 参赛：28 个去重后的完整 JAX Agent。
- 对局：1 组 × 100 局 = 2 局；每局 719 步。
- 每组使用 50 个相同独立随机事件，并交换双方座位再跑 50 局。
- 只纳入官方 1.32.7 逐步一致性验收通过的完整策略；路线骨架、未严格对齐代理和 BC/PPO 实验模型不计入。
- 排名采用带轻微正则的 Bradley-Terry 分数；平局按双方各半胜计算。

## 综合排名

| 排名 | Agent | BT/Elo | 总胜率 | 胜-平-负 | 平均现金 | 平均分差 | 最差对手（得分率） |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | public_g02_rc5_c166 | +276.5 | 100.0% | 2-0-0 | 104366 | +21091 | public_g01_boatlee_v16 (100.0%) |
| 2 | public_g04_soil_rain | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 3 | public_g06_v25 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 4 | public_g07_c95 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 5 | public_g09_c68_thunder | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 6 | public_g08_v14 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 7 | public_g10_four_hire | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 8 | public_g11_v21 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 9 | public_g15_v18_closed_loop | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 10 | public_g12_v13_r3 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 11 | public_g13_bruce_route1 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 12 | public_g14_v19_control | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 13 | gold_proxy_rank07_junichiro_morita | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 14 | public_g16_tran_cashflow | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 15 | gold_proxy_rank12_ai_b2b67_saas | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 16 | gold_proxy_rank14_recursion | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 17 | deniz_v111_8c4s_latest | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 18 | gold_proxy_rank19_manu_nicholas_jacob | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 19 | local_prt_v6 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 20 | boatlee_v20_multi_route | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 21 | rayk_k320_adaptive_rank1 | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 22 | tetsutani_adaptive_premium_queue | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 23 | kaito_v27_midgame_reset | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 24 | flexonafft_v59_multi_route | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 25 | x562_latest | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 26 | kaito_v36_latest | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 27 | tetsutani_adaptive_latest | -0.0 | 0.0% | 0-0-0 | 0 | +0 | N/A (0.0%) |
| 28 | public_g01_boatlee_v16 | -276.5 | 0.0% | 0-0-2 | 83275 | -21091 | public_g02_rc5_c166 (0.0%) |

## 运行与完整性

- GPU：cuda:0
- 实测吞吐：77 transitions/s
- Arena 运行时间（不含首次编译）：18.7 秒
- 全部终局完成：True
- price LUT 越界：0
- hand cap / market loop cap：0 / 0

## 去重别名

- `flex_multi_route_latest` → `public_g06_v25`（source/action semantics identical）
- `boatlee_v20_latest` → `boatlee_v20_multi_route`（source/action semantics identical）
- `kunal_2026_v1_latest` → `boatlee_v20_multi_route`（source/action semantics identical）
- `rayk_rank_agent_latest` → `rayk_k320_adaptive_rank1`（source/action semantics identical）

## 解释边界

本结果是在固定官方事件口径下的本地 JAX 对战强度，不等同于当前 Public 榜分；金牌 proxy 是已验收的本地模仿 Agent，不冒充原选手源码。
