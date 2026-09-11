# R2强化：实际对战消融结果

每项均冻结一个二进制和配置，使用11个原程序实时对手，不按seed或对手身份选择策略。代码修正只在独立副本，原版R2发布目录未修改。
注意开发小筛不是最终验收；同seed改策略后，官方空地杂草抽样可改变后续商店序列，不能把逐局差额当纯确定性修复效果。
不确定性按seed整组重采样，保持同seed的双座位和不同对手相关性；只有8个seed的区间本身也很不稳定。

| 开发面板/配置 | 局数 | 原胜率 | 新胜率 | 救回/丢旧胜 | 分差变化 | 商店序列改变 |
|---|---:|---:|---:|---:|---:|---:|
| economic_screen8/capital08 | 176 | 90.34% | 51.14% | 0/69 | -10,070.7 | 176/176 |
| economic_screen8/cash_labor | 176 | 90.34% | 45.45% | 2/81 | -7,079.6 | 176/176 |
| economic_screen8/cash_time03 | 176 | 90.34% | 59.09% | 1/56 | -4,132.7 | 176/176 |
| economic_screen8/cash_time08 | 176 | 90.34% | 62.50% | 15/64 | -5,832.8 | 176/176 |
| economic_screen8/crop135 | 176 | 90.34% | 42.05% | 14/99 | -8,932.0 | 176/176 |
| economic_screen8/externality1 | 176 | 90.34% | 68.75% | 1/39 | +748.3 | 176/176 |
| economic_screen8/hours14 | 176 | 90.34% | 77.27% | 0/23 | -683.0 | 151/176 |
| economic_screen8/work2 | 176 | 90.34% | 67.61% | 14/54 | -1,680.6 | 162/176 |
| competitive_screen8/delivery_pressure | 176 | 67.05% | 67.05% | 0/0 | -20.1 | 0/176 |
| competitive_screen8/externality3 | 176 | 67.05% | 80.68% | 48/24 | +3,290.6 | 176/176 |
| competitive_screen8/externality4 | 176 | 67.05% | 64.20% | 33/38 | -2,826.1 | 176/176 |
| competitive_screen8/hours8 | 176 | 67.05% | 73.30% | 33/22 | -1,900.4 | 176/176 |
| competitive_screen8/margin_labor | 176 | 67.05% | 68.75% | 37/34 | -135.5 | 176/176 |
| competitive_screen8/rival_replant | 176 | 67.05% | 35.23% | 21/77 | -6,054.8 | 176/176 |
| competitive_screen8/supply125 | 176 | 67.05% | 52.27% | 18/44 | +1,055.4 | 176/176 |
| competitive_screen8/work6 | 176 | 67.05% | 73.30% | 48/37 | +14.0 | 175/176 |
| competitive3_100/externality3 | 2200 | 68.23% | 66.05% | 250/298 | -515.3 | 2033/2200 |

完整开发面板 competitive3_100/externality3：胜率变化的seed级95% bootstrap区间 [-9.23%, +4.82%]。这不是未见种子验收。

| shipment_screen8/half/shipment_half | 176 | 67.05% | 62.50% | 46/54 | -2,331.7 | 175/176 |
| shipment_screen8/one/shipment_one | 176 | 67.05% | 67.61% | 35/34 | -1,052.9 | 176/176 |
| timing_screen8/quarter/timing_quarter | 176 | 67.05% | 79.55% | 42/20 | +3,317.5 | 175/176 |
| timing_screen8/quarter_half/timing_quarter_half | 176 | 67.05% | 57.95% | 42/58 | -970.2 | 176/176 |
| timing_screen8/quarter_one/timing_quarter_one | 176 | 67.05% | 79.55% | 42/20 | +4,106.2 | 176/176 |
| timing_full100/quarter/timing_quarter | 2200 | 68.23% | 68.14% | 422/424 | +844.8 | 2187/2200 |

完整开发面板 timing_full100/quarter/timing_quarter：胜率变化的seed级95% bootstrap区间 [-9.18%, +9.09%]。这不是未见种子验收。

| timing_full100/quarter_one/timing_quarter_one | 2200 | 68.23% | 63.82% | 382/479 | -117.6 | 2194/2200 |

完整开发面板 timing_full100/quarter_one/timing_quarter_one：胜率变化的seed级95% bootstrap区间 [-13.50%, +4.36%]。这不是未见种子验收。

| startup_screen8/startup_1/startup_1 | 176 | 67.05% | 21.59% | 0/80 | -9,883.6 | 176/176 |
| startup_screen8/startup_2/startup_2 | 176 | 67.05% | 67.05% | 0/0 | +0.0 | 0/176 |
| market_screen8/market_1/market_1 | 176 | 67.05% | 73.30% | 35/24 | +2,739.1 | 169/176 |
| market_screen8/market_2/market_2 | 176 | 67.05% | 69.32% | 52/48 | +3,860.8 | 171/176 |
| market_screen100/market_1/market_1 | 2200 | 68.23% | 68.23% | 246/246 | -401.7 | 1863/2200 |

完整开发面板 market_screen100/market_1/market_1：胜率变化的seed级95% bootstrap区间 [-6.82%, +6.91%]。这不是未见种子验收。

| market_screen100/market_2/market_2 | 2200 | 68.23% | 65.77% | 438/492 | +49.5 | 2160/2200 |

完整开发面板 market_screen100/market_2/market_2：胜率变化的seed级95% bootstrap区间 [-12.36%, +7.41%]。这不是未见种子验收。

| clock_screen8/clock_coupled_market0/clock_coupled_market0 | 176 | 67.05% | 71.59% | 16/8 | +522.3 | 138/176 |
| clock_screen8/clock_mpc_market0/clock_mpc_market0 | 176 | 67.05% | 67.05% | 4/4 | +1,014.5 | 38/176 |
| clock_screen100/clock_coupled_market0/clock_coupled_market0 | 2200 | 68.23% | 73.45% | 269/154 | -149.4 | 1567/2200 |

完整开发面板 clock_screen100/clock_coupled_market0/clock_coupled_market0：胜率变化的seed级95% bootstrap区间 [-0.55%, +11.18%]。这不是未见种子验收。

| horizon_screen8/horizon2/horizon2 | 176 | 67.05% | 80.11% | 40/17 | +3,398.3 | 176/176 |
| horizon_screen8/horizon2_service_margin/horizon2_service_margin | 176 | 67.05% | 74.43% | 32/19 | +3,723.9 | 176/176 |
| horizon_screen8/horizon3/horizon3 | 176 | 67.05% | 61.36% | 32/42 | -2,145.5 | 176/176 |
| horizon_screen8/no_mpc/no_mpc | 176 | 67.05% | 70.45% | 48/42 | +374.4 | 176/176 |
| horizon_screen8/service_margin/service_margin | 176 | 67.05% | 74.43% | 42/29 | +1,659.9 | 174/176 |
| horizon_screen100/horizon2/horizon2 | 2200 | 68.23% | 67.50% | 290/306 | -1,109.1 | 2092/2200 |

完整开发面板 horizon_screen100/horizon2/horizon2：胜率变化的seed级95% bootstrap区间 [-7.91%, +6.50%]。这不是未见种子验收。

| local_sale_screen8/local_sale_1/local_sale_1 | 176 | 67.05% | 78.41% | 24/4 | +2,768.7 | 76/176 |
| local_sale_screen8/local_sale_2/local_sale_2 | 176 | 67.05% | 75.00% | 18/4 | +1,158.2 | 70/176 |
| local_sale_screen100/local_sale_1/local_sale_1 | 2200 | 68.23% | 75.73% | 247/82 | +452.5 | 1041/2200 |

完整开发面板 local_sale_screen100/local_sale_1/local_sale_1：胜率变化的seed级95% bootstrap区间 [+2.64%, +12.59%]。这不是未见种子验收。

| local_sale_screen100/local_sale_2/local_sale_2 | 2200 | 68.23% | 70.95% | 140/80 | +34.7 | 905/2200 |

完整开发面板 local_sale_screen100/local_sale_2/local_sale_2：胜率变化的seed级95% bootstrap区间 [-1.18%, +6.68%]。这不是未见种子验收。

| local_sale_combination_screen8/sale_no_mpc/sale_no_mpc | 176 | 67.05% | 80.68% | 54/30 | +756.7 | 176/176 |
| local_sale_combination_screen8/sale_service/sale_service | 176 | 67.05% | 75.00% | 42/28 | +2,403.3 | 173/176 |
| local_sale_combination_screen8/sale_wide/sale_wide | 176 | 67.05% | 62.50% | 24/32 | +606.8 | 174/176 |
| local_sale_combination_screen8/sale_wide_service/sale_wide_service | 176 | 67.05% | 70.45% | 48/42 | +1,592.6 | 176/176 |
| local_sale_combination_screen8/wide_original/wide_original | 176 | 67.05% | 67.05% | 36/36 | -986.8 | 175/176 |
| local_sale_combination_screen100/sale_no_mpc/sale_no_mpc | 2200 | 68.23% | 63.14% | 389/501 | -2,083.2 | 2183/2200 |

完整开发面板 local_sale_combination_screen100/sale_no_mpc/sale_no_mpc：胜率变化的seed级95% bootstrap区间 [-15.05%, +4.82%]。这不是未见种子验收。

| finite_crop_screen8/finite1_sale0_v2/finite1_sale0_v2 | 176 | 67.05% | 55.68% | 4/24 | -620.4 | 173/176 |
| finite_crop_screen8/finite1_sale1_v2/finite1_sale1_v2 | 176 | 67.05% | 82.95% | 38/10 | +2,669.0 | 173/176 |
| finite_crop_screen100/finite1_sale1_v2/finite1_sale1_v2 | 2200 | 68.23% | 74.82% | 324/179 | +1,294.0 | 1894/2200 |

完整开发面板 finite_crop_screen100/finite1_sale1_v2/finite1_sale1_v2：胜率变化的seed级95% bootstrap区间 [+0.05%, +13.18%]。这不是未见种子验收。

| crop_clock_screen8/cropclock1_sale0/cropclock1_sale0 | 176 | 67.05% | 74.43% | 34/21 | +2,723.1 | 162/176 |
| crop_clock_screen8/cropclock1_sale1/cropclock1_sale1 | 176 | 67.05% | 76.70% | 38/21 | +4,101.3 | 162/176 |
| crop_clock_screen100/cropclock1_sale1/cropclock1_sale1 | 2200 | 68.23% | 79.27% | 404/161 | +1,175.1 | 1990/2200 |

完整开发面板 crop_clock_screen100/cropclock1_sale1/cropclock1_sale1：胜率变化的seed级95% bootstrap区间 [+4.09%, +18.05%]。这不是未见种子验收。

| crop_chain_screen8/cropchain1/cropchain1 | 176 | 67.05% | 84.66% | 32/1 | +7,561.9 | 176/176 |
| crop_chain_screen100/cropchain1/cropchain1 | 2200 | 68.23% | 78.32% | 363/141 | +2,150.1 | 2056/2200 |

完整开发面板 crop_chain_screen100/cropchain1/cropchain1：胜率变化的seed级95% bootstrap区间 [+3.64%, +16.77%]。这不是未见种子验收。

| crop_portfolio_screen8/cropportfolio1/cropportfolio1 | 176 | 67.05% | 72.16% | 56/47 | +2,615.1 | 174/176 |
| fert_portfolio_screen8/fertportfolio1/fertportfolio1 | 176 | 67.05% | 72.16% | 38/29 | +6,578.8 | 162/176 |
| crop_continuation_screen8/rotation_and_repeat/rotation_and_repeat | 176 | 67.05% | 44.32% | 30/70 | -1,376.8 | 176/176 |
| crop_continuation_screen8/rotation_only/rotation_only | 176 | 67.05% | 68.75% | 32/29 | +3,704.1 | 176/176 |
| startup_supply_screen8/startupsupply1/startupsupply1 | 176 | 67.05% | 83.52% | 34/5 | +5,029.4 | 176/176 |
| startup_supply_screen8/startupsupply2/startupsupply2 | 176 | 67.05% | 89.77% | 40/0 | +7,366.9 | 162/176 |

## 尚未达到最终目标

最终要求仍是冻结版本在100个未见seed×双座位×11对手中等权平均≥90%；未使用2609130000—2609130099。
新增R2P2中途交付触发器在首轮176局胜负不变，只有小额现金变化，不能宣布上分修复成功。
每场轨迹、配置、原程序哈希与官方核对证据均保留在对应子目录。
