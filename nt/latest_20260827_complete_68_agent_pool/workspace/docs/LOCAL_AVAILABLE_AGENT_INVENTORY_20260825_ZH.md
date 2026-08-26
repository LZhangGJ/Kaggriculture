# Kaggriculture 本地可用 Agent 总清单（2026-08-25）

## 1. 使用边界

工作区根目录：

`E:\ai_coding\kaggle\kaggriculture`

当前标准验证池共有 **68 个可调用配置**：

|类别|数量|含义|
|---|---:|---|
|冻结基线|1|FC24B|
|旧 Rank40 Replay 重建|35|2026-08-23 前40快照中保留的公开状态路线 Agent|
|历史公开精确/近精确 JAX|28|公开 Notebook、金牌代理和本地历史强拳|
|最新公开原生 JAX|1|Kaito V48|
|2026-08-25 当前 Top20 新增重建|3|tetsuya、Hasegawa new77、Crop Dusta current|
|合计|68|跨运行时最终行为去重前的临时总数|

权威机器清单：

`experiments/front40_fusion_v1/configs/complete_validation_pool_v1.json`

已检查该清单引用的97个配置、路线库、路由器、报告和回执，当前缺失文件为0。

重要限制：

- 68表示可运行配置，不表示68种完全独立策略；部分公开方案属于同一骨架的变体。
- `trace_*` 是根据 Replay 重建的公开状态 Agent，不等于原作者闭源 Agent。
- Replay Router 只能使用当前公开盘面、商店和自身状态；禁止按玩家身份、ReplayId、seed或未来事件路由。
- 今天新下载的当前 Rank21–40 只有 Replay，尚未迁入 JAX，因此不计入68个 Agent。

## 2. 建议另一线程优先使用的硬门池

|Agent|用途|当前已知特点|
|---|---|---|
|`fc24b_frozen_best`|冻结基准|本地历史综合基线|
|`kaito_v48`|快速牛羊主流硬门|对 FC24B 83.69%|
|`rank04_arman_v45`|强 Trace 骨架硬门|两阶段商店路由|
|`rank13_burntpotato_v12`|独立强路线硬门|两阶段路由|
|`top20_20260825_tetsuya_current_firstshop_v1`|动态动物配比|对 FC24B 68.85%，但怕 Kaito V48|
|`top20_20260825_hasegawa_new77_firstshop_v1`|低错误恢复|对 FC24B 59.91%，对 x562 79.69%|
|`top20_20260825_crop_dusta_current_unrestricted_v1`|重羊+混合作物覆盖|对 FC24B 36.23%，用于发现产业短板|

任何“综合最强”候选至少应先通过以上7个对手的全新事件、双座位、完整719步检查，再扩大到全部68成员。

## 3. 冻结基线

### FC24B

- 名称：`fc24b_frozen_best`
- 类型：原生 JAX。
- 配置：`experiments/fusion_champion_v1/configs/fc24b_frozen_best_v1.json`
- GPU 策略：`experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py`
- 官方 Python 提交包：`submission/55708153_fc24b_34agent_90pct_strict_jax_parity/`
- 状态：可作为 Arena 基准；已有官方/JAX parity。

## 4. 旧 Rank40 Replay 重建 Agent（35个）

这一组的 Rank 是 **2026-08-23 快照排名**，不是今天榜单排名。公共目录前缀：

`experiments/front40_fusion_v1/artifacts/`

|旧Rank|Agent|类型|主路由文件|
|---:|---|---|---|
|1|`rank01_hasegawa_v11`|两商店 Trace|`hasegawa_fc24b_ablation_blend_v10.json` + `hasegawa_fc24b_second_shop_gain0p30_train384_v8.json`|
|2|`rank02_crop_dusta_v29`|首店 Trace|`rank02_crop_dusta_robust_unrestricted_stable_win_all3panels_v29.json`|
|3|`rank03_subramanya_v18`|首店 Trace|`rank03_subramanya_wilson_unrestricted_map_v18.json`|
|4|`rank04_arman_v45`|两商店 Trace|`rank04_arman_stable_win_unrestricted_map_dev640_v8.json` + `rank04_arman_secondshop_exact5_dev640_v38.json`|
|5|`rank05_mimi_v29`|两商店 Trace|`rank05_mimi_stable_win_map_dev640_v6.json` + `rank05_mimi_secondshop_strong3_dev640_v15.json`|
|6|`rank06_kanta_v30`|两商店 Trace|`rank06_kanta_stable_win_map_dev640_v6.json` + `rank06_kanta_secondshop_acd_dev640_v18.json`|
|7|`rank07_active_v32`|两商店 Trace|`rank07_active_stable_win_map_dev640_v6.json` + `rank07_active_secondshop_risk7_canonical_v22.json`|
|8|`rank08_kobe_v46`|两商店 Trace|`rank08_kobe_all89_stable_win_map_dev640_v16.json` + `rank08_kobe_secondshop_safe8_fs_dev1152_v28.json`|
|9|`rank09_kawashigi_v115`|三商店树|`rank09_kawashigi_v115_pizza_fixed_route100.json`|
|10|`rank10_seb_v11`|首店 Trace|`rank10_seb_threepanel_minimax_firstshop_v4.json`|
|11|`rank11_peikopon_v7`|首店 Trace|`rank11_peikopon_threepanel_minimax_firstshop_v4.json`|
|13|`rank13_burntpotato_v12`|两商店 Trace|`rank13_burntpotato_two_panel_pooled_firstshop_v3.json` + `rank13_burntpotato_secondshop_exactprefix_cross_v6.json`|
|14|`rank14_kaan_diniz_validation_v1`|首店 Trace|`rank14_kaan_diniz_validation_firstshop_v1.json`|
|15|`rank15_recursion_v7`|固定/首店 Trace|`rank15_recursion_fixed_route80_v4.json`|
|17|`rank17_sarthak_v15`|两商店 Trace|`rank17_sarthak_three_panel_pooled_firstshop_v7.json` + `rank17_sarthak_secondshop_exactprefix_cross_v10.json`|
|18|`rank18_mandgeee_validation_v1`|首店 Trace|`rank18_mandgeee_validation_firstshop_v1.json`|
|19|`rank19_xiaowenhao404_validation_v1`|首店 Trace|`rank19_xiaowenhao404_validation_firstshop_v1.json`|
|20|`rank20_u_validation_v1`|首店 Trace|`rank20_u_validation_firstshop_v1.json`|
|21|`rank21_saikushal185_v9`|两商店 Trace|`rank21_saikushal185_minimax_firstshop_v4.json` + `rank21_saikushal185_secondshop_pooled_v9.json`|
|22|`rank22_victor_tufa_validation_v1`|首店 Trace|`rank22_victor_tufa_validation_firstshop_v1.json`|
|23|`rank23_xw_v9`|三商店树|`rank23_xw_thirdshop_three_panel_minimax_v9.json`|
|24|`rank24_stackknight_v4`|首店 Trace|`rank24_stackknight_two_panel_minimax_firstshop_v4.json`|
|26|`rank26_efe_validation_v1`|首店 Trace|`rank26_efe_validation_firstshop_v1.json`|
|27|`rank27_atakan_validation_v1`|首店 Trace|`rank27_atakan_validation_firstshop_v1.json`|
|28|`rank28_shiiin9_v4`|首店 Trace|`rank28_shiiin9_two_panel_minimax_firstshop_v4.json`|
|29|`rank29_lucaskna_v5`|首店 Trace|`rank29_lucaskna_two_panel_minimax_firstshop_v5.json`|
|31|`rank31_bplyc15_v4`|两商店 Trace|`rank31_bplyc15_single_panel_firstshop_v3.json` + `rank31_bplyc15_secondshop_single_panel_v4.json`|
|33|`rank33_unknownrobi_validation_v1`|首店 Trace|`rank33_unknownrobi_validation_firstshop_v1.json`|
|34|`rank34_arda_ceylan_v5`|两商店 Trace|`rank34_arda_ceylan_two_panel_minimax_firstshop_v4.json` + `rank34_arda_ceylan_secondshop_two_panel_v5.json`|
|35|`rank35_webmaking_validation_v1`|首店 Trace|`rank35_webmaking_validation_firstshop_v1.json`|
|36|`rank36_james_holland_v13`|首店 Trace|`rank36_james_holland_three_panel_robust_firstshop_v13.json`|
|37|`rank37_kenjo1209_v17`|两商店 Trace|`rank37_kenjo1209_two_panel_unrestricted_wilson_firstshop_v14.json` + `rank37_kenjo1209_secondshop_cross_source_two_panel_v17.json`|
|38|`rank38_ebisu_ya_v4`|首店 Trace|`rank38_ebisu_ya_single_panel_unrestricted_stable_win_firstshop_v4.json`|
|39|`rank39_lelelove1225_v14`|两商店 Trace|`rank39_lelelove1225_two_panel_unrestricted_stable_win_firstshop_v11.json` + `rank39_lelelove1225_secondshop_cross_source_two_panel_v14.json`|
|40|`rank40_ar_sekkat_v14`|两商店 Trace|`rank40_ar_sekkat_two_panel_unrestricted_stable_win_firstshop_v11.json` + `rank40_ar_sekkat_secondshop_cross_source_two_panel_v14.json`|

旧 Rank40 中没有作为独立 Agent 保留的项目：

- Rank12：最佳可部署胜率对 FC24B 仅1.95%。
- Rank16：筛选路线对 FC24B 全部为0。
- Rank25：固定/首店路由对 FC24B 为0。
- Rank32：最佳首店路由对 FC24B 仅3.12%。
- Rank30：就是 FC24B，已由单独冻结基线代替。

## 5. 历史公开精确/近精确 JAX Agent（28个）

统一入口：

`experiments/expert_business_agent_v2/tools/run_all_exact_jax_round_robin.py`

主要运行时数据：

- `experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz`
- `experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz`
- `experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz`

|Roster ID|Agent|来源/用途|
|---:|---|---|
|0|`public_g01_boatlee_v16`|Boatlee V16|
|1|`public_g02_rc5_c166`|RC5 C166|
|2|`public_g04_soil_rain`|Soil Rain / G04|
|3|`public_g06_v25`|公开 V25|
|4|`public_g07_c95`|C95|
|5|`public_g08_v14`|公开 V14|
|6|`public_g09_c68_thunder`|C68 Thunder|
|7|`public_g10_four_hire`|4-HIRE|
|8|`public_g11_v21`|公开 V21|
|9|`public_g12_v13_r3`|V13 R3|
|10|`public_g13_bruce_route1`|Bruce Route1|
|11|`public_g14_v19_control`|V19 Control|
|12|`public_g15_v18_closed_loop`|V18 Closed Loop|
|13|`public_g16_tran_cashflow`|Tran Cashflow|
|14|`gold_proxy_rank07_junichiro_morita`|旧金牌代理 Rank7|
|15|`gold_proxy_rank12_ai_b2b67_saas`|旧金牌代理 Rank12|
|16|`gold_proxy_rank14_recursion`|旧金牌代理 Rank14|
|17|`gold_proxy_rank19_manu_nicholas_jacob`|旧金牌代理 Rank19|
|18|`local_prt_v6`|本地 PRT V6|
|19|`boatlee_v20_multi_route`|Boatlee V20 多路线|
|20|`rayk_k320_adaptive_rank1`|K320 自适应|
|21|`tetsutani_adaptive_premium_queue`|tetsutani premium queue|
|22|`kaito_v27_midgame_reset`|Kaito V27|
|23|`flexonafft_v59_multi_route`|Flexonafft V59|
|24|`deniz_v111_8c4s_latest`|Deniz V111|
|25|`kaito_v36_latest`|Kaito V36|
|26|`x562_latest`|x562|
|27|`tetsutani_adaptive_latest`|tetsutani 最新本地迁移|

这里的“精确/近精确”是指已迁入固定 JAX 运行时；`gold_proxy_*` 明确只是基于 Replay 的代理，不应写成原作者 Agent 精确复刻。

## 6. 最新公开原生 JAX Agent

### Kaito V48

- 名称：`kaito_v48`
- 类型：原生公开方案 JAX 迁移。
- 路线库：`experiments/public_recent_20260825/artifacts/recent_v48_route_bank_v1.npz`
- 策略源码：`experiments/strategic_v5/src/strategic_v5/recent_public_20260825_gpu.py`
- 来源报告：`experiments/public_recent_20260825/reports/RECENTLY_RUN_PUBLIC_SCAN_20260825_ZH.md`
- 对 FC24B：857/1024，83.69%，平均分差约 +6,044。
- 状态：当前必须保留的快速牛羊主流硬门。

## 7. 2026-08-25 当前 Top20 新增重建（3个）

### tetsuya current

- 名称：`top20_20260825_tetsuya_current_firstshop_v1`
- 来源 submission：55734995。
- JAX Bank：`experiments/front40_fusion_v1/artifacts/tetsuya_new74_subset_v1.npz`
- 路由：`top20_20260825_tetsuya_new74_same_shop_map_seed1502001_v1.json`
- 对 FC24B 两套独立库合并：68.85%。
- 优点：动态牛/羊/鹅比例，对 Boatlee V20、K320、tetsutani 约71%–73%。
- 短板：对 Kaito V48 18.75%，对 Rank4 37.5%，对 Rank13 24.22%。
- 定位：强制对手和能力供体，不是通用提交骨架。

### Hasegawa new77

- 名称：`top20_20260825_hasegawa_new77_firstshop_v1`
- 来源 submission：55614463。
- JAX Bank：`experiments/front40_fusion_v1/artifacts/hasegawa_new77_subset_v1.npz`
- 路由：`top20_20260825_hasegawa_new77_same_shop_map_seed1502001_v1.json`
- 对 FC24B 两套独立库合并：59.91%。
- 优点：无效动作少、恢复稳定，对 x562 79.69%。
- 短板：对 Kaito V48 25%，对 Rank4 31.25%。
- 定位：恢复逻辑供体和强制对手。

### Crop Dusta current

- 名称：`top20_20260825_crop_dusta_current_unrestricted_v1`
- 来源 submission：55714246。
- JAX Bank：`experiments/front40_fusion_v1/artifacts/top20_20260825_crop_dusta_current_trace_bank_v1.npz`
- 路由：`top20_20260825_crop_dusta_unrestricted_map_seed1500001_v1.json`
- 对 FC24B：36.23%。
- 优点：完全不同的重羊+混合作物结构。
- 短板：每步最近 Replay 动态路由仅13.28%，无法复刻原 Top1 Agent 的持久计划。
- 定位：产业结构风险覆盖，不是提交候选。

详细报告：

`experiments/front40_fusion_v1/reports/TOP20_20260825_POTENTIAL_ROUTE_AUDIT_V1_ZH.md`

## 8. 今天已下载、但尚未成为 Agent 的数据

当前 Rank21–40 的20个最高分 submission Replay 已完整下载：

`replay/rank21_40_strongest_submissions_20260825_v1/`

- 1,808个唯一 Replay。
- 规则版本全部1.32.7。
- 1,808/1,808全量解析和SHA256通过。
- 这些玩家还没有编译成 JAX Agent，不能直接加入 Arena。

验收报告：

`replay/rank21_40_strongest_submissions_20260825_v1/DOWNLOAD_ACCEPTANCE_20260825_ZH.md`

## 9. 常用运行入口

### 历史28 Agent 内部 Arena

`experiments/expert_business_agent_v2/tools/run_all_exact_jax_round_robin.py`

### Trace Agent 对历史 Agent 组

`experiments/front40_fusion_v1/tools/run_trace_backbones_vs_old28_batched.py`

### Trace Agent 对 Trace Agent

`experiments/front40_fusion_v1/tools/run_trace_map_vs_trace_map.py`

### Trace Agent 对精确 Kaito V48

`experiments/front40_fusion_v1/tools/run_trace_map_vs_kaito_v48.py`

### 当前强制池机器清单

`experiments/front40_fusion_v1/configs/complete_validation_pool_v1.json`

## 10. 推荐给下一线程的读取顺序

1. 先读本文件，理解 Agent 边界和命名。
2. 再读 `complete_validation_pool_v1.json` 获取可执行路径。
3. 新候选先跑7个硬门池，不通过就不要直接扩大到68成员。
4. 需要分析今天的 Rank21–40 时，从 Replay 入手；在完成来源指纹、公开状态路由和硬错误验收前，不要把它们写成“已有 Agent”。
