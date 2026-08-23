# FC12G 批量雇工前杂草补偿保护：配对验收

生成时间：2026-08-22T02:40:00.119221+00:00

## 结论

该修复通过关键对手面板的配对安全门，可进入更大全池复验；但仍未达到最终逐对手 90% 总目标。

## 验收口径

- 官方规则版本：1.32.7。
- 每个对手：128 局；相同随机种子，双方换座。
- 原版与候选逐局按 `(opponent, seed, seat)` 配对，比较胜负与终局现金差。
- 该阶段只验收单一修复，不宣称已达到全池逐对手 90%。

## 逐对手结果

| 对手 | 原版胜率 | FC12G胜率 | 胜场变化 | 平均分差变化 | 救回败局 | 伤害原胜局 | 变化局数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| public_g01_boatlee_v16 | 99.22% | 99.22% | +0 | +0.0 | 0 | 0 | 0 |
| public_g02_rc5_c166 | 89.84% | 89.84% | +0 | +0.0 | 0 | 0 | 0 |
| public_g04_soil_rain | 88.28% | 88.28% | +0 | +0.0 | 0 | 0 | 0 |
| public_g06_v25 | 100.00% | 100.00% | +0 | +0.0 | 0 | 0 | 0 |
| public_g07_c95 | 98.44% | 98.44% | +0 | +0.0 | 0 | 0 | 0 |
| public_g08_v14 | 100.00% | 100.00% | +0 | +131.1 | 0 | 0 | 7 |
| public_g09_c68_thunder | 100.00% | 100.00% | +0 | +0.0 | 0 | 0 | 0 |
| public_g10_four_hire | 100.00% | 100.00% | +0 | +0.0 | 0 | 0 | 0 |
| public_g11_v21 | 98.44% | 98.44% | +0 | +62.6 | 0 | 0 | 5 |
| public_g12_v13_r3 | 98.44% | 98.44% | +0 | +92.7 | 0 | 0 | 6 |
| public_g13_bruce_route1 | 100.00% | 100.00% | +0 | +105.2 | 0 | 0 | 6 |
| public_g14_v19_control | 100.00% | 100.00% | +0 | +103.6 | 0 | 0 | 6 |
| public_g15_v18_closed_loop | 100.00% | 100.00% | +0 | +104.5 | 0 | 0 | 6 |
| public_g16_tran_cashflow | 100.00% | 100.00% | +0 | +100.6 | 0 | 0 | 6 |
| gold_proxy_rank07_junichiro_morita | 92.97% | 93.75% | +1 | +151.6 | 1 | 0 | 7 |
| gold_proxy_rank12_ai_b2b67_saas | 86.72% | 87.50% | +1 | +155.3 | 1 | 0 | 6 |
| gold_proxy_rank14_recursion | 92.97% | 92.97% | +0 | +0.0 | 0 | 0 | 0 |
| gold_proxy_rank19_manu_nicholas_jacob | 94.53% | 96.09% | +2 | +266.9 | 2 | 0 | 9 |
| local_prt_v6 | 87.50% | 89.06% | +2 | +324.8 | 2 | 0 | 9 |
| boatlee_v20_multi_route | 94.53% | 96.09% | +2 | +155.5 | 2 | 0 | 5 |
| rayk_k320_adaptive_rank1 | 85.94% | 88.28% | +3 | +149.5 | 3 | 0 | 5 |
| tetsutani_adaptive_premium_queue | 94.53% | 96.09% | +2 | +155.2 | 2 | 0 | 5 |
| kaito_v27_midgame_reset | 99.22% | 99.22% | +0 | +0.0 | 0 | 0 | 0 |
| flexonafft_v59_multi_route | 88.28% | 88.28% | +0 | +0.0 | 0 | 0 | 0 |
| deniz_v111_8c4s_latest | 89.84% | 89.84% | +0 | +0.0 | 0 | 0 | 0 |
| kaito_v36_latest | 97.66% | 97.66% | +0 | +0.0 | 0 | 0 | 0 |
| x562_latest | 93.75% | 95.31% | +2 | +223.4 | 2 | 0 | 9 |
| tetsutani_adaptive_latest | 94.53% | 96.09% | +2 | +155.5 | 2 | 0 | 5 |

## 总计

- 配对对局：3584。
- 胜场：3412 → 3429（+17）。
- 救回原败局：17；伤害原胜局：0。
- 现金差改善/退化/不变：102 / 0 / 3482。
- 候选最低逐对手胜率：87.50%。

## 证据文件

- FC12G：`experiments\fusion_champion_v1\receipts\fc12j_fc2b_mass_hire_guard_full_roster_seed594001_n64x2_v1.json`
- FC2B 原版：`experiments\fusion_champion_v1\receipts\fc12p_fc2b_source_full_roster_seed594001_n64x2_v1.json`
- 机器可读配对结果：`experiments\fusion_champion_v1\receipts\fc12q_fc12g_vs_fc2b_paired_full_roster_seed594001_n64x2_v1.json`
