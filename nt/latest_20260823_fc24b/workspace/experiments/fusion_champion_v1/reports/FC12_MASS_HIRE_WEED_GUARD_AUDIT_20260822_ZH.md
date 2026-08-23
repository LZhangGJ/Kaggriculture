# FC12G 批量雇工前杂草补偿保护：配对验收

生成时间：2026-08-22T01:25:18.086688+00:00

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
| gold_proxy_rank14_recursion | 89.06% | 89.06% | +0 | +0.0 | 0 | 0 | 0 |
| public_g04_soil_rain | 85.94% | 85.94% | +0 | +0.0 | 0 | 0 | 0 |
| flexonafft_v59_multi_route | 85.94% | 85.94% | +0 | +0.0 | 0 | 0 | 0 |
| public_g01_boatlee_v16 | 97.66% | 97.66% | +0 | +0.0 | 0 | 0 | 0 |
| gold_proxy_rank12_ai_b2b67_saas | 81.25% | 82.81% | +2 | +125.2 | 2 | 0 | 9 |
| public_g02_rc5_c166 | 89.84% | 89.84% | +0 | +0.0 | 0 | 0 | 0 |
| local_prt_v6 | 90.62% | 92.19% | +2 | +233.7 | 2 | 0 | 9 |
| deniz_v111_8c4s_latest | 89.84% | 89.84% | +0 | +0.0 | 0 | 0 | 0 |
| kaito_v36_latest | 96.88% | 96.88% | +0 | +0.0 | 0 | 0 | 0 |
| x562_latest | 92.19% | 93.75% | +2 | +186.7 | 2 | 0 | 9 |
| gold_proxy_rank07_junichiro_morita | 92.97% | 94.53% | +2 | +23.3 | 2 | 0 | 10 |
| gold_proxy_rank19_manu_nicholas_jacob | 94.53% | 95.31% | +1 | +161.9 | 1 | 0 | 9 |
| kaito_v27_midgame_reset | 97.66% | 97.66% | +0 | +0.0 | 0 | 0 | 0 |
| rayk_k320_adaptive_rank1 | 93.75% | 97.66% | +5 | +188.1 | 5 | 0 | 9 |
| boatlee_v20_multi_route | 95.31% | 96.88% | +2 | +193.7 | 2 | 0 | 9 |

## 总计

- 配对对局：1920。
- 胜场：1758 → 1774（+16）。
- 救回原败局：16；伤害原胜局：0。
- 现金差改善/退化/不变：57 / 7 / 1856。
- 候选最低逐对手胜率：82.81%。

## 证据文件

- FC12G：`experiments\fusion_champion_v1\receipts\fc12g_fc2b_mass_hire_guard_critical15_seed593001_n64x2_v1.json`
- FC2B 原版：`experiments\fusion_champion_v1\receipts\fc12h_fc2b_source_critical15_seed593001_n64x2_v1.json`
- 机器可读配对结果：`experiments\fusion_champion_v1\receipts\fc12i_fc12g_vs_fc2b_paired_critical15_seed593001_n64x2_v1.json`
