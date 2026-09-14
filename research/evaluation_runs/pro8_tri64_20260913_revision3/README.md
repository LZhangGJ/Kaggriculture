# Three-agent shared64 evaluation

This report contains the complete measurements for TRI_A08_r11_r3, TRI_A06_r6_r3_bundle_fix1, TRI_A06_r12_r3_fix1. The target is an overall strict win rate greater than 85%; the table below reports every planned game.

The official evaluator completed 4,608 games: 64 shared environment seeds, 12 frozen opponent entries and both seats, for 1,536 games per agent. All games completed 719 transitions with zero errors and 0 draws. The arena checked terminal cash against rewards and verified every full replay SHA256. A separate export check reconciled all raw result rows, every reported aggregate and the exported CSV.

| Agent | Strict wins | Overall | Representative (46 seeds) | Stress (18 seeds) | Mean final cash |
|---|---:|---:|---:|---:|---:|
| A08 r11 revision 3 | 1087/1536 | 70.77% | 71.11% | 69.91% | 104,564.11 |
| A06 r6 revision 3, combined funding and land comparison, fix 1 | 1286/1536 | 83.72% | 83.06% | 85.42% | 104,705.78 |
| A06 r12 calendar revision 3, fix 1 | 1217/1536 | 79.23% | 82.25% | 71.53% | 104,778.15 |

## Opponents

Each opponent contributes 128 games per agent. Entry weights are retained as requested: flexon_v5 and seven_turn share production files, and other entries also share substantial strategy templates. Names should not be treated as independent strategy families.

| Opponent | A08 r11 revision 3 | A06 r6 revision 3, combined funding and land comparison, fix 1 | A06 r12 calendar revision 3, fix 1 |
|---|---:|---:|---:|
| soil_v219g | 81/128 (63.28%) | 103/128 (80.47%) | 97/128 (75.78%) |
| moon_v215 | 98/128 (76.56%) | 110/128 (85.94%) | 105/128 (82.03%) |
| flexon_v5 | 81/128 (63.28%) | 103/128 (80.47%) | 93/128 (72.66%) |
| market_smart_v8 | 81/128 (63.28%) | 103/128 (80.47%) | 93/128 (72.66%) |
| nagatakengo_v70 | 128/128 (100.00%) | 128/128 (100.00%) | 128/128 (100.00%) |
| aurax_reactive_v1 | 98/128 (76.56%) | 110/128 (85.94%) | 105/128 (82.03%) |
| thomas_955_v2 | 106/128 (82.81%) | 105/128 (82.03%) | 99/128 (77.34%) |
| shop0909 | 81/128 (63.28%) | 103/128 (80.47%) | 93/128 (72.66%) |
| aurax_shop_v2 | 81/128 (63.28%) | 99/128 (77.34%) | 93/128 (72.66%) |
| seven_turn | 81/128 (63.28%) | 103/128 (80.47%) | 93/128 (72.66%) |
| ahmed_v27 | 84/128 (65.62%) | 104/128 (81.25%) | 95/128 (74.22%) |
| submission_56149565 | 87/128 (67.97%) | 115/128 (89.84%) | 123/128 (96.09%) |

## Public pool and seats

| Agent | Public11 | Original AFS R2 | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|
| A08 r11 revision 3 | 1000/1408 (71.02%) | 87/128 (67.97%) | 70.44% | 71.09% |
| A06 r6 revision 3, combined funding and land comparison, fix 1 | 1171/1408 (83.17%) | 115/128 (89.84%) | 84.51% | 82.94% |
| A06 r12 calendar revision 3, fix 1 | 1094/1408 (77.70%) | 123/128 (96.09%) | 79.43% | 79.04% |

## Protocol and interpretation

Seeds were sampled once, uniformly without replacement, from the 192 unused entries in the published representative256 + stress128 union after the exact candidate identities were frozen. All 596 registered or author-development seeds were excluded. This mixed development panel has 46 representative and 18 stress seeds; it is not the sealed Holdout. Previous panels used different seeds. Differences between panels are not a controlled estimate of a code change.

Wins require strictly greater terminal cash. The denominator includes all planned games; draws are not wins. Cash is diagnostic, with no cash acceptance gate. At least 1,306 wins out of 1,536 are required for the overall target. Individual opponent or seed-stratum pass flags in the raw JSON are diagnostics and do not mean an agent passed overall.

Original AFS R2 is submission_56149565, native SHA256 `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Policies run in isolated processes, reset each game, with the environment seed withheld. The local response watchdog is 120 seconds; this evaluation does not certify Kaggle execution-time limits.

All three tested revisions passed focused independent source/native and behavior checks before this panel. Author-side component checks, saved-observation probes and selected historical matches are separate development evidence; the measurements here use the fixed public11 plus original R2 pool. Follow-up revisions must use a newly frozen shared seed panel.

Evidence: [full metrics](RESULTS.json), [game rows](GAMES.csv), [row hashes](RESULT_ROW_HASHES.json), [portable protocol and identities](PORTABLE_PROTOCOL.json), [independent reconciliation](INDEPENDENT_CHECK.json). Full local replays remain retained; their SHA256 values are in the CSV.

[Exact strategy source and native releases](../../agents/pro8_20260913_revision3/README.md).
