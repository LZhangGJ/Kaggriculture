# Three-agent shared64 evaluation

This report contains the complete measurements for TRI_A08_r11_r2_fix1, TRI_A06_r6_r2, TRI_A06_r12_r2. The target is an overall strict win rate greater than 85%; the table below reports every planned game.

The official evaluator completed 4,608 games: 64 shared environment seeds, 12 frozen opponent entries and both seats, for 1,536 games per agent. All games completed 719 transitions with zero errors and 0 draws. The arena checked terminal cash against rewards and verified every full replay SHA256. A separate export check reconciled all raw result rows, every reported aggregate and the exported CSV.

| Agent | Strict wins | Overall | Representative (45 seeds) | Stress (19 seeds) | Mean final cash |
|---|---:|---:|---:|---:|---:|
| A08 r11 revision 2, fix 1 | 1097/1536 | 71.42% | 68.52% | 78.29% | 102,636.31 |
| A06 r6 revision 2 | 1017/1536 | 66.21% | 62.78% | 74.34% | 97,487.42 |
| A06 r12 calendar revision 2 | 1087/1536 | 70.77% | 68.24% | 76.75% | 98,706.82 |

## Opponents

Each opponent contributes 128 games per agent. Entry weights are retained as requested: flexon_v5 and seven_turn share production files, and other entries also share substantial strategy templates. Names should not be treated as independent strategy families.

| Opponent | A08 r11 revision 2, fix 1 | A06 r6 revision 2 | A06 r12 calendar revision 2 |
|---|---:|---:|---:|
| soil_v219g | 89/128 (69.53%) | 77/128 (60.16%) | 84/128 (65.62%) |
| moon_v215 | 85/128 (66.41%) | 88/128 (68.75%) | 88/128 (68.75%) |
| flexon_v5 | 88/128 (68.75%) | 75/128 (58.59%) | 81/128 (63.28%) |
| market_smart_v8 | 88/128 (68.75%) | 75/128 (58.59%) | 81/128 (63.28%) |
| nagatakengo_v70 | 128/128 (100.00%) | 128/128 (100.00%) | 128/128 (100.00%) |
| aurax_reactive_v1 | 85/128 (66.41%) | 88/128 (68.75%) | 89/128 (69.53%) |
| thomas_955_v2 | 94/128 (73.44%) | 86/128 (67.19%) | 89/128 (69.53%) |
| shop0909 | 88/128 (68.75%) | 75/128 (58.59%) | 81/128 (63.28%) |
| aurax_shop_v2 | 88/128 (68.75%) | 75/128 (58.59%) | 83/128 (64.84%) |
| seven_turn | 88/128 (68.75%) | 75/128 (58.59%) | 81/128 (63.28%) |
| ahmed_v27 | 94/128 (73.44%) | 74/128 (57.81%) | 84/128 (65.62%) |
| submission_56149565 | 82/128 (64.06%) | 101/128 (78.91%) | 118/128 (92.19%) |

## Public pool and seats

| Agent | Public11 | Original AFS R2 | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|
| A08 r11 revision 2, fix 1 | 1015/1408 (72.09%) | 82/128 (64.06%) | 70.70% | 72.14% |
| A06 r6 revision 2 | 916/1408 (65.06%) | 101/128 (78.91%) | 66.41% | 66.02% |
| A06 r12 calendar revision 2 | 969/1408 (68.82%) | 118/128 (92.19%) | 71.48% | 70.05% |

## Protocol and interpretation

Seeds were sampled once, uniformly without replacement, from the 256 unused entries in the published representative256 + stress128 union after the exact candidate identities were frozen. All 514 registered or author-development seeds were excluded. This mixed development panel has 45 representative and 19 stress seeds; it is not the sealed Holdout. Previous panels used different seeds. Differences between panels are not a controlled estimate of a code change.

Wins require strictly greater terminal cash. The denominator includes all planned games; draws are not wins. Cash is diagnostic, with no cash acceptance gate. At least 1,306 wins out of 1,536 are required for the overall target. Individual opponent or seed-stratum pass flags in the raw JSON are diagnostics and do not mean an agent passed overall.

Original AFS R2 is submission_56149565, native SHA256 `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Policies run in isolated processes, reset each game, with the environment seed withheld. The local response watchdog is 120 seconds; this evaluation does not certify Kaggle execution-time limits.

All three tested revisions passed focused independent source/native and behavior checks before this panel. Author-side component checks, saved-observation probes and selected historical matches are separate development evidence; the measurements here use the fixed public11 plus original R2 pool. Follow-up revisions must use a newly frozen shared seed panel.

Evidence: [full metrics](RESULTS.json), [game rows](GAMES.csv), [row hashes](RESULT_ROW_HASHES.json), [portable protocol and identities](PORTABLE_PROTOCOL.json), [independent reconciliation](INDEPENDENT_CHECK.json). Full local replays remain retained; their SHA256 values are in the CSV.

[Exact strategy source and native releases](../../agents/pro8_20260913_revision2/README.md).
