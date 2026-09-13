# Three-agent shared64 first-revision evaluation

This report contains the complete first-revision measurements for TRI_A08_r11_r1, TRI_A06_r6_r1 and TRI_A06_r12_r1. The target is an overall strict win rate greater than 85%; the table below reports every planned game.

The official evaluator completed 4,608 games: 64 shared environment seeds, 12 frozen opponent entries and both seats, for 1,536 games per agent. All games completed 719 transitions with zero errors and zero draws. The arena checked terminal cash against rewards and verified every full replay SHA256. A separate export check reconciled all raw result rows, every reported aggregate and the exported CSV.

| Agent | Strict wins | Overall | Representative (48 seeds) | Stress (16 seeds) | Mean final cash |
|---|---:|---:|---:|---:|---:|
| A08 r11 revision 1 | 1054/1536 | 68.62% | 67.53% | 71.88% | 107,498.50 |
| A06 r6 revision 1 | 1196/1536 | 77.86% | 78.04% | 77.34% | 105,068.61 |
| A06 r12 calendar revision 1 | 1099/1536 | 71.55% | 73.26% | 66.41% | 103,808.81 |

## Opponents

Each opponent contributes 128 games per agent. Entry weights are retained as requested: flexon_v5 and seven_turn share production files, and other entries also share substantial strategy templates. Names should not be treated as independent strategy families.

| Opponent | A08 r11 revision 1 | A06 r6 revision 1 | A06 r12 calendar revision 1 |
|---|---:|---:|---:|
| soil_v219g | 79/128 (61.72%) | 101/128 (78.91%) | 85/128 (66.41%) |
| moon_v215 | 96/128 (75.00%) | 79/128 (61.72%) | 86/128 (67.19%) |
| flexon_v5 | 79/128 (61.72%) | 99/128 (77.34%) | 83/128 (64.84%) |
| market_smart_v8 | 79/128 (61.72%) | 99/128 (77.34%) | 83/128 (64.84%) |
| nagatakengo_v70 | 128/128 (100.00%) | 128/128 (100.00%) | 128/128 (100.00%) |
| aurax_reactive_v1 | 96/128 (75.00%) | 79/128 (61.72%) | 86/128 (67.19%) |
| thomas_955_v2 | 97/128 (75.78%) | 95/128 (74.22%) | 96/128 (75.00%) |
| shop0909 | 79/128 (61.72%) | 99/128 (77.34%) | 83/128 (64.84%) |
| aurax_shop_v2 | 79/128 (61.72%) | 99/128 (77.34%) | 87/128 (67.97%) |
| seven_turn | 79/128 (61.72%) | 99/128 (77.34%) | 83/128 (64.84%) |
| ahmed_v27 | 78/128 (60.94%) | 101/128 (78.91%) | 84/128 (65.62%) |
| submission_56149565 | 85/128 (66.41%) | 118/128 (92.19%) | 115/128 (89.84%) |

## Public pool and seats

| Agent | Public11 | Original AFS R2 | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|
| A08 r11 revision 1 | 969/1408 (68.82%) | 85/128 (66.41%) | 69.14% | 68.10% |
| A06 r6 revision 1 | 1078/1408 (76.56%) | 118/128 (92.19%) | 78.26% | 77.47% |
| A06 r12 calendar revision 1 | 984/1408 (69.89%) | 115/128 (89.84%) | 71.74% | 71.35% |

## Protocol and interpretation

Seeds were sampled once, uniformly without replacement, from the 320 unused entries in the published representative256 + stress128 union after the exact candidate identities were frozen. All 450 registered or author-development seeds were excluded. This mixed development panel has 48 representative and 16 stress seeds; it is not the sealed Holdout. The original-agent baseline used 64 different seeds (35 representative, 29 stress). Differences between those two panels are not a controlled estimate of a code change.

Wins require strictly greater terminal cash. The denominator includes all planned games; draws are not wins. Cash is diagnostic, with no cash acceptance gate. At least 1,306 wins out of 1,536 are required for the overall target. Individual opponent or seed-stratum pass flags in the raw JSON are diagnostics and do not mean an agent passed overall.

Original AFS R2 is submission_56149565, native SHA256 `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Policies run in isolated processes, reset each game, with the environment seed withheld. The local response watchdog is 120 seconds; this evaluation does not certify Kaggle execution-time limits.

All three tested revisions passed focused independent source/native and behavior checks before this panel. Author-side small matches against each own parent are separate development evidence; the measurements here use the fixed public11 plus original R2 pool. Follow-up revisions must use a newly frozen shared seed panel.

Evidence: [full metrics](RESULTS.json), [game rows](GAMES.csv), [row hashes](RESULT_ROW_HASHES.json), [portable protocol and identities](PORTABLE_PROTOCOL.json), [independent reconciliation](INDEPENDENT_CHECK.json). Full local replays remain retained; their SHA256 values are in the CSV.

[Exact first-revision source and native releases](../../agents/pro8_20260913_revision1/README.md).
