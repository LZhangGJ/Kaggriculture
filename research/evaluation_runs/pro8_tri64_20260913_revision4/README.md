# Three-agent shared64 evaluation

This report contains the complete measurements for TRI_A08_r11_r4, TRI_A06_r6_r4, TRI_A06_r12_r4. The target is an overall strict win rate greater than 85%; the table below reports every planned game.

The official evaluator completed 4,608 games: 64 shared environment seeds, 12 frozen opponent entries and both seats, for 1,536 games per agent. All games completed 719 transitions with zero errors and 0 draws. The arena checked terminal cash against rewards and verified every full replay SHA256. A separate export check reconciled all raw result rows, every reported aggregate and the exported CSV.

| Agent | Strict wins | Overall | Representative (37 seeds) | Stress (27 seeds) | Mean final cash |
|---|---:|---:|---:|---:|---:|
| A08 r11 revision 4 | 1190/1536 | 77.47% | 74.55% | 81.48% | 105,827.87 |
| A06 r6 revision 4 | 1280/1536 | 83.33% | 84.57% | 81.64% | 107,692.17 |
| A06 r12 calendar revision 4 | 1238/1536 | 80.60% | 81.76% | 79.01% | 108,247.58 |

## Opponents

Each opponent contributes 128 games per agent. Entry weights are retained as requested: flexon_v5 and seven_turn share production files, and other entries also share substantial strategy templates. Names should not be treated as independent strategy families.

| Opponent | A08 r11 revision 4 | A06 r6 revision 4 | A06 r12 calendar revision 4 |
|---|---:|---:|---:|
| soil_v219g | 101/128 (78.91%) | 105/128 (82.03%) | 100/128 (78.12%) |
| moon_v215 | 92/128 (71.88%) | 106/128 (82.81%) | 101/128 (78.91%) |
| flexon_v5 | 99/128 (77.34%) | 101/128 (78.91%) | 98/128 (76.56%) |
| market_smart_v8 | 99/128 (77.34%) | 101/128 (78.91%) | 98/128 (76.56%) |
| nagatakengo_v70 | 128/128 (100.00%) | 128/128 (100.00%) | 128/128 (100.00%) |
| aurax_reactive_v1 | 92/128 (71.88%) | 106/128 (82.81%) | 101/128 (78.91%) |
| thomas_955_v2 | 99/128 (77.34%) | 111/128 (86.72%) | 96/128 (75.00%) |
| shop0909 | 99/128 (77.34%) | 101/128 (78.91%) | 98/128 (76.56%) |
| aurax_shop_v2 | 97/128 (75.78%) | 101/128 (78.91%) | 98/128 (76.56%) |
| seven_turn | 99/128 (77.34%) | 101/128 (78.91%) | 98/128 (76.56%) |
| ahmed_v27 | 100/128 (78.12%) | 99/128 (77.34%) | 97/128 (75.78%) |
| submission_56149565 | 85/128 (66.41%) | 120/128 (93.75%) | 125/128 (97.66%) |

## Public pool and seats

| Agent | Public11 | Original AFS R2 | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|
| A08 r11 revision 4 | 1105/1408 (78.48%) | 85/128 (66.41%) | 77.99% | 76.95% |
| A06 r6 revision 4 | 1160/1408 (82.39%) | 120/128 (93.75%) | 83.85% | 82.81% |
| A06 r12 calendar revision 4 | 1113/1408 (79.05%) | 125/128 (97.66%) | 79.56% | 81.64% |

## Protocol and interpretation

Seeds were sampled once, uniformly without replacement, from the 128 unused entries in the published representative256 + stress128 union after the exact candidate identities were frozen. All 676 registered or author-development seeds were excluded. This mixed development panel has 37 representative and 27 stress seeds; it is not the sealed Holdout. Previous panels used different seeds. Differences between panels are not a controlled estimate of a code change.

Wins require strictly greater terminal cash. The denominator includes all planned games; draws are not wins. Cash is diagnostic, with no cash acceptance gate. At least 1,306 wins out of 1,536 are required for the overall target. Individual opponent or seed-stratum pass flags in the raw JSON are diagnostics and do not mean an agent passed overall.

Original AFS R2 is submission_56149565, native SHA256 `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Policies run in isolated processes, reset each game, with the environment seed withheld. The local response watchdog is 120 seconds; this evaluation does not certify Kaggle execution-time limits.

All three tested revisions passed focused independent source/native and behavior checks before this panel. Author-side component checks, saved-observation probes and selected historical matches are separate development evidence; the measurements here use the fixed public11 plus original R2 pool. Follow-up revisions must use a newly frozen shared seed panel.

Evidence: [full metrics](RESULTS.json), [game rows](GAMES.csv), [row hashes](RESULT_ROW_HASHES.json), [portable protocol and identities](PORTABLE_PROTOCOL.json), [independent reconciliation](INDEPENDENT_CHECK.json). Full local replays remain retained; their SHA256 values are in the CSV.

[Strategy sources and tested natives](../../agents/pro8_20260913_revision4/README.md).
