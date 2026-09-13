# Three-agent shared64 baseline evaluation

All three original published agents missed the requested overall strict win-rate target of greater than 85%. This report contains completed measurements of the original A08 r11, A06 r6 and A06 r12 calendar agents. It is not a result for their first revisions.

The official evaluator completed 4,608 games: 64 shared environment seeds, 12 frozen opponent entries and both seats, for 1,536 games per agent. All games completed 719 transitions with zero errors and zero draws. The arena checked terminal cash against rewards and verified every full replay SHA256. A separate export check reconciled all raw result rows, every reported aggregate and the exported CSV.

| Agent | Strict wins | Overall | Representative (35 seeds) | Stress (29 seeds) | Mean final cash |
|---|---:|---:|---:|---:|---:|
| A08 r11 | 1120/1536 | 72.92% | 70.48% | 75.86% | 107,251.58 |
| A06 r6 | 1095/1536 | 71.29% | 70.60% | 72.13% | 102,993.51 |
| A06 r12 calendar | 1054/1536 | 68.62% | 67.14% | 70.40% | 104,525.41 |

## Opponents

Each opponent contributes 128 games per agent. Entry weights are retained as requested: flexon_v5 and seven_turn share production files, and other entries also share substantial strategy templates. Names should not be treated as independent strategy families.

| Opponent | A08 r11 | A06 r6 | A06 r12 calendar |
|---|---:|---:|---:|
| soil_v219g | 93/128 (72.66%) | 80/128 (62.50%) | 82/128 (64.06%) |
| moon_v215 | 94/128 (73.44%) | 95/128 (74.22%) | 86/128 (67.19%) |
| flexon_v5 | 91/128 (71.09%) | 81/128 (63.28%) | 80/128 (62.50%) |
| market_smart_v8 | 91/128 (71.09%) | 81/128 (63.28%) | 80/128 (62.50%) |
| nagatakengo_v70 | 128/128 (100.00%) | 128/128 (100.00%) | 128/128 (100.00%) |
| aurax_reactive_v1 | 94/128 (73.44%) | 95/128 (74.22%) | 86/128 (67.19%) |
| thomas_955_v2 | 87/128 (67.97%) | 95/128 (74.22%) | 76/128 (59.38%) |
| shop0909 | 91/128 (71.09%) | 81/128 (63.28%) | 80/128 (62.50%) |
| aurax_shop_v2 | 91/128 (71.09%) | 78/128 (60.94%) | 80/128 (62.50%) |
| seven_turn | 91/128 (71.09%) | 81/128 (63.28%) | 80/128 (62.50%) |
| ahmed_v27 | 92/128 (71.88%) | 82/128 (64.06%) | 81/128 (63.28%) |
| submission_56149565 | 77/128 (60.16%) | 118/128 (92.19%) | 115/128 (89.84%) |

## Public pool and seats

| Agent | Public11 | Original AFS R2 | Seat 0 | Seat 1 |
|---|---:|---:|---:|---:|
| A08 r11 | 1043/1408 (74.08%) | 77/128 (60.16%) | 73.18% | 72.66% |
| A06 r6 | 977/1408 (69.39%) | 118/128 (92.19%) | 70.70% | 71.88% |
| A06 r12 calendar | 939/1408 (66.69%) | 115/128 (89.84%) | 68.88% | 68.36% |

## Protocol and interpretation

Seeds were sampled once, uniformly without replacement, from the published representative256 + stress128 union after the exact candidate identities were frozen. This mixed development panel has 35 representative and 29 stress seeds; it is not the sealed Holdout. The earlier 20-seed results and this panel have different seed composition, so their rate difference is not a controlled estimate of a code change.

Wins require strictly greater terminal cash. The denominator includes all planned games; draws are not wins. Cash is diagnostic, with no cash acceptance gate. At least 1,306 wins out of 1,536 are required for the overall target. Individual opponent or seed-stratum pass flags in the raw JSON are diagnostics and do not mean an agent passed overall.

Original AFS R2 is submission_56149565, native SHA256 `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Policies run in isolated processes, reset each game, with the environment seed withheld. The local response watchdog is 120 seconds; this evaluation does not certify Kaggle execution-time limits.

The three first revisions have already passed focused local source/native and behavior checks. Their next frozen panel uses 64 different seeds (48 representative, 16 stress), the same 12 entries and both seats. No revised-agent win rate is claimed here.

Evidence: [full metrics](RESULTS.json), [game rows](GAMES.csv), [row hashes](RESULT_ROW_HASHES.json), [portable protocol and identities](PORTABLE_PROTOCOL.json), [independent reconciliation](INDEPENDENT_CHECK.json). Full local replays remain retained; their SHA256 values are in the CSV.
