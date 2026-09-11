# Original P16 versus route economics: 880 live games

This historical panel contains 20 seeds (`2609168000..2609168019`), 11 original live public opponents, both seats and two frozen agents: 440 games each. Original P16 wins 324, loses 116; route economics wins 346, loses 94. No draws occur. The raw win-rate difference is +5.00 percentage points, but the whole-seed 95% interval is [-0.91, +14.55].

The host dynamically links `fastkag::Simulator::step` from the already compiled frozen P16 library. Only a Python binding/audit interface was compiled, not the game rules or either strategy. Each side receives only its visible observation; the seed is not passed to agents.

| Metric | P16 | Route economics |
|---|---:|---:|
| Win rate | 73.64% | 78.64% |
| Mean final cash | 110,224.923 | 107,761.991 |
| Mean margin | 16,351.086 | 16,186.386 |
| Mean wages | 4,192.700 | 4,031.105 |
| Mean movement steps | 2,452.425 | 2,451.225 |
| Mean overnight storage loss | 14.941 | 11.102 |
| Daytime DROP loss | 0 | 0 |
| Invalid work actions | 0 | 0 |
| Protected-task omissions | 0 | 12 |
| Maximum batch decision, seconds | 0.542 | 2.343 |

The candidate rescues 30 losses and loses eight wins. Wages decrease in 232/440 cases: 112 of these also increase cash, while 120 decrease it. Another 95 cases increase both wages and cash. Lower labor cost alone is not a sufficient objective.

Only 64/440 pairs share the entire subsequent shop sequence. The official daily generator draws weeds on empty land before new shops, so changed empty land can change later shops even at a fixed seed. Paired differences measure the full policy/environment/opponent trajectory; isolate local route effects with fixed-observation interventions.

## Opponent detail

| Opponent | P16 wins/losses | Candidate wins/losses | Mean cash change | Mean wage change |
|---|---:|---:|---:|---:|
| Soil | 27/13 | 29/11 | -2,948.9 | -163.6 |
| Moon | 33/7 | 33/7 | -793.7 | -152.0 |
| Flexon | 27/13 | 29/11 | -3,002.1 | -186.9 |
| Market Smart | 27/13 | 29/11 | -3,002.9 | -186.9 |
| Nagatakengo | 40/0 | 40/0 | -6,036.0 | -223.4 |
| Aurax Reactive | 33/7 | 33/7 | -793.7 | -152.0 |
| Thomas 95.5 | 29/11 | 37/3 | +1,485.3 | +32.3 |
| Shop0909 | 27/13 | 29/11 | -3,001.7 | -186.9 |
| Aurax Shop | 27/13 | 29/11 | -2,994.8 | -184.2 |
| Seven Turn | 27/13 | 29/11 | -3,002.1 | -186.9 |
| Ahmed v27 | 27/13 | 29/11 | -3,001.7 | -186.9 |

For seed 2609168008 against Moon, the candidate saves 694 in wages but loses 9,465 in cash; margin changes from +16,236 to -3,347. Shop sequences diverge on day 10. For seed 2609168009 against Thomas, own cash rises 5,794 while opponent cash rises 15,815, still reversing a win; shops diverge on day 7. Neither final difference can be attributed solely to one hire cancellation.

## Confirmed execution defects

The candidate cannot replace P16 despite its higher aggregate wins. On day 16, seed 2609168005 against Moon and Aurax Reactive, both seats sow tomatoes at (9,0) only on the last step and omit watering; the tile becomes weeds after day change. Individual repair cannot reserve a watering task that is not yet valid on empty land. All four games are wins but still count as execution failures.

Seed 2609168007 against Nagatakengo, both seats, omits tomato sowing/watering at (1,9) and goose feeding/care at (4,9). Nine workers were hired that day, yet workers collect eggs/fertilizer and leave without service. The unresolved task audit runs from early in the day through its end. This exposes disagreement between hire-reduction trials, actual scheduling and dependency-chain preservation.

| Opponent / seed / seats | Day | Missing commitments |
|---|---:|---|
| Moon / 2609168005 / both | 16 | Water at (9,0), one per game |
| Aurax Reactive / 2609168005 / both | 16 | Water at (9,0), one per game |
| Nagatakengo / 2609168007 / both | 16 | Sow and water at (1,9); feed and care at (4,9), four per game |

These defects agree in Python and the existing C++ simulator. Later complete-workflow work addresses the mechanism generally; no seed or coordinate special cases belong in the policy.

## Verification and limits

The input audit covers 95 frozen files and output/replay hashes. C++ executes 632,720 action pairs. All 334 saved earlier Python games are replayed into C++, matching 480,960 bilateral observations and wage/storage/task audits. The overlapping 334 new live C++ games also match actions and observations. No claim is made that the remaining games had full Python revalidation.

The batch uses eight processes. The host does not enforce Kaggle timing/schema gates, and invalid-work counts exclude PASS and movement. Protected tasks cover accepted route/hire commitments only. Overnight loss and daytime discard are separate metrics.

The maximum per-game sum of time above one second is 0 for P16 and 4.097 seconds for the candidate. After stopping the batch, two slow trajectories independently reproduce all 719 actions, peaking at 0.671 and 0.623 seconds, with no steps above one second. This is local timing, not competition certification.

Several opponents share action libraries. Across this panel, Moon/Aurax Reactive and Flexon/Seven Turn each have identical opponent action sequences in all 80 tested conditions. They are retained at the original panel weights but are not 11 independent policy families.

`POOL_VALIDATION.json` preserves the summary. Full machine-local evidence is in `experiments/r2p16_vs_route_agent_pool_20260910`: `SUMMARY.json`, `PAIRED_CASES.csv`, `FUNCTIONAL_CASES.json`, native build/calibration receipts, serial timing, protocols and complete compressed replays. Selected historical receipts are already published under [economics-stage evidence](../../evidence/r2p16_route_agents_20260911/economics_stage).
