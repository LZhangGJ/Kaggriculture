# A06 dual-panel fusion: independent acceptance

Candidate: `v2_animal06`. Status: **NOT_MET**.

50 new shared seeds, both seats, 100 games per opponent. Internal: 13 original variants / 1,300 games. External: 10 frozen public entries / 1,000 games. Official 1.32.7 Python rules, live policies, 719 state transitions per completed game, zero execution errors. Draws do not count as wins.

| Panel | Wins / games | Strict win rate | 95% seed-cluster bootstrap interval | Mean cash margin |
|---|---:|---:|---:|---:|
| internal | 1242/1300 | 95.54% | 94.15%–96.85% | +11,464.8 |
| external | 832/1000 | 83.20% | 78.80%–86.50% | +9,740.9 |

External result after deduplicating identical main.py files (9 source entries): **81.89%**, 737/900.

The target is a 90% average strict win rate on each frozen panel. It is not a 90% guarantee against every opponent or a 90% confidence lower bound. The public pool is a September 23–24, 2026 snapshot; this is not a current online leaderboard or Kaggle sandbox certification.

## What changed

The native Cashflow C++ planner, executor, binary and public-supply sale wrapper are unchanged. Cashflow retains its heuristic that shifts half of positive field output value to the following day; this is a forecast, not future information.
An opening wheat buy/sell intent (10 units) uses cash, order-count and warehouse guards. Intraday admission/procurement of new projects is disabled. The hired-worker cap is 12. New-animal candidate ranking is multiplied by 0.6; this is a preference adjustment, not a fixed animal count or a fixed crop route.
No ML/RL, trained weights, opponent identity, opponent source code, hidden rival inventory or test seed is used for decision making. The learned_value header is a disabled stub. The agent still plans from the observed state.

## Remaining weaknesses

| Opponent | Wins-draws-losses / 100 | Strict win rate | Mean cash margin |
|---|---:|---:|---:|
| internal/r15c_rules | 99-0-1 | 99.0% | +13,176.4 |
| internal/r14_cashflow | 66-0-34 | 66.0% | +1,880.4 |
| internal/r14_daily_staff | 99-0-1 | 99.0% | +11,260.2 |
| internal/r14_f5l3 | 99-0-1 | 99.0% | +10,762.0 |
| internal/r14_asset_tail | 94-0-6 | 94.0% | +9,103.1 |
| internal/r14_frozen | 98-0-2 | 98.0% | +9,061.7 |
| internal/r14_terminal_switch | 99-0-1 | 99.0% | +10,769.8 |
| internal/rule_future65 | 100-0-0 | 100.0% | +20,871.9 |
| internal/r13_nml | 93-0-7 | 93.0% | +10,832.9 |
| internal/rule_r18 | 99-0-1 | 99.0% | +13,723.8 |
| internal/r14_liquidity | 100-0-0 | 100.0% | +13,626.6 |
| internal/r14_tl5 | 98-0-2 | 98.0% | +11,521.3 |
| internal/r22_continuation | 98-0-2 | 98.0% | +12,452.0 |
| Soil Remembers Rain | 95-0-5 | 95.0% | +12,732.3 |
| V15Stack | 97-0-3 | 97.0% | +11,333.7 |
| 2965 Master Hybrid Engine | 97-0-3 | 97.0% | +13,691.1 |
| V57 Funding Order Invariant | 97-0-3 | 97.0% | +13,566.9 |
| M4A MetaV4 | 32-0-68 | 32.0% | -2,085.8 |
| Demand Preserving Sale Timing (Soil identical source) | 95-0-5 | 95.0% | +12,732.3 |
| Master Engine V3 | 27-0-73 | 27.0% | -3,704.3 |
| More Wheat Smarter Sales | 98-0-2 | 98.0% | +11,989.1 |
| Farmer John | 97-0-3 | 97.0% | +13,577.5 |
| Pipe18 | 97-0-3 | 97.0% | +13,576.5 |

## Matched-seed external comparison

On the same 50 seeds and both seats: frozen V1 863/1000; frozen V2 832/1000. Paired win-rate difference: -3.10 percentage points; 95% seed-cluster interval -7.80 to +0.90 points.
This is a post-acceptance diagnosis without retuning. Do not attribute the raw difference between the original V1 and V2 reports to the parameter change: those original reports used different seed panels. Both runtimes are included in the release.

## Reproduction and timing

All 9 serial reruns reproduced both players' terminal cash and results exactly. Maximum local serial agent call in those checks: 1.108 seconds. This is a local timing observation, not platform certification.
The release includes the frozen runtime/source, SHA-256 manifest, official local referee, compressed per-game receipts and reproduce_one.py. The opponents come from the previously shared repository branch research/a06-dynamic-variants-public10-20260924, commit c00bcb4c5ed0cb43d750ad5a684e9f1ae318f80a.
Run under compatible Linux x86-64 / WSL. The inherited binary requires compatible GLIBC_2.32 and GLIBCXX_3.4.31 libraries. Rebuild from C++20 source if necessary, then recheck exact results.

```text
python reproduce_one.py --pool /path/to/research/a06_dynamic_variants_public10_20260924 --opponent internal/r14_cashflow --seat 0
```

The default seed is the first final holdout seed. The script verifies the runtime/referee and opponent hashes, then checks the exact recorded terminal cash. The historical README and reports inside agent/ belong to the parent package; use the root release README and evidence/ACCEPTANCE.json for this fusion.
