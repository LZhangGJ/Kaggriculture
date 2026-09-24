# Final dual-panel fusion review

**The fusion study, diagnostics and acceptance are complete. The 90% average win-rate target on both panels was not met. Retain V1 as the baseline; do not promote V2.**

| Version | Internal 13 | External 10 entries | Test panel |
|---|---:|---:|---|
| V1 | 1243/1300 = 95.62% | 861/1000 = 86.10% | First 50 seeds |
| V2, animal ranking multiplier 0.6 | 1242/1300 = 95.54% | 832/1000 = 83.20% | New 50 seeds |

Each acceptance panel used both seats and 100 games per opponent. Every game completed 719 transitions without execution errors. Draws are not wins. Public entries are the frozen September 23–24, 2026 snapshot.

## Matched-seed comparison

V1 was also replayed on the exact V2 external panel: **V1 863/1000; V2 832/1000**. The paired difference is -3.10 percentage points (95% seed-cluster interval -7.80 to +0.90). Both source versions remained frozen; there was no retuning.

Matched MetaV4 wins stayed at 32/100, while Master Engine V3 fell from 46 to 27 wins. An improved mean margin against MetaV4 did not translate into additional wins. The interval crosses zero, so universal statistical inferiority is not established; there is still no evidence supporting promotion.

## Retained changes

V1 preserves the Cashflow native planner/executor and public-output sale logic, adds the Liquidity opening wheat buy/sell intent, disables intraday admission/procurement of new projects, and caps hired workers at 12. No ML/RL, hidden rival state, opponent identity branches or test seed is used.
V2 additionally lowered new-animal candidate ranking to 0.6. Its development improvement did not survive the matched independent panel, so this change is not promoted. The 0.4 trial changed the opening purchases and caused a large regression.

## Remaining bottleneck and handoff

MetaV4 and Master Engine V3 remain the principal weaknesses. A global animal preference changes capital allocation, worker load and market interaction; it is not a general repair. Investigate where realized cash diverges from plan valuation, and make state-conditioned economic corrections before further promotion.
Use `agent_v1/main.py` for the retained baseline. `agent/main.py` is the rejected V2 comparison candidate. Both sources/runtimes, bilingual reports, per-game evidence and hashes are bundled. Detailed result and reproduction notes: ACCEPTANCE_EN.html / ACCEPTANCE_EN.md. The already-shared frozen opponent bundle is an external dependency.
There were 26 candidate definitions and 10,424 main game executions, including repeated controls. All 9 current-version serial reruns matched exact terminal cash; maximum observed serial call was 1.108s. This local referee/timing verification is not Kaggle sandbox certification.

V2 exceeded the strict one-second budget in the slowest serial case. V1's previously checked serial maximum was 0.959s. These are measured cases, not guarantees for all states or deployment platforms.
