# TRI_A08_r11_r1 — observed-stock price-floor sale-window DP

**Status: compiled, source/native matched, and frozen for central testing. This is not an accepted >85% agent.**

The only parent is the supplied **A08_r11**, not A08_r12 and not an A06 variant. Root `main.py` loads `policy/tri_a08_r11_r1.so`. This directory contains the complete production implementation, not a patch or a replacement policy. Production requires only `main.py`, `policy/`, and the matching shared library; the offline tests and research evidence are not runtime inputs.

## Change and reason

The parent's sale-window DP correctly avoided its cumulative-sales stock model when projected supply could reach the one-unit price floor. However, that rejection returned the decision to a local holding rule without rival-supply timing. Congestion therefore disabled precisely the competition-aware scheduling that could matter most.

The official engine still pays 1 for a unit sold at the price floor, but **does not add that unit to market inventory**. A final lockstep pair quoted above the floor can advance inventory two units and overshoot the serial saturation threshold by one. Consequently, `(tick, remaining quantity)` alone is not a sufficient DP state across the floor.

This revision adds an exact bounded `(tick, remaining quantity, actual market stock)` continuation **only for such floor-crossing windows**. It uses exact serial and lockstep stock transitions, retained integral quote tables, and the existing common-schedule scenario comparison. The original non-floor DP is unchanged. A current-observation anchor prevents the new branch from authorizing sales of later synthetic MPC inventory. Within this new floor branch only, the objective is actual own cash minus opponent cash; macro proposal weights greater than one are not treated as a cash exchange rate.

This is one sale-window repair, not a new supply predictor or a portfolio redesign. The same physical shed surplus after current unit execution, feed/material reservations, previously chosen same-day deadline, cash reserve, capacity guard, and action budget remain in force. No future harvest is credited as current cash. The new branch advances sales; it does not extend a holding deadline or independently alter worker routes. Later real feedback can still change subsequent plans.

The window remains at most five ticks (current tick plus four); batch size is at most 100 and reachable floor-stock span at most 1,024. Quote-domain or other eligibility failures use the parent behavior. Rival traffic is a hypothesis from completed observed sale-days, never enemy private inventory or saved future actions.

Exactly **5 of the 44 production text files** differ from the parent: `policy/sale_schedule_dp.hpp` (the mechanism), `policy/search.hpp` (current observation anchor), `policy/triad.hpp` (diagnostics), `main.py` (entry identity/default library), and `build.py` (build identity/feature switch). Configuration, Python observation codec, investment logic, rolling planner, executor, and the other 39 files retain parent hashes. See `SOURCE_DIFF.patch` and `SOURCE_FREEZE.json`.

## Entry and offline build

```python
from main import agent, reset
# action = agent(legal_observation, configuration)
# reset() between independent matches when reusing a Python process
```

The supplied library is Linux x86-64 ELF. Build without downloads using a C++20 compiler and Python's standard library:

```sh
python -B build.py
# Diagnostic ablation: disable only the added floor branch.
python -B build.py --sale-floor-dp 0 --out /tmp/tri_floor_off.so
```

`BUILD.json` and `policy/tri_a08_r11_r1.BUILD.json` are actual receipts for the delivered library, including every flag and all 44 source hashes. `BUILD.md` records the command, resource measurements, ABI inspection, and independent clean rebuild. The clean production-only rebuild produced the **same native SHA256**. Root default `main.agent` also passed two 48-observation common prefixes from a working directory outside the source tree, without specifying a binary override.

The library dynamically requires the inspected system C/C++ runtime; the highest recorded GLIBCXX and GLIBC symbol versions are GLIBCXX_3.4.31 and GLIBC_2.32. Rebuild on an incompatible host rather than assuming universal binary compatibility. This was not a Kaggle sandbox ABI or official agent-time-limit certification.

## Validation summary — scopes must not be mixed

| Check | Recorded outcome | What it establishes |
|---|---|---|
| Input and parent identity | Input archive SHA matched; all 44 parent source hashes matched | Correct lineage |
| All historical rows recounted | 393/480; public 368/440; original AFS R2 25/40 | Historical development results only |
| Saved-action official re-execution | 5,033 matching transitions; 420 matching player-days; cash residuals zero | Seven factual trajectories audited, not new wins |
| Original non-floor C++ tests | 420 problems; 1,680 exhaustive scenario optima; 17,108 transitions | Old DP regression coverage |
| New floor C++ tests | 702 problems; 2,592 independent exhaustive scenario optima; 6,174 primitives | Bounded floor-state correctness |
| ASan/UBSan | Same floor unit suite passed, leak detection enabled | Sanitizer coverage of unit solver, not whole-agent certification |
| Official market oracle | 4,974 cases, both seats and three orderings, all matched | Exact unit receipts and price-floor stock transitions |
| Additional guards | 31 observation/funding guard assertions; 9 resume boundary problems / 129 assertions; actual-margin negative regression passed | Authorization, capacity/reservations, and scoped objective protection |
| Floor-off ablation | All 5,033/5,033 supplied actions matched the parent | Removing the new feature recovers tested parent behavior |
| Final candidate legal prefixes | Four entire 719-step paths unchanged; three market-only first divergences | Localizes the first change without following saved future after divergence |
| Conditional official short branches | 32 branch runs / 16 paired comparisons | Conditional, explicitly hypothetical supply stress tests, not wins |
| Final full closed-loop games | **5 wins, 3 losses, 0 draws / 8**, against the supplied full A08_r11 | New parent-opponent development results, **not R2/public-pool acceptance** |

All eight final full games ran 719 steps with no runtime exceptions, zero cash-identity residuals, and no sell quantity beyond actual post-unit inventory. There were 10 unsuccessful `BUY_SEED` unit attempts per side across the panel (20 combined), equally present in parent and candidate; they are not silently counted as successful orders or recoverable revenue. Maximum observed policy-call time in this small panel was about 0.377 s. Neither this runtime sample nor these eight games establishes the central acceptance result.

Repeated oracle/unit runs are replications of the same cases, not additional independent sample sizes. The round contains 17 full parent-opponent games across three implementation checkpoints: one unanchored pilot, eight anchored/configured-weight games, and eight final games. Only the last eight evaluate the delivered native.

## Positive and negative evidence

In the original R2 worst-loss trajectory the parent held 24 wool units while the observed price went from 158 at step 530 to 1 at step 532. The original full-game deficit was 18,759. This is a mechanism clue, **not** a claim that its lost cash or victory can be recovered.

At the final candidate's first common-prefix divergence in that fixture (step 482), four-tick official branches with declared public-history-derived rival wool supply improved actual cash margin by +342, +2,222, or +2,222 depending on queue position; the no-additional-supply branch was unchanged. These are bounded conditional receipts, not a replayed R2 win.

The Thomas historical **+70** narrow-win path is unchanged for all 719 actions in the final candidate. A harmful intermediate price-floor choice that sacrificed 26 own cash to deny 13 rival cash was removed by the scoped true-cash-margin objective. Its negative branch now has zero difference in all four tested supply/order conditions.

**The R2 +2 narrow-win fixture is not certified safe.** Its first divergence is at step 434. A conditional early-rival branch improves actual margin by 291, but no additional rival supply or late queue positions cost 19 relative to the parent. The Aurax close-loss window also has an adverse -8 branch. These negatives are retained in `validation/logs/FINAL_BRANCH_COMPARISON.json`, not discarded or reclassified as wins.

## Limitations and acceptance

The original AFS R2 executable and the complete public-opponent set were not present in the attachment. No new genuine R2 or 12-opponent full matches were run here; the parent was used only as an explicitly named, non-PASS functional opponent. Saved replay futures after a different action were never used as a counterfactual trajectory. Short branches clear actual enemy private inventory, inject declared hypothetical supply, and obtain subsequent own observations from the official engine.

The trailing rival profile remains lagged and uncertain, and the scenario set does not model every multi-product queue interaction or long-term investment response. Exact DP values are conditional model values, not proof of realizable game-level benefit. Supply-estimation redesign and full opponent-pool validation remain unfinished. This candidate can regress despite the repaired floor-state transition.

The user's target is strict overall win rate >85%. The central mixed development panel is 64 seeds × 12 opponents × 2 seats = 1,536 games and requires **at least 1,306 wins**. It is not a sealed holdout. That acceptance panel was **not run in this chat**, and no result from it is claimed. The frozen candidate must be evaluated on the central updated-seed panel before promotion.

## Reproduce the small checks

Use fresh output directories **outside** this release to preserve frozen evidence. The runner is serial and never launches the central panel. It needs no third-party Python package.

```sh
python -B tests/verify_release.py --out /tmp/tri-release-check.json
python -B tests/run_validation.py unit --with-sanitizers --out /tmp/tri-unit
python -B tests/run_validation.py official-floor --out /tmp/tri-floor-oracle
python -B tests/run_validation.py prefixes --out /tmp/tri-prefixes
python -B tests/run_validation.py ablation-prefixes --out /tmp/tri-ablation
python -B tests/run_validation.py branches --out /tmp/tri-branches
python -B tests/run_validation.py audit --out /tmp/tri-saved-audit
# Optional: exactly eight original development-seed games against parent, not R2.
python -B tests/run_validation.py parent-matches --allow-full-games --out /tmp/tri-parent8
```

The portable unit and official-floor runner modes were actually exercised from the staged release. Raw phase drivers preserve their original absolute execution paths under `research/original_drivers`; they are audit records rather than the recommended portable entry point.

## Evidence layout

`validation/source_input/` is the unpacked, unmodified supplied bundle, including the complete parent, all 480 rows, selected-case replays, provided audits, and frozen official CPU referee. `validation/logs/` contains actual compilation, raw observations/actions, official unit cash logs, rejected-order accounting, timing, resource, ablation, prefix, and intermediate failed-attempt logs. `validation/cases/` declares conditional branch inputs. `research/checkpoints/` preserves the superseded implementations and matching libraries; **none is the root runtime**. `research/ablations/` holds the compiled floor-off variant. No original ZIP is nested and no old binary substitutes for the final one.

The source-input identity file also names A06 variants as supplied metadata; no A06 production policy was imported. An inherited test field named `historical_r8_action` is a stale label in the provided driver: its values are the attached **A08_r11** action trace. Some parent-history `.json` files contain gzip bytes; inspect the magic header or use `gzip.decompress`.

Resources were recorded 3 min 53 s after original dispatch: effective CPU quota 4 cores, cgroup limit 4 GiB, and 30% memory reserved. The host memory reading was recorded only as context, not used as an allocation limit. Build/probe measurements determined the small validation scale. Full matches were limited to one at a time. See `validation/logs/RESOURCE_SUMMARY.json` and `DELIVERY_TIMING.json` for actual timing and the unchanged hard deadline.
