# C01 economic revision round 2

## Result and scope

**Delivered candidate: `C01_R2_h2`. The fixed untouched author validation panel averaged 209,593.875 terminal cash across 16 seeds and both seats, 32 complete games. This exceeds the 200,000 author economic target by 9,593.875.**

This is an author-side result against the supplied legal PASS opponent. It is not the controller's fresh shared-panel qualification, a round-robin result, or competitive acceptance. Cash has no later competitive acceptance threshold. No claim is made about winning against active opponents.

The selected candidate changes exactly one policy configuration value: `scenario: 4` becomes `scenario: 2`. The Python and C++ policy implementations, executor, payback strength (`c01_payback: 1`) and legal-maturity treatment (`c01_maturity: 1`) are unchanged. A three-check entrypoint/reset unit test was added outside the policy. Regenerated native/build artifacts are separately identified.

## Reproduction of the supplied baseline

The feedback ZIP SHA-256 matches the supplied expected value. All 64 entries of its manifest match, and all 69 original extracted files remain byte-for-byte unchanged. The baseline was copied and rebuilt locally. Its 32 terminal results reproduce **every** disclosed controller C01 cell exactly, with mean **196,342.0625**. This validates the source/referee/configuration setup before comparing alternatives. The four-day baseline is not the C04 parent; the supplied matched C04 replays are separate historical evidence.

The low-cash seed 2027896586, seat 0, also served as the single-game speed measurement: 139,533 cash, 719 transitions, 8.506 seconds. It is disclosed development data, not fresh validation. Old smoke seed 291307001 was not used for a new full-game result.

## Matched development experiments

Every row below uses exactly the same 16 disclosed seeds in both seats. Each cell is a full game, not a forecast, partial-game balance or marked-to-market inventory estimate. Each completed experiment includes all 32 outcomes, even when the mean or an individual result is poor.

| Configuration | Games | Mean terminal cash | Mean wall seconds/game |
|---|---:|---:|---:|
| Four-day baseline | 32 | 196,342.0625 | 9.516 |
| One-day rollout | 32 | 199,795.96875 | 4.095 |
| Two-day rollout, selected | 32 | 205,292.90625 | 5.896 |
| Three-day rollout | 32 | 200,489.09375 | 7.563 |
| Two-day, payback strength 0 | 32 | 203,252.25 | 6.034 |
| Two-day, payback strength 0.25 | 32 | 190,662.625 | 5.618 |
| Two-day, payback strength 2 | 32 | 195,801.09375 | 7.447 |
| Diagnostic: no new starts from day 24 | 32 | 201,109.28125 | 6.513 |

All rows have 32 complete games and zero failed or unstarted games. The first seven configurations were eligible for selection by highest full-panel development mean. The day-24 cutoff was predeclared diagnostic-only and was not eligible. Candidate selection was completed before validation. The two-day, strength-1 configuration wins that fixed comparison.

Relative to the four-day baseline, h2 gains **8,950.84375** mean cash, with 21 improved and 11 worsened seat/seed cells. This is a full-panel policy comparison, not a claim of universal improvement. Its mean game time fell from 9.516 to 5.896 seconds under the logged four-worker evaluation setup. Timing includes observation encoding, Python/C++ calls, full replay and diagnostic trace serialization, and the referee. The strength-2 experiment overlapped a diagnostic compile, so its larger wall time should not be read as a pure policy-runtime comparison.

The two disclosed weakest baseline cells improve as follows:

| Seed, seat | Four-day baseline | Selected two-day | Change |
|---|---:|---:|---:|
| 2027896586, 0 | 139,533 | 169,480 | +29,947 |
| 1856155813, 0 | 142,327 | 190,960 | +48,633 |

## Investment timing, payback and execution

The inherited selector considers alternative economic plans and matched real-executor investment modes. It explicitly simulates a bounded public-information future, then values the remaining game with its terminal portfolio forecast. `scenario` selects the explicit rollout horizon in days. The final change reduces it from four days (up to 96 transitions) to two (up to 48); both are capped by the game's remaining transitions.

**The two-day rollout does not restrict payback accounting to two days.** The tail portfolio and marginal cash curves still extend through day 29. The inherited payback test compares with/without-investment cash flows, includes incremental wages, requires positive terminal marginal cash and recovery, and ranks viable opportunities with a capital-days penalty. Predicted cash does not become spendable budget. The payback-strength ablation did not justify weakening or strengthening the existing setting: zero, 0.25 and 2 all produced lower matched means than 1.

The explicit horizon experiment changes candidate selection and therefore the installed real executor. It is not a prediction-only change. Actual full-game trajectories demonstrate that the executor carries the selected plan into purchases, production and sales. The previously repaired legal crop-maturity admission, funded commitments, livestock service and feed handling, routing, harvesting, selling, staffing and land handling remain intact. No sale/leftover shortcut or general late-season cancellation rule was inserted into the submitted policy.

### Direct late-investment control

A separate diagnostic preserved h2's behavior before observed day 24, then prevented NEW investment decisions in both planning and execution. It retained already-funded work, incumbent maintenance and sales. All 32 replay pairs match exactly for the initial frame and first 576 transitions, immediately before the first changed day-24 decision. Thus earlier divergence is not an explanation for the result.

Suppressing new late starts **reduced terminal cash in all 32 games**, by a mean **4,183.625**. On the two critical seat-0 cells it lost 2,756 and 6,621 cash relative to h2. This rejects the blanket cancellation tested here. It does not prove every individual investment optimal, nor does it isolate each asset's return from downstream labor, routing or other decisions.

A post-freeze cash-path audit of these already completed paired games shows that retaining late starts initially put actual cash below the cutoff branch in all 32 cells, and that every cell recovered before terminal time. The median maximum actual cash deficit was **13,032**. Full step-by-step suffix cash paths, first and last recovery steps, and terminal differences are in `audit/LATE_ACTUAL_CASH_PATHS.json`. These are observed whole-policy cash differences, not predicted receipts or liquidation values.

### What the low-cash replays do and do not establish

The two supplied C01 replays and their matched parent replays were independently replayed with the frozen interpreter. Logging wrappers called the original market functions once and recorded successful and failed attempts separately; the cash reconciliation used only successful actual fills. Starting cash plus successful sales minus successful purchases and hires reconciled to terminal cash in all four cases. Physical planting and placement were checked against resulting tiles.

The first baseline case had 29,782 less milk revenue and 13,011 less strawberry revenue than its supplied parent; the second had 49,710 less strawberry revenue. These observations justify investigating plan composition and investment timing. They do **not** establish a cause by themselves: different upstream decisions also change work, inputs, prices and feasible market capacity. No leftover goods or daily cash differences were assumed sellable or treated as foregone revenue without a feasible execution test. The chosen fix is supported by the matched full-game horizon experiment, not by assigning all these revenue gaps to one apparent bottleneck.

## Fixed untouched author validation

The panel was declared at **2026-09-13T19:09:40.061339+00:00**, before candidate outcomes. The source/configuration/native runtime were frozen at **2026-09-13T19:20:03.512981+00:00**. The first validation game started at **2026-09-13T19:20:10.089444+00:00**; all 32 completed by **2026-09-13T19:20:56.843655+00:00**. The exact declared seeds and both seats were used once, with no additions, omissions, retuning or replacement games after looking at results.

The validation used the actual public **`main.py:agent(observation, configuration)`** entrypoint. Before it, one disclosed development game passed full 720-frame equality (initial plus 719 transitions) between the rebuilt final runtime and the selected development candidate. This equivalence game is logged separately and is not included in the validation mean. The final native binary matches the locally rebuilt baseline binary because the selected policy change is configuration-only.

| Measure | Value |
|---|---:|
| Complete validation games | 32 / 32 |
| Mean terminal cash | 209,593.875 |
| Seat 0 mean | 204,802.1875 |
| Seat 1 mean | 214,385.5625 |
| Median terminal cash | 212,944.5 |
| Minimum / maximum | 157,065 / 257,364 |
| Validation transitions | 23,008 |
| Game failures / unstarted games | 0 / 0 |
| Runtime and harness unchanged after validation | Yes |

| Declared seed | Seat 0 cash | Seat 1 cash | Seed-pair mean |
|---|---:|---:|---:|
| 345370069 | 194,070 | 215,867 | 204,968.5 |
| 1978370605 | 195,366 | 213,060 | 204,213 |
| 382902571 | 257,126 | 257,364 | 257,245 |
| 2142229822 | 224,252 | 244,895 | 234,573.5 |
| 101941154 | 212,829 | 210,113 | 211,471 |
| 886203829 | 223,888 | 213,302 | 218,595 |
| 1006236268 | 176,455 | 181,640 | 179,047.5 |
| 1294252994 | 199,991 | 220,486 | 210,238.5 |
| 893554823 | 201,994 | 201,785 | 201,889.5 |
| 363433203 | 157,065 | 180,586 | 168,825.5 |
| 1134682426 | 179,196 | 179,196 | 179,196 |
| 1706130316 | 187,680 | 229,829 | 208,754.5 |
| 1160145671 | 214,967 | 218,186 | 216,576.5 |
| 667109132 | 200,768 | 201,855 | 201,311.5 |
| 1187973481 | 213,546 | 224,363 | 218,954.5 |
| 546853526 | 237,642 | 237,642 | 237,642 |

Both seats on one seed are related observations; 32 games are not 32 independent random seeds. The 16-seed mean is above target, but it is not a statistical guarantee that another seed panel will pass. All validation seeds were excluded from the supplied known-used list and the disclosed development list. The declaration explicitly notes that lost round-1 histories cannot prove these seeds were absent from every unknown historical run. They were untouched within this revision round and were not used for policy selection.

## Resource, build and runtime receipts

Measured environment: Linux x86-64, Python 3.13.5, g++ (Debian 14.2.0-19) 14.2.0. CPU affinity permits five logical CPUs, but cgroup `cpu.max=400000 100000` limits execution to **four CPU cores**. Hard memory limit is **4 GiB** (`4294967296` bytes), not host-reported memory. Four game workers were used.

The initial native build took **26.772 seconds**. The final native build took **25.004 seconds**; native build plus 620 native regression checks took **29.888 seconds**. `/usr/bin/time -v` records build/test maximum resident set size of 570,744 KiB. The added public-entrypoint test passed three nonterminal checks: repeated step-0 reset after advancement, independent seat contexts, and idempotent close. Nonterminal unit work is not counted as a full game.

Validation completed in **46.764 seconds** wall time for the four-worker panel, with mean **5.769 seconds** per logged full game. The maximum measured validation action took **0.348969 seconds**, below the 5-second action limit. The full-game wall limit was 120 seconds. The largest validation worker process lifetime RSS was **97,712 KiB**, including its harness and serialization work. Container memory high-water is separately recorded and is not presented as policy-only usage.

## Coverage, failures and independent verification

There are **290 current-round full-game attempts**, all complete: eight 32-game development/diagnostic panels, 32 frozen validation games, one baseline speed game and one rebuilt-entrypoint equivalence game. They record **208,510 transitions**. The development/diagnostic repeats are deliberate paired tests, not extra fresh seeds. All current-round source-before/source-after hashes agree. Every game resets strategy state and receives only its own observation with no hidden seed/configuration exposed. The opposite seat executes the supplied legal PASS agent.

The independent verifier re-executed **all 290 recorded action sequences**, checked every recorded frame, matched trace actions to replay actions, checked the opponent's legal PASS action, verified replay/trace hashes, and confirmed exactly 719 transitions, both seats `DONE`, and reward equal to terminal farm money. It found **zero replay audit failures, zero unmatched start receipts and zero failed game attempts**. This is additional verification, not another set of 290 policy games.

All starts, results, timings, settings, full replays and per-step execution traces are included. Detailed candidate-selection debug output is retained at each dawn; ordinary step traces retain actions and execution diagnostics. `audit/FAILURES.json` is an explicit empty failure inventory rather than an absent failure log. Operational inspection/time-wrapper notes are preserved separately. Original feedback and round-1 artifacts remain intact under `input/` and `provenance/`; their historical records are not relabeled as newly executed evidence. No missing historical records were fabricated.

## Files and reproduction

The ZIP root is the runnable strategy. `submission/` is a second byte-identical frozen snapshot for audit provenance. `baseline/` and `variants/` contain the exact tested sources. `input/` contains the unmodified feedback package contents, including the supplied referee, 32 results, and four critical full replays. `runs/` contains every current-round attempt. `research/` contains evaluation, experiment, diagnostic and audit programs. `SOURCE_CHANGES.diff` and `SOURCE_HASHES.json` identify the precise source delta; `PACKAGE_MANIFEST.json` hashes every other ZIP entry.

See `README.md` for offline build, reset, reproduction and package verification commands. The included Linux native library uses only normal system C/C++ libraries. On a different ABI, rebuild from the included source; no network, model download, external runtime assets or evaluation records are needed by the policy.

**The next qualification remains the controller's fresh shared seed panel.** The source remains frozen regardless of that future result.
