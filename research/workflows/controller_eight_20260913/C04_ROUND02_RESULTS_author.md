# C04 round 2: complete source and economic qualification records

## Result: author qualification FAILED

The frozen candidate's mean terminal cash was **195,431.62500** across the complete predeclared panel: **16 seeds, both seats, 32 games against legal PASS**. The required mean was 200,000. The shortfall was **4,568.37500**. No validation seed was added, no failed game was excluded, and source was not changed after validation began.

On the same validation panel, the unchanged parent averaged **198,383.53125**. The candidate's paired mean difference was **-2,951.90625**. It improved 15 game pairs and worsened 17. The development improvement did not transfer to a higher mean on this particular holdout. These 16 seed blocks are a small author panel, not a claim of statistical superiority or a substitute for the controller's new shared panel.

The candidate is delivered unchanged despite this result. It is a complete runnable checkpoint, **not a qualified >=200,000 result**, not a round-robin result, and not a claim of >85% competitive acceptance.

## Identity and fixed protocol

Round start: 2026-09-13T19:09:34.106Z. Absolute cutoff: 2026-09-13T19:54:34.106Z. Source freeze: **2026-09-13T19:29:34.371166+00:00**. Packaging report generated: 2026-09-13T19:36:19.000040+00:00. Research/source selection ended at the freeze, more than ten minutes before the cutoff.

The validation declaration was written at 2026-09-13T19:10:20.132801+00:00, before any round-2 game results. The 16 draws exclude all 836 recoverable used seeds plus the stated smoke exclusion. Lost histories cannot be reconstructed, so absence from unknown lost histories is not claimed. The declaration, its frozen copy, panel start records and all results survive under research/audit and research/experiments. No resampling or post-validation fallback selection was performed.

Selected implementation: **regret_preview_cached**. Native SHA-256: `895d81d6b74847715cc86d7e09f8b8ffac83372155463c055eab7d7d396ce3cc`. Main entry, Python observation codec, economic configuration and compiler flags are byte-identical to the supplied baseline. Two existing production files change: `policy/triad.hpp` and `policy/executor/policy.hpp`. Four new offline test files are also frozen. `SOURCE_HASHES.json`, `IDENTITY.json` and `research/audit/production_changes.diff` give exact identities and edits.

## Complete strategy and the revision

The inherited strategy remains responsible for the entire game. It values crop and animal lifecycles using a conditional supply/demand forecast; protects paid commitments; allocates maintenance; buys seeds, animals, feed and land subject to observed finances; estimates labor; compiles pickup, travel, service, harvest and delivery routes; and rechecks execution and receipts on later observations. Its bounded economic/execution candidate comparisons and sale-risk mechanism remain present. Forecast cash is not recorded as terminal cash.

The first change enables the same complete-job regret-insertion fallback in worker sizing and route packing. The heuristic prioritizes jobs with fewer good placement alternatives. It accepts its alternative only when **all input jobs fit** and it either eliminates unassigned jobs or improves total/peak route cost. It does not deliberately drop a job merely to quote fewer workers. Worker starts, travel, operation duration, pickup types and applicable return/drop deadlines remain part of route cost. This is a heuristic, not a proof of globally minimal labor.

The second change handles a specific conservative preview failure. If the investment preview still has unassigned work, it invokes the executor's existing bounded exact tour/two-worker assignment refinement before rejecting project starts. The refinement uses the projected results of our own paid orders, not hidden future events or rival private inventory. Actual observations still reconcile conditional fills before execution. It does not enable the separate one-fewer-worker DP ablation used in earlier rejected trials.

A synthetic current-day fixture has nine startup jobs and 23 remaining ticks. Regret insertion alone leaves one unassigned. The old exploratory assertion that regret alone could fit all nine failed and is retained. The revised preview, compiled route and current-day execution each complete all nine starts. This is an execution-feasibility check, **not a terminal game or nine-job cash counterfactual**.

Exact-input memoization reduces repeated scheduling work within one observation. The key covers every Job field, worker starts, time budget, return requirement and algorithm mode; cache ownership is call-local. On all 32 development games, caching preserved all 23,008 actions/transitions and all recorded states. The unresolved-preview repair also preserved those 32 games exactly. Consequently, the observed development cash gain comes from the matched-regret variant; no extra real-game profit is attributed to the preview guard or cache on that panel.

## Development attempts, not holdout qualification

The controller's disclosed 32-game baseline mean was 195,548.00000. Four local baseline games reproduced both disclosed lowest-cash cases in both seats; the two low seat-0 cash values were 131,126 and 155,189. Original critical and matched-parent replays are retained unchanged.

| Attempt group | Completed games | Mean terminal cash |
|---|---:|---:|
| baseline_critical | 4 | 164508.50000 |
| fleet_hire_critical | 4 | 197999.75000 |
| fleet_hire_development | 32 | 196458.25000 |
| fleet_contract_development | 32 | 198414.65625 |
| regret_matched_development | 32 | 205430.40625 |
| regret_cached_development | 32 | 205430.40625 |
| regret_preview_cached_development | 32 | 205430.40625 |

The two four-game groups above reuse disclosed critical seeds. They are diagnostic pilots, not independent validation panels. Every 32-game development group uses the same disclosed 16 seeds and both seats. The matched-regret group gained 9,882.40625 over the disclosed baseline mean, with 21 improved and 11 worse cells. The cache and preview variants tie it exactly in actions and cash. The tie was resolved **before validation** in favor of the tested preview guard; the decision is recorded in `research/audit/selection_tiebreak.json`.

The early baseline and later buildable checkpoints, all variants, all source hashes, build attempts, complete game replays and debug/contract traces are retained. No parameter search or omitted failed game is hidden behind the selected mean.

## Validation and evidence checks

Every validation game has exactly **719 transitions**, both players **DONE**, and both terminal rewards equal the corresponding farm cash. The candidate and matched baseline each completed all 32 games, for 46,016 validation transitions. All validation pairs are listed in `VALIDATION_RESULTS.csv`; raw observations, actions, terminal checks, timings and replay hashes are in `research/experiments/validation_candidate` and `validation_baseline`.

Across the entire round there are **236 game attempts, 236 completed games, 0 game failures, and 169,684 recorded transitions**. These include repeated development, cache-equivalence, baseline and entrypoint checks; they are not 236 independent holdout games. The final replay audit independently replays every recorded action through the frozen interpreter and compares both players' state fields at every transition. All 236 audited games passed. Replay audits do not add new policy games.

The actual root `main.py:agent` was exercised for four full games in a reused module, including a repeated same-seed/same-seat game and a seat switch. Every one of the 2,876 action sequences matched the fresh-context development reference, with terminal checks. New step-0 observations reset state. `create_agent()` returns independent handles with explicit `close()`; `main.reset()` closes root seat contexts.

The final source's three offline C++ suites passed. They cover 250 generated route cases (247 complete schedules, with deadline/material/semantic-job checks), 250 exact-cache equality cases and 15 key-field mutations, and the nine-job preview/dispatch fixture. All original failed exploratory tests remain under `research/audit`, including the failed regret-only nine-job assertion. The initial unsupported streaming-tool call and a receipt-inspection command that ran before a build finished are recorded separately in `research/audit/tool_failures.jsonl`.

A clean copy rebuilt with the same compiler to a **byte-identical native** and passed the 44-setting ABI, independent-handle and close checks. The selected source/native and baseline source remained unchanged since freeze. All 73 original feedback-manifest entries were reverified before packaging.

## Resources, timing and runtime limits

Measured CPU quota: **4 cores** (`cpu.max=400000 100000`); affinity exposes five logical CPUs but was not treated as five usable cores. Memory limit: **4 GiB**. Compiler: GCC 14.2.0; Python: 3.13.5. Initial baseline build: 38.429 seconds; selected build: 54.269 seconds; clean rebuild: 37.554 seconds. The initial full-game measurements were about 5.3–5.7 seconds with complete records, rather than assuming the controller's faster environment.

Candidate validation took 190.944 seconds of runner wall time with two workers; matched baseline validation took 239.549 seconds with one worker. These are not like-for-like speed comparisons. The largest recorded candidate-validation action was 1.031797 seconds; the largest per-game sum of time above one second per action was 0.031797 seconds.

**LocalGame is not Kaggle's sandbox, timeout or full schema validator.** The frozen specification has actTimeout=1 and the local initial observation includes 60 seconds of overage, but this host does not decrement/enforce that overage. Some actions exceeded one second. Recorded times are supplied; Kaggle timeout/platform compatibility is not certified. The Linux x86-64 shared library requires compatible system libraries or an offline rebuild. `RUNTIME_AND_RESOURCES.json` and `research/audit/native_platform.txt` contain measured details.

## Source boundaries and remaining limitations

The supplied archive contains all 32 controller result rows but only two distinct full critical replays, plus byte-identical matched-parent copies. This delivery preserves what was supplied and adds every new full replay; it does not invent the other historical full replays. Earlier C04 round-1 research records were lost and are not claimed recovered. The original AUTHOR_INPUTS archive and surviving inventory are included unchanged. Inherited READMEs and receipts under research describe their original work, not tests newly conducted here; referenced historical test sources that were not supplied remain missing.

The two lowest-cash baseline games ended without positive unharvested yield. Empty late-day plans, idle actions, daily cash differences or leftover goods were not treated as proof of recoverable profit. Full closed-loop tests, rather than replay-derived alternative actions, determine every reported new terminal score. The new policy receives only its own private observation and public fields; seeds and full replays belong to the driver/audit files, not policy inputs. No remote execution, delegation or publication was used.

The held-out economic threshold was missed and the paired mean was lower than the unchanged baseline. No causal explanation for that regression is asserted without a further controlled test. There was no round robin, no 1,536-game competitive acceptance run and no claim that this checkpoint reaches the competitive strict-win requirement.
