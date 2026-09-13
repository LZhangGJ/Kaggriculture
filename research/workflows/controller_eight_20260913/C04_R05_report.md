# C04 ROUND05 final report

Prepared 2026-09-13T22:02:09.147649+00:00. Research was selected/frozen before all untouched validation. See the standalone delivery receipt for the final archive sealing time.

## Decision and result

**Economic qualification was not achieved on the author validation panel.** The frozen capital_rolling candidate returned 193,869.06250 mean terminal cash across all 32 predeclared candidate cells. Matched current baseline: 198,113.34375; paired change: -4,244.28125. No post-validation change, additional validation seed, or fallback selection occurred. The frozen candidate is delivered even though it failed. PASS cash is not competitive acceptance and this author panel does not replace the controller's future shared panel.

The selected candidate keeps C04's complete rolling planner and executor, changing capital_power from 0.4 to 0.7 and discount from 0.005 to 0.08. Both values are attributed to the supplied C08 reference. C08's complete source was not renamed or copied. The independent cash-exposure implementation was tested and rejected, with all its source and results preserved.

## Input provenance and identity

The input ZIP was 5,903,648 bytes with SHA-256 0b2d4f22a2475d5b91c21d231a3778612ecebde0217343090572aa132a9bdad1. Every one of 165 manifest entries was checked before use. All 70 supplied C04 and 75 C08 files matched the controller source identity manifests. The referee stayed unchanged; exact hashes and original supplied critical replays are retained under input/ and receipts/.

The controller's current baseline mean was 199,943.75 over 32 games. Older supplied panels used an older C04 configuration. We therefore reran the CURRENT baseline on those 64 cells rather than labeling older cash as matched results. The current panel's 32 baseline values are independently audited supplied records, with two cells reproduced locally. All revised-policy development outcomes and all 64 author-validation candidate/control games were executed here.

ROUND02's original revised source and full evidence remain lost. Its reported failed 195,431.625 validation is an inherited historical fact, not reconstructed evidence. New seed uniqueness is relative to the 1,062 recoverable IDs; unknowable lost histories cannot be certified.

## Complete strategy and execution

At dawn, the policy forecasts public demand and visible opposing production, assigns values to crops and livestock, retains existing commitments, and selects maintenance and optional investments. Crop rotation and animal service models use conditional prices, not future realized randomness. The nonzero scenario setting retains the inherited bounded one-day public-scenario proposal search and continuation estimate. Investment ranking divides discounted modeled contribution by capital raised to capital_power. The accepted values make delayed, capital-heavy projects less attractive, without changing actual wages or physical travel time.

The execution controller budgets actual hiring costs, checks cash and inventory, prepares market orders, reserves planting/placement inputs, assigns maintenance/harvest/start jobs to finite depot-aware worker routes, previews feasible starts, and recompiles against the current observation. All existing financial and route feasibility gates remain unchanged. work_price=1 is an internal shadow penalty, not the official wage. harvest_threshold=3, competition=0.5, max_animals=20, scenario=1, opening prior, maturity logic and terminal behavior remain C04's. Retaining gates is not a proof of globally optimal workforce allocation or platform timeout compliance.

Deployment: extract the separate final source/runtime ZIP and import main.agent(observation, configuration). Linux x86-64 native runtime is included. Offline rebuild: python3 -B build.py --cxx g++. No network, GPU, training dataset or third-party Python package is needed by the policy. Evidence harness replay serialization uses orjson; this is not a policy runtime dependency. main.create_agent() / close() are available for explicit local game lifecycles. Seeds are used only by LocalGame; configuration.seed remains None and no policy observation receives the seed.

## Bounded experiment sequence and failures

H1 direct: change scenario 1 to 0 only, attributed to C08. Current-panel gain +6,138.78125 looked promising but the two earlier panels averaged -10,360.53125 versus current baseline. Rejected.

H2 exposure: independently implemented maximum modeled cumulative cash deficit, including feed/input flows and incremental portfolio wages, in place of acquisition-only capital. It keeps the same exponent, time preference and physical executor. Five synthetic accounting cases passed, but forecasts do not prove actual future workforce feasibility. It improved its direct parent by +1,842.96875 over pooled development while still losing -3,017.79167 to current baseline; current-panel difference versus direct was -5,998.06250. Rejected. Daily cash, leftover crops and modeled workload were treated as associations, not proved recoverable profit. Complete helper, integration diff, instrumentation and tests remain in variants/exposure/.

H3 capital_reference: direct planning with C08's fixed financial settings (.7 / .08), without its other changes. Rejected, pooled difference -1,815.76042.

H4 capital_rolling: apply the same financial settings while retaining C04 rolling planning. Added after the direct-planning prior-panel defect was measured. This completes the bounded 2x2 planning-mode/financial-calibration comparison; no numeric sweep was used. Pooled development gain was just +161.59375 (about 0.079%), with large losses. It satisfied the declared selection rule but did not establish reliable superiority.

| Current C04 or variant | Disclosed panel 1 mean | Panel 2 mean | Current panel 3 mean | Pooled 96-cell mean |
|---|---:|---:|---:|---:|
| Current baseline | 209733.40625 | 203628.21875 | 199943.75000 | 204435.12500 |
| Direct | 198595.31250 | 194045.25000 | 206082.53125 | 199574.36458 |
| Independent exposure | 204401.71875 | 199765.81250 | 200084.46875 | 201417.33333 |
| Financial settings + direct | 197135.40625 | 205147.56250 | 205575.12500 | 202619.36458 |
| Selected financial settings + rolling | 200134.46875 | 205008.59375 | 208647.09375 | 204596.71875 |

Selection required all 96 development cells complete, pooled gain over baseline and positive panel-mean differences on at least two of three panels. Among eligible candidates the highest pooled mean was chosen. Only H4 qualified. Its development cells split 48 gains / 48 losses, 24 of 48 seed-pair differences positive; its largest loss was 88,492 cash at seed 320211102 seat 0. All development scores are selection-biased.

## Fixed author validation

Sixteen seed IDs were declared at 2026-09-13T21:36:10.073978+00:00, before any new outcomes, excluding all 1,062 known IDs. Source selection: 2026-09-13T21:54:16.618634+00:00; immutable freeze: 2026-09-13T21:55:25.439311+00:00. Validation started 2026-09-13T21:55:51.822377+00:00 and retained all 32 candidate cells plus 32 matched baseline cells. No seed was added and no fallback was selected afterward.

Candidate mean: **193,869.06250**; baseline mean: **198,113.34375**; paired change: **-4,244.28125**. Positive/negative/tied cells: 12 / 20 / 0. Positive seed pairs: 6 of 16. Full per-cell cash, status, timings, terminal checks and replay hashes are in analysis/VALIDATION_RESULTS.csv. Worst paired cell: {"baseline_cash": 215724.0, "baseline_error": null, "baseline_replay": "games/fixed_validation__baseline__1743809225__1/replay.jsonl.gz", "baseline_replay_sha256": "eb1f11f52e20047181d4b10b6099368cba9580894e777aa19fd41f3944105749", "baseline_reward_cash_agreement": true, "baseline_seconds": 4.883345182000085, "baseline_statuses": "DONE|DONE", "baseline_steps": 719, "baseline_valid": true, "candidate_cash": 162627.0, "candidate_error": null, "candidate_replay": "games/fixed_validation__final__1743809225__1/replay.jsonl.gz", "candidate_replay_sha256": "1532a875944782abef2b8fc39d853ae3d447db424ed8baec1498cd91bbf0585e", "candidate_reward_cash_agreement": true, "candidate_seconds": 4.730729445999941, "candidate_statuses": "DONE|DONE", "candidate_steps": 719, "candidate_valid": true, "delta": -53097.0, "seat": 1, "seed": 1743809225}.

The validation result is the relevant new generalization check; the tiny positive development mean does not override its failure. No claim is made that changing financial preferences repairs all worker-capacity weaknesses.

## Build, runtime and replay checks

Measured cgroup quota: four CPU cores (400000/100000), affinity five logical CPUs; memory limit 4 GiB. Compiler GCC 14.2.0; Python 3.13.5. At most two concurrent game/audit workers. Initial build: 40.88 seconds, 570,388 KiB peak compiler RSS. Initial complete games: 7.78345, 5.59940 seconds. Replay-serialization optimization was separately recorded; timing scopes are not silently mixed. Clean final rebuild: 48.50893 seconds and byte-identical to the development native library.

There are 518 newly attempted games, 518 valid and 0 failed/incomplete, covering 372,442 recorded transitions and 64 distinct seed IDs. Repeated development, smoke and reset checks are not independent new seeds. Each valid completed game has 719 transitions, both statuses DONE and reward equal to terminal cash. 518 of 518 audited attempts passed; 0 failed. 372,442 transitions were replayed. Replay audit reapplies every recorded action to a fresh frozen official interpreter and compares entire states, not only terminal totals.

Eight final API checks passed, including native-context independence, settings ABI, reset and invalid configuration rejection. Four full games through the actual root main.agent entrypoint reused the module and matched all action/state hashes from the selected development policy. Five synthetic exposure tests passed; those tests concern a rejected variant, not economic success. Existing test binaries/receipts in supplied runtime directories are historical unless explicitly listed as newly executed.

Highest observed action latency was 1.741712 seconds. The local harness uses a five-second action guard; Kaggle sandbox/time-limit compliance was NOT certified. Game memory readings are worker-process high-water marks. The attempt to use an unavailable interactive container session failed before any build started; a noninteractive build succeeded. This tooling error is retained. An initial report-packaging pass also failed on a missing duration-field name; its exact script, logs and traceback are preserved, and only the packaging script was repaired. All known technical and economic failures are in FAILURES.json. The initial packaging count warning was reconciled: 514 ordinary planned tasks plus four separately declared root-entrypoint cells equal 518 recorded attempts. No game was missing or added. The original packaging receipt is retained. Additional aggregation warnings: [].

## Delivery layout and preservation

First complete checkpoint was saved and linked at 21:36:10 UTC, within the initial five minutes. Accepted source checkpoint was linked before validation. The separately attached final runtime ZIP is 1,661,426 bytes, SHA-256 f5787f7a9324e124d989d7abafc2d6988948695219df36d497b83852458bb601. It was sealed before validation and never replaced afterward.

Evidence ZIPs are ordinary, independently readable archives, each under 50,000,000 bytes. Extract all parts into one directory; do not concatenate them. Source variants, current provided source/reference/referee files, declared/used seeds, every attempt/result/full replay, settings, tests, rejected changes, receipts, tools and exact checkpoint bytes are retained. The original input ZIP is referenced by its exact hash rather than embedding a large inherited archive. FILE_INVENTORY.tsv hashes all staged files except itself, avoiding circular hashes. Per-part manifests hash every payload member. Standalone DELIVERY.json and SHA256SUMS.txt give final part sizes, whole-file hashes, verification and completion time.
