# Route and procurement repair: 3,150 live matches

Date: 2026-09-11. Publication status: research handoff, **not a competition promotion**.

The local repair improves execution accounting but regresses against the original JointAFS R2 submission. Complete workflow and old recovery remain frozen comparison agents. The published runtime is the same binary used in the formal evaluation; preparing this handoff did not tune policy parameters or run a new tournament.

## Results and protocol

The existing compiled P16 C++ simulator advanced every match, with original live opponent programs. Each agent received only its own observation; no seed, hidden opponent inventory or future replay actions were provided. Each seed was played from both seats.

| Panel | Complete workflow | Old recovery | Latest repair |
|---|---:|---:|---:|
| 13 historical opponents, 25 fresh seeds | 421/650 (64.77%) | 417/650 (64.15%) | 440/650 (67.69%) |
| Original JointAFS R1, 100 fresh seeds | 101/200 (50.50%) | 102/200 (51.00%) | 99/200 (49.50%) |
| Original JointAFS R2, same 100 seeds | 122/200 (61.00%) | 125/200 (62.50%) | 97/200 (48.50%) |

There are no draws. Wins are divided by all games; a draw would remain in the denominator. Do not merge panels with different opponent/seed counts into a general strength estimate.

| Panel | Repair relative to | Win-rate change, percentage points | 95% interval | Mean own-cash change |
|---|---|---:|---:|---:|
| Historical 13 | Workflow | +2.92 | [-3.85, +10.31] | -1,212.82 |
| Historical 13 | Recovery | +3.54 | [-3.08, +10.92] | -1,401.98 |
| Original R1 | Workflow | -1.00 | [-9.50, +8.00] | +57.16 |
| Original R1 | Recovery | -1.50 | [-10.00, +6.50] | +566.30 |
| Original R2 | Workflow | -12.50 | [-22.00, -3.00] | +13.94 |
| Original R2 | Recovery | -14.00 | [-23.50, -4.50] | +596.95 |

Intervals use 10,000 paired bootstrap resamples clustered by seed, retaining seats and opponents together. Opponents include related policies, so matches are not independent samples of all possible strategies. The historical and R1 intervals include zero; R2 shows a clear local-panel regression.

## Per-opponent local win counts

| Opponent | Workflow | Recovery | Repair |
|---|---:|---:|---:|
| ahmed_v27 | 30/50 | 30/50 | 30/50 |
| aurax_reactive_v1 | 30/50 | 30/50 | 30/50 |
| aurax_shop_v2 | 32/50 | 32/50 | 34/50 |
| flexon_v5 | 30/50 | 30/50 | 34/50 |
| market_smart_v8 | 30/50 | 30/50 | 36/50 |
| moon_v215 | 30/50 | 30/50 | 31/50 |
| nagatakengo_v70 | 50/50 | 50/50 | 50/50 |
| seven_turn | 30/50 | 30/50 | 31/50 |
| shop0909 | 30/50 | 30/50 | 33/50 |
| soil_v219g | 30/50 | 30/50 | 30/50 |
| submission_56140347 | 32/50 | 31/50 | 34/50 |
| submission_56140351 | 33/50 | 30/50 | 31/50 |
| thomas_955_v2 | 34/50 | 34/50 | 36/50 |
| Original JointAFS R1 / 56146577 | 101/200 | 102/200 | 99/200 |
| Original JointAFS R2 / supplied archive | 122/200 | 125/200 | 97/200 |

In reverse, original R1 wins 298/600 (49.67%) against the three local agents. Original R2 wins 78/200 (39.0%) against workflow, 75/200 (37.5%) against recovery and 103/200 (51.5%) against repair: 256/600 (42.67%) in total. These totals cover only those three opponents, not the original public pool or leaderboard.

## What changed and what was verified

The repair retains unfinished recovery state, removes cash-only false alarms, prioritizes a feasible existing-asset schedule, coordinates useful work and replenishment in the same step, verifies quantitative warehouse placement and limits optional candidate search. It compares current-day cash/inventory outcomes plus a common next-day estimate. It does not retrain P16.

The recorded build passes 105 mechanism checks: 20 economics, 58 workflows and 27 recovery cases. Fourteen known-problem regression games pass frame-level checks with no invalid actions or unexplained registered omissions; they are excluded from formal rates.

Every formal game has 720 frames. Existing replay validation checks 4,536,000 bilateral observations and task metrics across the 3,150 games. A separate script independently recomputes terminal reward outcomes and paired differences. The 100 fresh seeds do not overlap 155 distinct seeds in the checked earlier protocols or the seven development seeds.

| Panel / agent | Invalid actions | Recorded unfinished tasks | Explicit deferrals | Unexplained omissions | Max seconds | Steps over 1 second |
|---|---:|---:|---:|---:|---:|---:|
| Historical / workflow | 0 | 0 | 0 | 0 | 2.407 | 416 |
| Historical / recovery | 0 | 368 | 168 | 200 | 1.866 | 369 |
| Historical / repair | 0 | 264 | 264 | 0 | 0.684 | 0 |
| R1 / workflow | 0 | 0 | 0 | 0 | 2.523 | 244 |
| R1 / recovery | 0 | 6 | 6 | 0 | 2.370 | 228 |
| R1 / repair | 0 | 1,072 | 1,072 | 0 | 0.691 | 0 |
| R2 / workflow | 0 | 0 | 0 | 0 | 3.492 | 213 |
| R2 / recovery | 0 | 546 | 290 | 256 | 3.221 | 200 |
| R2 / repair | 0 | 1,891 | 1,891 | 0 | 0.692 | 0 |

Registration scope differs by version. Zero unexplained omissions does not mean every original investment was performed, and count differences do not establish a causal improvement on an identical task set. The repair explicitly defers 3,227 tasks across its 1,050 formal games.

## Established defects and remaining economic problems

### Complete workflow

The agent preserves complete routes and can react to execution mismatches. It is not entirely without cash logic. It lacks the later dedicated shortage-recovery coordinator. A confirmed capacity case checks `DROP` but misses quantitative `PLACE`: at seed 572753638, step 530, a full warehouse makes milk placement ineffective, and a same-step sale cannot free space before worker actions. Removing this ineffective action did not improve final cash or flip that game; it is not a proven dominant loss cause.

### Old procurement recovery

The original diagnostic sample contains 66 recovery events: 50 cash-only alarms and 16 actual resource shortages. The 50 next-order batches are feasible in the real C++ simulator with the real simultaneous opposing actions. That establishes false intervention at those steps, not optimality of all subsequent investment.

Funding/purchase fallback steps can pause every worker. Recovery selects the first feasible minimum-extra-worker schedule rather than economically comparing incumbent, staged funding, joint orders and deferral under a common future scenario.

A confirmed early-exit case against original R1, seed 1508212750, seat 1, defers 27 investment tasks at step 241 with cash 9 and clears recovery after a partial plan. Cash rises to 439 next step, yet recovery is off. There are 24 other unfinished registered tasks. At step 242 the solver can find a valid purchase of two fertilizer plus three hires; by steps 244/245 the window is gone. This demonstrates a missed recovery opportunity, not a demonstrated full-match win reversal.

### Latest repair

The repair fixes those execution mechanisms, but its investment pruning and short-horizon economic comparison remain weak. It protects maintenance first, tests at most four ranked investment subsets, and explicitly moves some investments to the next day. Explicit deferral can discard a profitable timing window without being an unexplained accounting omission.

Against original R2, relative to workflow, average repair wages rise 44.38, own cash rises 13.94, opponent cash rises 1,313.71 and margin falls 1,299.77. Against recovery, own cash rises 596.95 but margin falls 1,308.79. The competitive consequence is worse despite superficially improved own cash.

The six selected largest win-to-loss pairs cover four seeds and first diverge at day 8, step 169 after equal observations. All have zero optional-budget stops at that step. Five immediately defer 44 tasks; one buys seeds and hires in the same step without deferral. They are targeted diagnostics, not a representative failure-frequency sample.

- Seed 3966297640, seat 0, relative to workflow: the old action hires three workers; repair moves/picks up, sells wool and wheat and defers 44 tasks. Final margin changes from +15,092 to -4,284 and wages rise 487.
- Seed 505640863, seat 1, relative to recovery: the old action sells four wool and pauses workers; repair picks up wheat, sells one wool, buys seven wheat seeds and hires four. No deferral is registered, but margin changes from +11,288 to -6,697.
- Historical seed 2166071744 against original P16, seat 1: first difference is day 10, step 216. Optional search is cut short and wheat purchase changes from seven to eight. No procurement recovery or omission occurs, but margin changes from +24,205 to -3,171.
- Original R1 seed 2351551869, seat 0: first difference is day 8, step 169 without a budget cutoff. Repair defers 42 investment tasks, saves 576 wages and earns 4,045 more cash, but the opponent earns 28,941 more; margin changes from +9,672 to -15,224.

These identify decision paths for investigation. Whole-match differences include later decisions, environment and opponent feedback; none is attributed entirely to a single purchase or deferral. Multiple changes were combined, and no feature-by-feature strength ablation has been completed.

## Timing scope

The optional budget is about 0.65 seconds of process CPU time, not a total wall-clock limit. The host records time but does not apply Kaggle timeout forfeits. Candidate selection near the budget can vary with load.

Ten serial timing reruns were selected separately and excluded from win rates. One earlier historical repair result flips from a loss to a win under a different load; the other outcomes remain unchanged, though cash can differ. Repair's selected serial maxima are approximately 0.643, 0.654 and 0.655 seconds. Old recovery still reaches 1.275 seconds in a serial R2-panel sample; workflow reaches 1.752 seconds in another.

Original R1's batch peak is 2.903 seconds with 188 steps over one second. Original R2's is 2.484 seconds with 163 such steps. Re-running R2's slowest batch game serially yields a 2.660-second opponent decision and one step over one second. This is a measured local slow step, not proof of an online forfeit.

## Identity clarification

See [IDENTITY.json](IDENTITY.json) and the acquisition receipts. Repository state `1a0d9c3c7746533222b9e745275dfa14b7cf7bef` was checked against the remote branch during identity verification.

- `latest_20260911_p16_jointafs_r1/public_submission/AFS/submission.tar.gz` is byte-identical to tested original R1, archive SHA-256 `c421a416b538df4eb4ad2da2d7598504c52dedd691537ab6fc246c3dec6c8d8d`.
- The supplied `submission (6).tar.gz` is a different archive, SHA-256 `363101251f64cd1967c0203812d30782c57daf39c117e7852811b26b8f6db804`, with entry `P16 JointAFS R2 LocalExchange`. Its submission-ID association is contextual, not an independently authenticated server hash.
- The default `latest_20260911_afs_workflow_repair_r1_r2/r2/policy/agent.so` is a separate workflow-repaired build and was not in this local 100-seed panel. Its repository-authored 1,100-game results are a different experiment. A different binary hash alone does not establish every code or behavioral difference.

Original R2's specific internal failures have not received the same exhaustive cash/recovery audit as the local variants. Do not transfer their bugs or omission guarantees to it merely because the agents share P16 ancestry.

## Shipped evidence and verification

- `RESULTS.json`, `INDEPENDENT_VERIFY.json`: archived formal statistics and independent terminal verification.
- `runs/<panel>/PROTOCOL.json`, `rows.json.gz`, `QA.json.gz`, `SUMMARY.json`: frozen jobs, per-game records, transition-check receipts and paired statistics.
- `runs/<panel>/LOSS_REVIEW.json`, `SERIAL_TIMING.json`: selected first-divergence evidence and separate timing samples.
- `REGRESSION_EVIDENCE.json`, `runs/regression_final/`: known-problem regression checks, excluded from formal rates.
- `POOL.json`, `SUBMISSIONS.json`, `DOWNLOAD_STATUS.json`, acquisition records and `IDENTITY.json`: opponent provenance and exact identities.
- `opponents/submission_56149565/`: original supplied R2 runtime files. Original R1 is already in its repository release directory; it is not duplicated here.
- `scripts/`: original evaluation/diagnostic source snapshots, requiring the original workspace layout, host and full replay assets.
- `PACKAGE_MANIFEST.json`: stored hashes and, for compressed JSON, original uncompressed hashes.

Run the portable verifier from the repository root:

```bash
python3 evidence/r2p16_route_repair_20260911/verify.py
```

This checks packaged bytes, source/build identities, seed/job pairing and recorded outcomes. It does **not** rerun the complete transition audit: the roughly 3.8 GB original experiment, including full replays, remains in the source workspace at `experiments/r2p16_route_repair_20260911`. Replay/audit hashes in the shipped rows identify those retained files. Serial reruns and development games are not silently added to win rates.

The next algorithmic experiment should compare staged funding that preserves investment, same-step replenishment and deferral over the same multi-day horizon, retain retry conditions for viable deferred projects, and ablate search-budget versus recovery changes on fresh frozen panels.
