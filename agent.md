# Kaggriculture agent development and validation

Updated: 2026-09-11. New and revised Markdown documentation for this handoff is in English.

The [two-approach handoff](docs/agent_approaches/README.md) now documents both the
route-clustering switch agent and this P16 planning line, including the two
original JointAFS submissions and the distinct later workflow-repair builds.
Use its [P16 method guide](docs/agent_approaches/P16_METHOD.md) for the detailed
architecture and development steps. The findings below remain specific to the
local P16 route/procurement experiments.

## Current conclusion

The latest local procurement-repair candidate improves execution checks but **must not replace the competition baseline**. Across 3,150 fresh live matches, it regresses against the user-supplied JointAFS R2 submission. Its 1,050 games contain no invalid actions or unexplained omissions of registered commitments, but include 3,227 explicitly deferred tasks. Deferral is not completion.

- [Latest candidate: source, build and usage](nt/latest_20260911_r2p16_route_repair/README.md)
- [Full evaluation, failure analysis and limitations](evidence/r2p16_route_repair_20260911/REPORT.md)
- [Earlier route-economics experiment](nt/latest_20260910_r2p16_route_economics/README.md)
- [Previously published workflow/recovery evidence](evidence/r2p16_route_agents_20260911/README_ZH.md)

This publishes research progress and the exact tested runtime. It does not submit to Kaggle, retrain the economic model, or establish a strongest local agent.

<a id="r2p16-route-agents"></a>
## Version identities

| Name in results | Meaning | Entry / reference |
|---|---|---|
| Frozen P16 | Baseline before the local route experiments | [P16](nt/latest_20260910_r2p16/README_ZH.md), commit `979fec14bdddd281f4d934494aa4a06673fba9b9` |
| Route economics | Delivery, warehouse and cash-aware scheduling; later found to omit protected tasks | [Entry](nt/latest_20260910_r2p16_route_economics/main.py), `build/revision5/route3.so` |
| Complete workflow (`workflow`) | Preserves feasible schedules and complete work chains | [Entry](nt/latest_20260910_r2p16_route_workflow/main.py), `build/revision2/route3.so` |
| Old procurement recovery (`recovery`) | Reconciles purchases and reconstructs remaining work | [Entry](nt/latest_20260910_r2p16_route_recovery/main.py), `build/revision3/route3.so` |
| Latest local repair (`fixed`) | Corrects false cash alarms, premature recovery completion and capacity checks; coordinates same-step work and bounds optional search | [Entry](nt/latest_20260911_r2p16_route_repair/main.py), `build/revision4/route3.so` |
| Original JointAFS R1 | Joint animal/feed/successor planning; submission `56146577` | [Original release](nt/latest_20260911_p16_jointafs_r1/public_submission/README_ZH.md) |
| Original JointAFS R2 | User-supplied `submission (6).tar.gz`; entry identifies `R2 LocalExchange` | [Identity receipt](evidence/r2p16_route_repair_20260911/IDENTITY.json) |
| JointAFS R1/R2 workflow repairs | Separate later builds, workflow repair enabled by default | [Separate package](nt/latest_20260911_afs_workflow_repair_r1_r2/README_ZH.md) |

"Old" and "new" refer to a specific experiment. Complete workflow is not original P16. Local procurement repair is not the separately published JointAFS workflow repair.

The original R1 archive in the repository is byte-identical to the downloaded file tested locally. The supplied R2 archive SHA-256 is `363101251f64cd1967c0203812d30782c57daf39c117e7852811b26b8f6db804`; its library SHA-256 is `1ead09a9bd48b20b512fb8fe57bbbbd87c86bb12fc9b1b42553e5c5b5bec121c`. Its association with submission `56149565` uses acquisition context and the entry label, not an independently authenticated Kaggle archive hash.

The separate JointAFS R2 workflow-repair default library has SHA-256 `179b204db64a32e08af3afb34ae1e6687737c71e4d506f37c4dd3ebe0ec591f2`. It was **not** an opponent in this 100-seed comparison. Different library hashes alone do not establish every source difference; compiler and linking choices also affect bytes.

## How the local agents decide

The route variants share P16's frozen economic configuration. They use rules, dynamic programming and bounded simulation search. This work does not train a new neural model; the learned candidate-value interface is not enabled in the frozen configuration.

1. Read actual cash, inventory, seeds, workers/bags, crops, animals, shops, market state, visible opponent assets and completed trades. Do not read seeds, private opponent stock or future replay actions.
2. Generate daily economic alternatives: crop/animal allocations, working-capital reserves and competition weights. Estimate output, purchases, maintenance, labor, delivery and sales.
3. Simulate the configured short horizon and estimate remaining value. Opponent activity and future markets are forecasts. Adopt today's choice and reconsider tomorrow.
4. Compile intentions into procurement and worker actions: supplies, movement, construction, sowing, watering, feeding, harvest and delivery.
5. Issue one step and reconcile the next real observation. Lightweight checks run each step. Heavier scheduling runs at work start or relevant deviations, with candidate and trigger limits.

Daily economic planning and per-step execution coexist. These changes do not recompute the entire economic planner from scratch every step. Their effects on cash, land, stocks and sales can nevertheless change subsequent investment and opponent responses.

## Development sequence

| Stage | Established problem | General change |
|---|---|---|
| Replay/day experiments | A short boundary route need not connect useful field work | Compare complete chains on day-6/day-18 observations, then full matches |
| Route economics | Fewer moves or lower wages can delay receipts, fill warehouses or leave purchases unfunded | Include delivery time, capacity, fills and actual cash; validate a schedule before cancelling hires |
| Complete workflows | Later reordering could discard a successful trial; individual task insertion broke dependencies | Preserve/revalidate full schedules; repair sow/water and pickup/feed chains |
| Procurement recovery | Planned supplies were treated as available despite missing fills | Reconcile actual resources/workers, raise funds and reconstruct work |
| Latest repair | Cash-only false alarms, paused workers, early recovery exit and incomplete capacity checks | Check actual fills, preserve useful actions, retry after resource growth and verify quantitative placement |
| Expanded evaluation | Cleaner execution counters did not imply better investment choices | Freeze programs and report wins, relative cash, wages and explicit deferrals separately |

The route algorithm is bounded local scheduling search with resource and precedence constraints, feasible-schedule retention and step simulation. It is not a global shortest-path or minimum-hire solver. Complete dependent chains must remain executable, while feasible maintenance may still be shared among workers.

Old recovery searches for a feasible remaining schedule with minimal added hires. If necessary it sells current stock, confirms proceeds and buys later. Funding/purchase steps can pause all workers. Investigation established a cash-only false alarm and a partial fallback that could clear recovery too early. A feasible corrective next step is not proof of a whole-match reversal.

Latest repair keeps pending maintenance recovery, retries after actual increases in cash/materials/workers, and coordinates worker actions, sales, purchases and hires in official order. It protects existing assets first, then tries up to four ranked investment subsets. It uses current-day outcomes plus a common next-day estimate. Optional search receives about 0.65 seconds of process CPU time, not an absolute whole-decision wall-clock guarantee.

<a id="route-repair-20260911"></a>
## Latest frozen evaluation

All three local binaries were frozen before evaluation. Matches used live original opponents and the existing compiled P16 C++ simulator, not replay-tape opponents. Each seed was played from both seats.

| Opponent panel | Complete workflow | Old recovery | Latest repair |
|---|---:|---:|---:|
| 13 historical opponents, 25 fresh seeds each | 421/650 (64.77%) | 417/650 (64.15%) | 440/650 (67.69%) |
| Original JointAFS R1, 100 fresh seeds | 101/200 (50.50%) | 102/200 (51.00%) | 99/200 (49.50%) |
| Original JointAFS R2, the same 100 seeds | 122/200 (61.00%) | 125/200 (62.50%) | 97/200 (48.50%) |

There were no draws. Wins are divided by all games; draws would remain in the denominator. Do not merge panels with different opponent composition into a general leaderboard estimate.

Against original R2, repair loses 12.5 percentage points versus workflow (95% seed-cluster interval [-22.0, -3.0]) and 14.0 versus recovery ([-23.5, -4.5]). Historical and R1 intervals include zero. Intervals use 10,000 paired bootstrap resamples of whole seeds, retaining seats and opponents together.

Across 3,150 matches, 4,536,000 bilateral observations and task metrics passed existing transition checks. A separate script recounted terminal rewards and paired outcomes. The 100 fresh seeds do not overlap 155 distinct checked prior seeds or seven development regression seeds.

### Why execution fixes are not a strength upgrade

The repair records 1,891 explicit deferrals against original R2, yet wins only 48.5%. Relative to workflow, mean wages rise by 44.38, own cash by 13.94 and opponent cash by 1,313.71; final margin falls by 1,299.77.

Six selected large win-to-loss comparisons, covering four distinct seeds, first diverge on day 8, step 169 after identical observations. None reaches the optional-search limit at that first difference. Five defer 44 tasks; one changes same-step buying/hiring without a deferral. These cases are not a prevalence estimate, but show that time truncation cannot explain every regression.

Registration differs among versions. Zero unexplained omissions does not mean all investment intentions were executed. An explicitly deferred investment can be a bad decision even if accounting is correct.

Ten serial timing reruns remain separate from formal win rates. One earlier repair outcome changes with runtime load. The original R2 archive also reaches 2.660 seconds in a serial sample. The host records time without Kaggle timeout forfeits; cash win rates are not timing certification.

## Earlier evidence: separate panels

| Experiment | Result | Interpretation |
|---|---|---|
| P16 versus route economics: 20 seeds, 11 opponents, both seats | 324 versus 346 wins / 440 each; 12 candidate omissions in six games | Execution defect; win-rate difference interval includes zero |
| Route economics versus workflow: 30 seeds, 11 opponents, both seats | Both 564/660; omissions 12 to 0; direct workflow record 16 wins, 16 losses, 28 draws | Completeness improved, no demonstrated strength gain |
| Workflow versus recovery: original 11 opponents | Both 88/110; all paired action sequences identical | This panel does not exercise recovery enough |
| Two then-latest submissions, 25 seeds, both seats | Workflow 74/100; recovery 69/100 | Larger seed sample reversed the earlier small-panel direction |

For that last panel, workflow/recovery win 39/50 and 34/50 against `56140347`, and 35/50 each against `56140351`. These are earlier opponents, not JointAFS R1/R2. Keep acquisition dates and cohort definitions attached to results.

## Next algorithmic work

1. Compare staged fundraising that retains investment, same-step replenishment and investment deferral over a common multi-day horizon.
2. Keep retry conditions for investments that later become feasible.
3. Measure relative cash and market/opponent feedback, not just own cash and wages. P16 already has public-flow forecasts; it is not opponent-blind.
4. Reduce redundant optional search and run ablations separating recovery, capacity and budget effects.
5. Freeze candidates before new seeds and independent opponent families. Known examples are regression data, never seed/coordinate policy branches.

JointAFS's animal/feed/successor search is a separate direction. Its workflow-repaired R2 default has not entered this local comparison; its repository-authored 1,100-game results cannot be substituted for our panel.

## Reproduction and publication

- Run [verify.py](evidence/r2p16_route_repair_20260911/verify.py) with Python 3. It checks manifests, build/source identities, lossless compression and the 3,150 recorded outcomes.
- Load `.so` files on Linux x86-64 / WSL. Preserve tested binaries; rebuild to a new path with GCC 13 and recorded flags.
- Original tournament scripts are source snapshots requiring the original workspace host, opponents and replays. They are not standalone portable runners.
- Keep full replays, intermediate builds and caches on the source machine. Compress new JSON over 256 KiB and record stored and uncompressed hashes.
- Include only the small final runtime binaries needed for this handoff, with manifests. Do not duplicate archives or development builds.
- Preserve negative findings, seed pairing, opponent homogeneity, timing limits and the distinction between repository-reported and locally reproduced evidence.
- Do not rewrite Git history in a progress push. A historical audit at `bd30034` measured 705.4 MB tracked and a 952 MB Git directory; these are not current size measurements.

## Earlier KEEP=2 / F3 reinforcement-learning work

This is a different line from P16 routing. The [full historical record at db109ce](https://github.com/LZhangGJ/Kaggriculture/blob/db109ce316151929622f2612223cc942dbb844de/agent.md) remains in Git. This English index preserves its conclusions without presenting old running statuses as current measurements.

- Four 1,000-round runs did not establish sustained improvement from simply extending training. All eight 1,000-versus-900 single-panel intervals included zero; the strong `aux_r0` run was already strong at 300 rounds.
- Same-recipe training variance was large. The offline value audit did not establish learned ranking superiority over uniform non-KEEP choice. Extrapolating one-step effects to whole-match win rates was invalid, and selecting the best noisy candidate creates winner's bias.
- [E1/E2/E3 evidence](evidence/e1_e2_e3_20260907/REPORT_ZH.md): E1's six new 300-round seeds produced zero of six positive confidence intervals, median -0.43 percentage points. E2 did not establish an out-of-pool gain because its assumed in-pool gain failed to reproduce. The historical record had only two of three E3 repetitions; this publication infers no later result.
- Terminal rewards confounded small decision effects with seeds, opponents and training randomness. Proposed remedies included paired baseline rewards and features for displaced investment, working capital, price impact and relative return.
- Do not extend the old narrow F3-PPO recipe or select checkpoints after final evaluation without replicated evidence and a declared protocol. Earlier 812-game portable validation belongs to the archived KEEP=2 package, not the new route candidate.
