# P16 planning, joint investment search and reliable execution

[Both approaches](README.md) · [Publication and identities](PUBLICATION.md) · [Local evaluation report](../../evidence/r2p16_route_repair_20260911/REPORT.md)

## 1. The central idea

This approach builds a farming plan from the current board. It estimates which crops and animals are worth operating, what land and supplies they require, how much cash must remain available and whether workers can complete the necessary actions. It then carries out one step, observes what really happened and updates its execution state.

The economic planner and the worker scheduler answer different questions. The planner asks, for example, whether an additional animal and its feed supply will improve the farm's future value. The scheduler must find an executable chain that buys or retrieves the feed, reaches the animal, feeds it on time and delivers output before a profitable sale. A good investment estimate does not automatically produce a feasible action sequence.

The development described here addressed that gap in two related directions: **better joint investment decisions** in the two JointAFS submissions, and **more reliable resource-constrained execution** in the workflow and procurement experiments. These are branches of the same P16 planning line, not interchangeable names for one binary.

## 2. What the model contains, and what was trained

The released configurations combine hand-designed rules, dynamic programming, explicit game simulation and bounded candidate search. They do not train a new neural model. A learned-value interface exists in the codebase, but it is not enabled in the frozen economic configuration used by the local route variants.

The principal components are:

| Component | What it does | Representative source |
|---|---|---|
| Observation decoding | Reconstructs actual farms, inventories, workers and public markets | [observation_codec.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/observation_codec.hpp) |
| Production planner | Proposes production paths, investment allocations and future value estimates | [planner.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/planner.hpp) |
| Maintenance and animal service | Models ongoing service obligations and resource use | [ongoing_maintenance_dp.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/ongoing_maintenance_dp.hpp), [animal_service_dp.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/animal_service_dp.hpp) |
| Public opponent forecast | Uses observed trades and production timing to estimate future public flow | [public_trade_ledger.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/public_trade_ledger.hpp), [public_flow_scenario.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/public_flow_scenario.hpp), [observed_crop_clock.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/observed_crop_clock.hpp) |
| Candidate search | Simulates alternative plans and compares their economic value | [search.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/search.hpp), [proposals.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/proposals.hpp) |
| Joint animal/feed/successor planning | Searches coupled changes to animals, wheat feed plots and the crop planted afterwards | [joint_candidates.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/joint_candidates.hpp), [joint_bundle_state.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/joint_bundle_state.hpp) |
| Worker executor | Converts chosen tasks into ordered movement and work | [executor/](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/executor/) |
| Local route economics and recovery | Preserves complete schedules and coordinates capacity, funding and retries | [route_economics.hpp](../../nt/latest_20260911_r2p16_route_repair/policy/route_economics.hpp), [route_recovery.hpp](../../nt/latest_20260911_r2p16_route_repair/policy/route_recovery.hpp) |
| Optional-search limit | Bounds the latest local repair's discretionary search | [decision_budget.hpp](../../nt/latest_20260911_r2p16_route_repair/policy/decision_budget.hpp) |
| Controller and deployment | Connects planning, execution, persistent state and the native/Python interface | [triad.hpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/triad.hpp), [bridge.cpp](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/bridge.cpp) |

The table maps the architecture to inspectable implementations. The local route-repair headers belong to their own experimental package; do not assume they are all compiled into the original JointAFS submissions.

The shared local-route economic configuration has SHA-256 `65efd0f5822d40a86440f1837640a54ee1ea60b8f8b61e3be99da2d97591f72c`. Freezing it isolates execution changes more effectively than simultaneously changing the planner parameters. It does not make the resulting full-game trajectories identical: execution changes alter future assets, cash and opponent responses.

## 3. Daily planning and step-by-step control

At a planning event, the agent reads actual cash, seeds, saleable stock, workers and bags, land, crop stages, animals, shops and market conditions. It also reads public opponent information. It does not inspect hidden match seeds, private opponent stock or future actions from the current replay.

The baseline economic search normally uses a short configured rollout, commonly one day, together with an estimate of the remaining economic value. Candidate changes apply to the current day's choice; later rollout policy follows a common continuation. The joint-investment search described below uses a longer common horizon for its candidates.

Dynamic programming generates and values production alternatives subject to the resources represented by the planner. Search compares alternatives using simulated outcomes and remaining value. It can account for opponent production and competition through public-flow forecasts; it is not opponent-blind. Nevertheless, the forecast is an estimate, not knowledge of the opponent's next orders.

The chosen economic intentions become work obligations: procure supplies, move or retrieve them, construct where needed, plant and water, service animals, harvest, carry output and sell. Every step then returns an action for each worker and the relevant market/hiring orders.

The next observation is essential. An issued purchase may not fill; a sowing action may fail; a warehouse may lack space; cash may arrive later than expected. The executor must compare actual effects against its expectations before treating the next task as feasible.

**The agent emits actions every step, while the heavier economic plan is refreshed daily or at relevant triggers.** Lightweight reconciliation and bounded rescheduling coexist with that daily plan. The latest experiments do not rerun an unlimited whole-farm optimizer from scratch on every step.

## 4. From shorter routes to complete work chains

The original route question was sensible: a worker who enters a field early can perform useful tasks along the way, whereas a visually short path along the boundary may miss opportunities. But shortest walking distance alone is the wrong objective.

Consider a worker carrying produce. A route with fewer moves may delay delivery until after the market phase that would have funded seed purchases. Another route may save one hired worker but leave an animal unfed. A third may reach storage when preceding workers have already filled it. In each case, geometric savings can reduce profit.

The local algorithm therefore searches a bounded set of schedules under resource and precedence constraints. It evaluates complete task chains by step simulation and retains a feasible schedule as a fallback. It is a local scheduling method, not a proof of globally shortest paths or minimum hiring for every board.

The necessary constraints include:

- **Precedence:** obtain seed before planting; complete dependent watering or pickup/feed tasks in a valid order.
- **Resources over time:** use actual cash and filled purchases at the point when they become available.
- **Capacity:** account for carried quantities and storage availability in worker execution order.
- **Deadlines and commitments:** preserve work promised to existing crops and animals, and distinguish deferred investment from completed work.
- **Competition:** value receipts and sales under market and opponent forecasts rather than only counting moves or wages.

A complete feasible trial must survive later route reordering. Finding a valid schedule and then rebuilding part of it without its dependencies can recreate the original defect. The workflow variants keep the witness schedule—the concrete sequence showing that the commitment can be completed—and revalidate it when the real state changes.

This makes the algorithm applicable to general observations and task sets, rather than hard-coding day 6, day 18, a particular seed or a farm coordinate. General applicability does not imply that every step has an improvement, that every board is feasible, or that hiring savings always increase the final margin.

## 5. Development sequence of the local route experiments

| Stage | Observed problem | Algorithmic response | Established limit |
|---|---|---|---|
| Route/day experiments | Boundary travel can bypass useful work | Compare chains of field actions and then full games | A local route saving is not a whole-game profit result |
| Route economics | Delivery, capacity and purchasing cash were missing from route value | Simulate delivery timing, stock movement, purchase fills and real cash | Some accepted plans still lost registered tasks |
| Complete workflow | Later reordering or partial insertion broke dependencies | Preserve/revalidate feasible full schedules and dependent chains | More reliable execution did not establish higher win rate |
| Procurement recovery | Planned materials were unavailable after missing purchases | Reconcile resources, fund purchases and reconstruct remaining work | Recovery could pause workers or exit prematurely |
| Latest local repair | False cash alarms, early completion and incomplete capacity checks | Check actual fills, keep recovery pending, allow useful work and validate quantities | Feasible deferral could still be economically harmful |
| Larger live panel | Small panels and completion counters overstated progress | Freeze programs, pair seeds/seats, compare final outcomes and execution diagnostics | Strong regression remained against original JointAFS R2 |

The complete-workflow and old-recovery versions are descendants of P16, not the untouched P16 baseline. The [frozen baseline](../../nt/latest_20260910_r2p16/) remains separately identified at commit `979fec14bdddd281f4d934494aa4a06673fba9b9`.

### What procurement recovery actually does

Old recovery looks for a feasible remaining schedule with minimal additional hiring. If funds are missing, it can sell current stock, observe the proceeds, buy later and reconstruct work. Some funding/purchase stages pause all workers. Investigation found that being below predicted cash could trigger recovery without a real missing fill, and a partial fallback could clear recovery before the outstanding work was complete.

The latest local repair checks actual resources and transaction effects rather than cash prediction alone. It keeps recovery pending when required work is only partly scheduled, retries after actual increases in cash, materials or workers, and retains useful worker actions during replenishment.

It protects existing maintenance first, then considers up to four ranked investment subsets. It compares feasible schedules through current-day outcomes plus a common next-day value estimate. Optional search receives approximately 0.65 seconds of process CPU time; necessary action generation continues. This is not an absolute whole-call wall-clock guarantee.

Official action ordering matters to the cash constraint. Worker work and deliveries precede the relevant market settlement; anticipated sales cannot fund a prior phase. Capacity must be checked after earlier workers' effects. An intended animal purchase also has to fill, not merely appear in the proposed order list.

## 6. The two JointAFS submissions: why bundle animals, feed and successor crops?

Animal investment, feed production and the next crop compete for the same land, cash and labor. Optimizing them separately can keep an extra planned animal while failing to recognize that a small feed plot followed by another crop creates a better combined return.

JointAFS expands this particular candidate space. It considers removing one or two **unpaid planned animals**, using one or two plots for wheat feed and then planting a successor crop. It does not sell or remove an already existing animal to implement this candidate.

The implementation limits the search to three shapes: one removed animal with one feed plot, one removed animal with two feed plots, and two removed animals with two feed plots. It enumerates wheat feed ages 2 through 4 and five successor crop choices, with one successor cycle. This is deliberately bounded search; it is not exhaustive joint optimization of every crop and animal sequence.

Candidate prefiltering shifts feed-harvest receipts to the next dawn so that a later harvest does not incorrectly fund an earlier purchase. The search retains a promising estimated candidate for each shape, then compares it with keeping the incumbent plan under a common rollout horizon. Because filtering is approximate, an excluded candidate could still have been good.

### Original R1: replan after reducing the animal allocation

R1 lowers the planned animal allocation and invokes planning again to accommodate feed and the successor. This introduces coordinated alternatives missing from the original P16 candidate set. However, replanning the whole farm can disturb unrelated investments or maintenance while making the local animal/feed change.

The source and original portable archive are in [latest_20260911_p16_jointafs_r1](../../nt/latest_20260911_p16_jointafs_r1/). The portable archive is byte-identical to the downloaded R1 file used locally. Its submission identifier is `56146577`.

### Original R2 design: keep the farm and exchange a small part

R2's local exchange begins with the incumbent plan. It removes only selected eligible planned-animal paths, inserts the feed/successor bundle and preserves unrelated production and commitments. Already paid animals or inventory must not be treated as freely removable plans; existing owned wheat seed can reduce the new purchasing requirement.

This narrower change aims to make the simulated comparison meaningful: the measured difference should arise mainly from the proposed bundle, not from an accidental whole-farm reshuffle. It still needs cash reserves, placement eligibility and later execution checks.

The original R2 runtime is [preserved here](../../evidence/r2p16_route_repair_20260911/opponents/submission_56149565/). Its entry identifies `P16 JointAFS R2 LocalExchange`. The associated user archive is `submission (6).tar.gz`. Acquisition context associates it with `56149565`; there is no independently authenticated official archive hash establishing that association.

The original archive does not include full C++ source. The [later R2 workflow-repair source](../../nt/latest_20260911_afs_workflow_repair_r1_r2/r2/policy/) exposes the local-exchange implementation used to explain this design, but exact source-to-binary parity with the original R2 archive is not established. The later package is provided under its own version, not relabeled as the original submission's exact source.

### Common simulation and persistent bundle state

In the inspectable joint implementation, the common rollout extends through the feed and successor transition, using a horizon based on `age + 2`, typically four to six days. Keeping the incumbent and trying a bundle use the same horizon, public rival-flow scenario and subsequent policy. Candidate value has the form:

```text
value = simulated_own_cash
      - competition_weight * simulated_opponent_cash
      + estimated_remaining_value
```

A bundle must beat keeping the incumbent, start all required successors in the simulated continuation and avoid bundle cancellation. This guards against crediting a successor that the executor never plants. Remaining value is still an estimate rather than a full exact solution through the end of the match.

Persistent state links the days together: await actual wheat planting, track the live feed crop, recognize the harvest transition and then confirm the successor planting. The successor has a deadline relative to harvest day; unexpected disappearance or an incompatible tile state can cancel the bundle. Completion here means the successors have been **planted**, not that all successors have later been harvested and sold.

The bundle reserves missing successor seeds and limited feed coverage, including up to two days of feed in the relevant transition. It does not prove that all future wages, warehouse capacity and market receipts are sufficient. That remaining execution gap motivates the workflow repair.

## 7. The later JointAFS workflow-repair builds are separate releases

The [R1/R2 workflow package](../../nt/latest_20260911_afs_workflow_repair_r1_r2/) contains full source and off/audit/on build variants. Its default R1 and R2 entries enable workflow repair. They are different binaries from the original submissions and from the local procurement-repair candidate.

The repair tracks promised work and preserves a feasible witness. It certifies effects against observations and keeps previously committed work when a new target would collide with resources. It also distinguishes the coordinate used to acquire supplies from the coordinate used to service the animal; conflating the two can break a valid chain. Simulation-internal rollouts disable live workflow state so a hypothetical candidate cannot pollute the actual game's commitments.

The separate package records 1,100-game author evaluations per variant: R1 workflow-on has 856 wins, 4 draws and 240 losses; R2 workflow-on has 824 wins, 4 draws and 272 losses. These match their corresponding workflow-off win/draw/loss counts in that panel. R2 unresolved workflow breaks decrease from 40 to zero and final misses from two to zero, while mean own cash rises by about 1.20.

This supports better execution consistency in that author's panel; it does not establish a higher win rate. A selected seed's large cash improvement still ended in a loss. These results were not rerun in this documentation task and do not substitute for the local fresh 100-seed comparison.

## 8. What the expanded local tournament established

Three frozen local agents played live opponents through the existing compiled P16 C++ simulator. The 13 historical opponents used 25 fresh seeds each. Original JointAFS R1 and R2 each used 100 fresh seeds. Every seed was played from both seats.

| Panel | Complete workflow | Old procurement recovery | Latest local repair |
|---|---:|---:|---:|
| Historical opponents, 650 games per candidate | 421 wins / 64.77% | 417 / 64.15% | 440 / 67.69% |
| Original JointAFS R1, 200 games | 101 / 50.50% | 102 / 51.00% | 99 / 49.50% |
| Original JointAFS R2, 200 games | 122 / 61.00% | 125 / 62.50% | 97 / 48.50% |

There were no draws. These are wins divided by all games, unlike the route-clustering study's half-credit-for-draws score. The original R2's reverse rates against workflow, recovery and repair are therefore 39.0%, 37.5% and 51.5%.

Against R2, the latest repair loses 12.5 percentage points relative to workflow, with a paired seed-cluster 95% interval of [-22.0, -3.0], and 14.0 relative to recovery, interval [-23.5, -4.5]. The historical and R1 intervals include zero. Pool composition and acquisition date remain part of the claim; these are not estimates of universal leaderboard performance.

The full panel contains 3,150 matches and 4,536,000 bilateral observations. Existing transition/task checks passed, and a separate recount checked terminal outcomes. The fresh seed receipt checks non-overlap with 155 distinct prior seeds and seven development seeds. Development regression games and serial timing reruns are excluded from formal win rates.

### Why the latest local repair should not replace the baseline

Its 1,050 games have zero invalid actions and zero unexplained omissions of registered commitments, but **3,227 explicitly deferred tasks**. Against R2 alone there are 1,891 deferrals. Registration differs by version, and explicitly dropping an investment from the active work set is not the same as completing it.

Relative to workflow against R2, mean wages rise by 44.38 and own cash by only 13.94, while opponent cash rises by 1,313.71. Final margin falls by 1,299.77. Own cash and lower error counts are insufficient objectives when the opponent benefits much more.

Six selected large win-to-loss comparisons across four seeds first diverge on day 8, step 169, after identical prior observations. None reaches the optional-search cutoff at that first difference. Five defer 44 tasks; one changes same-step purchases and hiring without a deferral. These selected examples show that search timeout is not the explanation for every regression; they do not estimate the frequency of each cause across all losses.

Runtime timing also needs separate treatment. The formal repair panel's maximum recorded decision is about 0.692 seconds, but the original R2 reaches 2.660 seconds in a serial sample. One earlier repair result changes under a serial timing rerun. The host records duration without enforcing Kaggle timeout forfeits. Cash outcomes are not competition-runtime certification.

## 9. A reproducible development and evaluation workflow

1. **Freeze the starting point.** Record source commit, configuration, compiler, build flags and binary hash. Name the actual parent version; do not use an ambiguous label such as “old.”
2. **Reproduce the failure from the real observation.** Compare issued orders with actual fills, capacity, cash and work completion. Keep the observation before the first divergence, not just the final loss screen.
3. **State a general contract.** Examples include “a committed feed chain remains schedulable” or “a purchase is available only after its actual fill.” Avoid policy branches on known seeds or replay coordinates.
4. **Change the smallest relevant algorithm.** Preserve feasible schedules, add a missing resource constraint, improve the candidate set or correct the valuation horizon. Keep the economic configuration fixed when isolating executor behavior.
5. **Validate contracts and compiled execution.** Use meaningful regression cases for the mechanism, then verify through the same compiled simulator and actual native policies used by the intended comparison.
6. **Freeze before the formal panel.** Separate development seeds, timing probes and formal held-out seeds. Play both seats against multiple independently sourced agents, preserving original opponent binaries.
7. **Report both strength and mechanism.** Count wins/draws/losses, paired margin, wages, timing, invalid actions, unexplained misses and explicit deferrals. Explain denominators and uncertainty.
8. **Inspect regressions before promotion.** Verify the first differing decisions under identical observations. A completion fix can be correct while its investment policy is weaker.

This is algorithm development through simulation and controlled evaluation. Replays are valuable for diagnosing mechanisms and building regression observations; replay homogeneity should not inflate confidence by being treated as independent strategic diversity.

## 10. Running and rebuilding the supplied packages

The native `.so` files target Linux x86-64. Use Linux or compatible WSL; native Windows Python cannot load them. Preserve the shipped files when rebuilding so the tested artifacts remain identifiable.

For original R1, extract the portable archive into its own runtime directory and use its `main.py`. The [developer package](../../nt/latest_20260911_p16_jointafs_r1/agent/) also contains `build.py`, `check_binary.py`, `verify_package.py`, tests and build/provenance receipts. Inspect that release's build interface before invoking it.

For original R2, load the [preserved entry](../../evidence/r2p16_route_repair_20260911/opponents/submission_56149565/main.py) together with its adjacent `afs_runtime` directory. No C++ rebuild of the original archive is claimed here. For the later workflow variants, inspect [build_variants.py](../../nt/latest_20260911_afs_workflow_repair_r1_r2/tools/build_variants.py), [run_panel.py](../../nt/latest_20260911_afs_workflow_repair_r1_r2/tools/run_panel.py) and their recorded build variants.

The latest local repair has a relocatable build script. From its package directory:

```bash
python3 -B build.py --unit --out /tmp/route-repair-check/route3.so
```

The archived build reports 105 checks: 20 economics, 58 workflow and 27 recovery cases. This publication reuses those recorded results; it does not claim a new native build unless separately recorded. The frozen runtime entry loads `build/revision4/route3.so`. Create a separate agent for each match and release its state with `close()` as described in the [package README](../../nt/latest_20260911_r2p16_route_repair/README.md).

From the repository root, verify the published evidence without rerunning games:

```bash
python evidence/r2p16_route_repair_20260911/verify.py
python evidence/r2p16_route_agents_20260911/verify.py
```

These check preserved files, identities and archived outcomes. Tournament scripts under the evidence directories are source snapshots tied to the original workspace host, opponents and replay paths; they are not a newly portable one-command tournament. Full multi-gigabyte replay sets and intermediate builds remain outside Git. Re-running formal matches requires reconstructing those runtime dependencies and keeping the simulator version aligned with the experiment.

## 11. What should change next

The evidence points toward better coordination over time. Compare staged fundraising that preserves a worthwhile investment, same-step replenishment and temporary deferral over the **same multi-day horizon**. Keep retry conditions for investments that become affordable again. Include opponent/market feedback and opportunity cost, not only present cash, wages or a clean commitment ledger.

Separate ablations should isolate recovery triggers, capacity checks, investment-subset selection and search budgeting. Reduce redundant optional simulation before expanding the candidate set. Then evaluate frozen programs on fresh seeds and newly acquired opponent families, including the separately identified JointAFS workflow-repair builds.

The current record supports specific execution improvements and identifies a competitive regression. It does not establish that either of the two original submissions, the later workflow builds or the latest local repair is the strongest agent on every opponent or seed.
