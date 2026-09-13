# Kaggriculture agent iteration workflow

[中文版本](README.zh-CN.md)

Updated: 2026-09-13. This document describes the controller-led workflow used for the ChatGPT 6 Pro agent experiments and the current three-lineage acceptance protocol. The results section is a dated snapshot, not a live status feed.

## Objective and responsibilities

The objective is to improve competitive performance through repeated source changes, local matches, and evidence-based feedback until each of the three current agent lineages exceeds **85% overall strict wins** against the specified opponent pool.

The initial exploration involved eight separate ChatGPT 6 Pro author conversations. The current focused loop uses three separate author conversations for **A08 r11**, **A06 r6**, and **A06 calendar r12**. Each author continues its own lineage; revisions retain an explicit parent identity.

| Role | Responsibility |
| --- | --- |
| Controller, running locally through Codex | Maintain versions and deadlines; prepare source and defeat evidence; dispatch author tasks; download and verify deliveries; freeze evaluation inputs; run and audit matches; publish artifacts and send the next feedback. |
| ChatGPT 6 Pro author | Inspect the actual source and evidence, investigate a concrete cause, implement and compile a change, run feasible checks, and deliver a usable ZIP within the round budget. |
| Local evaluation harness | Execute the pinned official game with isolated agent processes, save complete results and replays, and provide the evidence used for acceptance. |

Current acceptance is based on competitive wins. Mean terminal cash remains a diagnostic. Earlier cash thresholds and games against a passive opponent are no longer acceptance gates.

## The iteration loop

1. **Establish the current checkpoint.** Read the authoritative state, previous results, exact parent source/configuration/native library, and any active process or conversation. Preserve completed rounds. Do not launch duplicate work because a status observation timed out.
2. **Diagnose losses and protect narrow wins.** Use the completed panel to find representative public-opponent losses, R2 losses, close losses, and narrow-win controls. Reconcile observed cash changes with purchases, sales, production, labor, land use, maturity, and terminal inventory. Distinguish a planning-value error from a scheduling or execution failure.
3. **Dispatch one bounded author round.** Identify the parent and next revision, evidence, a concrete hypothesis, permitted changes, resource budget, start time, deadline, validation scope, and required files. Give each lineage its own conversation. Request actual source changes and a downloadable package.
4. **Collect and verify the delivery.** Preserve the original ZIP and its SHA-256. Check package integrity, complete production source, configuration, matching native library, build instructions, default entry point, and relevant positive and negative checks. Record what was independently verified and what remains only an author claim.
5. **Freeze before sampling.** Freeze candidate source/native identities, official engine, opponent files, runner, and protocol. Collect author development seeds and prior reservations. Draw the next shared 64 seeds once from the eligible source pool and save the immutable plan before running matches.
6. **Run the complete local panel.** Execute every declared seed × opponent × seat combination. Retain every result, error, and replay. Keep one full evaluation panel active at a time. Author smoke tests and small development comparisons are recorded separately.
7. **Audit, publish, and decide.** Verify completeness, terminal status, hashes, and metrics independently. Publish exact versions with English descriptions and reproducible evaluation evidence. If a lineage has not passed, return concrete failure evidence to its author and repeat with a new revision and fresh evaluation seeds.

The same loop covers economic planning, rolling scheduling, and low-level execution. Dynamic programming is useful when its state, horizon, constraints, objective, and fallback are explicit; its value must ultimately be supported by closed-loop competitive results.

## Author round budget: at most two hours

The 120-minute limit includes reading, analysis, implementation, compilation, validation, troubleshooting, and packaging. Follow-up messages do not reset the start time. The controller's full local panel is a separate workload.

| Time budget | Required action |
| --- | --- |
| First 5 minutes | Measure current CPU affinity and cgroup CPU quota, memory limit/current usage, available memory and disk, and Python/compiler versions. A previous environment report is only historical evidence. |
| Early in the round | Time a bounded compile and representative probe; record wall time and peak RSS. Estimate the feasible workload from measurements, including tool and packaging overhead. |
| By 90 minutes | Finish a defensible source checkpoint and the necessary focused checks. |
| By 100 minutes | Begin packaging and final verification. |
| Before 120 minutes | Deliver the real checkpoint and raw evidence, including failures and incomplete work. |

If effective CPU quota is unknown, use one worker conservatively. Bound parallelism by both CPU and memory, leaving at least 30% memory headroom. A short early-game probe does not establish the cost of a full game. Every subprocess timeout must fit inside the remaining round budget.

## Formal local evaluation settings

| Setting | Current protocol |
| --- | --- |
| Candidate lineages | A08 r11, A06 r6, A06 calendar r12; each revision has a frozen identity |
| Seed source | The published representative panel of 256 seeds and stress panel of 128 seeds |
| Sampling | 64 total, uniformly without replacement from their remaining eligible union; one shared draw for all three candidates |
| Exclusions | Previously reserved or tested seeds, plus all reported author development seeds |
| Opponents | 11 fixed public entries plus the true original AFS R2, identified as `submission_56149565` |
| Seats | Both seats, 0 and 1, for every seed/opponent pair |
| Game length | Complete official games with 719 transitions |
| Denominator | 64 × 12 × 2 = **1,536 games per candidate**; **4,608 games** for three candidates |
| Win definition | Strict terminal win; a draw does not count as a win |
| Acceptance | More than 85% overall: at least **1,306 strict wins out of 1,536** for each candidate |
| Additional reporting | Representative/stress subsets, each opponent, public-11 aggregate, R2, seat, terminal cash, and errors |

The exact opponent entries are:

`soil_v219g`, `moon_v215`, `flexon_v5`, `market_smart_v8`, `nagatakengo_v70`, `aurax_reactive_v1`, `thomas_955_v2`, `shop0909`, `aurax_shop_v2`, `seven_turn`, `ahmed_v27`, and `submission_56149565`.

Entries have equal weight under the requested protocol. Some public entries share code; the pool should not be described as 12 independent strategy families. Earlier local starter agents and other author revisions are outside this acceptance pool.

This is a **mixed development panel**, not the sealed Holdout: the sealed seed values were unavailable. Keep the representative/stress labels; their counts may differ between random draws. Do not redraw after seeing outcomes or reuse depleted seeds silently. If fewer than 64 eligible seeds remain, obtain a new seed source before another fresh panel.

The current harness isolates policies in fresh processes, withholds the environment seed from policy observations, and uses a response watchdog. These local settings do not certify compliance with a separate competition platform's runtime limits.

## Evidence and feedback rules

- Preserve the full declared denominator. Partial games, errors, or missing rows cannot disappear from reporting or support an acceptance claim.
- Verify source/native/engine/opponent/protocol hashes, the seed × opponent × seat Cartesian product, all terminal outcomes, and replay hashes. Independently recompute totals and subgroup metrics from raw rows.
- For selected feedback cases, replay saved joint actions through the pinned official engine and reconcile cash ledgers. This verifies the recorded game; it does not evaluate a changed policy.
- Use paired parent/candidate matches on identical seed/opponent/seat contexts for causal diagnosis when resources permit. Include narrow-win controls and retain all failures. Reusing feedback cases is development, not fresh acceptance.
- After a changed action, continuing the recorded future is not a valid counterfactual. Run both policies closed-loop to measure the changed strategy.
- Production decisions may use only current legal public observations and the agent's own private state. Do not introduce hidden-seed lookup, opponent-identity lookup, future-action scripts, or opponent-private-state access.
- Transfer source and failure evidence only within the authorized scope. Opponent source and full private simulation states require appropriate scope coverage; own-visible feedback is a possible reduced package. A rejected upload is not a dispatched task.

The prepared second-round feedback currently contains seven selected cases per lineage: five loss categories and two narrow-win controls. The local audit verifies 719 transitions and 60 player-days of cash accounting per selected game.

## Delivery and publication

An author delivery contains the entry point, complete production source, fixed configuration, matching Linux x86-64 native library, offline build instructions, compiler command/flags and source/native hashes, relevant checks, raw validation/resource/timing logs, and an English README. Failures and limitations remain visible.

The controller preserves each original ZIP, central verification receipt, frozen plan, machine-readable results, per-game rows, and replay identities. Publish new revisions in new directories, preserving older versions and reports. Verify the remote commit and published file bytes after pushing. Do not treat successful compilation, a favorable subset, or author-reported self-play wins as pool acceptance.

## Verified snapshot on 2026-09-13

The first revised panel is complete: 64 fresh shared seeds, comprising 48 representative and 16 stress seeds; 4,608 full games; zero errors and zero draws.

| First revision | Strict wins | Overall win rate | Versus original R2 |
| --- | --- | --- | --- |
| `TRI_A08_r11_r1` | 1,054 / 1,536 | 68.62% | 66.41% |
| `TRI_A06_r6_r1` | 1,196 / 1,536 | 77.86% | 92.19% |
| `TRI_A06_r12_r1` | 1,099 / 1,536 | 71.55% | 89.84% |

All three remain below the overall target. Different seed panels do not establish that a revision caused an improvement or regression. The second author round had not started at this snapshot: its full feedback uploads were awaiting resolution of an upload-scope approval rejection.

- [Original agents and English descriptions](../../agents/pro8_20260913/README.md)
- [First revised agents and English descriptions](../../agents/pro8_20260913_revision1/README.md)
- [Original 64-seed results](../../evaluation_runs/pro8_tri64_20260913_baseline/README.md)
- [First revised 64-seed results and protocol](../../evaluation_runs/pro8_tri64_20260913_revision1/README.md)
- [Published seed panels](https://github.com/LZhangGJ/Kaggriculture/tree/research/evaluation-seed-panels-20260912/research/evaluation_sets/2026-09-12-v1)
