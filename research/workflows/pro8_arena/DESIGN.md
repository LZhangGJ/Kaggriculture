# Pro8 research workflow and shared arena

Status: design only. No jobs, GitHub writes, publication, or submissions were enabled by this consultation. Preserve the active Pro8 run.

Design source: [GPT-6 Pro consultation and amendment](https://chatgpt.com/c/6aa76085-9f2c-83e9-b373-bd76d091a0f3). This document combines the answers; the amendment supersedes the original periodic reasoning cadence and teammate onboarding. Commands below are proposed, not installed tools.

## Purpose

Run continuous agent research while maintaining a shared, inspectable arena. Work fully with one user; allow four optional teammates to join research or submit independently. Keep the implementation to Python scripts, ordinary files, an OS scheduler, and a generated GitHub page. No API service, database, distributed locks, or custom web backend.

CPU only. Measure safe capacity and protect existing workloads. No GPU use, purchases, model substitutions, GitHub pushes/publication, or Kaggle submissions outside explicit authorization.

## Research loop

Codex reads relevant experiment history, assigns distinct questions, verifies returned artifacts, reviews completed evaluations, and sends authors focused feedback. Do not wait for every author to finish.

Eight author slots default to four independent complete agents, two phase/component improvements, one rolling planning experiment, and one replication/cross-review/technique-transfer task. Roles are flexible. Keep room for substantial rewrites. Use requested GPT-6 Pro for author work.

Author rounds last up to two hours, including packaging. Provide a stable rules/source packet plus a small task-specific brief. Preserve failed variants, conditions, source identity, measurements, and conclusions. Search previous attempts before dispatch; explain justified repeats. A negative result does not rule out an idea under changed conditions.

Phase labels describe decisions rather than fixed days. Rolling horizons and shared executors are hypotheses, not mandatory architecture. Validate components with completed games from varied reachable states and responsive opponents. Forecast gains or fixed replay actions alone do not establish improvement.

## Two independent ways to participate

Researchers may take assigned tasks and development-seed blocks, with or without their own swarms. One coordinator owns shared assignments and state. Participants return isolated result packages; no concurrent edits to central JSON. A teammate can leave without blocking the campaign.

Independent contributors use a private-repository **Submit agent** issue form referencing an authorized archive, or the local fallback:

```text
python manage.py submit --archive agent.zip --manifest agent.json
```

Minimum intake fields: name, author, version, archive location/hash, build command or prebuilt status, runtime/dependencies, run command, and resource requirements. No research task, Pro chat, experiment notes, or seed block is required. Source/provenance/seed history may be supplied; missing history alone does not reject an agent. Actual competition rules still apply. Unknown seed exposure limits freshness claims.

An intake receipt acknowledges the request. Artifact receipt requires download and hash verification. Check safe extraction, build, protocol, and resources. Build and run contributed code in an isolated environment without credentials, network, or host-workspace access and with CPU/memory/process/time/storage limits. Do not execute submitted commands on the credential-bearing controller or a broadly privileged self-hosted Actions runner.

Statuses: registered, pending, placement-rated, active, archived. Failures carry a reason and sanitized logs; incomplete evaluations show completed/planned counts. Every valid accepted submission gets placement evaluation regardless of predicted strength. Default: 32 seeds, both seats, against six weekly-frozen references/counters: **384 games**. This is provisional placement, not certification.

Reserve 20% of safe evaluation capacity for placement, lending idle capacity elsewhere. Schedule round-robin across authors and FIFO within each author. Apply the same policy to workflow and independent submissions. Deduplicate identical artifact/configuration identities while preserving attribution. Show backlog; do not promise fixed turnaround when arrivals exceed capacity.

## Active arena

| Size | Champion | Established | Public | Counters/specialists | Candidates |
|---|---:|---:|---:|---:|---:|
| 16 | 1 | 2 | 6 | 3 | 4 |
| 20 default | 1 | 3 | 8 | 4 | 4 |
| 24 | 1 | 3 | 10 | 4 | 6 |

One identity occupies one slot. Lend empty slots. The cap applies only to daily play, never registration, placement, or the historical leaderboard.

Select public agents for strength and distinct matchup behavior. Preserve opponents that expose weaknesses even if their average rank is low. New source/configuration versions get new identities and no inherited observations.

Review candidate slots daily after adequate coverage. Admission prioritizes missing coverage/counter value, then matched performance, then ready date. Review established retirement weekly. Retire duplicates, covered superseded versions, redundant matchup profiles, and unsuccessful candidates. Never retire solely for age/low rank or to hide a champion weakness. Preserve all evidence. Recheck archived counters weekly. Freeze roster changes during a tournament.

## Daily tournament

Default timezone America/New_York; freeze roster, versions, referee, and seed plan at midnight; report at 23:30. Times are configurable. Refresh public versions before the next freeze and record refresh failures.

Base schedule: every unordered pairing, 128 fresh seeds, both seats. At 20 agents this is **48,640 physical games**, or 4,864 appearances per agent. At 16: 30,720 games; at 24: 70,656. Shared seeds across pairings support matched comparisons. Both seats form one seed block, not independent evidence.

Benchmark complete games and concurrency first. Default arena budget is at most half of safe simulation capacity, leaving capacity for research; placement is a separate reserved share within the same total budget, not extra capacity. Use 128 seeds if projected completion fits 16 hours; otherwise predeclare 64 or 32 and label reduced precision. Never silently drop counters. Never overlap unfinished tournaments or accumulate catch-up runs. Continue the frozen run and issue an incomplete daily report.

Targeted follow-up may reach 512 or 1,024 cumulative unique seeds per pairing. Predeclare fixed batches, count physical games once, and keep adaptive follow-ups separate from balanced-base summaries. Never keep playing until a favorable threshold appears.

Daily new seeds add evidence for unchanged versions. Repeating an identical deterministic game does not. Keep original fixed benchmark, dated current public pool, and stress/counter results separate. Do not infer improvement from roster-driven rank changes.

## Ratings and champion decisions

Primary arena ranking: regularized Bradley-Terry-Davidson rating with a global seat effect and draw parameter. Base-round score is secondary; strict win rate and win-plus-half-draw score remain distinct. Show the matchup matrix because a scalar rating can hide counters.

Proposed implementation: small Python optimizer, weak quadratic regularization, and 500 whole-seed bootstrap refits for approximate intervals. Resample both seats and all pairings together; stratify cumulative resampling by day. Keep an unchanged anchor present when possible. Disconnected components must not share a claimed comparable rank. A fixed anchor does not eliminate opponent-pool changes. Validate estimator recovery and interval behavior before relying on these defaults.

Daily winner is the highest-rated agent in the completed balanced tournament; label incomplete results provisional and overlapping intervals uncertain. Keep daily, placement, and cumulative tables separate. The verified champion and best competition-ready artifact are separate records.

Incremental promotion proposal: at most one selected challenger per day, frozen before a fresh matched confirmation against representative, original, and protected-counter panels. Default 256 seeds; choose a larger fixed budget before testing if needed. Proposed tolerances:

- Family-balanced representative strict-win gain at least one point and paired 95% lower bound above zero.
- Simultaneous lower bounds for original/stress aggregate changes above minus two points.
- Protected opponent/seat lower bounds above minus five points; no other tested opponent/seat observed decline over ten points.

These are configurable operational choices, not validated optimal thresholds. Predeclare them and the resampling method; do not tune them after seeing a contender's result. Inconclusive candidates stay pending or enter a separately planned test; repeated looks need an explicit error-control plan. Zero-effect changes do not earn promotion.

Final public target is over 90% against each named notebook version, not an aggregate. Freeze the candidate and reserve undisclosed seeds; default certification budget 1,024 seeds per opponent, both seats. Pro proposed exact per-seat binomial lower bounds with error budget split across comparisons/planned looks; average seat bounds for equal-seat performance. Independence and sampling assumptions must hold. Sample size alone never guarantees a pass. The over-75% named top-player target requires official matches; private reactive agents cannot be replaced by replay tapes.

Keep confirmation seeds private and retire them after exposure. Track all tested/disclosed seeds across contributors. Do not claim global freshness when outside history is unknown.

## Minimal files and automation

```text
START.md, CONTRACT.md, config.json, STATE.json, CHAMPIONS.md
experiments/<id>/       # attempts, source identities, evidence, conclusions
inbox/<participant>/    # isolated deliveries
opponents/<snapshot>/  # pinned versions
runs/<id>/             # frozen manifest and game records
artifacts/index.csv    # hashes and authorized storage references
private/seeds.csv      # never published
site/                  # generated page and daily history
```

Proposed entry points: `manage.py` for intake/state/packets, `arena.py` for scheduling/resumable games, `report.py` for validated statistics and page generation. An OS scheduler runs `manage.py tick`; it does not invoke a reasoning model merely because time passed.

GitHub Actions proposals:

- `intake.yml`: parse authorized submission issues as data, validate metadata, report status. Never execute submitted fields.
- `checks.yml`: test trusted workflow code.
- `site.yml`: build/publish only approved export data within approved access and quota.

Keep CPU-heavy simulation local. Verify existing Actions quotas/site access before enabling; no paid purchase. Preserve local operation when GitHub is unavailable.

Scripts handle queueing, calculations, reports, known transient retries, and missing-game recovery. Codex wakes for material completed batches, research decisions, faults needing diagnosis, and champion review. Provide compact deltas and relevant notes; cache stable author packets by hash and batch feedback. The actual Codex wake mechanism must use a supported available integration, verified during implementation; ordinary shell scripts do not magically start reasoning sessions. Browser authentication/ambiguous sends still require attention and must not be blindly retried.

## One progress page

Homepage: verified champion and exact artifact/evidence date; daily arena leader; ratings/uncertainty/coverage; matchup W/L/D, per-seat and cash margins; fixed-benchmark plots; placement queue; active research and findings; new notebook versions; freshness/errors and incomplete counts.

Daily report: tournament result, supported changes versus roster effects, champion changes, notable matchups and sample sizes, entries/retirements with reasons, and linked evidence. Compute all numbers from validated records. Use templates rather than LLM-generated status prose.

Default preview is local/private. Publish a private GitHub Markdown page after approval; use access-controlled Pages only after verifying support/access. Private repository status alone does not prove Pages privacy. Public exports require an explicit allowlist and approval. Exclude reserved seeds, restricted source/artifacts, replays, and private findings. Keep large files outside Git. Archive access can stay limited to the author and evaluator. Approved ongoing publication scope can cover daily updates without repeated prompts.

## Acceptance before launch

Verify independent intake without research enrollment; fair placement beyond arena cap; bad-hash/unsafe-archive rejection; isolated builds; shared result contract; interrupted-run recovery with no duplicate counts; exact artifact traceability; statistical/seat/tie tests; honest incomplete reports; CPU limits; export privacy; and token-free idle scheduling. Import existing Pro8 artifacts/history without rewriting their evidence. Measure throughput before choosing arena size. No live run was modified by this design.
