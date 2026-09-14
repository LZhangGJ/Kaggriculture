# Pro8 arena

The controller, independent intake, evaluation scheduler, ratings, gates and reports run locally. Installation does not launch a campaign or publish anything. Preserve the existing Pro8 run.

## Setup

From the repository root, create and activate a virtual environment, then run:

```text
python -m pip install -r tools/arena/requirements.txt
python -m tools.arena.manage --root .arena init
python -m tools.arena.manage --root .arena check
python -m tools.arena.manage --root .arena tick
```

The last command creates `.arena/site/index.html`, initially empty. No synthetic results populate the leaderboard. Execution defaults off.

On an authorized CPU host with Docker:

```text
docker build -t kaggriculture-arena -f tools/arena/Dockerfile tools/arena
docker image inspect kaggriculture-arena --format {{.Id}}
```

Put the immutable image ID in `.arena/config.json`. The referee must be version 1.32.7. Measure resource use before raising workers or setting execution_enabled=true. Two containers and one referee run per game. No GPU access, network, credentials, or host workspace are mounted into agent containers. There is no unsafe host-execution fallback.

## Independent submissions

Copy `tools/arena/example-manifest.json`. ZIP the agent, then run:

```text
python -m tools.arena.manage --root .arena submit --archive agent.zip --manifest agent.json
```

Or use the private repository's **Submit agent** issue form once published. Attach the ZIP to an authorized private release and provide the asset ID and SHA-256. No research enrollment, task, Pro chat or experiment history is required. Unknown seed exposure limits freshness claims rather than ordinary admission.

For automated issue intake, configure github.repository, github.allowed_authors and github.intake_enabled. The local host needs existing `gh` authentication. Assets are fetched only from the approved repository. No GitHub writes occur in the local synchronizer. If release access is unavailable, supply the archive to the coordinator and use `import-issue --event event.json --archive agent.zip`.

Statuses are registered, pending, placement-rated, active and archived. Validation failures show a reason in local records. Every accepted valid agent gets placement outside the active cap: 32 seeds, both seats, six configured references, 384 games. Reference agents skip self-play. Twenty percent of evaluation slots go to placement with round-robin author scheduling and FIFO within each author. Idle allocation is lent to other jobs. Backlog is retained; there is no guaranteed deadline when arrivals exceed capacity.

## Agent interface

Build/run commands are argv arrays. The build executes inside `/work`; dependencies must be bundled or present in the image. Each stdin line is an object with observation and configuration. Emit one action object per stdout line and flush it. Diagnostics go to stderr. ARENA_AGENT_SEED supplies the declared random seed. See `tools/arena/example_agent.py`.

Python-callable Kaggle agents need a thin JSONL wrapper. Public notebooks need a trusted format-specific exporter; do not execute downloaded notebooks on the controller. Daily public_refresh_commands are explicit administrator-configured download/conversion argv commands. They write inbox JSON containing archive_path and manifest. The importer detects changed artifact hashes and preserves every version; roster replacement remains a reviewed decision.

## Research loop

Commands: `experiment ID --brief brief.json`, `search QUERY`, `finish ID --result result.json`. Brief fields are hypothesis, owner, parent, scope. Results require variants, evidence, conclusion and limitations. Duplicate active scopes are rejected. New attempts use new IDs.

Use up to eight GPT-6 Pro chats: four independent agents, two phase/component tasks, one rolling planner and one review/replication task. Keep roles flexible and preserve failed variants. Provide stable packets plus relevant new findings. Validate phase work in complete responsive games, not forecasts alone.

`events` returns compact pending decisions; `ack EVENT_HASH` marks handling. No model runs inside tick. A Codex task reads events when resuming. Automatic model waking requires a supported host integration; this repository does not invent a model API or control ChatGPT tabs from background shell scripts.

## Daily arena

Validate agents with `validate ID`. Set six placement_references. Apply a JSON roster using `roster FILE`: entries contain agent, category, reason. Default slots: 1 champion, 3 established, 8 public, 4 counters, 4 candidates. Keep weaker counters when they expose weaknesses. Review established retirement weekly and candidates daily; archive evidence rather than delete it. The registry is uncapped.

`plan daily-DATE --seeds 128` freezes a balanced round robin. Twenty agents produce 48,640 games. daily_enabled=true permits tick to start the next local-calendar day after the prior run finishes. No overlapping/catch-up tournaments. Predeclare fewer seeds when capacity requires it; incomplete reports stay provisional.

Each run pins agents, referee, image and execution limits. Ticks resume missing games and count each terminal result once. Infrastructure failures get up to three recorded attempts, then require diagnosis. Correct the fault and explicitly plan the next attempt; never erase failure history. Agent failures use forfeits, not dropped observations.

`topup NEW_RUN --source COMPLETED_RUN --candidate ID --opponent ID --target 512` adds only the missing seeds for that pair. Plain `plan --kind topup --seeds N` instead means N new seeds. Neither changes base-round scores or repeats existing games as new evidence.

Install the Windows timer only after configuration using `tools/arena/install-timer.ps1 -Root ABSOLUTE_ROOT -Python ABSOLUTE_PYTHON`. It runs deterministic ticks, with overlap disabled. A Linux OS timer can invoke the same CLI. No scheduler is installed automatically. The kernel lock also prevents concurrent controllers.

## Gates

`compare-plan RUN --candidate ID --incumbent ID --panel FILE --seeds 256` freezes a matched comparison. Panel entries have agent, family, panel (representative/original/stress) and protected. All three groups are required. `gate RUN` reports the result; `promote RUN` additionally verifies the incumbent has not changed. Initialize champion.json only from verified evidence.

Defaults: one-point representative gain with positive paired lower bound; two-point original/stress tolerance; five-point protected-cell tolerance; no other observed cell loss over ten points. These are approximate operational choices. Fixed thresholds and sample budgets are stored in the run. Do not extend a failing test until it passes.

For final public targets, `plan RUN --kind confirmation --candidate ID --references ID... --seeds 1024`, then `certify RUN`. Exact simultaneous per-seat lower bounds are averaged across seats and must exceed 90% for each opponent. The claim covers this frozen set and single planned attempt only. Repeated selection/looks require a declared error budget. The 75% top-player target requires official matches.

## Reports and publication

`report --bootstrap 500` produces private HTML, JSON and Markdown. Draw-aware ratings, whole-seed intervals, matchups, seats, cash, daily changes and history are separate from champion decisions. Descriptive daily changes are not significance claims. Different rosters affect ratings. Exact agent versions do not inherit observations. Cached summaries avoid repeated fits when data have not changed.

Actions perform metadata receipts, code tests and approved report exports. arena-site.yml first produces a private workflow artifact. Pages deployment requires ARENA_PAGES_APPROVED=true and verified access policy. Private repo status does not establish Pages privacy. Review generated exports before copying them to tracked arena-export for approved publication. Never put reserved seeds, private archives or credentials there. Large packages stay outside Git.

No pushes, purchases, publication, Kaggle submissions, model fallback or GPU use are enabled. All current output is local.

## Tests

```text
python -m unittest discover -s tests/arena -v
```

Tests cover unsafe archives, hashes, idempotence, balanced seats, seed reuse, overlap, result identity, draw accounting, placement beyond cap, issue parsing, disconnected ratings and no-effect rejection. A live Docker smoke is also required before accepting contributed code on a host; unit tests cannot establish sandbox availability.
