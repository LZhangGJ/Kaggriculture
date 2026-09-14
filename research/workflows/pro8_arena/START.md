# Pro8 arena

## Shared arena

**Submitting an agent?** Give your coding agent the [copy-and-paste submission workflow](SUBMIT_AGENT.md). It works without Pro8 and covers website upload, GitHub CLI submission and status checks.

[Open the live results page](../../arena_live/README.md). The coordinator runs on WRX90 using CPU only. It publishes checked summaries to `feature/pro8-arena`; it does not merge into main or submit to Kaggle.

You can submit an agent without running Pro8. The easiest route is **Submit agent** on the [private website](https://kaggriculture-arena.tail0d430d.ts.net/): sign in, name your agent, and upload its file. [Formats and status guide](PRIVATE_SITE.md#upload-an-agent).

The GitHub route remains available. Package your JSONL agent in a ZIP, upload it as a release asset in this private repository, then open an issue titled `[Arena] Your agent name`. Include this section, with your real asset ID and archive SHA-256:

````text
### Agent manifest
```json
{"name":"My agent","author":"your-github-login","version":"1","run":["python","main.py"],"archive_asset":123456,"sha256":"YOUR_ARCHIVE_SHA256"}
```
````

Repository collaborators can submit. The coordinator reads issues directly, so this works while the workflow remains on a feature branch. The issue form and Actions triggers need a separate merge into the default branch. ZIPs must be at most 256 MiB and contain no links. The agent reads one JSON object with `observation` and `configuration` per stdin line and writes one action object per stdout line; send logs to stderr. Use `tools.arena.kaggle_export.pack` to wrap an existing Python `agent` entry point.

New agents get placement games even when the active roster is full. Placement does not automatically replace the champion. Watch the results page for intake and evaluation status.

To join research, read existing experiment records before claiming a scope. If you only submit agents, no research setup is needed.

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

Statuses are registered, pending, placement-rated, active and archived. Validation failures show a reason in local records. Every accepted valid agent gets placement outside the active cap: 32 seeds, both seats, six configured references, 384 games. Reference agents skip self-play. Daily tournaments take priority. Outside that priority work, twenty percent of coordinator evaluation slots go to placement with round-robin author scheduling and FIFO within each author. Idle allocation is lent to other jobs. Backlog is retained; there is no guaranteed deadline when arrivals exceed capacity.

## Agent interface

Build/run commands are argv arrays. The build executes inside `/work`; dependencies must be bundled or present in the image. Each stdin line is an object with observation and configuration. Emit one action object per stdout line and flush it. Diagnostics go to stderr. ARENA_AGENT_SEED supplies the declared random seed. See `tools/arena/example_agent.py`.

Python-callable Kaggle agents need a thin JSONL wrapper. Public notebooks need a trusted format-specific exporter; do not execute downloaded notebooks on the controller. Daily public_refresh_commands are explicit administrator-configured download/conversion argv commands. They write inbox JSON containing archive_path and manifest. The importer detects changed artifact hashes and preserves every version; roster replacement remains a reviewed decision.

For daily leaderboard refresh, prefer `public_sources` in config; see `tools/arena/example-public-source.json`. Each entry names a notebook, trusted exporter command, and output receipt path. The exporter must replace that receipt on every successful check, even when the notebook is unchanged. The receipt contains archive_path and manifest, whose origin is `{ "kind": "public", "notebook": "owner/slug", "version": "exact-version" }`. The exporter must check the current upstream version; a cached package alone does not establish freshness.

Each configured notebook is checked once per local calendar day. One failed source does not stop others or evaluation. Failures and last successful checks appear on the page. Changed versions get separate identities and await sandbox validation. Once validated, they replace the matching public entry in the next roster; existing frozen tournaments retain their old versions. Every public entry carries a Public notebook badge and source/version details, including when serving as a counter rather than occupying a public slot. Sources still need configuration and the timer must be running for live daily refresh.

## Research loop

Commands: `experiment ID --brief brief.json`, `search QUERY`, `finish ID --result result.json`. Brief fields are hypothesis, owner, parent, scope. Results require variants, evidence, conclusion and limitations. Duplicate active scopes are rejected. New attempts use new IDs.

Use up to eight GPT-6 Pro chats: four independent agents, two phase/component tasks, one rolling planner and one review/replication task. Keep roles flexible and preserve failed variants. Provide stable packets plus relevant new findings. Validate phase work in complete responsive games, not forecasts alone.

`events` returns compact pending decisions; `ack EVENT_HASH` marks handling. No model runs inside tick. A Codex task reads events when resuming. Automatic model waking requires a supported host integration; this repository does not invent a model API or control ChatGPT tabs from background shell scripts.

## Continuous round robin

The mini PC runs a separate endless queue. WRX90 reserves all seeds, freezes the current roster into each round, sends two rounds ahead, and collects results at each controller batch. Each round uses four new seeds per pairing in both seats. Roster changes affect newly queued rounds. Daily tournament and placement jobs continue on WRX90.

The results page has a separate Elo table. Each exact agent version starts at 1500. A completed seat-swapped pair applies one K=32 update using its average score, with draws worth half. A database makes repeated result transfers idempotent. Ratings from different execution contracts stay separate; daily games do not enter this table. Treat ratings with few games as provisional.

The mini PC needs Docker, the same pinned image and referee, and the trusted `tools/arena` runtime. It does not need GitHub or Kaggle credentials. Put connection settings only in `.arena/private/continuous-host.json` and the host's SSH configuration. Never commit these files, keys, access URLs, private seeds, or agent bundles. The mini PC runs `python -m tools.arena.continuous execute .arena`; WRX90 performs synchronization and remains the sole publisher.

## Daily discovery and tournament

The shared host discovers public notebooks daily in Kaggle public-score order, highest first, and selects only notebooks updated in the preceding 24 hours. The API update timestamp is `lastRunTime` in UTC. It checks up to 1,000 entries and marks scans that hit this limit. It downloads up to four untracked outputs in that order per day, without running notebooks on the host. Accepted sources also receive daily version checks.

After a complete daily tournament, the controller selects its weakest public agent by strict win rate. One new public challenger at a time faces the same remaining roster as that agent, on 128 fresh seeds in both seats. It replaces the incumbent only if its mean win rate improves by at least two percentage points and the approximate one-sided 95% paired bootstrap lower bound exceeds zero, with no nonterminal challenger games. A changed roster invalidates that replacement decision. These are operational roster decisions, not a statistical guarantee across repeated trials. Current tournament manifests stay frozen, and retired agents keep their records.

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

New installations default to execution and publication off. The shared host has explicit approval for CPU evaluation and private summary publication. Kaggle submissions and GPU use remain off.

## Tests

```text
python -m unittest discover -s tests/arena -v
```

Tests cover unsafe archives, hashes, idempotence, balanced seats, seed reuse, overlap, result identity, draw accounting, placement beyond cap, issue parsing, disconnected ratings and no-effect rejection. A live Docker smoke is also required before accepting contributed code on a host; unit tests cannot establish sandbox availability.
