# Kaggriculture evaluation tools v2

V2 adds strict input checks, a four-seed parity preflight, five separate economic stress opponents, source-group reports and simultaneous pairwise intervals. It reuses the pinned game execution functions and runtime in `research/evaluation_sets/2026-09-12-v1`. That published bundle and its running comparison remain unchanged.

The default preflight covers all six candidates, four seeds, 21 opponents and both seats: **1,008 cases, 2,688 complete games, and 1,451,520 native/official observation comparisons** when it passes. Four candidates need the third package/adapter game. These games test execution; they do not contribute to win-rate estimates. The default development run has 96,768 games: 73,728 primary games and 23,040 diagnostic games. It does not run automatically after preflight.

## Start a new experiment

Use Linux or WSL for the pinned native binaries. Python 3.11 or later suffices for the runner. Analysis also needs NumPy. Do not substitute the repository root's older 1.32.6 engine. Run from the repository root; all input and output paths are explicit or repository-relative, with no drive-letter launcher assumptions.

```sh
python3 -B research/evaluation_tools/v2/panel_runner.py freeze --out artifacts/evaluation-v2/inputs
python3 -B research/evaluation_tools/v2/panel_runner.py run --freeze artifacts/evaluation-v2/inputs/FREEZE.json --phase preflight --out artifacts/evaluation-v2/preflight --workers 4
```

The freeze pins all runtime bytes, all public seed manifests including the stress pool, opponent identities, reference features, analysis thresholds, audit evidence and v2 code. To use another roster, pass `--roster path/to/roster.json` to `freeze`; its runtime must live in the selected `--bundle`. To restrict a verification run, repeat `--candidate ID` when freezing. A restricted preflight cannot authorize a larger roster.

Reserve the representative seeds and the entire stress selection pool from training generation. Log every development comparison and selection decision. Repeated tuning can overfit these panels, so use a fresh holdout for confirmation.

After a complete preflight, run development explicitly:

```sh
python3 -B research/evaluation_tools/v2/panel_runner.py run --freeze artifacts/evaluation-v2/inputs/FREEZE.json --phase development --preflight artifacts/evaluation-v2/preflight --out artifacts/evaluation-v2/development --workers 4
python -B research/evaluation_tools/v2/analyze_panels.py --freeze artifacts/evaluation-v2/inputs/FREEZE.json --preflight artifacts/evaluation-v2/preflight --development artifacts/evaluation-v2/development --out artifacts/evaluation-v2/results
```

Analysis may run on a different host from game execution. It verifies the recorded runner environment against preflight, and records its own Python and NumPy versions. Both primary and diagnostic games must finish before it reports a complete comparison. Diagnostic opponents never enter the primary overall score, condition comparisons or market summaries. Source groups describe known overlap; unclassified groups do not establish independent ancestry. See [ANALYSIS_PLAN.md](ANALYSIS_PLAN.md).

To test an unseen opponent population, prepare a separate bundle with a new pinned 16-entry opponent manifest and matching runtime files. Pass `--bundle PATH` before the runner's `freeze` or `run` command, complete its own preflight, and report it as a separate comparison. The five related stress controllers added here provide economic diagnostics; new opponent families still need their own evidence.

## Resume and repair

Repeat the same run command to resume. V2 checks the complete freeze, exact job list, environment, worker count, timeout and preflight receipts before reading or appending results. It repeats the input check after execution. A changed unplayed seed, changed opponent, changed analysis file or changed preflight receipt blocks continuation. Analysis enforces the same checks.

Each output directory has an exclusive `RUNNING.json` file while active. After an unclean process exit, first verify its recorded PID is no longer running before removing that one lock file. A partial final JSON line or a failed game blocks reuse; preserve the journal, document the cause and create a new versioned run. Never pick the more favorable retry. Do not modify frozen inputs to get a failed check to pass.

## Holdout

The old release cannot authorize v2. Do not use its seed values for another confirmation while the original comparison is running. After that comparison finishes, the custodian can supply its completed holdout `STATUS.json` and exact retired manifest. A new release also requires a new 256-seed manifest and a fresh campaign audit started after the v2 freeze. The audit must pass and include NPZ and remote registry checks; its `exclusions.json` must sit beside the report.

`holdout.py --help` lists these explicit inputs. Release checks disjointness against the retired original holdout, all three public seed manifests and the fresh campaign exclusions. The release pins its supporting files and belongs to one freeze. A holdout run claims one output directory, and only that run may resume it. Analysis of a holdout requires both `--holdout RUN_DIRECTORY` and `--release RELEASE.json`. Follow the same retirement rule for every later holdout; include all earlier retired sets in the campaign exclusions. This procedure depends on a complete audit and an honest custodian, not secret filesystem access.

The local 180-second game timeout and measured candidate-policy latency do not certify Kaggle's packaged execution limits. A separate packaged-agent runtime test remains required before submission. No command here submits an agent or promotes a champion.

## Tests

From the repository root:

```sh
python -B -m unittest discover -s research/evaluation_tools/v2/tests -v
```

The tests exercise input mutation, resume rejection, incomplete/duplicate results, preflight binding, holdout reuse and analysis separation. Synthetic results are test fixtures only. Full-game parity receipts are separate evidence under `verification/` when published.
