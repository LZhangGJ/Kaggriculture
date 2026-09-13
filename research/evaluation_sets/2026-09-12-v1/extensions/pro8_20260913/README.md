# Pro8 comparison on the shared benchmark

The [master leaderboard](../../results/ALL_RESULTS.md) combines A08 r11, A06 r6 and A06 r12 calendar with the six agents from the first comparison. Each new agent plays 20,480 games: 8,192 representative, 4,096 stress and 8,192 retired-holdout games. The extension adds 61,440 games, bringing the combined page to 184,320.

The representative panel measures performance on uniformly drawn eligible seeds against the fixed opponent mix. Stress covers selected economic extremes and contrasts. Its average is not a population estimate. The holdout is now a reused public benchmark for Pro8; these results do not provide fresh holdout confirmation. Do not pool the three panels into one score.

All three Pro8 packages come from [commit d65325334067cc6c2b616f346bc43231b1c11190](https://github.com/LZhangGJ/Kaggriculture/tree/d65325334067cc6c2b616f346bc43231b1c11190/research/agents/pro8_20260913). The evaluation calls each supplied `create_agent()` entry with its own codec, configuration and native library. It preserves the supplied bytes. Exact identities are in [COHORT.json](COHORT.json) and [inputs/roster.json](inputs/roster.json).

The frozen roster retains some descriptive fields inherited from the first campaign, including its selection text and total-game field. Those fields describe the original campaign. The authoritative Pro8 scope is the three `candidates`, the exact [PLAN.json](inputs/PLAN.json), this cohort receipt and [ANALYSIS_PLAN.md](tools/ANALYSIS_PLAN.md).

## Execution and validation

The run uses the same pinned official 1.32.7 rules, 16 opponents, both seats and 719 transitions per game. Two hosts each run 16 CPU workers. Workers recycle after 64 games. The local shard contains 20,544 games; WRX90 contains 40,896. Whole seed blocks use a fixed modulo-three assignment, independent of outcomes. No original baseline game is rerun.

Before primary play, each host completes 384 cases that compare the native simulator with official observations and a direct policy run. Both hosts agree exactly on actions, terminal cash, shops and market summaries: 1,536 full verification games and 1,105,920 official observation checks. These games do not enter the win rates. See [CROSS_HOST_PREFLIGHT.json](CROSS_HOST_PREFLIGHT.json).

The [join receipt](../../results/master_data/JOIN_VALIDATION.json) verifies 184,320 unique terminal games with no invalid, missing or duplicate cells. It also requires every original count, mean, confidence interval, paired difference, diagnostic, reference condition and realized-market result to match the original publication exactly. The new analysis retains the original 4,000 common seed bootstrap draws and all metric definitions. Draws count as zero wins. Pairwise intervals use matched seeds; repeated subgroup comparisons remain exploratory.

The [freeze](inputs/FREEZE.json) covers the runtime files, opponents, seeds, reference features, original raw journals, analysis plan and tool code. The [run records](runs/) contain separate manifests, status receipts and compressed game journals for each host and phase. The first six agents' [original data](../../results/RESULTS.json) and [original page](../../results/ALL_RESULTS_ORIGINAL_SIX.md) remain available.

## Rebuild results from the recorded games

Use Python 3.12 and NumPy 2.4.4, the recorded analysis version. Run from a full Git checkout that contains baseline commit `86cf19cd528cb071a1fed019e3bc84e45a007e62`. The preparation script uses that commit for the original bundle, adds the frozen Pro8 runtime and restores the recorded journals. It requires a new output directory and starts no games.

```sh
python research/evaluation_sets/2026-09-12-v1/extensions/pro8_20260913/reproduce.py --out /path/to/pro8-reproduction
python -B /path/to/pro8-reproduction/join_and_analyze.py
python /path/to/pro8-reproduction/render_master_leaderboard.py --results /path/to/pro8-reproduction/results --validation /path/to/pro8-reproduction/JOIN_VALIDATION.json --out /path/to/pro8-reproduction/ALL_RESULTS.md
```

The join writes to a new `joined/` directory and refuses to overwrite one. `RESULTS.json` includes creation time and run provenance, so its full-file hash may differ on reproduction; the original-candidate reconciliation and all numeric results must still match. Relative links in the rendered page target the repository layout.

## Repeat the games

Native execution requires Linux x86_64 and Python 3.12. Prepare a separate root on each host with `reproduce.py --out /path/to/new-run --inputs-only`. On the local host run `python -B /path/to/new-run/run_host.py --host local --phase preflight`; on the second host use `--host wrx90`. Keep the recorded limits: 16 workers per host, no GPU, and a separate WRX90 unit capped at 16 GiB with no swap. Give each run a new directory.

Copy the completed second host's `runs/preflight-wrx90/` directory to the first root. Run `check_cross_host.py` there and require `PASS`. Copy its `CROSS_HOST_PREFLIGHT.json` to the second root. Then run `run_host.py --host local --phase benchmark` on the first host and `run_host.py --host wrx90 --phase benchmark` on the second. Preserve failed evidence; do not select favorable retry outcomes. Once both finish, copy the second host's completed benchmark directory and host status to the first root, then run the join and renderer above.

Repeating these games reproduces a known benchmark. A fresh confirmation claim needs a new independent holdout. Timing from concurrent local and WRX90 runs does not establish Kaggle runtime compliance, and this evaluation makes no submission or champion change.
