# Unchanged public v37 on the shared benchmark

This extension adds Ahmed's unchanged public v37 to the [master leaderboard](../../results/ALL_RESULTS.md). It adds 20,480 games to the previous nine agents' 184,320 games. The combined page reports ten agents and 204,800 games, with overall, opponent and seat results for every panel, best rates in bold, minimum opponent rates, and a paired comparison with feed reserve.

The exact source is [main.py](runtime/public/ahmed-v37-original/main.py), SHA256 `94c1c2c05ae7cde8fca9ee3957b01112c1cbab82c7434aae6a5f24fa7485bc4c`. It is the literal Python file extracted from the public notebook, with all original author and license notices retained. [Source provenance](runtime/public/ahmed-v37-original/SOURCE.json) records the notebook. No policy source was changed. The lifecycle adapter matches the adapter used for v37 feed reserve and creates a fresh module for each game. Feed reserve adds one 77-line block to this parent; the matched comparison measures the effect of that addition on this benchmark.

The representative panel uses 256 uniformly drawn eligible seeds to estimate average performance against the fixed opponent roster. Stress uses 128 seeds selected for economic extremes and contrasts; its average is not a population estimate. The 256-seed holdout is now public and retired. Unchanged v37 and the earlier Pro8 additions reuse it as a known benchmark; they receive no fresh holdout confirmation. Each panel uses the same 16 pinned opponents and both seats: 8,192, 4,096 and 8,192 games per candidate. Do not pool the panels into one score.

## Execution and evidence

The policies use the pinned official 1.32.7 rules and 719 transitions per game. Each host first passes 128 full verification cases, comparing official observations and deterministic actions, cash, shops and market summaries. Both hosts must agree before primary play. This checks 512 full games and 368,640 official observations; verification games do not enter win rates. [Cross-host receipt](CROSS_HOST_PREFLIGHT.json).

Sixteen CPU workers on each host execute frozen, disjoint seed blocks: 6,848 games locally and 13,632 on WRX90. Workers recycle after 64 games. WRX90 runs in a separate unit with a 16 GiB memory cap and zero swap. The run uses no GPU. The exact [freeze](inputs/FREEZE.json), [job plan](inputs/PLAN.json), [analysis plan](tools/ANALYSIS_PLAN.md), [candidate roster](inputs/roster.json) and [raw game journals](runs/) preserve the evidence.

The join validates every expected candidate, seed, opponent and seat cell. It rejects invalid, missing and duplicate games and requires every previous candidate's published count, mean, confidence interval, paired difference and diagnostic to match exactly. The analysis retains the original 4,000 common whole-seed bootstrap draws. The [previous nine-agent cache](previous_nine/) preserves exact source bytes for reproduction; no previous games were rerun.

Minimum matchup rates combine both seats and take the lowest of the 16 opponent rates in each panel. The greater-than-50-percent criterion uses observed strict wins, with draws worth zero wins. It covers this fixed opponent roster, not a round robin among leaderboard candidates, and does not guarantee future wins. Repeated subgroup comparisons remain exploratory.

## Rebuild the report without playing games

Use Python 3.12 and NumPy 2.4.4. Run from a Git checkout containing baseline commit `86cf19cd528cb071a1fed019e3bc84e45a007e62`. Choose a new output directory.

```sh
python research/evaluation_sets/2026-09-12-v1/extensions/v37_original_20260913/reproduce.py --out /path/to/v37-reproduction
python -B /path/to/v37-reproduction/join_and_analyze.py
python /path/to/v37-reproduction/additional_comparisons.py
python /path/to/v37-reproduction/render_master_leaderboard.py --results /path/to/v37-reproduction/results --validation /path/to/v37-reproduction/JOIN_VALIDATION.json --out /path/to/v37-reproduction/ALL_RESULTS.md
```

The join refuses to overwrite an existing `joined/` directory. Result metadata includes creation time and run provenance, so the full result hash may differ on reproduction. Every numeric result and the previous-agent reconciliation must still match.

## Repeat the games

Prepare a new workspace on each Linux x86_64 host with `reproduce.py --inputs-only`. Run `run_host.py --host local --phase preflight` on the first host and `run_host.py --host wrx90 --phase preflight` on the second. Copy the second preflight directory to the first workspace, run `check_cross_host.py`, and require `PASS`. Copy its receipt back to the second host. Then run the corresponding `run_host.py --host HOST --phase benchmark` on each host, preserving the recorded resource limits. Copy the completed second-host journals and host status to the first workspace before joining.

Repeating these games reproduces a known benchmark. Fresh confirmation needs new independent seeds. These CPU runs do not establish Kaggle runtime compliance and make no champion change or submission.
