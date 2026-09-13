# Reproduce the completed comparison

Use Linux x86_64 and Python 3.12 for the pinned native policies. The recorded run used 16 local CPU workers. Candidate execution uses the standard library; result analysis also requires NumPy (recorded version 2.4.4). No GPU is required.

Run commands from the `research/evaluation_sets/2026-09-12-v1` directory. Choose an output directory outside this bundle. The runner verifies every runtime hash, resets policies for every game, and resumes only missing cells in the same versioned journal. It refuses duplicate cells and preserves invalid results.

```sh
python -B tools/check_bundle.py
python -B tools/validate_public.py
python -B evaluation/panel_runner.py --roster evaluation/roster.json --panels . --out /path/to/run/preflight --phase preflight --workers 16 --audit campaign_audit/report.json
python -B evaluation/panel_runner.py --roster evaluation/roster.json --panels . --out /path/to/run/development --phase development --workers 16 --audit campaign_audit/report.json --preflight /path/to/run/preflight/STATUS.json
mkdir -p sealed
cp retired_holdout/holdout.json sealed/holdout.json
python -B evaluation/panel_runner.py --roster evaluation/roster.json --panels . --out /path/to/run/holdout --phase holdout --workers 16 --audit campaign_audit/report.json --preflight /path/to/run/preflight/STATUS.json --release retired_holdout/RELEASE.json
python evaluation/analyze_panels.py --panels . --roster evaluation/roster.json --development /path/to/run/development --holdout /path/to/run/holdout --out /path/to/run/results
```

This repeats a known benchmark. It does not create fresh holdout evidence. The Windows campaign launcher records the original local paths and is provenance; use the portable runner commands above for another checkout.

Raw per-game records are in `results/raw/`. Each includes the seed, candidate and opponent identities, candidate seat, terminal cash, strict win/draw flags, action hash, timing, actual shop unlocks, and market summaries. Draws count as zero wins. `RESULTS.json` records all counts and paired seed-cluster intervals; the opponent tables are views of the same validated cells.

The market sampler reads 180 public pre-action observations, one every four turns. It records each product's prices, inventory and floor samples. Shop demand is action-dependent because occupancy changes the official weed/shop RNG consumption. The town center consumes 30 units of each of the eight non-fertilizer products during the default 719-transition game. No shop consumes melon. These demand counts do not measure either player's realized sales.

Optional terminal debug JSON is malformed for the frozen terminal-suffix native policy. The runner records the error, raw-byte hash and prefix; action decoding and terminal validation remain strict. The failed first preflight and successful repeated preflight are preserved under `evaluation/verification/`.
