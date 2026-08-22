# Official replay to V7 labels

The official daily replay dump is converted without duplicating observations.
Each compressed label bundle references its immutable raw JSON file by filename
and SHA-256 and contains:

- Kaggle's corrected action alignment: observation `t` uses the action stored at
  replay state `t + 1`.
- Player-specific private state with the shared player-1 `step` field restored.
- V7 Top-M inverse-planning task chains, task phase, ETA, route, and confidence.
- Per-step discounted liquidatable-net-asset value targets plus terminal outcome.
- Per-day soft budget targets: realized category spend is a lower bound and
  end-of-day cash is a reserve target.  Old bundles reconstruct these labels
  deterministically from the referenced raw replay when loaded.
- Exact trusted-interpreter resimulation audit and gold/silver/bronze quality tier.
- Team, action-trajectory hash, and behavior-cluster metadata for balanced sampling.

Public replays do not reveal an unambiguous high-level strategy-mode decision.
They therefore use `strategy_mode=-1` and explicitly disable initial-mode and
daily KEEP/SWITCH supervision.  Low-level task-chain and market actions remain
supervised.

Example:

```powershell
$env:PYTHONPATH = "src"
python scripts/build_official_v7_dataset.py `
  --replay-dir "D:\Kaggriculture\data\raw\replays\2026-08-21" `
  --source-date 2026-08-21 `
  --output-dir "D:\Kaggriculture\data\processed\official_v7_2026-08-21"
```

The default selection examines the 160 highest-average-score episodes and keeps
64 after per-team and exact-trajectory caps.  This prevents a large family of
cloned or near-identical public agents from dominating behavior cloning while
retaining a bounded relaxation pass when the public meta is unusually homogeneous.

Audit every compressed bundle and its referenced raw SHA-256 before training:

```powershell
$env:PYTHONPATH = "src"
python scripts/audit_official_v7_dataset.py `
  "D:\Kaggriculture\data\processed\official_v7_2026-08-21\manifest.json" `
  --output "D:\Kaggriculture\data\processed\official_v7_2026-08-21\audit.json"
```

Validate that the hierarchical BC entry point can join all raw observations to
their V7 labels without allocating a model:

```powershell
$env:PYTHONPATH = "src"
python scripts/train_hierarchical_bc.py `
  --official-v7-manifest "D:\Kaggriculture\data\processed\official_v7_2026-08-21\manifest.json" `
  --official-only `
  --validate-data-only
```

Remove `--validate-data-only` to train.  Official samples use a neutral
`BALANCED` mode only as network conditioning; `initial_mode_active` and
`daily_switch_active` remain false, so public replays never supervise an
invented strategy-mode or KEEP/SWITCH label.

The BC budget loss supervises reserve fraction, the spending-category simplex,
and minimum realized envelopes.  It regularizes the emergency allowance toward
zero instead of treating the expert's latent exact budget as observable.
