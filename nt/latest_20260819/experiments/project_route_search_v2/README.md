# Project Route Search V2

This package is the implementation home for the V1.1 three-layer Kaggriculture
route-search design:

```text
economic project -> obligations and scheduling -> legal primitive actions
```

## Current status: M2.6 accepted

M0 freezes interfaces and experiment contracts. M1 adds:

- fixed-shape JAX schema declarations;
- route-family, domain, species-coverage, budget, and performance-gate configs;
- a CPU and JAX `NullOpponent` implementation;
- cumulative search panels `16 -> 32 -> 128` plus an independent official
  64-seed holdout;
- a deterministic freeze tool, machine-readable receipt, and tests.
- episode-safe controller factories and exact state snapshots;
- terminal record and inactive-hand cleanup;
- per-lane episode reset inside a `State + Controller` `lax.scan` carry;
- a poison-before-reset audit covering both seats, 1,000 episodes, batch 8,
  and 719 controller steps per episode.

M2 adds one deliberately small, fully executable crop-business loop:

- one `CROP_LOT` project in controller slot zero;
- fixed-shape crop obligations for seed purchase, planting plus same-day water,
  daily water, harvest, shed deposit, sale, and terminal liquidation;
- a NullOpponent 719-step JAX rollout using the existing simulator and the
  previously parity-tested V5 primitive action compiler;
- 100 frozen-seed semantic acceptance and an RTX 3090 performance receipt.

It **does not** yet contain animal projects, multi-project scheduling, a
candidate generator, route search, official Python step parity for the new M2
controller, or a high-potential route. M2 is a minimal wheat closure test, not a
competitive strategy result.

M2.5 expands the crop-side controller to the old R2 candidate-101 operating
plan without copying the old executor:

- staged tomato capacity `12 -> 18 -> 28` and wheat support `0 -> 8 -> 6`;
- daily workforce targets `9 -> 11 -> 7`;
- second-land purchase on day 6 with a cash floor;
- collision-free multi-unit watering, harvesting, planting, and deposit tasks;
- ordered cash-safe SELL, BUY_LAND, BUY_SEED, and HIRE market plans;
- sell cadence, investment stop, and terminal liquidation;
- exact official Python 1.32.7 full-state step parity on two seeds and both
  seats (2,880 frames including initial states).

M2.5 is an execution-capacity milestone, not a claim that fixed large-scale
tomato production is robustly profitable. On the frozen 100-seed NullOpponent
panel, market-demand variation makes its median cash close to the smaller M2
controller even though the upper tail is much higher. Route/threshold search is
therefore still required.

M2.6 replaces the historical primary/support aliases with one unified
`[batch, phase, crop]` genome and makes the crop search surface executable:

- all five crops through the same project/task/action path;
- one to six phases with expansion, shrink, stop and restart;
- target land/workforce, per-crop planting/harvest/fertilizer controls;
- four coarse layouts, weed recovery, active abandonment and maintenance
  admission;
- per-product sell cadence, floor and fraction, exact sequential market
  quotes, shed pressure, cash commitments and terminal liquidation;
- a deterministic crop-genome sampler and a 64-candidate search smoke.

All 25 searchable fields have schema, activation and behavior-effect receipts.
Representative full seasons, a 256-season broad random smoke, exact official
Python 1.32.7 seat-swapped parity and a same-contract 2048-batch performance
comparison all pass. This proves the major crop branches are wired and
searchable; it does not prove route profitability or competitive strength.

The source-of-truth design is:

`gpt_review/gpt/KAGGRICULTURE_E0_THREE_LAYER_PROJECT_SEARCH_DESIGN_V1_1_ZH.md`

The rules reference is Kaggriculture `1.32.7`. The source event bank is reused
without regeneration:

`experiments/route_playbook_v1/artifacts/events/event_panels_v1.npz`

## Reproduce M0

Run from the repository root with the repository virtual environment:

```powershell
& '.venv\python.exe' 'experiments\project_route_search_v2\tools\freeze_e0_contract_v1.py' `
  --event-bank 'experiments\route_playbook_v1\artifacts\events\event_panels_v1.npz' `
  --output 'experiments\project_route_search_v2\configs\seed_panels_v1.json'

& '.venv\python.exe' -m pytest 'experiments\project_route_search_v2\tests' -q `
  -c 'experiments\project_route_search_v2\pyproject.toml'
```

Expected generated artifacts:

- `configs/seed_panels_v1.json`
- `artifacts/events/e0_seed_panels_v1.npz`
- `receipts/m0_contract_freeze_v1.json`

Reproduce the M1 stress audit for either seat:

```powershell
& '.venv\python.exe' `
  'experiments\project_route_search_v2\tools\run_m1_controller_reset_stress_v1.py' `
  --batch-size 8 --episodes 1000 --steps-per-episode 719 --player 0 `
  --output 'experiments\project_route_search_v2\receipts\m1_controller_reset_stress_v1.json'
```

Reproduce the formal M2 GPU acceptance from WSL2:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc `
  'cd /mnt/e/ai_coding/kaggle/kaggriculture && `
   export XLA_PYTHON_CLIENT_PREALLOCATE=false && `
   export PYTHONPATH=/mnt/e/ai_coding/kaggle/kaggriculture/experiments/project_route_search_v2/src:/mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim/src:/mnt/e/ai_coding/kaggle/kaggriculture/experiments/strategic_v5/src && `
   gpu_sim/.venv-wsl-jax011-cuda12/bin/python `
   experiments/project_route_search_v2/tools/run_m2_crop_acceptance_v1.py `
   --semantic-seasons 100 --performance-batch 2048 --steps 719 `
   --repetitions 5 --crop-id 0 --target-tiles 4 --player 0 --require-gpu `
   --output experiments/project_route_search_v2/receipts/m2_crop_acceptance_v1.json'
```

M2 machine receipt:

- `receipts/m2_crop_acceptance_v1.json`

M2.5 machine receipts:

- `receipts/m25_acceptance_v1.json`
- `receipts/m25_parity_trace_generation_v1.json`
- `receipts/m25_official_parity_v1.json`

M2.6 machine receipts:

- `receipts/m26_parameter_effect_coverage_v1.json`
- `receipts/m26_branch_coverage_v1.json`
- `receipts/m26_crop_search_smoke_v1.json`
- `receipts/m26_parity_trace_generation_v1.json`
- `receipts/m26_official_parity_v1.json`
- `receipts/m26_performance_comparison_v1.json`

M2.6 acceptance report:

- `reports/M26_UNIFIED_CROP_SEARCH_SURFACE_ACCEPTANCE_20260818_ZH.md`

M3.5 crop/animal resource-loop receipts:

- `receipts/m35_branch_coverage_v1.json`
- `receipts/m35_random_safety_v1.json`
- `receipts/m35_performance_comparison_v1.json`
- `receipts/m35_parity_trace_generation_v1.json`
- `receipts/m35_official_parity_v1.json`
- `artifacts/traces/m35_parity_trace_v1.npz`

M3.5 acceptance report:

- `reports/M35_CROP_ANIMAL_RESOURCE_LOOP_ACCEPTANCE_20260818_ZH.md`

M3.5 route-quality objective is terminal bank cash.  Terminal inventory/map
value is retained as a soft economic diagnostic, not a hard correctness gate;
rule, ledger and unexplained execution errors remain hard failures.

## Frozen truth boundary

Any future search result is, at most, `BEST_OBSERVED_WITHIN_FROZEN_DOMAIN`
until it passes the official Python 1.32.7, unseen 64-seed, seat-swapped
acceptance route. NullOpponent results measure uncontested economic potential,
not competitive strength.
