# Kaggriculture JAX GPU simulator

This directory contains a parity-first, fixed-shape JAX rewrite of the official
`kaggle-environments==1.32.6` Kaggriculture interpreter for RTX 3090 self-play.
The official implementation under `reference/` is immutable and remains the
source of truth.

## Verified status

- Frozen official source/config hashes: `receipts/reference_verification.json`.
- Three canonical 720-frame traces, including the boatlee v16 agent: exact.
- 100 unseen seeds (`10000..10099`), 72,000 official frames: exact integer,
  market price, status, and reward parity.
- 16 state-aware randomized seeds with deliberately invalid actions, 11,520
  frames and up to 10 hands: exact parity.
- Static hand bound: 32 hands / 33 units. Across 119 recorded seasons the
  observed maximum is 14; all cap-hit counters are zero.
- GPU-resident heterogeneous-checkpoint Arena and homogeneous self-play PPO
  collection/update are implemented and tested.

The authoritative machine-readable evidence is under `receipts/`. Do not train
against a simulator revision unless the parity receipts have been regenerated
and pass.

## Throughput on the local RTX 3090

Compilation and the first execution are excluded. One environment transition
advances both players by one turn; player-samples/s is exactly twice the listed
environment rate.

Full official season (719 actual interpreter transitions, 720 recorded frames),
simulator-only:

| Batch | Env transitions/s | Player samples/s | Peak JAX memory |
|---:|---:|---:|---:|
| 256 | 187,548 | 375,096 | 95 MB |
| 1,024 | 630,237 | 1,260,473 | 95 MB |
| 4,096 | 1,561,634 | 3,123,269 | 283 MB |

The minimum `50,000/s` gate and formal `300,000/s` target both pass. The exact
receipt is `receipts/benchmark_simulator_only_full_season.json`.

With a 608,029-parameter `(128, 128)` policy, 3,301 actor-visible inputs, two
market slots, action sampling/inference, and environment stepping all on GPU:

| Batch | Policy + simulator env/s | Full PPO env/s |
|---:|---:|---:|
| 256 | 23,660 | 17,429 |
| 1,024 | 79,035 | 57,291 |
| 4,096 | 188,236 | not measured (rollout storage is unnecessary at this size) |

`Full PPO` includes self-play collection, GAE, clipped policy loss, value loss,
entropy, backpropagation, gradient clipping, and Adam. See
`receipts/benchmark_policy_and_ppo.json`.

## Runtime setup

JAX CPU can run on native Windows, but the NVIDIA CUDA backend is not supported
natively. Use Ubuntu 24.04 under WSL2 for the RTX 3090 path.

From PowerShell:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim; python3 -m venv .venv-wsl; .venv-wsl/bin/python -m pip install --upgrade pip; .venv-wsl/bin/python -m pip install -r requirements-wsl.in; .venv-wsl/bin/python -m pip install -e .'
```

The resolved environment is frozen in `requirements-wsl.lock.txt`. Verify the
actual CUDA device:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim; export XLA_PYTHON_CLIENT_PREALLOCATE=false; .venv-wsl/bin/python tools/verify_jax_runtime.py'
```

Expected backend/device: `gpu`, `cuda:0`, NVIDIA GeForce RTX 3090.

## Exactness design

The hot path is a pure JAX pytree with fixed tensor shapes and integer/bool game
state:

- `State`: both farms, 10x10 tile tensors, 33 unit slots, ordered carried
  inventories, private shed/seeds, shared market/town, rewards and diagnostics.
- `Action`: 33 unit actions and 10 ordered market slots per player.
- `step_env`: independent pure function suitable for `jit` and general `vmap`.
- `batched_step_sync`: high-throughput synchronized self-play path. Day/terminal
  control flow is outside `vmap`, while each environment remains independent.
- `lax.scan`: complete device-resident seasons and policy rollouts.

Two Python-specific behaviors are frozen exactly rather than approximated:

1. `market_price` uses Python binary64 math and bankers `round`. The generator
   evaluates official code into an int16 LUT for inventories `-32768..65535`.
   `price_lut_oob` must remain zero.
2. End-of-day randomness uses Python `random.Random((seed*1000003)^day)` and
   consumes draws only for empty tiles. The frozen event bank stores each of the
   200 possible conditional weed draws and the exact `random.choice` result for
   every possible consumed-draw count, so arbitrary policies preserve the
   official MT19937 stream. The shipped bank contains 256 seeds: `0..127` for
   training/development and `10000..10127` for held-out verification. A seed
   outside that bank is rejected instead of silently using the wrong random
   stream; add new seeds in `tools/generate_static_tables.py` and regenerate the
   table before using them.

The official framework records frame 0 before any action and ends at frame 719;
there are therefore 720 states but 719 calls to the interpreter. All benchmark
and parity reports state which unit they count.

## Hand bound

`MAX_HANDS=32` (`MAX_UNITS=33`). Hiring 32 hands in one day costs 5,702,886 at
the official Fibonacci multiplier; attempting a 33rd from zero hires requires
9,227,464 total cash. A request beyond the tensor bound is a no-op and increments
`hand_cap_hits`. The same action/state shape is used in simulator rollout, Arena,
PPO, and exported policy code. Evidence: `receipts/hand_cap_analysis.json`.

## Main commands

Run tests:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim; export XLA_PYTHON_CLIENT_PREALLOCATE=false; .venv-wsl/bin/pytest -q'
```

Regenerate the exact derived tables from the frozen official source (native
official environment):

```powershell
& 'E:\ai_coding\kaggle\kaggriculture\.venv\python.exe' 'E:\ai_coding\kaggle\kaggriculture\gpu_sim\tools\generate_static_tables.py'
```

Verify the 100 held-out seeds and randomized differential corpus:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim; export XLA_PYTHON_CLIENT_PREALLOCATE=false; .venv-wsl/bin/python tools/verify_heldout_parity.py; .venv-wsl/bin/python tools/verify_random_differential.py'
```

Benchmark:

```powershell
wsl.exe -d Ubuntu-24.04 -- bash -lc 'cd /mnt/e/ai_coding/kaggle/kaggriculture/gpu_sim; export XLA_PYTHON_CLIENT_PREALLOCATE=false; .venv-wsl/bin/python tools/benchmark_simulator.py --batch-sizes 256 1024 4096 --rollout-steps 719 --repetitions 5 --receipt receipts/benchmark_simulator_only_full_season.json; .venv-wsl/bin/python tools/benchmark_policy_training.py'
```

## Layout

- `reference/`: immutable official snapshots, derived exact tables, and traces.
- `src/kaggriculture_jax/`: state/action types, rules, codec, policy, Arena, PPO.
- `tests/`: reset, invalid action, batching, canonical parity, Arena, PPO tests.
- `tools/`: table/trace generators, parity verifiers, profilers and benchmarks.
- `receipts/`: versioned machine-readable evidence.
