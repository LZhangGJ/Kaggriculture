# Kaggriculture local simulation lab

This workspace contains the official Kaggriculture 1.32.6 engine, selected public
research notebooks, and a fast local runner for large self-play experiments.

## What is where

- `official/competition/` — competition-provided rules and agent guide
- `vendor/kaggle-environments/` — Kaggle's official engine source
- `research/notebooks/` — selected Discussion-linked and high-value public notebooks
- `src/kaggriculture_lab/fast_env.py` — low-overhead CPU training runner
- `src/kaggriculture_lab/gpu_engine.py` — pure-tensor CUDA transition engine
- `benchmarks/benchmark_gpu_engine.py` — CUDA transition throughput benchmark
- `tests/test_gpu_engine.py` — stepwise differential tests against the official runner

## Environment

The local virtual environment is pinned to `kaggle-environments==1.32.6`:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe ".[dev]"
```

Use a normal wheel install here. On Windows, an editable install writes a `.pth`
containing the workspace's Chinese path, which Python 3.11 may decode with CP932 and
reject during startup.

## Run one fast episode

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -c "from kaggriculture_lab import run_fast_episode; print(run_fast_episode(seed=1))"
```

Custom agents may be built-in names, paths to Python files exposing `agent(obs)`, or
`module:callable` references:

```python
from kaggriculture_lab import run_fast_episode

result = run_fast_episode("agents/my_agent.py", "starter", seed=9001)
print(result.rewards, result.statuses)
```

## CPU tournament

Use spawn-safe string agent specifications when `workers > 1`:

```python
from kaggriculture_lab import run_duel

games = run_duel(
    "agents/candidate.py",
    "agents/baseline.py",
    seeds=range(10_000, 10_100),
    both_seats=True,
    workers=8,
)
```

## Pure-tensor GPU engine

`src/kaggriculture_lab/gpu_engine.py` is a fixed-shape structure-of-arrays rewrite
of the official 1.32.6 transition. Thousands of complete games live on one CUDA
device; `reset()` and `step()` never construct observation dictionaries. It covers
unit actions, crops, animals, inventory, hiring, land, per-unit market matching,
town demand, decay, daily refresh, and terminal rewards.

The official dictionary interpreter is still the submission oracle. The tensor
engine specializes the competition defaults (10x10, two players), caps simultaneous
hands and per-order quantities at configurable RL-safe bounds, and uses a stateless
GPU RNG with the same distributions but not Python `random.Random` bit identity.
With random events disabled, deterministic differential tests compare it directly
against the official engine.

Related GPU components:

- `src/kaggriculture_lab/gpu_policy.py` encodes dictionary observations, builds legal
  action masks, batches all farmers/hands/market decisions, and runs a shared
  PyTorch policy/value network.
- `benchmarks/benchmark_gpu_engine.py` measures pure CUDA transition throughput.
- `benchmarks/benchmark_gpu_policy.py` measures the fallback CPU-transition plus
  CUDA-policy path.
- `scripts/train_gpu_bc.py` is a warm-start trainer that clones trusted scripted
  agents before replay BC or self-play RL.

Linux/CUDA quick start:

```bash
conda create -y -n kaggriculture python=3.11
conda run -n kaggriculture python -m pip install -e '.[dev,gpu]'
conda run -n kaggriculture pytest -q
conda run -n kaggriculture python benchmarks/benchmark_gpu_engine.py --envs 16384 --steps 700 --hands 0 --profile move
conda run -n kaggriculture python scripts/train_gpu_bc.py --envs 128 --updates 2000
```

For two GPUs, launch independent actor/trainer processes with disjoint seeds and
`CUDA_VISIBLE_DEVICES=0` / `CUDA_VISIBLE_DEVICES=1`.  A single process intentionally
uses one GPU so transition state and policy tensors never cross devices.

On `doraemon03` the deployed checkout is `/homes/lzhang/Kaggriculture`. The reusable
environment is `.venv`, backed by PyTorch 2.5.1 + CUDA 11.8 from the existing
`trans` environment:

```bash
cd /homes/lzhang/Kaggriculture
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/src \
TRITON_CACHE_DIR=/tmp/lzhang-kaggriculture-triton-v4 \
  .venv/bin/python benchmarks/benchmark_gpu_engine.py \
  --envs 16384 --steps 700 --device cuda --profile mixed --hands 16
```

Measured on an RTX 3090 while GPU 1 had only about 1.26 GiB free because another
job owned most of the card:

| Batch | Hands/player | Workload | Equivalent 720-turn games/s |
|---:|---:|---|---:|
| 16,384 | 16 | all units move, 700-turn run | 9,485.4 |
| 16,384 | 16 | farmer interacts, all hands move, 700-turn run | 8,049.6 |
| 16,384 | 16 | movement + one seed order every turn | 2,951.3 |
| 16,384 | 16 | every unit performs a fused board interaction | 7,699.6 |
| 16,384 | 16 | every unit performs PICKUP | 5,922.8 |
| 16,384 | 16 | every unit performs DROP | 1,119.5 |
| 16,384 | 16 | every unit performs PLACE | 903.5 |

Movement is fused across all active unit slots. Board and inventory interactions
retain official sequential unit order. DIG, WATER, HARVEST, FERTILIZE, BUILD,
FEED, COLLECT, CARE, and PLANT run through one optional Triton kernel per active
unit slot. DROP, PICKUP, and PLACE use separate operation-specialized Triton
variants so they do not inflate the common board kernel. Fixed-price seed and animal
orders are settled in a single exact batch; dynamically priced product buys and
sales retain per-unit matching. The figures measure transitions only and exclude
policy-network inference/training.

Against the optimized 8-process CPU runner at 57.25 games/s, the long-run movement
profile is about 166x faster, the mixed profile about 141x faster, and the fused
all-interaction profile about 134x faster. The first invocation JIT-compiles and
caches Triton kernels; benchmark warm-up excludes this one-time cost.

## Verification and benchmark

```powershell
$env:PYTHONPATH = "$PWD\src"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe benchmarks\benchmark_engine.py
```

On the current Ryzen 7 3800X workstation, a 1,024-game run reached 57.25 games/s
with 8 processes, versus about 0.51 games/s through the official full framework.
See `research/README.md` for the recorded setup and caveats.

The fast runner trusts agents: it skips schema validation, timeout enforcement,
defensive observation copies, replay recording, and log capture. Always validate a
candidate with the official runner before submission.
