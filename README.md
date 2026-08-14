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
- `benchmarks/benchmark_fixed_market_kernel.py` — isolated fixed-market Triton microbenchmark
- `benchmarks/profile_gpu_hotpath.py` — short CPU/CUDA operator-level profiler
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

For the native Windows GPU environment, run the pinned setup script. It installs
PyTorch 2.11 with CUDA 12.8 and the matching Triton-Windows 3.6 runtime, then checks
that CUDA and Triton can see the local GPU:

```powershell
.\scripts\setup_windows_gpu.ps1
```

On RTX 50-series GPUs, keep PyTorch at 2.7 or newer, Triton at 3.3 or newer, and
CUDA at 12.8 or newer. The script pins a tested combination rather than relying on
the default PyPI CPU wheel.

## Official replay data

The official daily replay index is the source of truth. This downloads the newest
published day by default, extracts it under `D:\Kaggriculture\data\raw\replays\YYYY-MM-DD`, and
validates both the episode count and total bytes against the official manifest:

```powershell
.\scripts\download_official_replays.ps1
```

Pass `-Date YYYY-MM-DD` for a specific day, or `-Destination D:\path\to\replays`
for another D-drive location. The repository's `data` entry is a directory junction
to `D:\Kaggriculture\data`, so existing relative paths continue to work without
storing replay files on C. Raw replay data is intentionally ignored by Git; a single
extracted daily dataset is roughly 20 GiB.

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
engine specializes the competition defaults (10x10, two players) and caps
simultaneous hands and per-order quantities at configurable RL-safe bounds. Daily
weed and town-unlock events consume the same Python `random.Random` stream as the
official engine; full-season differential tests compare every observable field,
reward, status, and terminal flag step by step.

The CUDA engine accepts the official runner's 31-bit episode-seed domain
(`0 <= seed < 2**31`). Seeds outside that range are rejected instead of silently
producing a different random stream.

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

The native Windows setup was validated on an RTX 5070 Ti (16 GiB) with PyTorch
2.11.0+cu128 and Triton-Windows 3.6.0. A 4,096-environment movement benchmark
reached a median 24.87 million joint turns/s, equivalent to 34,538 complete
720-turn games/s; all 24 CPU, CUDA, Triton, and official-engine differential tests
passed. Use short cache paths outside the OneDrive checkout as configured by the
setup script.

On `doraemon02` the previous deployed checkout is `/homes/lzhang/Kaggriculture`. The reusable
environment is `.venv`, backed by PyTorch 2.5.1 + CUDA 11.8 from the existing
`trans` environment:

```bash
cd /homes/lzhang/Kaggriculture
CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 PYTHONPATH=$PWD/src \
TRITON_CACHE_DIR=/tmp/lzhang-kaggriculture-triton-v6 \
TORCHINDUCTOR_CACHE_DIR=/tmp/lzhang-kaggriculture-inductor-v1 \
  .venv/bin/python benchmarks/benchmark_gpu_engine.py \
  --envs 16384 --steps 700 --device cuda --profile mixed --hands 16
```

Measured on an RTX 3090 while GPU 1 had only about 1.26 GiB free because another
job owned most of the card:

| Batch | Initial hands/player | Workload | Equivalent 720-turn games/s |
|---:|---:|---|---:|
| 16,384 | 0 | move actions, 3x700-turn median | 35,976.1 |
| 16,384 | 16 | mixed board/move, 700 turns with official daily hand reset | 31,912.7 |
| 16,384 | 16 | all 17 units move, saturated 23-turn day | 25,935.5 |
| 16,384 | 16 | farmer interacts and 16 hands move, saturated 23-turn day | 22,166.7 |
| 16,384 | 0 | one fixed-price seed order, 3x700-turn median | 31,630.9 |
| 16,384 | 16 | every unit performs a board interaction, saturated 23-turn day | 15,367.5 |
| 16,384 | 16 | every unit performs PICKUP | 5,922.8 |
| 16,384 | 16 | every unit performs DROP | 1,119.5 |
| 16,384 | 16 | every unit performs PLACE | 903.5 |
| 16,384 | 0 | buy 16 dynamically priced products, 3x700-turn median | 31,379.2 |
| 16,384 | 0 | sell 16 dynamically priced products, 3x700-turn median | 33,140.2 |

Movement is fused across all active unit slots. Board and inventory interactions
retain official sequential unit order. DIG, WATER, HARVEST, FERTILIZE, BUILD,
FEED, COLLECT, CARE, and PLANT run through one optional Triton kernel per active
unit slot. DROP, PICKUP, and PLACE use separate operation-specialized Triton
variants so they do not inflate the common board kernel. A market-only compiled
router skips empty order/type combinations. HIRE, BUY_LAND, BUY_SEED, and
BUY_ANIMAL each use an operation-specialized Triton launch for an active order.
Dynamically priced product buys and sales fuse up to 16 sequential quotes and
settlements into one Triton launch while preserving the official shared
pre-settlement price snapshot for both players. Their maximum round count is
returned by the same compact market control transfer, avoiding an extra device
synchronization. The
town-demand update and all nine price refreshes share one phase-specialized Triton
launch. Movement, the full-board plant decay scan, and end-of-day board/farm reset
also use fused kernels. The figures measure transitions only and exclude
policy-network inference/training. Long-run rows retain the official rule that
hired hands are reset each day; the 23-turn rows isolate a saturated 17-unit day.
The sparse action classifier is a fixed-shape `torch.compile` graph; use
`--no-compile-routing` for faster cold starts at lower steady-state throughput.

For RL workloads bounded to 2 market orders and quantities up to 16, enable the
single-kernel market interpreter. It processes every fixed and dynamically priced
order on device in exact order, removing the second market routing transfer while
leaving no-market turns on the smaller action-routing graph:

```python
config = GpuEngineConfig(
    max_market_orders=2,
    max_market_quantity=16,
    fuse_market_kernel=True,
)
```

The general operation-specialized path remains the default for configurations that
need quantities above 16. The benchmark equivalent is `--fuse-market-kernel`.

Against the optimized 8-process CPU runner at 57.25 games/s, the long-run movement
profile is about 628x faster, the mixed profile about 557x faster, and the saturated
all-interaction profile about 268x faster. The first invocation JIT-compiles and
caches Triton/Inductor kernels; benchmark warm-up excludes this one-time cost. New
fixed batch/configuration variants can take several minutes to compile, so
long-running actors should reuse both cache directories shown above.
For shared servers, `benchmark_gpu_engine.py --warmup-steps 25 --repeats 3`
reports the median of several timed runs without repeating process/JIT startup.

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
