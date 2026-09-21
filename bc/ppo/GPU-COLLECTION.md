# GPU collection

This adapter keeps the exact BC model and action grammar. JAX advances the game;
Torch builds observations, resolves worker requests, decodes actions, and stores
the rollout on the GPU. DLPack shares simulator state between the two libraries.

`--collector gpu` selects this path in `ppo.train`. The default remains the
official engine. GPU collection accepts neural opponents only. Python notebook
agents still belong in the official-engine evaluation panel.

## Runtime

The tested environment is Torch 2.11.0 with CUDA 13 and JAX/jaxlib 0.11.0.
Set `PPO_JAX_SITE_PACKAGES` only when JAX comes from a separate compatible
environment. Set `PPO_JAX_SIM_SRC` to the patched simulator's `src` directory.
The checkpoint contract records simulator source and static-table hashes.
Do not substitute the upstream simulator without checking current-engine parity.

Run from the repository root with `PYTHONPATH=bc`:

```sh
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TORCHINDUCTOR_COMPILE_THREADS=8
export PPO_COMPILE_WORKERS=1
export PPO_COMPILE_MARKET=1
export PPO_WORKER_BUCKETS=1
export PPO_DONATE_SIM_STATE=1
export PPO_FAST_MARKET_HEAD=1
export PPO_FAST_WORKER_HEAD=1
export PPO_ROLE_WORKER_BUCKETS=1
export PPO_FINE_WORKER_BUCKETS=1
export PPO_SPLIT_BURN_ENCODER=1
export PPO_SEQUENCE_GRU=1
export PPO_PACK_STATS=1
export PPO_FUSED_ADAM=1
python -m torch.distributed.run --standalone --nproc_per_node=2 -m ppo.train \
  --collector gpu --bc-reference /path/to/exact-bc.pt \
  --league /path/to/neural-league.json --output /new/output/directory \
  --games 512 --workers 8 --minibatch 128 --updates 2
```

This is a bounded integration-test command, not a selected production recipe.
`games`, `workers`, and `minibatch` are global across ranks. Each rank owns its
games and neural opponents. Each model owns separate CUDA graph buffers.
GPU sharding balances each opponent family across ranks. This preserves the
global games, seeds, and opponent mix while keeping policy batch shapes stable.
Resume through the existing `--resume` interface; the contract checks simulator,
model, league, configuration, code, and runtime identity.

## What stays on the CPU

Fresh official-engine random-event streams are prepared before collection.
The CPU also schedules kernels, assigns opponents, saves checkpoints, and runs
official-engine evaluation. With worker buckets enabled, one scalar worker-count
read per turn selects a captured decoder width. Observations and worker
resolution remain on the GPU. Disable buckets for a fixed 33-worker graph.

## Correctness and timing

- Full collection means 719 transitions and terminal games, not simulator steps.
- Features preserve the BC checkpoint's representation and private-information
  boundaries. Recurrent anchors and probabilities stay FP32.
- Disable the eval-only Transformer fast path so collection and differentiable
  replay use the same attention path. TF32 and BF16 did not pass density checks.
- The first PPO minibatch checks unchanged-policy log probabilities. Simulator
  capacity counters and terminal status are checked after each season.
- Compile time belongs to cold-start timing. Report warm collection separately
  from PPO, event preparation, checkpoint writes, and total wall time.
- A full PPO pass and an early KL stop are different workloads. Always report
  optimizer steps, trained learner turns, and whether KL stopped the update.

All benchmarks use disposable outputs. They do not establish playing strength.

A 1,024-game diagnostic completed its first full PPO pass but stalled during its
second update near GPU memory capacity. It was stopped and its completed
checkpoint preserved. Do not use its first result as evidence of sustained
1,024-game stability. The 512-game recipe completed two full PPO updates; its warm
batch with shared head projections took 129.94 seconds (62.16 collection,
67.70 PPO), or 3.94 full games/s across both GPUs. Finer per-role worker buckets,
split burn-in encoding, sequence GRUs and packed/fused updates reduced that to
121.52 seconds (57.83 collection, 63.61 PPO), or 4.21 games/s. Each update trained all
552,192 learner turns in 90 optimizer steps, with no KL stop.

The head adapters share repeated source, target, item and worker projections and
skip projections of absent market references. They preserve checkpoint weights
and action semantics. CPU output/gradient comparisons and both full GPU updates
passed; FP32 reassociation can change sampled trajectories (510 of 512 first-game
cash pairs matched the baseline). This is not a bitwise rollout guarantee.

Sequence GRUs share the checkpoint's original GRUCell parameters. Their first
forward runs during initialization, before CUDA graph capture, because PyTorch
may repack parameter storage. Keep that warm-up: capturing earlier left graphs
reading stale weights after an optimizer update. A regression check covers
storage stability, outputs and gradients, including the burn-in boundary.

PPO reports per-rank batch/forward/statistics/backward/optimizer timings.
Collection reports nested encoder/worker/market timings and worker-bucket counts.
Nested timings are components of role_decode, not extra elapsed time.
