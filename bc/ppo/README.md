# Exact-action recurrent PPO

This package builds on the released BC model without changing its hashed source. It provides batched sampling, CPU actors, shared-memory numeric transfer, exact action-density replay, recurrent PPO, frozen BC regularization, a small opponent league, checkpoints, and official-game evaluation.

**Status (September 19 GPU tests):** CPU checks, strict-FP32 GPU density replay, the four-game official evaluator, and one/two-GPU complete-game updates pass. BF16 density parity failed: before any update, sampled versus replayed probability ratios differed by up to 11.1% on the test trace. Use FP32 with TF32 disabled (the trainer now enforces the latter); do not use `--bf16` for a campaign. Two-GPU resume passed (optimizer steps 15 to 30). The 64-game, 16-worker FP32 test completed all seasons and an update in 268.27 seconds (0.239 games/sec). Detailed tests are tracked under `/home/keith/kaggriculture-ppo-tests-20260919`. No production PPO campaign has been started. Small-batch GPU utilization is low; collection needs measured optimization.

## Environment

Use the parent BC README's Python 3.14, PyTorch 2.11 and official engine 1.32.7 environment. Run commands from `bc/`. No extra training framework is required. The checkpoint loader verifies the BC source, engine, architecture and quantity vocabularies.

## CPU checks while BC runs

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m ppo.test_cpu
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 -m ppo.test_ddp
```

These run short official-engine sequences and synthetic PPO targets, not full training games. They cover sampled action-density replay, independent canonical feature reconstruction, interior recurrent replay, finite gradients, zero-LR invariance, a nonzero update, checkpoint/optimizer serialization, GAE terminal semantics, shared-memory actor parity, disjoint game ownership, evaluation result accounting, and uneven two-rank CPU updates including continuation after a KL stop. The tests do not use GPU memory or modify production jobs.

## Before a GPU launch

1. Select the BC reference with the planned matched game panel. Do not choose it from training loss alone.
2. Copy `league.example.json` and pin any extra opponent files by SHA-256. The portable default uses 50% self-play, 40% frozen BC and 10% starter because it does not bundle an unverified public agent. To use the planned fourth opponent, allocate 15% to a measured-fast, trusted script and reduce frozen BC to 25%.
3. Supply a real milestone panel. `panel.example.json` is a format example and sanity panel, not evidence of competitive strength. Each external script or neural checkpoint needs its SHA. Reserve its seeds from collection.
4. Run the deferred one/two-GPU complete-game smoke and measure end-to-end throughput before a long campaign.

Prepared launch command, **not yet run**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
torchrun --standalone --nproc-per-node=2 -m ppo.train \
  --bc-reference /path/to/selected-bc.pt \
  --league /path/to/league.json --output /path/to/new-ppo-run \
  --games 64 --workers 16 --minibatch 128 --updates 1
```

Both ranks collect from their own CPU pool and perform their own batched GPU inference. Both then run a synchronous DDP update. The policy version stays fixed throughout collection. `--workers` is global. CPU actors use one numerical-library thread each. `--updates 0` continues until the run directory contains `STOP`; it finishes the current update before exiting. `--resume /path/to/update-N.pt` restores weights, optimizer, per-rank Torch RNG states, update number and matchup statistics, and verifies the saved contract. No automatic restart occurs after failure. A stale lock requires inspection before a manual restart.

Every completed update saves a numbered checkpoint and `latest.json`; the BC reference is never overwritten. `lifecycle.json`, `metrics.jsonl` and `failure.json` record progress and failure. Checkpoints preserve the BC model schema so its existing inference loader can read the unchanged model, with PPO metadata stored separately.

## Probability and learning contract

The PPO action is one complete player-turn request. Its log probability sums all worker commands (including PASS), applicable quantities, market requests, and selected END. No END is invented at the request cap. BC supervision weights never enter the PPO ratio.

The worker resolver preserves the whole-turn PLANT pre-scan and cancellation rule. Market requests are not treated as observed fills. The rollout records only causal observations and request prefixes. Each game seat has separate actor, critic and reference states.

Initial settings: one PPO pass, clip 0.1, gamma 1, GAE lambda 0.95, 16 burn-in plus 48 loss turns, actor LR 2e-5, critic LR 5e-5. The WDL head supplies P(win)-P(loss), with terminal WDL cross-entropy. Rewards are terminal +1/0/-1. Frozen-reference KL and entropy are means over visited factors; PPO uses the unweighted joint density. The trainer rejects BF16 pending collector/replay density parity; log probabilities and losses use FP32. All these settings need measurement.

Both current-policy self-play seats contribute training data. Frozen-opponent seats never do. Learner policy faults receive terminal losses; opponent faults cannot become free learner wins. Infrastructure errors abort collection rather than fabricate rewards. Normal games end at the official terminal. Synthetic short tests are never reported as games/sec benchmarks.

## League and evaluation

Historical opponents are frozen, hash-pinned checkpoints with the same representation contract. History's total allocation remains fixed while within-history sampling mixes uniform coverage with recent difficulty. Admission is explicit; saving a checkpoint alone does not promote it or establish a distinct strategy. The helper caps history at eight and cannot replace the permanent BC reference. `--history-mass` defaults to 0.25 across all snapshots, funded by self-play; adding more snapshots does not keep reducing self-play. Set the other allocations explicitly for the planned mature mix.

```bash
python -m ppo.league --league league.json --checkpoint candidate.pt \
  --id snapshot-001 --family animal-heavy
python -m ppo.evaluate --checkpoint candidate.pt --panel milestone.json --output eval/candidate
python -m ppo.evaluate --compare eval/candidate eval/incumbent
```

Do not run a different league under an existing resume contract. To change opponents, start a new output directory with `--bc-reference ORIGINAL_BC.pt --learner-init PREVIOUS_PPO.pt --league NEW_LEAGUE.json`. This keeps the original frozen BC reference and loads only learner weights, with a fresh optimizer. It records both checkpoint hashes. `--resume` instead restores the exact prior stage, including its optimizer; these options cannot be combined.

Use fixed matched panels for comparison and separate panels with fresh rotating seeds for promotion review. `--eval-panel PATH --eval-every 10` reserves the panel's seeds and writes an immutable `eval-request-N.json` after each milestone. It does not launch evaluation or make DDP wait. Run `python -m ppo.evaluate --request RUN/eval-request-N.json` separately when CPU resources are available. The request pins the checkpoint SHA, panel and output directory. The comparison tool reports a small-panel screening result and never submits to the arena or promotes automatically.

Compare outcomes by opponent and seat, with cash and margin as diagnostics. Distinct epochs are not automatically distinct strategies. Opponents absent from PPO may still occur in BC's all-replay dataset.

## Measurements and limits

Metrics include per-rank collection time, valid games, full seasons, learner turns, action factors, shared-memory bytes/wait time, actor feature/environment/script times, update time, KL, clipping, entropy, gradient norm and end-to-end valid games/sec. Actor times sum process work and may overlap; sampler time includes IPC. Do not add them as disjoint wall-clock phases.

The sampler preserves the official Python engine and exact action grammar. It sends compact quantity statistics and builds quantity features on the GPU only for selected commands that use a quantity. Actor requests share one barrier per action depth across policy groups, while each policy keeps its own weights and recurrent states. Worker prefixes update incrementally, with a canonical rebuild when a newly blocked crop cancels earlier PLANT requests. The post-worker farm summary uses the exact own-farm subset of the encoder.

Shared-memory views are temporary. Learner records own copies before another actor command can overwrite the arena. Pinned staging buffers wait for their previous DMA event before reuse. Frozen opponents do not retain training records. Recurrent anchors are saved at the starts needed by the configured burn-in and training windows.

Strict FP32 with TF32 disabled remains the tested collection/replay precision. BF16 did not pass the earlier pre-update density check; the trainer rejects `--bf16` before opening a run. Skipping unused quantity draws changes random-number consumption, so identical seeds need not produce identical trajectories to the old sampler. The action distributions must still agree when replaying fixed actions.

The matched 64-game, two-GPU test improved from 268.27 to 158.70 seconds end to end (0.239 to 0.403 valid full games/second), with one PPO pass in each run. All 64 games completed 719 turns without faults. This is a bounded test, not a production training run or proof of playing strength. CPU feature construction and synchronous action-depth barriers remain substantial; no simulator rewrite is included.

References checked: [PyTorch 2.11 DDP](https://docs.pytorch.org/docs/2.11/generated/torch.nn.parallel.DistributedDataParallel.html), [Python 3.14 shared memory](https://docs.python.org/3.14/library/multiprocessing.shared_memory.html).

## Production run (2026-09-20/21, `broad-ppo-20260920`)

Indefinite continuation from CP117 with lambda 1, LR 1e-5 actor / 5e-5 critic, 512 full 719-turn games per update on 2 GPUs.
Per update: 256 self-play games, 64 against retained older policies, 192 against the real public arena programs
(`arena-mix-v2`: arena families are sampled with weights that rank 0 recomputes after every update from an EMA of the learner cash margin per family, `arena_opponents.update_family_weights`)
(`arena_opponents.py`, `hybrid.py`; pool hash-pinned, sandboxed native code, no proxies).

Objective (`shaped-reward-v4`, `replay.py` / `rollout.py` / `gpu_replay.py` / `shaped_reward.py`): terminal return
`sign(margin) + clip(margin / 25000, -1, 1)`; baseline `V = utility(outcome logits).detach() + value_shaped(critic.detach())`
where `value_shaped` is a zero-initialised `Linear(256, 1)` attached at load time (`exact_model.py` stays byte-identical) and
trained with Huber loss in its own AdamW group at 100x the critic LR; the 3-class outcome head and its cross-entropy are unchanged.
Migrations freeze actor/trunk gradients for two warm-up updates. Exports (`league_runtime.snapshot`) strip the head so pinned evaluators keep the old contract.

Collection (`collection-overlap-v1`): 24 official actor processes per rank; the official sampler draws from a dedicated per-update
generator; captured graphs use `thread_local` error mode. `overlap_collection` stays 0 (a child-process overlap fails the full-prefix density guard).

Migration kinds accepted on resume (`train.py`): `shaped-reward-v1..v4`, `collection-overlap-v1`, `arena-pool-update-v1` (daily pool additions, `ops/pool_update.py`).
Package hash of this tree: `34c674a10c5c707b58b915515b321e58f9bd6cff8091952c7d7f957181e7e9f6`. Controller: `broad_controller.py` + `run_controller.py` (snapshots every 10 updates,
128-game dev panel on 8 real programs, nominations every 50 updates, paired 64-seed confirmations; promotions need CI > 0, no family drop > 20 pp).
