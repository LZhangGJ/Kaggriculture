# Broad PPO (minimal)

Code for the broad-PPO Kaggriculture agent, and nothing else:

- `ppo/`: the PPO trainer (`ppo/train.py`) plus the collection, reward, league and arena-opponent code it imports.
- Top-level `*.py`, `kgrl/`: the policy model and feature code the trainer and agent import.
- `gpu_sim/`: the JAX GPU simulator the games run in, plus its static lookup table.
- `ext_agent.py`: the inference entry point used by our Kaggle and arena submissions.

It does not include checkpoints, opponent bots, the promotion controller or ops scripts.

## Train (current production recipe)

This is 8 GPUs, 4096 games per update, 3576 of them against arena bots, with PFSP family weighting.

```bash
export PYTHONPATH=$PWD PPO_JAX_SIM_SRC=$PWD/gpu_sim/src PPO_JAX_ONE_DEVICE=1 PPO_ARENA_SANDBOX=chroot \
  PPO_COMPILE_MARKET=1 PPO_COMPILE_WORKERS=1 PPO_DONATE_SIM_STATE=1 PPO_EXT_LR_SCALE=10 PPO_HEAD_LR_SCALE=10 \
  PPO_FAST_MARKET_HEAD=1 PPO_FAST_WORKER_HEAD=1 PPO_FINE_WORKER_BUCKETS=1 PPO_ROLE_WORKER_BUCKETS=1 \
  PPO_WORKER_BUCKETS=1 PPO_FUSED_ADAM=1 PPO_SEQUENCE_GRU=1 PPO_SPLIT_BURN_ENCODER=1 PPO_SYNC_CHILD=1 \
  PPO_PACK_STATS=1 PPO_CHILD_ALLOC_CONF=expandable_segments:False \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=1
torchrun --standalone --nproc_per_node=8 -m ppo.train --collector gpu --games 4096 --workers 32 --minibatch 1024 \
  --updates 0 --seed 290000000 --bc-reference <BC_CHECKPOINT> --league <LEAGUE_JSON> --live-league <MANIFEST_JSON> \
  --output <OUTPUT_DIR> --resume <CHECKPOINT> --arena-pool <ARENA_POOL_JSON> \
  --lr 1e-5 --critic-lr 5e-5 --gae-lambda 0.99 --ppo-passes 2 --drift-kl 0.02 --market-entropy-coef 0.015 \
  --family-weighting pfsp --family-focus '{}' --arena-games 3576 --arena-in-gpu 1 --arena-bot-workers 48 \
  --official-workers 48 --async-collect 1 --async-official 1 --overlap-collection 0 --arena-parity-games 0 \
  --reward-shape linear --reward-dense 1 --reward-beta 1.0 --reward-sigma 40000 \
  --reward-cash-weight 0.25 --reward-cash-center 112000 \
  --anchor 0.02 --anchor-decay 0.99 --anchor-min 0.01 --anchor-start-update 1006 --distill-coef 0 \
  --start-state-bank <START_STATE_BANK> --start-state-frac 0.25 --start-state-days 7,24 \
  --value-trunk-grad 1 --value-loss mse --policy-warmup-until 1259
```

The `<...>` values are local files:

- `<BC_CHECKPOINT>`: the behaviour-cloning checkpoint the run started from.
- `<LEAGUE_JSON>` and `<MANIFEST_JSON>`: the league files.
- `<OUTPUT_DIR>`: where checkpoints are written.
- `<CHECKPOINT>`: the checkpoint to resume from.
- `<ARENA_POOL_JSON>`: the hash-pinned list of arena opponent bots.
- `<START_STATE_BANK>`: the start-state bank.

The arena opponents themselves are not in this branch.

Requirements are `requirements.txt` plus torch 2.11 (CUDA) and jax (the simulator runs on the GPU).

## Inference

```python
from ext_agent import load_agent
agent = load_agent('policy.pt', device='cpu', greedy=True)
action, _ = agent.act(observation)
```
