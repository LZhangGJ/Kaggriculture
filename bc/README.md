# Kaggriculture behavior cloning

Train or run the **ExactWorkerMarketPolicyV1** policy used in our September 18 BC campaign. This directory includes replay downloading, lossless action preprocessing, a configuration/action audit, numeric caching, recurrent training, inference, evaluation, and completed checkpoints.

## Start from a checkpoint

Use Linux or WSL2 and **Python 3.14**. The cache uses Python 3.14's `compression.zstd`. Install GitHub CLI (`gh`) and authenticate with access to this repository. Do not copy anyone else's credentials.

```bash
git clone --branch feature/reproducible-exact-bc https://github.com/LZhangGJ/Kaggriculture.git
cd Kaggriculture/bc
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Our training build; choose a compatible official PyTorch wheel for your GPU.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip install --no-deps kaggle-environments==1.32.7
python fetch_checkpoint.py --name epoch6
python check_environment.py --checkpoint checkpoints/epoch6.pt
python evaluate.py --checkpoint checkpoints/epoch6.pt --opponent starter --output runs/starter
```

For CPU inference/preprocessing, replace `cu130` with `cpu`. GPU training requires CUDA and uses BF16 by default; `--fp32` is available. Consult the [official PyTorch installer](https://pytorch.org/get-started/locally/) for driver compatibility, retaining Torch 2.11.0. Do not upgrade the game engine: **kaggle-environments 1.32.7** and its engine/spec hashes are part of the checkpoint contract. Install that wheel with `--no-deps`: its full dependency list includes unrelated games, including pygame, whose Python 3.14 install can fail. `requirements.txt` supplies the dependencies used by this headless Kaggriculture workflow. Warnings about other unavailable games during import do not indicate a Kaggriculture failure.

`fetch_checkpoint.py --name epoch7` downloads the later completed checkpoint. `--name encoder` downloads the earlier encoder initialization used by this campaign. Downloads verify the SHA-256 values in `metadata/checkpoints.json`. Large weights live in the repository release, not Git history. Epoch 6 is the default because it has arena and weaker-opponent calibration results; epoch 7 is newer, not established as stronger.

## Rebuild the replay dataset and train

All commands below run from `bc/`, with the environment active. Use your own Kaggle credentials and accept the competition/dataset terms as needed. The [official organizer index](https://www.kaggle.com/datasets/kaggle/kaggriculture-episodes-index) contains selected high-rated episodes, not every competition game.

```bash
python -m pip install kaggle==2.2.2
python download_replays.py --output data/downloads
python prepare_replays.py --downloads data/downloads --output data/canonical --workers 32
python audit_exact.py --root data/canonical --output data/audit --workers 32
python cache_exact.py --root data/canonical --audit data/audit --output data/cache --workers 32
python check_cache.py --cache data/cache
python fetch_checkpoint.py --name encoder
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 train_exact.py \
  --cache data/cache --output runs/epoch-0001 --batch-games 64 --workers 4 \
  --epochs 1 --initialize-encoder checkpoints/encoder.pt
```

Use `--nproc-per-node=1` for one GPU. `--batch-games` is the global trajectory batch, divided across ranks; it must divide evenly. `--workers` on the trainer is per GPU, unlike preprocessing. Defaults match the production run: 16-turn truncated-backpropagation chunks, recurrent state across the full 719-turn season, AdamW at 1e-4, gradient clipping at 1, both seats, losses included, and no dev/test holdouts. Omit `--initialize-encoder` to start completely fresh; that differs from this campaign.

### Replay scope and storage

The default frozen organizer index covers **August 15–September 16, 2026**, the dates containing this campaign's 1.32.7 games. The production corpus contains **22,185 games / 44,370 seat trajectories / 31,902,030 turns**. Other engine versions and incomplete games are excluded and counted. No eligible seat is excluded based on win/loss. No sample is copied to inflate coverage.

To add future dates, use `--refresh-index --through-date YYYY-MM-DD`; changed source data needs a new cache. To try a smaller window, set `--from-date` and `--through-date`. That is a smaller experiment, not a reproduction of the full campaign. Default preprocessing includes every eligible game downloaded. It records exact raw action slots; cache construction regenerates the reviewed worker/market labels and quantity choices, rather than using old macro-label masks.

Raw ZIPs remain compressed. Do not plan storage from their compressed size alone: canonical JSON and numeric caches are additional copies, and production canonical files may reference reused trajectories outside their top-level folder. The full production numeric cache alone is about **128 GiB**. Use an SSD with ample free space and start with a one-day window if capacity is uncertain. Download receipts record archive hashes and count gaps; missing organizer episodes cannot be reconstructed.

Preprocessing and cache jobs resume completed matching receipts. Keep data in a stable directory because manifests contain absolute paths. Do not move a cache and then assume `--resume` will work. Caches use Python pickle internally; load only caches you built or trust.

### Continue training and evaluate

```bash
python evaluate.py --checkpoint runs/epoch-0001/latest.pt --opponent starter --output runs/eval-0001
OMP_NUM_THREADS=1 torchrun --standalone --nproc-per-node=2 train_exact.py \
  --cache data/cache --output runs/epoch-0002 --batch-games 64 --workers 4 \
  --epochs 1 --resume runs/epoch-0001/latest.pt
```

`--resume` preserves weights and optimizer, advances the epoch shuffle, and requires a **completed epoch using that exact cache digest**. A locally rebuilt cache has different path-dependent provenance, so the downloadable production checkpoints are for inference/PPO initialization; do not force them through BC `--resume` by deleting checks. Resume your own checkpoints on your own cache. The supplied encoder checkpoint lets you reproduce the production initialization on a rebuilt corpus.

The trainer writes `metrics.jsonl`, `run-config.json`, and atomic `latest.pt` checkpoints. The stored epoch index is zero-based. Mid-epoch snapshots can be evaluated but cannot resume this trainer. Use a fresh output directory for each epoch. This package does not install timers or start any background jobs automatically.

`evaluate.py` runs the official 720-frame game, both seats per seed, and reports wins/losses and final cash. Failed/unfinished games never count as wins. `--opponent path/to/agent.py` supports a trusted Python module exporting `agent(obs, config)`; it executes in your process, so use the arena sandbox for untrusted submissions. Built-in `starter` is a weak single-tile carrot loop. It is only a functionality baseline.

## Model and PPO starting point

See [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md) for the detailed feature representation and tensor dimensions, [the diagram](model-architecture.svg), and [ARCHITECTURE.md](ARCHITECTURE.md) for the workflow. The model has roughly 5.31M parameters: a shared spatial/entity encoder, actor/critic recurrent states, separate autoregressive worker and market decoders, and legal quantity choices. It receives causal observations and public history. The worker phase updates an exact own-action ledger before market decoding; it does not observe the opponent's pending action or pretend to know market fills.

Despite the earlier “macro BC” directory name, this version learns **actual worker and market commands each turn**. It is not a separate high-level planner issuing goals to a separately trained micro policy.

```python
from exact_decoder import ExactAgent

policy = ExactAgent.from_checkpoint('checkpoints/epoch6.pt', device='cpu', greedy=True)
policy.reset()  # reset between games; one instance/state per concurrently running game
action, info = policy.act(observation)
# action is the official farmer/hands/market dictionary.
model = policy.model  # ExactWorkerMarketPolicyV1 with the pretrained weights
```

For exploratory stochastic rollouts use `greedy=False`; `act()` samples the joint gate/command density and conditional quantity probabilities. `act()` uses `no_grad` and returns detached diagnostics; **it is not a differentiable PPO update API**. PPO collection must retain observations, recurrent states, ordered actions, quantity choices and old log probabilities. Recompute the same autoregressive density with gradients, using the same masks, worker projection, market order, and vocabularies. `ExactChunk` shows differentiable teacher-forced scoring; its BC-weighted loss is not the PPO policy ratio. The outcome head is a three-class win/draw/loss classifier, not a trained scalar-return PPO critic. Adapt/train a value head appropriate to your return target.

Keep this checkpoint as a frozen BC reference/opponent. Create your own optimizer for PPO. The BC optimizer is included in the completed checkpoint for provenance, but blindly resuming it is not a PPO algorithm. This branch provides the BC foundation, not a tested end-to-end PPO trainer. No cached private opponent information or future target is an input feature.

## Checkpoint compatibility

`exact_identity.py` pins the relevant model, decoder, feature, mechanics, and engine hashes. `.gitattributes` preserves their original line endings. Do not auto-format those files before loading the checkpoint. `check_environment.py --checkpoint ...` detects drift. For architecture/feature experiments, deliberately create a new compatibility contract and verify the weight mapping rather than turning off validation globally.

## What has been measured

Epochs 1–7 completed training and the fixed 32-game panel. Every checkpoint beat PASS 8/8 and lost 0/8 versus each of AFS R2, Day 9, and historical public farming v4. Epoch 6 also scored 16/16 versus the official starter and the old land allocator, 1/16 versus archived Kaggriculture 2026 v1, and 0/16 versus EcoBot v7 and v23. These small calibration panels establish some useful play, not competitive strength or a competition percentile. Training loss is not a win-rate estimate.

The fixed-panel receipts are under `metadata/`. See `VALIDATION.md` for the portability checks made before publishing. Changes here do not alter the running WRX90 training job.
