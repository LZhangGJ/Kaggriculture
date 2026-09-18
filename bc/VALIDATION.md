# Validation — September 18, 2026

The package was copied into a separate checkout. The production trainer and jobs were not changed.

## Checked

- Python 3.14.4 clean virtual environment installed Torch 2.11.0+cpu, the minimal requirements, and kaggle-environments 1.32.7 with `--no-deps`.
- `check_environment.py` loaded epoch 6, validated all checkpoint source/engine hashes, and built the model without the production source directory on PYTHONPATH.
- A real organizer replay passed through `prepare_replays.py`, `audit_exact.py`, and `cache_exact.py`: two seats, 1,438 player-turns. The resulting raw input objects matched the existing production canonical inputs for every turn.
- `check_cache.py` ran a 16-turn recurrent chunk through the model and backpropagation on CPU, with finite loss and gradients. It also passed in the clean environment.
- The portable `evaluate.py` ran epoch 6 against the built-in starter for seed 9182201 in both seats. Both games reached 720 frames / DONE. Cash was 124,910 versus 3,667 and 123,699 versus 3,667, exactly matching the earlier production-code evaluation. The clean environment reproduced both results.
- The organizer index downloaded successfully with Kaggle CLI 2.2.2.
- The unchanged production trainer has already completed seven two-GPU epochs. This packaging task did not launch another GPU training run.

Receipts are in `metadata/`. These checks do not claim a fresh full-corpus rebuild, bit-identical training on another GPU, or a working PPO trainer. Cross-machine GPU kernels can change numerical results. The full Kaggle dependency bundle was tried and failed at pygame metadata generation on Python 3.14; the documented headless install avoids that unrelated dependency.

## Artifact identity

`metadata/checkpoints.json` pins the release assets. Epoch 6 and epoch 7 are completed training checkpoints including optimizer state. The encoder asset is the historical initialization used by production. `metadata/epoch6-cache-identity.json` contains the original feature/label/source contract. Rebuilt canonical/cache manifests have new local path provenance; do not erase that distinction to force exact BC resume.
