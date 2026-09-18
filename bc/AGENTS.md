# Working on the BC pipeline

Read README.md first. Work from this directory so the flat Python modules and bundled kgrl support resolve locally. Use Python 3.14 and the pinned engine; avoid unrelated environment dependencies as described in the guide.

The files listed in exact_identity.SOURCES, cache_identity.py, kgrl/contracts.py, and kgrl/commitments/mechanics.py are checkpoint-hashed. Preserve their bytes and line endings when loading the released weights. For a deliberate model change, define and test a new compatibility/migration contract rather than bypassing validation.

Use all eligible current-engine games and both seats. Raw observation frame t supervises the recorded action at frame t+1. Future fields and the other player's private state must not enter features. Preserve raw action requests; exact labeling occurs in cache_exact.py.

Keep data, caches, local environments, and weights out of Git. Small metadata receipts are allowed. Never add credentials. Do not connect to or change WRX90 jobs just because historical metadata mentions them.

Useful checks: check_environment.py for engine and checkpoint compatibility; check_cache.py for a real cached recurrent forward/backward pass; evaluate.py for full-game behavior. Run checks relevant to the change. Evaluation must report failed games separately, and training loss must not be described as playing strength.

This is a BC foundation for PPO experiments. The current value head predicts W/D/L; act() is no-grad inference. Read the PPO section before implementing updates.
