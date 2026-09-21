# Operations scripts for the broad-PPO continuation

- `ppo_export.py` — builds `ppo.json` for the arena dashboard's "PPO run" tab every 5 minutes (cron on WRX90).
- `pool_update.py` — daily (04:30 UTC) boundary migration that adds new public roster agents to the training pool (`arena-pool-update-v1`).
- `prune_vast_incoming.py` — hourly retention of drained arena transport bundles on the Vast evaluation host.

All three are read-only over training except `pool_update.py`, which stops the trainer at an update boundary, validates one update on the new pool, imports it once and resumes.
