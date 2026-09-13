# Independent economic search and PPO pilot

This experiment compares independently generated economic plans with a small recurrent PPO policy. Both can buy crops, animals and land, hire up to 16 hands, assign work and sell products. The champion remains a frozen opponent. No competitor actions or model weights provide training labels.

The pilot uses the official 1.32.7 reference from the merged evaluation bundle. It adds a separate runner because these agents have observation-based Python entry points rather than the existing six candidates' native ABI. It does not change the published evaluation bundle or consume its holdout outcomes.

Completed pilot: [result and next experiment](verification/REPORT.md), [full evidence](verification/README.md). Neither family qualified to advance.

## Scope

- E1 represents a season as five six-day stages of production, labor, land, cash-reserve and sale targets. A new feedback executor turns those targets into legal commands on each observation. Beam search constructs stages; large-neighborhood search changes allocations and uses CP-SAT for minimum-change allocation repair. The solver does not solve a full worker schedule. LNS alternates uniform and UCB operator selection; this pilot does not isolate the effect of that controller.
- E2 uses a 147,458-parameter GRU policy, complete 719-turn sequences and a conditional command decoder. The policy starts from random weights. It supports all 24 official commands plus the internal market STOP choice, up to ten market slots, and all integer quantities within its resource bounds. Binomial quantity sampling uses double precision for stable log-probability replay.
- Workers update an exact local ledger in order. Market purchases use a conservative cash bound because simultaneous rival trades can affect prices. Sales contribute only the guaranteed price floor to further same-turn spending. These masks can exclude affordable actions; they do not expose hidden inventory.
- Training uses our three independently written crop/livestock controls. PPO also uses current self-play in half its games. The final pilot uses four frozen benchmark opponents and both seats. This limited pool is not a leaderboard estimate.

## Run

Use Linux with Python 3.12, NumPy 2.4.4, PyTorch 2.11.0 CPU and OR-Tools 9.15.6755. The existing opponent binaries require a recent libstdc++; Ubuntu 24.04 works. The older default Ubuntu 22.04 runtime on this host does not load those binaries.

From the repository root:

```sh
python -m research.independent_pilot.build
python -m unittest discover -s research/independent_pilot/tests -v
python -m research.independent_pilot.preflight --out artifacts/independent-pilot/preflight-ubuntu24.json
python -m research.independent_pilot.learning_check --out artifacts/independent-pilot/learning-check-ubuntu24.json
# Run the external-opponent smoke described below before freezing.
python -m research.independent_pilot.protocol --out artifacts/independent-pilot/run-v1/inputs --base-commit VERIFIED_40_CHARACTER_COMMIT --reserved /path/to/current/reserved/holdout.json --seconds 180
python -m research.independent_pilot.run_trials --root artifacts/independent-pilot/run-v1
python -m research.independent_pilot.evaluate --root artifacts/independent-pilot/run-v1 --phase selection --workers 2
python -m research.independent_pilot.evaluate --root artifacts/independent-pilot/run-v1 --phase pilot_test --workers 2
python -m research.independent_pilot.analyze --root artifacts/independent-pilot/run-v1
```

The external-opponent smoke calls `evaluate.worker` for the `control` candidate against `native:r1`, `native:r2`, `extra:old-champion` and `extra:hysteresis`, with a replay path for each game. Save the four passing rows to `artifacts/independent-pilot/eval-smoke/results.json`. The freeze also reads the two check receipts at the paths above. These verification games remain outside all score estimates.

The reserved-manifest input serves only a membership exclusion check. Its values and outcomes are not printed or used as features. A formal final comparison still needs a fresh custodian release and current campaign audit. The historical exclusions here are a snapshot, not a complete audit of every running campaign.

The freeze pins source, native binaries, the opponent runtime, public evaluation seeds, historical exclusions, pilot seeds and analysis rules. Trials validate those files before and after running. Each trial gets 180 process CPU seconds and finishes its last complete rollout/update. Two configurations and three seeds give each family an 18 CPU-minute training/search budget. The timed CPU loops exclude model and optimizer initialization. The pipeline records elapsed time including startup; it does not meter initialization CPU separately. Correctness checks and evaluation have separate receipts. A failed trial stops the runner; there is no automatic retry.

Configuration selection averages the three training seeds on eight selection worlds. One configuration per family then runs on eight fresh pilot worlds, alongside untrained policies and the independent rule control. Each test world retains four opponents and both seats as one resampling cluster. All games and failures remain visible. The benchmark contract keeps strict win rate primary and match score secondary.

In each evaluation phase, the first world checks every observation against the official interpreter and records complete replays for every candidate/opponent/seat. The other worlds use the verified native engine. This is a local execution check, not Kaggle runtime certification. No command submits an agent or promotes a champion.

## Provenance

The starting native engine and direct decoder came from the user's earlier full-action PPO work: `full-actions-20260907/native.cpp`, SHA256 `945b60c6c2460b32ebf1920d80ee6ac2595cf13d5c75812187b646d23bb6f68f`. This copy adds an observation-only actor, conservative market bounds and independent plan execution. The new GRU/ordered PPO implementation avoids importing that campaign's legacy script, teacher and checkpoint machinery. Existing campaigns remain untouched.

The CP-SAT path receives proposals that already passed deterministic feasibility repair, so this pilot cannot show a benefit from the solver. A later ablation must compare unrepaired proposals with and without solver repair.

See [JAX_AUDIT.md](JAX_AUDIT.md) for the team simulator and its remaining integration work. See the frozen run's `PROTOCOL.json`, journals, replay files and `RESULTS.json` for the experiment evidence.
