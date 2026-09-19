# Local learned-agent progress — 2026-09-19

This is an auditable source and metadata snapshot of the local training program. It is separate from the reference `bc/` implementation. **The current >75% pure-win objective has not been achieved.**

## Is training effective?

The models demonstrably learn the supervised tasks. We have not established improved playing strength from the latest changes. Lower training/validation loss is not a substitute for fresh-seed games.

| Candidate | Evaluation | Wins / games | Pure win rate | Draws / errors |
|---|---|---:|---:|---:|
| `joint_win65_20260918`, selected step 500 | Completed fresh final | 358 / 480 | 74.58% | 0 / 0 |
| `learned_expand75_20260918` | Completed development | 177 / 240 | 73.75% | 0 / 0 |
| `market_quantity75_20260918` | Completed development | 169 / 240 | 70.42% | 0 / 0 |
| `retained_market75_20260919` | Development started at snapshot time | Pending | Pending | Pending |

These panels use different seeds and have different sizes; this table is not a paired causal estimate. The older experiment's own metadata used a 65% target. Its 74.58% final result does **not** satisfy the current >75% target. Failed development candidates did not consume their reserved final panels.

Evidence of learning and its limits:

- The market combination vocabulary expanded from 1,349 to 1,470 to 1,514 classes, derived only from training action labels. The in-progress retention round has 1,554 classes.
- The `market_quantity75_20260918` round learned all 44 new training examples' full market actions. However, on 218 previously expanded training examples, exact type sequences fell from 209 to 44; full market actions fell from 99 to 23. This is evidence of catastrophic forgetting on training data, not an evaluation-replay diagnosis. See [training-only comparison](audits/market_quantity/LEARNED_MARKET_COMPARISON.json).
- The retention round samples all historical rare combinations as well as new ones. Its fresh collection completed 60 distinct-seed games / 43,140 frames. Training stopped after 3,000 updates and selected step 1,000. Validation loss fell from 0.24609 to 0.24207, while full-action exact accuracy fell slightly from 94.72% to 94.62%. It retains the independent market quantity head and freezes the observation encoder and unit controller. Development games have started; its win-rate effect remains unknown.
- A subsequent audit of the selected retention checkpoint found recovery on its 286 historical rare examples: exact type sequences improved from 92 to 244, and full market actions from 68 to 154. However, exact types and full actions on this round's 44 newly introduced examples both remained 0/44. Vocabulary growth therefore does not prove the selected model learned the new classes. This cohort includes additional fresh training examples and is not the same 218-example cohort above. See [retention comparison](audits/retained_market/LEARNED_MARKET_COMPARISON.json). The sampling/selection tradeoff remains unresolved.
- A separate autoregressive market prototype learned all 27 selected training examples after 50 updates, including every request length from 0 to 10. It passed causal-prefix, EOS masking, codec, and frozen-unit checks. It does not use a whole-combination lookup table. This is a small-sample implementation check, **not** held-out generalization or a successful match candidate. Probe weights were discarded. See [prototype receipt](autoregressive_prototype/PROTOTYPE_CHECK.json).

The current experiments use teacher behavior cloning. They do not directly optimize match rewards with PPO. The latest changes train market heads, not the entire agent. The next architecture experiment is to train the autoregressive market decoder on the full accumulated data; post-worker resource state and cross-turn memory should be tested separately.

## Fixed acceptance contract

- Unchanged pool of all 15 opponents, equal opponent/seat counts.
- A different environment seed for every game, including both seats, each candidate, training, development and final acceptance. Registration excludes all discovered prior experiment seed metadata. `PLAN.json` and `SEED_AUDIT.json` are retained as receipts.
- Strict pure wins: wins / all scheduled games. Draws never count as wins. Zero runtime errors.
- Development: at least 181 wins in 240 games. Final: at least 721 wins in an independent 960-game panel.
- Learned changes only: no rules tailored to evaluation replays, no teacher at inference, no evaluation trajectories in training.
- Use the match runner's `SUMMARY.json` for results; do not perform a separate replay win-rate rescan.
- Windows native job CPU cap at 70%, plus whole-machine headroom feedback. Unrelated processes are outside the job's control.

The local runner uses official engine transitions but does not enforce the competition sandbox or its time limits. Prototype warm single-thread CPU inference medians were 17.37 ms for the existing template policy and 24.06 ms for the autoregressive policy. These omit feature construction, codec and cold start; they are not an official timeout guarantee.

## Contents and reproduction limits

- `experiments/`: source bytes, frozen model definitions, protocols, small dataset receipts, completed result summaries, and an explicitly time-stamped training snapshot.
- `audits/`: candidate provenance and training-only rare-pattern checks.
- `autoregressive_prototype/`: runnable local learnability check and its result.
- `source_receipts/`: dataset and unchanged opponent-pool identities.
- [Exact BC investigation](EXACT_BC_REVIEW_ZH.md): source-based borrowing recommendations and limitations of the reference branch's evaluation.
- `SNAPSHOT_MANIFEST.json`: hashes and original paths for copied files; `PACKAGE_CHECK.json`: publication checks.

This is **not** a self-contained training distribution. Weights (`.pt`), raw replays, numeric arrays, Python environments, teacher implementation and opponent executables are not included. The scripts preserve the original `F:/Kaggriculture/experiments/...` paths and source bytes so recorded hashes remain meaningful. On another machine, restore the named inputs from the receipts and adapt paths in a separately registered experiment. Do not bypass hash validation, reuse a started seed, or rerun a historical `prepare.py` against its existing plan.

Completed candidate checkpoint identities:

| Candidate | SHA-256 |
|---|---|
| 74.58% baseline | `cff21b96c0c41b4dadedd23cafbba91d9b5780d4280d644dee3b107f24d57914` |
| Expanded vocabulary | `2a31a9aa7aca82f22363f89c955c9a524875adfde5258ce2195efd4ac0823cb2` |
| Independent market quantity | `a642906d6b8737869efc55dbb52512c633285ef8a2138cafadc248c865bc7120` |

Snapshot time: see `SNAPSHOT_MANIFEST.json` (UTC). Live training can advance after this snapshot; an archived `STATUS.json` is not a live monitor.
