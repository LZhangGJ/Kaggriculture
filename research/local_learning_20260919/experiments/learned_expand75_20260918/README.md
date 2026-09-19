# Learned market expansion, round 1

The objective is a pure win rate strictly greater than 75% against the unchanged 15-agent pool, with improvements learned through training. Draws count in the denominator and are not wins. No evaluation replay is used to create rules, labels, or a vocabulary.

The starting checkpoint is the first joint run's selected step 500. Its completed independent result is 358/480 wins (74.5833%). The subsequent low-rate continuation stopped at 750 steps without improving its validation selection metric; it is not used as initialization.

This round reserves 60 distinct training seeds, 240 different development seeds, and 960 different final seeds. Every match, opponent and seat has its own seed. All are excluded from registered historical training, validation, development and final seeds. The final checkpoint is selected before the final panel starts; no substitute checkpoint can reuse that panel.

The frozen teacher plays actual games against all 15 opponents, two matches per opponent and seat. Only training observations and canonical teacher labels are stored. Each action is checked through the existing codec and the historical action input uses the same canonical representation. All matches are retained regardless of outcome. No opponent identifier or seed is a policy input.

The vocabulary is automatically the union of the initial checkpoint's templates and templates in the original, supplemental and fresh training data. There is no hard-coded target vocabulary size and no inclusion of validation-only templates. Old classifier rows and template embeddings are mapped by the template contents. This gives a repeatable expansion mechanism when new training data is added; the policy remains a template classifier, not an autoregressive command generator.

Training changes only the market pattern classifier and template embedding. The shared encoder, unit controller and quantity network are frozen. Template embeddings still influence market quantity predictions through the existing frozen quantity network. Eight-action initialization parity, label-independent inference, actual gradient flow, a 30-update rare-template learning check, and frozen-weight checks run before real training. That learning check is discarded and real training restarts from the expanded initialization.

Sampling weights are 80% original, 5% public supplemental, 5% local supplemental and 10% fresh data; 5% of batch slots are replaced by uniformly sampled new-template training examples. Twenty percent of eligible inputs use the model's prediction on the preceding teacher frame as previous-action history. This is history denoising, not on-policy DAgger or PPO.

At most 6,000 batches of 64 are trained, validating every 500 steps with patience 4. The trained checkpoint with lowest validation loss is selected; the initialization is a reference, not an eligible trained candidate. A drop of more than 0.5 percentage points in validation full-action accuracy prevents game evaluation. Development requires at least 181/240 wins and zero errors. Final requires at least 721/960 wins and zero errors.

The existing Windows Job CPU controller applies a 70% maximum hard cap to owned work and reduces it when whole-machine CPU increases; its target is 55% with a 60% pause threshold. Unrelated programs remain independent. Evaluation uses CUDA FP32 serial model calls with two CPU game workers and does not claim official competition timeout acceptance.

`pipeline.py` runs collection, training, development, and (only after development passes) the reserved final panel. `check_protocol.py` checks seed separation and acceptance boundaries. `run/CHECKS.json` records the real learning preflight. Runner `SUMMARY.json` files are authoritative; no separate replay win-rate recomputation is performed.
