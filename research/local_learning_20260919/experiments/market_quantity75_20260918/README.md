# Independent market quantity branch

This is a training experiment for the active >75% pure-win objective. The running 1,470-template candidate remains unchanged in `F:/Kaggriculture/experiments/learned_expand75_20260918`.

The prototype copies the existing quantity MLP (128 → 256 → 1002) into a separate market quantity MLP. Unit quantities keep the original network. The copied weights preserve initial behavior; the market branch can then be trained independently. This adds 290,538 parameters, for 2,704,095 total parameters. Command templates and the quantity vocabulary are unchanged by this prototype.

`model_joint_numeric.load_checkpoint` loads legacy checkpoints by copying the shared quantity weights into the new branch. Partially missing new-head weights are rejected. Existing checkpoints containing all new-head weights load strictly.

`check_quantity_branch.py` uses the frozen 6,000-step candidate with SHA-256 `2a31a9aa7aca82f22363f89c955c9a524875adfde5258ce2195efd4ac0823cb2`. It compares all forward outputs against the original forward implementation, checks initial actions, trains only the new branch on 16 existing training examples, and checks all frozen tensors and unit outputs again. It does not inspect evaluation replays, consume any match seed, save the fitted probe weights, or change the active evaluator.

The check passed: initial forward difference 0, eight matching initial actions, and quantity loss 0.976711 → 0.000774 after 30 probe updates. Unit actions on identical probe inputs, market types, and all frozen tensors stayed unchanged. This establishes initialization compatibility and small-sample learnability only; it is not evidence of held-out improvement or a >75% win rate.

The full training pipeline reuses the existing collector, codec, evaluator and CPU limiter. It retains the original training data and all three supplemental datasets, collects 60 new balanced teacher games, and adds any newly observed training-only command templates. It trains the market pattern head, pattern embedding and new quantity head for up to 6,000 steps, with validation every 500 steps and four-check early stopping. The shared encoder, unit controller and original unit quantity head remain frozen. All probe updates are discarded before real training.

Every new match has its own seed: 60 training, 240 development and 960 final games, disjoint from every registered historical seed and from each other. All 15 opponents and both seats are balanced. Promotion requires more than 75% pure wins with zero errors on the complete development panel, then at least 721/960 pure wins with zero errors on the untouched final panel. Draws are not wins. Use runner SUMMARY.json; do not rescan evaluation replays for win-rate verification.

The pipeline waits on the previous pipeline's exact Windows process creation identity, including any final evaluation it starts. It trains only after DEVELOPMENT_BELOW_TARGET, FINAL_BELOW_TARGET or VALIDATION_REGRESSION. It skips this round if the previous experiment reports TARGET_MET, and rejects running, failed or unknown prior states. Thus it never overlaps the previous candidate's evaluation or consumes new matches after an accepted result.

Training, collection and evaluation use the existing native Windows job CPU cap of at most 70%, plus whole-machine load feedback. The prototype check proves initialization compatibility and small-sample learnability only; this architecture still needs actual training and fresh-seed acceptance before promotion.
