# Retain learned rare market patterns

The previous quantity-head experiment learned all 44 newly introduced training examples, but its exact market actions on 218 previously introduced rare training examples fell from 99 to 23. These are training-label diagnostics, not match replays or held-out performance.

This round keeps the same model architecture and optimizer settings. The fixed 5% rare-example allocation now includes every training command class introduced since the original 1,349-class vocabulary, including older expansions. Previously the allocation included only the current round's new classes. All accumulated training datasets remain available. Sixty additional teacher games add training-only templates automatically.

The model starts from the frozen 1,514-class quantity-head checkpoint. Only the market classifier, pattern embedding and market quantity MLP train; the shared encoder, unit controller and unit quantity MLP remain frozen. A sequential read warms the immutable base training arrays before random sampling, avoiding the observed cold-disk bottleneck without changing examples or their sampling order.

The existing pipeline waits for the preceding experiment to finish. It skips this round if that experiment achieves its final target, and starts only after a verified rejection. It reserves 60 training, 240 development and 960 final seeds, all distinct from historical registrations and from each other. Acceptance requires more than 75% pure wins and zero errors on the entire 15-agent pool: at least 181/240 development wins followed by at least 721/960 independent final wins. CPU limits remain unchanged.

`check_protocol.py` checks the sampling distinction with a concrete retained-class example, seed isolation, seat/opponent balance, dependency behavior, and strict win-rate boundaries. `train.py` additionally verifies real new-class gradients, initial output compatibility, data provenance and frozen weights. Runner SUMMARY.json remains authoritative; no postgame replay win-rate rescan is used.
