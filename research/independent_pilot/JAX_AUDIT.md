# Team JAX simulator audit

The JAX simulator is available on `agent/p16-local-review-transfer-20260910` at commit `75ae03d3d517cf3da35c42914a78a064838334de`. The merged evaluation branch and current `main` contain an older PyTorch/Triton engine; that is a different implementation.

The relevant newer JAX source is [nt/latest_20260823_fc24b/workspace/gpu_sim](https://github.com/LZhangGJ/Kaggriculture/tree/75ae03d3d517cf3da35c42914a78a064838334de/nt/latest_20260823_fc24b/workspace/gpu_sim). Its README names reference version 1.32.7. The earlier top-level `nt/gpu_sim` names 1.32.6; do not substitute that version without checking it against the pinned reference.

The newer README reports 1.80 million simulator transitions/second at batch 4,096 on an RTX 3090. It also lists an earlier actor-plus-PPO measurement of 57,291 transitions/second at batch 1,024. These are team records, not measurements repeated on this host. Simulator-only throughput omits inference, learning and data movement. The newer source snapshot contains 18 files and does not include the complete receipt/test/table bundle advertised by its README.

The implementation is a useful candidate for larger training runs:

- It has a fixed-shape state, ten simulator market slots, ordered inventory handling and diagnostics for hand caps, market-loop caps and price-table limits. Its observation encoder selects the viewing player's private state.
- Its event-table design indexes shop draws by the number of weed draws actually consumed. That preserves the action-dependent random stream instead of treating same-seed shops as a fixed sequence.
- The shipped policy has two market slots and eight quantity choices: 0, 1, 2, 3, 4, 8, 16 and 32. Its feed-forward action heads therefore differ from this experiment's recurrent, sequential resource checks and full integer quantity support.

Before using it for E2, pin the 1.32.7 source and tables, regenerate events for fresh training seeds, port the ordered decoder and actor boundary, then run the same focused and full-game differential tests. Require zero bound/lookup diagnostics. Measure compilation, complete rollout-plus-update throughput and peak memory on the assigned GPU. The current CPU pilot makes no claim about JAX performance or PPO's eventual GPU learning curve.

No competitor policy or replay labels are needed to reuse this simulator. Its game engine can support independently developed policies.
