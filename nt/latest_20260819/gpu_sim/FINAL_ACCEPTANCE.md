# Kaggriculture JAX GPU simulator final acceptance

Audit result: **PASS**

## Goal gates

- **PASS — official_reference_immutable_and_hashed**: Frozen and installed official source/config/framework hashes match. Evidence: `receipts/reference_verification.json`
- **PASS — wsl2_jax_rtx3090_runtime**: JAX 0.11.0 executes compiled work on NVIDIA GeForce RTX 3090, 24576, 610.47. Evidence: `receipts/jax_runtime.json`
- **PASS — exact_python_price_and_rng_tables**: Python round market LUT and conditional MT19937 event tables derive from frozen source. Evidence: `receipts/static_tables_v2.json`
- **PASS — canonical_full_rule_parity**: Pass, starter, and v16 full seasons compare every recorded state. Evidence: `receipts/reference_traces.json and receipts/test_suite.json`
- **PASS — 100_unseen_seeds_720_frames_zero_error**: Seeds 10000..10099, all farm/private/market/town/status/reward fields exact. Evidence: `receipts/heldout_reference_traces.json and receipts/heldout_parity.json`
- **PASS — random_and_invalid_action_differential**: 16 state-aware randomized seasons include deliberately invalid actions. Evidence: `receipts/random_differential_reference.json and receipts/random_differential_parity.json`
- **PASS — static_hand_bound_with_instrumentation**: MAX_HANDS=32, observed=14, cap counter tested. Evidence: `receipts/hand_cap_analysis.json`
- **PASS — jit_vmap_scan_batch_independence**: Pure step, synchronized fast batch, independent rollouts, and lax.scan tests pass. Evidence: `receipts/test_suite.json`
- **PASS — gpu_arena_and_ppo**: Heterogeneous architectures, self-play collection, GAE and clipped PPO update stay on GPU. Evidence: `receipts/benchmark_policy_and_ppo.json and receipts/test_suite.json`
- **PASS — simulator_50k_minimum_300k_target**: All batches exceed 50k/s; batches 1024 and 4096 exceed 300k/s; compilation excluded. Evidence: `receipts/benchmark_simulator_only_full_season.json`
- **PASS — reproducible_environment_commands_and_docs**: Pinned WSL environment, verification, parity, benchmark and training commands documented. Evidence: `requirements-wsl.lock.txt and README.md`

## Headline measurements

- Held-out official frames compared: 72,000
- Random/invalid differential frames compared: 11,520
- Simulator-only batch 256: 200,271 env transitions/s
- Simulator-only batch 1024: 744,525 env transitions/s
- Simulator-only batch 4096: 1,803,833 env transitions/s
- Policy+sim batch 256: 24,615 env transitions/s
- Policy+sim batch 1024: 90,684 env transitions/s
- Policy+sim batch 4096: 198,471 env transitions/s
- Full PPO batch 256: 18,321 env transitions/s
- Full PPO batch 1024: 58,200 env transitions/s

All throughput numbers exclude compilation and synchronize the final device result.

## Declared runtime boundary

The exact MT19937 event bank contains seeds `0..127` for training/development
and `10000..10127` for held-out verification. Seeds outside this bank are
rejected and must be added by regenerating the frozen event table; they are
never silently approximated.
