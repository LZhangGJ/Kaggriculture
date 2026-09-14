# TRI_A08_r11_r4 — complete lightweight production package

This is a packaging-only redelivery of the frozen r4 strategy, not an economic fix and not a new competition result. All 50 production/build inputs match the original source manifest; the native SHA256 is `acf904cb79039a80612d69437f26b358cec471e74400f1ac1275369466e9c27f`.

The original full evidence ZIP is preserved unchanged: 44,495,151 bytes, SHA256 `5b6d34a83d2309ea707cc1252a4f3dd3986412590f69009b99454de1a18bda2b`. Historical evidence is deliberately not nested here.

## Offline build and checks
Requires Linux x86-64, Python 3, and a C++20 compiler (GCC or Clang). No network, pip packages or files outside this directory are needed for these commands:

```sh
python build.py
python tests/run_checks.py
# Optional additional sanitizer run (does not replace the optimized tests):
python tests/run_checks.py --sanitize
sha256sum -c SOURCE_SHA256SUMS.txt
```

Use `main.agent(observation, configuration)` or `main.create_agent()`; the root entry loads the fixed config and included native. The default build runs both exact-native correctness gates before replacing the native. All r4 rolling-route, task-handoff, animal labor, floor-sale DP and investment capabilities remain enabled. Default compiler flags, source hashes and gate results are in `policy/tri_a08_r11_r4.BUILD.json`. `BUILD.json` retains the original frozen build receipt.

## Recovery verification
A clean input-only directory was actually rebuilt using GCC 14.2 and the default optimized flags. Native bytes match the original. The supplied default `tests/run_checks.py` passed. Raw resource, build and test logs are retained without compiled test executables. GCC 13.3 was not available locally. A first external tool timeout interrupted its timing wrapper; it was not counted as a completed measured build. The fully timed retry is the reported build.

`DEVELOPMENT_SEEDS.json` records development fixtures and original conditional scenario seeds; referenced full historical traces reside in the original evidence ZIP, not this lightweight package. The default correctness tests use the included small fixtures and frozen official rules only.

No new full competitive games were run for redelivery. The user's running r3 panel is not r4's score. Download-client success cannot be verified from this environment.
