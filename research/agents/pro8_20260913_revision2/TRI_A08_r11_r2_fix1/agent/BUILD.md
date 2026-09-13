# BUILD — TRI_A08_r11_r2_fix1

## Exact deployed build

Compiler: `g++ (Debian 14.2.0-19) 14.2.0`. Python: 3.13.5. Linux x86-64.
Feature flags are unchanged from the r2 default build, including
`A08_ANIMAL_COLLECTION_LABOR=1`; optimization is `-O3`, with `-ffp-contract=off`.
No native-architecture-specific `-march=native` and no global deoptimization.

Actual command (absolute paths and staging filename preserved):

```sh
g++ -DA08_ANIMAL_COLLECTION_LABOR=1 -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared -Wl,-Bsymbolic /mnt/data/fix1_work/candidate/policy/bridge.cpp /mnt/data/fix1_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/fix1_work/candidate/policy/tri_a08_r11_r2_fix1.building-20260913T164128033231Z.so
```

The staging library was checked by this actual command before atomic promotion:

```sh
/opt/pyvenv/bin/python /mnt/data/fix1_work/candidate/tests/native_choice_gate.py --library /mnt/data/fix1_work/candidate/policy/tri_a08_r11_r2_fix1.building-20260913T164128033231Z.so --enabled 1 --out /mnt/data/fix1_work/candidate/build_runs/20260913T164128033231Z/native_choice_gate.json
```

The stage was renamed to `policy/tri_a08_r11_r2_fix1.so` only after the gate passed.
Source and native byte identities are recorded in `IDENTITY.json`,
`SOURCE_SHA256SUMS.txt` and the adjacent `.BUILD.json` receipt.

Production native SHA256: `91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed`.
Source manifest SHA256: `d8fa06d155ec38c75358c220dcdbb3d05da29ad2694414ddf67fd0f119e77f36`.

## Independently rebuilt artifact

`fix1/clean_rebuild/` was populated with the 44 production source/configuration
files and the standalone native checker, without a native library. Running
`python clean_rebuild/build.py` compiled and gated an identical GCC 14.2 binary.
See `fix1/logs/clean_offline_rebuild.*`, `fix1/logs/clean_rebuild_hashes.txt`,
`fix1/clean_rebuild/policy/tri_a08_r11_r2_fix1.BUILD.json`.

Clang 17 full optimized native and its receipt are retained as
`fix1/repro/fix1_clang17.so` and `.BUILD.json`. It is not the default production
library. Its table digest and all 5,033 tested observation actions match GCC 14.2;
its binary hash intentionally differs. These compilers use the installed Linux
headers/standard library. This is not a claim of testing on Ubuntu 24.04 itself.
`fix1/logs/native_platform.txt` records ELF, dynamic dependencies and symbol versions.

## Exact-native fail-closed gate

The gate is `tests/native_choice_gate.py` (SHA256
`1c4866667e35d9fd9d167b001fd83a89a5d10a8f391fcd2cdf15a5d83ae1c51d`). It checks both stored score fields,
independent optimal action/tie order, one-step Bellman realization and full policy
rollout. Build failure, mismatch or gate failure prevents promotion and preserves
existing production bytes. Full logs and the staging artifact are retained.
It is mandatory; there is no skip flag.

The self-contained new build-refusal test confirms rejection with the exact
positive-return assertion, while preserving BOTH the original native and receipt.
The fault injection changes only a disposable Python checker copy. It must not be
mistaken for a GCC 13 reproduction or a failed production candidate.

## Compiler-specific boundary and attribution limit

The only new optimization boundary is `AnimalServiceDP::solve`: GCC `noipa`,
Clang `noinline`. GCC documents `noipa` as preventing interprocedural optimization
between a function and its callers; it is not an instruction to compile the body
at `-O0`:
https://gcc.gnu.org/onlinedocs/gcc-13.3.0/gcc/Common-Function-Attributes.html#index-noipa-function-attribute

Original GCC 14 unit symbols contain a `solve [clone .constprop.0]`; the final
GCC 14 symbols retain the un-cloned solver boundary. See the two symbol logs in
`fix1/repro/`. That demonstrates the intended boundary in the tested compiler;
it does NOT establish that GCC 13 constant propagation caused the central failure.
No GCC bug number, diagnosed UB, or GCC 13 pass is asserted.
