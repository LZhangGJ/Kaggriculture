# Actual build record — TRI_A08_r11_r3

Production compiler: `g++ (Debian 14.2.0-19) 14.2.0`.
Production native SHA256: `49c98f2c1930cf741bdd9e2eb7da9af398117bb64f903f9375c7da9350e8fa64`.

## Actual command (original measured workspace)

```sh
g++ -DA08_DELIVERY_HANDOFF=1 -DA08_ANIMAL_COLLECTION_LABOR=1 -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared -Wl,-Bsymbolic /mnt/data/r3_work/candidate/policy/bridge.cpp /mnt/data/r3_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/r3_work/candidate/policy/tri_a08_r11_r3.building-20260913T193514593996Z.so
```

The command writes a timestamped staging library. `build.py` then calls the
unchanged exact-native animal action/value gate before atomic promotion to
`policy/tri_a08_r11_r3.so`. The actual command, exit codes, stdout, stderr,
gate result and source hashes are in `BUILD.json` and `build_runs/`.
`SOURCE_SHA256SUMS.txt` additionally records the mandatory gate dependency.

## Offline reproduction

```sh
python3 -B build.py --cxx g++
python3 -B tests/run_checks.py --sanitize
```

Python's standard library and a Linux x86-64 C++20 compiler suffice; no network,
Kaggle installation, third-party Python package or absent old test directory is
needed. GCC 14.2 and Clang 17 were actually tested at -O3. GCC 13.3 is not
installed here and has NOT been newly tested. Rebuild under the central
compiler and run the same gate; different compiler binary hashes need not match.
`-march=x86-64` is used, not host-specific `-march=native`.

A clean, independently copied source directory reproduced the production
library byte for byte with GCC 14.2. Its full sources, command and native are in
`development/clean_rebuild/`; its measured results are in
`validation/raw/clean_rebuild_result.json` and `clean_build_precheck.*`.
An intentionally rejecting gate was also built in a separate copy: both the
existing native and receipt remained byte-identical, and the rejected staging
library was retained. See `development/atomic_gate_negative/` and
`validation/raw/atomic_gate_negative_result.json`. This is an expected failure
injection, not a failing production gate.

For an ablation, never overwrite the default production output:

```sh
python3 -B build.py --delivery-handoff 0 --out /tmp/a08_r3_ablation.so
python3 -B build.py --cxx clang++ --out /tmp/a08_r3_clang.so
```

These switches existed only for controlled development. Default configuration
remains fixed. The shipped alternative native files in
`development/final_variants/` are NOT loaded by the root entry.
