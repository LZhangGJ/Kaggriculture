# Actual selected build

Compiler: `g++ (Debian 14.2.0-19) 14.2.0`.

Production command (actual staging output, promoted only after both linked gates passed):

```sh
g++ -DA08_COMPLETE_ROLLING_ROUTES=1 -DA08_DELIVERY_HANDOFF=1 -DA08_ANIMAL_COLLECTION_LABOR=1 -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared -Wl,-Bsymbolic /mnt/data/TRI_A08_r11_r4_release/policy/bridge.cpp /mnt/data/TRI_A08_r11_r4_release/policy/executor/vendor/simulator.cpp -o /mnt/data/TRI_A08_r11_r4_release/policy/tri_a08_r11_r4.building-20260913T222421576853Z.so
```

Flags:

```text
-DA08_COMPLETE_ROLLING_ROUTES=1
-DA08_DELIVERY_HANDOFF=1
-DA08_ANIMAL_COLLECTION_LABOR=1
-DA08_SALE_FLOOR_DP=1
-DA08_SALE_SCHEDULE_DP=1
-DA08_PAID_CONTINUATION=1
-std=c++20
-O3
-DNDEBUG
-march=x86-64
-ffp-contract=off
-DR2_STARTUP_SUPPLY_MODE=2
-DR2_LOCAL_SALE_TIMING=1
-DR2_FINITE_FERTILIZER=1
-DR2_CROP_CLOCK_MODE=1
-DR2_OBSERVE_PUBLIC_TRADES=1
-DR2_MARKET_INTEGRAL=1
-DR2_SALE_CLOCK_MODE=2
-DA08_COMPLETE_CROP_STOPS=1
-DA08_LAND_DP_MODE=2
-DA08_REALIZATION_MODE=0
-DA08_CROP_PORTFOLIO_MODE=1
-DA08_CASH_LEDGER=0
-DA08_PREFIX_DAYS=5
-fPIC
-shared
-Wl,-Bsymbolic
```

All 50 production/build inputs including both gate scripts are recorded in `SOURCE_SHA256SUMS.txt` and `BUILD.json`. Actual staging commands/exit codes/stdout/stderr are retained in `build_runs/`. Independent clean-source rebuild: `development/final_clean_rebuild/`; the emitted native matches selected bytes. Clang 17 optimized output is separately identified and tested; GCC 13.3 is not installed locally.

Selected native SHA256: `acf904cb79039a80612d69437f26b358cec471e74400f1ac1275369466e9c27f`.
