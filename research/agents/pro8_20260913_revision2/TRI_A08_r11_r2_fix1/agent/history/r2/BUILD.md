# Actual build and source identity

This is an actual compiled artifact, not a proposed command. Default candidate
compile: `logs/r2_build.stdout`, `.stderr`, `.time`, `.exit`; exact source/native
receipt: `policy/tri_a08_r11_r2.BUILD.json`.

Compiler: `g++ (Debian 14.2.0-19) 14.2.0`.

Actual candidate command:
```sh
g++ -DA08_ANIMAL_COLLECTION_LABOR=1 -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared -Wl,-Bsymbolic /mnt/data/a08_r2_work/candidate/policy/bridge.cpp /mnt/data/a08_r2_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/a08_r2_work/candidate/policy/tri_a08_r11_r2.so
```

Flags (unchanged parent features, plus one collection-work switch):
```text
-DA08_ANIMAL_COLLECTION_LABOR=1 -DA08_SALE_FLOOR_DP=1 -DA08_SALE_SCHEDULE_DP=1 -DA08_PAID_CONTINUATION=1 -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_STARTUP_SUPPLY_MODE=2 -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_MARKET_INTEGRAL=1 -DR2_SALE_CLOCK_MODE=2 -DA08_COMPLETE_CROP_STOPS=1 -DA08_LAND_DP_MODE=2 -DA08_REALIZATION_MODE=0 -DA08_CROP_PORTFOLIO_MODE=1 -DA08_CASH_LEDGER=0 -DA08_PREFIX_DAYS=5 -fPIC -shared
```

The command used full mounted paths during this run. For the same portable
relative build from any extracted location, run `python build.py`; no paths to
old round tests are required. No `-march=native`, OpenMP or worker pool was used.

Native SHA256: `afca1d69fde38a0474546b70ecee38911477548dfde9aadc685c58a2a0b88a2f`.
44-source manifest SHA256: `9046ac7badd335511633d0770906cb5982da636a2cfa82844f13fe0f64cd6708`.
Config SHA256: `65efd0f5822d40a86440f1837640a54ee1ea60b8f8b61e3be99da2d97591f72c`.
Each source/configuration hash is in `SOURCE_SHA256SUMS.txt` and the JSON receipt.

The independent clean tree default build is preserved in
`probes/clean_rebuild/`. Its receipt has the actual separate command/path.
`probes/r2_clean_build_comparison.json` records byte identity. The first clean
batch had an incomplete outer-wrapper time receipt; the successful isolated
repeat is `logs/r2_clean_rebuild_repeat.*`. It took 28.78 seconds wall and
594,712KiB maximum RSS. Do not infer successful timing from the incomplete file.

Parent rebuild: `probes/parent_rebuild_repeat.so` and its BUILD receipt,
`logs/parent_build_repeat.*`. The parent artifact bytes are also retained in
`reference/parent/policy/tri_a08_r11_r1.so`.

Ablation native and receipt: `probes/ablation.so`, `probes/ablation.BUILD.json`.
That switch-off binary is not the submission. Focused instrument/test compiler
commands are in `logs/*.command.json` and timestamped `rerun/` directories.

`tests/run_checks.py --sanitize` rebuilds C++ focused tests using the production
feature flags and optionally AddressSanitizer/UndefinedBehaviorSanitizer. The
production native itself is the optimized release binary, not a sanitizer build.

## Dynamic runtime dependency check

`logs/elf_dependencies.txt` records the actual ELF header, loader dependencies
and imported symbol versions. The release is ELF64 x86-64 and requires system
symbols including `GLIBC_2.32` and `GLIBCXX_3.4.31`. Runtime loading was tested
here; arbitrary older Linux images are not implicitly certified. Rebuild
offline on the target host using the included sources when its loader differs.
