# Actual offline build

Compiler: g++ (Debian 14.2.0-19) 14.2.0

Production command (original absolute working paths):

```sh
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2 /mnt/data/r3_fix1_work/candidate/policy/bridge.cpp /mnt/data/r3_fix1_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/r3_fix1_work/candidate/policy/a06.so
```

Rebuild from this archive root:

```sh
python3 build.py
python3 build.py --unit
python3 tests/verify_entry.py --root . --evidence feedback --out entry_check.json
```

No network, pip, downloaded code, or opponent source is required. A C++20 compiler and Python standard library are required. The unchanged flags use portable Linux x86-64, not -march=native. The author compiler is GCC14.2.0; GCC13.3 testing reported by central applied to parent r3, not to this fix. See BUILD.json, policy/a06.BUILD.json, logs/fix1_build.*, logs/rebuild.*, and logs/independent_rebuild_receipt.json for actual receipts. Unit-test assertions are enabled (-O1, no -DNDEBUG).
