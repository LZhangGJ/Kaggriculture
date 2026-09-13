# Actual build and resource receipt

Production compiler: `g++ (Debian 14.2.0-19) 14.2.0`. Python measured at startup:3.13.5.
Target:Linux x86-64 ELF shared library; ISA flag:`-march=x86-64`.

## Actual successful production command

```sh
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2 /mnt/data/a06_r12_r2_work/candidate/policy/bridge.cpp /mnt/data/a06_r12_r2_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/a06_r12_r2_work/candidate/policy/a06.so
```

Actual flags are also frozen in `COMPILER_FLAGS.json`; no `-march=native` or
downloaded build dependency is used.

Primary wall time:29.417s; maximum RSS:607836KiB; exit0.
The timed initial parent rebuild was29.16s/RSS608084KiB and reproduced the
parent native exactly.

## Actual independent rebuild command

```sh
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2 /mnt/data/a06_r12_r2_work/independent_build/policy/bridge.cpp /mnt/data/a06_r12_r2_work/independent_build/policy/executor/vendor/simulator.cpp -o /mnt/data/a06_r12_r2_work/independent_build/policy/a06.so
```

Wall time:28.388s; maximum RSS:607968KiB; exit0.
This directory contained only production source/config/build inputs, not prior
evidence or opponents. The rebuilt native is byte-identical and its root
runtime separately reproduced336 early actions.

Native SHA256:`638626de9398f0aed45e98394742ee22df875dd805f2647dee6d5666eb1c7755`

All48 build/runtime input hashes and both actual
compiler receipts are recorded in `BUILD.json`; canonical source-input hash
is in `SOURCE_FREEZE.json`. `NATIVE_ABI.txt` under evidence records actual ELF
format, dependency inspection and required symbol versions.

## Quota and budget

Initial affinity0-4 exposed5 CPUs; cgroup CPU quota400000/100000 capped the
entitlement at4 CPUs. Memory entitlement4294967296 bytes. Heavy builds/tests
were serial and reserved30% memory. Final cgroup memory.peak was recorded in
`evidence/RESOURCE_FINAL.json`; there were no OOM events. Host memory figures
were logged as context only.

The complete staged six-stage offline suite exited0 in
55.698s. Raw command/exit/time/RSS/stdout/stderr
records are retained. Neither that suite nor the historical probes ran a new
complete game.
