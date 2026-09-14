# Build and runtime receipt

Final native SHA256: `7d7fd9540328b4d87df7b9970c03d63837831713efd781b86359a33de76836f5`.
Actual compiler: `g++ (Debian 14.2.0-19) 14.2.0`. Python3.13.5. Linux x86-64, explicit
`-march=x86-64`, not host-native ISA. No pip install or network is needed for
production rebuild, root inference or the C++ unit suites.

Actual final command (original working directory paths are preserved):

```sh
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2 /mnt/data/TRI_A06_r12_r3_work/candidate/policy/bridge.cpp /mnt/data/TRI_A06_r12_r3_work/candidate/policy/executor/vendor/simulator.cpp -o /mnt/data/TRI_A06_r12_r3_work/candidate/policy/a06.so
```

Equivalent offline commands from the extracted delivery root:

```sh
timeout 150s python -B build.py
# Optional new tests exist and may now be requested:
timeout 180s python -B tests/run_units.py
python -B tests/verify_entry.py --out /tmp/a06_r3_entry.json
```

All flags: `-std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2`.
`policy/a06.BUILD.json` is the actual final receipt, including50 input hashes,
compiler, command, native hash and measured build duration. Root `BUILD.json`
references independent build validation rather than guessing source/native
compatibility. The first frozen-parent probe compiled in30.76s, peak607840KiB,
and matched the parent's binary. Final production compile took27.77s wall,
peak609280KiB; independent directory compile27.51s, peak609224KiB, native bytes
identical. All commands were serial with explicit timeouts. Compile stdout and
stderr, including empty successful stderr, are retained as actual logs.

The distributed binary is dynamically linked. Observed required symbols include
GLIBC2.32 and GLIBCXX3.4.31; see exact ELF/version/ldd records in `logs/native_*`.
An older Linux runtime may require the supplied offline rebuild. GCC13.3 rebuilt
the parent centrally; this round used the actual available GCC14.2.0, not a claimed
central-toolchain build. Cross-compiler byte identity is not promised.

For diagnostics only, the supplied Python official engine uses its vendored
source; optional action-schema auditing uses already-installed `jsonschema`
(Draft7). Production has no `jsonschema` dependency. Re-run research in a copy to
avoid overwriting frozen raw logs. Example final bounded batch (four trajectories,
not full matches): `python -B research/run_final_batch.py 2 v3`. The historical
subversions and manual research fork are explicitly not production binaries.
