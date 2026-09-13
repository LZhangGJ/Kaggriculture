# Actual build receipt — TRI_A06_r12_r1

This receipt is for the complete production library delivered at `policy/a06.so`.
`BUILD.json` and `policy/a06.BUILD.json` are byte-identical machine-readable receipts
from the clean delivery-directory compilation; historical parent receipts are
under `evidence/parent_metadata/` and are not current build evidence.

## Actual invocation

Working directory: `/mnt/data/TRI_A06_r12_r1`.

```bash
/usr/bin/time -v -o evidence/logs/delivery_clean_build.time \
  timeout 120s python -B build.py --cxx /usr/bin/g++ \
  >evidence/logs/delivery_clean_build.stdout \
  2>evidence/logs/delivery_clean_build.stderr
cp policy/a06.BUILD.json BUILD.json
```

The build script executed exactly:

```bash
/usr/bin/g++ -std=c++20 -O3 -DNDEBUG -march=x86-64 -ffp-contract=off -DR2_LOCAL_SALE_TIMING=1 -DR2_FINITE_FERTILIZER=1 -DR2_CROP_CLOCK_MODE=1 -DR2_OBSERVE_PUBLIC_TRADES=1 -DR2_SALE_CLOCK_MODE=0 -fPIC -shared -Wl,-Bsymbolic -DR2_MARKET_INTEGRAL=0 -DR2_STARTUP_SUPPLY_MODE=2 -DA06_EXEC_MODE=2 /mnt/data/TRI_A06_r12_r1/policy/bridge.cpp /mnt/data/TRI_A06_r12_r1/policy/executor/vendor/simulator.cpp -o /mnt/data/TRI_A06_r12_r1/policy/a06.so
```

Compiler: `g++ (Debian 14.2.0-19) 14.2.0`. Build receipt timestamp:
`2026-09-13T02:54:12.925215+00:00`. Exit status 0. Builder elapsed
26.206604 seconds; command wall time 26.85 seconds; maximum RSS
608,024 KiB. Raw compiler stdout/stderr and `/usr/bin/time -v` output are retained.
The library is an x86-64 ELF shared object, 1,250,728 bytes,
SHA256 `d642c5650c916365136bfc5ce5c47f82240c27dc34d9baf691a580376b56ec0b`. It is byte-identical to the final native used in the eight
candidate closed-loop games and the seven final common-prefix probes.

All compiler flags are recorded verbatim in `COMPILER_FLAGS.json`. They include
baseline x86-64 (not `-march=native`), C++20, O3, DNDEBUG and disabled FMA contraction.
Unit tests are built separately with assertions enabled and O2; their exact
commands, source hashes and outputs are in `tests/UNIT_RESULTS.json`.

## Offline rebuilding

Use `python -B build.py --cxx g++` from a working copy. The script compiles the two
translation units plus their local headers/includes; no download, installation,
network access, generated placeholder or foreign strategy is needed. The Python
entry and codec use the standard library. `--out /path/to/a06.so` is supported.
`--unit` additionally invokes the seven unit suites. The new output receives an
adjacent `.BUILD.json` receipt. Preserve the original frozen `BUILD.json` for
comparison; hashes produced by a different compiler need not be byte-identical.

The 47-item source/config/build-input hash mapping was checked against
`SOURCE_FREEZE.json` and the live files. Complete payload hashes, including tests,
reference fixtures and raw logs, are separately in `MANIFEST.json`. A rebuild or
unit rerun intentionally changes generated logs and may invalidate that original
full-payload manifest; verify a fresh extraction before modifying it.

Dependencies are real dynamically linked Linux libc/libstdc++/libgcc/libm, not
bundled runtime shims. The exact `file`, `ldd` and `readelf --version-info` output
is in `evidence/native_linux_dependencies.txt`. No portability to an older runtime
than the symbol requirements is claimed; rebuild in that target environment.
