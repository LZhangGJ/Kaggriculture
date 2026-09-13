# TRI_A08_r11_r2_fix1

**Status: source/build repair candidate, not a centrally accepted agent.**
The exact parent is TRI_A08_r11_r2 (ZIP `2f1557875d550000ae492c3814c84c78d5402e99b3a754308dc3907aa32b2178`). The complete
production parent, including its tested native, is retained in `reference/r2/`;
the earlier r1 ancestor is unchanged in `reference/parent/`.

## What is and is not established

Central Ubuntu 24.04 / GCC 13.3 reported an optimized r2 test with
`feed=0` but `value=65`, although the unique optimal action is `feed=1`.
This record is preserved verbatim in substance in `fix1/central_failure.json`.
GCC 13 is not installed in this runtime, and attempts to retrieve it failed.
**Neither the original GCC 13 failure nor its disappearance after this change
has been reproduced locally. Undefined behavior versus a compiler defect remains
unresolved.** The repair isolates code generation, makes the chosen action/value
structurally inseparable at selection, and adds a fail-closed test of the actual
linked native. It is not a claim that an identified GCC bug has been fixed.

Actual local compilers are GCC 14.2.0 and Clang 17.0.0. Both original and fixed
standalone reductions pass on these compilers; the unmodified original full
unit is retained as the central reproducer, not replaced with a passing reduction.
`fix1/repro/matrix_gcc13_unavailable/RESULT.json` records the missing compiler.

## Focused source repair

`policy/animal_service_dp.hpp` now chooses one immutable candidate index from the
three legal service actions, preserving the original enumeration order and
`1e-9` tie rule, and commits both `Choice` and `value` from that same winner.
The service reward formula and r2 animal-collection-labor improvement are unchanged.
GCC's `noipa` attribute places `solve` behind a genuine optimized call boundary;
Clang uses `noinline`. The body is still compiled at **`-O3`**. There is no global
`-O0`, no `optimize("O0")`, no removed assertion, and no disabled production feature.
This is a targeted code-generation workaround pending the GCC 13 recheck.

`policy/bridge.cpp` adds `td_service_dp_snapshot`, a read-only synthetic-input
validation seam calling that exact linked solver. It owns a fresh DP object and
cannot inspect or mutate a running agent. It is never called by strategy decisions.

`build.py` compiles to a unique staging file, checks the resulting native's action,
value, independent Bellman optimum, tie choice and full-policy realization, then
promotes it only after success. A failing check preserves the prior native and
receipt and retains the rejected staging file and all logs. `main.py` loads only
`policy/tri_a08_r11_r2_fix1.so`; there is no silent fallback to an ancestor.

Only four of 44 production source/configuration files differ from r2: the service
DP, bridge, entrypoint and build script. The other 40 are byte-identical, including
`triad.hpp`, fixed configuration, positive-return investment, rolling scheduling,
low-level execution and all existing sales-DP boundaries.
The exact patch is `fix1/PRODUCTION_DIFF.patch`.

## Offline build and checks

Linux x86-64, a C++20 compiler, Python standard library, GNU `time` and `timeout`
are required. There is no package installation or network dependency.

```sh
python tests/verify_package.py  # First verify the delivered inventory.
python build.py --cxx g++
python tests/run_checks.py --sanitize
```

A rebuild deliberately changes its receipt and appends logs, so run inventory
verification before rebuilding, or on a fresh unpack. A different compiler is
not expected to emit the same native bytes. It must pass the exact-native gate
and behavioral checks. To test the central compiler explicitly:

```sh
python build.py --cxx g++-13
python tests/run_checks.py --cxx g++-13 --sanitize
python tests/repro_matrix.py --cxx g++-13 --out /tmp/a08_fix1_gcc13_new_directory
```

The matrix includes original standalone, boundary-only, argmax-only, combined
repair and the original full unit, each with `-O3`, `-O0` and `-O1` ASan/UBSan.
Original failures are diagnostic; fixed-case failures return a failure status.
No pretense is made that a locally passing reduction reproduces a foreign compiler.

```sh
python tests/test_build_refusal.py --out /tmp/a08_fix1_expected_rejection_new_directory
python tests/compare_saved_entries.py \
  --left "$PWD/reference/r2/main.py" --right "$PWD/main.py" \
  --traces "$PWD/evidence/own_traces" \
  --out /tmp/a08_fix1_paired_new_directory --require-exact
```

The refusal test deliberately mutates a Python checker copy only, outside the
production tree. It is an expected-failure safety test, not a GCC bug reproduction.
All test output directory arguments must be fresh directories.

## Actual validation

Both GCC 14.2 and Clang 17 optimized full builds passed the expanded C++ unit:
148,821 assertions, 5,120 brute-force cases, 4,608 path cases and 20,496 action-state
checks. ASan/UBSan versions passed as separate evidence, not substitutes for `-O3`.
The exact-native gate made 530 DP constructions and validated 29,456 states with
171,628 successful checks before the deliberate mutation. The rich control is
`value=Choice.value=65, feed=1, care=0`; the poor control retires the animal.
The injected `feed=0, value=65` mismatch is rejected.

Official local execution checks passed 4,608 single-farm lifecycles, 16,128 day
states and 90,914 checks. These do not prove global scheduler feasibility.
The default entry/receipt/config/ABI/isolation checks passed 183 assertions.

Actual paired calls on all seven saved own-observation traces: **5,033/5,033 actions
match original r2**; **5,033/5,033 match between GCC 14.2 and Clang 17 fix1 builds**.
The two narrow-win saved traces each match r2 for 719/719 calls. These are fixed
observation probes, not new matches, not counterfactual futures and not evidence
that either narrow win is retained in closed loop.

A clean-directory GCC 14.2 rebuild produced identical native bytes. A separate
expected-failure build test preserved the previous native and its receipt.
See `VALIDATION_REPORT.md` and raw results under `fix1/` and `rerun/`.

**New closed-loop games: 0. New official panel: 0. No new win rate is claimed.**
The centrally supplied r1 1,054/1,536 result remains historical. The acceptance
threshold remains strictly greater than 85% (at least 1,306/1,536 wins), to be
measured by the central fresh-seed panel after identity freezing.

## Identity and preserved evidence

- Production native SHA256: `91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed`
- Source manifest SHA256: `d8fa06d155ec38c75358c220dcdbb3d05da29ad2694414ddf67fd0f119e77f36`
- Configuration SHA256: `65efd0f5822d40a86440f1837640a54ee1ea60b8f8b61e3be99da2d97591f72c`
- Exact compiler commands, flags, native gate and all source hashes:
  `policy/tri_a08_r11_r2_fix1.BUILD.json` and `BUILD.md`.

`history/r2/` contains the original r2 top-level claims and manifests unchanged;
they are historical, not fix1 results. Existing `analysis/`, `logs/`, `probes/`,
`development/` and `checkpoints/` originated in the r2 ZIP and are not relabelled as
new work. New repair records are in `fix1/`, new time-stamped tests in `rerun/`,
and transactional build logs in `build_runs/`. Earlier failed/partial tool
wrappers, network retrieval failures, intermediate build receipts, and the central
failure are retained. No full match panel or new development game seeds were run.

The original round start/deadline remain 2026-09-13 15:22:07.194 / 17:22:07.194 UTC.
Resource/timing records distinguish these from the repair's 16:26:13 UTC start.
