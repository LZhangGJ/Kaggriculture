# Build receipt and offline reproduction

The root `main.py:agent` loads `policy/a06.so` explicitly and reads the fixed
`policy/config.json`. All 40 production source/configuration files are present.
`build.py` and the compiler flags file are byte-identical to the A06_r6 parent.
No network, package download, training checkpoint, or external repository is
needed to compile this agent.

Run in a **disposable copy** because builds/tests write logs and receipts:

```sh
python3 -B build.py --cxx g++ --out /tmp/TRI_A06_r6_r1_rebuilt.so
python3 -B tests/run_units.py --cxx g++
python3 -B validation/run_floor_oracle.py
python3 -B validation/test_runtime_privacy.py
python3 -B validation/compare_prefixes.py
python3 -B validation/audit_new_games.py
```

`BUILD.json` contains the actual compiler command, every flag, the compiler
version, hashes of every compiled source and configuration, runtime hashes,
and measured wall/RSS receipts. `policy/a06.BUILD.json` is the untouched actual
builder receipt for the delivered native, including the absolute source paths
used during this turn. These paths are evidence, not installation requirements.

The tested compiler is GCC 14.2.0, using C++20, `-O3 -DNDEBUG -march=x86-64`,
`-ffp-contract=off`, `-fPIC -shared -Wl,-Bsymbolic`, and the exact feature defines
in `COMPILER_FLAGS.json`. Importantly `R2_MARKET_INTEGRAL=0` stays unchanged.
The artifact is Linux x86-64, dynamically linked. Recorded parent and candidate
symbol-version requirements match; see `validation/logs/native_platform.txt`.
The environment must provide compatible glibc/libstdc++ (including the recorded
GLIBC_2.32 / GLIBCXX_3.4.31 requirements). This is not a certification of the
Kaggle sandbox or its CPU/time/memory limits. Other compiler versions are not
promised to reproduce identical binary bytes.

The original work scripts and actual logs are under `validation/`. Portable
validation scripts at that level adjust only the flattened candidate-root path;
verbatim scripts used in the measured run are under
`validation/original_work_scripts/`. Their immutable original dispatch monitor
stops at this round's deadline; it is not required for a later offline rebuild.
Run the direct Python commands above in a disposable copy instead.

`validation/run_closed_loop.py` reproduces the small **parent-only** head-to-head
panel using its already-fixed seeds and matching native hashes, and overwrites
those copied logs. It does not contain the central 1536-game benchmark or the
original AFS R2/public opponent executable pool. Do not count replays or repeated
runs of the same panel as additional independent test games.
