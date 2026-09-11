# Tools, simulators and the exact opponent pool

Updated: 2026-09-11. This file documents the tools used for the [two agent approaches](docs/agent_approaches/README.md). It separates runnable artifact checks, the frozen local match toolchain and the offline route-training pipeline.

## 1. Opponents included in this branch

The [evaluation/opponents/](evaluation/opponents/) directory contains the **15 original live opponents** used in the latest 3,150-game P16 comparison. These are complete runtime directories, including their Python files, action data, configuration, native libraries and the license/notice files supplied with the original packages. No opponent was rebuilt or edited for this handoff.

[evaluation/POOL.json](evaluation/POOL.json) records every entry point, file size and SHA-256, with the original acquisition location and panel membership. Its hashes are checked against the pre-existing [formal evaluation pool receipt](evidence/r2p16_route_repair_20260911/POOL.json).

| Opponent ID | Runnable entry | Where it was used |
|---|---|---|
| `soil_v219g` | [main.py](evaluation/opponents/soil_v219g/main.py) | Historical panel |
| `moon_v215` | [main.py](evaluation/opponents/moon_v215/main.py) | Historical panel |
| `flexon_v5` | [main.py](evaluation/opponents/flexon_v5/main.py) | Historical panel |
| `market_smart_v8` | [main.py](evaluation/opponents/market_smart_v8/main.py) | Historical panel |
| `nagatakengo_v70` | [main.py](evaluation/opponents/nagatakengo_v70/main.py) | Historical panel |
| `aurax_reactive_v1` | [main.py](evaluation/opponents/aurax_reactive_v1/main.py) | Historical panel |
| `thomas_955_v2` | [main.py](evaluation/opponents/thomas_955_v2/main.py) | Historical panel |
| `shop0909` | [main.py](evaluation/opponents/shop0909/main.py) | Historical panel; earlier Shop Router notebook |
| `aurax_shop_v2` | [main.py](evaluation/opponents/aurax_shop_v2/main.py) | Historical panel |
| `seven_turn` | [main.py](evaluation/opponents/seven_turn/main.py) | Historical panel; earlier Seven-Turn Rescue notebook |
| `ahmed_v27` | [main.py](evaluation/opponents/ahmed_v27/main.py) | Historical panel; earlier Ahmed notebook |
| `submission_56140347` | [main.py](evaluation/opponents/submission_56140347/main.py) | Historical panel; earlier own submission |
| `submission_56140351` | [main.py](evaluation/opponents/submission_56140351/main.py) | Historical panel; earlier own submission |
| `submission_56146577` | [main.py](evaluation/opponents/submission_56146577/main.py) | Original JointAFS R1, 100-seed panel |
| `submission_56149565` | [main.py](evaluation/opponents/submission_56149565/main.py) | Original JointAFS R2, 100-seed panel |

The historical panel contains 13 opponents, 25 seeds and both seats, yielding 650 games per candidate. Each original JointAFS opponent uses 100 seeds and both seats, yielding 200 games per candidate. With three local candidates, the total is `3 * (650 + 200 + 200) = 3,150` games.

The three notebooks first acquired on September 9 are already represented here. Their original runtime-file hashes match `shop0909`, `seven_turn` and `ahmed_v27`, respectively. Additional notebook metadata in the later pool does not change those runtime files. The policies share route data and must not be counted as three independent strategy families. The historical selection distinguished the latest tested notebook version from the version that earned a qualifying public score; ratings are dated acquisition evidence, not current rankings.

Original JointAFS R2 is the user-supplied archive associated with `56149565`; the association uses acquisition context and its entry label. The later R1/R2 workflow-repair builds are different programs. See [exact version identities](docs/agent_approaches/PUBLICATION.md).

The local candidate programs remain in their original source packages: [frozen P16](nt/latest_20260910_r2p16/), [complete workflow](nt/latest_20260910_r2p16_route_workflow/), [procurement recovery](nt/latest_20260910_r2p16_route_recovery/) and [latest local repair](nt/latest_20260911_r2p16_route_repair/). They are candidate arms, not extra members of the 15-opponent panel.

## 2. Verify a checkout without running games

From the repository root, Python 3 and its standard library are sufficient:

```bash
python evaluation/verify.py
python docs/agent_approaches/verify.py
python evidence/r2p16_route_repair_20260911/verify.py
python evidence/r2p16_route_agents_20260911/verify.py
```

| Tool | Checks |
|---|---|
| [evaluation/verify.py](evaluation/verify.py) | All 15 opponent identities, 61 runtime files and eight frozen match-tool files against their manifests and original pool receipt |
| [Two-approach artifact verifier](docs/agent_approaches/verify.py) | Imported route-package bytes, unchanged source Git blobs and selected P16 submission identities |
| [Latest evidence verifier](evidence/r2p16_route_repair_20260911/verify.py) | Preserved build/source identities and the 3,150 recorded outcomes |
| [Earlier evidence verifier](evidence/r2p16_route_agents_20260911/verify.py) | Earlier workflow/recovery records and associated file identities |

Hash checks establish the identity of the copied artifacts. They do not rerun matches, reconstruct every replay transition or certify competition timing. Git attributes preserve the external agents' original bytes, including line endings. Large duplicate action tapes remain in the locations expected by their unmodified programs; Git can reuse identical blobs.

## 3. The compiled simulator used for the local P16 matches

The actual match loop used the **already compiled P16 C++ simulator**. A thin CPython extension exposes that simulator and records work/capacity diagnostics; it does not contain a freshly reimplemented game engine.

The [tools/p16_match_host/](tools/p16_match_host/) directory now preserves the original tool files:

| File | Role |
|---|---|
| [native_pool.py](tools/p16_match_host/native_pool.py) | Live candidate/opponent loop; calls the compiled transition, records actions, timing, cash, work and compressed replays |
| [pool_test.py](tools/p16_match_host/pool_test.py) | Shared path/configuration helpers and the earlier Python-reference panel used for calibration |
| [native_bridge.cpp](tools/p16_match_host/native_bridge.cpp) | Observation/action conversion and Python binding around the frozen simulator |
| [native_audit.inc](tools/p16_match_host/native_audit.inc) | Work-completion, invalid-action and resource metrics used to generate the bridge |
| [build_native.py](tools/p16_match_host/build_native.py) | Original bridge-build recipe; links the existing P16 library rather than rebuilding its simulator |
| [_pool_sim extension](tools/p16_match_host/_pool_sim.cpython-312-x86_64-linux-gnu.so) | Exact historical CPython 3.12 / Linux x86-64 bridge |
| [NATIVE_BUILD.json](tools/p16_match_host/NATIVE_BUILD.json) | Compiler command, dynamic dependencies, hashes and simulator symbol check |
| [NATIVE_CALIBRATION.json](tools/p16_match_host/NATIVE_CALIBRATION.json) | Archived stepwise comparison with completed official-interpreter reference games |

The bridge SHA-256 is `9f56f04c642212ea2a7720e95fb8c0f6f8989ad826904622488bc8ed298cc0b7`. It resolves simulator symbols from P16's [startupsupply2.so](nt/latest_20260910_r2p16/policy/startupsupply2.so), SHA-256 `4d7ef55bb79e4b312445a67628dd15609421e8c9afff21e8192a9b1b055f875b`. The matching simulator header and source are retained under [executor/vendor/](nt/latest_20260910_r2p16/policy/executor/vendor/).

The original extension has an **absolute dynamic-library dependency on the source workstation's baseline path**. The original scripts also retain experiment-relative paths. They are frozen evidence of the toolchain, not a portable prebuilt installation. Native Windows Python cannot load these ELF libraries, and the bridge specifically needs CPython 3.12.

The native transition is the match engine. The official Python source is loaded for configuration defaults and used separately as the comparison authority. Loading those defaults does not mean the match loop uses the Python interpreter for transitions.

### Running the archived panels on the original workspace

On the source machine, with its original experiment directories present, use the existing Linux/WSL CPython 3.12 environment. From the workspace directory containing both `Kaggriculture/` and `experiments/`:

```bash
python3.12 experiments/r2p16_route_repair_20260911/run.py \
  --stage historical --tag new_historical_run \
  --arms workflow recovery fixed --workers 8

python3.12 experiments/r2p16_route_repair_20260911/run.py \
  --stage latest --tag new_jointafs_run \
  --arms workflow recovery fixed --workers 8
```

The runner freezes inputs and rejects a changed protocol under an existing tag. These commands use its recorded seed protocol. Re-running them is a replication of that protocol, not a fresh holdout. For a new strength experiment, define and freeze a new seed set first.

The [archived copy of run.py](evidence/r2p16_route_repair_20260911/scripts/run.py) is included for inspection. Executing that copy in its relocated evidence directory will not automatically reconstruct the original workspace paths. Full replay data and old calibration reference games are also outside Git.

### Using the frozen engine from a different Linux checkout

Keep the historical bridge intact. If its absolute dependency does not exist on another machine, build a new bridge in an external output directory while linking the shipped, unchanged P16 library. With GCC 13, CPython 3.12 development headers and pybind11 installed, the equivalent binding build is:

```bash
# Run from the repository root. This creates a new binding, not a new simulator.
P16_PACKAGE="$PWD/nt/latest_20260910_r2p16"
HOST_OUTPUT=/tmp/kaggriculture-match-host
mkdir -p "$HOST_OUTPUT"
g++-13 -std=c++20 -O3 -DNDEBUG -fPIC -shared -Wl,-Bsymbolic \
  $(python3.12 -m pybind11 --includes) \
  -I"$P16_PACKAGE/policy/executor/vendor" \
  tools/p16_match_host/native_bridge.cpp \
  -L"$P16_PACKAGE/policy" -l:startupsupply2.so \
  -Wl,-rpath,"$P16_PACKAGE/policy" \
  -o "$HOST_OUTPUT/_pool_sim$(python3.12-config --extension-suffix)"
```

This relocation recipe is documented, not executed during this publication. Record the new binding's hash and dynamic dependencies. As in the original build receipt, `fastkag::Simulator::step` must remain undefined in the binding and resolve from the frozen P16 library. Revalidate transition parity for the new environment before treating it as equivalent evaluation infrastructure.

To adapt the original Python host, point `native_pool.BASELINE` to the repository's frozen P16 package, `SOURCE` to `evaluation/`, `PACKAGE` and `BINARY` to the chosen local route candidate, and `OUT` to a fresh external run directory. Its job tuple is `(arm, opponent_id, seed, candidate_seat)`. Set `OLD` to the intended reference-replay directory only when checking that historical parity. Load the new bridge directory before the archived host directory on `sys.path`. The complete latest-repair wrapper additionally records deferred tasks; keep that wrapper when reproducing its diagnostics.

## 4. Agent loading and process isolation

Use [FreshPublic](nt/latest_20260910_r2p16/run.py) to load an opponent directory. It sets the working directory and import path so adjacent `actions.json`, Python helpers and native runtime packages can be found, and restores module/path state on close. Copying only `main.py` from a multi-file agent is insufficient.

Use separate processes per match when running the pool. Several agents retain global policy or native state, and the loader temporarily changes process-wide state. The historical native panel uses spawned workers with one task per child. Do not invoke multiple `FreshPublic` agents concurrently in threads within one process.

The original R2 and other submission wrappers load their shipped `.so` files through `ctypes`. Their policy libraries require Linux x86-64; the Python-version-specific restriction belongs to the simulator bridge, not to the plain `ctypes` wrapper itself.

The engine's seed is recorded in the experiment protocol and replay metadata, but is removed from the agent observations and configuration. Opponents act on live observations. Their own embedded route tapes are policy assets, not access to the other player's future actions.

## 5. Replay, diagnosis and result tools

| Tool | Purpose | Input/location constraint |
|---|---|---|
| [analyze.py](evidence/r2p16_route_repair_20260911/scripts/analyze.py) | Aggregate candidate/panel metrics and compare outcomes | Original run directories; inspect historical path constants |
| [independent_verify.py](evidence/r2p16_route_repair_20260911/scripts/independent_verify.py) | Independent terminal reward and result checks | Full source replay collection for transition-level reconstruction |
| [loss_review.py](evidence/r2p16_route_repair_20260911/scripts/loss_review.py) | Collect failed obligations and loss diagnostics | Source replay/audit files |
| [regression_evidence.py](evidence/r2p16_route_repair_20260911/scripts/regression_evidence.py) | Compare selected first divergences under matching observations | Selected paired replay/audit records |
| [timing.py](evidence/r2p16_route_repair_20260911/scripts/timing.py) | Serial timing reruns separated from formal strength results | Original host and runtime environment |

The match host writes `<name>.json.gz` for replay frames, `<name>.audit.json.gz` for work/timing diagnostics and `<name>.result.json` for the terminal receipt and hashes. Published [compressed per-game records](evidence/r2p16_route_repair_20260911/runs/) allow result recounting; full multi-gigabyte replays remain on the source machine.

Report wins, draws and losses with the full denominator, retaining failed runs. Pair seats and seeds when comparing candidates. Inspect relative cash, wages, invalid actions, unexplained omissions and explicit deferrals separately. A deferred task is not completed work. The local host records time without imposing Kaggle timeout forfeits, so cash win rates are not sandbox timing certification.

## 6. Tools for route clustering and learned switching

The other approach has its own complete [offline pipeline](agents/route_clustering_switch_agent/scripts/run_pipeline.sh), [requirements](agents/route_clustering_switch_agent/requirements.txt) and [native simulator](agents/route_clustering_switch_agent/fast_kaggriculture/). It clusters replay intentions, audits route representatives, simulates static routes and switches, trains trees, evaluates held-out seeds and exports the runtime. Follow the [method guide](agents/route_clustering_switch_agent/docs/METHOD.md), including the manual quality gate before large searches.

Its 175 route-carrier opponents are already shipped as [route metadata](agents/route_clustering_switch_agent/runtime/route_library.json) and [compressed action tapes](agents/route_clustering_switch_agent/runtime/route_actions.json.zlib), executed through a common base policy. They are different from the 15 live-agent directories above.

```bash
cd agents/route_clustering_switch_agent
python -I main.py
python -I runtime/main.py
PYTHONPATH=src python -m pytest -q tests
```

The offline native package targets default rules version 1.32.7 and has its own differential tests. Do not assume that the route-search simulator, the P16 binding and the older general repository lab are interchangeable merely because each runs Kaggriculture.

## 7. What was checked in this follow-up

All 61 copied opponent files match the original 15-agent pool receipt. The eight copied match-tool files are preserved without modification. The earlier three notebook runtimes match their aliases in this pool.

A Linux CPython 3.12 smoke check loaded each copied opponent in a fresh process and ran four initial live steps against a PASS policy using the original compiled P16 simulator. [SMOKE.json](evaluation/SMOKE.json) records those checks. This verifies initial loading, required assets and basic action shape; it does not establish full-game equivalence, new win rates or competition timing. No new tournament or training run was performed.
