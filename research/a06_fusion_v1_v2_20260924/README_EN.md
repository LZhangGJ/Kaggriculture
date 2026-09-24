# A06 Fusion V1 / V2

V1 is the retained baseline at `v1/main.py`; V2 is the unpromoted comparison at `v2/main.py`. Each folder contains 67 files, byte-identical to its evaluated frozen version, including Python/C++ sources, configuration, Linux runtime, and build script. Neither uses ML/RL.

## Changes

Both preserve the original Cashflow agent's native planner/executor, cash-flow valuation, and sale adjustment based on publicly visible mature opponent output. They add Liquidity's opening wheat buy-10/sell-10 intention with cash, storage and order-count guards; disable intraday new-project admission and procurement; and cap hired workers at 12. Daily economic planning and existing task execution remain active.

V1 uses `animal_bias=1.0`. V2 changes this new-animal candidate score multiplier to `0.6`; it does not impose a fixed animal count. Native source and binary are unchanged. Neither policy uses opponent names, test seeds, private opponent state, or actual future markets.

## Evidence

| Opponent | V1 | V2 |
|---|---|---|
| Teammate V306 | 119/200 (59.5%) | 123/200 (61.5%) |
| Teammate V463 | 110/200 (55.0%) | 123/200 (61.5%) |

All 800 main games reached 719 transitions with no errors or draws. The opponent's original sampled policy completed all 17 model-controlled days in every game with no fallbacks or illegal choices. Both candidates used the same 100 environment seeds and both seats. Independent recorded policy RNG seeds were paired across candidates. Eight smoke reruns and six serial reruns exactly matched both action hashes and terminal cash; they are excluded from the 800 games.

Earlier separate holdouts: V1 internal 95.62%, external 86.10%; V2 internal 95.54%, external 83.20%. Those holdouts used different seed sets. A subsequent same-seed external comparison was V1 86.30% versus V2 83.20%. The dual-90% target was not met. V2 was not promoted; its small teammate-panel advantage is not proof of a stable or general improvement. See the full reports and seed-cluster intervals in `evidence/`.

Observed serial maximum calls against V463 were 1.108 s for V1 and 1.214 s for V2; a V2 call against V306 took 1.557 s. These local official-1.32.7 games did not enforce Kaggle sandbox timeout forfeits. They are strategy comparisons, not hosted runtime certification.

## Reproduce

Use the full version folder: `main.py` imports adjacent helpers and loads configuration and `policy/a06.so`. The supplied library is Linux x86-64, requiring GLIBC_2.32 and GLIBCXX_3.4.31 or compatible runtime libraries; it is not a Windows DLL. Each version has a `build.py` for rebuilding the C++20 source. Revalidate after rebuilding with another compiler.

From this package directory:

```text
python verify_package.py
python reproduce_one.py --candidate-version v1 --pool ../a06_dynamic_variants_public10_20260924 --opponent internal/r14_cashflow --seat 0
python reproduce_teammate.py --candidate-version v2 --opponent student_v463 --opponent-root /path/to/extracted/student-v463 --seat 0
```

The first replay uses the already-shared internal/public opponent pool. Teammate replays require the original fully extracted submission folder, provided with `--opponent-root`; opponent models/archives are not duplicated here. Original archive hashes:

- V306: `908235e386d73252505a7b3879079d07e835d1c0ec78533f13a69ea1f2c2bdaa`
- V463: `7a8fcabeb04096e330bacc3e6d6acb42a9cb3b93567c0bfb1b4cb1de05092445`

The teammate replay verifies source hashes, seeds the original sampled opponent independently of the environment, and compares final cash and full action hashes with the saved receipt. Evaluated environment: Python 3.12, CPU PyTorch 2.9.1, NumPy 2.2.6, one thread per process. PyTorch is only needed for the teammate opponent; V1/V2 themselves do not use it.

Original experiment scripts and receipts are retained under `evidence/teammate/`; those archived scripts may contain old workspace paths. Use the root scripts for portable replay. Historical reports' `agent_v1/` and `agent/` correspond to this package's `v1/` and `v2/`. Inherited parent READMEs/reports inside version folders are historical provenance, not this fusion's performance claims. The root `MANIFEST.json` is the current integrity manifest.
