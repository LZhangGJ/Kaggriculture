# TRI_A06_r6_r1

**Status: a compiled correctness-repair candidate for central evaluation, not an
accepted >85% agent and not a demonstrated win-rate improvement.**

This is an independent iteration of the supplied **A06_r6 only**. The complete
production policy, root `main.py`, fixed configuration, Linux native library,
offline builder, tests, raw logs and source/native identities are included.
No other agent lineage was substituted.

## Change

Only **`policy/planner.hpp`** changes among the 40 production source/configuration
files. In `Planner::trade`, a positive forecast sale that crosses the 1-unit cash
price floor now uses the existing sequential cash/stock primitive. A successful
floor-priced sale pays cash but does **not** add market supply. The parent's bulk
average-quote shortcut could overcount or undercount inventory at this boundary,
and pass an incorrect recovery price into lifecycle and maintenance valuation.

The fix is restricted to that boundary. The parent formula remains bit-identical
for the tested non-crossing flows and purchase paths. This is not a global switch
to exact market integration: `R2_MARKET_INTEGRAL=0` stays unchanged. The root entry,
Python observation codec, all settings, build flags and all executor sources are
byte-identical to the parent. Positive-return investment, rolling candidate
selection, paid commitments, maintenance reconciliation, labor/transport/resource
execution, and the r6 public-supply sale-risk mechanism are retained.

The soil and aurax-shop loss traces contain a concrete example: from wool stock
10056, selling 10 units produces conditional cash 41 and counted stock 10059.
The old forecast estimated cash 37.7777777778 and stock 10066. This is a **model
state error**, not a claim that the cash difference can be recovered from the
historical game: the official execution already paid the official amount.
Thomas, R2 and narrow-win examples are also documented.

See `docs/CAUSE_AND_SCOPE.md`, `validation/diagnostics/production_change.diff`,
and `validation/diagnostics/market_floor_evidence.json` for the exact change,
source-level coupling, and distinctions between actual transactions and
one-sided conditional calculations.

## Actual validation in this turn

| Evidence | Actual result | Meaning / limitation |
|---|---|---|
| Parent source/native check | All 40 recorded source hashes matched; parent rebuilt byte-identically | Confirms the sole lineage |
| Initial legal-observation probe | 216/216 parent actions reproduced | Timing/RSS probe, not new games |
| Seven supplied historical replays | 719 matching actions and official transitions, 60 reconciled player-days each | Historical reproduction only |
| Existing plus new C++ suites | 5/5 compiled and passed | Full raw output and actual commands included |
| New boundary tests | 2061 positive cases, 729 bit-identical negative cases, 234 fractional cases, 9 demand-recovery cases | Tests the production `Planner::trade` |
| Independent Python official oracle | 2061 production-C++ cases / 73179 successful unit commits matched | One-sided conditional primitives, not games |
| Legal historical prefix comparison | Seven cases stop at their first revised action | No saved suffix used as a counterfactual |
| Root/ABI/unknown-field tests | Four legal cases in both seats passed; 44-setting ABI matched | Fabricated seed/name/private/future canaries were not encoded |
| **New actual closed-loop games** | **8 wins, 8 losses, 0 draws, 16/16 complete; 719 steps each; zero runtime exceptions** | **Eight predeclared fresh seeds, both seats, against exact A06_r6 parent only** |
| Independent audit of the new logs | 11504 transitions and 960 player-days reconciled | Replays the 16 actual games; does not add games |

The new head-to-head strict win rate is **50%**. It does **not** establish a
win-rate advantage over the parent. Mean cash margin was +1259.1875, but cash
margin is diagnostic and does not override the win result. The seed panel was
written before the first revised game and was not resampled after outcomes.
There were no PASS-opponent games and no full 1536-game launch.

The physically extracted ZIP rebuilt to a byte-identical native. Its actual
root `main.py:agent` matched 1438 calls across the two seat variants of a recorded
new game. `DELIVERY_CHECKS.json` contains these successful receipts. They are
reproducibility checks, not additional tournament wins.

## Historical and central results must stay separate

The supplied historical development baseline is unchanged: **374/480 overall**,
**339/440 public**, and **35/40 original AFS R2**. All 480 rows are retained in
`validation/input/HISTORICAL_ALL_480_ROWS.json`. These are not this revision's
scores and are not a sealed holdout.

The central mixed development panel is 64 seeds × 12 opponents × 2 seats = 1536
games, requiring **at least 1306 strict wins**. It was not run here; its result
was not available in this turn. No executable instance of the eleven named
public opponents or true original AFS R2 was obtained and run for this revision.
Consequently, neither >85% overall nor preservation of the historical R2 win
rate is verified. This candidate should be evaluated under its frozen identity;
it should not silently replace the parent based on forecast correctness alone.

## Run and rebuild

The callable entry is **`main.py:agent(observation, configuration)`**. Keep
`main.py` and `policy/` together. The root entry loads `policy/a06.so` explicitly;
`create_agent()` supplies an independent context for a local driver, and
`reset()` releases both root seat contexts. Each actual policy receives only
its own private state and public fields. Seed values in the validation folder
belong to the environment driver, not to production policy lookup logic.

In a disposable copy, with a compatible Linux x86-64 C++20 toolchain:

```sh
sha256sum -c SHA256SUMS.txt
python3 -B build.py --cxx g++ --out /tmp/TRI_A06_r6_r1_rebuilt.so
python3 -B tests/run_units.py --cxx g++
python3 -B validation/run_floor_oracle.py
python3 -B validation/test_runtime_privacy.py
python3 -B validation/compare_prefixes.py
python3 -B validation/audit_new_games.py
```

Run the checksum verification **before** mutating files with tests/builds; those
commands write logs and receipts. `BUILD.md` and `BUILD.json` contain the actual
compiler/flags/commands and offline reproduction details. No network download is
required. The tested compiler is GCC 14.2.0; compiler-version changes are not
promised to produce byte-identical native output. Linux symbol-version
requirements are recorded; Kaggle sandbox/resource-limit compatibility has not
been certified.

## Identity

Input ZIP SHA256:
`bbd5112684c8cb910d46d163c450b75035360c42a073bbb3975a005d984e6b12`

Parent native SHA256:
`4119ae39b07c5fd2ef7f79d2f30ee3ee55f0dcd8980707d133a75b92e1140342`

**This candidate native SHA256:**
`174ba2ad749cb3213a72a03302ef50b08a37a579848ba922bde132a19b8b6c23`

Changed `policy/planner.hpp` SHA256:
`19e9c39dee13e22d84f1e91cd84df046f0d5e27b756b6176546b13c87653df5c`

`IDENTITY.json`, `BUILD.json`, `policy/a06.BUILD.json` and `SHA256SUMS.txt` identify
all sources, configuration, native output and evidence. The original parent
input remains under `validation/input/`; it is a reference, not the root runtime.
There is no nested original-delivery ZIP.

## Resource and deadline record

Original dispatch: **2026-09-13T02:17:41.523Z**. Hard deadline:
**2026-09-13T04:17:41.523Z**. Initial resource measurements were recorded at
02:18:23.287461Z. The substantive source fix was written at 02:29:20.170644Z and
packaging began at 02:38:32.519362Z. Final task elapsed time and receipt inventory
are in `TIME_AND_RESOURCES.json`.

CPU affinity exposed five logical CPUs, but the cgroup quota was **4 CPUs**, so
four was the effective CPU limit. The cgroup memory limit was **4 GiB**, with a
70% operating cap (**2.8 GiB**); host `MemAvailable` was recorded but never used
as the allocation limit. Tests used one game worker, with at most one compiler
plus one validation process concurrently after measurement. The observed peak
cgroup usage before packaging was **1412153344 bytes**, well below that cap.
The revised compile measured 72.12 seconds and a peak child-tree RSS of 663876 KiB.
The 16 actual games took 314.62 seconds with a 116580 KiB child-tree peak.

See the raw `validation/logs/*.receipt.json`, initial resource receipt and
`docs/TOOLING_NOTES.md`. The initial baseline compile's outer tool wrapper timed
out while its compiler continued successfully; no peak RSS from that failed
wrapper is claimed. Later measurements use completed bounded-runner receipts.

## Remaining work and limitations

The central public/R2 comparison, >85% acceptance, Kaggle sandbox limits, and
broader fresh-seed generalization remain unverified. The parent still uses an
approximate daily rival-flow/demand model, ordinary non-crossing bulk quotes,
and a conditional rather than clairvoyant market forecast. The floor repair
must not be interpreted as future-price certainty or spendable predicted cash.
Failed orders, bought-then-sold resources, and unsold goods were diagnostic leads,
not automatically counted as recoverable profit.
