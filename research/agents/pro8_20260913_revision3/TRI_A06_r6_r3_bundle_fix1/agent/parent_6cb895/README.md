# TRI_A06_r6_r3

**Status: runnable source/native candidate for a new central 1,536-game panel.
No new matches were run here. Strict overall win rate >85% is not established.**

## Identity and entry

The sole parent is the attached **b10d r2 search.hpp version**. The separate
79a769 triad.hpp delivery is NOT the parent. Its whole-land-bundle comparison is
ported explicitly; its centrally reported diagnostic outcomes are not r3 results.

- Parent native: `b10d86c102f68e5863559e6c049369571cbc123aa4db71db908a10c1ab14eab1`
- Candidate native: `6cb895db1484045699e8b81f11ed1d753507e02d9d57af81d037845dbf73441e`
- Entry: root `main.py:agent(observation, configuration)`.
- Runtime: Linux x86-64 native `policy/a06.so`; Python standard library only.
- Fixed `main.py`, `policy/agent.py`, `policy/config.json`, compiler flags,
  `policy/search.hpp`, and the correct r1 `policy/planner.hpp` floor-price repair
  are unchanged from b10d.

Of the parent's 40 tracked production/build-input files, **38 are byte-identical**.
Existing `policy/executor/policy.hpp` and `policy/triad.hpp` change; new
`policy/executor/labor_funding.hpp` is added. The complete production policy is
present, not a patch, opponent substitution or placeholder. `parent/` is a complete
read-only parent copy, verified against `validation/immutable_parent_hashes.json`.
`IDENTITY.json`, `SOURCE_HASHES.json`, `BUILD` and `PRODUCTION_DIFF.patch` distinguish
all identities and changes.

## What changed

### 1. Fund the selected crew before optional capital, then verify actual arrivals

The previous preparation queue was sales, all purchases, then HIRE. Consuming
that queue token did not prove the hire had happened. Actual trading, partial
fills and the ten-order boundary could leave too few workers. Subsequent sales
could restore cash without restoring the selected maintenance crew.

The new order is **sales, today's required feed, selected wages, remaining land /
animals / seeds / fertilizer / feed-buffer purchases**. Every originally admitted
order and quantity is retained for the same chosen plan; initial order count and
page count are unchanged. Economic admission and executable preparation previews
see that same ordering. This is a priority repair, not a ban on expansion or feed.

The controller also remembers the selected daily staffing target. After preparation,
if the real roster is smaller, it checks a bounded recovery action on every eligible
current observation. Recovery is allowed only for still-unassigned, currently
intended service of existing crops/animals, with actual unreserved inputs, a route
that fits the remaining day, a spare slot in the current ten-order page, and positive
service-utility-minus-current-wage. The utility is explicitly a heuristic, not
recoverable cash. Replacement-cost credit is limited to at-risk incumbents whose
existing service policy still opts in.

Funding is checked after the actual chosen own-unit prefix and currently submitted
own sales/material orders. It does not count future harvest, bags not deposited,
unsubmitted sales, rival private stock or actual future events. Already outstanding
HIRE orders are observed before issuing another hiring decision. Required feed is
not displaced by wages, and pending projects/incumbent pickup reservations are not
borrowed by a new worker.

No hypothetical worker plan is installed. On the next real observation, only
actually arrived workers get new routes. The hiring hook does not discard incumbent
routes. The unchanged scheduler can still perform its normal legal recoordination.
It does not generically rebuild the entire day from imaginary prepared resources.

### 2. Compare the entire land-plus-project bundle

`triad::Controller::plan` now compares expansion with deferring land from the same
pre-plan state, using the complete admitted portfolio after executable preview.
The per-slot loop had previously charged land before testing project increments;
positive project increments did not establish that the entire bundle beat waiting.

This is the bounded comparison ported from 79a769, evaluated under the new funding
execution too. A rejected expansion installs the *whole* alternative controller:
portfolio, project paths, prices, commitments, orders, resource reservations and
service decisions. No partial target/queue transplant occurs. Positive relative
conditional values can still expand; tomorrow can reconsider land. The b10d search
layer and its paired land alternatives remain intact.

The comparison is still conditional model value, NOT executable terminal cash or
proof of a win. It is an additional economic guard paired with a real execution repair.

## Offline build and tests

From this directory, with Python 3 and a C++20 g++ toolchain:

```sh
python3 -B tests/verify_delivery.py  # before rebuild changes receipts
python3 -B build.py --cxx g++
python3 -B build.py --cxx g++ --unit
python3 -B tests/check_entry.py
python3 -B tests/reproduce_parent.py
```

All non-toolchain dependencies, frozen official rule files and test fixtures are
inside the package. No pip installation, network connection, opponent implementation,
training data download or runtime compilation is required for the shipped agent.
Actual build used Debian g++ 14.2.0-19, `-O3 -std=c++20 -march=x86-64`, the unchanged
full flags in `COMPILER_FLAGS.json`, and Python 3.13.5. `BUILD` records exact commands,
compiler, flags and source/native hashes. A compatible C++ runtime is needed to load
the Linux library; rebuilding is the offline portability path.

`tests/export_parent_fixtures.py` optionally rebuilds a parent inspection shim,
reproduces the first 315 parent actions of the R2 diagnostic exactly and regenerates
the two retained service/route fixtures. That shim and the fixture-loading test API
are not part of the production native interface.

## Actual validation, with denominators and scope

| Check | Result |
|---|---:|
| Exact b10d rebuild against attached native | byte-identical |
| Five current own-visible parent histories | 3,595 / 3,595 actions exact |
| Independent saved-state plan / funding invariants | 360 states across all 12 supplied histories |
| Whole land comparisons | 43: expansion higher 29, defer higher/equal 14 |
| Funding order changes | 249 / 360 independent states |
| Procurement orders checked, same multiset | 3,997 |
| Focused recovery positive/negative fixtures | 21 pass |
| Price-floor primitive regressions | 154 pass |
| Conditional same-day execution branches | 37 |
| Official own-unit + own-market prefix comparisons | 827 / 827 |
| Actual production library through root entry, matching test branches | 667 / 667 |
| Cloned production controller / callback checks | 638 / 638 |
| Independent root/reset checks | 45 states plus 12 stopped common-prefix probes |
| Undefined-behavior instrumented repeat | passed after correcting a test-only oracle type |
| Independent ZIP extraction, offline default build and `--unit` | exit 0; native byte-identical; all tests repeated |
| New closed-loop matches | **0** |

In the conditional root branches, 9 branches actually paid
for land and 9 of those started projects
on newly unlocked tiles; 123 such starts and
315 successful existing-animal feeds were recorded.
These are execution checks, not farming income or battle performance.

The official differential checker applies the chosen own actions and own market
orders with no opposing orders, then only known same-day town demand and deterministic
crop decay. It stops before hour 23 / midnight processing; there is no future weed,
shop or opponent script. All actual branch actions, cash, roster counts, fill counts,
full own-prefix signatures and comparisons are retained in `build/r3_units_final/`.
These bounded scenarios are not matches, PASS cash benchmarks or a replacement for
central evaluation.

## Evidence of recovery, not a claimed rescued match

Two controlled service tests begin at the actual later-cash observations with the
parent's exported remaining routes, targets and selected service fields. New-project
admission is disabled in both treatment and control to isolate the labor mechanism.
No source-match observations after the intervention are used.

| Actual starting state | Recovery action | Conditional at-risk unwatered plants at hour 23: control / recovery |
|---|---|---:|
| day9 step219: cash644, 2 hands | 3 HIRE filled; wages10; real next roster5 | 16 / 0 |
| day13 step315: cash2490, 6 hands | 2 HIRE filled; wages34; real next roster8 | 8 / 1 |

“At risk” means a current plant is still unwatered after already being dry one day.
It is **not** a terminal death or cash outcome. The second control's eight at-risk
plants are not the source game's six recorded deaths: the scenario and endpoint
are explicitly different. Likewise, the normal cold r3 day9 preparation paid seven
hands on its first page and still bought land, but that is not the original R2
opponent's closed-loop future.

## Limits and acceptance

Both protected narrow-win histories change their first action at step0 because
funding order changes. Their wins are **not** verified preserved. All twelve common
prefix probes stop at their first difference (step0); no saved suffix is interpreted
as a candidate trajectory. Legacy seven histories are preserved and used as independent
states and bounded tests, rather than an invalid after-divergence game replay.

Thomas and market-smart's zero-physical-loss defeats and strawberry floor sales are
not fully explained by staffing. The bundle guard may change admitted supply, but
sales timing, competitive demand, price forecasts and the quantity of profitable
supply remain uncertain. More surviving crops may even worsen price competition.
This patch does not assert that every low-price sale is avoidable or every expansion
is harmful. Existing routes can still be time-constrained; one modeled high-risk
plant remained in the day13 service experiment.

The supplied parent `PROGRESS.json` is an **incomplete** snapshot: 789 completed,
537 wins, 252 nonwins, at most1284 wins even if every remaining game wins. All supplied partial rows
and the original scheduled denominator1536 remain intact. No final parent rate is invented.
The alternate's four central-reported diagnostic comparisons are in
`evidence/SUPPLEMENT_REPORTED_BY_CENTRAL.md`, separately labeled.

Freeze this r3 identity and run the fresh shared64-seed, 12-opponent, both-seat,
1,536-game panel. The unchanged acceptance criterion is **at least1306 strict wins**.
Internal scores, wages saved, unit-test passes and conditional risk counts cannot
satisfy that criterion.

## Resources, time and failures

The initial resource probe measured affinity0-4 but cgroup `cpu.max=400000 100000`:
**4 effective CPUs**, not five. Memory limit4GiB; compiles and probes were serial
with 30% memory headroom. Initial parent compile28.35s, peakRSS579624KiB; the UBSan
compile used a 2.4GB virtual-memory bound and peaked below750MiB RSS. Native call
latency in 57 cold-entry probes had median0.04753s and maximum0.25351s locally;
these are not Kaggle sandbox measurements.

Original start18:44:08 UTC and hard deadline20:44:08 UTC were never reset. The first
observable resource probe was19:29:01 UTC, so the first-five-minute requirement was
missed; `validation/timing.json` states this rather than claiming compliance. Production
was frozen at the timestamp in `validation/production_freeze.json`; later work is
verification/packaging only. Final packaging times and resource records are retained.

A test-only int32 narrowing bug at the theoretical wheat/egg floor boundary caused
one new primitive test to fail. Its original source, binary, failed output and corrected
repeat are retained. No production price-floor algorithm was reverted. Tool-session
availability and bounded-command wait failures also have genuine recovery records.
See `FAILURES.md`; failed attempts are not discarded or called successful.

Final clean-extraction integrity is recorded in `validation/DELIVERY_INTEGRITY.json`
and `validation/clean_rebuild/`. The root entry checks and all 3,595 parent action
reproductions also passed from that separate extracted tree.

`RESOURCES_AND_TIME.json` contains the final pre-archive resource/time snapshot.
`MANIFEST.sha256` covers every archived payload file except itself. Verify it before
rebuilding, since an actual rebuild intentionally writes new receipt/log files.
