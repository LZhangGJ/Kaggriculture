# TRI_A06_r12_r2

**Status: a source-built candidate for the controller's next evaluation. Overall
strict win rate >85% has NOT been demonstrated. This author iteration ran zero
new matches.**

The sole parent is `TRI_A06_r12_r1`, the A06 calendar-r12 lineage. The original
calendar optimizer, its r1 late finite-crop admission repair, positive-return
investment selection, rolling scheduler, delivery logic, resource reservations,
and r11 recovery remain present. No opponent implementation was fetched or used.

## The single production change

**Make an existing, survival-critical workforce wage reservation effective at
market execution, before discretionary capital purchases spend its backing.**

The parent `dp7::Controller::prepare_orders()` subtracts `hirecost(h)` before
admitting purchases against conditional sale proceeds. Its final order queue,
however, was `sales -> admitted purchases -> HIREs`. Actual competing supply,
input prices and partial fills can invalidate the sale-funded budget. Purchases
can finish first while the intended maintenance workforce never arrives. That
breaks the connection between an incumbent continuous crop's conditional future
value and the work needed to keep the plant alive.

The new `policy/executor/ongoing_labor_reserve.hpp` changes only preparation queue
assembly. It activates only when all of these conditions hold:

1. The existing plan already requests hires and has sales; currently observed
   cash alone does not cover its admitted purchases plus those wages.
2. An existing tomato or strawberry is unwatered today and has at least one
   consecutive unwatered day: another omitted watering risks death at refresh.
3. The actual job list contains WATER for that incumbent, not DIG/PLANT for a
   replacement, and it still has an official production wave after today and
   no later than day29.

Under this guard the order is:

```
existing sales
-> admitted priority-0 WHEAT needed by today's jobs
-> exactly the already-planned HIRE orders
-> the other admitted purchases, in their original relative order
```

Future feed buffers and fertilizer are not reclassified as priority-0 feed.
No order or quantity is added, no new money is booked, and neither the staffing
cap nor the calendar is changed. Queue length and the ten-orders-per-frame limit
are unchanged. All ungated cases use the exact parent queue order. Orders still
settle against actual cash. The new helper never marks a failed purchase as
funded; existing observation-based stock/worker checks and recompilation remain
in place.

The only functional source edits are the queue-construction call in
`policy/executor/policy.hpp` and this new helper header. `main.py` changes only
its version-identification docstring. `provenance/SOURCE_DIFF.patch` is the full
source diff. Config and compiler flags are byte-identical to the parent.

## What the supplied observations actually establish

All seven own-visible traces were read. The historical own-crop audit applies
only official own-unit actions, deterministic plant decay and plant refresh to
each actual observation, then compares with the next actual observation. It
matched all **5,033 own ongoing-crop transitions**, and harvested quantities
reconciled with the supplied own ledger. These are parent-history checks, not
candidate rollouts.

The initially suspected generic harvest/warehouse loss did not explain the
three highlighted large losses: recorded strawberry production, harvesting
and sale quantities were respectively 227/227/227 for market_smart, 190/190/190
for reactive120540881, and 210/210/210 for the R2 worst loss. No held-capacity
production loss was observed for these strawberries. This does not validate
the optimizer's entire future-value model; it only rules out treating those
already-produced units as uncollected cash.

There is a different concrete failure in
`submission_56149565_852484341_seat0`:

- At step216/day9/hour0, cash was482 and there were no hired hands. The parent
  requested seven hires but placed land, animals and seeds before them.
- After its two preparation frames, cash was0 and only one hire had actually
  succeeded. The market-order requests and next own-private stocks show partial
  seed fills as well. The parent later hired three more hands through r11.
- Nineteen visible ongoing plants needed survival watering at the day start;
  sixteen strawberries died at the step239 refresh before ever producing.
- The own-only current-quote estimate left3481 after sales and immediate feed;
  the visible actual fixed-price fills imply3321 available at that point: a160
  **net price-execution shortfall**. This combines sale receipts and feed costs;
  it is not attributed to a particular unseen opponent order.

The r11 rescue mechanism intentionally considers bundles with current output,
so it is not a general replacement for funding immature crop maintenance. It
was not expanded in this iteration. See `evidence/OBSERVED_MECHANISM.json`,
`evidence/raw_logs/own_crop_audit.json`, the raw event log and the original own
traces/ledgers for the evidence. **Neither a failed order nor a dead plant is
claimed to represent recoverable terminal profit.**

## Validation completed

| Check | Actual result | Interpretation |
|---|---|---|
| Input provenance | Attached ZIP SHA256 and all72 manifest entries match | Exact current source package |
| Parent identity | All48 original parent files unchanged; rebuilt parent native matches | No lineage replacement |
| Parent entry reproduction | 7 x719 =5,033 exact own actions | Historical reproduction only |
| Source unit suite | One C++ suite,18,403 assertions,zero failures | Guard boundaries, operation/quantity conservation, stable priority order |
| Official atomic market fixtures | 15 explicitly synthetic market conditions,32 assertions | No match, no opponent-private data, no invented cash |
| Fixed-plan execution fixtures | Four market conditions x two queue arrangements; all eight complete | Actual stocks/worker count recompile, official own routes and refresh |
| Fixed-route feasibility | No ineffective prescribed actions, no overdrafts/negative stocks, routes fit22 remaining hours | Conditional component reachability, not a live candidate day |
| Root `main.agent` smoke | 336/336 early actions exact; reset and malformed-input tests pass | Real root loader and matching native |
| Common-prefix probes | 1,176 common actions plus seven first-divergence actions | Each case stopped immediately at first changed action |
| Staged offline suite | All six stages pass | Reproducible tests included |
| Independent production rebuild | See BUILD/rebuild receipt: byte-identical native | Sources match the shipped Linux library |

### What the stress fixtures mean

The atomic tests call the frozen official market commit/hire functions for only
the known own farm. Changes to current public MILK/FERTILIZER/WHEAT inventories
are explicit **synthetic stress inputs**, not inferred historical opponent
orders. With no shock, both orderings obtain the same seven planned hires.
Several reduced-receipt fixtures preserve more hires with the repair. An
extreme feed-price/low-receipt negative fixture still cannot fund all seven:
the repair never fabricates a wage reserve or borrows future crop cash.

The route fixture reproduces the parent context through step216, holds its
investment/service intentions fixed, settles the two preparation batches,
and recompiles the unchanged executor against the resulting real fixture
stocks and hands. It credits no extra work during the second preparation
frame. It then applies the compiled own-unit routes with official rules until
that day's refresh. It does not run opponents, random events, later sales,
intraday reoptimization or a full episode.

For the synthetic +5 MILK/+2 FERTILIZER inventory shock, the old ordering gives
2 hands and waters3 of19 at-risk plants; the repaired ordering gives7 hands and
waters19 of19. Both prescribe only effective actions within the day. With no
shock both water19/19. **These are bounded component tests, not16 recovered
plants in a real candidate game, not recovered cash, and not wins.**

### Narrow-win protection is not proven

| Historical case | Parent margin | First changed action step |
|---|---:|---:|
| aurax_reactive_v1_521596133_seat0 | +42 |120 (day5/hour0)|
| submission_56149565_1123506227_seat0 | +67 |168 (day7/hour0)|

The other first divergences are144,216,168,216,144 in the supplied selected-case
order. At every first divergence the complete queued market operation/quantity
multiset is unchanged; the order differs. This does **not** prove later policy,
state, prices or terminal outcomes remain equal. No saved suffix after a
changed action was fed back as the candidate's real future.

## Historical score and controller acceptance

The supplied full1536-row panel was independently recounted: parent1099 wins,
437 losses,zero draws/errors,all719 steps. Public11:984/1408; originalAFS R2:
115/128. Overall1099/1536 =71.549479%. These are supplied historical parent
results, not new candidate results. All1536 rows and original metrics are
retained in `evidence/input/`; the seven examples never replace the denominator.

The next controller evaluation must freeze this candidate identity and use its
fresh64-seed12-opponent,two-seat,1536-game panel. **At least1306 strict wins**
are required. Neither that test nor an overall win-rate improvement is claimed
here. Different-round seeds do not establish causal changes in win rate.

`DEVELOPMENT_SEEDS.json` records all64 public-result seeds seen and the five
unique seeds used for the seven historical observation examples. No new seed
was drawn, and no seed/opponent label is consumed by production policy code.

## Build and reproduce offline

Requirements: Linux x86-64, Python3, a C++20 compiler, GNU `timeout`/`time` for
the supplied bounded runner. Production uses only the Python standard library
and the shipped C++ library; it needs neither Kaggle installation nor network.
The official source copy is used only by offline tests.

```
python build.py
python build.py --unit
python tests/run_all.py
python analysis_tools/audit_ongoing.py
```

The first command rebuilds `policy/a06.so`; the second also runs the new focused
source tests. `run_all.py` is serial and bounds each child command with timeout.
Its six stages write individual commands, outputs, errors and results under
`build/validation/`. It builds a separate **test-only parent-context library**;
that library is never loaded by production `main.py`. Historical own-transition
re-auditing is a separate explicit command, not a tournament.

`BUILD.json` and `policy/a06.BUILD.json` record the actual successful production
command, g++ version, compiler flags, configuration, each source/input SHA256
and the native SHA256. `SOURCE_FREEZE.json` fixes current build inputs.
`BUILD.md` and `evidence/raw_logs/` contain wall-clock/RSS and independent rebuild
records. No downloaded dependencies or fabricated build results are used.

## Provenance, resources and retained failures

`provenance/parent_agent/` contains the exact48 input parent files, and
`provenance/TRI_A06_r12_r1_immutable_parent.tar.gz` preserves a read-only snapshot.
Both are separate from the active root policy. Hashes are verified again on
final extraction. Initial Python imports added two derived bytecode caches to
the read-only-mode working directory because execution used root privileges;
all original48 files remained unchanged. Those caches were recorded, removed,
and bytecode writes disabled for subsequent tests. They are not included in
the immutable archive.

Initial measured affinity exposed5 logical CPUs, but cgroup quota was
400000/100000, so effective CPU budget was4. Memory limit was4GiB with about286MiB
initial cgroup usage. Production builds/tests used serial workers and reserved
at least30% of memory. Host MemAvailable was recorded only as context, never as
the container entitlement. Python3.13.5 and g++14.2.0 were measured, not assumed.

`FAILURES_AND_LIMITATIONS.md` retains two failed *negative test-fixture
expectations*, their exact source versions and raw errors; they did not require
production changes. It also records initial tool orchestration failures and
missing outer timing receipts rather than inventing measurements. Successful
retries have explicit exit/wall/RSS receipts. All work uses the original
2026-09-13 15:24:05.645Z dispatch and17:24:05.645Z deadline; supplements did not
reset it. Final operation times are in `TIMING.json` and the external delivery
receipt.

This repair can shift purchases across preparation frames and alter responses
to market competition. It does not guarantee future staffing, maintenance,
transport revenue or preservation of the two narrow wins. The full controller
win-rate test remains the decisive next check.
