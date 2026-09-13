# TRI_A06_r6_r2

**Status: frozen source/entry-tested candidate for central closed-loop evaluation.
No new closed-loop games or win-rate claim are included.**

The only selected agent is `main.py` at the archive root, loading
`policy/a06.so`. The exact parent is TRI_A06_r6_r1. Its complete 45-file production
copy is under `provenance/parent/`, with original hashes and read-only archive
permissions. No other lineage or opponent implementation was substituted.

## The single production change

Only `policy/search.hpp` was edited. It adds a **matched no-new-land-today
alternative when every original economic proposal requires BUY_LAND**.
The intervention addresses a missing executable alternative, not an assumption
that expansion or a 7,000 land bill is inherently bad.

The parent planner charges the next quadrant before selecting all its uses.
At the audited R2 loss, day 18 / observation 432, all six original scored
proposals required buying the fourth quadrant. The selected post-preview bundle
initially contained only four tomato targets on that quadrant. Without a
same-input waiting alternative the selector could compare several ways of
expanding, but not those choices against keeping the land cash available today.

The final implementation:

1. Preserves every original proposal, including all positive expansion paths.
2. Does nothing when any original prepared queue already avoids BUY_LAND.
3. Otherwise replans matched alternatives from the incoming controller and
   commitment book, with today's land cap equal to the actually owned quadrants.
   It does **not** inherit the expanded proposal's unfunded new-plot book.
4. Uses the existing exact execution mechanism, one-day public-flow rollout,
   rival-cash competition term and common next-day continuation to choose.
5. Installs the selected cap into the real controller for today. The existing
   next-day restoration returns expansion eligibility to the fixed base setting.

Proposal IDs use a separate +32 band, and the debug choice counters were widened
from 32 to 64 entries. The existing +16 execution-mode pairing remains intact.
There is no new ROI threshold, search over seeds, opponent-name gate, learned
model, observation leak, permanent expansion ban, or removal of strategy modules.
All other production source/configuration and compiler flags are unchanged,
including the r1 market-floor fix in `policy/planner.hpp`.

## What the own-visible evidence actually supports

The simple unused-land hypothesis was **refuted**. Across the seven supplied
parent histories, all 21 land purchases were productively used on the purchase
day. Each case spent 1,000 + 2,000 + 4,000 on land, and all 210 own-player day cash
ledgers reconcile exactly. The input reports no failed market orders.

In the worst R2 case, the fourth quadrant was bought on action 432. The first
successful plant action on it was action 435, visible in observation 436.
Its later historical output included 36 tomatoes, 36 wheat and 27 carrots; it
received 92 effective water actions and 13 fertilizer actions. Thus it was neither
unused nor a pure 4,000 cash leak. Its initial bundle alone also cannot represent
all later rolling uses.

`validation/analysis/trace_land_evidence/` retains successful plot operations,
visible maturation events, whole-farm warehouse arrivals, and cash reconciliation.
Goods become fungible in carried inventory and the shed: **whole-farm sale
proceeds are not attributed to a particular quadrant**, and no recovered-cash
claim is made. The parent bundle-removal model values in the diagnostic audit are
conditional quotes, not realized marginal profits.

The narrower confirmed issue is candidate-set coverage: at that particular
buying decision the old pool contained no executable wait alternative. Whether
correcting this omission improves new-seed win rate remains a central test.

## Validation performed on the selected final source

| Check | Actual result |
|---|---|
| Input identity | ZIP SHA and all 69 manifest entries verified |
| Immutable parent | All 45 original files exact; native matches supplied SHA |
| Parent rebuild | Offline rebuild byte-identical to the supplied parent native |
| Parent actual root entry | 5,033 / 5,033 saved actions exactly reproduced |
| Final actual root entry | 5,033 calls, no runtime exception or malformed action |
| Full action-trace preservation | Six cases: 719 / 719 each; includes both narrow wins |
| First changed action | Worst R2 loss only: observation 432, day 18 |
| Original candidate preservation | 1,692 score/plan comparisons bit-exact before/at first divergence |
| Focused production-class fixtures | 35 states passed; 193 original and 130 deferred candidates checked |
| Existing commitment checks | 5,810 repeated funded-commitment checks across candidates |
| Bounded executable exercises | 37 declared public-flow branches, 888 ticks total |
| Official rule differential | 37 / 37 own unit+market prefixes exactly matched |

The exercises verify preserved expansion can actually purchase land, capped
alternatives cannot purchase or target locked land, and the land cap restores
next day. They also test candidate-ID uniqueness, original proposal preservation,
independent replanning, absence of extra branches when a wait choice already
exists, full land, and terminal-day cases. Assertions throw explicitly and are
not disabled by the inherited `-DNDEBUG` build flag.

The official differential compares one own unit phase and own market queue under
an explicitly absent-other-orders condition, stopping before town demand, decay,
day-end events or future RNG. It is a rule-prefix test, **not a PASS match**.
The 24-tick exercises use the parent's declared public-flow scenario, **not a
real opponent**. Unknown private inventory is never loaded from an opponent.

The seven root-entry probes call the production `main.py`, not only a helper
policy. No saved post-divergence observation is treated as the r2 action's true
future. In the changed R2 case, only the prefix through observation 432 supports
the first-decision comparison; later replay calls test entry robustness only.

## Narrow-win protection and remaining risks

- Thomas 1395450535 seat 1 (+129 in the parent): all 719 own actions unchanged.
- Original AFS R2 2029247155 seat 0 (+35 in the parent): all 719 own actions unchanged.
- The other four unchanged traces include the selected market_smart and soil
  losses. This candidate does **not** demonstrate a repair of those losses.

Exact own-action preservation is not a newly measured match result. The sole
changed R2 trace has no counterfactual terminal cash or outcome here. Conditional
future supply, continuation value, demand and opponent response remain modeling
approximations. Added alternatives can still choose worse actions in new games.
Kaggle's sandbox timing/validation and the full opponent pool were not run here.

## Full denominator and acceptance

The supplied **parent r1** full64 results are retained verbatim and independently
recounted: 1,196 / 1,536 wins (77.8646%), public eleven 1,078 / 1,408,
original AFS R2 118 / 128; all 719 steps, zero errors and ties.
These are not r2 scores. The selected seven are a diagnostic subset, not a new
win-rate denominator. Different-round seed panels are not a causal A/B estimate.

**R2 new closed-loop games: 0.** Central evaluation must freeze this identity,
use its new 64-seed panel against all twelve real opponents in both seats, and
require at least **1,306 / 1,536 strict wins**. Acceptance remains unmeasured.

## Offline build and reproducible checks

Runtime needs Python 3 standard library and a compatible Linux x86-64 C++ runtime.
No network download or external Python package is required to build or test.
Run commands from the archive root:

```sh
python build.py
python build.py --unit
```

`--unit` is now implemented by the included focused tests and uses the bundled
own-visible fixtures by default. It does not recreate a full match system.
For actual-entry probes:

```sh
PYTHONDONTWRITEBYTECODE=1 python validation/tools/probe_cases.py \
  --agent "$PWD/main.py" \
  --input "$PWD/evidence/own_visible" \
  --out "$PWD/build/entry-probes"
```

For exact parent reproduction replace the agent path by
`$PWD/provenance/parent/main.py` and add `--require-exact`.
`BUILD.md`, `policy/a06.BUILD.json`, and `PRODUCTION_AND_TEST_SHA256.json` record
actual commands, compiler, unchanged flags, hashes and build inputs.

## Audit layout and development history

`validation/` contains raw stdout/stderr, command/exit/time/resource receipts,
source diffs, parent diagnostics, final tests, and executable analysis tools.
`evidence/own_visible/` contains the supplied seven traces/ledgers, original full64
rows/metrics, identities and unchanged official rules. The original MANIFEST
maps `agent/` to `provenance/parent/`, as explained in its SCOPE_NOTE.

`development/broad_matched_deferral/` is a **retained, unselected** earlier form of
the same change, with full source and matching native. It unnecessarily widened
already two-sided choices and changed both narrow-win prefixes. The final gate
was narrowed for that structural reason, not by seed or opponent. Its raw results
are not pooled with the final results. Outer-tool orchestration failures and their
independent successful reruns are preserved in tooling_incidents.json.

No old larger replay bundle or outside opponent source was opened. The complete
supplied 64 development seeds remain in the evidence; no new match seeds were
sampled. `provenance/parent/` is the immutable parent, not another candidate.

## Frozen runtime identity

Native SHA256: `b10d86c102f68e5863559e6c049369571cbc123aa4db71db908a10c1ab14eab1`

Root main SHA256: `694e92607a1b96cff6488f89187234ff6900b7ab72d77a2e4abfb20518f682e7`

The unpacked rebuild matched this native byte-for-byte and the unpacked root
reproduced all 5,033 selected-candidate probe actions. Full delivery checks are
in validation/DELIVERY_CHECKS.json. Resource/timing records are in
RESOURCE_AND_TIMING.md and validation/RESOURCE_AND_TIMING.json.
