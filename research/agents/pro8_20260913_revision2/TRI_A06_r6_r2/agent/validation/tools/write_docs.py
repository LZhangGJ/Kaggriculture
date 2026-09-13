from pathlib import Path
import datetime,hashlib,json,shlex
w=Path('/mnt/data/a06_r6_r2_work');r=w/'delivery'
ident=json.load(open(r/'IDENTITY.json'));build=json.load(open(r/'policy/a06.BUILD.json'));unit=json.load(open(w/'analysis/narrow_choice_units/SUMMARY.json'));probes=json.load(open(w/'analysis/narrow_root_probes/SUMMARY.json'));preserve=json.load(open(w/'analysis/narrow_preservation.json'))
md='''# TRI_A06_r6_r2

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
PYTHONDONTWRITEBYTECODE=1 python validation/tools/probe_cases.py \\
  --agent "$PWD/main.py" \\
  --input "$PWD/evidence/own_visible" \\
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
'''
(r/'README.md').write_text(md)
cmd=shlex.join(build['command'])
source_manifest=json.load(open(r/'PRODUCTION_AND_TEST_SHA256.json'))
flags=json.load(open(r/'COMPILER_FLAGS.json'))
buildmd=f'''# BUILD — TRI_A06_r6_r2

## Selected binary and ancestry

- Selected native: `{ident['native_sha256']}`
- Parent native: `{ident['parent_native_sha256']}`
- Input own-visible ZIP: `{ident['input_zip_sha256']}`
- Root main.py: `{ident['main_sha256']}`
- Changed source policy/search.hpp: `{source_manifest['policy/search.hpp']}`
- Compiler flags file: `{ident['compiler_flags_sha256']}`
- Fixed runtime config: `{ident['config_sha256']}`

The 45 inherited files comprise 43 unchanged files, one changed production header,
and the rebuilt native. Full hashes are in PRODUCTION_AND_TEST_SHA256.json;
all 40 policy source/config inputs hashed by the build are also recorded in
policy/a06.BUILD.json. Other new files are tests, evidence and documentation.

## Actual selected compilation

Compiler: `{build['compiler']}`. Architecture target: generic Linux x86-64.
Actual command executed successfully in this container:

```sh
{cmd}
```

The production build subprocess recorded {build['seconds']:.6f} seconds. The
bounded supervisor's wall/RSS and successful exit are in
`validation/logs/narrow_compile.{{receipt.json,time,exit}}`.
The flags were inherited without edits:

```text
{' '.join(flags)}
```

No fast-math change, host-specific -march=native, or runtime compilation is used.
The root offline build entry is `python build.py`; `build_a06.py` is retained as
provided. A compiler failure is surfaced rather than replaced with a stale native.
The optional `--unit` now runs the included tests; the original input omitted them.

## Parent reproducibility

`validation/parent_rebuild.so` was compiled before any production edit and matched
the supplied parent byte-for-byte. Its actual command, all policy input hashes
and compiler are in `validation/parent_rebuild.BUILD.json`; bounded run details
are `validation/logs/parent_compile.*`.

`provenance/PARENT_FILES_SHA256.json` pins all original 45 files. The source/native
copy is read-only and hash-verified. An initial Python import generated bytecode
despite root-level read-only permissions; that disposable cache was removed.
No original source, configuration or native bytes were modified. See
`validation/analysis/parent_integrity.json` for the exact cleanup record.

## Test builds

The actual-production-class test library includes bridge.cpp from the selected
source, then adds test-only entry points; it is not the deployed policy library.
Actual commands/compiler/native SHA are in
`validation/analysis/narrow_choice_units/compile.receipt.json`, with raw stdout,
stderr and exit records. Tests invoke explicit throwing checks, so -DNDEBUG
cannot silently disable validation. The official interpreter copy is unchanged.

The unpacked archive's default build and `--unit` are re-executed separately.
Their command receipts, byte-for-byte comparison and root-entry checks are listed
in `validation/DELIVERY_CHECKS.json` after completion. No build or probe result
is interpreted as a real-opponent win-rate result.
'''
(r/'BUILD.md').write_text(buildmd)
(r/'ANALYSIS.md').write_text('''# Evidence-led decision record

## Rejected explanations

The seven ledgers all show 7,000 land expenditure, but that alone says nothing
about marginal profitability. The original unused/orphaned-land explanation is
not supported: all 21 purchases were used on the same day; actual later crops,
animal production and maintenance are recorded in trace_land_evidence. All 210
own-player days balance cash exactly. No failed-market-order refund was invented.

For example, at R2 1623731725 seat 1 day 18, the initial fourth-quadrant plan has
four tomato targets (positions 55, 56, 57, 65). Its later recorded use spans more
plantings and commodities. Removing only those initial paths and refunding land
in the model produces a negative initial-bundle contribution, but omits later
rolling uses and changes in opponent supply. That is a diagnostic clue, not
proof of an unprofitable land purchase and not an estimate of recoverable cash.

## Confirmed source-level gap

In policy/triad.hpp, Controller::plan reserves and charges a next quadrant when
current slots are exhausted (the next_land_cost / planned_land block), before
finishing the asset choices and the preview-admission pruning. The original
SearchController only compared the generated expansion-capable portfolio
variants. At the R2 day-18 state all six original scored candidates included the
land transaction: no wait alternative was available to price its opportunity
cost through the same executable mechanism.

The final search.hpp change adds that alternative only when all original queues
share the land transaction. Waiting is therefore a matched feasible intervention,
not an additional cash credit, theoretical production sold before delivery, or a
permanent maximum-land reduction. It replans from the incoming paid commitment
book, caps both the real controller and executor, and restores eligibility next
day via the existing restoration code. The common continuation can still choose
to expand later; only today's intervention differs.

## Why the broad development form was not selected

The broad form added matched waiting branches even when other original candidates
already offered not buying land. It changed four saved histories, including both
narrow wins. There were no measured new losses, but broadening already two-sided
choices was unnecessary to fix the demonstrated missing-choice problem. The
final all-originals-BUY_LAND gate is a current legal action-set property, not a
label/seed/timing table. Its complete source and native are retained for audit.

The final saved-observation behavior is much narrower: only the demonstrated
R2 day-18 state is the first differing action; the other six histories, including
both narrow wins, reproduce all 719 parent actions. The conditional selector
score at that first divergence favors candidate 39 (land cap 3) by approximately
3,355.12 over the best old candidate. That is a model-score difference, NOT
actual cash saved, terminal profit, a reversed loss, or a new win.

## Source and information boundary

Only search.hpp changes at runtime. No probe, full64 row, seed, opponent label,
ledger or trace is loaded by main.py. Legal observations pass the same codec and
contain current public farms/market/shops and only the acting player's private
state. PublicFlowScenario's deterministic assumed arrivals and expected demand
are inherited, not enemy private inventory or actual future scripts. The
unchanged simulator may initialize a synthetic RNG internally; synthetic future
shop draws are discarded by PublicFlowScenario and are not treated as observed
future events. The separate official first-prefix tests do not advance a clock.

## Remaining uncertainty

A missing feasible alternative is demonstrably repaired; win-rate improvement is
not established. The parent's one-day horizon, continuation approximation and
public-flow model can misrank alternatives. Exact narrow-win action preservation
is a regression property on the supplied own histories, not independent new-game
validation. Market_smart and soil selected losses remain action-identical and are
not presented as fixed. Central new-seed 1,536-game evaluation is required.
''')
print('Wrote English README, BUILD and ANALYSIS')
