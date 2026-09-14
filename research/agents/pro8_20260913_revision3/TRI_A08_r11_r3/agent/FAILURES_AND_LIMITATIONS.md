# Failures, rejected hypotheses and limits

## Genuine development failures retained

1. The first transport prototype lacked an explicit whole-fleet seed funding
proof before retiming a route containing PLANT. The added commitment assertion
failed after 16,423 checks. Its actual native also accepted a synthetic version
of the R2 observation with four seeds but five planned plant operations. This
was a real missing safety proof, not a demonstrated seed failure in the
historical game. The final source requires current seeds to cover all fleet
PLANT operations and retains the real five-seed/five-plant positive case.
See `development/checkpoints/prototype_v1/`,
`validation/raw/seed_guard_prefixed_failure.json` and
`validation/raw/seed_budget_missing_proof_prototype.json`.

2. A late review found a test-driver defect: the driver rechecked simultaneous
PLANT demand after the first unit consumed a seed. Two available seeds and two
requests could incorrectly execute only one. The official interpreter freezes
blocked crops before executing any unit. The old driver, a failing minimal
reproduction and old outputs are retained in
`development/checkpoints/harness_atomic_before_fix/`. Both current drivers now
freeze admission before application. Ten independent 2/3-worker, 0–4-seed
cases check both atomic acceptance and debits. All official own-prefix checks
and all 24 conditional branches were rerun. Production source and native did
not change for this harness repair. Only `atomic_checks_*`, `atomic_branches*`
and the portable reruns are current evidence for official execution; earlier
`seedguard_checks_*` official subtests and earlier branch runs are superseded.
The earlier actual-native animal gates, C++ properties and direct entry probes
are unaffected by this Python driver defect.

3. The initial parent-build wrapper and first prototype full-probe wrapper did
not retain a complete outer GNU-time receipt after the tool supervisor ended.
The compiler/agent processes did finish and have their separate output,
receipts and identity checks, but their incomplete outer RSS/timing files are
not counted as complete timed measurements. Later timed builds and probes are
used for resource claims. Raw partial logs remain in `validation/raw/`.

## Deliberate negative controls (passing tests of rejection)

The independent actual-native gate rejects an injected feed/value mismatch.
A separately injected build-gate failure leaves the prior native and receipt
unchanged and retains the rejected staging library. These are expected failures,
not evidence that the final production library failed its gate. Transport
controls reject unsupported inputs, missed expiry, insufficient seeds, full
warehouse, unprofitable floor-price detours and non-current authorization.

## Rejected or incomplete causal claims

Some initial mixed-cargo/depot cases were false positives: the parent already
unloads safe surplus through `early_deposit_action`. Their exploratory route
outputs are retained, but they do not justify the final change. A positive
example was instead established for a remaining-route deadline rejection and
an available receiver; the old midroute flag remains unchanged.

No-clearance own-prefix tests show no cash gain and no reduction in total night
loss. Early warehouse placement alone is not recovered return. In the +118
case, declared pressure increases candidate loss from 3 to 4 even though its
conditional within-day cash is higher. This negative is reported, not filtered.

The limited group transfer will miss other good routes. Conservative input and
capacity checks can reject feasible transfers. The inherited auto-deposit
forecast is not a complete future-yield forecast. The current quote cannot
know future rival supply or market sale order. Single-day effects may change
next-day production, labor allocation or competitive prices. No guarantee is
made about all historical lost units, the old cash margins, or a win-rate gain.

GCC 13.3 is not available locally. GCC 14.2 and Clang 17 -O3, matched-native
checks and separate sanitizers passed, but this is not GCC 13.3 validation.
Sanitizers cover the focused C++ transport property program, not an entire
719-step match. The small conditional branch host is not a Kaggle sandbox or
a full interpreter tournament. Final full-panel acceptance remains untested.
