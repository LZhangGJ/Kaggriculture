# G001 forced-trigger worker identity gate

## Reproduced failure

The 719-action G001 tape was executed in two native simulators with the same
seed and `weed_spawn_chance=1.0`. Player 0 used either the unmodified executor
or only repair mask bit 8. This is an intentionally adversarial regression,
not formal trade data and not a reward claim.

An early version of the compiler exposed this deterministic worker-identity
failure:

- transaction 1 opens for hand actor 1 at tile `(0,2)` on day 1;
- all hands are destroyed by `Simulator::end_of_day`;
- a different hand is later hired into numeric actor slot 1;
- at step 81 (day 3, hour 9), that new hand reaches `(0,2)`;
- the old transaction is selected and an effect receipt is staged;
- steps 82 and 83 confirm effects against the old transaction.

Therefore `(actor index, tile)` is not a stable transaction key across days.
Simply retaining the current map lets a newly hired worker inherit an old
worker's overlays and source debts. Simply clearing the map would avoid that
identity collision but silently destroy real tile/lifecycle debt.

## Required ownership model

Worker ownership and semantic debt must be separated.

- A live worker has a generation-qualified identity, for example
  `(day_hired, hire_ordinal, current_actor_slot)`; the farmer has a stable
  identity of its own.
- A crop/structure obligation has a stable transaction id and tile identity.
  Purchase debt and delayed non-MOVE source effects attach to this semantic
  transaction, not to a reusable actor slot.
- When a hired worker disappears at day end, its open transaction becomes an
  unclaimed tile obligation. An hour-23 effect may be observed on the adjacent
  next tick using its original generation-qualified lease; it must be settled
  before the numeric slot can be interpreted as a new worker.
- A new worker may claim an unclaimed obligation only through an explicit
  proposal after observing the exact tile, desired item, legal prerequisite,
  and arbitration ownership. Numeric actor-slot equality is not a claim.
- The claim is staged against the exact final unit manifest. Only its next
  observation receipt transfers progress to the new worker generation.
- Old worker MOVE actions are never persisted or transferred. Only semantic
  non-MOVE source debt survives orphaning. The new worker follows its own
  current-day ordered MOVE plan.

## Strong gates

An identity repair is incomplete unless all gates below pass.

1. **Conservation:** every pre-rollover open transaction is exactly one of
   active, unclaimed, completed, or explicitly retired-with-reason after
   rollover. No debt is dropped or duplicated.
2. **No stale receipt:** a receipt contains worker generation, actor slot,
   transaction id, exact tile, and adjacent submission step. A token that was
   not committed for that generation cannot advance debt. A valid hour-23
   token may settle the effect actually emitted before teardown.
3. **No implicit reuse:** a new hand with the same numeric actor slot and tile
   cannot emit a binding for the old transaction before an exact claim commit.
4. **Safe claim:** a confirmed claim preserves transaction id, desired crop,
   tile, confirmed harvest count, and outstanding non-MOVE source debts.
5. **MOVE isolation:** orphaning/claiming transfers no MOVE. For every
   day+worker generation whose availability window is unchanged, the enabled
   ordered MOVE sequence equals the raw source sequence. Under forced weeds,
   the legacy baseline's previous-tape replay is allowed to differ and is
   reported as the overlay owner rather than treated as an enabled overwrite.
6. **Purchase continuity:** an open seed acquisition remains attached to the
   semantic tile transaction while unclaimed. Inventory-only handoff is made
   to the claiming generation; it never confirms PLANT.
7. **Boundary receipts:** hour-23 submissions are either receipt-settled with
   their original generation before orphaning, or rejected stale while debt is
   preserved. Weed creation and hand teardown cannot fabricate success.
8. **Terminal boundary:** the final bound HARVEST has a next-tick receipt or is
   reported pending; ordinary unbound HARVEST is checked by simulator unit
   effect. Final DROP is verified with `preview_unit_phase` before market sells.
9. **Actual-route stress:** the native 719-turn forced G001 report has zero
   accepted invalid lease tokens, zero uncommitted pending bindings, and zero
   enabled same-active-window raw-source MOVE mismatches. Actor availability
   differences are reported separately as economic-feasibility cascades.
   Enabled-no-trigger parity alone is insufficient.

## Comparator and current terminal audit

Both simulators must use the same `Config`, seed, market arm, and
`neutral_special_economy` value. The harness asserts the full `Config`
fingerprint and that the only repair-option difference is bit 8. Reports made
before this assertion, where only the enabled side used neutral special
economy, are invalid comparators and must not be used as gates.

The current report exports every terminal open plot transaction, including
origin provenance, source epoch, tile and desired state, outstanding source
count, current claimant, pending state, receipt counters, and the last observed
receipt result. `origin_actor_generation != current day` is not pollution: it
is expected provenance for a persistent plot debt. Pollution requires an
invalid lease token to be accepted or a pending binding to appear without an
exact commit.

Terminal debt is split into:

- **critical source debt:** at least one original stationary source obligation
  remains; both transaction and unit counts are reported;
- **recoverable critical debt:** critical debt whose terminal tile remains
  compatible with its desired crop;
- **lifecycle-controller only:** source debt is zero, but the configured crop
  lifecycle (for example a second confirmed harvest) is not complete;
- **terminal-unleased:** no current actor occupies the plot at episode end;
  this is not deletion and the transaction remains fully serialized.

The read-only harness is `src/native_g001_forced_regression.cpp`. It emits an
explicit non-formal JSON report and returns nonzero while a strong gate fails.
