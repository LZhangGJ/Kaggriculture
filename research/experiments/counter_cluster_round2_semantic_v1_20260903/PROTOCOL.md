# Counter-cluster Round 2: semantic Candidate8 retest

Frozen on 2026-09-03 before inspecting any Round2 game outcome. Round1 files
are read-only inputs and are not overwritten.

## Question

With v3 G003 fixed through step 143, do the 48 already-selected Candidate8
plans remain useful on new states when stored as semantic `PlanDelta` intents
and executed by the live Candidate8 planner? Can one intent sequence cover more
than one opponent route under the frozen counter-cluster rule?

## Only changed variable

- Round1 treatment: replay a materialized 719-action tape.
- Round2 treatment: at days 6, 9, 12, 18, and 24, match the stored semantic
  intent against candidates feasible in the current state; apply the freshly
  normalized match, then let Candidate8 choose concrete actions and replan from
  the live board, inventory, market, and shop state.
- Candidate rank is used only in the original discovery state to recover the
  selected intent and audit exact discovery reproduction. A fresh game never
  reuses a rank or a fixed action tape.
- A semantic identity contains only `family_id`, `target_delta[8]`,
  `effective_delay_days`, `schedule_profile`, `market_profile`,
  `recovery_profile`, `suffix_project`, `market_item`, and `recovery_issue`.
  Capacity-derived deltas, estimates, and the 64-bit signature are diagnostics,
  not identity.
- Exact-intent lookup probes a copy of planner state. If the intent is
  unavailable or infeasible, discard the probe, log `matched=false`, and let
  the ordinary planner act from the untouched state without forcing a
  checkpoint replan. Never substitute a nearby candidate or ordinal rank.

Everything else is held fixed: panel and splits, opening and prefix, genome,
48 Round1 search choices, discovery/train/validation seeds, seats, search
settings, coverage thresholds, and portfolio cap.

## Frozen design

- Panel: 24 routes, split 12 train / 6 validation / 6 locked holdout.
- Opening: `103928643:1`, canonical action SHA-256
  `fcd500a8aaef48f001c58e85902194b37c35cfe5062f1451416141c0a7d1f28f`.
- Prefix: actions 0--143; semantic execution starts at step 144 (day 6).
- Search choices: all 48 rows in the locked Round1 `search_results.json`; no
  new search and no residual-path search in this retest.
- Discovery: seeds 9106001--9106002, both seats.
- Train: seeds 9107001--9107008, both seats.
- Validation: seeds 9108001--9108008, both seats.
- Holdout: seeds 9109001--9109016, both seats. These seeds and holdout action
  tapes are locked and must not be loaded or evaluated in this retest.

A fist covers a route only if score is at least 0.625, score uplift over the
fixed opening is at least 0.125, and mean reward margin is positive. Greedy set
cover may select at most 12 fists. Similarity alone cannot merge fists or
routes.

## Gates and interpretation

1. In each original discovery game, rank execution and recovered semantic
   execution must match every stage, terminal rewards, and full action trace.
2. Fresh-seed reports must include exact-intent match rate as well as outcome;
   an unmatched stage is not evidence that the stored fist was executed.
3. Build the response clusters only from the train matrix. If no train fist
   covers any route, stop before validation.
4. Round1 validation outcomes have already been viewed. Therefore Round2
   validation is explicitly a paired comparison, not untouched confirmation.
5. The holdout remains the only locked confirmatory layer and requires a later,
   separately authorized run after its representative fist set is frozen.

The machine-readable authority is `config.json`; hashes of inherited inputs
are in `inputs.lock.json`. A hash of the Round2 semantic executor must be
recorded before the first simulation.
