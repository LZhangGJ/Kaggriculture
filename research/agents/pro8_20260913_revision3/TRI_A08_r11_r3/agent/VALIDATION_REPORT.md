# Validation report — TRI_A08_r11_r3

Status: **candidate for central testing; no new competitive win rate established**.
Exact shipped production native: `49c98f2c1930cf741bdd9e2eb7da9af398117bb64f903f9375c7da9350e8fa64`.
All counts below refer to the final seed-funded production source/native.
The official-unit and bounded-branch results use the corrected atomic PLANT
harness; superseded results and both genuine failures are retained separately.

## 1. Input and provenance

Both new attachments match the user-specified SHA256 hashes. All 75 manifest
entries were verified. The exact parent source/native set has 46 files;
`parent/` is a fresh, read-only copy of that input, not a copy of a mutated old
workspace. `PARENT_SHA256SUMS.txt` can verify it without rebuilding it.
The full manifest and source-package parent identity remain in
`evidence/source_metadata/`; full feedback files are in `evidence/feedback/`.

`validation/raw/input_result_audit.json` independently checks all 902 supplied
partial rows: 649 wins, 253 non-wins, no ties/errors, 719 transitions each,
with unique game keys. This remains an incomplete parent panel; the maximum
possible final wins from this snapshot is 1,283/1,536. No 902-row percentage is
reported as a completed formal rate. The four own cash ledgers reconcile all
120 available own-player days against the trace endpoints and cash flows.
The sender's 60-player-day-per-game audit includes the opponent; no new access
to that opponent's private ledger is claimed here. All 38 distinct game seeds
present in the partial rows, the four trace IDs and synthetic test scope are
preserved in `validation/raw/DEVELOPMENT_SEEDS.json`. No new game seeds were drawn.

## 2. Observed structural rejection and implemented change

Read-only instrumentation of the parent reproduced all 2,876 actions across
four complete own-observation traces. It identifies per-worker remaining DROP,
reserved-input/mixed cargo, remaining tasks and expected night deposit; it does
not change actions. The matching diagnostic sources/native are retained under
`development/parent_diagnostic/` and `development/diagnostic.so`.

R2 step 613: the parent's forecast is 103 units, shed safety is one, so its
excess indicator is four. Unit 0 has actual S=2/M=6, no reserved material job,
no remaining DROP, and 10 actions left from cell 32. A 14-action depot detour
fails the single-worker completion test. Unit 4 has seven actions left.
Transferring cell-20 WATER/HARVEST intact to unit 4 gives 10- and 11-action
routes. The donor's cell-23 HARVEST/PLANT/WATER remains executable, and five
actual wheat seeds cover all five planned fleet PLANT operations. The final
native installs this handoff at 613 and first changes an emitted action at 614.
The paired own-state branch observes the actual eight-unit DROP at 616.

See `validation/raw/audit_submission_56149565_1782098909_seat1.json.gz`,
`seedguard_final_submission_56149565_1782098909_seat1.json.gz`,
`atomic_checks_gcc/delivery_official.json`, and the current
`branch_submission_56149565_1782098909_seat1_613_candidate_clear1_supply0.json.gz`.
All numbers are zero-based steps/days and unit 0 is the farmer. The proposal
quote is not the historical item's lost quantity multiplied by a current price.

The public loss also receives a real handoff from the root entry at step 460.
Initial apparent mixed-cargo/depot opportunities were discarded as causal
explanations when the parent's existing partial-unload logic already handled
them. The retained exploratory `route_options.json` therefore contains rejected
hypotheses and is not a list of recovered opportunities.

## 3. Correctness and build checks

| Check | Final measured result | Scope |
|---|---|---|
| Required actual-native animal gate | 530 DP solves, 29,456 states, 171,628 checks; injected wrong action rejected | Choice/value consistency in the loaded production library |
| Positive rich animal case | value=65, choice.value=65, feed=1, care=0 | Original positive-maintenance behavior retained |
| GCC 14.2 and Clang 17 production -O3 transport tests | Each 8,640 states, 16,440 assertions, 864 selected proposals | Route/order/resource property matrix, not games |
| Additional GCC/Clang ASan+UBSan | Both pass focused C++ transport tests | Separate from, not a replacement for -O3 |
| Actual-native proposal + official own-unit prefixes | 249 saved snapshot calls; 63 selected proposals; 1,427 assertions | 49 synthetic and 14 snapshot proposals; all requested productive/maintenance effects retained |
| Atomic seed admission regression | 10 two/three-unit cases covering 0–4 seeds | Whole-tick blocked set fixed before execution |
| Clean independent GCC rebuild | Byte-identical final native | Actual native gate also passed |
| Injected build-gate failure | Rejected staging library retained; previous native and receipt unchanged | Deliberate expected failure in a separate copy |
| Default root `agent()` | 2,876 valid calls, zero exceptions; all match `create_agent()` | Complete supplied own-observation probes |
| Cross-compiler entry comparison | 2,876/2,876 GCC vs Clang actions identical | Same saved observations, not new trajectories |
| New feature disabled | 2,876/2,876 original-parent actions identical | Native ablation preserves original behavior |

The proposal/unit tests check route targets and deadlines, the whole-fleet
same-plot operation sequence, successful production/maintenance effects, seed
debits, item conservation and real DROP arrival/capacity. They independently
recompute quantity, extra actions, quote and score from the selected native
proposal. Eleven negative controls reject infeasible/unsupported inputs;
a positive PICKUP/FEED receiver case verifies the input pickup prefix is kept.
The no-market control deliberately confirms that early DROP alone does not
reduce total loss after night overflow.

Current raw test directories are `validation/raw/atomic_checks_gcc/` and
`validation/raw/atomic_checks_clang/`. Actual -O3 flags are the corresponding
build receipt flags, not a reduced-optimization substitute. The macro
`A08_ANIMAL_COLLECTION_LABOR=1` remains enabled. The old required animal native
gate file is unchanged. GCC 13.3 is unavailable here: central GCC 13.3 execution
remains a required independent check. Dynamic symbol requirements and ELF
architecture were inspected in `native_platform_check.json`; a successful local
load is not a substitute for the central machine's own test.

## 4. Saved-observation behavior, not a changed-policy replay

| Supplied case | Parent action matches | First emitted action difference |
|---|---:|---:|
| aurax loss (-10,695) | 699/719 | 460 |
| R2 loss (-7,910) | 709/719 | 614 |
| market_smart close win (+118) | 699/719 | 508 |
| R2 close win (+289) | 719/719 | none |

Totals are 2,826/2,876 matching parent actions. An installed plan can differ one
step before the emitted action changes. Saved observations after a divergence
are not a valid candidate future; these rows test loading, output validity,
feature activation and compiler behavior only. They establish no cash or win.
The +289 case stays identical on its saved history, while +118 changes. Neither
has a new competitive close-win-protection result.

## 5. Bounded own-state execution and negative economics

For each of four branch points, the policy is initialized only from the matching
prefix. Each later own observation is newly generated from the official unit,
market, town-consumption, decay, production-maintenance and auto-deposit
functions. The earlier harness's atomic PLANT issue was corrected and all
branches rerun. Every endpoint comparison retains matching productive tiles
and seed consumption between parent and candidate.

These are 24 bounded branches totaling 372 generated own ticks, **zero games**.
They hold the rival's public farm static, use no actual rival private state or
rival actions, and stop before any next-day policy action. They omit unknown
next-day random weeds/shops and the now-irrelevant worker reset. Thus the cash
numbers below are conditional within-day differences, not actual match cash
recovered or terminal profit estimates.

### Own market enabled; no subsequent rival order

| Case and branch start | Generated ticks each | Parent cash change | Candidate cash change | Night units lost, parent → candidate |
|---|---:|---:|---:|---:|
| aurax loss (-10,695), step 460 | 20 | 7746 | 8376 | 15 → 10 |
| R2 loss (-7,910), step 613 | 11 | 0 | 86 | 9 → 1 |
| market_smart close win (+118), step 508 | 20 | 3873 | 4193 | 3 → 0 |
| R2 close win (+289), step 613 | 11 | 396 | 396 | 0 → 0 |

### Own SELL disabled: explicit no-clearance negative

Cash gain is zero in both branches for every case. Parent/candidate night loss
is respectively 76/76, 9/9, 51/51 and 0/0. This refutes the inference that moving
items into the warehouse by itself recovers sale revenue or avoids total loss.

### Synthetic market-pressure sensitivity

A fixed, declared inventory shock adds three units per tick in each of
STRAWBERRY, MELON, MILK and WOOL before market clearing. This is not a reconstructed
opponent strategy or evidence about the true rival's future supply.

| Case | Conditional cash difference, candidate − parent | Night loss, parent → candidate |
|---|---:|---:|
| aurax loss (-10,695) | +567 | 15 → 10 |
| R2 loss (-7,910) | +8 | 9 → 1 |
| market_smart close win (+118) | +220 | 3 → 4 |
| R2 close win (+289) | +0 | 0 → 0 |

**The +118 case has a real negative within this test condition: night loss rises
from three to four.** The conditional cash increase does not prove a win and
cannot cancel this risk by assertion. These three conditions are a small
execution/sensitivity set, not a parameter search or replacement for the central
12-opponent panel. Current results are the `own_branch_summary_*` files and
`atomic_branches*` logs; the portable release-script reruns are in
`validation/raw/portable_branches*/`.

## 6. Delivery, limits and acceptance

The final source uses no case ID, seed table, opponent name, opponent inventory
or saved observation file. The only runtime input is the ordinary legal
observation plus its own internal controller state. Experimental exports are
read-only and are never invoked by the root playing entry.

The original animal labor improvement, floor-sale DP, input recovery, investment
and scheduling code remain. Remaining restrictions can miss feasible transfers;
future supply and final competitive opportunity cost remain uncertain. The
parent's incomplete panel and old round results cannot be used to claim a
causal win-rate effect across changed seed sets. No all-75/all-23-unit salvage
claim, no recovered-loss cash estimate, and no >85% claim is made.

See `FAILURES_AND_LIMITATIONS.md` for the seed-proof failure, harness failure,
partial timing wrappers, rejected hypotheses and raw evidence locations.
`validation/RESOURCE_END.json` and `validation/RESOURCE_LOG_INDEX.json` record
the actual environment and timed command resources. The release was assembled
well before the original 100-minute packaging target; the budget was not reset.
Full ZIP/extracted-root validation is recorded in the accompanying external
`TRI_A08_r11_r3_POST_UNPACK_CHECK.json` so the ZIP can remain immutable.

**Formal new competitive games: 0. Acceptance remains at least 1,306 strict
wins from 1,536 new central games; draws are not wins.**
