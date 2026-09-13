# Evidence-led decision record

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
