# TRI_A06_r6_r1: a bounded correction to competitive supply valuation

## The single production change

Only `policy/planner.hpp`, inside `competitive::Planner::trade`, changes. When a
positive conditional sale flow extends past the product's price-floor threshold,
use the already-existing `ConditionalMarket::execute` for that flow. All other
flows keep the parent's numerical formula. This is not a general switch to
`R2_MARKET_INTEGRAL=2`; the production macro stays **0**. Purchases, ordinary
non-crossing sales, settings, root entry, observation codec and executor remain
byte-identical to the supplied A06_r6.

The old approximate path computed a bulk average quote and then did
`if (q < 0 || p > 1) inv += q`. A quote above 1 did not mean every unit added
market supply. Conversely, an average quote rounded down to 1 did not mean that
no earlier unit added supply. The frozen official engine's `_commit_unit` pays
cash for every successful sale, but adds inventory only when the quoted unit
price is greater than 1. The positive crossing branch now respects both cash
and counted inventory. Starting inventory already above the threshold is not
artificially reduced; subsequent floor-priced units simply add no supply.

## Why this affects resources and maintenance

The `Planner::value` objective is used for whole-lifecycle asset comparisons and
for the market price curves fed to maintenance planning. Its own/rival flow
ordering is also used by the existing service marginal calculation. Fictitious
floor inventory can therefore distort the future recovery price assigned to
outputs from paid animals, crop maintenance and new projects. The fix does not
turn that forecast into spendable cash: actual procurement, funded commitments,
feed/fertilizer reservation, route scheduling, resource exchange, market order
execution and observed receipts continue through the original machinery.

The parent still coordinates positive-return investments, rolls the current day
through its bounded execution candidate search, reconciles maintenance, and uses
the original A06 fleet/execution mechanism. Its two bounded execution candidate
modes and original public-supply sale-risk certificate remain enabled. No module
was removed, no portfolio module was replaced, and no cash-target objective was
introduced. The production competition weight remains 2.

## Observed support, without counterfactual revenue claims

All seven supplied historical traces were independently reproduced using the
parent and frozen official engine: 719 matching parent actions and transitions,
plus 60 reconciled player-days, per trace. All 480 historical result rows were
read, not just the selected losses. Historical overall/public/R2 results remain
374/480, 339/440 and 35/40; none is a revised-agent result.

At step 458 in both the soil and aurax-shop worst-case histories, an own wool
sale started at inventory 10056 and successfully sold 10 units. The official
one-sided conditional batch pays 41 and ends at inventory 10059. The old
primitive estimates 37.7777777778 and ends at 10066, counting seven nonexistent
units of additional supply. These primitive calculations are not an assertion
that 3.2222222222 cash could be recovered by a different policy. The actual
historical transaction already paid the official amount.

The Thomas history also exposes the boundary: at step 530, strawberry inventory
10045 and a 21-unit sale correspond to conditional cash 315 and final inventory
10062, versus the old estimate 296.3333333333 and inventory 10066. The narrow-win
and R2 histories contain analogous boundaries. Full per-case observations,
actual successful unit counts, simultaneous-order actual revenue, and separate
one-sided conditional comparisons are in `diagnostics/market_floor_evidence.json`.
They are explicitly separated because the official market quotes both players
in per-unit lockstep; a one-sided primitive does not certify an actual rival's
future order or all simultaneous-order outcomes.

The initial bought-then-sold wheat observation was not promoted to a causal
profit claim. The original candidate search deliberately includes a liquid
variant with a feed buffer. Proving that every such unit is waste, or that a
failed order would produce recoverable profit, would require a separate
resource/time-window counterfactual. No feed-buffer toggle or extra maintenance
module was changed in this revision.

## Evidence boundaries

- The first differing revised action terminates each comparison with a saved
  historical replay. No saved suffix is used to calculate a revised final score.
- New games are a small, predeclared, paired-seat, fresh-seed head-to-head versus
  the **supplied A06_r6 parent**. Both agents react to each actual new observation.
  These are neither PASS games nor the requested public/R2 competition pool.
- No executable instance of the eleven named public entries or true original
  AFS R2 was obtained and run in this turn. Their revised win rates and protection
  of the historical R2 advantage remain unverified, pending central evaluation.
- The current 1536-game mixed development panel is not a sealed holdout. It was
  not run here, and its pending outcome is not supplied or inferred.

## Remaining modeling limitations

The existing conditional daily demand and early/late rival-supply assumptions
remain approximations. The fix does not infer private rival inventories,
future shops, future random scripts, or exact rival intent. Non-crossing bulk
cash quotes retain the parent's approximation deliberately. No claim is made
that a correct market boundary by itself establishes >85% overall win rate.
