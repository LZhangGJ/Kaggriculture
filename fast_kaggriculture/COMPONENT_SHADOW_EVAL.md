# G001 component planner shadow evaluation

`component_shadow_eval` is an offline, read-only owner-coverage audit. It feeds
the unmodified native G001 action into `Simulator`, maps that action and the
pre-action observation to conservative component-planner certificates, and
runs `compile_component_scoped_route` on a shadow manifest. The shadow manifest
is never submitted to the simulator.

The report therefore contains no reward, money, margin, or promotion claim.
`applies_actions` and `reward_metrics_present` are both hard-coded `false`.

## Certificate boundary

Every native unit operation has a mapper branch and a simulator-backed fixture.
Legal MOVE sources receive an exact source-id/from/to certificate. Crop tile
effects receive exact pre/post tile certificates when the actor-only preview
and the complete unit manifest prove the same causal result. Invalid crop
actions may create an observation-driven crop objective. No-effect actions are
omitted from the raw effect queue.

The mapper now uses the independently differential-tested
`effectCertificateSemanticsCpp` predictor for DROP, PICKUP, PLACE, FERTILIZE,
FEED, COLLECT_FERTILIZER, and HARVEST. It records exact before/after quantities
for actor inventory, shed, and every changed tile field. Every certificate is
bound to the lower-actor-slot prefix by a nonzero state hash and day-scoped
actor generation. Shared typed cells produce dependency keys. Concurrent
PLANT whose result depends on manifest-wide seed validation remains opaque
rather than falsely attributed to one actor.

The source-conservation component mode still seals actors containing a typed
native operation that its crop scheduler cannot execute. The typed sidecar is
not permission to reinterpret that byte. `prefix_authority_failures` is a hard
shadow gate; a native receipt path would additionally need to retain and match
the prefix token, and is not implemented here.

The board certificate is constructed from the actual observed source-id MOVE
transition at every tick. Missing, mismatched, blocked, or out-of-bounds MOVE
proof fails the component planner's global merge gate.

## Fixed panel

The bounded panel is intentionally descriptive:

- one seat-0 game with seed `25772138000` and weed rate `1.0`;
- seeds `25772138701..25772138732`, both seats, weed rate `0.005`;
- the same raw G001 baseline is executed in every game;
- objectives are coalesced within a day and never applied to later observations.

Run it with:

```bash
cmake --build build-shadow-release --target component_shadow_mapper_tests component_shadow_eval -j2
./build-shadow-release/component_shadow_mapper_tests
./build-shadow-release/component_shadow_eval \
  --output build-shadow-release/component-shadow-full.json
```

The original source-conservation mode produced zero safe assignments even
after typed certificates reduced global opacity. It remains a negative owner
coverage result.

The report also contains a separate `exact_move_slot` crop-only shadow mode.
It fixes every original MOVE at the same actor/day/hour with the exact payload
and recomputes the position timeline. Known crop actions become plot-owned
intents; each stationary service slot takes at most one state-driven step.
Unfinished intents cross visits and days. Rejected proposals preserve the raw
slot transactionally, and any same-tile unknown/prefix-bound raw effect seals
that tile/turn. A legal ongoing tomato/strawberry HARVEST is preserved as raw
and does not create a replant debt; the planner does not speculate about its
post-state. Animal lifecycle scheduling is not implemented, so the exact-slot
result is not a native readiness claim.

`deviation_only` is the native-candidate-shaped exact-slot arm. It preserves
every unconfirmed non-MOVE byte and opens a plot intent only when the exact
predictor proves the current crop source has the wrong/no effect. The full
recompile arm remains an oracle/coverage comparison only.

`event_local_elastic_day_*` is a second, non-executable scenario shadow. Its
input begins at an already observed trigger (`trigger_source[0] == true`), so
it cannot consume or advance a pre-trigger prefix. MOVE sources are rolling
ordered tokens. A MOVE may be delayed only into certified PASS/no-effect sinks
when the remaining horizon can discharge every MOVE and hard raw token. The
reported capacity proof is recomputed from terminal token counts and the MOVE
source/payload validators; it is not an input assertion.

The elastic seed vector is explicitly a scenario receipt trace. It may contain
future values only for the offline observed-trace/upper-bound comparison and
is therefore marked `scenario_shadow_only` and `future_receipt_trace` in JSON.
A future native owner would have to invoke the compiler each tick with only the
current receipt; it must not preview later fills. Trigger windows in this
evaluator are independent and may overlap, so their totals are diagnostics,
not a globally executable manifest.

## Report interpretation

`objectives` counts newly observed, coalesced crop objective candidates.
`unscheduled_objectives` is a sum of per-segment planner decisions, so an
objective carried across several segments can appear more than once. A shadow
commit is treated as an opportunity and removed from the shadow carry; it is
not treated as a simulator receipt. Carry is cleared at the day boundary, so
the report does not claim cross-day recovery completeness.

`potential_move_delay` is computed only from a merge-safe shadow manifest and
exact raw source ids. It is not an observed action edit or reward metric.

`exact_move_slot.assignments` counts individual DIG/PLANT/WATER/HARVEST
transitions, not completed objectives. `cross_day_debt_created` counts debt
present at day boundaries, so the same long-lived debt may contribute on more
than one boundary. Since the proposed manifest is never executed,
`cross_day_debt_retired` is a counterfactual service opportunity, not an
observation receipt. The native gate requires `debt_lost`, MOVE slot/payload
changes, position failures, and prefix-authority failures all to remain zero.

The seed upper-bound arms never grant seed to an executable action. They only
separate purchase/receipt capacity from route/service-slot capacity. Neither
the fixed nor elastic shadow is submitted to `Simulator`, and neither supports
the animal FEED/CARE/yield lifecycle planner yet.
