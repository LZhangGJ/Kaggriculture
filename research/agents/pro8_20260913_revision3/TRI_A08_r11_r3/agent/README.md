# TRI_A08_r11_r3

**Frozen candidate for central evaluation, not an accepted >85% agent.** Exact
parent: `TRI_A08_r11_r2_fix1`, native
`91828b8a9f6882030ba5247a8e40f15d788a9219fe64667502ebd6a5d211beed`.
The 46-file parent production/build input set is copied byte-for-byte to
`parent/`, with a separate `PARENT_SHA256SUMS.txt`. The root production entry is
`main.py:agent(observation, configuration)`; its only default native is
`policy/tri_a08_r11_r3.so`. Nothing under `parent/`, `development/`, `evidence/`
or `validation/` is used by the playing agent.

## One structural repair: deliver by sharing a remaining field job

The parent already has midroute delivery and partial unloading at a warehouse.
Turning those switches on again would not repair the observed losses. Its
single-worker test requires the cargo worker to complete its entire remaining
route after the warehouse detour, including the parent's extra spare tick.
This can reject useful delivery even when a different worker has enough slack
to finish a resource-free field job without losing production.

The strongest concrete example is the R2 loss observation at zero-based step
613, day 25, hour 13. Farmer unit 0 is at cell 32 with two strawberries and six
melons. Its remaining route has 10 actions; a direct depot-44 detour retaining
all those actions needs 14. Worker 4 has a 7-action route. Moving the whole
WATER/HARVEST group for cell 20 to worker 4 yields routes of 10 and 11 actions,
respectively. No action is deleted. The farmer retains its HARVEST/PLANT/WATER
group at cell 23; all five currently planned wheat seeds are actually funded
by the five seeds in the observation. This changes the real entry's installed
plans at step 613, and its first emitted action difference is step 614.
A bounded own-state execution reaches DROP at step 616 and deposits all eight
units before the nightly capacity bottleneck. It is not evidence that all 23
units lost in that historical game, or its cash deficit, are recoverable.

The new code is in `policy/executor/delivery_handoff.hpp`. It runs after the
existing dispatch logic only if that logic installed no new midroute delivery.
It is authorized for the current real observation, not imagined future MPC
states. It transfers at most one complete, contiguous same-plot group composed
of WATER, HARVEST, CARE or COLLECT_FERTILIZER. Only the donor and receiver plans
are rewritten. Existing field-task order across the fleet, all tasks and
harvest expiries must be preserved, and all remaining work must still fit
before the day ends.

## Resource and value boundaries

The donor is rejected if it holds an animal or has remaining DROP, PICKUP,
FEED, FERTILIZE or PLACE obligations. Planned seed use for the *whole fleet*
must be covered by current private seeds; future purchases are not credited.
A funded PLANT operation may remain on the donor's route, but is not moved.
The receiver's prefix through its last PICKUP is unchanged, and a receiver
with a pre-existing DROP is not used. Thus this revision does not strip away
feeding/fertilizer inputs or delete positive production work to improve a
warehouse statistic.

Warehouse admission uses a conservative bound on current stock and other
workers' visible cargo/harvests that could reach a depot before the new DROP.
It assumes no intervening sale succeeds. The old runtime DROP capacity guard
also remains. A limited finite-crop watering increment is included only in
this safety bound. The older `expected_auto_deposit` forecast itself still
omits some watering increments; that broader forecast was not rewritten.

The opportunity filter quotes only current sale products, caps the quote by
estimated overflow, and charges positive extra-action cost. It excludes wheat
and fertilizer as sale proceeds. This is a conditional ranking, not a promise
of cash or a prediction of victory. The unchanged market DP must actually
observe deposited stock and choose the SELL orders. No market sale, no
recovered income: that negative control is explicitly tested.

The animal collection-labor correction, action/value-safe animal DP, floor-price
sale DP, positive-return investment logic, rolling scheduler, original input
recovery and fixed `policy/config.json` are retained. Of 44 original production
source/config/build files, 39 are byte-identical; five changed and two source
files were added. The mandatory `tests/native_choice_gate.py` is unchanged.
`SOURCE_DIFF.json` and `SOURCE_DIFF.patch` enumerate the exact changes.

## Run and reproduce

```sh
python3 -B build.py
python3 -B tests/run_checks.py --sanitize
```

The default checks validate the actual native's animal choice/value consistency,
the actual native's delivery proposals against official own-unit functions,
and transport property cases at the production -O3 flags. `--sanitize` adds a
separate ASan/UBSan C++ test; it is not substituted for the -O3 run. See
`BUILD.md` for measured commands and cross-compiler limitations.

Saved-observation entry check (zero new games):

```sh
python3 -B tests/probe_observations.py \
  --agent "$PWD/main.py" \
  --trace "$PWD/evidence/feedback/own_traces/submission_56149565_1782098909_seat1.json.gz" \
  --limit 719 --out /tmp/a08_r3_entry.json
```

The output file must not already exist. Use `--require-exact` only for the
parent reproduction or a deliberately identical ablation, not for a changed
candidate. After the first action divergence, saved future observations are
not the candidate's trajectory.

Optional **bounded own-state** execution checks, not a competition runner:

```sh
python3 -B tests/own_branch_checks.py --out-dir /tmp/a08_r3_own_branch
python3 -B tests/own_branch_checks.py --no-clearance --out-dir /tmp/a08_r3_no_sales
python3 -B tests/own_branch_checks.py --supply 3 --out-dir /tmp/a08_r3_pressure
```

Each starts from one supplied own observation and initializes the policy using
only the earlier matching prefix. Later own observations are generated from
public rules, not replayed from the saved future. The rival's public farm is
held static and no actual rival private data or action sequence is used.
The pressure condition injects a declared synthetic inventory shock of three
units per tick in each of STRAWBERRY, MELON, MILK and WOOL; it is NOT an inferred
or fully feasible rival strategy. Unknown next-day weeds/shop draws and worker
reset are not simulated, and the agent is not called on that next day. Only
within-day cash, task execution, inventory and pre-random endpoint comparisons
are supported by these checks. They cannot establish a win or terminal return.

## Evidence and unfinished acceptance

`VALIDATION_REPORT.md` gives the evidence hierarchy and exact positive/negative
results. `validation/raw/` contains successes, rejected cases, raw action
records, generated branch records, commands, stdout, stderr and timing.
Development checkpoints and superseded test outputs are retained, not erased.
The round budget remains 2026-09-13 19:00:45–21:00:45 UTC; resource and timing
records distinguish cgroup limits from host-visible figures.

The input panel is explicitly incomplete: 902 completed games, 649 wins,
253 non-wins, from a planned 1,536. Even winning every remaining game would
reach only 1,283, below 1,306. This is parent progress, not a final panel
win rate or a candidate result. All supplied rows remain in the denominator,
all 38 seed values present in those partial rows are recorded, and no new
competitive game seeds were drawn here. The shared representative/stress
pool is development data, not a sealed holdout.

**New competitive games in this round: 0.** The +289 saved trace remains action
identical; the +118 trace changes from step 508 and has a negative inventory
result in the declared pressure condition. Neither close win is claimed to be
protected in a new match. The central new-64-seed, 12-opponent, dual-seat panel
must still determine whether the candidate obtains at least 1,306 strict wins.
