# Agent-native route-level training

This training path treats existing Kaggriculture agents as complete route
executors. It does not require JAX, and it does not ask PPO to learn farmer
movement, hiring, inventory legality, and market ordering simultaneously.

## What is trained

The model learns:

```text
Q(public decision state, parameterized plan)
  -> probability that the route completes
  -> probability of winning
  -> expected terminal margin
  -> expected terminal cash
```

The target priority is lexicographic:

```text
completion > win > margin > own terminal cash
```

Low-level execution remains inside each candidate agent. A route may therefore
use any implementation: a fixed EBA action stream, a task scheduler, a search
agent, a PyTorch policy, or a hybrid rule agent.

## Why the common prefix is not repeated

For each `(seed, seat, opponent, decision_step)` scenario:

1. the prefix agent runs once;
2. every candidate and an independent opponent copy observe the same prefix so
   stateful agents are warmed correctly;
3. the fast official local environment is copied at the decision point;
4. each candidate continues from that identical state to the end of the season.

The branch uses `FastKaggricultureEnv`, not the slow full Kaggle framework. This
keeps arbitrary Python agents compatible. The Triton tensor engine remains
available for future fully tensorized route executors, but it is not a training
dependency.

## Route configuration

Copy `configs/route_training.example.json` and replace the built-in smoke agents
with file paths or `module:callable` specifications:

```json
{
  "prefix_agent": "agents/champion.py",
  "opponents": [
    "agents/public_g02.py",
    "agents/public_g04.py",
    "agents/market_exploiter.py"
  ],
  "decision_steps": [120, 168, 240, 360, 600],
  "routes": [
    {
      "name": "10c4s75l",
      "agent": "agents/route_10c4s75l.py",
      "plan": {
        "cow_target": 10,
        "sheep_target": 4,
        "land_target": 75,
        "max_hands": 5,
        "cash_reserve": 1200,
        "first_hire_step": 48,
        "sale_offset": 0,
        "liquidation_step": 672
      }
    }
  ]
}
```

File-based agents are imported under unique module names, so module-level state
from one route cannot contaminate another. Agents should still reset their own
episode state when `observation.step == 0`.

## Generate counterfactual data

```bash
PYTHONPATH=src python scripts/generate_route_dataset.py \
  --config configs/my_routes.json \
  --output artifacts/route_counterfactuals.jsonl \
  --seed-start 10000 \
  --num-seeds 128 \
  --workers 8
```

Both seats are evaluated by default. Use `--decision-step` to override the JSON
list and `--resume` to append only missing `(scenario, route)` rows.

Every route in a scenario has the same `context_hash`. Dataset validation fails
if a supposedly paired group contains different decision states.

## Train the selector

```bash
PYTHONPATH=src python scripts/train_route_selector.py \
  --dataset artifacts/route_counterfactuals.jsonl \
  --output artifacts/route_selector.pt \
  --epochs 200 \
  --batch-scenarios 32 \
  --device cuda
```

The split is made by `scenario_id`, not by individual rows, preventing routes
from the same seed/state from leaking across training and validation. The main
selection metric is top-1 route accuracy inside each held-out counterfactual
group.

## Farmer, hands, and hiring

`route_planner.py` supplies a small task-DAG scheduler for route agents. Tasks
have positions, release times, deadlines, dependencies, values, and allowed
worker roles. The beam scheduler accounts for Manhattan travel and worker
availability. `evaluate_one_hire` compares the best schedule with and without
one additional hand after subtracting the official Fibonacci hire cost.

This is intentionally a rolling 24--96 step scheduler. It should be called again
when a shop unlocks, a weed appears, a maintenance deadline is threatened, or a
high-level route switch is accepted. It is not intended to enumerate every
movement sequence for all 719 turns.

## Recommended experiment order

1. Freeze the current champion as the prefix and fallback agent.
2. Add a small set of materially different complete agents: 10C4S, 8C6S,
   6C12S, 11C4S, high-wool, plant, and conservative liquidation.
3. Collect paired data against the weakest full-pool matchups first.
4. Train the selector and require held-out top-1 improvement plus official
   double-seat validation.
5. Only after route selection is reliable, allow the model to alter hiring,
   sale timing, or liquidation as separate plan features.
