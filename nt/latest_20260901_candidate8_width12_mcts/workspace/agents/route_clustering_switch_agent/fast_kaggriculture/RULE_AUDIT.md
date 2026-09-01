# Kaggriculture 1.32.7 rule audit

Source of truth audited on 2026-08-20:

`/root/miniforge3/envs/torch-npu/lib/python3.11/site-packages/kaggle_environments/envs/kaggriculture/kaggriculture.py`

The upstream package and this clean-room implementation are Apache-2.0.

## Interpreter ordering

For each recorded turn:

1. Process player 0 units in farmer/hand order, then player 1 units.
2. Validate PLANT atomically per crop and player before any unit action.
3. Process market order slots in lockstep.
4. Apply town demand and refresh all market prices.
5. Decay finite-life crops.
6. At the final hour, refresh plants/animals, spawn weeds, drop inventories,
   despawn hands, reset farmers, then possibly unlock one shop.
7. Advance day/hour; final reward is public cash only.

## Market semantics

- HIRE and BUY_LAND are atomic and resolved in player order for each order slot.
- SELL/BUY orders iterate one unit at a time.
- Both players receive a quote from the same pre-commit inventory, then commits
  occur in player order.
- BUY_PRODUCT quotes at inventory minus one, making an unchanged buy/sell round
  trip neutral.
- Prices refresh after each order slot, not after each unit.
- A sale at the price floor adds cash but does not add market inventory.
- Shed capacity constrains product/animal buys.

## Unit and lifecycle edge cases

- Movement onto locked land is legal; tile mutation is not.
- Shed operations precede the locked-tile guard.
- DROP discards overflow and empties the entire unit inventory.
- Seeds never occupy shed or unit inventory.
- DIG cannot remove an occupied animal structure.
- Two unwatered/unfed daily refreshes create a weed / release the animal.
- Animal care bonus is banked only when both cared and fed, and consumed only on
  a fed production day.
- Weed/shop RNG is one shared per-day CPython MT19937 stream, keyed by
  `(episode_seed * 1_000_003) ^ day`, consumed player 0 then player 1.

## Implementation map

| Official function | C++ implementation |
|---|---|
| `_apply_unit_action` | `Simulator::apply_unit` |
| `_process_market` / `_commit_unit` | `process_market` / `commit_unit` |
| `_town_consume` | `town_consume` |
| `_decay_plants` | `decay_plants` |
| `_daily_refresh_*`, `_spawn_weeds`, inventory reset | `end_of_day` |
| `market_price`, `_refresh_prices` | `market_price`, `refresh_prices` |
| `interpreter` | `Simulator::step` |

## Known boundary

Sparse `configuration.marketParams` overrides are not implemented. All default
competition market parameters are exact. Do not use the C++ simulator for a
custom-market experiment until overrides are added and differentially tested.
