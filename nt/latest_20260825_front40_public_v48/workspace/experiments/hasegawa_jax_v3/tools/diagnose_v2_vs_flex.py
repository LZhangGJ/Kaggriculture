"""Export stepwise public-economic diagnostics for V2 versus Flex."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v2 import hasegawa_step_with_external_v2, initialize_hasegawa_carry_v2, load_hasegawa_trace_bank_v2  # noqa: E402
from kaggriculture_jax.constants import MarketOp, NUM_ANIMALS, NUM_CROPS, NUM_PRODUCTS, TileKind  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
from strategic_v5.public_g02_gpu import initialize_public_g02_carry_v1, public_rc5_weed_player_action_v1  # noqa: E402


def product_yield(states, player):
    batch = states.step.shape[0]
    crop = states.tile_crop[:, player].reshape(batch, -1)
    animal = states.tile_animal[:, player].reshape(batch, -1)
    amount = states.tile_yield[:, player].reshape(batch, -1).astype(jnp.int32)
    values = []
    for product in range(NUM_PRODUCTS):
        if product < NUM_CROPS:
            values.append(jnp.sum(jnp.where(crop == product, amount, 0), axis=1))
        elif product < NUM_CROPS + NUM_ANIMALS:
            values.append(jnp.sum(jnp.where(animal == product - NUM_CROPS, amount, 0), axis=1))
        else:
            values.append(jnp.zeros((batch,), dtype=jnp.int32))
    return jnp.stack(values, axis=1)


def farm_counts(states, player):
    batch = states.step.shape[0]
    crop = states.tile_crop[:, player].reshape(batch, -1)
    animal = states.tile_animal[:, player].reshape(batch, -1)
    kind = states.tile_kind[:, player].reshape(batch, -1)
    return jnp.concatenate((
        jnp.stack([jnp.sum(animal == index, axis=1) for index in range(NUM_ANIMALS)], axis=1),
        jnp.stack([jnp.sum(crop == index, axis=1) for index in range(NUM_CROPS)], axis=1),
        jnp.sum(kind != TileKind.LOCKED, axis=1)[:, None],
        jnp.sum(states.unit_active[:, player, 1:], axis=1)[:, None],
    ), axis=1).astype(jnp.int32)


def sale_quantities(action, player):
    active = jnp.arange(action.market_op.shape[-1])[None, :] < action.market_count[:, player, None]
    op = action.market_op[:, player]
    item = action.market_item[:, player]
    amount = action.market_amount[:, player]
    return jnp.stack([
        jnp.sum(jnp.where(active & (op == MarketOp.SELL) & (item == product), jnp.maximum(amount, 0), 0), axis=1)
        for product in range(NUM_PRODUCTS)
    ], axis=1)


def make_rollout(hbank, exact_bank, tables, h_player):
    opponent = 1 - h_player

    @jax.jit
    def rollout(initial, events):
        batch = initial.step.shape[0]
        route_ids = jnp.full((batch,), 9, dtype=jnp.int32)

        def body(value, _):
            states, hcarry, flex_carry = value
            flex_action, flex_carry = public_rc5_weed_player_action_v1(
                states, exact_bank, route_ids, flex_carry, opponent
            )
            next_states, hcarry, diagnostics, joint = hasegawa_step_with_external_v2(
                states, hcarry, hbank, flex_action, h_player, events, tables
            )
            features = (
                states.money,
                states.market_price,
                states.market_inventory,
                states.shed[:, h_player, :NUM_PRODUCTS],
                states.shed[:, opponent, :NUM_PRODUCTS],
                product_yield(states, h_player),
                product_yield(states, opponent),
                farm_counts(states, h_player),
                farm_counts(states, opponent),
                sale_quantities(joint, h_player),
                sale_quantities(joint, opponent),
                diagnostics.invalid_unit_intent_count,
                diagnostics.resync_unit_count,
                diagnostics.market_trim_count,
                diagnostics.branch_id,
                states.town_count,
                states.town_shops,
            )
            return (next_states, hcarry, flex_carry), features

        hcarry = initialize_hasegawa_carry_v2(batch)
        flex_carry = initialize_public_g02_carry_v1(batch)
        (terminal, hcarry, _), scanned = jax.lax.scan(
            body, (initial, hcarry, flex_carry), None, length=719
        )
        return terminal.money, terminal.done, hcarry, scanned

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--exact-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed-start", type=int, default=154001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    hbank = load_hasegawa_trace_bank_v2(args.hasegawa_bank)
    exact_bank = load_bank(args.exact_bank)
    _ = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    base_seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    detail = {}
    rows = []
    for seat in (0, 1):
        weed, shops = build_events_v1(base_seeds.tolist())
        events = Events(jnp.asarray(weed), jnp.asarray(shops))
        initial = jax.vmap(reset)(jnp.asarray(base_seeds))
        result = make_rollout(hbank, exact_bank, tables, seat)(initial, events)
        jax.block_until_ready(result[0])
        money, done, carry, scanned = jax.device_get(result)
        names = (
            "money", "market_price", "market_inventory", "own_shed", "opponent_shed",
            "own_yield", "opponent_yield", "own_farm", "opponent_farm",
            "own_sales", "opponent_sales", "invalid", "resync", "market_trim",
            "branch", "town_count", "town_shops",
        )
        for name, value in zip(names, scanned, strict=True):
            detail[f"seat{seat}__{name}"] = value
        detail[f"seat{seat}__terminal_money"] = money
        detail[f"seat{seat}__done"] = done
        detail[f"seat{seat}__seeds"] = base_seeds
        margin = money[:, seat] - money[:, 1 - seat]
        rows.append({
            "seat": seat,
            "games": args.batch,
            "win_rate": float(np.mean(margin > 0)),
            "mean_margin": float(np.mean(margin)),
            "hard_total": int(np.sum(carry.hard_counter_total)),
            "invalid_total": int(np.sum(carry.invalid_intent_total)),
            "resync_total": int(np.sum(carry.resync_total)),
            "all_done": bool(np.all(done)),
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **detail)
    receipt = {
        "schema": "kaggriculture.hasegawa_v2_vs_flex.stepwise_diagnostics",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed_start": args.seed_start,
        "games_per_seat": args.batch,
        "rows": rows,
        "artifact": str(args.output),
        "status": "PASS" if all(row["all_done"] and row["hard_total"] == 0 for row in rows) else "FAIL",
    }
    args.output.with_suffix(".json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
