"""Screen all Hasegawa programs after the first shop becomes public.

Every candidate shares the same bootstrap program until the first shop is
visible.  A candidate program may only take over once ``town_count > 0``.  Route selection
therefore uses no future random information.  The output supports choosing one
robust route per visible first-shop class on training seeds.
"""

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
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_PREFIX_LOCK  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402
from strategic_v5.public_g02_gpu import initialize_public_g02_carry_v1, public_rc5_weed_player_action_v1  # noqa: E402


SHOP_NAMES = tuple(sorted((
    "BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE",
    "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET",
)))


def make_rollout(hbank, exact_bank, tables, h_player):
    opponent = 1 - h_player

    @jax.jit
    def rollout(initial, events, forced_route):
        batch = initial.step.shape[0]
        flex_route = jnp.full((batch,), 9, dtype=jnp.int32)

        def body(value, _):
            states, hcarry, flex_carry = value
            activate = states.town_count > 0
            route = jnp.where(activate, forced_route, hcarry.branch_id).astype(jnp.int16)
            flex_action, flex_carry = public_rc5_weed_player_action_v1(
                states, exact_bank, flex_route, flex_carry, opponent
            )
            states, hcarry, _, _ = hasegawa_step_with_external_v3(
                states,
                hcarry,
                hbank,
                flex_action,
                h_player,
                events,
                tables,
                ROUTER_PREFIX_LOCK,
                None,
                forced_route=route,
                forced_lock=activate,
            )
            return (states, hcarry, flex_carry), None

        hcarry = initialize_hasegawa_carry_v3(batch, hbank.bootstrap_route_id)
        flex_carry = initialize_public_g02_carry_v1(batch)
        (terminal, hcarry, _), _ = jax.lax.scan(
            body, (initial, hcarry, flex_carry), None, length=719
        )
        return terminal.money, terminal.done, hcarry, terminal.town_shops[:, 0]

    return rollout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hasegawa-bank", type=Path, required=True)
    parser.add_argument("--exact-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--route-chunk", type=int, default=17)
    parser.add_argument("--seed-start", type=int, default=154001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    hbank = load_hasegawa_trace_bank_v3(args.hasegawa_bank)
    exact_bank = load_bank(args.exact_bank)
    _ = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    route_count = int(hbank.source_reward.shape[0])
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    cash = np.zeros((2, args.batch, route_count), np.int32)
    opponent_cash = np.zeros_like(cash)
    invalid = np.zeros_like(cash)
    resync = np.zeros_like(cash)
    hard = np.zeros_like(cash)
    done_all = True
    first_shop = np.full((2, args.batch), -1, np.int8)
    for seat in (0, 1):
        rollout = make_rollout(hbank, exact_bank, tables, seat)
        for start in range(0, route_count, args.route_chunk):
            valid_routes = np.arange(start, min(start + args.route_chunk, route_count), dtype=np.int32)
            padded = np.pad(valid_routes, (0, args.route_chunk - len(valid_routes)), mode="edge")
            expanded_seeds = np.repeat(seeds, args.route_chunk)
            forced = np.tile(padded, args.batch)
            weed, shops = build_events_v1(expanded_seeds.tolist())
            events = Events(jnp.asarray(weed), jnp.asarray(shops))
            initial = jax.vmap(reset)(jnp.asarray(expanded_seeds))
            result = rollout(initial, events, jnp.asarray(forced, dtype=jnp.int16))
            jax.block_until_ready(result[0])
            money, done, carry, shops = jax.device_get(result)
            shape = (args.batch, args.route_chunk)
            n = len(valid_routes)
            cash[seat][:, valid_routes] = money[:, seat].reshape(shape)[:, :n]
            opponent_cash[seat][:, valid_routes] = money[:, 1 - seat].reshape(shape)[:, :n]
            invalid[seat][:, valid_routes] = carry.invalid_intent_total.reshape(shape)[:, :n]
            resync[seat][:, valid_routes] = carry.resync_total.reshape(shape)[:, :n]
            hard[seat][:, valid_routes] = carry.hard_counter_total.reshape(shape)[:, :n]
            first_shop[seat] = shops.reshape(shape)[:, 0]
            done_all = done_all and bool(np.all(done))
            print(json.dumps({"seat": seat, "routes": [int(valid_routes[0]), int(valid_routes[-1])], "done": bool(np.all(done))}), flush=True)
    margin = cash - opponent_cash
    source_episode = np.asarray(hbank.source_episode_id)
    source_reward = np.asarray(hbank.source_reward)
    source_first_shop = np.asarray(hbank.source_shop_sequence)[:, 0]
    groups = []
    for shop in range(8):
        mask = first_shop[0] == shop
        if not np.any(mask):
            continue
        group_margin = margin[:, mask, :].reshape(-1, route_count)
        group_invalid = invalid[:, mask, :].reshape(-1, route_count)
        win_rate = np.mean(group_margin > 0, axis=0)
        mean_margin = np.mean(group_margin, axis=0)
        # The deployable route is selected by win rate, then margin, while
        # preferring source traces from the same visible first-shop family.
        same_source = source_first_shop == shop
        order = np.lexsort((source_reward, same_source.astype(np.int8), mean_margin, win_rate))[::-1]
        top = []
        for route_id in order[:10]:
            top.append({
                "route_id": int(route_id),
                "source_episode_id": int(source_episode[route_id]),
                "source_reward": int(source_reward[route_id]),
                "source_first_shop": SHOP_NAMES[int(source_first_shop[route_id])],
                "same_first_shop": bool(same_source[route_id]),
                "games": int(group_margin.shape[0]),
                "win_rate": float(win_rate[route_id]),
                "mean_margin": float(mean_margin[route_id]),
                "invalid_mean": float(np.mean(group_invalid[:, route_id])),
            })
        groups.append({
            "first_shop_id": shop,
            "first_shop": SHOP_NAMES[shop],
            "seed_count": int(np.sum(mask)),
            "top_routes": top,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"), seeds=seeds, first_shop=first_shop,
        cash=cash, opponent_cash=opponent_cash, margin=margin,
        invalid=invalid, resync=resync, hard=hard,
        source_episode_id=source_episode, source_reward=source_reward,
        source_first_shop=source_first_shop,
    )
    payload = {
        "schema": "kaggriculture.hasegawa_v3.postshop_route_screen",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "seed_start": args.seed_start,
        "seed_count": args.batch,
        "route_count": route_count,
        "games": int(2 * args.batch * route_count),
        "all_done": done_all,
        "groups": groups,
        "status": "PASS" if done_all and int(np.sum(hard)) == 0 else "FAIL",
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(args.output)}))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
