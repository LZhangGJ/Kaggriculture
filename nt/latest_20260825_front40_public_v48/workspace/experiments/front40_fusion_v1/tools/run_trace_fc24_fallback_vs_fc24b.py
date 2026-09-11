"""Evaluate a trace opening/router with FC24B fallback on weak public shops."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/hasegawa_jax_v3/src",
    ROOT / "experiments/hasegawa_jax_v2/src",
    ROOT / "experiments/hasegawa_jax_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from hasegawa_jax_v3 import (  # noqa: E402
    hasegawa_step_with_external_v3,
    initialize_hasegawa_carry_v3,
    load_hasegawa_trace_bank_v3,
)
from hasegawa_jax_v3.agent import ROUTER_FIRST_SHOP_MAP  # noqa: E402
from kaggriculture_jax.constants import SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_events_v1, load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def select_carry(mask: jax.Array, old, new):
    def choose(old_value, new_value):
        shaped = mask.reshape((mask.shape[0],) + (1,) * (old_value.ndim - 1))
        return jnp.where(shaped, new_value, old_value)

    return jax.tree_util.tree_map(choose, old, new)


def make_fc_policy(player, tables, latest_bank, old_bank, runtime):
    @jax.jit
    def policy(states, carry):
        return fc24_terminal_crop_salvage_player_action_v1(
            states,
            tables,
            latest_bank,
            old_bank,
            runtime,
            carry,
            player,
        )

    return policy


def make_fallback_policy(player, tables, latest_bank, old_bank, runtime, weak_shop_mask):
    @jax.jit
    def policy(states, carry):
        action, proposed_carry = fc24_terminal_crop_salvage_player_action_v1(
            states,
            tables,
            latest_bank,
            old_bank,
            runtime,
            carry,
            player,
        )
        first_shop = jnp.clip(states.town_shops[:, 0].astype(jnp.int32), 0, 7)
        use_fallback = (states.town_count > 0) & weak_shop_mask[first_shop]
        carry = select_carry(use_fallback, carry, proposed_carry)
        return action, carry, use_fallback

    return policy


def make_trace_step(trace_player, tables):
    @jax.jit
    def one_step(states, trace_carry, opponent_action, fallback_action, use_fallback, events, trace_bank, route_map):
        states, trace_carry, _, _ = hasegawa_step_with_external_v3(
            states,
            trace_carry,
            trace_bank,
            opponent_action,
            trace_player,
            events,
            tables,
            ROUTER_FIRST_SHOP_MAP,
            route_map,
            15_000,
            5_000,
            2,
            controlled_action_override=fallback_action,
            controlled_action_override_mask=use_fallback,
        )
        return states, trace_carry

    return one_step


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--route-map", type=Path, required=True)
    parser.add_argument("--latest-bank", type=Path, required=True)
    parser.add_argument("--old-bank", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument(
        "--fallback-shop-ids",
        default="1,3,6",
        help="Comma-separated public SHOP_NAMES indices that use FC24B fallback.",
    )
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed-start", type=int, default=236_001)
    parser.add_argument(
        "--compilation-cache",
        type=Path,
        default=ROOT / "experiments/front40_fusion_v1/artifacts/jax_compilation_cache",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.compilation_cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(args.compilation_cache.resolve()))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    trace_bank = load_hasegawa_trace_bank_v3(args.trace_bank)
    mapping = json.loads(args.route_map.read_text(encoding="utf-8"))
    route_map = jnp.asarray(mapping["route_ids"], dtype=jnp.int16)
    if route_map.shape != (8,):
        raise ValueError(f"route map must have shape (8,), got {route_map.shape}")
    fallback_ids = tuple(sorted({int(value) for value in args.fallback_shop_ids.split(",") if value}))
    if any(value < 0 or value >= 8 for value in fallback_ids):
        raise ValueError(f"invalid fallback shop IDs: {fallback_ids}")
    weak_shop_mask = jnp.asarray(
        [shop_id in fallback_ids for shop_id in range(8)], dtype=jnp.bool_
    )

    latest_bank = load_bank(args.latest_bank)
    old_bank = load_bank(args.old_bank)
    runtime = load_high_potential_runtime_tables_v1(args.runtime)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.batch, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(seeds))

    rows = []
    per_game = []
    for trace_player in (0, 1):
        opponent_policy = make_fc_policy(
            1 - trace_player, tables, latest_bank, old_bank, runtime
        )
        fallback_policy = make_fallback_policy(
            trace_player, tables, latest_bank, old_bank, runtime, weak_shop_mask
        )
        trace_step = make_trace_step(trace_player, tables)
        state = initial
        trace_carry = initialize_hasegawa_carry_v3(
            args.batch, trace_bank.bootstrap_route_id
        )
        candidate_fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(
            args.batch
        )
        opponent_fc_carry = initialize_fusion_champion_terminal_salvage_carry_v1(
            args.batch
        )
        start = time.perf_counter()
        for _ in range(719):
            opponent_action, opponent_fc_carry = opponent_policy(state, opponent_fc_carry)
            fallback_action, candidate_fc_carry, use_fallback = fallback_policy(
                state, candidate_fc_carry
            )
            state, trace_carry = trace_step(
                state,
                trace_carry,
                opponent_action,
                fallback_action,
                use_fallback,
                events,
                trace_bank,
                route_map,
            )
        jax.block_until_ready(state.money)
        elapsed = time.perf_counter() - start
        state, trace_carry = jax.device_get((state, trace_carry))
        opponent = 1 - trace_player
        own = np.asarray(state.money[:, trace_player], dtype=np.int64)
        rival = np.asarray(state.money[:, opponent], dtype=np.int64)
        first_shop = np.asarray(state.town_shops[:, 0], dtype=np.int8)
        fallback = np.isin(first_shop, fallback_ids)
        win = own > rival
        row = {
            "candidate_seat": trace_player,
            "games": args.batch,
            "wins": int(np.sum(win)),
            "win_rate": float(np.mean(win)),
            "mean_margin": float(np.mean(own - rival)),
            "candidate_cash_mean": float(np.mean(own)),
            "fc24b_cash_mean": float(np.mean(rival)),
            "fallback_games": int(np.sum(fallback)),
            "fallback_wins": int(np.sum(win & fallback)),
            "fallback_win_rate": float(np.mean(win[fallback])) if np.any(fallback) else None,
            "trace_games": int(np.sum(~fallback)),
            "trace_wins": int(np.sum(win & ~fallback)),
            "trace_win_rate": float(np.mean(win[~fallback])) if np.any(~fallback) else None,
            "all_done": bool(np.all(state.done)),
            "hard_counter_total": int(np.sum(trace_carry.hard_counter_total)),
            "invalid_intent_total": int(np.sum(trace_carry.invalid_intent_total)),
            "resync_total": int(np.sum(trace_carry.resync_total)),
            "elapsed_seconds": elapsed,
            "transitions_per_second": args.batch * 719 / elapsed,
        }
        rows.append(row)
        for index, seed in enumerate(seeds):
            per_game.append(
                {
                    "seed": int(seed),
                    "candidate_seat": trace_player,
                    "first_shop_id": int(first_shop[index]),
                    "first_shop": SHOP_NAMES[int(first_shop[index])],
                    "used_fallback": bool(fallback[index]),
                    "candidate_cash": int(own[index]),
                    "fc24b_cash": int(rival[index]),
                    "margin": int(own[index] - rival[index]),
                    "win": bool(win[index]),
                }
            )
        print(json.dumps(row, ensure_ascii=False), flush=True)

    games = sum(row["games"] for row in rows)
    shop_rows = []
    for shop_id, shop_name in enumerate(SHOP_NAMES):
        subset = [row for row in per_game if row["first_shop_id"] == shop_id]
        if not subset:
            continue
        shop_rows.append(
            {
                "shop_id": shop_id,
                "shop": shop_name,
                "mode": "FC24B_FALLBACK" if shop_id in fallback_ids else "CROP_DUSTA_TRACE",
                "games": len(subset),
                "wins": sum(row["win"] for row in subset),
                "win_rate": sum(row["win"] for row in subset) / len(subset),
                "mean_margin": float(np.mean([row["margin"] for row in subset])),
            }
        )
    payload = {
        "schema": "kaggriculture.front40_fusion.trace-fc24-fallback-vs-fc24b.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if all(row["all_done"] and row["hard_counter_total"] == 0 for row in rows) else "FAIL",
        "device": str(jax.devices()[0]),
        "candidate": args.candidate_name,
        "seed_start": args.seed_start,
        "batch_per_seat": args.batch,
        "route_map": str(args.route_map),
        "fallback_shop_ids": list(fallback_ids),
        "fallback_shops": [SHOP_NAMES[index] for index in fallback_ids],
        "runtime_inputs": ["current_state", "own_private_inventory", "public_first_shop"],
        "aggregate": {
            "games": games,
            "wins": sum(row["wins"] for row in rows),
            "win_rate": sum(row["wins"] for row in rows) / games,
            "mean_margin": sum(row["mean_margin"] * row["games"] for row in rows) / games,
        },
        "by_seat": rows,
        "by_first_shop": shop_rows,
        "per_game": per_game,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "aggregate": payload["aggregate"], "output": str(args.output)}, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
