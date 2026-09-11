#!/usr/bin/env python3
"""Search old frozen routes for task suffixes executable from an X562 prefix."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc16_shadow_old_suffix_player_action_v1,
    fc2b_shadow_old_suffix_player_action_v1,
    initialize_fusion_champion_carry_v6,
    initialize_fusion_champion_moon_suffix_carry_v1,
    initialize_k320_old_suffix_carry_v1,
    initialize_x562_old_suffix_carry_v1,
    k320_old_suffix_player_action_v1,
    x562_old_suffix_player_action_v1,
    x562_prt_suffix_rule_player_action_v1,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


def extract_decision_features(state, own: int) -> list[dict]:
    """Serialize only own-private and mutually visible decision-time state."""

    state = jax.device_get(state)
    rival = 1 - own
    batch = int(np.asarray(state.step).shape[0])
    rows = []
    for index in range(batch):
        own_active = np.asarray(state.unit_active[index, own], dtype=bool)
        rival_active = np.asarray(state.unit_active[index, rival], dtype=bool)
        town_count = int(state.town_count[index])
        rows.append({
            "state_step": int(state.step[index]),
            "money_own": int(state.money[index, own]),
            "money_rival": int(state.money[index, rival]),
            "hires_today_own": int(state.hires_today[index, own]),
            "hires_today_rival": int(state.hires_today[index, rival]),
            "unlocked_count_own": int(state.unlocked_count[index, own]),
            "unlocked_count_rival": int(state.unlocked_count[index, rival]),
            "tile_kind_own": np.asarray(state.tile_kind[index, own], dtype=int).reshape(-1).tolist(),
            "tile_kind_rival": np.asarray(state.tile_kind[index, rival], dtype=int).reshape(-1).tolist(),
            "tile_crop_own": np.asarray(state.tile_crop[index, own], dtype=int).reshape(-1).tolist(),
            "tile_crop_rival": np.asarray(state.tile_crop[index, rival], dtype=int).reshape(-1).tolist(),
            "tile_animal_own": np.asarray(state.tile_animal[index, own], dtype=int).reshape(-1).tolist(),
            "tile_animal_rival": np.asarray(state.tile_animal[index, rival], dtype=int).reshape(-1).tolist(),
            "tile_yield_own": np.asarray(state.tile_yield[index, own], dtype=int).reshape(-1).tolist(),
            "tile_yield_rival": np.asarray(state.tile_yield[index, rival], dtype=int).reshape(-1).tolist(),
            "tile_neglect_own": np.asarray(state.tile_neglect[index, own], dtype=int).reshape(-1).tolist(),
            "tile_neglect_rival": np.asarray(state.tile_neglect[index, rival], dtype=int).reshape(-1).tolist(),
            "unit_pos_own": np.asarray(state.unit_pos[index, own][own_active], dtype=int).tolist(),
            "unit_pos_rival": np.asarray(state.unit_pos[index, rival][rival_active], dtype=int).tolist(),
            "unit_inventory_own": np.asarray(
                state.unit_inventory[index, own][own_active], dtype=int
            ).tolist(),
            "shed_own": np.asarray(state.shed[index, own], dtype=int).tolist(),
            "seeds_own": np.asarray(state.seeds[index, own], dtype=int).tolist(),
            "market_inventory": np.asarray(state.market_inventory[index], dtype=int).tolist(),
            "market_price": np.asarray(state.market_price[index], dtype=int).tolist(),
            "town_shops": np.asarray(
                state.town_shops[index, :town_count], dtype=int
            ).tolist(),
            "active_units_own": int(np.sum(own_active)),
            "active_units_rival": int(np.sum(rival_active)),
            "animal_counts_own": [
                int(np.sum(np.asarray(state.tile_animal[index, own]) == item))
                for item in range(9)
            ],
            "animal_counts_rival": [
                int(np.sum(np.asarray(state.tile_animal[index, rival]) == item))
                for item in range(9)
            ],
            "crop_counts_own": [
                int(np.sum(np.asarray(state.tile_crop[index, own]) == item))
                for item in range(9)
            ],
            "crop_counts_rival": [
                int(np.sum(np.asarray(state.tile_crop[index, rival]) == item))
                for item in range(9)
            ],
            "tile_yield_sum_own": int(
                np.sum(np.asarray(state.tile_yield[index, own]))
            ),
            "tile_yield_sum_rival": int(
                np.sum(np.asarray(state.tile_yield[index, rival]))
            ),
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=540301)
    parser.add_argument("--seeds", type=int, default=16)
    parser.add_argument("--opponent", default="gold_proxy_rank14_recursion")
    parser.add_argument("--switch-step", type=int, default=168)
    parser.add_argument(
        "--prefix",
        choices=("stable_x562", "k320", "fc2b_shadow", "fc16_shadow"),
        default="stable_x562",
    )
    parser.add_argument("--routes", required=True, help="Comma-separated old-bank route IDs")
    parser.add_argument(
        "--switch-steps",
        default="",
        help="Optional comma-separated switch steps; evaluates every route/step pair",
    )
    parser.add_argument("--save-decision-features", action="store_true")
    parser.add_argument(
        "--feature-step",
        type=int,
        default=-1,
        help="Feature snapshot step; defaults to --switch-step",
    )
    parser.add_argument(
        "--feature-variant-index",
        type=int,
        default=0,
        help="Variant whose state supplies features (0 is the baseline)",
    )
    parser.add_argument(
        "--prt-step288-selector",
        action="store_true",
        help="Evaluate the frozen public-state selector instead of an always-on suffix",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    feature_step = args.switch_step if args.feature_step < 0 else args.feature_step

    selected = [int(value) for value in args.routes.split(",") if value]
    if not selected or any(route < 0 or route >= 87 for route in selected):
        raise ValueError("invalid routes")
    requested_steps = [
        int(value) for value in args.switch_steps.split(",") if value
    ]
    if requested_steps:
        if any(step < 0 or step > 718 for step in requested_steps):
            raise ValueError("--switch-steps values must be in [0,718]")
        variants = [(-1, args.switch_step)] + [
            (route, step) for route in selected for step in requested_steps
        ]
    else:
        variants = [(-1, args.switch_step)] + [
            (route, args.switch_step) for route in selected
        ]
    if not 0 <= args.feature_variant_index < len(variants):
        raise ValueError("invalid --feature-variant-index")
    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    old_receipt_path = ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    latest_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(old_bank_path),
        "latest_bank": load_bank(latest_bank_path),
        "runtime": load_high_potential_runtime_tables_v1(runtime_path),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    names = [row["name"] for row in rr.ROSTER]
    opponent_id = names.index(args.opponent)
    base_seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    expanded = np.tile(base_seeds, len(variants))
    weed, shops = build_events_v1(expanded.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    enabled = jnp.repeat(
        jnp.asarray([route >= 0 for route, _ in variants]), args.seeds
    )
    suffix_route = jnp.repeat(
        jnp.asarray([max(route, 0) for route, _ in variants], dtype=jnp.int16),
        args.seeds,
    )
    switch_step = jnp.repeat(
        jnp.asarray([step for _, step in variants], dtype=jnp.int16), args.seeds
    )
    simulator = rr.make_simulator_step(resources["tables"])

    def make_candidate_policy(player: int):
        @jax.jit
        def policy(states, carry, enabled_arg, switch_step_arg, suffix_route_arg):
            if args.prt_step288_selector:
                if args.prefix != "stable_x562":
                    raise ValueError("PRT selector currently requires stable_x562 prefix")
                return x562_prt_suffix_rule_player_action_v1(
                    states,
                    resources["runtime"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    carry,
                    enabled_arg,
                    player,
                )
            if args.prefix == "k320":
                return k320_old_suffix_player_action_v1(
                    states,
                    resources["tables"],
                    resources["runtime"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    carry,
                    enabled_arg,
                    switch_step_arg,
                    suffix_route_arg,
                    player,
                )
            if args.prefix == "fc2b_shadow":
                return fc2b_shadow_old_suffix_player_action_v1(
                    states,
                    resources["tables"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    resources["runtime"],
                    carry,
                    enabled_arg,
                    switch_step_arg,
                    suffix_route_arg,
                    player,
                )
            if args.prefix == "fc16_shadow":
                return fc16_shadow_old_suffix_player_action_v1(
                    states,
                    resources["tables"],
                    resources["latest_bank"],
                    resources["old_bank"],
                    resources["runtime"],
                    carry,
                    enabled_arg,
                    switch_step_arg,
                    suffix_route_arg,
                    player,
                )
            return x562_old_suffix_player_action_v1(
                states,
                resources["runtime"],
                resources["latest_bank"],
                resources["old_bank"],
                carry,
                enabled_arg,
                switch_step_arg,
                suffix_route_arg,
                player,
            )
        return policy

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    started = perf_counter()
    batch = len(expanded)
    money = []
    orientation_features = []
    for seat in (0, 1):
        states = jax.vmap(reset)(jnp.asarray(expanded))
        candidate_carry = (
            initialize_fusion_champion_moon_suffix_carry_v1(batch)
            if args.prefix == "fc16_shadow"
            else initialize_fusion_champion_carry_v6(batch)
            if args.prefix == "fc2b_shadow"
            else initialize_k320_old_suffix_carry_v1(batch)
            if args.prefix == "k320"
            else initialize_x562_old_suffix_carry_v1(batch)
        )
        opponent_carry = rr.initialize_agent_carry(opponent_id, batch, resources["router"])
        candidate_policy = make_candidate_policy(seat)
        opponent_policy = rr.make_agent_policy(opponent_id, 1 - seat, **resources)
        decision_features = None
        for action_index in range(719):
            if args.save_decision_features and action_index == feature_step:
                start = args.feature_variant_index * args.seeds
                stop = start + args.seeds
                prefix_state = jax.tree_util.tree_map(
                    lambda value: value[start:stop], states
                )
                decision_features = extract_decision_features(prefix_state, seat)
            candidate_action, candidate_carry = candidate_policy(
                states,
                candidate_carry,
                enabled,
                switch_step,
                suffix_route,
            )
            opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
            states = (
                simulator(states, candidate_action, opponent_action, events)
                if seat == 0
                else simulator(states, opponent_action, candidate_action, events)
            )
        jax.block_until_ready(states.money)
        terminal = jax.device_get(states)
        if not bool(np.all(np.asarray(terminal.done))):
            raise AssertionError("not all games DONE")
        if (
            int(np.sum(np.asarray(terminal.hand_cap_hits)))
            or int(np.sum(np.asarray(terminal.market_loop_cap_hits)))
            or int(np.sum(np.asarray(terminal.price_lut_oob)))
        ):
            raise AssertionError("simulator safety counter hit")
        money.append(np.asarray(terminal.money, dtype=np.int64).reshape(len(variants), args.seeds, 2))
        if args.save_decision_features:
            if decision_features is None:
                raise AssertionError("decision features were not captured")
            orientation_features.append(decision_features)

    first, second = money
    rows = []
    margin_rows = []
    for index, (route, variant_switch_step) in enumerate(variants):
        own = np.concatenate((first[index, :, 0], second[index, :, 1]))
        rival = np.concatenate((first[index, :, 1], second[index, :, 0]))
        margins = own - rival
        margin_rows.append(margins)
        metadata = None if route < 0 else old_receipt["skeletons"][route]
        rows.append({
            "suffix_route": route,
            "switch_step": int(variant_switch_step),
            "name": args.prefix if route < 0 else metadata["opponent"],
            "source_route": "baseline" if route < 0 else metadata["route"],
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(own)),
            "mean_opponent_cash": float(np.mean(rival)),
            "mean_margin": float(np.mean(margins)),
            "per_game": [
                {
                    "seed": int(base_seeds[game % args.seeds]),
                    "candidate_seat": 0 if game < args.seeds else 1,
                    "margin": int(margin),
                }
                for game, margin in enumerate(margins.tolist())
            ],
        })
    matrix = np.stack(margin_rows, axis=0)
    oracle = np.max(matrix, axis=0)
    payload = {
        "schema": "kaggriculture.fusion_champion.old_suffix_discovery.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "official_package_version": "1.32.7",
        "opponent": args.opponent,
        "prefix": args.prefix,
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "switch_step": args.switch_step,
        "switch_steps": requested_steps,
        "feature_step": int(feature_step),
        "feature_variant_index": int(args.feature_variant_index),
        "selector": "prt_step288_tree_v1" if args.prt_step288_selector else None,
        "seat_protocol": "same seeds with seats swapped",
        "rows": rows,
        "oracle": {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)),
            "mean_margin": float(np.mean(oracle)),
        },
        "elapsed_seconds": perf_counter() - started,
    }
    if args.save_decision_features:
        payload["decision_features"] = [
            {
                "seed": int(base_seeds[index]),
                "candidate_seat": seat,
                **orientation_features[seat][index],
            }
            for seat in (0, 1)
            for index in range(args.seeds)
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS",
        "best": sorted(
            [{key: value for key, value in row.items() if key != "per_game"} for row in rows],
            key=lambda row: (row["score_rate"], row["mean_margin"]),
            reverse=True,
        )[:6],
        "oracle": payload["oracle"],
        "output": str(args.output.resolve()),
    }), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
