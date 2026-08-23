#!/usr/bin/env python3
"""FC15 versus the frozen 2026-08-22 public-six JAX controllers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/kawashigi_counterfactual_ranker_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/strategic_v5/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc15_fc14_x562_split_weed_hire_guard_player_action_v1,
    fc15_kaito_market_skill_player_action_v1,
    fc15_moon_embedded_market_observer_player_action_v1,
    fc15_moon_market_observer_player_action_v1,
    fc15_moon_parameter_player_action_v1,
    fc15_opening_transaction_player_action_v1,
    fc17_opening_hedge_player_action_v1,
    fc17_ueddy_s192_player_action_v1,
    fc19_moon_h4_wheat8_player_action_v1,
    fc20_wheat8_virtual_baseline_player_action_v1,
    fc21_wheat8_feed_cash_guard_player_action_v1,
    fc22_feed_value_guard_player_action_v1,
    fc24_terminal_crop_salvage_player_action_v1,
    initialize_fusion_champion_carry_v3,
    initialize_fusion_champion_feed_value_carry_v1,
    initialize_fusion_champion_kaito_market_carry_v1,
    initialize_fusion_champion_moon_market_carry_v1,
    initialize_fusion_champion_moon_suffix_carry_v1,
    initialize_fusion_champion_terminal_salvage_carry_v1,
)
from kaggriculture_jax.constants import PRODUCTS, SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import load_bank  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public6_20260822_gpu import (  # noqa: E402
    boatlee_v21_player_action_v1,
    initialize_boatlee_v21_carry_v1,
    initialize_kaito_v39_carry_v1,
    initialize_latest_e284_carry_v1,
    initialize_moon_v92_carry_v1,
    initialize_soil_v26h_carry_v1,
    kaito_v39_history_gate_player_action_v1,
    prvsiyan_moon_v92_player_action_v1,
    prvsiyan_soil_v26h_player_action_v1,
    salem_harvestforge_x_player_action_v1,
    steven_e284_hadouken_player_action_v1,
)
from strategic_v5.latest_public8_gpu import initialize_x562_carry_v1  # noqa: E402


OPPONENTS = {
    "boatlee_v21_latest": (
        boatlee_v21_player_action_v1,
        initialize_boatlee_v21_carry_v1,
    ),
    "prvsiyan_soil_v26h_latest": (
        prvsiyan_soil_v26h_player_action_v1,
        initialize_soil_v26h_carry_v1,
    ),
    "prvsiyan_moon_v92_latest": (
        prvsiyan_moon_v92_player_action_v1,
        initialize_moon_v92_carry_v1,
    ),
    "kaito_v39_history_gate_latest": (
        kaito_v39_history_gate_player_action_v1,
        initialize_kaito_v39_carry_v1,
    ),
    "steven_e284_hadouken_latest": (
        steven_e284_hadouken_player_action_v1,
        initialize_latest_e284_carry_v1,
    ),
    "salem_harvestforge_x_latest": (
        salem_harvestforge_x_player_action_v1,
        initialize_x562_carry_v1,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def compact_decision_features(state, own: int) -> list[dict]:
    """Serialize public/own-private aggregates at the route decision point."""

    state = jax.device_get(state)
    rival = 1 - own
    rows = []
    for index in range(int(np.asarray(state.step).shape[0])):
        town_count = int(state.town_count[index])
        own_animals = np.asarray(state.tile_animal[index, own], dtype=np.int16)
        rival_animals = np.asarray(state.tile_animal[index, rival], dtype=np.int16)
        own_crops = np.asarray(state.tile_crop[index, own], dtype=np.int16)
        rival_crops = np.asarray(state.tile_crop[index, rival], dtype=np.int16)
        rows.append(
            {
                "state_step": int(state.step[index]),
                "money_own": int(state.money[index, own]),
                "money_rival": int(state.money[index, rival]),
                "hires_today_own": int(state.hires_today[index, own]),
                "hires_today_rival": int(state.hires_today[index, rival]),
                "unlocked_count_own": int(state.unlocked_count[index, own]),
                "unlocked_count_rival": int(state.unlocked_count[index, rival]),
                "active_units_own": int(np.sum(np.asarray(state.unit_active[index, own]))),
                "active_units_rival": int(np.sum(np.asarray(state.unit_active[index, rival]))),
                "animal_counts_own": [int(np.sum(own_animals == item)) for item in range(len(PRODUCTS))],
                "animal_counts_rival": [int(np.sum(rival_animals == item)) for item in range(len(PRODUCTS))],
                "crop_counts_own": [int(np.sum(own_crops == item)) for item in range(len(PRODUCTS))],
                "crop_counts_rival": [int(np.sum(rival_crops == item)) for item in range(len(PRODUCTS))],
                "tile_yield_sum_own": int(np.sum(np.asarray(state.tile_yield[index, own]))),
                "tile_yield_sum_rival": int(np.sum(np.asarray(state.tile_yield[index, rival]))),
                "shed_own": np.asarray(state.shed[index, own], dtype=int).tolist(),
                "seeds_own": np.asarray(state.seeds[index, own], dtype=int).tolist(),
                "market_inventory": np.asarray(state.market_inventory[index], dtype=int).tolist(),
                "market_price": np.asarray(state.market_price[index], dtype=int).tolist(),
                "town_shops": np.asarray(state.town_shops[index, :town_count], dtype=int).tolist(),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=823001)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument(
        "--candidate",
        choices=(
            "fc15",
            "fc15_moon_market",
            "fc15_moon_embedded_market",
            "fc15_kaito_maker",
            "fc15_kaito_both",
            "fc16_moon_h4",
            "fc17_ueddy_s192",
            "fc17_opening_hedge",
            "fc18_wheat8",
            "fc19_moon_h4_wheat8",
            "fc20_wheat8_virtual_baseline",
            "fc21_wheat8_feed_cash_guard",
            "fc22_feed_value_guard",
            "fc24_terminal_crop_salvage",
        ),
        default="fc15",
    )
    parser.add_argument(
        "--opponents", default=",".join(OPPONENTS),
        help="Comma-separated subset of the frozen six.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-decision-features", action="store_true")
    parser.add_argument("--feature-step", type=int, default=192)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    selected = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if len(selected) != len(set(selected)) or any(value not in OPPONENTS for value in selected):
        raise ValueError("invalid or duplicate opponent")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if args.candidate in ("fc15", "fc18_wheat8"):
        candidate_fn = fc15_fc14_x562_split_weed_hire_guard_player_action_v1
        candidate_init = initialize_fusion_champion_carry_v3
    elif args.candidate == "fc15_moon_market":
        candidate_fn = fc15_moon_market_observer_player_action_v1
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate == "fc15_moon_embedded_market":
        candidate_fn = fc15_moon_embedded_market_observer_player_action_v1
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate in ("fc15_kaito_maker", "fc15_kaito_both"):
        candidate_fn = None
        candidate_init = initialize_fusion_champion_kaito_market_carry_v1
    elif args.candidate == "fc16_moon_h4":
        candidate_fn = None
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate == "fc19_moon_h4_wheat8":
        candidate_fn = fc19_moon_h4_wheat8_player_action_v1
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate == "fc20_wheat8_virtual_baseline":
        candidate_fn = fc20_wheat8_virtual_baseline_player_action_v1
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate == "fc21_wheat8_feed_cash_guard":
        candidate_fn = fc21_wheat8_feed_cash_guard_player_action_v1
        candidate_init = initialize_fusion_champion_moon_market_carry_v1
    elif args.candidate == "fc22_feed_value_guard":
        candidate_fn = fc22_feed_value_guard_player_action_v1
        candidate_init = initialize_fusion_champion_feed_value_carry_v1
    elif args.candidate == "fc24_terminal_crop_salvage":
        candidate_fn = fc24_terminal_crop_salvage_player_action_v1
        candidate_init = initialize_fusion_champion_terminal_salvage_carry_v1
    elif args.candidate == "fc17_ueddy_s192":
        candidate_fn = fc17_ueddy_s192_player_action_v1
        candidate_init = initialize_fusion_champion_moon_suffix_carry_v1
    else:
        candidate_fn = fc17_opening_hedge_player_action_v1
        candidate_init = initialize_fusion_champion_moon_suffix_carry_v1

    tables = load_tables()
    old_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    latest8_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    latest6_bank_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public6_20260822_route_bank_v1.npz"
    runtime_path = ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    old_bank = load_bank(old_bank_path)
    latest8_bank = load_bank(latest8_bank_path)
    latest6_bank = load_bank(latest6_bank_path)
    runtime = load_high_potential_runtime_tables_v1(runtime_path)
    simulator = rr.make_simulator_step(tables)

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    batch = int(seeds.size)
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in selected:
        opponent_fn, opponent_init = OPPONENTS[opponent_name]
        orientation_money = []
        orientation_states = []
        orientation_candidate_carries = []
        orientation_opponent_carries = []
        orientation_decision_features = []
        for candidate_seat in (0, 1):
            player = candidate_seat
            rival = 1 - candidate_seat

            @jax.jit
            def candidate_policy(states, carry):
                if args.candidate in ("fc19_moon_h4_wheat8", "fc20_wheat8_virtual_baseline", "fc21_wheat8_feed_cash_guard", "fc22_feed_value_guard", "fc24_terminal_crop_salvage"):
                    return candidate_fn(
                        states,
                        tables,
                        latest8_bank,
                        old_bank,
                        runtime,
                        carry,
                        player,
                    )
                if args.candidate == "fc18_wheat8":
                    batch_shape = states.step.shape
                    return fc15_opening_transaction_player_action_v1(
                        states,
                        tables,
                        latest8_bank,
                        old_bank,
                        runtime,
                        carry,
                        jnp.full(batch_shape, 8, dtype=jnp.int16),
                        jnp.full(batch_shape, 12, dtype=jnp.int16),
                        jnp.full(batch_shape, 6, dtype=jnp.int16),
                        player,
                    )
                if args.candidate == "fc16_moon_h4":
                    batch_shape = states.step.shape
                    return fc15_moon_parameter_player_action_v1(
                        states,
                        tables,
                        latest8_bank,
                        old_bank,
                        runtime,
                        carry,
                        jnp.full(batch_shape, 4, dtype=jnp.int16),
                        jnp.full(batch_shape, 120, dtype=jnp.int16),
                        jnp.full(batch_shape, 2, dtype=jnp.int16),
                        jnp.full(batch_shape, 6, dtype=jnp.int16),
                        player,
                    )
                if args.candidate in ("fc17_ueddy_s192", "fc17_opening_hedge"):
                    return candidate_fn(
                        states,
                        tables,
                        latest8_bank,
                        old_bank,
                        runtime,
                        carry,
                        player,
                    )
                if args.candidate in ("fc15_kaito_maker", "fc15_kaito_both"):
                    enabled = jnp.ones(states.step.shape, dtype=jnp.bool_)
                    return fc15_kaito_market_skill_player_action_v1(
                        states,
                        tables,
                        latest8_bank,
                        old_bank,
                        runtime,
                        carry,
                        enabled if args.candidate == "fc15_kaito_both" else ~enabled,
                        enabled,
                        player,
                    )
                return candidate_fn(
                    states,
                    tables,
                    latest8_bank,
                    old_bank,
                    runtime,
                    carry,
                    player,
                )

            @jax.jit
            def opponent_policy(states, carry):
                return opponent_fn(
                    states, runtime, latest6_bank, carry, rival
                )

            states = jax.vmap(reset)(jnp.asarray(seeds))
            candidate_carry = candidate_init(batch)
            opponent_carry = opponent_init(batch)
            decision_features = None
            for action_index in range(719):
                if args.save_decision_features and action_index == args.feature_step:
                    decision_features = compact_decision_features(states, player)
                candidate_action, candidate_carry = candidate_policy(
                    states, candidate_carry
                )
                opponent_action, opponent_carry = opponent_policy(
                    states, opponent_carry
                )
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if candidate_seat == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError(f"{opponent_name} seat{candidate_seat}: unfinished")
            if any(
                int(np.sum(np.asarray(value)))
                for value in (
                    terminal.hand_cap_hits,
                    terminal.market_loop_cap_hits,
                    terminal.price_lut_oob,
                )
            ):
                raise AssertionError(f"{opponent_name} seat{candidate_seat}: safety counter")
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
            orientation_states.append(terminal)
            orientation_candidate_carries.append(jax.device_get(candidate_carry))
            orientation_opponent_carries.append(jax.device_get(opponent_carry))
            orientation_decision_features.append(decision_features)

        first, second = orientation_money
        candidate_cash = np.concatenate((first[:, 0], second[:, 1]))
        opponent_cash = np.concatenate((first[:, 1], second[:, 0]))
        margins = candidate_cash - opponent_cash
        per_game = []
        for index, seed in enumerate(seeds.tolist()):
            for candidate_seat, money, terminal in (
                (0, first, orientation_states[0]),
                (1, second, orientation_states[1]),
            ):
                margin = int(money[index, candidate_seat] - money[index, 1 - candidate_seat])
                count = int(np.asarray(terminal.town_count)[index])
                shop_ids = np.asarray(terminal.town_shops)[index, :count].astype(int).tolist()
                candidate_carry = orientation_candidate_carries[candidate_seat]
                if args.candidate in ("fc15", "fc18_wheat8"):
                    candidate_base_carry = candidate_carry
                elif args.candidate in ("fc17_ueddy_s192", "fc17_opening_hedge"):
                    candidate_base_carry = candidate_carry.base.base
                elif args.candidate == "fc22_feed_value_guard":
                    candidate_base_carry = candidate_carry.base.base
                elif args.candidate == "fc24_terminal_crop_salvage":
                    candidate_base_carry = candidate_carry.base.base.base
                else:
                    candidate_base_carry = candidate_carry.base
                opponent_carry = orientation_opponent_carries[candidate_seat]
                opponent_route = None
                opponent_market_overlay = None
                if opponent_name == "boatlee_v21_latest":
                    opponent_route = int(np.asarray(opponent_carry.route)[index])
                    opponent_market_overlay = bool(
                        np.asarray(opponent_carry.market_overlay)[index]
                    )
                elif opponent_name == "prvsiyan_soil_v26h_latest":
                    opponent_route = 9
                elif opponent_name == "prvsiyan_moon_v92_latest":
                    opponent_route = int(
                        np.asarray(opponent_carry.base.ray_route_id)[index]
                    )
                game_row = {
                        "seed": int(seed),
                        "candidate_seat": candidate_seat,
                        "candidate_cash": int(money[index, candidate_seat]),
                        "opponent_cash": int(money[index, 1 - candidate_seat]),
                        "margin": margin,
                        "result": "win" if margin > 0 else ("tie" if margin == 0 else "loss"),
                        "town_shop_ids": shop_ids,
                        "town_shops": [SHOP_NAMES[value] for value in shop_ids],
                        "fc15_sheep_pressure": bool(
                            np.asarray(candidate_base_carry.sheep_pressure)[index]
                        ),
                        "fc15_k320_route": int(
                            np.asarray(candidate_base_carry.k320.ray_route_id)[index]
                        ),
                        "fc15_x562_route": int(
                            np.asarray(candidate_base_carry.x562_suffix.x562.route_id)[index]
                        ),
                        "fc15_prt_switched": bool(
                            np.asarray(candidate_base_carry.x562_suffix.switched)[index]
                        ),
                        "opponent_route": opponent_route,
                        "opponent_market_overlay": opponent_market_overlay,
                        "terminal_market_prices": {
                            name: int(value)
                            for name, value in zip(
                                PRODUCTS,
                                np.asarray(terminal.market_price)[index].astype(int).tolist(),
                                strict=True,
                            )
                        },
                    }
                if args.candidate in ("fc17_ueddy_s192", "fc17_opening_hedge"):
                    game_row["fc17_opening_match"] = bool(
                        np.asarray(candidate_carry.opening_match)[index]
                    )
                    game_row["fc17_suffix_switched"] = bool(
                        np.asarray(candidate_carry.switched)[index]
                    )
                if args.save_decision_features:
                    if orientation_decision_features[candidate_seat] is None:
                        raise AssertionError("decision features were not captured")
                    game_row["decision_features"] = orientation_decision_features[candidate_seat][index]
                per_game.append(game_row)
        row = {
            "opponent": opponent_name,
            "games": int(margins.size),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_candidate_cash": float(np.mean(candidate_cash)),
            "mean_opponent_cash": float(np.mean(opponent_cash)),
            "mean_margin": float(np.mean(margins)),
            "median_margin": float(np.median(margins)),
            "candidate_seat0_wins": int(np.sum(first[:, 0] > first[:, 1])),
            "candidate_seat1_wins": int(np.sum(second[:, 1] > second[:, 0])),
            "per_game": per_game,
        }
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "per_game"}), flush=True)

    source = ROOT / "experiments/strategic_v5/src/strategic_v5/latest_public6_20260822_gpu.py"
    fc15_source = ROOT / "experiments/fusion_champion_v1/src/fusion_champion_v1/policy_gpu.py"
    payload = {
        "schema": "kaggriculture.fc15-vs-latest-public6-panel.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "candidate": args.candidate,
        "seed_start": int(args.seed_start),
        "seed_count": batch,
        "games_per_opponent": batch * 2,
        "seat_protocol": "same seeds with seats swapped",
        "feature_step": int(args.feature_step) if args.save_decision_features else None,
        "elapsed_seconds": perf_counter() - started,
        "rows": rows,
        "min_score_rate": min(row["score_rate"] for row in rows),
        "mean_score_rate": float(np.mean([row["score_rate"] for row in rows])),
        "sources": {
            "fc15": {"path": str(fc15_source), "sha256": sha256(fc15_source)},
            "latest_public6_jax": {"path": str(source), "sha256": sha256(source)},
            "old_bank": {"path": str(old_bank_path), "sha256": sha256(old_bank_path)},
            "latest8_bank": {"path": str(latest8_bank_path), "sha256": sha256(latest8_bank_path)},
            "latest6_bank": {"path": str(latest6_bank_path), "sha256": sha256(latest6_bank_path)},
            "runtime": {"path": str(runtime_path), "sha256": sha256(runtime_path)},
        },
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
