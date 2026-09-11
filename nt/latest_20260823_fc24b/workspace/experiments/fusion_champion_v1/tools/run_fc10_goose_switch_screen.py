#!/usr/bin/env python3
"""Screen when a Kobe C6/S6/G2 route can safely replace the FC2B trunk."""

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

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/fusion_champion_v1/src",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "experiments/expert_business_agent_v2/tools",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    _select_action,
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import PRODUCTS, SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from route_playbook_v1.trace_core import (  # noqa: E402
    initialize_trace_player_carry_v1,
    load_skeleton_bank_v1,
    skeleton_player_action_v1,
)
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import load_high_potential_runtime_tables_v1  # noqa: E402


DEFAULT_OPPONENTS = (
    "rayk_k320_adaptive_rank1",
    "gold_proxy_rank12_ai_b2b67_saas",
    "local_prt_v6",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goose-bank", type=Path, required=True)
    parser.add_argument("--seed-list", required=True)
    parser.add_argument("--switch-steps", default="0,168,192,216,719")
    parser.add_argument(
        "--trigger-mode",
        choices=("forced", "public-melon"),
        default="forced",
    )
    parser.add_argument("--opponent-melon-min", type=int, default=18)
    parser.add_argument("--feature-step", type=int, default=192)
    parser.add_argument(
        "--third-yarn-policy",
        choices=("cancel", "continue"),
        default="cancel",
        help=(
            "What to do when a public-melon goose prebuild sees YARN_STORE as "
            "the third shop. 'cancel' returns to the FC2B shadow policy; "
            "'continue' keeps the goose route for a causal ablation."
        ),
    )
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    seeds = np.asarray(
        [int(value.strip()) for value in args.seed_list.split(",") if value.strip()],
        dtype=np.int32,
    )
    switch_steps = np.asarray(
        [int(value.strip()) for value in args.switch_steps.split(",") if value.strip()],
        dtype=np.int32,
    )
    if seeds.size < 1 or len(set(seeds.tolist())) != int(seeds.size):
        raise ValueError("seeds must be non-empty and unique")
    if switch_steps.size < 2 or 719 not in switch_steps:
        raise ValueError("switch variants must include 719 as the FC2B control")
    if len(set(switch_steps.tolist())) != int(switch_steps.size):
        raise ValueError("switch steps must be unique")
    if np.any((switch_steps < 0) | (switch_steps > 719)):
        raise ValueError("switch steps must be in [0,719]")
    base_batch = int(seeds.size)
    variants = int(switch_steps.size)
    expanded_batch = base_batch * variants
    expanded_seeds = np.tile(seeds, variants)
    expanded_switch = np.repeat(switch_steps, base_batch)

    old_receipt_path = (
        ROOT
        / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json"
    )
    old_bank_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz"
    )
    latest_bank_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz"
    )
    runtime_path = (
        ROOT
        / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz"
    )
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    resources = {
        "old_bank": load_bank(old_bank_path),
        "latest_bank": load_bank(latest_bank_path),
        "runtime": load_high_potential_runtime_tables_v1(runtime_path),
        "tables": load_tables(),
        "router": build_router_arrays(old_receipt),
        "boatlee_trace": load_boatlee_trace_v1(),
    }
    goose_bank = load_skeleton_bank_v1(args.goose_bank.resolve())
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(
        jnp.asarray(np.tile(weed, (variants, 1, 1))),
        jnp.asarray(np.tile(shops, (variants, 1, 1))),
    )
    simulator = rr.make_simulator_step(resources["tables"])
    names = [row["name"] for row in rr.ROSTER]
    requested = [value.strip() for value in args.opponents.split(",") if value.strip()]
    if not requested or any(name not in names for name in requested):
        raise ValueError("unknown or empty opponent list")
    ids = {name: index for index, name in enumerate(names)}
    route = jnp.zeros((expanded_batch,), dtype=jnp.int32)
    switch = jnp.asarray(expanded_switch)
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    rows = []
    started = perf_counter()
    for opponent_name in requested:
        opponent_id = ids[opponent_name]
        orientations = []
        trigger_orientations = []
        feature_orientations = []
        for candidate_seat in (0, 1):
            states = jax.vmap(reset)(jnp.asarray(expanded_seeds))
            fc2b_carry = initialize_fusion_champion_carry_v3(expanded_batch)
            goose_carry = initialize_trace_player_carry_v1(expanded_batch)
            opponent_carry = rr.initialize_agent_carry(
                opponent_id, expanded_batch, resources["router"]
            )
            route_checked = jnp.zeros((expanded_batch,), dtype=jnp.bool_)
            route_pending = jnp.zeros((expanded_batch,), dtype=jnp.bool_)
            route_cancelled = jnp.zeros((expanded_batch,), dtype=jnp.bool_)
            opponent_policy = rr.make_agent_policy(
                opponent_id, 1 - candidate_seat, **resources
            )

            @jax.jit
            def candidate_policy(
                current,
                base_carry,
                route_carry,
                checked,
                pending,
                cancelled,
            ):
                base_action, base_carry = (
                    fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
                        current,
                        resources["tables"],
                        resources["latest_bank"],
                        resources["old_bank"],
                        resources["runtime"],
                        base_carry,
                        candidate_seat,
                    )
                )
                goose_action, route_carry = skeleton_player_action_v1(
                    current,
                    resources["tables"],
                    goose_bank,
                    route,
                    route_carry,
                    candidate_seat,
                )
                step = current.step.astype(jnp.int32)
                if args.trigger_mode == "forced":
                    use_goose = step >= switch
                else:
                    town_ready = current.town_count >= 2
                    evaluate_now = (~checked) & (step >= switch) & town_ready
                    first_two = current.town_shops[:, :2]
                    yarn = SHOP_NAMES.index("YARN_STORE")
                    milk_support = jnp.asarray(
                        [
                            SHOP_NAMES.index("PIZZA_SHOP"),
                            SHOP_NAMES.index("ICE_CREAM_SHOP"),
                            SHOP_NAMES.index("SMOOTHIE_SHOP"),
                        ],
                        dtype=jnp.int8,
                    )
                    no_yarn = jnp.all(first_two != yarn, axis=1)
                    no_milk = ~jnp.any(
                        first_two[..., None] == milk_support[None, None, :],
                        axis=(1, 2),
                    )
                    rival = 1 - candidate_seat
                    opponent_melon = jnp.sum(
                        current.tile_crop[:, rival] == PRODUCTS.index("MELON"),
                        axis=(1, 2),
                    )
                    pending = pending | (
                        evaluate_now
                        & no_yarn
                        & no_milk
                        & (opponent_melon >= args.opponent_melon_min)
                    )
                    checked = checked | evaluate_now
                    third_yarn = (
                        (current.town_count >= 3)
                        & (current.town_shops[:, 2] == yarn)
                    )
                    if args.third_yarn_policy == "cancel":
                        cancelled = cancelled | (pending & third_yarn)
                    use_goose = pending & (~cancelled)
                return (
                    _select_action(use_goose, goose_action, base_action),
                    base_carry,
                    route_carry,
                    checked,
                    pending,
                    cancelled,
                )

            feature_state = None
            for step_index in range(719):
                if step_index == args.feature_step:
                    feature_state = states
                (
                    candidate_action,
                    fc2b_carry,
                    goose_carry,
                    route_checked,
                    route_pending,
                    route_cancelled,
                ) = candidate_policy(
                    states,
                    fc2b_carry,
                    goose_carry,
                    route_checked,
                    route_pending,
                    route_cancelled,
                )
                opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
                states = (
                    simulator(states, candidate_action, opponent_action, events)
                    if candidate_seat == 0
                    else simulator(states, opponent_action, candidate_action, events)
                )
            if feature_state is None:
                raise RuntimeError("feature step was not captured")
            terminal, feature_state, route_pending, route_cancelled = jax.device_get(
                (states, feature_state, route_pending, route_cancelled)
            )
            if not bool(np.all(np.asarray(terminal.done))):
                raise AssertionError("not all games DONE")
            if any(
                int(np.sum(np.asarray(getattr(terminal, field))))
                for field in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
            ):
                raise AssertionError("simulator safety counter hit")
            orientations.append(np.asarray(terminal.money, dtype=np.int64))
            trigger_orientations.append(
                {
                    "pending": np.asarray(route_pending, dtype=np.bool_),
                    "cancelled": np.asarray(route_cancelled, dtype=np.bool_),
                }
            )
            # Only the unswitched FC2B block is a common, deployable-state
            # feature source for later trigger analysis.
            control_index = int(np.flatnonzero(switch_steps == 719)[0])
            start = control_index * base_batch
            stop = start + base_batch
            rival = 1 - candidate_seat
            feature_orientations.append(
                {
                    "own_money": np.asarray(feature_state.money[start:stop, candidate_seat]),
                    "opp_money": np.asarray(feature_state.money[start:stop, rival]),
                    "own_animals": np.asarray(feature_state.tile_animal[start:stop, candidate_seat]),
                    "opp_animals": np.asarray(feature_state.tile_animal[start:stop, rival]),
                    "own_crops": np.asarray(feature_state.tile_crop[start:stop, candidate_seat]),
                    "opp_crops": np.asarray(feature_state.tile_crop[start:stop, rival]),
                    "own_hands": np.asarray(feature_state.unit_active[start:stop, candidate_seat]),
                    "opp_hands": np.asarray(feature_state.unit_active[start:stop, rival]),
                    "market_inventory": np.asarray(feature_state.market_inventory[start:stop]),
                    "market_price": np.asarray(feature_state.market_price[start:stop]),
                    "town_count": np.asarray(feature_state.town_count[start:stop]),
                    "town_shops": np.asarray(feature_state.town_shops[start:stop]),
                }
            )

        per_game = []
        for variant_index, switch_step in enumerate(switch_steps.tolist()):
            lo = variant_index * base_batch
            hi = lo + base_batch
            first = orientations[0][lo:hi]
            second = orientations[1][lo:hi]
            own = np.concatenate((first[:, 0], second[:, 1]))
            rival = np.concatenate((first[:, 1], second[:, 0]))
            margins = own - rival
            row = {
                "opponent": opponent_name,
                "switch_step": int(switch_step),
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                "mean_candidate_cash": float(np.mean(own)),
                "mean_opponent_cash": float(np.mean(rival)),
                "mean_margin": float(np.mean(margins)),
                "seat0_score_rate": float(np.mean(first[:, 0] > first[:, 1])),
                "seat1_score_rate": float(np.mean(second[:, 1] > second[:, 0])),
                "route_prebuild_count": int(
                    np.sum(trigger_orientations[0]["pending"][lo:hi])
                    + np.sum(trigger_orientations[1]["pending"][lo:hi])
                ),
                "route_cancelled_by_third_yarn_count": int(
                    np.sum(trigger_orientations[0]["cancelled"][lo:hi])
                    + np.sum(trigger_orientations[1]["cancelled"][lo:hi])
                ),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            for candidate_seat, money in ((0, first), (1, second)):
                own_values = money[:, candidate_seat]
                rival_values = money[:, 1 - candidate_seat]
                for index, seed in enumerate(seeds.tolist()):
                    per_game.append(
                        {
                            "seed": int(seed),
                            "candidate_seat": candidate_seat,
                            "switch_step": int(switch_step),
                            "candidate_cash": int(own_values[index]),
                            "opponent_cash": int(rival_values[index]),
                            "margin": int(own_values[index] - rival_values[index]),
                            "route_prebuild": bool(
                                trigger_orientations[candidate_seat]["pending"][
                                    lo + index
                                ]
                            ),
                            "route_cancelled_by_third_yarn": bool(
                                trigger_orientations[candidate_seat]["cancelled"][
                                    lo + index
                                ]
                            ),
                        }
                    )

        feature_rows = []
        for candidate_seat, feature in enumerate(feature_orientations):
            for index, seed in enumerate(seeds.tolist()):
                town_count = int(feature["town_count"][index])
                town_ids = feature["town_shops"][index, :town_count].astype(int).tolist()
                own_animals = feature["own_animals"][index]
                opp_animals = feature["opp_animals"][index]
                own_crops = feature["own_crops"][index]
                opp_crops = feature["opp_crops"][index]
                feature_rows.append(
                    {
                        "seed": int(seed),
                        "candidate_seat": candidate_seat,
                        "step": args.feature_step,
                        "own_money": int(feature["own_money"][index]),
                        "opp_money": int(feature["opp_money"][index]),
                        "own_geese": int(np.sum(own_animals == 0)),
                        "own_cows": int(np.sum(own_animals == 1)),
                        "own_sheep": int(np.sum(own_animals == 2)),
                        "opp_geese": int(np.sum(opp_animals == 0)),
                        "opp_cows": int(np.sum(opp_animals == 1)),
                        "opp_sheep": int(np.sum(opp_animals == 2)),
                        "own_wheat": int(np.sum(own_crops == PRODUCTS.index("WHEAT"))),
                        "own_melon": int(np.sum(own_crops == PRODUCTS.index("MELON"))),
                        "opp_wheat": int(np.sum(opp_crops == PRODUCTS.index("WHEAT"))),
                        "opp_melon": int(np.sum(opp_crops == PRODUCTS.index("MELON"))),
                        "own_units": int(np.sum(feature["own_hands"][index])),
                        "opp_units": int(np.sum(feature["opp_hands"][index])),
                        "town_shops": [SHOP_NAMES[value] for value in town_ids],
                        "market_inventory": feature["market_inventory"][index].astype(int).tolist(),
                        "market_price": feature["market_price"][index].astype(int).tolist(),
                    }
                )

        # Keep opponent-specific paired data together without bloating every
        # aggregate row.
        for row in rows:
            if row["opponent"] == opponent_name:
                row["per_game"] = [
                    item
                    for item in per_game
                    if item["switch_step"] == row["switch_step"]
                ]
                row["feature_rows_step192_fc2b_control"] = feature_rows

    payload = {
        "schema": "kaggriculture.fusion-champion.fc10-goose-switch-screen.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": (
            "Development screen on a Replay-derived goose route. Switch-step "
            "selection is not a deployable trigger and must be revalidated on held-out seeds."
        ),
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "goose_bank": str(args.goose_bank.resolve()),
        "goose_bank_sha256": sha256(args.goose_bank),
        "seed_count": base_batch,
        "seed_values": seeds.astype(int).tolist(),
        "switch_steps": switch_steps.astype(int).tolist(),
        "trigger_mode": args.trigger_mode,
        "opponent_melon_min": args.opponent_melon_min,
        "third_yarn_policy": args.third_yarn_policy,
        "feature_step": args.feature_step,
        "seat_protocol": "same independent events with seats swapped",
        "rows": rows,
        "elapsed_seconds": perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
