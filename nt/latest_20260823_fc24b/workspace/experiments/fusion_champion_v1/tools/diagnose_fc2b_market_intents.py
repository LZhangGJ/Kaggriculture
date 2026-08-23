#!/usr/bin/env python3
"""Action-level market-intent audit for FC2B versus frozen opponents.

Opponent actions are written only as offline diagnostic truth.  They are never
fed back into the candidate policy.  Any later online rule must be rebuilt from
the actor-visible state that existed before its decision.
"""

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
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from fusion_champion_v1.policy_gpu import (  # noqa: E402
    fc2b_rank14_plus_clone_aware_preempt_player_action_v1,
    initialize_fusion_champion_carry_v3,
)
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    PRODUCTS,
    MarketOp,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


DEFAULT_OPPONENTS = (
    "gold_proxy_rank12_ai_b2b67_saas",
    "local_prt_v6",
)
SNAPSHOT_AFTER_ACTION = tuple(range(23, 719, 24)) + (718,)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


@jax.jit
def sell_intent(action) -> jax.Array:
    slots = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    active = slots < action.market_count[:, None]
    valid = active & (action.market_op == MarketOp.SELL)
    amount = jnp.maximum(action.market_amount, 0)
    return jnp.stack(
        [
            jnp.sum(
                jnp.where(valid & (action.market_item == product), amount, 0),
                axis=1,
            )
            for product in range(NUM_PRODUCTS)
        ],
        axis=1,
    ).astype(jnp.int32)


def daily_totals(step_values: np.ndarray) -> np.ndarray:
    # 719 executed actions; pad the unused final slot of day 30.
    padded = np.pad(step_values, ((0, 0), (0, 1), (0, 0)))
    return padded.reshape(len(step_values), 30, 24, NUM_PRODUCTS).sum(axis=2)


def exact_events(step_values: np.ndarray, game: int) -> list[dict]:
    rows = []
    for step in np.flatnonzero(np.any(step_values[game] > 0, axis=1)):
        products = {
            PRODUCTS[product]: int(step_values[game, step, product])
            for product in range(NUM_PRODUCTS)
            if step_values[game, step, product] > 0
        }
        rows.append({"state_step": int(step), "day": int(step // 24 + 1), "sell": products})
    return rows


def first_steps(step_values: np.ndarray) -> np.ndarray:
    present = step_values > 0
    any_present = np.any(present, axis=1)
    first = np.argmax(present, axis=1)
    return np.where(any_present, first, -1).astype(np.int16)


def group_summary(
    candidate_step: np.ndarray,
    opponent_step: np.ndarray,
    snapshots: dict[str, np.ndarray],
    margins: np.ndarray,
    mask: np.ndarray,
) -> dict:
    games = int(np.sum(mask))
    if games == 0:
        return {"games": 0}
    candidate_daily = daily_totals(candidate_step[mask])
    opponent_daily = daily_totals(opponent_step[mask])
    candidate_first = first_steps(candidate_step[mask])
    opponent_first = first_steps(opponent_step[mask])

    def first_stats(values: np.ndarray) -> list[dict]:
        rows = []
        for product in range(NUM_PRODUCTS):
            valid = values[:, product] >= 0
            rows.append(
                {
                    "product": PRODUCTS[product],
                    "games_with_sale": int(np.sum(valid)),
                    "mean_first_state_step": (
                        float(np.mean(values[valid, product])) if np.any(valid) else -1.0
                    ),
                }
            )
        return rows

    return {
        "games": games,
        "mean_terminal_margin": float(np.mean(margins[mask])),
        "mean_total_candidate_sell_intent": np.mean(
            candidate_step[mask].sum(axis=1), axis=0
        ).tolist(),
        "mean_total_opponent_sell_intent": np.mean(
            opponent_step[mask].sum(axis=1), axis=0
        ).tolist(),
        "candidate_first_sale": first_stats(candidate_first),
        "opponent_first_sale": first_stats(opponent_first),
        "mean_daily_candidate_sell_intent": np.mean(candidate_daily, axis=0).tolist(),
        "mean_daily_opponent_sell_intent": np.mean(opponent_daily, axis=0).tolist(),
        "mean_daily_cumulative_sell_lead": np.mean(
            np.cumsum(candidate_daily - opponent_daily, axis=1), axis=0
        ).tolist(),
        "mean_daily_candidate_cash": np.mean(
            snapshots["candidate_money"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_cash": np.mean(
            snapshots["opponent_money"][mask], axis=0
        ).tolist(),
        "mean_daily_market_price": np.mean(
            snapshots["market_price"][mask], axis=0
        ).tolist(),
        "mean_daily_market_inventory": np.mean(
            snapshots["market_inventory"][mask], axis=0
        ).tolist(),
        "mean_daily_candidate_animal_count": np.mean(
            snapshots["candidate_animal_count"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_animal_count_public": np.mean(
            snapshots["opponent_animal_count"][mask], axis=0
        ).tolist(),
        "mean_daily_candidate_animal_yield": np.mean(
            snapshots["candidate_animal_yield"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_animal_yield_public": np.mean(
            snapshots["opponent_animal_yield"][mask], axis=0
        ).tolist(),
        "mean_daily_candidate_crop_count": np.mean(
            snapshots["candidate_crop_count"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_crop_count_public": np.mean(
            snapshots["opponent_crop_count"][mask], axis=0
        ).tolist(),
        "mean_daily_candidate_crop_yield": np.mean(
            snapshots["candidate_crop_yield"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_crop_yield_public": np.mean(
            snapshots["opponent_crop_yield"][mask], axis=0
        ).tolist(),
        "mean_daily_candidate_units": np.mean(
            snapshots["candidate_units"][mask], axis=0
        ).tolist(),
        "mean_daily_opponent_units_public": np.mean(
            snapshots["opponent_units"][mask], axis=0
        ).tolist(),
    }


def run_orientation(
    opponent_id: int,
    candidate_seat: int,
    seeds: np.ndarray,
    events: Events,
    resources: dict,
):
    batch = len(seeds)
    states = jax.vmap(reset)(jnp.asarray(seeds))
    candidate_carry = initialize_fusion_champion_carry_v3(batch)
    opponent_carry = rr.initialize_agent_carry(
        opponent_id, batch, resources["router"]
    )
    simulator = rr.make_simulator_step(resources["tables"])

    @jax.jit
    def candidate_policy(current, carry):
        return fc2b_rank14_plus_clone_aware_preempt_player_action_v1(
            current,
            resources["tables"],
            resources["latest_bank"],
            resources["old_bank"],
            resources["runtime"],
            carry,
            candidate_seat,
        )

    opponent_policy = rr.make_agent_policy(
        opponent_id, 1 - candidate_seat, **resources
    )
    candidate_steps = []
    opponent_steps = []
    snapshots = {
        "candidate_money": [],
        "opponent_money": [],
        "candidate_shed": [],
        "market_price": [],
        "market_inventory": [],
        "candidate_animal_count": [],
        "opponent_animal_count": [],
        "candidate_animal_yield": [],
        "opponent_animal_yield": [],
        "candidate_crop_count": [],
        "opponent_crop_count": [],
        "candidate_crop_yield": [],
        "opponent_crop_yield": [],
        "candidate_units": [],
        "opponent_units": [],
    }
    for action_index in range(719):
        candidate_action, candidate_carry = candidate_policy(states, candidate_carry)
        opponent_action, opponent_carry = opponent_policy(states, opponent_carry)
        candidate_steps.append(sell_intent(candidate_action))
        opponent_steps.append(sell_intent(opponent_action))
        states = (
            simulator(states, candidate_action, opponent_action, events)
            if candidate_seat == 0
            else simulator(states, opponent_action, candidate_action, events)
        )
        if action_index in SNAPSHOT_AFTER_ACTION:
            snapshots["candidate_money"].append(states.money[:, candidate_seat])
            snapshots["opponent_money"].append(states.money[:, 1 - candidate_seat])
            snapshots["candidate_shed"].append(
                states.shed[:, candidate_seat, :NUM_PRODUCTS]
            )
            snapshots["market_price"].append(states.market_price)
            snapshots["market_inventory"].append(states.market_inventory)
            candidate_tiles_animal = states.tile_animal[:, candidate_seat]
            opponent_tiles_animal = states.tile_animal[:, 1 - candidate_seat]
            candidate_tiles_crop = states.tile_crop[:, candidate_seat]
            opponent_tiles_crop = states.tile_crop[:, 1 - candidate_seat]
            snapshots["candidate_animal_count"].append(
                jnp.stack(
                    [jnp.sum(candidate_tiles_animal == item, axis=(1, 2)) for item in range(NUM_ANIMALS)],
                    axis=1,
                )
            )
            snapshots["opponent_animal_count"].append(
                jnp.stack(
                    [jnp.sum(opponent_tiles_animal == item, axis=(1, 2)) for item in range(NUM_ANIMALS)],
                    axis=1,
                )
            )
            snapshots["candidate_animal_yield"].append(
                jnp.stack(
                    [
                        jnp.sum(
                            jnp.where(
                                candidate_tiles_animal == item,
                                states.tile_yield[:, candidate_seat],
                                0,
                            ),
                            axis=(1, 2),
                        )
                        for item in range(NUM_ANIMALS)
                    ],
                    axis=1,
                )
            )
            snapshots["opponent_animal_yield"].append(
                jnp.stack(
                    [
                        jnp.sum(
                            jnp.where(
                                opponent_tiles_animal == item,
                                states.tile_yield[:, 1 - candidate_seat],
                                0,
                            ),
                            axis=(1, 2),
                        )
                        for item in range(NUM_ANIMALS)
                    ],
                    axis=1,
                )
            )
            snapshots["candidate_crop_count"].append(
                jnp.stack(
                    [jnp.sum(candidate_tiles_crop == item, axis=(1, 2)) for item in range(NUM_CROPS)],
                    axis=1,
                )
            )
            snapshots["opponent_crop_count"].append(
                jnp.stack(
                    [jnp.sum(opponent_tiles_crop == item, axis=(1, 2)) for item in range(NUM_CROPS)],
                    axis=1,
                )
            )
            snapshots["candidate_crop_yield"].append(
                jnp.stack(
                    [
                        jnp.sum(
                            jnp.where(
                                candidate_tiles_crop == item,
                                states.tile_yield[:, candidate_seat],
                                0,
                            ),
                            axis=(1, 2),
                        )
                        for item in range(NUM_CROPS)
                    ],
                    axis=1,
                )
            )
            snapshots["opponent_crop_yield"].append(
                jnp.stack(
                    [
                        jnp.sum(
                            jnp.where(
                                opponent_tiles_crop == item,
                                states.tile_yield[:, 1 - candidate_seat],
                                0,
                            ),
                            axis=(1, 2),
                        )
                        for item in range(NUM_CROPS)
                    ],
                    axis=1,
                )
            )
            snapshots["candidate_units"].append(
                jnp.sum(states.unit_active[:, candidate_seat], axis=1)
            )
            snapshots["opponent_units"].append(
                jnp.sum(states.unit_active[:, 1 - candidate_seat], axis=1)
            )

    jax.block_until_ready(states.money)
    terminal = jax.device_get(states)
    if not bool(np.all(np.asarray(terminal.done))):
        raise AssertionError("not all games DONE")
    if any(
        int(np.sum(np.asarray(getattr(terminal, name))))
        for name in ("hand_cap_hits", "market_loop_cap_hits", "price_lut_oob")
    ):
        raise AssertionError("simulator safety counter hit")
    return {
        "money": np.asarray(terminal.money, dtype=np.int64),
        "candidate_step": np.asarray(
            jax.device_get(jnp.stack(candidate_steps, axis=1)), dtype=np.int32
        ),
        "opponent_step": np.asarray(
            jax.device_get(jnp.stack(opponent_steps, axis=1)), dtype=np.int32
        ),
        "snapshots": {
            key: np.asarray(jax.device_get(jnp.stack(value, axis=1)))
            for key, value in snapshots.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument(
        "--seed-list",
        default="",
        help="Optional comma-separated seed list; overrides --seed-start/--seeds.",
    )
    parser.add_argument("--opponents", default=",".join(DEFAULT_OPPONENTS))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")
    if not args.seed_list and args.seed_start is None:
        raise ValueError("either --seed-list or --seed-start is required")

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
    ids = {name: index for index, name in enumerate(names)}
    opponents = [name.strip() for name in args.opponents.split(",") if name.strip()]
    if len(opponents) != len(set(opponents)) or any(name not in ids for name in opponents):
        raise ValueError("invalid opponents")

    if args.seed_list:
        seeds = np.asarray(
            [int(value.strip()) for value in args.seed_list.split(",") if value.strip()],
            dtype=np.int32,
        )
        if seeds.size < 1 or len(set(seeds.tolist())) != int(seeds.size):
            raise ValueError("--seed-list must contain unique seeds")
    else:
        seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    started = perf_counter()
    pairs = []
    for opponent in opponents:
        orientations = [
            run_orientation(ids[opponent], seat, seeds, events, resources)
            for seat in (0, 1)
        ]
        candidate_cash = np.concatenate(
            (orientations[0]["money"][:, 0], orientations[1]["money"][:, 1])
        )
        opponent_cash = np.concatenate(
            (orientations[0]["money"][:, 1], orientations[1]["money"][:, 0])
        )
        margins = candidate_cash - opponent_cash
        candidate_step = np.concatenate(
            (orientations[0]["candidate_step"], orientations[1]["candidate_step"]),
            axis=0,
        )
        opponent_step = np.concatenate(
            (orientations[0]["opponent_step"], orientations[1]["opponent_step"]),
            axis=0,
        )
        snapshots = {
            key: np.concatenate(
                (orientations[0]["snapshots"][key], orientations[1]["snapshots"][key]),
                axis=0,
            )
            for key in orientations[0]["snapshots"]
        }
        masks = {
            "all": np.ones(len(margins), dtype=bool),
            "wins": margins > 0,
            "losses": margins < 0,
        }
        per_game = []
        candidate_daily = daily_totals(candidate_step)
        opponent_daily = daily_totals(opponent_step)
        for game in range(len(margins)):
            source = game % len(seeds)
            per_game.append(
                {
                    "seed": int(seeds[source]),
                    "candidate_seat": 0 if game < len(seeds) else 1,
                    "candidate_cash": int(candidate_cash[game]),
                    "opponent_cash": int(opponent_cash[game]),
                    "margin": int(margins[game]),
                    "result": "win" if margins[game] > 0 else ("tie" if margins[game] == 0 else "loss"),
                    "candidate_sell_events": exact_events(candidate_step, game),
                    "opponent_sell_events_offline_only": exact_events(opponent_step, game),
                    "daily_candidate_sell_intent": candidate_daily[game].astype(int).tolist(),
                    "daily_opponent_sell_intent_offline_only": opponent_daily[game].astype(int).tolist(),
                    "daily_candidate_cash": snapshots["candidate_money"][game].astype(int).tolist(),
                    "daily_opponent_cash_public": snapshots["opponent_money"][game].astype(int).tolist(),
                    "daily_candidate_shed": snapshots["candidate_shed"][game].astype(int).tolist(),
                    "daily_market_price": snapshots["market_price"][game].astype(int).tolist(),
                    "daily_market_inventory": snapshots["market_inventory"][game].astype(int).tolist(),
                    "daily_candidate_animal_count": snapshots["candidate_animal_count"][game].astype(int).tolist(),
                    "daily_opponent_animal_count_public": snapshots["opponent_animal_count"][game].astype(int).tolist(),
                    "daily_candidate_animal_yield": snapshots["candidate_animal_yield"][game].astype(int).tolist(),
                    "daily_opponent_animal_yield_public": snapshots["opponent_animal_yield"][game].astype(int).tolist(),
                    "daily_candidate_crop_count": snapshots["candidate_crop_count"][game].astype(int).tolist(),
                    "daily_opponent_crop_count_public": snapshots["opponent_crop_count"][game].astype(int).tolist(),
                    "daily_candidate_crop_yield": snapshots["candidate_crop_yield"][game].astype(int).tolist(),
                    "daily_opponent_crop_yield_public": snapshots["opponent_crop_yield"][game].astype(int).tolist(),
                    "daily_candidate_units": snapshots["candidate_units"][game].astype(int).tolist(),
                    "daily_opponent_units_public": snapshots["opponent_units"][game].astype(int).tolist(),
                }
            )
        row = {
            "candidate": "fc2b_rank14_plus_k320_clone_aware_preempt",
            "opponent": opponent,
            "games": int(len(margins)),
            "wins": int(np.sum(margins > 0)),
            "ties": int(np.sum(margins == 0)),
            "losses": int(np.sum(margins < 0)),
            "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
            "mean_margin": float(np.mean(margins)),
            "groups": {
                name: group_summary(
                    candidate_step, opponent_step, snapshots, margins, mask
                )
                for name, mask in masks.items()
            },
            "per_game": per_game,
        }
        pairs.append(row)
        print(
            json.dumps(
                {key: value for key, value in row.items() if key not in ("groups", "per_game")}
            ),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.fusion_champion.fc2b-market-intent-audit.v2",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": args.seed_start,
        "seed_values": seeds.astype(int).tolist(),
        "seed_count": int(seeds.size),
        "seat_protocol": "same seeds with seats swapped",
        "products": list(PRODUCTS),
        "crops": list(CROPS),
        "animals": list(ANIMALS),
        "diagnostic_boundary": (
            "opponent action intents are offline attribution truth only; they are not legal online features"
        ),
        "sources": [
            {"path": str(path.resolve()), "sha256": sha256(path)}
            for path in (old_bank_path, old_receipt_path, latest_bank_path, runtime_path)
        ],
        "elapsed_seconds": perf_counter() - started,
        "pairs": pairs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
