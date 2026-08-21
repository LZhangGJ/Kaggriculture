#!/usr/bin/env python3
"""Daily public/economic trajectory audit for strict JAX agents.

The tool reuses only agents that are already part of the frozen strict-parity
round-robin roster.  It records state-derived facts, not opponent identities as
online features.  Results are canonicalized so the named candidate is always
the left side even when seats are swapped.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "experiments/expert_business_agent_v2/tools"))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    NUM_PRODUCTS,
    PRODUCTS,
    TileKind,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import build_router_arrays, load_bank  # noqa: E402
from strategic_v5.boatlee_v16_gpu import load_boatlee_trace_v1  # noqa: E402
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    load_high_potential_runtime_tables_v1,
)


DEFAULT_PAIRS = (
    ("rayk_k320_adaptive_rank1", "gold_proxy_rank14_recursion"),
    ("x562_latest", "gold_proxy_rank14_recursion"),
    ("gold_proxy_rank07_junichiro_morita", "gold_proxy_rank14_recursion"),
)
SNAPSHOT_AFTER_ACTION = tuple(range(23, 719, 24)) + (718,)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def counts(values: np.ndarray, count: int) -> np.ndarray:
    return np.stack([np.sum(values == index, axis=(-2, -1)) for index in range(count)], axis=1)


def extract_snapshot(state, own: int) -> dict[str, np.ndarray]:
    state = jax.device_get(state)
    rival = 1 - own
    own_kind = np.asarray(state.tile_kind[:, own])
    rival_kind = np.asarray(state.tile_kind[:, rival])
    own_crop = np.asarray(state.tile_crop[:, own])
    rival_crop = np.asarray(state.tile_crop[:, rival])
    own_animal = np.asarray(state.tile_animal[:, own])
    rival_animal = np.asarray(state.tile_animal[:, rival])
    own_yield = np.asarray(state.tile_yield[:, own])
    rival_yield = np.asarray(state.tile_yield[:, rival])

    own_crop_yield = np.stack(
        [np.sum(np.where(own_crop == index, own_yield, 0), axis=(-2, -1)) for index in range(len(CROPS))],
        axis=1,
    )
    rival_crop_yield = np.stack(
        [np.sum(np.where(rival_crop == index, rival_yield, 0), axis=(-2, -1)) for index in range(len(CROPS))],
        axis=1,
    )
    own_animal_yield = np.stack(
        [np.sum(np.where(own_animal == index, own_yield, 0), axis=(-2, -1)) for index in range(len(ANIMALS))],
        axis=1,
    )
    rival_animal_yield = np.stack(
        [np.sum(np.where(rival_animal == index, rival_yield, 0), axis=(-2, -1)) for index in range(len(ANIMALS))],
        axis=1,
    )
    return {
        "money_own": np.asarray(state.money[:, own], dtype=np.float64),
        "money_rival": np.asarray(state.money[:, rival], dtype=np.float64),
        "hands_own": np.sum(np.asarray(state.unit_active[:, own, 1:]), axis=1),
        "hands_rival": np.sum(np.asarray(state.unit_active[:, rival, 1:]), axis=1),
        "unlocked_own": np.sum(own_kind != int(TileKind.LOCKED), axis=(-2, -1)),
        "unlocked_rival": np.sum(rival_kind != int(TileKind.LOCKED), axis=(-2, -1)),
        "crops_own": counts(own_crop, len(CROPS)),
        "crops_rival": counts(rival_crop, len(CROPS)),
        "animals_own": counts(own_animal, len(ANIMALS)),
        "animals_rival": counts(rival_animal, len(ANIMALS)),
        "crop_yield_own": own_crop_yield,
        "crop_yield_rival": rival_crop_yield,
        "animal_yield_own": own_animal_yield,
        "animal_yield_rival": rival_animal_yield,
        "shed_own": np.asarray(state.shed[:, own, :NUM_PRODUCTS]),
        "shed_rival": np.asarray(state.shed[:, rival, :NUM_PRODUCTS]),
        "seeds_own": np.asarray(state.seeds[:, own]),
        "seeds_rival": np.asarray(state.seeds[:, rival]),
        "market_price": np.asarray(state.market_price),
        "market_inventory": np.asarray(state.market_inventory),
        "town_count": np.asarray(state.town_count),
        "town_shops": np.asarray(state.town_shops),
    }


def extract_policy_state(carry, agent_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return route, route-lock flag, and controller-family marker."""
    batch = int(carry[0].shape[0]) if not hasattr(carry, "_fields") else None
    if agent_id == 20:
        return (
            np.asarray(jax.device_get(carry.ray_route_id)),
            np.asarray(jax.device_get(carry.ray_route_locked)),
            np.full(carry.ray_route_id.shape, -1, dtype=np.int8),
        )
    if 14 <= agent_id <= 18:
        return (
            np.asarray(jax.device_get(carry.skeleton_id)),
            np.ones(carry.skeleton_id.shape, dtype=bool),
            np.asarray(jax.device_get(carry.family)),
        )
    if agent_id == 26:
        high = np.asarray(jax.device_get(carry.expert_high))
        decided = np.asarray(jax.device_get(carry.expert_decided))
        return np.where(high, 13, 12), decided, np.full(high.shape, -1, dtype=np.int8)
    if batch is None:
        first = jax.tree_util.tree_leaves(carry)[0]
        batch = int(first.shape[0])
    return (
        np.full((batch,), -1, dtype=np.int16),
        np.zeros((batch,), dtype=bool),
        np.full((batch,), -1, dtype=np.int8),
    )


def mean_rows(values: np.ndarray, mask: np.ndarray) -> list | float:
    if not np.any(mask):
        shape = values.shape[1:]
        return 0.0 if not shape else np.zeros(shape, dtype=np.float64).tolist()
    result = np.mean(values[mask], axis=0)
    return float(result) if result.ndim == 0 else result.tolist()


def summarize_snapshots(
    snapshots: list[tuple[int, dict[str, np.ndarray]]], margins: np.ndarray
) -> list[dict]:
    groups = {
        "all": np.ones(margins.shape, dtype=bool),
        "wins": margins > 0,
        "losses": margins < 0,
    }
    rows = []
    for action_index, snapshot in snapshots:
        row = {"after_action_index": action_index, "state_step": action_index + 1, "groups": {}}
        for name, mask in groups.items():
            routes_own = snapshot["route_own"][mask].astype(int)
            routes_rival = snapshot["route_rival"][mask].astype(int)
            route_pairs: dict[str, int] = {}
            for own_route, rival_route in zip(routes_own, routes_rival, strict=True):
                key = f"{own_route}:{rival_route}"
                route_pairs[key] = route_pairs.get(key, 0) + 1
            shop_prefixes: dict[str, int] = {}
            for shops, count in zip(
                snapshot["town_shops"][mask], snapshot["town_count"][mask], strict=True
            ):
                key = ",".join(str(int(value)) for value in shops[: int(count)])
                shop_prefixes[key] = shop_prefixes.get(key, 0) + 1
            row["groups"][name] = {
                "games": int(np.sum(mask)),
                **{key: mean_rows(value, mask) for key, value in snapshot.items()},
                "route_pair_hist": route_pairs,
                "town_shop_prefix_hist": shop_prefixes,
            }
        rows.append(row)
    return rows


def serialize_public_per_game(
    snapshots: list[tuple[int, dict[str, np.ndarray]]],
    margins: np.ndarray,
    seeds: np.ndarray,
) -> list[dict]:
    """Persist decision-time features without exposing rival private inventory."""

    batch = int(len(seeds))
    rows = []
    for game_index, margin in enumerate(margins.tolist()):
        source_index = game_index % batch
        days = []
        for action_index, snapshot in snapshots:
            town_count = int(snapshot["town_count"][game_index])
            days.append(
                {
                    "after_action_index": int(action_index),
                    "state_step": int(action_index + 1),
                    "money_own": int(snapshot["money_own"][game_index]),
                    "money_rival": int(snapshot["money_rival"][game_index]),
                    "hands_own": int(snapshot["hands_own"][game_index]),
                    "hands_rival": int(snapshot["hands_rival"][game_index]),
                    "unlocked_own": int(snapshot["unlocked_own"][game_index]),
                    "unlocked_rival": int(snapshot["unlocked_rival"][game_index]),
                    "crops_own": snapshot["crops_own"][game_index].astype(int).tolist(),
                    "crops_rival": snapshot["crops_rival"][game_index].astype(int).tolist(),
                    "animals_own": snapshot["animals_own"][game_index].astype(int).tolist(),
                    "animals_rival": snapshot["animals_rival"][game_index].astype(int).tolist(),
                    "crop_yield_own": snapshot["crop_yield_own"][game_index].astype(int).tolist(),
                    "crop_yield_rival": snapshot["crop_yield_rival"][game_index].astype(int).tolist(),
                    "animal_yield_own": snapshot["animal_yield_own"][game_index].astype(int).tolist(),
                    "animal_yield_rival": snapshot["animal_yield_rival"][game_index].astype(int).tolist(),
                    "shed_own": snapshot["shed_own"][game_index].astype(int).tolist(),
                    "seeds_own": snapshot["seeds_own"][game_index].astype(int).tolist(),
                    "market_price": snapshot["market_price"][game_index].astype(int).tolist(),
                    "market_inventory": snapshot["market_inventory"][game_index].astype(int).tolist(),
                    "town_shops": snapshot["town_shops"][game_index, :town_count].astype(int).tolist(),
                    "route_own_diagnostic": int(snapshot["route_own"][game_index]),
                    "route_rival_diagnostic": int(snapshot["route_rival"][game_index]),
                }
            )
        rows.append(
            {
                "seed": int(seeds[source_index]),
                "candidate_seat": 0 if game_index < batch else 1,
                "margin": int(margin),
                "result": "win" if margin > 0 else ("tie" if margin == 0 else "loss"),
                "daily": days,
            }
        )
    return rows


def run_orientation(
    agent0: int,
    agent1: int,
    seeds: np.ndarray,
    events: Events,
    resources: dict,
) -> tuple[np.ndarray, list[tuple[int, dict[str, np.ndarray]]]]:
    states = jax.vmap(reset)(jnp.asarray(seeds))
    router = resources["router"]
    carry0 = rr.initialize_agent_carry(agent0, len(seeds), router)
    carry1 = rr.initialize_agent_carry(agent1, len(seeds), router)
    policy0 = rr.make_agent_policy(agent0, 0, **resources)
    policy1 = rr.make_agent_policy(agent1, 1, **resources)
    simulator = rr.make_simulator_step(resources["tables"])
    snapshots: list[tuple[int, dict[str, np.ndarray]]] = []
    for action_index in range(719):
        action0, carry0 = policy0(states, carry0)
        action1, carry1 = policy1(states, carry1)
        states = simulator(states, action0, action1, events)
        if action_index in SNAPSHOT_AFTER_ACTION:
            snapshot = extract_snapshot(states, own=0)
            route0, lock0, family0 = extract_policy_state(carry0, agent0)
            route1, lock1, family1 = extract_policy_state(carry1, agent1)
            snapshot.update(
                {
                    "route_own": route0,
                    "route_rival": route1,
                    "route_locked_own": lock0,
                    "route_locked_rival": lock1,
                    "route_family_own": family0,
                    "route_family_rival": family1,
                }
            )
            snapshots.append((action_index, snapshot))
    jax.block_until_ready(states.money)
    terminal = jax.device_get(states)
    if not bool(np.all(np.asarray(terminal.done))):
        raise AssertionError("not all games reached DONE")
    if int(np.sum(np.asarray(terminal.hand_cap_hits))) != 0:
        raise AssertionError("hand cap hit")
    if int(np.sum(np.asarray(terminal.market_loop_cap_hits))) != 0:
        raise AssertionError("market loop cap hit")
    if int(np.sum(np.asarray(terminal.price_lut_oob))) != 0:
        raise AssertionError("price LUT out of bounds")
    return np.asarray(terminal.money, dtype=np.int64), snapshots


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=540001)
    parser.add_argument("--seeds", type=int, default=64)
    parser.add_argument(
        "--pairs",
        default="",
        help="Optional comma-separated candidate:opponent pairs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/receipts/fc1_pair_trajectory_seed540001_n64x2_v1.json",
    )
    parser.add_argument(
        "--save-per-game",
        action="store_true",
        help="Persist daily decision features for each seed/seat game.",
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1:
        raise ValueError("seeds must be positive")

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
    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    ids = {row["name"]: index for index, row in enumerate(rr.ROSTER)}
    selected_pairs = (
        [tuple(value.split(":", 1)) for value in args.pairs.split(",") if value]
        if args.pairs
        else list(DEFAULT_PAIRS)
    )
    if any(len(pair) != 2 or pair[0] not in ids or pair[1] not in ids for pair in selected_pairs):
        raise ValueError("invalid --pairs value")

    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    pair_rows = []
    for candidate, opponent in selected_pairs:
        first_money, first_snapshots = run_orientation(
            ids[candidate], ids[opponent], seeds, events, resources
        )
        second_money, second_raw = run_orientation(
            ids[opponent], ids[candidate], seeds, events, resources
        )
        second_snapshots = [
            (step, extract_snapshot_from_canonical(snapshot))
            for step, snapshot in second_raw
        ]
        candidate_cash = np.concatenate((first_money[:, 0], second_money[:, 1]))
        opponent_cash = np.concatenate((first_money[:, 1], second_money[:, 0]))
        margins = candidate_cash - opponent_cash
        combined_snapshots = []
        for (step_a, snap_a), (step_b, snap_b) in zip(first_snapshots, second_snapshots, strict=True):
            if step_a != step_b:
                raise AssertionError("snapshot step mismatch")
            combined_snapshots.append(
                (step_a, {key: np.concatenate((snap_a[key], snap_b[key]), axis=0) for key in snap_a})
            )
        pair_row = {
                "candidate": candidate,
                "opponent": opponent,
                "games": int(margins.size),
                "wins": int(np.sum(margins > 0)),
                "ties": int(np.sum(margins == 0)),
                "losses": int(np.sum(margins < 0)),
                "score_rate": float(np.mean(margins > 0) + 0.5 * np.mean(margins == 0)),
                "mean_candidate_cash": float(np.mean(candidate_cash)),
                "mean_opponent_cash": float(np.mean(opponent_cash)),
                "mean_margin": float(np.mean(margins)),
                "median_margin": float(np.median(margins)),
                "seat0_wins": int(np.sum((first_money[:, 0] - first_money[:, 1]) > 0)),
                "seat1_wins": int(np.sum((second_money[:, 1] - second_money[:, 0]) > 0)),
                "daily": summarize_snapshots(combined_snapshots, margins),
            }
        if args.save_per_game:
            pair_row["per_game"] = serialize_public_per_game(
                combined_snapshots, margins, seeds
            )
        pair_rows.append(pair_row)
        print(
            json.dumps(
                {
                    key: pair_rows[-1][key]
                    for key in pair_rows[-1]
                    if key not in ("daily", "per_game")
                }
            ),
            flush=True,
        )

    payload = {
        "schema": "kaggriculture.fusion_champion.pair_trajectory.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "backend": jax.default_backend(),
        "device": str(jax.devices()[0]),
        "official_package_version": "1.32.7",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "seat_protocol": "same seeds with seats swapped",
        "sources": [
            {"path": str(path), "sha256": sha256(path)}
            for path in (old_bank_path, old_receipt_path, latest_bank_path, runtime_path)
        ],
        "pairs": pair_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "output": str(args.output.resolve())}), flush=True)
    return 0


def extract_snapshot_from_canonical(snapshot: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Swap own/rival labels for an orientation where candidate occupied seat 1."""
    result: dict[str, np.ndarray] = {}
    for key, value in snapshot.items():
        if key.endswith("_own"):
            result[key] = snapshot[key[:-4] + "_rival"]
        elif key.endswith("_rival"):
            result[key] = snapshot[key[:-6] + "_own"]
        else:
            result[key] = value
    return result


if __name__ == "__main__":
    raise SystemExit(main())
