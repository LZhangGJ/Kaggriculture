#!/usr/bin/env python3
"""Export the exact 66 actor-visible step-145 features from the JAX arena."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "experiments" / "route_playbook_v1" / "src",
    ROOT / "experiments" / "gold_adaptive_rule_v2" / "tools",
):
    sys.path.insert(0, str(path))

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax.constants import ANIMALS, CROPS, SHOP_NAMES, TileKind
from kaggriculture_jax.state import load_tables, reset
from route_playbook_v1.event_bank import load_route_event_bank_v1, select_route_events_v1
from export_gpu_route_context_features_v1 import load_trace_bank, make_rollout


DECISION_STEP = 145
ANCHOR_STEPS = (73, 121)
SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
TREND_INDICES = (8, 11, 12, 14, 15, 23, 24, 27, 28)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def base_feature_names() -> list[str]:
    names = [f"shop_{name}" for name in SHOPS]
    names += [f"price_{name}" for name in PRODUCTS]
    names += [f"market_inventory_{name}" for name in PRODUCTS]
    names += [f"opp_animal_{name}" for name in ANIMALS]
    names += [f"opp_crop_{name}" for name in CROPS]
    names += ["opp_unlocked", "opp_hands", "opp_money"]
    names += [f"own_animal_{name}" for name in ANIMALS]
    names += [f"own_crop_{name}" for name in CROPS]
    names += ["own_unlocked", "own_hands", "own_money"]
    if len(names) != 48:
        raise AssertionError(len(names))
    return names


BASE_NAMES = base_feature_names()
FEATURE_NAMES = BASE_NAMES + [
    f"delta_from_step{step}_{BASE_NAMES[index]}"
    for step in ANCHOR_STEPS
    for index in TREND_INDICES
]


def extract_base48(states, player: int):
    opponent = 1 - player
    shop_ids = jnp.asarray([SHOP_NAMES.index(name) for name in SHOPS], dtype=jnp.int32)
    shops = states.town_shops.astype(jnp.int32)
    shop_counts = jnp.stack([jnp.sum(shops == value, axis=1) for value in shop_ids], axis=1)

    def farm_values(which: int):
        batch_size = states.step.shape[0]
        animal = states.tile_animal[:, which].reshape(batch_size, -1)
        crop = states.tile_crop[:, which].reshape(batch_size, -1)
        kind = states.tile_kind[:, which].reshape(batch_size, -1)
        animal_counts = jnp.stack([jnp.sum(animal == index, axis=1) for index in range(len(ANIMALS))], axis=1)
        crop_counts = jnp.stack([jnp.sum(crop == index, axis=1) for index in range(len(CROPS))], axis=1)
        unlocked = jnp.sum(kind != int(TileKind.LOCKED), axis=1, keepdims=True)
        hands = jnp.sum(states.unit_active[:, which, 1:].astype(jnp.int32), axis=1, keepdims=True)
        money = states.money[:, which : which + 1]
        return jnp.concatenate([animal_counts, crop_counts, unlocked, hands, money], axis=1)

    result = jnp.concatenate(
        [shop_counts, states.market_price, states.market_inventory, farm_values(opponent), farm_values(player)],
        axis=1,
    ).astype(jnp.float32)
    if result.shape[1] != 48:
        raise AssertionError(result.shape)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bank", type=Path, required=True)
    parser.add_argument("--event-bank", type=Path, required=True)
    parser.add_argument("--candidate-id", type=int, default=0)
    parser.add_argument("--opponent-ids", type=int, nargs="+", required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int32)
    opponent_ids = np.repeat(np.asarray(args.opponent_ids, dtype=np.int32), len(seeds))
    batch_seeds = np.tile(seeds, len(args.opponent_ids))
    event_seeds, event_bank = load_route_event_bank_v1(args.event_bank)
    events = select_route_events_v1(batch_seeds.tolist(), event_seeds, event_bank)
    initial = jax.vmap(reset)(jnp.asarray(batch_seeds))
    candidate_ids = jnp.full(opponent_ids.shape, args.candidate_id, dtype=jnp.int32)
    opponent_ids_jax = jnp.asarray(opponent_ids)
    bank = load_trace_bank(args.trace_bank)
    tables = load_tables()
    features = []
    for seat in (0, 1):
        base_by_step = {}
        for step in (*ANCHOR_STEPS, DECISION_STEP):
            state = make_rollout(bank, tables, seat, step)(initial, events, candidate_ids, opponent_ids_jax)
            values = extract_base48(state, seat)
            jax.block_until_ready(values)
            base_by_step[step] = values
        current = base_by_step[DECISION_STEP]
        trend_index = jnp.asarray(TREND_INDICES, dtype=jnp.int32)
        extended = jnp.concatenate(
            [current, *(current[:, trend_index] - base_by_step[step][:, trend_index] for step in ANCHOR_STEPS)],
            axis=1,
        )
        jax.block_until_ready(extended)
        features.append(np.asarray(extended, dtype=np.float32))
    feature_array = np.stack(features, axis=0).reshape(2, len(args.opponent_ids), len(seeds), 66)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=feature_array,
        feature_names=np.asarray(FEATURE_NAMES),
        opponent_ids=np.asarray(args.opponent_ids, dtype=np.int16),
        seeds=seeds,
        candidate_id=np.asarray(args.candidate_id, dtype=np.int16),
        decision_step=np.asarray(DECISION_STEP, dtype=np.int16),
    )
    receipt = {
        "schema": "kawashigi-jax-context-features66-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "truth_boundary": "JAX GPU screening only; official Python 1.32.7 remains the referee",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "trace_bank": str(args.trace_bank.resolve()),
        "trace_bank_sha256": sha256(args.trace_bank),
        "event_bank": str(args.event_bank.resolve()),
        "event_bank_sha256": sha256(args.event_bank),
        "candidate_id": args.candidate_id,
        "opponent_ids": args.opponent_ids,
        "seed_start": args.seed_start,
        "seed_count": args.seed_count,
        "decision_step": DECISION_STEP,
        "anchor_steps": list(ANCHOR_STEPS),
        "feature_names": FEATURE_NAMES,
        "feature_shape": list(feature_array.shape),
        "information_boundary": ["actor-visible public state only", "no opponent identity", "no seed", "no future events", "no terminal label"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output),
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "backend", "feature_shape", "output")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
