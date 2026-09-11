#!/usr/bin/env python3
"""GPU counterfactual panel with six public-state dynamic opponent routers."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import NamedTuple

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "gpu_sim" / "src",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "experiments" / "route_playbook_v1" / "src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.constants import (  # noqa: E402
    ANIMAL_STRUCTURE,
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    NUM_ANIMALS,
    NUM_CROPS,
    NUM_PRODUCTS,
    NUM_SHED_ITEMS,
    SHED_ACCESS,
    SHED_CAPACITY,
    SHOP_NAMES,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from route_playbook_v1.trace_core import (  # noqa: E402
    CalibrationTraceV1,
    _append_terminal_liquidation_v1,
    initialize_trace_player_carry_v1,
    skeleton_player_raw_action_v1,
    skeleton_player_action_v1,
)
from strategic_v5.boatlee_v16_gpu import (  # noqa: E402
    BoatleePlayerCarryV1,
    BoatleeTraceV1,
    _rank_sell_slots,
    boatlee_player_action_v1,
    initialize_boatlee_player_carry_v1,
    load_boatlee_trace_v1,
)
from strategic_v5.c95_gpu import (  # noqa: E402
    C95PlayerCarryV1,
    c95_player_action_v1,
    initialize_c95_player_carry_v1,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    PublicG02CarryV1,
    initialize_public_g02_carry_v1,
    public_g02_player_action_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import public_v25_player_action_v1  # noqa: E402
from strategic_v5.public_v14_gpu import public_v14_player_action_v1  # noqa: E402
from strategic_v5.public_v21_gpu import public_v21_player_action_v1  # noqa: E402
from strategic_v5.public_v13_r3_gpu import public_v13_r3_player_action_v1  # noqa: E402
from strategic_v5.public_c68_gpu import public_c68_player_action_v1  # noqa: E402
from strategic_v5.public_four_hire_gpu import (  # noqa: E402
    FourHireCarryV1,
    initialize_four_hire_carry_v1,
    public_four_hire_player_action_v1,
)
from strategic_v5.public_tran_cashflow_gpu import (  # noqa: E402
    TranCashflowCarryV1,
    initialize_tran_cashflow_carry_v1,
    public_tran_cashflow_player_action_v1,
)
from strategic_v5.public_bruce_route1_gpu import (  # noqa: E402
    BruceRoute1CarryV1,
    initialize_bruce_route1_carry_v1,
    public_bruce_route1_player_action_v1,
)
from strategic_v5.public_v19_control_gpu import (  # noqa: E402
    public_v19_control_player_action_v1,
)
from strategic_v5.public_v18_closed_loop_gpu import (  # noqa: E402
    V18ClosedLoopCarryV1,
    initialize_v18_closed_loop_carry_v1,
    public_v18_closed_loop_player_action_v1,
)


SHOPS = ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "YARN_STORE", "ICE_CREAM_SHOP", "PET_CAFE", "SMOOTHIE_SHOP", "FARMERS_MARKET")
PRODUCTS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER")
TREND_INDICES = (8, 11, 12, 14, 15, 23, 24, 27, 28)
ANCHOR_STEPS = (73, 121)
DECISION_STEP = 145
_SHED_ACCESS = jnp.asarray(SHED_ACCESS, dtype=jnp.int8)
_ANIMAL_STRUCTURE = jnp.asarray(ANIMAL_STRUCTURE, dtype=jnp.int8)
_PRT6_PREMIUM_IDS = (
    PRODUCTS.index("STRAWBERRY"),
    PRODUCTS.index("MELON"),
    PRODUCTS.index("MILK"),
    PRODUCTS.index("WOOL"),
)
_PRT6_PREEMPT_HORIZONS = (6, 5, 4, 3, 2, 1)


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
    return names


BASE_NAMES = base_feature_names()
FEATURE_NAMES = BASE_NAMES + [
    f"delta_from_step{step}_{BASE_NAMES[index]}"
    for step in ANCHOR_STEPS
    for index in TREND_INDICES
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load_bank(path: Path) -> CalibrationTraceV1:
    with np.load(path, allow_pickle=False) as data:
        return CalibrationTraceV1(*(jnp.asarray(data[field]) for field in CalibrationTraceV1._fields))


def _farm_values(states, which: int):
    batch = states.step.shape[0]
    animal = states.tile_animal[:, which].reshape(batch, -1)
    crop = states.tile_crop[:, which].reshape(batch, -1)
    kind = states.tile_kind[:, which].reshape(batch, -1)
    animal_counts = jnp.stack([jnp.sum(animal == index, axis=1) for index in range(len(ANIMALS))], axis=1)
    crop_counts = jnp.stack([jnp.sum(crop == index, axis=1) for index in range(len(CROPS))], axis=1)
    unlocked = jnp.sum(kind != int(TileKind.LOCKED), axis=1, keepdims=True)
    hands = jnp.sum(states.unit_active[:, which, 1:].astype(jnp.int32), axis=1, keepdims=True)
    money = states.money[:, which : which + 1]
    return jnp.concatenate((animal_counts, crop_counts, unlocked, hands, money), axis=1)


def extract_base48(states, player: int):
    opponent = 1 - player
    shop_ids = jnp.asarray([SHOP_NAMES.index(name) for name in SHOPS], dtype=jnp.int32)
    shops = states.town_shops.astype(jnp.int32)
    active = jnp.arange(shops.shape[1])[None, :] < states.town_count[:, None]
    counts = jnp.stack([jnp.sum(active & (shops == value), axis=1) for value in shop_ids], axis=1)
    return jnp.concatenate((counts, states.market_price, states.market_inventory, _farm_values(states, opponent), _farm_values(states, player)), axis=1).astype(jnp.float32)


def extract_tree56(states, player: int):
    opponent = 1 - player
    shop_ids = jnp.asarray([SHOP_NAMES.index(name) for name in SHOPS], dtype=jnp.int32)
    first = states.town_shops[:, 0].astype(jnp.int32)
    second = states.town_shops[:, 1].astype(jnp.int32)
    ordered = (
        jnp.stack([first == value for value in shop_ids], axis=1),
        jnp.stack([second == value for value in shop_ids], axis=1),
    )
    return jnp.concatenate((*ordered, states.market_price, states.market_inventory, _farm_values(states, opponent), _farm_values(states, player)), axis=1).astype(jnp.float32)


class RouterArrays(NamedTuple):
    kind: jax.Array
    default_id: jax.Array
    decision: jax.Array
    route_count: jax.Array
    route_ids: jax.Array
    centroid: jax.Array
    mean: jax.Array
    scale: jax.Array
    public04_ids: jax.Array
    v8_prt_ids: jax.Array
    v8_prt_centroid: jax.Array
    v8_prt_mean: jax.Array
    v8_prt_scale: jax.Array
    v8_prt_count: jax.Array
    v8_prt_decision: jax.Array
    v8_alt_ids: jax.Array
    v8_alt_centroid: jax.Array
    v8_alt_mean: jax.Array
    v8_alt_scale: jax.Array
    v8_alt_count: jax.Array
    v8_alt_decision: jax.Array
    tree_feature: jax.Array
    tree_threshold: jax.Array
    tree_left: jax.Array
    tree_right: jax.Array
    tree_alternate: jax.Array
    prt6_ids: jax.Array
    prt6_tree_feature: jax.Array
    prt6_tree_threshold: jax.Array
    prt6_tree_left: jax.Array
    prt6_tree_right: jax.Array
    prt6_tree_route: jax.Array
    prt6_opponent_id: jax.Array
    public04_opponent_id: jax.Array
    v8_opponent_id: jax.Array
    c95_opponent_id: jax.Array
    c95_skeleton_id: jax.Array
    public_g02_opponent_id: jax.Array
    public_g02_skeleton_id: jax.Array
    public_g02_rival_schedule: jax.Array
    public_g04_opponent_id: jax.Array
    public_g04_skeleton_id: jax.Array
    public_v25_opponent_id: jax.Array
    public_v25_skeleton_id: jax.Array
    public_v14_opponent_id: jax.Array
    public_v14_skeleton_id: jax.Array
    public_v21_opponent_id: jax.Array
    public_v21_skeleton_id: jax.Array
    public_v21_prototype_signature: jax.Array
    public_v21_prototype_sales: jax.Array
    public_v13_opponent_id: jax.Array
    public_v13_skeleton_id: jax.Array
    public_v13_hazard_enabled: jax.Array
    public_v13_hazard_cap: jax.Array
    public_c68_opponent_id: jax.Array
    public_c68_skeleton_id: jax.Array
    public_four_hire_opponent_id: jax.Array
    public_four_hire_kaito_skeleton_id: jax.Array
    public_four_hire_ray_skeleton_id: jax.Array
    public_tran_cashflow_opponent_id: jax.Array
    public_tran_cashflow_skeleton_id: jax.Array
    public_bruce_route1_opponent_id: jax.Array
    public_bruce_route1_skeleton_id: jax.Array
    public_v19_control_opponent_id: jax.Array
    public_v19_control_skeleton_id: jax.Array
    public_v18_closed_loop_opponent_id: jax.Array
    public_v18_board_skeleton_id: jax.Array
    public_v18_expert_skeleton_ids: jax.Array
    public_v18_feature_scale: jax.Array
    public_v18_market_bias_by_seat: jax.Array
    public_v18_prototypes_by_day: jax.Array
    public_v18_distance_strength: jax.Array
    public_v18_stay_bonus: jax.Array
    public_g05_opponent_id: jax.Array
    public_g05_seat_skeleton_ids: jax.Array


def _pad_routes(spec: dict, max_routes: int) -> tuple[list[int], list[list[float]]]:
    ids = list(spec.get("skeleton_ids", []))
    centroids = [list(row) for row in spec.get("centroids", [])]
    ids += [ids[0] if ids else 0] * (max_routes - len(ids))
    centroids += [[0.0] * 48 for _ in range(max_routes - len(centroids))]
    return ids[:max_routes], centroids[:max_routes]


def build_router_arrays(receipt: dict) -> RouterArrays:
    max_routes = 6
    kind_map = {
        "fixed": 0,
        "shop_dominance": 1,
        "centroid": 2,
        "nested_tree": 3,
        "boatlee_v16_exact": 4,
        "prt_tree": 5,
        "c95_exact": 6,
        "public_g02_exact": 7,
        "public_g04_exact": 8,
        "public_v25_exact": 9,
        "public_v14_exact": 10,
        "public_v21_exact": 11,
        "public_v13_r3_exact": 12,
        "public_c68_exact": 13,
        "public_four_hire_exact": 14,
        "public_tran_cashflow_exact": 15,
        "public_bruce_route1_exact": 16,
        "public_v19_control_exact": 17,
        "public_v18_closed_loop_exact": 18,
        "public_g05_trace_proxy": 19,
    }
    kinds, defaults, decisions, counts, ids, centroids, means, scales = [], [], [], [], [], [], [], []
    for spec in receipt["opponents"]:
        route_ids, route_centroids = _pad_routes(spec, max_routes)
        kinds.append(kind_map[spec["kind"]])
        defaults.append(spec["default_skeleton_id"])
        decisions.append(spec.get("decision_step", -1))
        counts.append(len(spec.get("skeleton_ids", [])))
        ids.append(route_ids)
        centroids.append(route_centroids)
        means.append(spec.get("feature_mean", [0.0] * 48))
        scales.append(spec.get("feature_scale", [1.0] * 48))
    by_name = {row["name"]: row for row in receipt["opponents"]}
    public04 = by_name["public04_read_market"]
    v8 = by_name["public_opening_router_v8"]
    prt_ids, prt_centroids = _pad_routes(v8["prt"], max_routes)
    alt_ids, alt_centroids = _pad_routes(v8["alternate"], max_routes)
    tree = v8["tree"]
    prt6 = by_name.get("local_prt_v6")
    if prt6 is None:
        prt6_ids = [0] * max_routes
        prt6_tree = [{"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "route": 0}]
        prt6_opponent_id = -1
    else:
        prt6_ids, _ = _pad_routes(prt6, max_routes)
        prt6_tree = prt6["tree"]
        prt6_opponent_id = int(prt6["opponent_id"])
    c95 = by_name.get("public_g07_c95")
    if c95 is None:
        c95_opponent_id = -1
        c95_skeleton_id = 0
    else:
        c95_opponent_id = int(c95["opponent_id"])
        c95_skeleton_id = int(c95["default_skeleton_id"])
    public_g02 = by_name.get("public14_rank_agent_v17")
    if public_g02 is None:
        public_g02_opponent_id = -1
        public_g02_skeleton_id = 0
        public_g02_rival_schedule = [[0] * 4 for _ in range(719)]
    else:
        public_g02_opponent_id = int(public_g02["opponent_id"])
        public_g02_skeleton_id = int(public_g02["default_skeleton_id"])
        public_g02_rival_schedule = public_g02.get(
            "rival_schedule", [[0] * 4 for _ in range(719)]
        )
    public_g04 = by_name.get("public06_soil_rain")
    if public_g04 is None:
        public_g04_opponent_id = -1
        public_g04_skeleton_id = 0
    else:
        public_g04_opponent_id = int(public_g04["opponent_id"])
        public_g04_skeleton_id = int(public_g04["default_skeleton_id"])
    public_v25 = by_name.get("public_g06_v25")
    if public_v25 is None:
        public_v25_opponent_id = -1
        public_v25_skeleton_id = 0
    else:
        public_v25_opponent_id = int(public_v25["opponent_id"])
        public_v25_skeleton_id = int(public_v25["default_skeleton_id"])
    public_v14 = by_name.get("public_g08_v14")
    if public_v14 is None:
        public_v14_opponent_id = -1
        public_v14_skeleton_id = 0
    else:
        public_v14_opponent_id = int(public_v14["opponent_id"])
        public_v14_skeleton_id = int(public_v14["default_skeleton_id"])
    public_v21 = by_name.get("public_g11_v21")
    if public_v21 is None:
        public_v21_opponent_id = -1
        public_v21_skeleton_id = 0
        public_v21_prototype_signature = [[[0] * 47 for _ in range(719)]]
        public_v21_prototype_sales = [[[False] * NUM_PRODUCTS for _ in range(719)]]
    else:
        public_v21_opponent_id = int(public_v21["opponent_id"])
        public_v21_skeleton_id = int(public_v21["default_skeleton_id"])
        public_v21_prototype_signature = public_v21["prototype_signature"]
        public_v21_prototype_sales = public_v21["prototype_sales"]
    public_v13 = by_name.get("public_g12_v13_r3")
    if public_v13 is None:
        public_v13_opponent_id = -1
        public_v13_skeleton_id = 0
        public_v13_hazard_enabled = [[False] * 4 for _ in range(719)]
        public_v13_hazard_cap = [[0] * 4 for _ in range(719)]
    else:
        public_v13_opponent_id = int(public_v13["opponent_id"])
        public_v13_skeleton_id = int(public_v13["default_skeleton_id"])
        public_v13_hazard_enabled = public_v13["hazard_enabled"]
        public_v13_hazard_cap = public_v13["hazard_cap"]
    public_c68 = by_name.get("public_g09_c68_thunder")
    if public_c68 is None:
        public_c68_opponent_id = -1
        public_c68_skeleton_id = 0
    else:
        public_c68_opponent_id = int(public_c68["opponent_id"])
        public_c68_skeleton_id = int(public_c68["default_skeleton_id"])
    public_four_hire = by_name.get("public_g10_four_hire")
    if public_four_hire is None:
        public_four_hire_opponent_id = -1
        public_four_hire_kaito_skeleton_id = 0
        public_four_hire_ray_skeleton_id = 0
    else:
        public_four_hire_opponent_id = int(public_four_hire["opponent_id"])
        public_four_hire_kaito_skeleton_id = int(
            public_four_hire["kaito_skeleton_id"]
        )
        public_four_hire_ray_skeleton_id = int(
            public_four_hire["ray_skeleton_id"]
        )
    public_tran_cashflow = by_name.get("public_g16_tran_cashflow")
    if public_tran_cashflow is None:
        public_tran_cashflow_opponent_id = -1
        public_tran_cashflow_skeleton_id = 0
    else:
        public_tran_cashflow_opponent_id = int(
            public_tran_cashflow["opponent_id"]
        )
        public_tran_cashflow_skeleton_id = int(
            public_tran_cashflow["default_skeleton_id"]
        )
    public_bruce_route1 = by_name.get("public_g13_bruce_route1")
    if public_bruce_route1 is None:
        public_bruce_route1_opponent_id = -1
        public_bruce_route1_skeleton_id = 0
    else:
        public_bruce_route1_opponent_id = int(
            public_bruce_route1["opponent_id"]
        )
        public_bruce_route1_skeleton_id = int(
            public_bruce_route1["default_skeleton_id"]
        )
    public_v19_control = by_name.get("public_g14_v19_control")
    if public_v19_control is None:
        public_v19_control_opponent_id = -1
        public_v19_control_skeleton_id = 0
    else:
        public_v19_control_opponent_id = int(
            public_v19_control["opponent_id"]
        )
        public_v19_control_skeleton_id = int(
            public_v19_control["default_skeleton_id"]
        )
    public_v18_closed_loop = by_name.get("public_g15_v18_closed_loop")
    if public_v18_closed_loop is None:
        public_v18_closed_loop_opponent_id = -1
        public_v18_board_skeleton_id = 0
        public_v18_expert_skeleton_ids = [0] * 4
        public_v18_feature_scale = [1.0] * 29
        public_v18_market_bias_by_seat = [[0.0] * 4 for _ in range(2)]
        public_v18_prototypes_by_day = [
            [[0.0] * 29 for _ in range(30)] for _ in range(4)
        ]
        public_v18_distance_strength = 0.0
        public_v18_stay_bonus = 0.0
    else:
        public_v18_closed_loop_opponent_id = int(
            public_v18_closed_loop["opponent_id"]
        )
        public_v18_board_skeleton_id = int(
            public_v18_closed_loop["board_skeleton_id"]
        )
        public_v18_expert_skeleton_ids = public_v18_closed_loop["skeleton_ids"]
        public_v18_feature_scale = public_v18_closed_loop["v18_feature_scale"]
        public_v18_market_bias_by_seat = public_v18_closed_loop[
            "market_bias_by_seat"
        ]
        public_v18_prototypes_by_day = public_v18_closed_loop[
            "prototypes_by_day"
        ]
        public_v18_distance_strength = float(
            public_v18_closed_loop["distance_strength"]
        )
        public_v18_stay_bonus = float(public_v18_closed_loop["stay_bonus"])
    public_g05 = by_name.get("public_g05_structured_econ")
    if public_g05 is None:
        public_g05_opponent_id = -1
        public_g05_seat_skeleton_ids = [0, 0]
    else:
        public_g05_opponent_id = int(public_g05["opponent_id"])
        public_g05_seat_skeleton_ids = public_g05["seat_skeleton_ids"]
    return RouterArrays(
        kind=jnp.asarray(kinds, dtype=jnp.int8),
        default_id=jnp.asarray(defaults, dtype=jnp.int32),
        decision=jnp.asarray(decisions, dtype=jnp.int16),
        route_count=jnp.asarray(counts, dtype=jnp.int8),
        route_ids=jnp.asarray(ids, dtype=jnp.int32),
        centroid=jnp.asarray(centroids, dtype=jnp.float32),
        mean=jnp.asarray(means, dtype=jnp.float32),
        scale=jnp.asarray(scales, dtype=jnp.float32),
        public04_ids=jnp.asarray(public04["skeleton_ids"], dtype=jnp.int32),
        v8_prt_ids=jnp.asarray(prt_ids, dtype=jnp.int32),
        v8_prt_centroid=jnp.asarray(prt_centroids, dtype=jnp.float32),
        v8_prt_mean=jnp.asarray(v8["prt"]["feature_mean"], dtype=jnp.float32),
        v8_prt_scale=jnp.asarray(v8["prt"]["feature_scale"], dtype=jnp.float32),
        v8_prt_count=jnp.asarray(len(v8["prt"]["skeleton_ids"]), dtype=jnp.int8),
        v8_prt_decision=jnp.asarray(v8["prt"]["decision_step"], dtype=jnp.int16),
        v8_alt_ids=jnp.asarray(alt_ids, dtype=jnp.int32),
        v8_alt_centroid=jnp.asarray(alt_centroids, dtype=jnp.float32),
        v8_alt_mean=jnp.asarray(v8["alternate"]["feature_mean"], dtype=jnp.float32),
        v8_alt_scale=jnp.asarray(v8["alternate"]["feature_scale"], dtype=jnp.float32),
        v8_alt_count=jnp.asarray(len(v8["alternate"]["skeleton_ids"]), dtype=jnp.int8),
        v8_alt_decision=jnp.asarray(v8["alternate"]["decision_step"], dtype=jnp.int16),
        tree_feature=jnp.asarray([row["feature"] for row in tree], dtype=jnp.int16),
        tree_threshold=jnp.asarray([row["threshold"] for row in tree], dtype=jnp.float32),
        tree_left=jnp.asarray([row["left"] for row in tree], dtype=jnp.int16),
        tree_right=jnp.asarray([row["right"] for row in tree], dtype=jnp.int16),
        tree_alternate=jnp.asarray([row.get("family") == v8["alternate_label"] for row in tree], dtype=jnp.bool_),
        prt6_ids=jnp.asarray(prt6_ids, dtype=jnp.int32),
        prt6_tree_feature=jnp.asarray([row["feature"] for row in prt6_tree], dtype=jnp.int16),
        prt6_tree_threshold=jnp.asarray([row["threshold"] for row in prt6_tree], dtype=jnp.float32),
        prt6_tree_left=jnp.asarray([row["left"] for row in prt6_tree], dtype=jnp.int16),
        prt6_tree_right=jnp.asarray([row["right"] for row in prt6_tree], dtype=jnp.int16),
        prt6_tree_route=jnp.asarray([row.get("route", 0) for row in prt6_tree], dtype=jnp.int16),
        prt6_opponent_id=jnp.asarray(prt6_opponent_id, dtype=jnp.int16),
        public04_opponent_id=jnp.asarray(public04["opponent_id"], dtype=jnp.int16),
        v8_opponent_id=jnp.asarray(v8["opponent_id"], dtype=jnp.int16),
        c95_opponent_id=jnp.asarray(c95_opponent_id, dtype=jnp.int16),
        c95_skeleton_id=jnp.asarray(c95_skeleton_id, dtype=jnp.int32),
        public_g02_opponent_id=jnp.asarray(public_g02_opponent_id, dtype=jnp.int16),
        public_g02_skeleton_id=jnp.asarray(public_g02_skeleton_id, dtype=jnp.int32),
        public_g02_rival_schedule=jnp.asarray(public_g02_rival_schedule, dtype=jnp.int16),
        public_g04_opponent_id=jnp.asarray(public_g04_opponent_id, dtype=jnp.int16),
        public_g04_skeleton_id=jnp.asarray(public_g04_skeleton_id, dtype=jnp.int32),
        public_v25_opponent_id=jnp.asarray(public_v25_opponent_id, dtype=jnp.int16),
        public_v25_skeleton_id=jnp.asarray(public_v25_skeleton_id, dtype=jnp.int32),
        public_v14_opponent_id=jnp.asarray(public_v14_opponent_id, dtype=jnp.int16),
        public_v14_skeleton_id=jnp.asarray(public_v14_skeleton_id, dtype=jnp.int32),
        public_v21_opponent_id=jnp.asarray(public_v21_opponent_id, dtype=jnp.int16),
        public_v21_skeleton_id=jnp.asarray(public_v21_skeleton_id, dtype=jnp.int32),
        public_v21_prototype_signature=jnp.asarray(public_v21_prototype_signature, dtype=jnp.int16),
        public_v21_prototype_sales=jnp.asarray(public_v21_prototype_sales, dtype=jnp.bool_),
        public_v13_opponent_id=jnp.asarray(public_v13_opponent_id, dtype=jnp.int16),
        public_v13_skeleton_id=jnp.asarray(public_v13_skeleton_id, dtype=jnp.int32),
        public_v13_hazard_enabled=jnp.asarray(public_v13_hazard_enabled, dtype=jnp.bool_),
        public_v13_hazard_cap=jnp.asarray(public_v13_hazard_cap, dtype=jnp.int16),
        public_c68_opponent_id=jnp.asarray(public_c68_opponent_id, dtype=jnp.int16),
        public_c68_skeleton_id=jnp.asarray(public_c68_skeleton_id, dtype=jnp.int32),
        public_four_hire_opponent_id=jnp.asarray(
            public_four_hire_opponent_id, dtype=jnp.int16
        ),
        public_four_hire_kaito_skeleton_id=jnp.asarray(
            public_four_hire_kaito_skeleton_id, dtype=jnp.int32
        ),
        public_four_hire_ray_skeleton_id=jnp.asarray(
            public_four_hire_ray_skeleton_id, dtype=jnp.int32
        ),
        public_tran_cashflow_opponent_id=jnp.asarray(
            public_tran_cashflow_opponent_id, dtype=jnp.int16
        ),
        public_tran_cashflow_skeleton_id=jnp.asarray(
            public_tran_cashflow_skeleton_id, dtype=jnp.int32
        ),
        public_bruce_route1_opponent_id=jnp.asarray(
            public_bruce_route1_opponent_id, dtype=jnp.int16
        ),
        public_bruce_route1_skeleton_id=jnp.asarray(
            public_bruce_route1_skeleton_id, dtype=jnp.int32
        ),
        public_v19_control_opponent_id=jnp.asarray(
            public_v19_control_opponent_id, dtype=jnp.int16
        ),
        public_v19_control_skeleton_id=jnp.asarray(
            public_v19_control_skeleton_id, dtype=jnp.int32
        ),
        public_v18_closed_loop_opponent_id=jnp.asarray(
            public_v18_closed_loop_opponent_id, dtype=jnp.int16
        ),
        public_v18_board_skeleton_id=jnp.asarray(
            public_v18_board_skeleton_id, dtype=jnp.int32
        ),
        public_v18_expert_skeleton_ids=jnp.asarray(
            public_v18_expert_skeleton_ids, dtype=jnp.int32
        ),
        public_v18_feature_scale=jnp.asarray(
            public_v18_feature_scale, dtype=jnp.float32
        ),
        public_v18_market_bias_by_seat=jnp.asarray(
            public_v18_market_bias_by_seat, dtype=jnp.float32
        ),
        public_v18_prototypes_by_day=jnp.asarray(
            public_v18_prototypes_by_day, dtype=jnp.float32
        ),
        public_v18_distance_strength=jnp.asarray(
            public_v18_distance_strength, dtype=jnp.float32
        ),
        public_v18_stay_bonus=jnp.asarray(
            public_v18_stay_bonus, dtype=jnp.float32
        ),
        public_g05_opponent_id=jnp.asarray(
            public_g05_opponent_id, dtype=jnp.int16
        ),
        public_g05_seat_skeleton_ids=jnp.asarray(
            public_g05_seat_skeleton_ids, dtype=jnp.int32
        ),
    )


class DynamicCarry(NamedTuple):
    trace: object
    skeleton_id: jax.Array
    family: jax.Array
    prt6_due_step: jax.Array
    prt6_due: jax.Array


class FullOpponentCarry(NamedTuple):
    dynamic: DynamicCarry
    boatlee: BoatleePlayerCarryV1
    c95: C95PlayerCarryV1
    public_g02: PublicG02CarryV1
    public_g04: PublicG02CarryV1
    public_v25: PublicG02CarryV1
    public_v14: PublicG02CarryV1
    public_v21: PublicG02CarryV1
    public_v13: PublicG02CarryV1
    public_c68: PublicG02CarryV1
    public_four_hire: FourHireCarryV1
    public_tran_cashflow: TranCashflowCarryV1
    public_bruce_route1: BruceRoute1CarryV1
    public_v18_closed_loop: V18ClosedLoopCarryV1


def initialize_dynamic_carry(batch_size: int, opponent_ids: jax.Array, router: RouterArrays) -> DynamicCarry:
    return DynamicCarry(
        trace=initialize_trace_player_carry_v1(batch_size),
        skeleton_id=router.default_id[opponent_ids],
        family=jnp.full((batch_size,), -1, dtype=jnp.int8),
        prt6_due_step=jnp.full((batch_size,), -1, dtype=jnp.int16),
        prt6_due=jnp.zeros((batch_size, NUM_PRODUCTS), dtype=jnp.int16),
    )


def initialize_full_opponent_carry(
    batch_size: int, opponent_ids: jax.Array, router: RouterArrays
) -> FullOpponentCarry:
    return FullOpponentCarry(
        dynamic=initialize_dynamic_carry(batch_size, opponent_ids, router),
        boatlee=initialize_boatlee_player_carry_v1(batch_size),
        c95=initialize_c95_player_carry_v1(batch_size),
        public_g02=initialize_public_g02_carry_v1(batch_size),
        public_g04=initialize_public_g02_carry_v1(batch_size),
        public_v25=initialize_public_g02_carry_v1(batch_size),
        public_v14=initialize_public_g02_carry_v1(batch_size),
        public_v21=initialize_public_g02_carry_v1(batch_size),
        public_v13=initialize_public_g02_carry_v1(batch_size),
        public_c68=initialize_public_g02_carry_v1(batch_size),
        public_four_hire=initialize_four_hire_carry_v1(batch_size),
        public_tran_cashflow=initialize_tran_cashflow_carry_v1(batch_size),
        public_bruce_route1=initialize_bruce_route1_carry_v1(batch_size),
        public_v18_closed_loop=initialize_v18_closed_loop_carry_v1(batch_size),
    )


def choose_centroid(features, centroid, mean, scale, route_ids, count):
    normalized = (features - mean) / scale
    distance = jnp.sum((normalized[:, None, :] - centroid) ** 2, axis=2)
    valid = jnp.arange(centroid.shape[1])[None, :] < count[:, None]
    choice = jnp.argmin(jnp.where(valid, distance, jnp.inf), axis=1)
    return jnp.take_along_axis(route_ids, choice[:, None], axis=1)[:, 0]


def evaluate_v8_tree(features, router: RouterArrays):
    index = jnp.zeros((features.shape[0],), dtype=jnp.int16)
    batch = jnp.arange(features.shape[0])
    for _ in range(8):
        feature = router.tree_feature[index]
        leaf = feature < 0
        safe = jnp.clip(feature, 0, features.shape[1] - 1)
        value = features[batch, safe]
        following = jnp.where(value <= router.tree_threshold[index], router.tree_left[index], router.tree_right[index])
        index = jnp.where(leaf, index, following).astype(jnp.int16)
    return router.tree_alternate[index]


def evaluate_prt6_tree(features, router: RouterArrays):
    index = jnp.zeros((features.shape[0],), dtype=jnp.int16)
    batch = jnp.arange(features.shape[0])
    for _ in range(8):
        feature = router.prt6_tree_feature[index]
        leaf = feature < 0
        safe = jnp.clip(feature, 0, features.shape[1] - 1)
        value = features[batch, safe]
        following = jnp.where(
            value <= router.prt6_tree_threshold[index],
            router.prt6_tree_left[index],
            router.prt6_tree_right[index],
        )
        index = jnp.where(leaf, index, following).astype(jnp.int16)
    return router.prt6_tree_route[index]


def _compact_market(action: Action, keep: jax.Array) -> Action:
    """Stable-compacts active market orders, matching Python list deletion."""

    slot = jnp.arange(MAX_MARKET_ORDERS)[None, :]
    keep = keep & (slot < action.market_count[:, None])
    key = jnp.where(keep, slot, slot + MAX_MARKET_ORDERS)
    order = jnp.argsort(key, axis=1, stable=True)
    op = jnp.take_along_axis(action.market_op, order, axis=1)
    item = jnp.take_along_axis(action.market_item, order, axis=1)
    amount = jnp.take_along_axis(action.market_amount, order, axis=1)
    count = jnp.sum(keep, axis=1).astype(jnp.int8)
    active = slot < count[:, None]
    return action._replace(
        market_op=jnp.where(active, op, MarketOp.NONE).astype(jnp.int8),
        market_item=jnp.where(active, item, -1).astype(jnp.int8),
        market_amount=jnp.where(active, amount, 0).astype(jnp.int32),
        market_count=count,
    )


def _prt6_repay_shift(
    action: Action,
    step: jax.Array,
    due_step: jax.Array,
    due: jax.Array,
    enabled: jax.Array,
) -> tuple[Action, jax.Array, jax.Array]:
    """Remove an earlier borrowed sale from its original route step."""

    due_now = enabled & (due_step.astype(jnp.int32) == step.astype(jnp.int32))
    due_late = enabled & (due_step >= 0) & (
        due_step.astype(jnp.int32) < step.astype(jnp.int32)
    )
    active = (
        jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    )
    amount = action.market_amount
    remove = jnp.zeros_like(active)
    for product_id in _PRT6_PREMIUM_IDS:
        matching = (
            active
            & (action.market_op == MarketOp.SELL)
            & (action.market_item == product_id)
        )
        requested = jnp.where(matching, jnp.maximum(amount, 0), 0)
        prefix = jnp.cumsum(requested, axis=1) - requested
        reduction = jnp.minimum(
            requested,
            jnp.maximum(
                due[:, product_id].astype(jnp.int32)[:, None] - prefix, 0
            ),
        )
        repaid = amount - jnp.where(due_now[:, None], reduction, 0)
        remove = remove | (due_now[:, None] & matching & (repaid <= 0))
        amount = jnp.where(due_now[:, None] & matching, repaid, amount)

    action = action._replace(market_amount=amount.astype(jnp.int32))
    action = _compact_market(action, ~remove)
    clear = due_now | due_late
    return (
        action,
        jnp.where(clear, -1, due_step).astype(jnp.int16),
        jnp.where(clear[:, None], 0, due).astype(jnp.int16),
    )


def _prt6_route_distance(states) -> jax.Array:
    """Exact public-signature distance used by the Python PRT V6 gate."""

    crop = states.tile_crop.astype(jnp.int32)
    animal = states.tile_animal.astype(jnp.int32)
    crop_counts = (
        jax.nn.one_hot(
            jnp.clip(crop, 0, NUM_CROPS - 1), NUM_CROPS, dtype=jnp.int32
        )
        * (crop >= 0)[..., None]
    ).sum(axis=(2, 3))
    animal_counts = (
        jax.nn.one_hot(
            jnp.clip(animal, 0, NUM_ANIMALS - 1),
            NUM_ANIMALS,
            dtype=jnp.int32,
        )
        * (animal >= 0)[..., None]
    ).sum(axis=(2, 3))
    kinds = states.tile_kind.astype(jnp.int32)
    empty_structure = animal < 0
    kind_counts = jnp.stack(
        (
            jnp.sum(empty_structure & (kinds == TileKind.COOP), axis=(2, 3)),
            jnp.sum(
                empty_structure & (kinds == TileKind.PASTURE), axis=(2, 3)
            ),
            jnp.sum(kinds == TileKind.WEED, axis=(2, 3)),
        ),
        axis=-1,
    )
    signature = jnp.concatenate(
        (crop_counts, animal_counts, kind_counts), axis=-1
    )
    hands = jnp.sum(states.unit_active, axis=2).astype(jnp.int32) - 1
    return (
        jnp.sum(jnp.abs(signature[:, 0] - signature[:, 1]), axis=1)
        + jnp.abs(hands[:, 0] - hands[:, 1])
        + 3
        * jnp.abs(
            states.unlocked_count[:, 0].astype(jnp.int32)
            - states.unlocked_count[:, 1].astype(jnp.int32)
        )
    )


def _prt6_projected_shed(states, action: Action, player: int) -> jax.Array:
    """Project same-step DROP/PLACE deposits before market execution."""

    batch_size = states.step.shape[0]
    batch = jnp.arange(batch_size)
    projected = states.shed[:, player].astype(jnp.int32)
    positions = states.unit_pos[:, player].astype(jnp.int32)
    inventories = states.unit_inventory[:, player].astype(jnp.int32)
    inventory_order = states.unit_inventory_order[:, player].astype(jnp.int32)

    for unit in range(MAX_UNITS):
        present = unit < action.unit_count.astype(jnp.int32)
        position = positions[:, unit]
        at_shed = jnp.any(
            jnp.all(
                position[:, None, :]
                == _SHED_ACCESS.astype(jnp.int32)[None, :, :],
                axis=2,
            ),
            axis=1,
        )
        op = action.unit_op[:, unit]
        item = action.unit_item[:, unit].astype(jnp.int32)
        requested = jnp.maximum(action.unit_amount[:, unit], 0).astype(jnp.int32)
        drop = present & at_shed & (op == UnitOp.DROP)

        # Python dictionaries preserve acquisition order.  The JAX simulator
        # records that order explicitly so a capacity-limited DROP is exact.
        order_key = jnp.where(
            inventory_order[:, unit] >= 0,
            inventory_order[:, unit],
            32767,
        )
        item_order = jnp.argsort(order_key, axis=1, stable=True)
        for ordinal in range(NUM_SHED_ITEMS):
            item_id = item_order[:, ordinal].astype(jnp.int32)
            available = jnp.take_along_axis(
                inventories[:, unit], item_id[:, None], axis=1
            )[:, 0]
            room = jnp.maximum(
                SHED_CAPACITY - jnp.sum(projected, axis=1), 0
            )
            take = jnp.where(drop, jnp.minimum(jnp.maximum(available, 0), room), 0)
            projected = projected.at[batch, item_id].add(take)

        safe_item = jnp.clip(item, 0, NUM_SHED_ITEMS - 1)
        x = jnp.clip(position[:, 0], 0, 9)
        y = jnp.clip(position[:, 1], 0, 9)
        kind = states.tile_kind[batch, player, y, x]
        tile_animal = states.tile_animal[batch, player, y, x]
        animal_id = item - NUM_PRODUCTS
        safe_animal = jnp.clip(animal_id, 0, NUM_ANIMALS - 1)
        places_animal = (
            (animal_id >= 0)
            & (animal_id < NUM_ANIMALS)
            & (kind == _ANIMAL_STRUCTURE[safe_animal])
            & (tile_animal < 0)
        )
        place_shed = (
            present
            & at_shed
            & (op == UnitOp.PLACE)
            & (~places_animal)
            & (item >= 0)
            & (item < NUM_SHED_ITEMS)
            & (requested > 0)
        )
        available = inventories[batch, unit, safe_item]
        room = jnp.maximum(SHED_CAPACITY - jnp.sum(projected, axis=1), 0)
        take = jnp.where(
            place_shed,
            jnp.minimum(jnp.minimum(requested, jnp.maximum(available, 0)), room),
            0,
        )
        projected = projected.at[batch, safe_item].add(take)
    return projected


def _append_market_order(
    action: Action,
    append: jax.Array,
    item: int,
    amount: jax.Array,
) -> Action:
    batch = jnp.arange(action.market_op.shape[0])
    slot = jnp.clip(
        action.market_count.astype(jnp.int32), 0, MAX_MARKET_ORDERS - 1
    )
    action = action._replace(
        market_op=action.market_op.at[batch, slot].set(
            jnp.where(append, MarketOp.SELL, action.market_op[batch, slot])
        ),
        market_item=action.market_item.at[batch, slot].set(
            jnp.where(append, item, action.market_item[batch, slot])
        ),
        market_amount=action.market_amount.at[batch, slot].set(
            jnp.where(append, amount, action.market_amount[batch, slot])
        ),
        market_count=(
            action.market_count + append.astype(action.market_count.dtype)
        ).astype(jnp.int8),
    )
    return action


def _prt6_preempt_shift(
    states,
    action: Action,
    bank: CalibrationTraceV1,
    skeleton_id: jax.Array,
    player: int,
    due_step: jax.Array,
    due: jax.Array,
    enabled: jax.Array,
) -> tuple[Action, jax.Array, jax.Array]:
    """Exact six-horizon premium-sale preemption used by PRT V6."""

    step = states.step.astype(jnp.int32)
    base_enabled = (
        enabled
        & (step >= 120)
        & (step < 680)
        & (jnp.sum(due.astype(jnp.int32), axis=1) == 0)
        & (_prt6_route_distance(states) <= 6)
        & (action.market_count < MAX_MARKET_ORDERS)
    )
    remaining = _prt6_projected_shed(states, action, player)
    active = (
        jnp.arange(MAX_MARKET_ORDERS)[None, :] < action.market_count[:, None]
    )
    safe_item = jnp.clip(
        action.market_item.astype(jnp.int32), 0, NUM_PRODUCTS - 1
    )
    planned = jnp.zeros(
        (states.step.shape[0], NUM_PRODUCTS), dtype=jnp.int32
    )
    planned = planned.at[
        jnp.arange(states.step.shape[0])[:, None], safe_item
    ].add(
        jnp.where(
            active & (action.market_op == MarketOp.SELL),
            jnp.maximum(action.market_amount, 0),
            0,
        )
    )
    remaining = jnp.maximum(remaining[:, :NUM_PRODUCTS] - planned, 0)

    selected_action = action
    selected_due_step = due_step
    selected_due = due
    chosen = jnp.zeros_like(base_enabled)
    for horizon in _PRT6_PREEMPT_HORIZONS:
        future_step = step + horizon
        in_range = future_step < bank.market_op.shape[1]
        safe_step = jnp.clip(future_step, 0, bank.market_op.shape[1] - 1)
        future_op = bank.market_op[skeleton_id, safe_step]
        future_item = bank.market_item[skeleton_id, safe_step]
        future_amount = bank.market_amount[skeleton_id, safe_step]
        future_active = (
            jnp.arange(MAX_MARKET_ORDERS)[None, :]
            < bank.market_count[skeleton_id, safe_step][:, None]
        )

        trial_action = action
        trial_remaining = remaining
        trial_due = jnp.zeros_like(due)
        shifted = jnp.zeros_like(base_enabled)
        for product_id in _PRT6_PREMIUM_IDS:
            future_quantity = jnp.sum(
                jnp.where(
                    future_active
                    & (future_op == MarketOp.SELL)
                    & (future_item == product_id),
                    jnp.maximum(future_amount, 0),
                    0,
                ),
                axis=1,
            )
            target = jnp.minimum(
                jnp.minimum(trial_remaining[:, product_id], future_quantity),
                30,
            )
            append = (
                base_enabled
                & in_range
                & (future_quantity >= 4)
                & (states.market_price[:, product_id] > 1)
                & (target > 0)
                & (trial_action.market_count < MAX_MARKET_ORDERS)
            )
            trial_action = _append_market_order(
                trial_action, append, product_id, target
            )
            trial_remaining = trial_remaining.at[:, product_id].add(
                -jnp.where(append, target, 0)
            )
            trial_due = trial_due.at[:, product_id].set(
                jnp.where(append, target, trial_due[:, product_id])
            )
            shifted = shifted | append

        take = base_enabled & (~chosen) & shifted
        selected_action = _select_action(take, trial_action, selected_action)
        selected_due_step = jnp.where(
            take, step + horizon, selected_due_step
        ).astype(jnp.int16)
        selected_due = jnp.where(
            take[:, None], trial_due, selected_due
        ).astype(jnp.int16)
        chosen = chosen | take
    return selected_action, selected_due_step, selected_due


def dynamic_player_action(states, tables, bank, opponent_ids, carry: DynamicCarry, player: int, router: RouterArrays):
    step = states.step.astype(jnp.int16)
    def update_router(value):
        skeleton, family = value
        features48 = extract_base48(states, player)
        generic_selected = choose_centroid(
            features48,
            router.centroid[opponent_ids],
            router.mean[opponent_ids],
            router.scale[opponent_ids],
            router.route_ids[opponent_ids],
            router.route_count[opponent_ids],
        )
        generic_due = (router.kind[opponent_ids] == 2) & (step == router.decision[opponent_ids])
        skeleton = jnp.where(generic_due, generic_selected, skeleton)

        # Public04: YARN present selects high unless ICE_CREAM then YARN dominates.
        shops = states.town_shops.astype(jnp.int32)
        active = jnp.arange(shops.shape[1])[None, :] < states.town_count[:, None]
        yarn = SHOP_NAMES.index("YARN_STORE")
        ice = SHOP_NAMES.index("ICE_CREAM_SHOP")
        has_yarn = jnp.any(active & (shops == yarn), axis=1)
        dominated = (states.town_count >= 2) & (shops[:, 0] == ice) & (shops[:, 1] == yarn)
        public04_selected = jnp.where(has_yarn & ~dominated, router.public04_ids[1], router.public04_ids[0])
        public04_id = router.public04_opponent_id.astype(jnp.int32)
        public04_due = (opponent_ids == public04_id) & (
            step == router.decision[public04_id]
        )
        skeleton = jnp.where(public04_due, public04_selected, skeleton)

        # V8 first selects PRT versus rank1, then each family selects its own route.
        v8_id = router.v8_opponent_id.astype(jnp.int32)
        v8_family_due = (opponent_ids == v8_id) & (
            step == router.decision[v8_id]
        )
        alternate = evaluate_v8_tree(extract_tree56(states, player), router)
        family = jnp.where(v8_family_due, alternate.astype(jnp.int8), family)
        family_default = jnp.where(alternate, router.v8_alt_ids[0], router.v8_prt_ids[0])
        skeleton = jnp.where(v8_family_due, family_default, skeleton)

        batch_size = states.step.shape[0]
        prt_ids = jnp.broadcast_to(router.v8_prt_ids[None, :], (batch_size, router.v8_prt_ids.shape[0]))
        prt_centroid = jnp.broadcast_to(router.v8_prt_centroid[None, :, :], (batch_size, *router.v8_prt_centroid.shape))
        prt_selected = choose_centroid(
            features48, prt_centroid,
            jnp.broadcast_to(router.v8_prt_mean, features48.shape),
            jnp.broadcast_to(router.v8_prt_scale, features48.shape),
            prt_ids, jnp.full((batch_size,), router.v8_prt_count, dtype=jnp.int8),
        )
        alt_ids = jnp.broadcast_to(router.v8_alt_ids[None, :], (batch_size, router.v8_alt_ids.shape[0]))
        alt_centroid = jnp.broadcast_to(router.v8_alt_centroid[None, :, :], (batch_size, *router.v8_alt_centroid.shape))
        alt_selected = choose_centroid(
            features48, alt_centroid,
            jnp.broadcast_to(router.v8_alt_mean, features48.shape),
            jnp.broadcast_to(router.v8_alt_scale, features48.shape),
            alt_ids, jnp.full((batch_size,), router.v8_alt_count, dtype=jnp.int8),
        )
        prt_due = (opponent_ids == v8_id) & (family == 0) & (step == router.v8_prt_decision)
        alt_due = (opponent_ids == v8_id) & (family == 1) & (step == router.v8_alt_decision)
        skeleton = jnp.where(prt_due, prt_selected, skeleton)
        skeleton = jnp.where(alt_due, alt_selected, skeleton)

        # Local PRT V6 uses a public 56-feature tree at step 168 to select one
        # of six frozen route streams.
        prt6_id = router.prt6_opponent_id.astype(jnp.int32)
        prt6_due = (opponent_ids == prt6_id) & (
            step == router.decision[jnp.maximum(prt6_id, 0)]
        )
        prt6_choice = evaluate_prt6_tree(extract_tree56(states, player), router)
        prt6_selected = router.prt6_ids[prt6_choice]
        skeleton = jnp.where(prt6_due, prt6_selected, skeleton)
        return skeleton, family

    # The broad GOLD proxy pool also contains routers that commit on steps
    # 192, 216, and 470.  Keep the small static union so XLA only computes the
    # expensive public feature block on real routing steps.
    routing_step = jnp.any(
        states.step[0]
        == jnp.asarray((72, 120, 145, 152, 168, 192, 216, 470), dtype=states.step.dtype)
    )
    skeleton, family = jax.lax.cond(
        routing_step,
        update_router,
        lambda value: value,
        (carry.skeleton_id, carry.family),
    )

    # G05 is explicitly approximate: use the medoid tape collected for the
    # actual seat, then retain the generic GPU trace repair/liquidation layer.
    g05_mask = router.kind[opponent_ids] == 19
    g05_skeleton = jnp.full(
        skeleton.shape,
        router.public_g05_seat_skeleton_ids[player],
        dtype=jnp.int32,
    )
    skeleton = jnp.where(g05_mask, g05_skeleton, skeleton)

    raw_action, trace_carry = skeleton_player_raw_action_v1(
        states, bank, skeleton, carry.trace, player
    )
    generic_market = _rank_sell_slots(
        states,
        tables,
        raw_action.market_op,
        raw_action.market_item,
        raw_action.market_amount,
        raw_action.market_count,
    )
    generic_action = _append_terminal_liquidation_v1(
        states,
        raw_action._replace(
            market_op=generic_market[0],
            market_item=generic_market[1],
            market_amount=generic_market[2],
        ),
        player,
    )

    prt6_mask = router.kind[opponent_ids] == 5
    prt6_action, prt6_due_step, prt6_due = _prt6_repay_shift(
        raw_action,
        states.step,
        carry.prt6_due_step,
        carry.prt6_due,
        prt6_mask,
    )
    prt6_market = _rank_sell_slots(
        states,
        tables,
        prt6_action.market_op,
        prt6_action.market_item,
        prt6_action.market_amount,
        prt6_action.market_count,
    )
    prt6_action = prt6_action._replace(
        market_op=prt6_market[0],
        market_item=prt6_market[1],
        market_amount=prt6_market[2],
    )
    prt6_action, prt6_due_step, prt6_due = _prt6_preempt_shift(
        states,
        prt6_action,
        bank,
        skeleton,
        player,
        prt6_due_step,
        prt6_due,
        prt6_mask,
    )
    prt6_action = _append_terminal_liquidation_v1(
        states, prt6_action, player
    )
    action = _select_action(prt6_mask, prt6_action, generic_action)
    return action, DynamicCarry(
        trace_carry,
        skeleton,
        family,
        jnp.where(prt6_mask, prt6_due_step, carry.prt6_due_step).astype(
            jnp.int16
        ),
        jnp.where(prt6_mask[:, None], prt6_due, carry.prt6_due).astype(
            jnp.int16
        ),
    )


def _select_action(mask: jax.Array, selected: Action, fallback: Action) -> Action:
    return Action(
        *(
            jnp.where(
                mask.reshape((mask.shape[0],) + (1,) * (left.ndim - 1)),
                left,
                right,
            )
            for left, right in zip(selected, fallback, strict=True)
        )
    )


def full_opponent_action(
    states,
    tables,
    bank,
    boatlee_trace: BoatleeTraceV1,
    opponent_ids,
    carry: FullOpponentCarry,
    player: int,
    router: RouterArrays,
):
    """Dispatch exact special policies and route-state-machine opponents.

    All branches remain GPU native.  Both carries are advanced because JAX
    needs fixed structures; only the carry selected by ``opponent_ids`` can
    influence the emitted action for that row.
    """

    generic_action, dynamic_carry = dynamic_player_action(
        states,
        tables,
        bank,
        opponent_ids,
        carry.dynamic,
        player,
        router,
    )
    boatlee_action, boatlee_carry = boatlee_player_action_v1(
        states,
        tables,
        boatlee_trace,
        carry.boatlee,
        player,
    )
    c95_skeleton = jnp.full(
        opponent_ids.shape, router.c95_skeleton_id, dtype=jnp.int32
    )
    c95_action, c95_carry = c95_player_action_v1(
        states,
        tables,
        bank,
        c95_skeleton,
        carry.c95,
        player,
    )
    public_g02_skeleton = jnp.full(
        opponent_ids.shape, router.public_g02_skeleton_id, dtype=jnp.int32
    )
    public_g02_action, public_g02_carry = public_g02_player_action_v1(
        states,
        bank,
        public_g02_skeleton,
        router.public_g02_rival_schedule,
        carry.public_g02,
        player,
    )
    public_g04_skeleton = jnp.full(
        opponent_ids.shape, router.public_g04_skeleton_id, dtype=jnp.int32
    )
    public_g04_action, public_g04_carry = public_rc5_weed_player_action_v1(
        states,
        bank,
        public_g04_skeleton,
        carry.public_g04,
        player,
    )
    public_v25_skeleton = jnp.full(
        opponent_ids.shape, router.public_v25_skeleton_id, dtype=jnp.int32
    )
    public_v25_action, public_v25_carry = public_v25_player_action_v1(
        states,
        tables,
        bank,
        public_v25_skeleton,
        carry.public_v25,
        player,
    )
    public_v14_skeleton = jnp.full(
        opponent_ids.shape, router.public_v14_skeleton_id, dtype=jnp.int32
    )
    public_v14_action, public_v14_carry = public_v14_player_action_v1(
        states,
        tables,
        bank,
        public_v14_skeleton,
        carry.public_v14,
        player,
    )
    public_v21_skeleton = jnp.full(
        opponent_ids.shape, router.public_v21_skeleton_id, dtype=jnp.int32
    )
    public_v21_action, public_v21_carry = public_v21_player_action_v1(
        states,
        bank,
        public_v21_skeleton,
        router.public_v21_prototype_signature,
        router.public_v21_prototype_sales,
        carry.public_v21,
        player,
    )
    public_v13_skeleton = jnp.full(
        opponent_ids.shape, router.public_v13_skeleton_id, dtype=jnp.int32
    )
    public_v13_action, public_v13_carry = public_v13_r3_player_action_v1(
        states,
        bank,
        public_v13_skeleton,
        router.public_v13_hazard_enabled,
        router.public_v13_hazard_cap,
        carry.public_v13,
        player,
    )
    public_c68_skeleton = jnp.full(
        opponent_ids.shape, router.public_c68_skeleton_id, dtype=jnp.int32
    )
    public_c68_action, public_c68_carry = public_c68_player_action_v1(
        states,
        tables,
        bank,
        public_c68_skeleton,
        carry.public_c68,
        player,
    )
    public_four_hire_kaito = jnp.full(
        opponent_ids.shape,
        router.public_four_hire_kaito_skeleton_id,
        dtype=jnp.int32,
    )
    public_four_hire_ray = jnp.full(
        opponent_ids.shape,
        router.public_four_hire_ray_skeleton_id,
        dtype=jnp.int32,
    )
    public_four_hire_action, public_four_hire_carry = (
        public_four_hire_player_action_v1(
            states,
            tables,
            bank,
            public_four_hire_kaito,
            public_four_hire_ray,
            carry.public_four_hire,
            player,
        )
    )
    public_tran_cashflow_skeleton = jnp.full(
        opponent_ids.shape,
        router.public_tran_cashflow_skeleton_id,
        dtype=jnp.int32,
    )
    public_tran_cashflow_action, public_tran_cashflow_carry = (
        public_tran_cashflow_player_action_v1(
            states,
            bank,
            public_tran_cashflow_skeleton,
            carry.public_tran_cashflow,
            player,
        )
    )
    public_bruce_route1_skeleton = jnp.full(
        opponent_ids.shape,
        router.public_bruce_route1_skeleton_id,
        dtype=jnp.int32,
    )
    public_bruce_route1_action, public_bruce_route1_carry = (
        public_bruce_route1_player_action_v1(
            states,
            bank,
            public_bruce_route1_skeleton,
            carry.public_bruce_route1,
            player,
        )
    )
    public_v19_control_skeleton = jnp.full(
        opponent_ids.shape,
        router.public_v19_control_skeleton_id,
        dtype=jnp.int32,
    )
    public_v19_control_action = public_v19_control_player_action_v1(
        states,
        bank,
        public_v19_control_skeleton,
        player,
    )
    public_v18_board_skeleton = jnp.full(
        opponent_ids.shape,
        router.public_v18_board_skeleton_id,
        dtype=jnp.int32,
    )
    public_v18_closed_loop_action, public_v18_closed_loop_carry = (
        public_v18_closed_loop_player_action_v1(
            states,
            bank,
            public_v18_board_skeleton,
            router.public_v18_expert_skeleton_ids,
            router.public_v18_feature_scale,
            router.public_v18_market_bias_by_seat,
            router.public_v18_prototypes_by_day,
            router.public_v18_distance_strength,
            router.public_v18_stay_bonus,
            carry.public_v18_closed_loop,
            player,
        )
    )
    exact_boatlee = router.kind[opponent_ids] == 4
    exact_c95 = router.kind[opponent_ids] == 6
    exact_public_g02 = router.kind[opponent_ids] == 7
    exact_public_g04 = router.kind[opponent_ids] == 8
    exact_public_v25 = router.kind[opponent_ids] == 9
    exact_public_v14 = router.kind[opponent_ids] == 10
    exact_public_v21 = router.kind[opponent_ids] == 11
    exact_public_v13 = router.kind[opponent_ids] == 12
    exact_public_c68 = router.kind[opponent_ids] == 13
    exact_public_four_hire = router.kind[opponent_ids] == 14
    exact_public_tran_cashflow = router.kind[opponent_ids] == 15
    exact_public_bruce_route1 = router.kind[opponent_ids] == 16
    exact_public_v19_control = router.kind[opponent_ids] == 17
    exact_public_v18_closed_loop = router.kind[opponent_ids] == 18
    selected = _select_action(exact_boatlee, boatlee_action, generic_action)
    selected = _select_action(exact_c95, c95_action, selected)
    selected = _select_action(exact_public_g02, public_g02_action, selected)
    selected = _select_action(exact_public_g04, public_g04_action, selected)
    selected = _select_action(exact_public_v25, public_v25_action, selected)
    selected = _select_action(exact_public_v14, public_v14_action, selected)
    selected = _select_action(exact_public_v21, public_v21_action, selected)
    selected = _select_action(exact_public_v13, public_v13_action, selected)
    selected = _select_action(exact_public_c68, public_c68_action, selected)
    selected = _select_action(
        exact_public_four_hire, public_four_hire_action, selected
    )
    selected = _select_action(
        exact_public_tran_cashflow, public_tran_cashflow_action, selected
    )
    selected = _select_action(
        exact_public_bruce_route1, public_bruce_route1_action, selected
    )
    selected = _select_action(
        exact_public_v19_control, public_v19_control_action, selected
    )
    selected = _select_action(
        exact_public_v18_closed_loop, public_v18_closed_loop_action, selected
    )

    def select_carry(mask, selected_carry, old_carry):
        if hasattr(selected_carry, "_fields"):
            return type(selected_carry)(
                *(
                    select_carry(mask, left, right)
                    for left, right in zip(
                        selected_carry, old_carry, strict=True
                    )
                )
            )
        return jnp.where(
            mask.reshape(
                (mask.shape[0],) + (1,) * (selected_carry.ndim - 1)
            ),
            selected_carry,
            old_carry,
        )

    return (
        selected,
        FullOpponentCarry(
            dynamic_carry,
            select_carry(exact_boatlee, boatlee_carry, carry.boatlee),
            select_carry(exact_c95, c95_carry, carry.c95),
            select_carry(
                exact_public_g02, public_g02_carry, carry.public_g02
            ),
            select_carry(
                exact_public_g04, public_g04_carry, carry.public_g04
            ),
            select_carry(
                exact_public_v25, public_v25_carry, carry.public_v25
            ),
            select_carry(
                exact_public_v14, public_v14_carry, carry.public_v14
            ),
            select_carry(
                exact_public_v21, public_v21_carry, carry.public_v21
            ),
            select_carry(
                exact_public_v13, public_v13_carry, carry.public_v13
            ),
            select_carry(
                exact_public_c68, public_c68_carry, carry.public_c68
            ),
            select_carry(
                exact_public_four_hire,
                public_four_hire_carry,
                carry.public_four_hire,
            ),
            select_carry(
                exact_public_tran_cashflow,
                public_tran_cashflow_carry,
                carry.public_tran_cashflow,
            ),
            select_carry(
                exact_public_bruce_route1,
                public_bruce_route1_carry,
                carry.public_bruce_route1,
            ),
            select_carry(
                exact_public_v18_closed_loop,
                public_v18_closed_loop_carry,
                carry.public_v18_closed_loop,
            ),
        ),
    )


def pair(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def make_rollout(
    bank,
    tables,
    router: RouterArrays,
    candidate_player: int,
    boatlee_trace: BoatleeTraceV1 | None = None,
):
    opponent_player = 1 - candidate_player
    if boatlee_trace is None:
        boatlee_trace = load_boatlee_trace_v1()

    @jax.jit
    def rollout(initial, events, candidate_ids, opponent_ids):
        batch_size = initial.step.shape[0]
        zero48 = jnp.zeros((batch_size, 48), dtype=jnp.float32)

        def body(value, _):
            (
                states,
                candidate_carry,
                opponent_carry,
                decision1,
                decision20,
                decision120,
                anchor73,
                anchor121,
                decision145,
                decision168,
                decision216,
            ) = value
            def capture(captured):
                old1, old20, old120, old73, old121, old145, old168, old216 = captured
                actor_features = extract_base48(states, candidate_player)
                old1 = jnp.where((states.step == 1)[:, None], actor_features, old1)
                old20 = jnp.where((states.step == 20)[:, None], actor_features, old20)
                old120 = jnp.where((states.step == 120)[:, None], actor_features, old120)
                old73 = jnp.where((states.step == 73)[:, None], actor_features, old73)
                old121 = jnp.where((states.step == 121)[:, None], actor_features, old121)
                old145 = jnp.where((states.step == 145)[:, None], actor_features, old145)
                old168 = jnp.where((states.step == 168)[:, None], actor_features, old168)
                old216 = jnp.where((states.step == 216)[:, None], actor_features, old216)
                return old1, old20, old120, old73, old121, old145, old168, old216

            capture_step = jnp.any(
                states.step[0]
                == jnp.asarray((1, 20, 73, 120, 121, 145, 168, 216), dtype=states.step.dtype)
            )
            (
                decision1,
                decision20,
                decision120,
                anchor73,
                anchor121,
                decision145,
                decision168,
                decision216,
            ) = jax.lax.cond(
                capture_step,
                capture,
                lambda captured: captured,
                (
                    decision1,
                    decision20,
                    decision120,
                    anchor73,
                    anchor121,
                    decision145,
                    decision168,
                    decision216,
                ),
            )
            candidate_action, candidate_carry = skeleton_player_action_v1(
                states, tables, bank, candidate_ids, candidate_carry, candidate_player
            )
            opponent_action, opponent_carry = full_opponent_action(
                states,
                tables,
                bank,
                boatlee_trace,
                opponent_ids,
                opponent_carry,
                opponent_player,
                router,
            )
            actions = pair(candidate_action, opponent_action) if candidate_player == 0 else pair(opponent_action, candidate_action)
            states = batched_step_sync(states, actions, events, tables)
            return (
                states,
                candidate_carry,
                opponent_carry,
                decision1,
                decision20,
                decision120,
                anchor73,
                anchor121,
                decision145,
                decision168,
                decision216,
            ), None

        result, _ = jax.lax.scan(
            body,
            (
                initial,
                initialize_trace_player_carry_v1(batch_size),
                initialize_full_opponent_carry(batch_size, opponent_ids, router),
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
            ),
            xs=None,
            length=719,
        )
        (
            states,
            _,
            _,
            decision1,
            decision20,
            decision120,
            anchor73,
            anchor121,
            decision145,
            decision168,
            decision216,
        ) = result
        trend = jnp.asarray(TREND_INDICES, dtype=jnp.int32)
        features66 = jnp.concatenate((
            decision145,
            decision145[:, trend] - anchor73[:, trend],
            decision145[:, trend] - anchor121[:, trend],
        ), axis=1)
        return (
            states.money,
            states.done,
            features66,
            decision1,
            decision20,
            decision120,
            decision168,
            decision216,
        )

    return rollout


def make_kind_rollout(
    bank,
    tables,
    router: RouterArrays,
    candidate_player: int,
    kind_id: int,
    boatlee_trace: BoatleeTraceV1 | None = None,
):
    """Compile only one controller kind instead of the whole dispatcher.

    ``kind_id`` is a Python-static closure value.  XLA therefore sees exactly
    one opponent policy and does not spend minutes compiling the other
    eighteen unreachable controllers.
    """

    opponent_player = 1 - candidate_player
    if boatlee_trace is None:
        boatlee_trace = load_boatlee_trace_v1()

    def initialize_opponent(batch_size, opponent_ids):
        if kind_id in (0, 1, 2, 3, 5, 19):
            return initialize_dynamic_carry(batch_size, opponent_ids, router)
        if kind_id == 4:
            return initialize_boatlee_player_carry_v1(batch_size)
        if kind_id == 6:
            return initialize_c95_player_carry_v1(batch_size)
        if kind_id in (7, 8, 9, 10, 11, 12, 13):
            return initialize_public_g02_carry_v1(batch_size)
        if kind_id == 14:
            return initialize_four_hire_carry_v1(batch_size)
        if kind_id == 15:
            return initialize_tran_cashflow_carry_v1(batch_size)
        if kind_id == 16:
            return initialize_bruce_route1_carry_v1(batch_size)
        if kind_id == 17:
            return jnp.zeros((batch_size,), dtype=jnp.bool_)
        if kind_id == 18:
            return initialize_v18_closed_loop_carry_v1(batch_size)
        raise ValueError(f"unsupported grouped controller kind {kind_id}")

    def opponent_action(states, opponent_ids, carry):
        batch_size = states.step.shape[0]
        if kind_id in (0, 1, 2, 3, 5, 19):
            return dynamic_player_action(
                states, tables, bank, opponent_ids, carry, opponent_player, router
            )
        if kind_id == 4:
            return boatlee_player_action_v1(
                states, tables, boatlee_trace, carry, opponent_player
            )
        if kind_id == 6:
            skeleton = jnp.full(
                (batch_size,), router.c95_skeleton_id, dtype=jnp.int32
            )
            return c95_player_action_v1(
                states, tables, bank, skeleton, carry, opponent_player
            )
        if kind_id == 7:
            skeleton = jnp.full(
                (batch_size,), router.public_g02_skeleton_id, dtype=jnp.int32
            )
            return public_g02_player_action_v1(
                states,
                bank,
                skeleton,
                router.public_g02_rival_schedule,
                carry,
                opponent_player,
            )
        if kind_id == 8:
            skeleton = jnp.full(
                (batch_size,), router.public_g04_skeleton_id, dtype=jnp.int32
            )
            return public_rc5_weed_player_action_v1(
                states, bank, skeleton, carry, opponent_player
            )
        if kind_id == 9:
            skeleton = jnp.full(
                (batch_size,), router.public_v25_skeleton_id, dtype=jnp.int32
            )
            return public_v25_player_action_v1(
                states, tables, bank, skeleton, carry, opponent_player
            )
        if kind_id == 10:
            skeleton = jnp.full(
                (batch_size,), router.public_v14_skeleton_id, dtype=jnp.int32
            )
            return public_v14_player_action_v1(
                states, tables, bank, skeleton, carry, opponent_player
            )
        if kind_id == 11:
            skeleton = jnp.full(
                (batch_size,), router.public_v21_skeleton_id, dtype=jnp.int32
            )
            return public_v21_player_action_v1(
                states,
                bank,
                skeleton,
                router.public_v21_prototype_signature,
                router.public_v21_prototype_sales,
                carry,
                opponent_player,
            )
        if kind_id == 12:
            skeleton = jnp.full(
                (batch_size,), router.public_v13_skeleton_id, dtype=jnp.int32
            )
            return public_v13_r3_player_action_v1(
                states,
                bank,
                skeleton,
                router.public_v13_hazard_enabled,
                router.public_v13_hazard_cap,
                carry,
                opponent_player,
            )
        if kind_id == 13:
            skeleton = jnp.full(
                (batch_size,), router.public_c68_skeleton_id, dtype=jnp.int32
            )
            return public_c68_player_action_v1(
                states, tables, bank, skeleton, carry, opponent_player
            )
        if kind_id == 14:
            kaito = jnp.full(
                (batch_size,), router.public_four_hire_kaito_skeleton_id, dtype=jnp.int32
            )
            ray = jnp.full(
                (batch_size,), router.public_four_hire_ray_skeleton_id, dtype=jnp.int32
            )
            return public_four_hire_player_action_v1(
                states, tables, bank, kaito, ray, carry, opponent_player
            )
        if kind_id == 15:
            skeleton = jnp.full(
                (batch_size,), router.public_tran_cashflow_skeleton_id, dtype=jnp.int32
            )
            return public_tran_cashflow_player_action_v1(
                states, bank, skeleton, carry, opponent_player
            )
        if kind_id == 16:
            skeleton = jnp.full(
                (batch_size,), router.public_bruce_route1_skeleton_id, dtype=jnp.int32
            )
            return public_bruce_route1_player_action_v1(
                states, bank, skeleton, carry, opponent_player
            )
        if kind_id == 17:
            skeleton = jnp.full(
                (batch_size,), router.public_v19_control_skeleton_id, dtype=jnp.int32
            )
            return (
                public_v19_control_player_action_v1(
                    states, bank, skeleton, opponent_player
                ),
                carry,
            )
        if kind_id == 18:
            board = jnp.full(
                (batch_size,), router.public_v18_board_skeleton_id, dtype=jnp.int32
            )
            return public_v18_closed_loop_player_action_v1(
                states,
                bank,
                board,
                router.public_v18_expert_skeleton_ids,
                router.public_v18_feature_scale,
                router.public_v18_market_bias_by_seat,
                router.public_v18_prototypes_by_day,
                router.public_v18_distance_strength,
                router.public_v18_stay_bonus,
                carry,
                opponent_player,
            )
        raise ValueError(f"unsupported grouped controller kind {kind_id}")

    @jax.jit
    def rollout(initial, events, candidate_ids, opponent_ids):
        batch_size = initial.step.shape[0]
        zero48 = jnp.zeros((batch_size, 48), dtype=jnp.float32)

        def body(value, _):
            (
                states,
                candidate_carry,
                opponent_carry,
                f1,
                f20,
                f120,
                f73,
                f121,
                f145,
                f168,
                f216,
            ) = value

            def capture(captured):
                old1, old20, old120, old73, old121, old145, old168, old216 = captured
                features = extract_base48(states, candidate_player)
                old1 = jnp.where((states.step == 1)[:, None], features, old1)
                old20 = jnp.where((states.step == 20)[:, None], features, old20)
                old120 = jnp.where((states.step == 120)[:, None], features, old120)
                old73 = jnp.where((states.step == 73)[:, None], features, old73)
                old121 = jnp.where((states.step == 121)[:, None], features, old121)
                old145 = jnp.where((states.step == 145)[:, None], features, old145)
                old168 = jnp.where((states.step == 168)[:, None], features, old168)
                old216 = jnp.where((states.step == 216)[:, None], features, old216)
                return old1, old20, old120, old73, old121, old145, old168, old216

            capture_step = jnp.any(
                states.step[0]
                == jnp.asarray((1, 20, 73, 120, 121, 145, 168, 216), dtype=states.step.dtype)
            )
            f1, f20, f120, f73, f121, f145, f168, f216 = jax.lax.cond(
                capture_step,
                capture,
                lambda captured: captured,
                (f1, f20, f120, f73, f121, f145, f168, f216),
            )
            candidate_action, candidate_carry = skeleton_player_action_v1(
                states, tables, bank, candidate_ids, candidate_carry, candidate_player
            )
            rival_action, opponent_carry = opponent_action(
                states, opponent_ids, opponent_carry
            )
            actions = (
                pair(candidate_action, rival_action)
                if candidate_player == 0
                else pair(rival_action, candidate_action)
            )
            states = batched_step_sync(states, actions, events, tables)
            return (
                states,
                candidate_carry,
                opponent_carry,
                f1,
                f20,
                f120,
                f73,
                f121,
                f145,
                f168,
                f216,
            ), None

        result, _ = jax.lax.scan(
            body,
            (
                initial,
                initialize_trace_player_carry_v1(batch_size),
                initialize_opponent(batch_size, opponent_ids),
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
                zero48,
            ),
            xs=None,
            length=719,
        )
        states, _, _, f1, f20, f120, f73, f121, f145, f168, f216 = result
        trend = jnp.asarray(TREND_INDICES, dtype=jnp.int32)
        features66 = jnp.concatenate(
            (f145, f145[:, trend] - f73[:, trend], f145[:, trend] - f121[:, trend]),
            axis=1,
        )
        return states.money, states.done, features66, f1, f20, f120, f168, f216

    return rollout


def run_four_hire_group_stepwise(
    initial,
    events,
    candidate_ids,
    bank,
    tables,
    router: RouterArrays,
    candidate_player: int,
):
    """Run Four-Hire with reusable single-step graphs.

    XLA optimization becomes pathological when the composite V25+C68 policy
    is nested inside a 719-step scan.  Keeping policy and simulator as two
    small cached graphs preserves exact semantics and avoids that compile.
    """

    opponent_player = 1 - candidate_player
    batch_size = initial.step.shape[0]
    kaito = jnp.full(
        (batch_size,), router.public_four_hire_kaito_skeleton_id, dtype=jnp.int32
    )
    ray = jnp.full(
        (batch_size,), router.public_four_hire_ray_skeleton_id, dtype=jnp.int32
    )

    @jax.jit
    def policy(states, candidate_carry, opponent_carry):
        candidate_action, candidate_carry = skeleton_player_action_v1(
            states,
            tables,
            bank,
            candidate_ids,
            candidate_carry,
            candidate_player,
        )
        opponent_action, opponent_carry = public_four_hire_player_action_v1(
            states,
            tables,
            bank,
            kaito,
            ray,
            opponent_carry,
            opponent_player,
        )
        return candidate_action, opponent_action, candidate_carry, opponent_carry

    @jax.jit
    def advance(states, candidate_action, opponent_action):
        actions = (
            pair(candidate_action, opponent_action)
            if candidate_player == 0
            else pair(opponent_action, candidate_action)
        )
        return batched_step_sync(states, actions, events, tables)

    @jax.jit
    def capture(states):
        return extract_base48(states, candidate_player)

    states = initial
    candidate_carry = initialize_trace_player_carry_v1(batch_size)
    opponent_carry = initialize_four_hire_carry_v1(batch_size)
    captured = {}
    for step in range(719):
        if step in (1, 20, 73, 120, 121, 145, 168, 216):
            captured[step] = capture(states)
        candidate_action, opponent_action, candidate_carry, opponent_carry = policy(
            states, candidate_carry, opponent_carry
        )
        states = advance(states, candidate_action, opponent_action)
    trend = jnp.asarray(TREND_INDICES, dtype=jnp.int32)
    features66 = jnp.concatenate(
        (
            captured[145],
            captured[145][:, trend] - captured[73][:, trend],
            captured[145][:, trend] - captured[121][:, trend],
        ),
        axis=1,
    )
    return (
        states.money,
        states.done,
        features66,
        captured[1],
        captured[20],
        captured[120],
        captured[168],
        captured[216],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, required=True)
    parser.add_argument("--features-output", type=Path, required=True)
    parser.add_argument("--outcomes-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument(
        "--opponent-indices",
        default="",
        help="Optional comma-separated compiled opponent indices; defaults to all.",
    )
    parser.add_argument(
        "--candidate-indices",
        default="",
        help=(
            "Optional comma-separated source candidate indices. This is useful "
            "for cheap shared-prefix feature extraction without rerunning every route."
        ),
    )
    parser.add_argument(
        "--outcome-only",
        action="store_true",
        help=(
            "Allow candidate routes to diverge before the feature checkpoint. "
            "The outcome tensor remains valid, but shared-state features are "
            "diagnostic only and are not admitted as counterfactual labels."
        ),
    )
    parser.add_argument(
        "--monolithic-dispatch",
        action="store_true",
        help=(
            "Compile all controller kinds into one graph. The default groups "
            "opponents by kind to avoid pathological whole-pool XLA compile time."
        ),
    )
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = Path.home() / ".cache" / "kaggriculture_jax"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    bank_path = args.bank.resolve()
    bank_receipt_path = args.bank_receipt.resolve()
    receipt = json.loads(bank_receipt_path.read_text(encoding="utf-8"))
    bank = load_bank(bank_path)
    router = build_router_arrays(receipt)
    tables = load_tables()
    seeds = np.arange(args.seed_start, args.seed_start + args.seed_count, dtype=np.int32)
    requested_opponent_indices = [
        int(value.strip())
        for value in args.opponent_indices.split(",")
        if value.strip()
    ]
    if requested_opponent_indices:
        if any(
            index < 0 or index >= len(receipt["opponents"])
            for index in requested_opponent_indices
        ):
            raise ValueError("opponent index outside the compiled bank")
        if len(requested_opponent_indices) != len(set(requested_opponent_indices)):
            raise ValueError("duplicate opponent index")
        opponent_values = np.asarray(requested_opponent_indices, dtype=np.int16)
    else:
        opponent_values = np.arange(len(receipt["opponents"]), dtype=np.int16)
    batch_seeds = np.tile(seeds, len(opponent_values))
    batch_opponents = np.repeat(opponent_values, len(seeds))
    # The official event stream depends on the seed, not the opponent.  Build
    # each seed once and tile it in the same opponent-major order as
    # ``batch_seeds`` instead of regenerating identical Python MT19937 draws
    # once per opponent.
    weed_by_seed, shops_by_seed = build_events_v1(seeds.tolist())
    weed = np.tile(weed_by_seed, (len(opponent_values), 1, 1))
    shops = np.tile(shops_by_seed, (len(opponent_values), 1, 1))
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    initial = jax.vmap(reset)(jnp.asarray(batch_seeds))
    opponent_ids = jnp.asarray(batch_opponents, dtype=jnp.int32)
    all_candidate_source_ids = list(receipt["candidate_ids"])
    requested_candidate_indices = [
        int(value.strip())
        for value in args.candidate_indices.split(",")
        if value.strip()
    ]
    if requested_candidate_indices:
        if any(
            index < 0 or index >= len(all_candidate_source_ids)
            for index in requested_candidate_indices
        ):
            raise ValueError(
                f"candidate index outside [0,{len(all_candidate_source_ids) - 1}]"
            )
        candidate_source_ids = [
            all_candidate_source_ids[index] for index in requested_candidate_indices
        ]
    else:
        requested_candidate_indices = list(range(len(all_candidate_source_ids)))
        candidate_source_ids = all_candidate_source_ids
    candidate_count = len(candidate_source_ids)
    wins = np.zeros((candidate_count, 2, 1, len(opponent_values), len(seeds)), dtype=np.bool_)
    margins = np.zeros_like(wins, dtype=np.int32)
    feature_by_candidate = np.zeros(
        (candidate_count, 2, len(opponent_values), len(seeds), 66),
        dtype=np.float32,
    )
    feature1_by_candidate = np.zeros(
        (candidate_count, 2, len(opponent_values), len(seeds), 48),
        dtype=np.float32,
    )
    feature20_by_candidate = np.zeros(
        (candidate_count, 2, len(opponent_values), len(seeds), 48),
        dtype=np.float32,
    )
    feature120_by_candidate = np.zeros(
        (candidate_count, 2, len(opponent_values), len(seeds), 48),
        dtype=np.float32,
    )
    feature168_by_candidate = np.zeros_like(feature120_by_candidate)
    feature216_by_candidate = np.zeros_like(feature120_by_candidate)
    timings = []
    all_done = True
    if args.monolithic_dispatch:
        rollouts = [make_rollout(bank, tables, router, seat) for seat in (0, 1)]
        for candidate_index, skeleton_id in enumerate(candidate_source_ids):
            candidate_ids = jnp.full(opponent_ids.shape, skeleton_id, dtype=jnp.int32)
            for seat, rollout in enumerate(rollouts):
                started = perf_counter()
                (
                    money,
                    done,
                    features,
                    features1,
                    features20,
                    features120,
                    features168,
                    features216,
                ) = rollout(
                    initial, events, candidate_ids, opponent_ids
                )
                jax.block_until_ready(money)
                elapsed = perf_counter() - started
                money_np = np.asarray(money, dtype=np.int64)
                done_np = np.asarray(done, dtype=bool)
                feature_np = np.asarray(features, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 66
                )
                feature1_np = np.asarray(features1, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 48
                )
                feature20_np = np.asarray(features20, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 48
                )
                feature120_np = np.asarray(features120, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 48
                )
                feature168_np = np.asarray(features168, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 48
                )
                feature216_np = np.asarray(features216, dtype=np.float32).reshape(
                    len(opponent_values), len(seeds), 48
                )
                own = money_np[:, seat]
                other = money_np[:, 1 - seat]
                wins[candidate_index, seat, 0] = (own > other).reshape(
                    len(opponent_values), len(seeds)
                )
                margins[candidate_index, seat, 0] = (own - other).reshape(
                    len(opponent_values), len(seeds)
                )
                feature_by_candidate[candidate_index, seat] = feature_np
                feature1_by_candidate[candidate_index, seat] = feature1_np
                feature20_by_candidate[candidate_index, seat] = feature20_np
                feature120_by_candidate[candidate_index, seat] = feature120_np
                feature168_by_candidate[candidate_index, seat] = feature168_np
                feature216_by_candidate[candidate_index, seat] = feature216_np
                all_done &= bool(np.all(done_np))
                timings.append({
                    "candidate_index": candidate_index,
                    "seat": seat,
                    "controller_kind": "monolithic",
                    "games": int(len(batch_seeds)),
                    "seconds": elapsed,
                    "transitions_per_second": len(batch_seeds) * 719 / elapsed,
                })
    else:
        kind_values = np.asarray(jax.device_get(router.kind), dtype=np.int16)[
            opponent_values
        ]
        groups = [
            (int(kind_id), np.flatnonzero(kind_values == kind_id))
            for kind_id in np.unique(kind_values)
        ]
        grouped_rollouts = {
            (seat, kind_id): make_kind_rollout(
                bank, tables, router, seat, kind_id
            )
            for seat in (0, 1)
            for kind_id, _ in groups
        }
        for candidate_index, skeleton_id in enumerate(candidate_source_ids):
            for seat in (0, 1):
                for kind_id, group_indices in groups:
                    group_opponents = opponent_values[group_indices]
                    print(
                        json.dumps(
                            {
                                "event": "group_start",
                                "candidate_index": candidate_index,
                                "seat": seat,
                                "controller_kind": kind_id,
                                "opponent_ids": group_opponents.astype(int).tolist(),
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )
                    group_batch_seeds = np.tile(seeds, len(group_indices))
                    group_batch_opponents = np.repeat(group_opponents, len(seeds))
                    group_weed = np.tile(weed_by_seed, (len(group_indices), 1, 1))
                    group_shops = np.tile(shops_by_seed, (len(group_indices), 1, 1))
                    group_events = Events(
                        jnp.asarray(group_weed), jnp.asarray(group_shops)
                    )
                    group_initial = jax.vmap(reset)(
                        jnp.asarray(group_batch_seeds, dtype=jnp.int32)
                    )
                    group_opponent_ids = jnp.asarray(
                        group_batch_opponents, dtype=jnp.int32
                    )
                    group_candidate_ids = jnp.full(
                        group_opponent_ids.shape, skeleton_id, dtype=jnp.int32
                    )
                    started = perf_counter()
                    if kind_id == 14:
                        (
                            money,
                            done,
                            features,
                            features1,
                            features20,
                            features120,
                            features168,
                            features216,
                        ) = (
                            run_four_hire_group_stepwise(
                                group_initial,
                                group_events,
                                group_candidate_ids,
                                bank,
                                tables,
                                router,
                                seat,
                            )
                        )
                    else:
                        rollout = grouped_rollouts[(seat, kind_id)]
                        (
                            money,
                            done,
                            features,
                            features1,
                            features20,
                            features120,
                            features168,
                            features216,
                        ) = rollout(
                            group_initial,
                            group_events,
                            group_candidate_ids,
                            group_opponent_ids,
                        )
                    jax.block_until_ready(money)
                    elapsed = perf_counter() - started
                    money_np = np.asarray(money, dtype=np.int64)
                    done_np = np.asarray(done, dtype=bool)
                    group_shape = (len(group_indices), len(seeds))
                    own = money_np[:, seat]
                    other = money_np[:, 1 - seat]
                    wins[candidate_index, seat, 0, group_indices, :] = (
                        own > other
                    ).reshape(group_shape)
                    margins[candidate_index, seat, 0, group_indices, :] = (
                        own - other
                    ).reshape(group_shape)
                    feature_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features, dtype=np.float32).reshape(
                        *group_shape, 66
                    )
                    feature1_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features1, dtype=np.float32).reshape(
                        *group_shape, 48
                    )
                    feature20_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features20, dtype=np.float32).reshape(
                        *group_shape, 48
                    )
                    feature120_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features120, dtype=np.float32).reshape(
                        *group_shape, 48
                    )
                    feature168_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features168, dtype=np.float32).reshape(
                        *group_shape, 48
                    )
                    feature216_by_candidate[
                        candidate_index, seat, group_indices, :, :
                    ] = np.asarray(features216, dtype=np.float32).reshape(
                        *group_shape, 48
                    )
                    all_done &= bool(np.all(done_np))
                    timings.append({
                        "candidate_index": candidate_index,
                        "seat": seat,
                        "controller_kind": kind_id,
                        "opponent_ids": group_opponents.astype(int).tolist(),
                        "games": int(len(group_batch_seeds)),
                        "seconds": elapsed,
                        "transitions_per_second": (
                            len(group_batch_seeds) * 719 / elapsed
                        ),
                    })
                    print(
                        json.dumps(
                            {
                                "event": "group_complete",
                                "candidate_index": candidate_index,
                                "seat": seat,
                                "controller_kind": kind_id,
                                "games": int(len(group_batch_seeds)),
                                "seconds": elapsed,
                            },
                            separators=(",", ":"),
                        ),
                        flush=True,
                    )

    max_feature_error = float(np.max(np.abs(feature_by_candidate - feature_by_candidate[0:1])))
    max_feature1_error = float(
        np.max(np.abs(feature1_by_candidate - feature1_by_candidate[0:1]))
    )
    max_feature20_error = float(
        np.max(np.abs(feature20_by_candidate - feature20_by_candidate[0:1]))
    )
    max_feature120_error = float(
        np.max(np.abs(feature120_by_candidate - feature120_by_candidate[0:1]))
    )
    max_feature168_error = float(
        np.max(np.abs(feature168_by_candidate - feature168_by_candidate[0:1]))
    )
    max_feature216_error = float(
        np.max(np.abs(feature216_by_candidate - feature216_by_candidate[0:1]))
    )
    features = feature_by_candidate[0]
    features1 = feature1_by_candidate[0]
    features20 = feature20_by_candidate[0]
    features120 = feature120_by_candidate[0]
    features168 = feature168_by_candidate[0]
    features216 = feature216_by_candidate[0]
    args.features_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.features_output,
        features=features,
        feature_names=np.asarray(FEATURE_NAMES),
        features1=features1,
        feature_names1=np.asarray(BASE_NAMES),
        feature_step1=np.asarray(1, dtype=np.int16),
        features20=features20,
        feature_names20=np.asarray(BASE_NAMES),
        feature_step20=np.asarray(20, dtype=np.int16),
        features120=features120,
        feature_names120=np.asarray(BASE_NAMES),
        feature_step120=np.asarray(120, dtype=np.int16),
        features168=features168,
        feature_names168=np.asarray(BASE_NAMES),
        feature_step168=np.asarray(168, dtype=np.int16),
        features216=features216,
        feature_names216=np.asarray(BASE_NAMES),
        feature_step216=np.asarray(216, dtype=np.int16),
        opponent_ids=opponent_values,
        seeds=seeds,
        candidate_id=np.asarray(candidate_source_ids[0], dtype=np.int16),
        candidate_ids=np.asarray(candidate_source_ids, dtype=np.int16),
        features1_by_candidate=feature1_by_candidate,
        features120_by_candidate=feature120_by_candidate,
        features168_by_candidate=feature168_by_candidate,
        features216_by_candidate=feature216_by_candidate,
        decision_step=np.asarray(DECISION_STEP, dtype=np.int16),
    )
    args.outcomes_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.outcomes_output,
        candidate_ids=np.asarray(candidate_source_ids, dtype=np.int16),
        opponent_ids=opponent_values,
        seeds=seeds,
        wins=wins,
        margins=margins,
    )
    total_games = sum(row["games"] for row in timings)
    total_seconds = sum(row["seconds"] for row in timings)
    result = {
        "schema": "kawashigi-jax-dynamic-counterfactual-panel-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PARITY_PENDING"
            if all_done and (args.outcome_only or max_feature_error == 0.0)
            else "FAIL"
        ),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "seed_start": int(seeds[0]),
        "seed_count": len(seeds),
        "event_streams_generated": len(seeds),
        "event_stream_reuse_factor": len(opponent_values),
        "context_count": int(2 * len(opponent_values) * len(seeds)),
        "game_count": int(wins.size),
        "all_done": all_done,
        "same_state_feature_max_abs_error": max_feature_error,
        "same_state_pass": max_feature_error == 0.0,
        "step1_feature_max_abs_error": max_feature1_error,
        "step1_same_state_pass": max_feature1_error == 0.0,
        "step20_feature_max_abs_error": max_feature20_error,
        "step20_same_state_pass": max_feature20_error == 0.0,
        "step120_feature_max_abs_error": max_feature120_error,
        "step120_same_state_pass": max_feature120_error == 0.0,
        "step168_feature_max_abs_error": max_feature168_error,
        "step168_same_state_pass": max_feature168_error == 0.0,
        "step216_feature_max_abs_error": max_feature216_error,
        "step216_same_state_pass": max_feature216_error == 0.0,
        "requested_candidate_indices": requested_candidate_indices,
        "requested_opponent_indices": opponent_values.astype(int).tolist(),
        "outcome_only": bool(args.outcome_only),
        "dispatch_mode": (
            "monolithic" if args.monolithic_dispatch else "grouped_by_kind"
        ),
        "timings": timings,
        "compile_inclusive_transitions_per_second": total_games * 719 / total_seconds,
        "bank": str(bank_path),
        "bank_sha256": sha256(bank_path),
        "bank_receipt": str(bank_receipt_path),
        "bank_receipt_sha256": sha256(bank_receipt_path),
        "features_output": str(args.features_output.resolve()),
        "features_sha256": sha256(args.features_output),
        "outcomes_output": str(args.outcomes_output.resolve()),
        "outcomes_sha256": sha256(args.outcomes_output),
        "truth_boundary": (
            "GPU dynamic route-router outcome screening; official Python 1.32.7 "
            "parity is required before promotion. With --outcome-only, feature "
            "rows from divergent candidates are diagnostic and are not "
            "counterfactual same-state labels."
        ),
    }
    args.receipt_output.parent.mkdir(parents=True, exist_ok=True)
    args.receipt_output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())
