#!/usr/bin/env python3
"""All accepted, de-duplicated JAX agents: 100-game round robin.

The Arena uses 50 independent official event seeds and repeats each seed with
the seats swapped.  Source-identical aliases are recorded but do not get an
extra rating entry.  The hot path stays on GPU; Python only advances the
already-jitted single-step graph and writes a checkpoint after every chunk.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
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
    ROOT / "experiments" / "kawashigi_counterfactual_ranker_v2" / "tools",
    ROOT / "experiments" / "strategic_v5" / "src",
    ROOT / "experiments" / "route_playbook_v1" / "src",
    ROOT / "gpu_sim" / "src",
):
    sys.path.insert(0, str(path))

from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402
from run_jax_dynamic_counterfactual_panel import (  # noqa: E402
    DynamicCarry,
    FullOpponentCarry,
    build_router_arrays,
    dynamic_player_action,
    full_opponent_action,
    initialize_dynamic_carry,
    initialize_full_opponent_carry,
    load_bank,
)
from strategic_v5.boatlee_v16_gpu import (  # noqa: E402
    boatlee_player_action_v1,
    initialize_boatlee_player_carry_v1,
    load_boatlee_trace_v1,
)
from strategic_v5.c95_gpu import (  # noqa: E402
    c95_player_action_v1,
    initialize_c95_player_carry_v1,
)
from strategic_v5.high_potential_v20_gpu import (  # noqa: E402
    MODE_BOATLEE,
    MODE_RAY_K320,
    MODE_TETSUTANI,
    MODE_TETSUTANI_LATEST,
    HighPotentialV20CarryV1,
    high_potential_v20_player_action_v1,
    initialize_high_potential_v20_carry_v1,
    load_high_potential_runtime_tables_v1,
)
from strategic_v5.latest_public8_gpu import (  # noqa: E402
    KaitoV36CarryV1,
    X562CarryV1,
    deniz_v111_player_action_v1,
    initialize_kaito_v36_carry_v1,
    initialize_x562_carry_v1,
    kaito_v36_player_action_v1,
    x562_player_action_v1,
)
from strategic_v5.public_g02_gpu import (  # noqa: E402
    PublicG02CarryV1,
    initialize_public_g02_carry_v1,
    public_g02_player_action_v1,
    public_rc5_weed_player_action_v1,
)
from strategic_v5.public_v25_gpu import (  # noqa: E402
    public_v25_player_action_v1,
    public_v27_player_action_exact_v1,
)
from strategic_v5.public_v14_gpu import public_v14_player_action_v1  # noqa: E402
from strategic_v5.public_v21_gpu import public_v21_player_action_v1  # noqa: E402
from strategic_v5.public_v13_r3_gpu import public_v13_r3_player_action_v1  # noqa: E402
from strategic_v5.public_c68_gpu import public_c68_player_action_v1  # noqa: E402
from strategic_v5.public_four_hire_gpu import (  # noqa: E402
    initialize_four_hire_carry_v1,
    public_four_hire_player_action_v1,
)
from strategic_v5.public_tran_cashflow_gpu import (  # noqa: E402
    initialize_tran_cashflow_carry_v1,
    public_tran_cashflow_player_action_v1,
)
from strategic_v5.public_bruce_route1_gpu import (  # noqa: E402
    initialize_bruce_route1_carry_v1,
    public_bruce_route1_player_action_v1,
)
from strategic_v5.public_v19_control_gpu import public_v19_control_player_action_v1  # noqa: E402
from strategic_v5.public_v18_closed_loop_gpu import (  # noqa: E402
    initialize_v18_closed_loop_carry_v1,
    public_v18_closed_loop_player_action_v1,
)


OLD_IDS = np.asarray(
    # Strict official-1.32.7 parity only. G03/G05 and unaccepted gold proxies
    # are intentionally absent.
    (0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 27, 29, 34, 35),
    dtype=np.int32,
)

ROSTER = (
    {"name": "public_g01_boatlee_v16", "kind": "old", "old_id": 0},
    {"name": "public_g02_rc5_c166", "kind": "old", "old_id": 1},
    {"name": "public_g04_soil_rain", "kind": "old", "old_id": 3},
    {
        "name": "public_g06_v25",
        "kind": "old",
        "old_id": 5,
        "aliases": ("flex_multi_route_latest",),
    },
    {"name": "public_g07_c95", "kind": "old", "old_id": 6},
    {"name": "public_g08_v14", "kind": "old", "old_id": 7},
    {"name": "public_g09_c68_thunder", "kind": "old", "old_id": 8},
    {"name": "public_g10_four_hire", "kind": "old", "old_id": 9},
    {"name": "public_g11_v21", "kind": "old", "old_id": 10},
    {"name": "public_g12_v13_r3", "kind": "old", "old_id": 11},
    {"name": "public_g13_bruce_route1", "kind": "old", "old_id": 12},
    {"name": "public_g14_v19_control", "kind": "old", "old_id": 13},
    {"name": "public_g15_v18_closed_loop", "kind": "old", "old_id": 14},
    {"name": "public_g16_tran_cashflow", "kind": "old", "old_id": 15},
    {"name": "gold_proxy_rank07_junichiro_morita", "kind": "old", "old_id": 22},
    {"name": "gold_proxy_rank12_ai_b2b67_saas", "kind": "old", "old_id": 27},
    {"name": "gold_proxy_rank14_recursion", "kind": "old", "old_id": 29},
    {"name": "gold_proxy_rank19_manu_nicholas_jacob", "kind": "old", "old_id": 34},
    {"name": "local_prt_v6", "kind": "old", "old_id": 35},
    {
        "name": "boatlee_v20_multi_route",
        "kind": "high",
        "mode": MODE_BOATLEE,
        "aliases": ("boatlee_v20_latest", "kunal_2026_v1_latest"),
    },
    {
        "name": "rayk_k320_adaptive_rank1",
        "kind": "high",
        "mode": MODE_RAY_K320,
        "aliases": ("rayk_rank_agent_latest",),
    },
    {
        "name": "tetsutani_adaptive_premium_queue",
        "kind": "high",
        "mode": MODE_TETSUTANI,
    },
    {"name": "kaito_v27_midgame_reset", "kind": "legacy", "route_id": 8},
    {"name": "flexonafft_v59_multi_route", "kind": "legacy", "route_id": 9},
    {"name": "deniz_v111_8c4s_latest", "kind": "deniz", "route_id": 10},
    {"name": "kaito_v36_latest", "kind": "kaito", "route_id": 11},
    {"name": "x562_latest", "kind": "x562"},
    {
        "name": "tetsutani_adaptive_latest",
        "kind": "high",
        "mode": MODE_TETSUTANI_LATEST,
    },
)

GROUP_BY_ID = {
    **{index: "old_a" for index in range(0, 5)},
    **{index: "old_b" for index in range(5, 10)},
    **{index: "old_c" for index in range(10, 14)},
    **{index: "dynamic" for index in range(14, 19)},
    19: "high",
    20: "high",
    21: "high",
    22: "legacy",
    23: "legacy",
    24: "latest",
    25: "latest",
    26: "latest",
    27: "high",
}


class UniversalCarry(NamedTuple):
    old: FullOpponentCarry
    high: HighPotentialV20CarryV1
    legacy: PublicG02CarryV1
    deniz: PublicG02CarryV1
    kaito: KaitoV36CarryV1
    x562: X562CarryV1


class LatestCarry(NamedTuple):
    deniz: PublicG02CarryV1
    kaito: KaitoV36CarryV1
    x562: X562CarryV1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def select_tree(mask: jax.Array, selected, fallback):
    if hasattr(selected, "_fields"):
        return type(selected)(
            *(select_tree(mask, left, right) for left, right in zip(selected, fallback, strict=True))
        )
    return jnp.where(
        mask.reshape((mask.shape[0],) + (1,) * (selected.ndim - 1)),
        selected,
        fallback,
    )


def select_action(mask: jax.Array, selected: Action, fallback: Action) -> Action:
    return select_tree(mask, selected, fallback)


def pair_actions(left: Action, right: Action) -> Action:
    return Action(*(jnp.stack((a, b), axis=1) for a, b in zip(left, right, strict=True)))


def initialize_universal_carry(global_ids: jax.Array, old_id_lut, router) -> UniversalCarry:
    batch = global_ids.shape[0]
    safe = jnp.clip(global_ids, 0, old_id_lut.shape[0] - 1)
    old_ids = old_id_lut[safe]
    return UniversalCarry(
        old=initialize_full_opponent_carry(batch, old_ids, router),
        high=initialize_high_potential_v20_carry_v1(batch),
        legacy=initialize_public_g02_carry_v1(batch),
        deniz=initialize_public_g02_carry_v1(batch),
        kaito=initialize_kaito_v36_carry_v1(batch),
        x562=initialize_x562_carry_v1(batch),
    )


def make_advance(old_bank, latest_bank, runtime, tables, router, boatlee_trace):
    old_id_lut = jnp.asarray(
        np.concatenate((OLD_IDS, np.zeros((len(ROSTER) - len(OLD_IDS),), dtype=np.int32)))
    )

    def one_player(states, global_ids, carry: UniversalCarry, player: int):
        old_ids = old_id_lut[jnp.clip(global_ids, 0, old_id_lut.shape[0] - 1)]
        action, old_carry = full_opponent_action(
            states,
            tables,
            old_bank,
            boatlee_trace,
            old_ids,
            carry.old,
            player,
            router,
        )

        high_carry = carry.high
        for global_id, mode in (
            (19, MODE_BOATLEE),
            (20, MODE_RAY_K320),
            (21, MODE_TETSUTANI),
            (27, MODE_TETSUTANI_LATEST),
        ):
            candidate_action, candidate_carry = high_potential_v20_player_action_v1(
                states, tables, latest_bank, runtime, carry.high, player, mode
            )
            mask = global_ids == global_id
            action = select_action(mask, candidate_action, action)
            high_carry = select_tree(mask, candidate_carry, high_carry)

        legacy_carry = carry.legacy
        for global_id, route_id in ((22, 8), (23, 9)):
            route = jnp.full(global_ids.shape, route_id, dtype=jnp.int32)
            candidate_action, candidate_carry = public_v27_player_action_exact_v1(
                states, runtime, latest_bank, route, carry.legacy, player
            )
            mask = global_ids == global_id
            action = select_action(mask, candidate_action, action)
            legacy_carry = select_tree(mask, candidate_carry, legacy_carry)

        route = jnp.full(global_ids.shape, 10, dtype=jnp.int32)
        deniz_action, deniz_carry = deniz_v111_player_action_v1(
            states, latest_bank, route, carry.deniz, player
        )
        deniz_mask = global_ids == 24
        action = select_action(deniz_mask, deniz_action, action)
        deniz_carry = select_tree(deniz_mask, deniz_carry, carry.deniz)

        route = jnp.full(global_ids.shape, 11, dtype=jnp.int32)
        kaito_action, kaito_carry = kaito_v36_player_action_v1(
            states, runtime, latest_bank, route, carry.kaito, player
        )
        kaito_mask = global_ids == 25
        action = select_action(kaito_mask, kaito_action, action)
        kaito_carry = select_tree(kaito_mask, kaito_carry, carry.kaito)

        x562_action, x562_carry = x562_player_action_v1(
            states, runtime, latest_bank, carry.x562, player
        )
        x562_mask = global_ids == 26
        action = select_action(x562_mask, x562_action, action)
        x562_carry = select_tree(x562_mask, x562_carry, carry.x562)

        return action, UniversalCarry(
            old=old_carry,
            high=high_carry,
            legacy=legacy_carry,
            deniz=deniz_carry,
            kaito=kaito_carry,
            x562=x562_carry,
        )

    @jax.jit
    def advance(states, ids0, ids1, carry0, carry1, events):
        action0, carry0 = one_player(states, ids0, carry0, 0)
        action1, carry1 = one_player(states, ids1, carry1, 1)
        states = batched_step_sync(
            states, pair_actions(action0, action1), events, tables
        )
        return states, carry0, carry1

    return advance, old_id_lut


def initialize_group_carry(group: str, global_ids: jax.Array, old_id_lut, router):
    batch = global_ids.shape[0]
    old_ids = old_id_lut[jnp.clip(global_ids, 0, old_id_lut.shape[0] - 1)]
    if group in ("old_a", "old_b", "old_c"):
        return initialize_full_opponent_carry(batch, old_ids, router)
    if group == "dynamic":
        return initialize_dynamic_carry(batch, old_ids, router)
    if group == "high":
        return initialize_high_potential_v20_carry_v1(batch)
    if group == "legacy":
        return initialize_public_g02_carry_v1(batch)
    if group == "latest":
        return LatestCarry(
            deniz=initialize_public_g02_carry_v1(batch),
            kaito=initialize_kaito_v36_carry_v1(batch),
            x562=initialize_x562_carry_v1(batch),
        )
    raise ValueError(f"unknown controller group: {group}")


def make_group_advance(
    group0: str,
    group1: str,
    old_bank,
    latest_bank,
    runtime,
    tables,
    router,
    boatlee_trace,
    old_id_lut,
):
    """Compile one modest graph for one directed controller-family pair."""

    def group_action(group: str, states, global_ids, carry, player: int):
        if group == "old_a":
            boatlee_action, boatlee_carry = boatlee_player_action_v1(
                states, tables, boatlee_trace, carry.boatlee, player
            )
            g02_route = jnp.full(
                global_ids.shape, router.public_g02_skeleton_id, dtype=jnp.int32
            )
            g02_action, g02_carry = public_g02_player_action_v1(
                states,
                old_bank,
                g02_route,
                router.public_g02_rival_schedule,
                carry.public_g02,
                player,
            )
            g04_route = jnp.full(
                global_ids.shape, router.public_g04_skeleton_id, dtype=jnp.int32
            )
            g04_action, g04_carry = public_rc5_weed_player_action_v1(
                states, old_bank, g04_route, carry.public_g04, player
            )
            v25_route = jnp.full(
                global_ids.shape, router.public_v25_skeleton_id, dtype=jnp.int32
            )
            v25_action, v25_carry = public_v25_player_action_v1(
                states, tables, old_bank, v25_route, carry.public_v25, player
            )
            c95_route = jnp.full(
                global_ids.shape, router.c95_skeleton_id, dtype=jnp.int32
            )
            c95_action, c95_carry = c95_player_action_v1(
                states, tables, old_bank, c95_route, carry.c95, player
            )
            action = boatlee_action
            for agent_id, candidate in (
                (1, g02_action),
                (2, g04_action),
                (3, v25_action),
                (4, c95_action),
            ):
                action = select_action(global_ids == agent_id, candidate, action)
            return action, carry._replace(
                boatlee=select_tree(global_ids == 0, boatlee_carry, carry.boatlee),
                public_g02=select_tree(global_ids == 1, g02_carry, carry.public_g02),
                public_g04=select_tree(global_ids == 2, g04_carry, carry.public_g04),
                public_v25=select_tree(global_ids == 3, v25_carry, carry.public_v25),
                c95=select_tree(global_ids == 4, c95_carry, carry.c95),
            )

        if group == "old_b":
            v14_route = jnp.full(
                global_ids.shape, router.public_v14_skeleton_id, dtype=jnp.int32
            )
            v14_action, v14_carry = public_v14_player_action_v1(
                states, tables, old_bank, v14_route, carry.public_v14, player
            )
            c68_route = jnp.full(
                global_ids.shape, router.public_c68_skeleton_id, dtype=jnp.int32
            )
            c68_action, c68_carry = public_c68_player_action_v1(
                states, tables, old_bank, c68_route, carry.public_c68, player
            )
            four_kaito = jnp.full(
                global_ids.shape,
                router.public_four_hire_kaito_skeleton_id,
                dtype=jnp.int32,
            )
            four_ray = jnp.full(
                global_ids.shape,
                router.public_four_hire_ray_skeleton_id,
                dtype=jnp.int32,
            )
            four_action, four_carry = public_four_hire_player_action_v1(
                states,
                tables,
                old_bank,
                four_kaito,
                four_ray,
                carry.public_four_hire,
                player,
            )
            v21_route = jnp.full(
                global_ids.shape, router.public_v21_skeleton_id, dtype=jnp.int32
            )
            v21_action, v21_carry = public_v21_player_action_v1(
                states,
                old_bank,
                v21_route,
                router.public_v21_prototype_signature,
                router.public_v21_prototype_sales,
                carry.public_v21,
                player,
            )
            v13_route = jnp.full(
                global_ids.shape, router.public_v13_skeleton_id, dtype=jnp.int32
            )
            v13_action, v13_carry = public_v13_r3_player_action_v1(
                states,
                old_bank,
                v13_route,
                router.public_v13_hazard_enabled,
                router.public_v13_hazard_cap,
                carry.public_v13,
                player,
            )
            action = v14_action
            for agent_id, candidate in (
                (6, c68_action),
                (7, four_action),
                (8, v21_action),
                (9, v13_action),
            ):
                action = select_action(global_ids == agent_id, candidate, action)
            return action, carry._replace(
                public_v14=select_tree(global_ids == 5, v14_carry, carry.public_v14),
                public_c68=select_tree(global_ids == 6, c68_carry, carry.public_c68),
                public_four_hire=select_tree(
                    global_ids == 7, four_carry, carry.public_four_hire
                ),
                public_v21=select_tree(global_ids == 8, v21_carry, carry.public_v21),
                public_v13=select_tree(global_ids == 9, v13_carry, carry.public_v13),
            )

        if group == "old_c":
            bruce_route = jnp.full(
                global_ids.shape,
                router.public_bruce_route1_skeleton_id,
                dtype=jnp.int32,
            )
            bruce_action, bruce_carry = public_bruce_route1_player_action_v1(
                states, old_bank, bruce_route, carry.public_bruce_route1, player
            )
            v19_route = jnp.full(
                global_ids.shape,
                router.public_v19_control_skeleton_id,
                dtype=jnp.int32,
            )
            v19_action = public_v19_control_player_action_v1(
                states, old_bank, v19_route, player
            )
            v18_route = jnp.full(
                global_ids.shape,
                router.public_v18_board_skeleton_id,
                dtype=jnp.int32,
            )
            v18_action, v18_carry = public_v18_closed_loop_player_action_v1(
                states,
                old_bank,
                v18_route,
                router.public_v18_expert_skeleton_ids,
                router.public_v18_feature_scale,
                router.public_v18_market_bias_by_seat,
                router.public_v18_prototypes_by_day,
                router.public_v18_distance_strength,
                router.public_v18_stay_bonus,
                carry.public_v18_closed_loop,
                player,
            )
            tran_route = jnp.full(
                global_ids.shape,
                router.public_tran_cashflow_skeleton_id,
                dtype=jnp.int32,
            )
            tran_action, tran_carry = public_tran_cashflow_player_action_v1(
                states, old_bank, tran_route, carry.public_tran_cashflow, player
            )
            action = bruce_action
            for agent_id, candidate in (
                (11, v19_action),
                (12, v18_action),
                (13, tran_action),
            ):
                action = select_action(global_ids == agent_id, candidate, action)
            return action, carry._replace(
                public_bruce_route1=select_tree(
                    global_ids == 10, bruce_carry, carry.public_bruce_route1
                ),
                public_v18_closed_loop=select_tree(
                    global_ids == 12, v18_carry, carry.public_v18_closed_loop
                ),
                public_tran_cashflow=select_tree(
                    global_ids == 13, tran_carry, carry.public_tran_cashflow
                ),
            )

        if group == "dynamic":
            old_ids = old_id_lut[
                jnp.clip(global_ids, 0, old_id_lut.shape[0] - 1)
            ]
            return dynamic_player_action(
                states, tables, old_bank, old_ids, carry, player, router
            )

        if group == "high":
            action = None
            selected_carry = carry
            for agent_id, mode in (
                (19, MODE_BOATLEE),
                (20, MODE_RAY_K320),
                (21, MODE_TETSUTANI),
                (27, MODE_TETSUTANI_LATEST),
            ):
                candidate_action, candidate_carry = high_potential_v20_player_action_v1(
                    states, tables, latest_bank, runtime, carry, player, mode
                )
                if action is None:
                    action = candidate_action
                else:
                    action = select_action(global_ids == agent_id, candidate_action, action)
                selected_carry = select_tree(
                    global_ids == agent_id, candidate_carry, selected_carry
                )
            return action, selected_carry

        if group == "legacy":
            route8 = jnp.full(global_ids.shape, 8, dtype=jnp.int32)
            action8, carry8 = public_v27_player_action_exact_v1(
                states, runtime, latest_bank, route8, carry, player
            )
            route9 = jnp.full(global_ids.shape, 9, dtype=jnp.int32)
            action9, carry9 = public_v27_player_action_exact_v1(
                states, runtime, latest_bank, route9, carry, player
            )
            mask = global_ids == 23
            return select_action(mask, action9, action8), select_tree(mask, carry9, carry8)

        if group == "latest":
            route10 = jnp.full(global_ids.shape, 10, dtype=jnp.int32)
            deniz_action, deniz_carry = deniz_v111_player_action_v1(
                states, latest_bank, route10, carry.deniz, player
            )
            route11 = jnp.full(global_ids.shape, 11, dtype=jnp.int32)
            kaito_action, kaito_carry = kaito_v36_player_action_v1(
                states, runtime, latest_bank, route11, carry.kaito, player
            )
            x562_action, x562_carry = x562_player_action_v1(
                states, runtime, latest_bank, carry.x562, player
            )
            action = select_action(global_ids == 25, kaito_action, deniz_action)
            action = select_action(global_ids == 26, x562_action, action)
            return action, LatestCarry(
                deniz=select_tree(global_ids == 24, deniz_carry, carry.deniz),
                kaito=select_tree(global_ids == 25, kaito_carry, carry.kaito),
                x562=select_tree(global_ids == 26, x562_carry, carry.x562),
            )

        raise ValueError(f"unknown controller group: {group}")

    @jax.jit
    def advance(states, ids0, ids1, carry0, carry1, events):
        action0, carry0 = group_action(group0, states, ids0, carry0, 0)
        action1, carry1 = group_action(group1, states, ids1, carry1, 1)
        states = batched_step_sync(
            states, pair_actions(action0, action1), events, tables
        )
        return states, carry0, carry1

    return advance


def initialize_agent_carry(agent_id: int, batch: int, router):
    if agent_id == 0:
        return initialize_boatlee_player_carry_v1(batch)
    if agent_id in (1, 2, 3, 5, 6, 8, 9, 22, 23, 24):
        return initialize_public_g02_carry_v1(batch)
    if agent_id == 4:
        return initialize_c95_player_carry_v1(batch)
    if agent_id == 7:
        return initialize_four_hire_carry_v1(batch)
    if agent_id == 10:
        return initialize_bruce_route1_carry_v1(batch)
    if agent_id == 11:
        return jnp.zeros((batch,), dtype=jnp.int8)
    if agent_id == 12:
        return initialize_v18_closed_loop_carry_v1(batch)
    if agent_id == 13:
        return initialize_tran_cashflow_carry_v1(batch)
    if 14 <= agent_id <= 18:
        old_id = int(ROSTER[agent_id]["old_id"])
        return initialize_dynamic_carry(
            batch, jnp.full((batch,), old_id, dtype=jnp.int32), router
        )
    if agent_id in (19, 20, 21, 27):
        return initialize_high_potential_v20_carry_v1(batch)
    if agent_id == 25:
        return initialize_kaito_v36_carry_v1(batch)
    if agent_id == 26:
        return initialize_x562_carry_v1(batch)
    raise ValueError(f"no carry initializer for agent {agent_id}")


def make_agent_policy(
    agent_id: int,
    player: int,
    old_bank,
    latest_bank,
    runtime,
    tables,
    router,
    boatlee_trace,
):
    """One exact policy graph, independent of the opponent policy graph."""

    @jax.jit
    def policy(states, carry):
        shape = states.step.shape
        if agent_id == 0:
            return boatlee_player_action_v1(
                states, tables, boatlee_trace, carry, player
            )
        if agent_id == 1:
            route = jnp.full(shape, router.public_g02_skeleton_id, dtype=jnp.int32)
            return public_g02_player_action_v1(
                states,
                old_bank,
                route,
                router.public_g02_rival_schedule,
                carry,
                player,
            )
        if agent_id == 2:
            route = jnp.full(shape, router.public_g04_skeleton_id, dtype=jnp.int32)
            return public_rc5_weed_player_action_v1(
                states, old_bank, route, carry, player
            )
        if agent_id == 3:
            route = jnp.full(shape, router.public_v25_skeleton_id, dtype=jnp.int32)
            return public_v25_player_action_v1(
                states, tables, old_bank, route, carry, player
            )
        if agent_id == 4:
            route = jnp.full(shape, router.c95_skeleton_id, dtype=jnp.int32)
            return c95_player_action_v1(
                states, tables, old_bank, route, carry, player
            )
        if agent_id == 5:
            route = jnp.full(shape, router.public_v14_skeleton_id, dtype=jnp.int32)
            return public_v14_player_action_v1(
                states, tables, old_bank, route, carry, player
            )
        if agent_id == 6:
            route = jnp.full(shape, router.public_c68_skeleton_id, dtype=jnp.int32)
            return public_c68_player_action_v1(
                states, tables, old_bank, route, carry, player
            )
        if agent_id == 7:
            kaito = jnp.full(
                shape, router.public_four_hire_kaito_skeleton_id, dtype=jnp.int32
            )
            ray = jnp.full(
                shape, router.public_four_hire_ray_skeleton_id, dtype=jnp.int32
            )
            return public_four_hire_player_action_v1(
                states, tables, old_bank, kaito, ray, carry, player
            )
        if agent_id == 8:
            route = jnp.full(shape, router.public_v21_skeleton_id, dtype=jnp.int32)
            return public_v21_player_action_v1(
                states,
                old_bank,
                route,
                router.public_v21_prototype_signature,
                router.public_v21_prototype_sales,
                carry,
                player,
            )
        if agent_id == 9:
            route = jnp.full(shape, router.public_v13_skeleton_id, dtype=jnp.int32)
            return public_v13_r3_player_action_v1(
                states,
                old_bank,
                route,
                router.public_v13_hazard_enabled,
                router.public_v13_hazard_cap,
                carry,
                player,
            )
        if agent_id == 10:
            route = jnp.full(
                shape, router.public_bruce_route1_skeleton_id, dtype=jnp.int32
            )
            return public_bruce_route1_player_action_v1(
                states, old_bank, route, carry, player
            )
        if agent_id == 11:
            route = jnp.full(
                shape, router.public_v19_control_skeleton_id, dtype=jnp.int32
            )
            return public_v19_control_player_action_v1(
                states, old_bank, route, player
            ), carry
        if agent_id == 12:
            route = jnp.full(
                shape, router.public_v18_board_skeleton_id, dtype=jnp.int32
            )
            return public_v18_closed_loop_player_action_v1(
                states,
                old_bank,
                route,
                router.public_v18_expert_skeleton_ids,
                router.public_v18_feature_scale,
                router.public_v18_market_bias_by_seat,
                router.public_v18_prototypes_by_day,
                router.public_v18_distance_strength,
                router.public_v18_stay_bonus,
                carry,
                player,
            )
        if agent_id == 13:
            route = jnp.full(
                shape, router.public_tran_cashflow_skeleton_id, dtype=jnp.int32
            )
            return public_tran_cashflow_player_action_v1(
                states, old_bank, route, carry, player
            )
        if 14 <= agent_id <= 18:
            old_id = int(ROSTER[agent_id]["old_id"])
            opponent_ids = jnp.full(shape, old_id, dtype=jnp.int32)
            return dynamic_player_action(
                states, tables, old_bank, opponent_ids, carry, player, router
            )
        if agent_id in (19, 20, 21, 27):
            mode = int(ROSTER[agent_id]["mode"])
            return high_potential_v20_player_action_v1(
                states, tables, latest_bank, runtime, carry, player, mode
            )
        if agent_id in (22, 23):
            route = jnp.full(
                shape, int(ROSTER[agent_id]["route_id"]), dtype=jnp.int32
            )
            return public_v27_player_action_exact_v1(
                states, runtime, latest_bank, route, carry, player
            )
        if agent_id == 24:
            route = jnp.full(shape, 10, dtype=jnp.int32)
            return deniz_v111_player_action_v1(
                states, latest_bank, route, carry, player
            )
        if agent_id == 25:
            route = jnp.full(shape, 11, dtype=jnp.int32)
            return kaito_v36_player_action_v1(
                states, runtime, latest_bank, route, carry, player
            )
        if agent_id == 26:
            return x562_player_action_v1(
                states, runtime, latest_bank, carry, player
            )
        raise ValueError(f"no policy for agent {agent_id}")

    return policy


def make_simulator_step(tables):
    @jax.jit
    def simulator_step(states, action0, action1, events):
        return batched_step_sync(
            states, pair_actions(action0, action1), events, tables
        )

    return simulator_step


def fit_bradley_terry(pair_rows: list[dict], count: int) -> np.ndarray:
    """L2-stabilized Bradley-Terry MLE; ties count as half a win per side."""

    theta = np.zeros((count,), dtype=np.float64)
    reg = 0.05
    for _ in range(100):
        gradient = -reg * theta
        hessian = np.eye(count, dtype=np.float64) * reg
        for row in pair_rows:
            i, j = row["agent_a_id"], row["agent_b_id"]
            games = row["games"]
            observed = row["a_wins"] + 0.5 * row["ties"]
            delta = float(np.clip(theta[i] - theta[j], -30.0, 30.0))
            probability = 1.0 / (1.0 + np.exp(-delta))
            residual = observed - games * probability
            weight = games * probability * (1.0 - probability)
            gradient[i] += residual
            gradient[j] -= residual
            hessian[i, i] += weight
            hessian[j, j] += weight
            hessian[i, j] -= weight
            hessian[j, i] -= weight
        # Pin the otherwise unidentifiable global offset through the L2 term,
        # then explicitly center for stable presentation.
        update = np.linalg.solve(hessian + np.eye(count) * 1e-9, gradient)
        theta += update
        theta -= np.mean(theta)
        if float(np.max(np.abs(update))) < 1e-9:
            break
    return theta * (400.0 / np.log(10.0))


def write_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_report(payload: dict) -> str:
    lines = [
        "# 全部严格验收 JAX Agent 两两 100 局结果",
        "",
        f"生成时间：{payload['generated_at_utc']}",
        f"结论：**{payload['status']}**",
        "",
        "## 口径",
        "",
        f"- 参赛：{payload['roster_count']} 个去重后的完整 JAX Agent。",
        f"- 对局：{payload['pair_count']} 组 × 100 局 = {payload['games_total']:,} 局；每局 719 步。",
        "- 每组使用 50 个相同独立随机事件，并交换双方座位再跑 50 局。",
        "- 只纳入官方 1.32.7 逐步一致性验收通过的完整策略；路线骨架、未严格对齐代理和 BC/PPO 实验模型不计入。",
        "- 排名采用带轻微正则的 Bradley-Terry 分数；平局按双方各半胜计算。",
        "",
        "## 综合排名",
        "",
        "| 排名 | Agent | BT/Elo | 总胜率 | 胜-平-负 | 平均现金 | 平均分差 | 最差对手（得分率） |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload.get("ranking", []):
        lines.append(
            f"| {row['rank']} | {row['agent']} | {row['bt_elo']:+.1f} | "
            f"{100 * row['score_rate']:.1f}% | {row['wins']}-{row['ties']}-{row['losses']} | "
            f"{row['mean_cash']:.0f} | {row['mean_margin']:+.0f} | "
            f"{row['worst_opponent']} ({100 * row['worst_score_rate']:.1f}%) |"
        )
    lines += [
        "",
        "## 运行与完整性",
        "",
        f"- GPU：{payload.get('device', 'UNKNOWN')}",
        f"- 实测吞吐：{payload.get('transitions_per_second', 0):,.0f} transitions/s",
        f"- Arena 运行时间（不含首次编译）：{payload.get('arena_seconds', 0):,.1f} 秒",
        f"- 全部终局完成：{payload.get('all_done', False)}",
        f"- price LUT 越界：{payload.get('price_lut_oob_total', 0)}",
        f"- hand cap / market loop cap：{payload.get('hand_cap_hits_total', 0)} / {payload.get('market_loop_cap_hits_total', 0)}",
        "",
        "## 去重别名",
        "",
    ]
    aliases = payload.get("aliases", [])
    if aliases:
        for item in aliases:
            lines.append(f"- `{item['alias']}` → `{item['canonical']}`（{item['reason']}）")
    else:
        lines.append("- 无。")
    lines += [
        "",
        "## 解释边界",
        "",
        "本结果是在固定官方事件口径下的本地 JAX 对战强度，不等同于当前 Public 榜分；金牌 proxy 是已验收的本地模仿 Agent，不冒充原选手源码。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-bank",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/jax_full37_mixed_exact_proxy_bank_v1.npz",
    )
    parser.add_argument(
        "--old-receipt",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/receipts/jax_full37_mixed_exact_proxy_bank_v1.json",
    )
    parser.add_argument(
        "--latest-bank",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_route_bank_v1.npz",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/latest_public8_runtime_tables_v1.npz",
    )
    parser.add_argument("--seed-start", type=int, default=530001)
    parser.add_argument("--seeds", type=int, default=50)
    parser.add_argument("--pairs-per-chunk", type=int, default=8)
    parser.add_argument("--pair-limit", type=int, default=0)
    parser.add_argument("--steps", type=int, default=719)
    parser.add_argument(
        "--event-bank-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/artifacts/all_exact_round_robin_events_seed530001_n50_v1.npz",
    )
    parser.add_argument(
        "--receipt-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/receipts/all_exact_jax_round_robin_n28_seed530001_n50x2_v1.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "experiments/expert_business_agent_v2/reports/ALL_EXACT_JAX_ROUND_ROBIN_100_GAMES_20260820_ZH.md",
    )
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    if args.seeds < 1 or args.pairs_per_chunk < 1 or not (1 <= args.steps <= 719):
        raise ValueError("invalid seeds, pairs-per-chunk or steps")
    if len(OLD_IDS) != 19 or len(ROSTER) != 28:
        raise AssertionError("frozen roster changed unexpectedly")
    if tuple(row["old_id"] for row in ROSTER[:19]) != tuple(OLD_IDS.tolist()):
        raise AssertionError("old roster/LUT mismatch")

    cache = ROOT / ".jax_cache" / "all_exact_round_robin"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))

    old_receipt_path = args.old_receipt.resolve()
    old_receipt = json.loads(old_receipt_path.read_text(encoding="utf-8"))
    old_bank = load_bank(args.old_bank.resolve())
    latest_bank = load_bank(args.latest_bank.resolve())
    runtime = load_high_potential_runtime_tables_v1(args.runtime.resolve())
    router = build_router_arrays(old_receipt)
    tables = load_tables()
    boatlee_trace = load_boatlee_trace_v1()
    old_id_lut = jnp.asarray(
        np.concatenate(
            (OLD_IDS, np.zeros((len(ROSTER) - len(OLD_IDS),), dtype=np.int32))
        )
    )

    seeds = np.arange(args.seed_start, args.seed_start + args.seeds, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    event_path = args.event_bank_output.resolve()
    event_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(event_path, event_seeds=seeds, weed_spawn=weed, shop_choice=shops)

    pairs = list(itertools.combinations(range(len(ROSTER)), 2))
    if args.pair_limit:
        pairs = pairs[: args.pair_limit]
    games_per_pair = args.seeds * 2
    pair_rows: list[dict] = []
    compile_seconds = 0.0
    compile_records: list[dict] = []
    policies = {}
    simulator_step = make_simulator_step(tables)
    simulator_compiled = False
    arena_seconds = 0.0
    all_done = True
    hand_cap_hits_total = 0
    market_loop_cap_hits_total = 0
    price_lut_oob_total = 0
    started_all = perf_counter()
    device = jax.devices()[0]

    aliases = [
        {"alias": alias, "canonical": row["name"], "reason": "source/action semantics identical"}
        for row in ROSTER
        for alias in row.get("aliases", ())
    ]
    provenance_paths = (
        args.old_bank.resolve(),
        old_receipt_path,
        args.latest_bank.resolve(),
        args.runtime.resolve(),
        ROOT / "experiments/expert_business_agent_v2/receipts/weak6_jax_stepwise_parity_v3.json",
        ROOT / "experiments/expert_business_agent_v2/receipts/high_potential_exact5_historical_regression_after_latest8_v1.json",
        ROOT / "experiments/expert_business_agent_v2/receipts/latest_public8_jax_acceptance_v1.json",
        Path(__file__).resolve(),
    )

    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    for pair_index, (left, right) in enumerate(pairs):
        orientation_money = []
        pair_seconds = 0.0
        for orientation, (agent0, agent1) in enumerate(((left, right), (right, left))):
            states = jax.vmap(reset)(jnp.asarray(seeds))
            carry0 = initialize_agent_carry(agent0, args.seeds, router)
            carry1 = initialize_agent_carry(agent1, args.seeds, router)
            key0, key1 = (agent0, 0), (agent1, 1)
            new0, new1 = key0 not in policies, key1 not in policies
            if new0:
                policies[key0] = make_agent_policy(
                    agent0,
                    0,
                    old_bank,
                    latest_bank,
                    runtime,
                    tables,
                    router,
                    boatlee_trace,
                )
            if new1:
                policies[key1] = make_agent_policy(
                    agent1,
                    1,
                    old_bank,
                    latest_bank,
                    runtime,
                    tables,
                    router,
                    boatlee_trace,
                )
            policy0, policy1 = policies[key0], policies[key1]

            match_started = perf_counter()
            compile_for_match = 0.0
            for step in range(args.steps):
                call_started = perf_counter()
                action0, carry0 = policy0(states, carry0)
                if new0 and step == 0:
                    jax.block_until_ready(action0.unit_op)
                    seconds = perf_counter() - call_started
                    compile_for_match += seconds
                    compile_records.append(
                        {"agent_id": agent0, "agent": ROSTER[agent0]["name"], "seat": 0, "seconds": seconds}
                    )
                    print(json.dumps({"phase": "compiled_policy", **compile_records[-1]}), flush=True)

                call_started = perf_counter()
                action1, carry1 = policy1(states, carry1)
                if new1 and step == 0:
                    jax.block_until_ready(action1.unit_op)
                    seconds = perf_counter() - call_started
                    compile_for_match += seconds
                    compile_records.append(
                        {"agent_id": agent1, "agent": ROSTER[agent1]["name"], "seat": 1, "seconds": seconds}
                    )
                    print(json.dumps({"phase": "compiled_policy", **compile_records[-1]}), flush=True)

                call_started = perf_counter()
                states = simulator_step(states, action0, action1, events)
                if not simulator_compiled and step == 0:
                    jax.block_until_ready(states.step)
                    seconds = perf_counter() - call_started
                    compile_for_match += seconds
                    compile_records.append({"agent": "simulator_step", "seat": -1, "seconds": seconds})
                    simulator_compiled = True
                    print(json.dumps({"phase": "compiled_simulator", "seconds": seconds}), flush=True)

            jax.block_until_ready(states.money)
            terminal = jax.device_get(states)
            match_seconds = perf_counter() - match_started
            compile_seconds += compile_for_match
            steady = max(0.0, match_seconds - compile_for_match)
            arena_seconds += steady
            pair_seconds += match_seconds
            orientation_money.append(np.asarray(terminal.money, dtype=np.int64))
            done = np.asarray(terminal.done)
            all_done = all_done and bool(np.all(done)) if args.steps == 719 else all_done
            hand_cap_hits_total += int(np.sum(np.asarray(terminal.hand_cap_hits)))
            market_loop_cap_hits_total += int(np.sum(np.asarray(terminal.market_loop_cap_hits)))
            price_lut_oob_total += int(np.sum(np.asarray(terminal.price_lut_oob)))

        first, second = orientation_money
        a_cash = np.concatenate((first[:, 0], second[:, 1]))
        b_cash = np.concatenate((first[:, 1], second[:, 0]))
        margins = a_cash - b_cash
        wins = int(np.sum(margins > 0))
        losses = int(np.sum(margins < 0))
        ties = int(np.sum(margins == 0))
        pair_rows.append(
            {
                "agent_a_id": left,
                "agent_a": ROSTER[left]["name"],
                "agent_b_id": right,
                "agent_b": ROSTER[right]["name"],
                "games": games_per_pair,
                "a_wins": wins,
                "ties": ties,
                "a_losses": losses,
                "a_score_rate": (wins + 0.5 * ties) / games_per_pair,
                "a_mean_cash": float(np.mean(a_cash)),
                "b_mean_cash": float(np.mean(b_cash)),
                "a_mean_margin": float(np.mean(margins)),
                "a_median_margin": float(np.median(margins)),
                "a_seat0_wins": int(np.sum((first[:, 0] - first[:, 1]) > 0)),
                "a_seat1_wins": int(np.sum((second[:, 1] - second[:, 0]) > 0)),
            }
        )

        checkpoint = {
            "schema": "kaggriculture.all_exact_jax_round_robin.v1",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "RUNNING",
            "roster_count": len(ROSTER),
            "pair_count_planned": len(pairs),
            "pair_count_complete": len(pair_rows),
            "games_per_pair": games_per_pair,
            "pair_results": pair_rows,
        }
        write_checkpoint(args.receipt_output.resolve(), checkpoint)
        print(
            json.dumps(
                {
                    "phase": "chunk",
                    "chunk": pair_index + 1,
                    "chunks": len(pairs),
                    "pairs_complete": len(pair_rows),
                    "pairs_total": len(pairs),
                    "seconds": pair_seconds,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    elapsed = perf_counter() - started_all
    full_run = args.steps == 719 and not args.pair_limit and args.seeds == 50
    ratings = fit_bradley_terry(pair_rows, len(ROSTER))
    totals = [
        {"wins": 0, "ties": 0, "losses": 0, "cash": [], "margin": [], "pairs": []}
        for _ in ROSTER
    ]
    for row in pair_rows:
        i, j = row["agent_a_id"], row["agent_b_id"]
        totals[i]["wins"] += row["a_wins"]
        totals[i]["ties"] += row["ties"]
        totals[i]["losses"] += row["a_losses"]
        totals[j]["wins"] += row["a_losses"]
        totals[j]["ties"] += row["ties"]
        totals[j]["losses"] += row["a_wins"]
        totals[i]["cash"].append(row["a_mean_cash"])
        totals[j]["cash"].append(row["b_mean_cash"])
        totals[i]["margin"].append(row["a_mean_margin"])
        totals[j]["margin"].append(-row["a_mean_margin"])
        totals[i]["pairs"].append((row["agent_b"], row["a_score_rate"]))
        totals[j]["pairs"].append((row["agent_a"], 1.0 - row["a_score_rate"]))

    order = np.argsort(-ratings)
    ranking = []
    for rank, agent_id in enumerate(order.tolist(), 1):
        total = totals[agent_id]
        games = total["wins"] + total["ties"] + total["losses"]
        worst_name, worst_score = min(total["pairs"], key=lambda item: item[1]) if total["pairs"] else ("N/A", 0.0)
        ranking.append(
            {
                "rank": rank,
                "agent_id": agent_id,
                "agent": ROSTER[agent_id]["name"],
                "bt_elo": float(ratings[agent_id]),
                "games": games,
                "wins": total["wins"],
                "ties": total["ties"],
                "losses": total["losses"],
                "score_rate": (total["wins"] + 0.5 * total["ties"]) / max(games, 1),
                "mean_cash": float(np.mean(total["cash"])) if total["cash"] else 0.0,
                "mean_margin": float(np.mean(total["margin"])) if total["margin"] else 0.0,
                "worst_opponent": worst_name,
                "worst_score_rate": float(worst_score),
            }
        )

    integrity_pass = (
        (all_done if args.steps == 719 else True)
        and hand_cap_hits_total == 0
        and market_loop_cap_hits_total == 0
        and price_lut_oob_total == 0
    )
    status = "PASS" if full_run and integrity_pass else ("SMOKE_PASS" if integrity_pass else "FAIL")
    games_total = len(pair_rows) * games_per_pair
    payload = {
        "schema": "kaggriculture.all_exact_jax_round_robin.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "official_package_version": "1.32.7",
        "backend": jax.default_backend(),
        "jax_version": jax.__version__,
        "device": str(device),
        "roster_count": len(ROSTER),
        "roster": [dict(row) for row in ROSTER],
        "aliases": aliases,
        "excluded": {
            "route_skeletons": 46,
            "reason": "not complete strict-parity agents",
            "experimental_bc_ppo": "excluded",
            "unaccepted_proxy_agents": "excluded",
        },
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "event_seeds": seeds.tolist(),
        "event_bank": str(event_path),
        "event_bank_sha256": sha256(event_path),
        "steps": args.steps,
        "seat_protocol": "50 seeds as A-seat0/B-seat1 plus same 50 seeds swapped",
        "games_per_pair": games_per_pair,
        "pair_count": len(pair_rows),
        "pair_count_planned": len(pairs),
        "games_total": games_total,
        "transitions_total": games_total * args.steps,
        "pairs_per_chunk": args.pairs_per_chunk,
        "compile_seconds": compile_seconds,
        "compile_records": compile_records,
        "arena_seconds": arena_seconds,
        "wall_seconds": elapsed,
        "transitions_per_second": games_total * args.steps / max(arena_seconds, 1e-9),
        "all_done": all_done,
        "hand_cap_hits_total": hand_cap_hits_total,
        "market_loop_cap_hits_total": market_loop_cap_hits_total,
        "price_lut_oob_total": price_lut_oob_total,
        "integrity_pass": integrity_pass,
        "provenance": [
            {"path": str(path), "sha256": sha256(path)} for path in provenance_paths
        ],
        "pair_results": pair_rows,
        "ranking": ranking,
        "truth_boundary": (
            "Gold proxy entries are strict-parity local imitation agents, not original gold source. "
            "This fixed-event local JAX ranking is not a current Public leaderboard score."
        ),
    }
    write_checkpoint(args.receipt_output.resolve(), payload)
    report_path = args.report_output.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(make_report(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": status,
                "receipt": str(args.receipt_output.resolve()),
                "report": str(report_path),
                "pairs": len(pair_rows),
                "games": games_total,
                "transitions_per_second": payload["transitions_per_second"],
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0 if integrity_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
