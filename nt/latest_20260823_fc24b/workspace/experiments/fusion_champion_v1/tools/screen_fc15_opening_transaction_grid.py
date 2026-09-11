#!/usr/bin/env python3
"""One-step exact JAX screen of FC15 opening transaction quantities."""

from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
    ROOT / "gpu_sim/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from kaggriculture_jax.constants import (  # noqa: E402
    MAX_MARKET_ORDERS,
    MAX_UNITS,
)
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402


BASE_MARKET_OP = np.asarray((1, 1, 1, 1, 1, 5, 5, 3, 3, 4), dtype=np.int8)
BASE_MARKET_ITEM = np.asarray((-1, -1, -1, -1, -1, 10, 11, 0, 4, 0), dtype=np.int8)
BASE_MARKET_AMOUNT = np.asarray((0, 0, 0, 0, 0, 2, 2, 7, 12, 6), dtype=np.int32)


def make_action(wheat_seed, melon_seed, wheat_product) -> Action:
    batch = int(len(wheat_seed))
    unit_op = np.zeros((batch, MAX_UNITS), dtype=np.int8)
    unit_item = np.full((batch, MAX_UNITS), -1, dtype=np.int8)
    unit_amount = np.zeros((batch, MAX_UNITS), dtype=np.int32)
    unit_op[:, 0] = 14
    unit_amount[:, 0] = 1
    market_op = np.tile(BASE_MARKET_OP, (batch, 1))
    market_item = np.tile(BASE_MARKET_ITEM, (batch, 1))
    market_amount = np.tile(BASE_MARKET_AMOUNT, (batch, 1))
    market_amount[:, 7] = wheat_seed
    market_amount[:, 8] = melon_seed
    market_amount[:, 9] = wheat_product
    return Action(
        unit_op=jnp.asarray(unit_op),
        unit_item=jnp.asarray(unit_item),
        unit_amount=jnp.asarray(unit_amount),
        unit_count=jnp.ones((batch,), dtype=jnp.int8),
        market_op=jnp.asarray(market_op),
        market_item=jnp.asarray(market_item),
        market_amount=jnp.asarray(market_amount),
        market_count=jnp.full((batch,), MAX_MARKET_ORDERS, dtype=jnp.int8),
    )


def main() -> int:
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    cache = ROOT / ".jax_cache/fusion_champion_v1"
    cache.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(cache))
    grid = np.asarray(
        [
            (wheat_seed, melon_seed, wheat_product)
            for wheat_seed in range(4, 11)
            for melon_seed in range(8, 16)
            for wheat_product in range(3, 10)
        ],
        dtype=np.int16,
    )
    batch = int(grid.shape[0])
    candidate = make_action(grid[:, 0], grid[:, 1], grid[:, 2])
    opponent = make_action(
        np.full(batch, 7, dtype=np.int16),
        np.full(batch, 12, dtype=np.int16),
        np.full(batch, 6, dtype=np.int16),
    )
    seeds = np.full(batch, 843999, dtype=np.int32)
    weed, shops = build_events_v1(seeds.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    states0 = jax.vmap(reset)(jnp.asarray(seeds))
    simulator = jax.jit(rr.make_simulator_step(load_tables()))

    results = []
    for seat in (0, 1):
        states = (
            simulator(states0, candidate, opponent, events)
            if seat == 0
            else simulator(states0, opponent, candidate, events)
        )
        results.append(jax.device_get(states))

    rows = []
    for index, (wheat_seed, melon_seed, wheat_product) in enumerate(grid.tolist()):
        seats = []
        for seat, state in enumerate(results):
            seats.append(
                {
                    "seat": seat,
                    "money": int(np.asarray(state.money)[index, seat]),
                    "wheat_seed": int(np.asarray(state.seeds)[index, seat, 0]),
                    "melon_seed": int(np.asarray(state.seeds)[index, seat, 4]),
                    "shed_wheat": int(np.asarray(state.shed)[index, seat, 0]),
                    "cow": int(np.asarray(state.shed)[index, seat, 10]),
                    "sheep": int(np.asarray(state.shed)[index, seat, 11]),
                    "hires": int(np.asarray(state.hires_today)[index, seat]),
                }
            )
        if all(row["money"] <= 20 for row in seats):
            economic_perturbation = max(
                abs(row["wheat_seed"] - 7) * 10
                + abs(row["melon_seed"] - 12) * 80
                + abs(row["shed_wheat"] - 5) * 25
                + abs(row["money"] - 22)
                for row in seats
            )
            rows.append(
                {
                    "requested": {
                        "wheat_seed": wheat_seed,
                        "melon_seed": melon_seed,
                        "wheat_product": wheat_product,
                    },
                    "post_step": seats,
                    "min_total_seed": min(
                        row["wheat_seed"] + row["melon_seed"] for row in seats
                    ),
                    "min_shed_wheat": min(row["shed_wheat"] for row in seats),
                    "max_money": max(row["money"] for row in seats),
                    "economic_perturbation_from_baseline": economic_perturbation,
                }
            )
    rows.sort(
        key=lambda row: (
            row["economic_perturbation_from_baseline"],
            -row["min_shed_wheat"],
            -row["min_total_seed"],
        )
    )
    print(
        {
            "grid_size": batch,
            "eligible_both_seats": len(rows),
            "top20": rows[:20],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
