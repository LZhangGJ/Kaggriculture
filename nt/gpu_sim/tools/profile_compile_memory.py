"""Report XLA buffer estimates for simulator sub-systems."""

from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp

import kaggriculture_jax.simulator as sim
from kaggriculture_jax import Events, empty_action, load_event_bank, load_tables, reset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    args = parser.parse_args()
    n = args.batch_size
    _, bank = load_event_bank()
    tables = load_tables()
    states = jax.vmap(reset)(jnp.arange(n, dtype=jnp.int32) % 256)
    actions = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (n, *value.shape)), empty_action()
    )
    indices = jnp.arange(n) % 256
    events = Events(bank.weed_spawn[indices], bank.shop_choice[indices])
    day = jnp.zeros((n,), dtype=jnp.int8)
    functions = [
        (
            "units",
            jax.vmap(sim._process_units, in_axes=(0, 0, 0)),
            (states, actions, day),
        ),
        (
            "market",
            jax.vmap(sim._process_market, in_axes=(0, 0, None)),
            (states, actions, tables),
        ),
        (
            "town",
            jax.vmap(sim._town_consume, in_axes=(0, None)),
            (states, tables),
        ),
        ("decay", jax.vmap(sim._decay_plants), (states,)),
        (
            "end_of_day",
            jax.vmap(sim._end_of_day, in_axes=(0, 0, 0)),
            (states, events, day),
        ),
        (
            "step",
            jax.vmap(sim.step_env, in_axes=(0, 0, 0, None)),
            (states, actions, events, tables),
        ),
    ]
    for name, function, fn_args in functions:
        compiled = jax.jit(function).lower(*fn_args).compile()
        print(name, compiled.memory_analysis())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
