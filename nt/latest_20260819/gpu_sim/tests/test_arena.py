from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    Events,
    PolicyValueNet,
    encode_observations,
    load_event_bank,
    load_tables,
    make_arena_rollout,
    reset,
    summarize_side_swapped,
)


def test_actor_observation_hides_opponent_private_state() -> None:
    state = reset(0)
    baseline = encode_observations(state)
    changed = state._replace(
        shed=state.shed.at[1, 0].set(77),
        seeds=state.seeds.at[1, 0].set(55),
        unit_inventory=state.unit_inventory.at[1, 0, 0].set(33),
    )
    modified = encode_observations(changed)
    np.testing.assert_array_equal(baseline[0], modified[0])
    assert not np.array_equal(np.asarray(baseline[1]), np.asarray(modified[1]))


def test_heterogeneous_checkpoint_arena_runs_entirely_on_gpu() -> None:
    seed_values, bank = load_event_bank()
    tables = load_tables()
    batch_size = 8
    states = jax.vmap(reset)(jnp.asarray(seed_values[:batch_size], dtype=jnp.int32))
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    sample = encode_observations(reset(0))[0:1]
    model_a = PolicyValueNet(hidden_sizes=(32,))
    model_b = PolicyValueNet(hidden_sizes=(48, 24))
    params_a = model_a.init(jax.random.key(1), sample)
    params_b = model_b.init(jax.random.key(2), sample)
    assert jax.tree.leaves(params_a)[0].shape != jax.tree.leaves(params_b)[0].shape

    a_vs_b = jax.jit(
        make_arena_rollout(
            model_a.apply, model_b.apply, rollout_steps=8, deterministic=True
        )
    )
    b_vs_a = jax.jit(
        make_arena_rollout(
            model_b.apply, model_a.apply, rollout_steps=8, deterministic=True
        )
    )
    first = a_vs_b(states, events, tables, params_a, params_b, jax.random.key(3))
    second = b_vs_a(states, events, tables, params_b, params_a, jax.random.key(4))
    jax.block_until_ready((first, second))
    np.testing.assert_array_equal(first.final_state.step, np.full(batch_size, 8))
    assert first.player0_outcome.shape == (batch_size,)
    assert first.final_state.step.devices().pop().platform == "gpu"
    summary = summarize_side_swapped(first, second)
    assert int(summary.games) == 2 * batch_size
    assert int(summary.policy_a_wins + summary.policy_b_wins + summary.draws) == 2 * batch_size

