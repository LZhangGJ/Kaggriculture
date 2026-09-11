from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import (
    Events,
    PPOConfig,
    PolicyValueNet,
    create_train_state,
    encode_observations,
    generalized_advantage_estimate,
    load_event_bank,
    load_tables,
    make_ppo_update,
    make_selfplay_collector,
    reset,
)


def test_gpu_selfplay_collection_and_real_ppo_gradient_update() -> None:
    seed_values, bank = load_event_bank()
    tables = load_tables()
    batch_size = 4
    states = jax.vmap(reset)(jnp.asarray(seed_values[:batch_size], dtype=jnp.int32))
    events = Events(bank.weed_spawn[:batch_size], bank.shop_choice[:batch_size])
    model = PolicyValueNet(hidden_sizes=(16,))
    sample = encode_observations(reset(0))[0:1]
    params = model.init(jax.random.key(10), sample)
    config = PPOConfig(learning_rate=1e-3)
    train_state = create_train_state(model.apply, params, config)

    collector = jax.jit(
        make_selfplay_collector(model.apply, rollout_steps=4, deterministic=False)
    )
    rollout = collector(states, events, tables, train_state.params, jax.random.key(11))
    advantages, returns = generalized_advantage_estimate(
        rollout.transitions,
        rollout.bootstrap_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )
    updater = jax.jit(make_ppo_update(model.apply, config))
    new_state, metrics = updater(
        train_state, rollout.transitions, advantages, returns
    )
    jax.block_until_ready((new_state, metrics))

    assert int(new_state.step) == 1
    assert rollout.transitions.observations.shape[:3] == (4, batch_size, 2)
    assert rollout.final_state.step.devices().pop().platform == "gpu"
    assert all(np.isfinite(np.asarray(value)) for value in metrics)
    changed = any(
        not np.array_equal(np.asarray(before), np.asarray(after))
        for before, after in zip(
            jax.tree.leaves(train_state.params),
            jax.tree.leaves(new_state.params),
            strict=True,
        )
    )
    assert changed
