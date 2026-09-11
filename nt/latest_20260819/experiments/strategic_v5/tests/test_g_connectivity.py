from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from kaggriculture_jax import load_event_bank, load_tables
from strategic_v5 import (
    full_learned_step_v1,
    full_two_policy_step_v1,
    initialize_full_learned_carry_v1,
    initialize_full_learned_params_v1,
    make_full_g0_audit_collector_v1,
)


TABLES = load_tables()


def _small_carry(batch_size: int, key_seed: int = 1):
    event_ids, bank = load_event_bank()
    events = jax.tree.map(lambda value: value[:batch_size], bank)
    return initialize_full_learned_carry_v1(
        jnp.asarray(event_ids[:batch_size]), events, jax.random.key(key_seed)
    )


def _assert_tree_equal(left, right) -> None:
    for one, two in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        if jax.dtypes.issubdtype(one.dtype, jax.dtypes.prng_key):
            np.testing.assert_array_equal(
                np.asarray(jax.random.key_data(one)),
                np.asarray(jax.random.key_data(two)),
            )
        elif jnp.issubdtype(one.dtype, jnp.inexact):
            np.testing.assert_allclose(np.asarray(one), np.asarray(two), rtol=1e-6, atol=1e-6)
        else:
            np.testing.assert_array_equal(np.asarray(one), np.asarray(two))


def test_two_policy_step_matches_shared_parameter_selfplay() -> None:
    carry = _small_carry(2, 3)
    params = initialize_full_learned_params_v1(jax.random.key(4))
    selfplay = jax.jit(
        lambda value: full_learned_step_v1(
            value, TABLES, params, deterministic=True, decision_interval=8
        )
    )
    paired = jax.jit(
        lambda value: full_two_policy_step_v1(
            value,
            TABLES,
            params,
            params,
            deterministic=True,
            decision_interval=8,
        )
    )
    selfplay_carry, _ = selfplay(carry)
    paired_carry = paired(carry)
    _assert_tree_equal(selfplay_carry, paired_carry)


def test_g0_audit_collector_saves_task_masks_and_ledgers() -> None:
    carry = _small_carry(2, 5)
    params = initialize_full_learned_params_v1(jax.random.key(6))
    collector = jax.jit(
        make_full_g0_audit_collector_v1(
            rollout_steps=9,
            sample_stride=8,
            deterministic=False,
            decision_interval=8,
            include_final_sample=True,
        )
    )
    rollout = collector(carry, TABLES, params)
    assert rollout.transitions.old_logprob.shape == (2, 2, 2)
    np.testing.assert_array_equal(
        np.asarray(rollout.snapshots.state_step[:, 0]), np.asarray([0, 8])
    )
    assert rollout.transitions.task_masks.shape[:4] == (2, 2, 2, 8)
    assert rollout.transitions.task_selected_indices.shape == (2, 2, 2, 8)
    assert rollout.snapshots.player0_controller.unit_tasks.task_type.shape[:2] == (2, 2)
    assert rollout.snapshots.player1_controller.market_tasks.task_type.shape[:2] == (2, 2)
    for ledger in (
        rollout.snapshots.player0_ledger,
        rollout.snapshots.player1_ledger,
    ):
        assert bool(jnp.all(ledger.cash_reserved <= ledger.cash_available))
        assert bool(jnp.all(ledger.shed_reserved_out <= ledger.shed_available))
        assert bool(
            jnp.all(
                ledger.unit_inventory_reserved
                <= ledger.unit_inventory_available
            )
        )
        assert bool(jnp.all(ledger.seeds_reserved <= ledger.seeds_available))
