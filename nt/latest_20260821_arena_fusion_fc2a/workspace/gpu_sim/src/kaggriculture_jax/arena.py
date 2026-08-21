"""Heterogeneous-checkpoint GPU Arena and self-play rollout factories."""

from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .policy import (
    PolicyHeads,
    combine_player_actions,
    encode_observations,
    heads_to_player_action,
)
from .simulator import batched_step_sync
from .types import Events, State, StaticTables


PolicyApply = Callable[[object, jax.Array], PolicyHeads]


class ArenaResult(NamedTuple):
    final_state: State
    player0_outcome: jax.Array
    player0_value: jax.Array
    player1_value: jax.Array


def make_arena_rollout(
    apply_player0: PolicyApply,
    apply_player1: PolicyApply,
    *,
    rollout_steps: int = 719,
    deterministic: bool = True,
):
    """Build a JIT-able rollout for two possibly different architectures.

    Separate apply functions and parameter pytrees allow checkpoint A and B to
    have different hidden sizes/structures.  There are no host transfers inside
    the rollout.
    """

    def rollout(
        initial_states: State,
        events: Events,
        tables: StaticTables,
        params_player0: object,
        params_player1: object,
        key: jax.Array,
    ) -> ArenaResult:
        def body(carry, _):
            states, rng = carry
            rng, key0, key1 = jax.random.split(rng, 3)
            observations = jax.vmap(encode_observations)(states)
            heads0 = apply_player0(params_player0, observations[:, 0])
            heads1 = apply_player1(params_player1, observations[:, 1])
            action0, value0 = heads_to_player_action(
                heads0, key0, states.unit_active[:, 0], deterministic
            )
            action1, value1 = heads_to_player_action(
                heads1, key1, states.unit_active[:, 1], deterministic
            )
            actions = combine_player_actions(action0, action1)
            next_states = batched_step_sync(states, actions, events, tables)
            return (next_states, rng), (value0, value1)

        (final_states, _), values = jax.lax.scan(
            body, (initial_states, key), xs=None, length=rollout_steps
        )
        money_delta = final_states.money[:, 0] - final_states.money[:, 1]
        outcome = jnp.sign(money_delta).astype(jnp.int8)
        return ArenaResult(
            final_state=final_states,
            player0_outcome=outcome,
            player0_value=values[0][-1],
            player1_value=values[1][-1],
        )

    return rollout


class SideSwappedSummary(NamedTuple):
    games: jax.Array
    policy_a_wins: jax.Array
    policy_b_wins: jax.Array
    draws: jax.Array
    policy_a_score: jax.Array


def summarize_side_swapped(
    a_as_player0: ArenaResult, b_as_player0: ArenaResult
) -> SideSwappedSummary:
    # First result: A is p0. Second result: B is p0, so negate for A.
    a_outcomes = jnp.concatenate(
        (a_as_player0.player0_outcome, -b_as_player0.player0_outcome)
    )
    wins = jnp.sum(a_outcomes > 0)
    losses = jnp.sum(a_outcomes < 0)
    draws = jnp.sum(a_outcomes == 0)
    games = a_outcomes.size
    return SideSwappedSummary(
        games=jnp.asarray(games, dtype=jnp.int32),
        policy_a_wins=wins,
        policy_b_wins=losses,
        draws=draws,
        policy_a_score=(wins + 0.5 * draws) / games,
    )
