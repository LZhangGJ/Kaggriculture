"""Parity-first JAX simulator for Kaggriculture."""

from .codec import encode_actions, stack_actions
from .arena import ArenaResult, make_arena_rollout, summarize_side_swapped
from .policy import PolicyHeads, PolicyValueNet, encode_observations
from .training import (
    PPOConfig,
    SelfPlayRollout,
    create_train_state,
    generalized_advantage_estimate,
    make_ppo_update,
    make_selfplay_collector,
)
from .state import empty_action, events_for_seed, load_event_bank, load_tables, reset
from .simulator import batched_step, batched_step_sync, step_env
from .types import Action, Events, State, StaticTables

__all__ = [
    "Action",
    "ArenaResult",
    "Events",
    "PolicyHeads",
    "PolicyValueNet",
    "PPOConfig",
    "SelfPlayRollout",
    "State",
    "StaticTables",
    "empty_action",
    "encode_actions",
    "encode_observations",
    "create_train_state",
    "generalized_advantage_estimate",
    "batched_step",
    "batched_step_sync",
    "events_for_seed",
    "load_event_bank",
    "load_tables",
    "make_arena_rollout",
    "make_ppo_update",
    "make_selfplay_collector",
    "reset",
    "stack_actions",
    "step_env",
    "summarize_side_swapped",
]
