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
from .simulator import (
    batched_project_action_phases,
    batched_project_action_phases_sync,
    batched_project_unit_phase,
    batched_step,
    batched_step_from_projected_unit_phase_sync,
    batched_step_sync,
    project_action_phases_env,
    project_unit_phase_env,
    step_env,
)
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
    "batched_step_from_projected_unit_phase_sync",
    "batched_step_sync",
    "batched_project_action_phases",
    "batched_project_action_phases_sync",
    "batched_project_unit_phase",
    "events_for_seed",
    "load_event_bank",
    "load_tables",
    "make_arena_rollout",
    "make_ppo_update",
    "make_selfplay_collector",
    "reset",
    "stack_actions",
    "step_env",
    "project_action_phases_env",
    "project_unit_phase_env",
    "summarize_side_swapped",
]
