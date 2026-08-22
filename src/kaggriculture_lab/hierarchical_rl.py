"""On-policy and exact action-replay helpers for the V7 hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from .candidate_proposal import evaluate_proposal_draws
from .hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    HierarchicalPolicyBatch,
    HierarchicalReplayBatch,
    _as_tensors,
    _policy_batch_impl,
    encode_hierarchical_batch,
    evaluate_budget_action_features,
    evaluate_market_sequence,
    evaluate_task_matching,
    opponent_conditioned_mode_logits,
)
from .hierarchical_schema import (
    BUDGET_CATEGORIES,
    BudgetPlan,
    HierarchicalMemory,
    PlannedMarketBudget,
    STRATEGY_MODES,
)


@dataclass(frozen=True)
class ReplayEvaluation:
    log_probs: torch.Tensor
    entropies: torch.Tensor
    values: torch.Tensor
    proposal_log_probs: torch.Tensor
    proposal_entropies: torch.Tensor
    budget_log_probs: torch.Tensor
    budget_entropies: torch.Tensor
    horizon_values: torch.Tensor


def hierarchical_actor_critic_batch(
    model: HierarchicalKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: list[HierarchicalMemory],
) -> HierarchicalPolicyBatch:
    """Sample a differentiable constrained action for every observation."""

    return _policy_batch_impl(
        model,
        observations,
        device,
        memories,
        deterministic=False,
    )


def _evaluate_mode_replay(
    mode_logits: torch.Tensor,
    switch_logits: torch.Tensor,
    replay: HierarchicalReplayBatch,
    opponent_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mode_logits = opponent_conditioned_mode_logits(
        mode_logits, opponent_uncertainty, replay.observations
    )
    modes = replay.strategy_modes.to(mode_logits.device)
    switches = replay.mode_switches.to(mode_logits.device)
    previous = replay.previous_modes.to(mode_logits.device)
    initial = replay.initial_mode_active.to(mode_logits.device)
    daily = replay.daily_mode_active.to(mode_logits.device)
    mode_distribution = torch.distributions.Categorical(logits=mode_logits)
    switch_distribution = torch.distributions.Categorical(logits=switch_logits)
    log_probs = torch.zeros(mode_logits.shape[0], device=mode_logits.device)
    entropies = torch.zeros_like(log_probs)
    if bool(initial.any()):
        log_probs[initial] = mode_distribution.log_prob(modes)[initial]
        entropies[initial] = mode_distribution.entropy()[initial]
    if bool(daily.any()):
        if bool((switches[daily] < 0).any()):
            raise ValueError("daily mode replay is missing KEEP/SWITCH choices")
        log_probs[daily] = switch_distribution.log_prob(switches)[daily]
        entropies[daily] = switch_distribution.entropy()[daily]
        switched = daily & (switches == 1)
        kept = daily & (switches == 0)
        if bool(kept.any()) and not bool((modes[kept] == previous[kept]).all()):
            raise ValueError("KEEP replay changed strategy mode")
        for row in torch.nonzero(switched, as_tuple=False).flatten().tolist():
            allowed = torch.ones(
                mode_logits.shape[-1], dtype=torch.bool, device=mode_logits.device
            )
            allowed[int(previous[row])] = False
            floor = torch.finfo(mode_logits.dtype).min
            distribution = torch.distributions.Categorical(
                logits=mode_logits[row].masked_fill(~allowed, floor)
            )
            log_probs[row] = log_probs[row] + distribution.log_prob(modes[row])
            entropies[row] = entropies[row] + distribution.entropy()
    return log_probs, entropies


def evaluate_hierarchical_replay(
    model: HierarchicalKaggriculturePolicy,
    replay: HierarchicalReplayBatch,
    device: torch.device | str,
) -> ReplayEvaluation:
    """Recompute every stochastic factor for the exact sampled V7 action."""

    tensors = _as_tensors(replay.tensors, device)
    context = model.encode_state(*tensors[:4], tensors[16], tensors[17])
    proposal = model.propose(context)
    proposal_log_prob, proposal_entropy = evaluate_proposal_draws(
        proposal.task_logits,
        replay.observations,
        replay.proposal_draws,
        locked_mask=tensors[0][:, 0, 1],
    )
    modes = replay.strategy_modes.to(context.state.device)
    mode_log_prob, mode_entropy = _evaluate_mode_replay(
        context.mode_logits,
        context.switch_logits,
        replay,
        context.opponent_uncertainty,
    )
    budget_output = model.predict_budget(context, modes)
    budget_log_prob, budget_entropy = evaluate_budget_action_features(
        budget_output,
        replay.budget_plan_features.to(
            device=context.state.device, dtype=context.state.dtype
        ),
        replay.budget_action_active.to(context.state.device),
    )
    assignment_logits = model.score_assignments(
        context, *tensors[4:12], modes
    )
    assignment_log_prob, assignment_entropy = evaluate_task_matching(
        assignment_logits, tensors[12], tensors[13], replay.matching
    )
    assignments = replay.assignments.to(context.state.device)
    valid_assignments = assignments >= 0
    assignment_weights = torch.zeros_like(
        assignment_logits, dtype=context.state.dtype
    )
    assignment_weights.scatter_(
        2,
        assignments.clamp_min(0).unsqueeze(-1),
        valid_assignments.to(context.state.dtype).unsqueeze(-1),
    )
    plan_context = model.assignment_plan_context(
        tensors[5], tensors[6], tensors[11], tensors[15], assignment_weights
    )
    budget_features = replay.budget_plan_features.to(
        device=context.state.device, dtype=context.state.dtype
    )
    market_context = plan_context + model.budget_context_projection(
        budget_features
    )
    budget_plans = [
        BudgetPlan.from_fractions(
            day=int(day),
            money=float(money),
            reserve_fraction=float(features[0]),
            category_fractions=features[1 : 1 + len(BUDGET_CATEGORIES)],
            emergency_fraction=float(features[-1]),
            minimum_reserve_fraction=float(minimum_reserve),
        )
        for features, money, day, minimum_reserve in zip(
            replay.budget_plan_features.tolist(),
            replay.budget_cash_at_plan.tolist(),
            replay.budget_days.tolist(),
            replay.budget_minimum_reserve.tolist(),
            strict=True,
        )
    ]
    budget_ledgers = [
        PlannedMarketBudget.from_observation(
            observation,
            plan,
            spent_by_category=spent,
            emergency_spent=emergency,
        )
        for observation, plan, spent, emergency in zip(
            replay.observations,
            budget_plans,
            replay.budget_spent_before.tolist(),
            replay.budget_emergency_before.tolist(),
            strict=True,
        )
    ]
    required_orders = tensors[15].gather(
        2, assignments.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    required_orders = torch.where(
        valid_assignments, required_orders, torch.zeros_like(required_orders)
    )
    market_log_prob, market_entropy = evaluate_market_sequence(
        model,
        context,
        modes,
        tensors[14],
        replay.observations,
        replay.market_indices,
        plan_context=market_context,
        required_orders=required_orders,
        budget_ledgers=budget_ledgers,
    )
    return ReplayEvaluation(
        log_probs=(
            proposal_log_prob
            + mode_log_prob
            + budget_log_prob
            + assignment_log_prob
            + market_log_prob
        ),
        entropies=(
            proposal_entropy
            + mode_entropy
            + budget_entropy
            + assignment_entropy
            + market_entropy
        )
        / (
            4.0
            + replay.budget_action_active.to(
                device=context.state.device, dtype=context.state.dtype
            )
        ),
        values=context.values,
        proposal_log_probs=proposal_log_prob,
        proposal_entropies=proposal_entropy,
        budget_log_probs=budget_log_prob,
        budget_entropies=budget_entropy,
        horizon_values=context.horizon_values,
    )


@torch.no_grad()
def hierarchical_values(
    model: HierarchicalKaggriculturePolicy,
    observations: Sequence[Any],
    device: torch.device | str,
    memories: list[HierarchicalMemory],
) -> torch.Tensor:
    """Evaluate bootstrap values without sampling tasks, modes, or market orders."""

    batch = encode_hierarchical_batch(observations, memories)
    tensors = _as_tensors(batch, device)
    return model.encode_state(*tensors[:4], tensors[16], tensors[17]).values
