"""Clipped PPO self-play for the replayable V7 hierarchical policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.board_rl import (
    generalized_advantage_estimates,
    multi_horizon_transition_targets,
    paired_liquidatable_net_asset_potentials,
    productive_resource_potentials,
    relative_money_potential,
)
from kaggriculture_lab.gpu_policy import flatten_environment_observations
from kaggriculture_lab.hierarchical_policy import HierarchicalKaggriculturePolicy
from kaggriculture_lab.hierarchical_rl import (
    evaluate_hierarchical_replay,
    hierarchical_actor_critic_batch,
    hierarchical_values,
)
from kaggriculture_lab.hierarchical_schema import HierarchicalMemory
from kaggriculture_lab.league import (
    assemble_seat_actions,
    league_mixture_from_payoff,
    sample_league_member,
)


def load_compatible_checkpoint(
    model: HierarchicalKaggriculturePolicy,
    state_dict: dict[str, torch.Tensor],
) -> list[str]:
    """Load matching tensors while leaving new V8 heads initialized."""

    current = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in current and current[key].shape == value.shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(set(state_dict) - set(compatible))


def model_from_checkpoint(
    checkpoint: dict[str, object],
    defaults: argparse.Namespace,
    device: torch.device,
) -> HierarchicalKaggriculturePolicy:
    model = HierarchicalKaggriculturePolicy(
        hidden_size=int(checkpoint.get("hidden_size", defaults.hidden_size)),
        board_width=int(checkpoint.get("board_width", defaults.board_width)),
        d_model=int(checkpoint.get("d_model", defaults.d_model)),
        transformer_layers=int(
            checkpoint.get("transformer_layers", defaults.transformer_layers)
        ),
        transformer_heads=int(
            checkpoint.get("transformer_heads", defaults.transformer_heads)
        ),
        proposal_top_k=int(
            checkpoint.get("proposal_top_k", defaults.proposal_top_k)
        ),
        opponent_clusters=int(checkpoint.get("opponent_clusters", 8)),
    ).to(device)
    load_compatible_checkpoint(model, checkpoint["model"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=16)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--board-width", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--proposal-top-k", type=int, default=24)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--win-bonus", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--horizon-value-coef", type=float, default=0.20)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-clip", type=float, default=0.20)
    parser.add_argument("--value-clip", type=float, default=0.20)
    parser.add_argument("--minibatch-steps", type=int, default=8)
    parser.add_argument("--target-kl", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Optional V3-V7 behavior-cloning or RL checkpoint.",
    )
    parser.add_argument(
        "--league-checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Frozen opponent checkpoint; repeat to build a league.",
    )
    parser.add_argument(
        "--league-payoff-json",
        type=Path,
        help="Optional cluster payoff report supplying PSRO/CVaR mixture weights.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hierarchical_ppo_v7.pt")
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="JSON executor-failure log path (defaults beside --output).",
    )
    args = parser.parse_args()
    if min(
        args.envs,
        args.updates,
        args.rollout_steps,
        args.episode_steps,
        args.ppo_epochs,
        args.minibatch_steps,
    ) <= 0:
        parser.error("environment and rollout counts must be positive")
    if not 0.0 < args.ppo_clip < 1.0 or args.value_clip < 0.0:
        parser.error("PPO clip values are invalid")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    checkpoint = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=True
        )
        if checkpoint.get("architecture") not in {
            "hierarchical_candidateformer_v3",
            "hierarchical_candidateformer_a2c_v3",
            "hierarchical_candidateformer_v4",
            "hierarchical_candidateformer_a2c_v4",
            "hierarchical_candidateformer_v5",
            "hierarchical_candidateformer_a2c_v5",
            "hierarchical_candidateformer_v6",
            "hierarchical_candidateformer_a2c_v6",
            "hierarchical_candidateformer_v7",
            "hierarchical_candidateformer_a2c_v7",
            "hierarchical_candidateformer_ppo_v7",
            "hierarchical_candidateformer_v7_budget_v1",
            "hierarchical_candidateformer_ppo_v7_budget_v1",
            "hierarchical_candidateformer_ppo_v7_budget_v2",
            "hierarchical_candidateformer_v8_belief_v1",
            "hierarchical_candidateformer_ppo_v8_belief_v1",
        }:
            parser.error("--init-checkpoint is not a V3-V7 hierarchical checkpoint")
        for key in (
            "hidden_size",
            "board_width",
            "d_model",
            "transformer_layers",
            "transformer_heads",
            "proposal_top_k",
        ):
            if key in checkpoint:
                setattr(args, key, int(checkpoint[key]))

    model = HierarchicalKaggriculturePolicy(
        hidden_size=args.hidden_size,
        board_width=args.board_width,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        proposal_top_k=args.proposal_top_k,
    ).to(device).eval()
    if checkpoint is not None:
        skipped = load_compatible_checkpoint(model, checkpoint["model"])
        if skipped:
            print(f"initialized_v8_tensors={len(skipped)}")
    league_models: list[HierarchicalKaggriculturePolicy] = []
    for path in args.league_checkpoint:
        league_checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        league_model = model_from_checkpoint(league_checkpoint, args, device)
        league_model.eval().requires_grad_(False)
        league_models.append(league_model)
    payoff_report = (
        json.loads(args.league_payoff_json.read_text(encoding="utf-8"))
        if args.league_payoff_json is not None
        else None
    )
    league_mixture = (
        league_mixture_from_payoff(payoff_report, len(league_models))
        if league_models
        else np.empty(0, dtype=np.float64)
    )
    league_selection_counts = [0 for _ in league_models]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    environments = VectorFastEnv(
        args.envs, configuration={"episodeSteps": args.episode_steps}
    )
    observations = environments.reset(range(args.seed, args.seed + args.envs))
    memories = [HierarchicalMemory() for _ in range(2 * args.envs)]
    next_seed = args.seed + args.envs
    failure_counts: dict[str, int] = {}
    task_failure_counts: dict[str, int] = {}
    recent_failures: list[dict[str, object]] = []
    task_transition_counts: dict[str, int] = {}
    recent_task_transitions: list[dict[str, object]] = []
    current_task_states: list[dict[str, object]] = []
    mode_switches = 0
    mode_history: list[dict[str, object]] = []
    internal_traces: list[dict[str, object]] = []
    proposal_monitor_history: list[dict[str, float | int]] = []
    budget_monitor_history: list[dict[str, float | int]] = []

    def consume_executor_diagnostics(rows: list[HierarchicalMemory]) -> None:
        nonlocal mode_switches, current_task_states
        for memory in rows:
            diagnostic = memory.executor_diagnostics()
            for reason, count in diagnostic["by_reason"].items():
                failure_counts[reason] = failure_counts.get(reason, 0) + count
            for key, count in diagnostic["by_task_and_reason"].items():
                task_failure_counts[key] = task_failure_counts.get(key, 0) + count
            recent_failures.extend(diagnostic["recent_failures"])
            task_state = diagnostic["task_state_machine"]
            for key, count in task_state["by_task_and_phase"].items():
                task_transition_counts[key] = task_transition_counts.get(key, 0) + count
            recent_task_transitions.extend(task_state["recent_transitions"])
            current_task_states.extend(task_state["current"])
            strategy = diagnostic["strategy"]
            mode_switches += int(strategy["switches"])
            mode_history.extend(strategy["history"])
            internal_traces.extend(memory.internal_trace_history)
        if len(recent_failures) > 256:
            del recent_failures[:-256]
        if len(recent_task_transitions) > 512:
            del recent_task_transitions[:-512]
        if len(current_task_states) > 2 * args.envs * 17:
            del current_task_states[: -(2 * args.envs * 17)]
        if len(mode_history) > 256:
            del mode_history[:-256]
        if len(internal_traces) > 512:
            del internal_traces[:-512]

    for update in range(1, args.updates + 1):
        league_active = bool(league_models)
        learner_seat = (update - 1) % 2
        opponent_model = None
        if league_active:
            league_index = sample_league_member(league_mixture, rng)
            league_selection_counts[league_index] += 1
            opponent_model = league_models[league_index]
        rollout_replays = []
        rollout_log_probs: list[torch.Tensor] = []
        rollout_proposal_log_probs: list[torch.Tensor] = []
        rollout_values: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []
        rollout_proposal_entropies: list[torch.Tensor] = []
        rollout_budget_log_probs: list[torch.Tensor] = []
        rollout_resource_before: list[torch.Tensor] = []
        rollout_resource_after: list[torch.Tensor] = []

        for _ in range(args.rollout_steps):
            flat = flatten_environment_observations(observations)
            full_before = torch.tensor(
                paired_liquidatable_net_asset_potentials(flat),
                dtype=torch.float32,
                device=device,
            )
            learner_rows = (
                list(range(learner_seat, len(flat), 2))
                if league_active
                else list(range(len(flat)))
            )
            opponent_rows = (
                list(range(1 - learner_seat, len(flat), 2))
                if league_active
                else []
            )
            learner_observations = [flat[index] for index in learner_rows]
            learner_memories = [memories[index] for index in learner_rows]
            before = full_before[learner_rows]
            resource_before = torch.as_tensor(
                productive_resource_potentials(learner_observations), device=device
            )
            with torch.no_grad():
                policy = hierarchical_actor_critic_batch(
                    model, learner_observations, device, learner_memories
                )
            if policy.replay is None:
                raise RuntimeError("hierarchical policy did not return PPO replay data")
            if league_active:
                opponent_observations = [flat[index] for index in opponent_rows]
                opponent_memories = [memories[index] for index in opponent_rows]
                with torch.no_grad():
                    opponent_policy = hierarchical_actor_critic_batch(
                        opponent_model,
                        opponent_observations,
                        device,
                        opponent_memories,
                    )
                paired_actions = assemble_seat_actions(
                    policy.actions, opponent_policy.actions, learner_seat
                )
            else:
                paired_actions = [
                    [policy.actions[2 * index], policy.actions[2 * index + 1]]
                    for index in range(args.envs)
                ]
            results = environments.step(paired_actions)
            terminal_observations = [result.observations for result in results]
            terminal_flat = flatten_environment_observations(terminal_observations)
            full_after = torch.tensor(
                paired_liquidatable_net_asset_potentials(terminal_flat),
                dtype=torch.float32,
                device=device,
            )
            terminal_learner = [terminal_flat[index] for index in learner_rows]
            after = full_after[learner_rows]
            resource_after = torch.as_tensor(
                productive_resource_potentials(terminal_learner), device=device
            )
            env_dones = [result.done for result in results]
            dones = torch.tensor(
                env_dones
                if league_active
                else [done for done in env_dones for _ in range(2)],
                dtype=torch.bool,
                device=device,
            )
            rewards = args.reward_scale * (after - before)
            if args.win_bonus:
                money_outcome = torch.tensor(
                    [relative_money_potential(item) for item in terminal_learner]
                    if league_active
                    else [relative_money_potential(item) for item in terminal_flat],
                    dtype=torch.float32,
                    device=device,
                )
                rewards = rewards + dones * args.win_bonus * torch.sign(money_outcome)

            rollout_replays.append(policy.replay)
            rollout_log_probs.append(policy.log_probs.detach())
            rollout_proposal_log_probs.append(policy.proposal_log_probs.detach())
            rollout_values.append(policy.values.detach())
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            rollout_proposal_entropies.append(policy.proposal_entropies.detach())
            rollout_budget_log_probs.append(policy.budget_log_probs.detach())
            rollout_resource_before.append(resource_before)
            rollout_resource_after.append(resource_after)
            if any(env_dones):
                if not all(env_dones):
                    raise RuntimeError("fixed-length vector games ended asynchronously")
                observations = environments.reset(
                    range(next_seed, next_seed + args.envs)
                )
                consume_executor_diagnostics(memories)
                memories = [HierarchicalMemory() for _ in range(2 * args.envs)]
                next_seed += args.envs
            else:
                observations = terminal_observations

        values = torch.stack(rollout_values)
        rewards = torch.stack(rollout_rewards)
        dones = torch.stack(rollout_dones)
        horizon_targets, horizon_mask = multi_horizon_transition_targets(
            torch.stack(rollout_resource_before),
            torch.stack(rollout_resource_after),
            dones,
        )
        with torch.no_grad():
            bootstrap = (
                torch.zeros_like(values[-1])
                if dones[-1].all()
                else hierarchical_values(
                    model,
                    (
                        [
                            flatten_environment_observations(observations)[index]
                            for index in range(
                                learner_seat,
                                2 * args.envs,
                                2,
                            )
                        ]
                        if league_active
                        else flatten_environment_observations(observations)
                    ),
                    device,
                    (
                        [memories[index] for index in range(
                            learner_seat, 2 * args.envs, 2
                        )]
                        if league_active
                        else memories
                    ),
                )
            )
            advantages, returns = generalized_advantage_estimates(
                rewards,
                dones,
                values.detach(),
                bootstrap,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
            )
            advantages = (advantages - advantages.mean()) / (
                advantages.std(unbiased=False) + 1e-8
            )

        old_log_probs = torch.stack(rollout_log_probs)
        old_proposal_log_probs = torch.stack(rollout_proposal_log_probs)
        old_budget_log_probs = torch.stack(rollout_budget_log_probs)
        proposal_weight_before = model.proposal_task_head.weight.detach().clone()
        budget_weight_before = model.budget_category_head.weight.detach().clone()
        budget_active_count = sum(
            int(replay.budget_action_active.sum()) for replay in rollout_replays
        )
        ppo_policy_losses: list[float] = []
        ppo_value_losses: list[float] = []
        ppo_horizon_losses: list[float] = []
        ppo_entropies: list[float] = []
        ppo_proposal_entropies: list[float] = []
        ppo_proposal_kls: list[float] = []
        ppo_budget_kls: list[float] = []
        ppo_kls: list[float] = []
        ppo_clip_fractions: list[float] = []
        grad_norm_value = 0.0
        early_stop = False
        for _epoch in range(args.ppo_epochs):
            order = torch.randperm(args.rollout_steps).tolist()
            for start in range(0, args.rollout_steps, args.minibatch_steps):
                indices = order[start : start + args.minibatch_steps]
                evaluations = [
                    evaluate_hierarchical_replay(
                        model, rollout_replays[index], device
                    )
                    for index in indices
                ]
                new_log_probs = torch.cat(
                    [evaluation.log_probs for evaluation in evaluations]
                )
                new_values = torch.cat(
                    [evaluation.values for evaluation in evaluations]
                )
                new_horizon_values = torch.cat(
                    [evaluation.horizon_values for evaluation in evaluations]
                )
                entropy = torch.cat(
                    [evaluation.entropies for evaluation in evaluations]
                ).mean()
                proposal_entropy = torch.cat(
                    [evaluation.proposal_entropies for evaluation in evaluations]
                ).mean()
                old_log_prob = torch.cat(
                    [old_log_probs[index] for index in indices]
                )
                old_proposal_log_prob = torch.cat(
                    [old_proposal_log_probs[index] for index in indices]
                )
                new_proposal_log_prob = torch.cat(
                    [evaluation.proposal_log_probs for evaluation in evaluations]
                )
                new_budget_log_prob = torch.cat(
                    [evaluation.budget_log_probs for evaluation in evaluations]
                )
                old_budget_log_prob = torch.cat(
                    [old_budget_log_probs[index] for index in indices]
                )
                budget_active = torch.cat(
                    [
                        rollout_replays[index].budget_action_active.to(device)
                        for index in indices
                    ]
                )
                old_value = torch.cat([values[index] for index in indices])
                advantage = torch.cat([advantages[index] for index in indices])
                return_target = torch.cat([returns[index] for index in indices])
                horizon_target = torch.cat(
                    [horizon_targets[index] for index in indices]
                )
                horizon_active = torch.cat(
                    [horizon_mask[index] for index in indices]
                )
                log_ratio = new_log_probs - old_log_prob
                ratio = log_ratio.exp()
                unclipped = ratio * advantage
                clipped = ratio.clamp(
                    1.0 - args.ppo_clip, 1.0 + args.ppo_clip
                ) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                if args.value_clip:
                    clipped_value = old_value + (new_values - old_value).clamp(
                        -args.value_clip, args.value_clip
                    )
                    value_loss = 0.5 * torch.maximum(
                        (new_values - return_target).square(),
                        (clipped_value - return_target).square(),
                    ).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(new_values, return_target)
                horizon_loss = (
                    F.smooth_l1_loss(
                        new_horizon_values[horizon_active],
                        horizon_target[horizon_active],
                    )
                    if horizon_active.any()
                    else new_horizon_values.sum() * 0.0
                )
                loss = (
                    policy_loss
                    + args.value_coef * value_loss
                    + args.horizon_value_coef * horizon_loss
                    - args.entropy_coef * entropy
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm
                )
                optimizer.step()
                with torch.no_grad():
                    approximate_kl = ((ratio - 1.0) - log_ratio).mean()
                    proposal_log_ratio = (
                        new_proposal_log_prob - old_proposal_log_prob
                    )
                    proposal_ratio = proposal_log_ratio.exp()
                    proposal_kl = (
                        (proposal_ratio - 1.0) - proposal_log_ratio
                    ).mean()
                    if bool(budget_active.any()):
                        budget_log_ratio = (
                            new_budget_log_prob[budget_active]
                            - old_budget_log_prob[budget_active]
                        )
                        budget_ratio = budget_log_ratio.exp()
                        budget_kl = (
                            (budget_ratio - 1.0) - budget_log_ratio
                        ).mean()
                    else:
                        budget_kl = torch.zeros((), device=device)
                    clip_fraction = (
                        (ratio - 1.0).abs() > args.ppo_clip
                    ).float().mean()
                ppo_policy_losses.append(float(policy_loss.detach()))
                ppo_value_losses.append(float(value_loss.detach()))
                ppo_horizon_losses.append(float(horizon_loss.detach()))
                ppo_entropies.append(float(entropy.detach()))
                ppo_proposal_entropies.append(float(proposal_entropy.detach()))
                ppo_proposal_kls.append(float(proposal_kl))
                ppo_budget_kls.append(float(budget_kl))
                ppo_kls.append(float(approximate_kl))
                ppo_clip_fractions.append(float(clip_fraction))
                grad_norm_value = float(grad_norm)
                if args.target_kl > 0 and float(approximate_kl) > args.target_kl:
                    early_stop = True
                    break
            if early_stop:
                break

        if update == 1 or update % args.log_every == 0:
            policy_loss_value = sum(ppo_policy_losses) / max(len(ppo_policy_losses), 1)
            value_loss_value = sum(ppo_value_losses) / max(len(ppo_value_losses), 1)
            horizon_loss_value = sum(ppo_horizon_losses) / max(
                len(ppo_horizon_losses), 1
            )
            entropy_value = sum(ppo_entropies) / max(len(ppo_entropies), 1)
            proposal_entropy_value = sum(ppo_proposal_entropies) / max(
                len(ppo_proposal_entropies), 1
            )
            proposal_kl_value = sum(ppo_proposal_kls) / max(
                len(ppo_proposal_kls), 1
            )
            budget_kl_value = sum(ppo_budget_kls) / max(
                len(ppo_budget_kls), 1
            )
            kl_value = sum(ppo_kls) / max(len(ppo_kls), 1)
            clip_fraction_value = sum(ppo_clip_fractions) / max(
                len(ppo_clip_fractions), 1
            )
            print(
                f"update={update} policy={policy_loss_value:.4f} "
                f"value={value_loss_value:.4f} entropy={entropy_value:.4f} "
                f"horizon={horizon_loss_value:.4f} "
                f"grad={grad_norm_value:.3f} kl={kl_value:.5f} "
                f"clipfrac={clip_fraction_value:.3f} "
                f"proposal_entropy={proposal_entropy_value:.4f} "
                f"proposal_kl={proposal_kl_value:.5f} "
                f"budget_kl={budget_kl_value:.5f} "
                f"rollout_return={rewards.sum(dim=0).mean().item():.4f}"
            )
        proposal_delta = float(
            (model.proposal_task_head.weight.detach() - proposal_weight_before)
            .abs()
            .sum()
        )
        proposal_monitor_history.append(
            {
                "update": update,
                "entropy": sum(ppo_proposal_entropies)
                / max(len(ppo_proposal_entropies), 1),
                "approximate_kl": sum(ppo_proposal_kls)
                / max(len(ppo_proposal_kls), 1),
                "task_head_l1_delta": proposal_delta,
                "old_log_prob_mean": float(old_proposal_log_probs.mean()),
            }
        )
        budget_delta = float(
            (model.budget_category_head.weight.detach() - budget_weight_before)
            .abs()
            .sum()
        )
        budget_monitor_history.append(
            {
                "update": update,
                "active_daily_actions": budget_active_count,
                "approximate_kl": sum(ppo_budget_kls)
                / max(len(ppo_budget_kls), 1),
                "category_head_l1_delta": budget_delta,
                "old_log_prob_mean": float(
                    old_budget_log_probs[old_budget_log_probs != 0].mean()
                    if bool((old_budget_log_probs != 0).any())
                    else 0.0
                ),
            }
        )
        if update == 1 or update % args.log_every == 0:
            print(
                f"budget_actions={budget_active_count} "
                f"budget_head_delta={budget_delta:.6f}"
            )

    consume_executor_diagnostics(memories)
    executor_diagnostics = {
        "schema": "kaggriculture.executor-failures.v1",
        "failure_count": int(sum(failure_counts.values())),
        "by_reason": dict(sorted(failure_counts.items())),
        "by_task_and_reason": dict(sorted(task_failure_counts.items())),
        "recent_failures": recent_failures,
        "task_state_machine": {
            "transition_count": int(sum(task_transition_counts.values())),
            "by_task_and_phase": dict(sorted(task_transition_counts.items())),
            "recent_transitions": recent_task_transitions,
            "current": current_task_states,
        },
        "strategy": {
            "mode_switches": mode_switches,
            "history": mode_history,
        },
        "agent_trace": {
            "schema": "kaggriculture.agent-trace.v2",
            "count": len(internal_traces),
            "recent": internal_traces,
        },
        "planner": {
            "task_chain_trace_schema": "kaggriculture.agent-trace.v2",
            "persistent_runtime_task_chains": True,
            "strict_candidate_feasibility": True,
            "joint_task_resource_market": True,
            "explicit_daily_budget_plan": True,
            "hard_budget_constrained_market_decoder": True,
            "explicit_budget_ppo_action": True,
            "budget_action_frequency": "daily",
            "budget_action_distribution": "Beta+Dirichlet+scaled-Beta",
            "proposal_reinforce": True,
            "exact_action_replay": True,
            "optimizer": "clipped_ppo",
            "league_training": bool(league_models),
            "league_mixture": league_mixture.tolist(),
            "league_selection_counts": league_selection_counts,
            "league_opponent_frequency": "update",
        },
        "proposal_monitor": {
            "schema": "kaggriculture.proposal-monitor.v1",
            "updates": proposal_monitor_history,
            "task_head_total_l1_delta": sum(
                float(row["task_head_l1_delta"])
                for row in proposal_monitor_history
            ),
        },
        "budget_monitor": {
            "schema": "kaggriculture.budget-ppo-monitor.v1",
            "updates": budget_monitor_history,
            "category_head_total_l1_delta": sum(
                float(row["category_head_l1_delta"])
                for row in budget_monitor_history
            ),
            "active_daily_actions": sum(
                int(row["active_daily_actions"])
                for row in budget_monitor_history
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": "hierarchical_candidateformer_ppo_v8_belief_v1",
            "hidden_size": args.hidden_size,
            "board_width": args.board_width,
            "d_model": args.d_model,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "proposal_top_k": args.proposal_top_k,
            "proposal_reinforce": True,
            "exact_action_replay": True,
            "joint_task_resource_market": True,
            "explicit_daily_budget_plan": True,
            "hard_budget_constrained_market_decoder": True,
            "explicit_budget_ppo_action": True,
            "budget_action_frequency": "daily",
            "budget_action_distribution": "Beta+Dirichlet+scaled-Beta",
            "opponent_belief_conditioning": True,
            "uncertainty_cash_floor": True,
            "robust_opening_mode": True,
            "resource_growth_horizons": [24, 72, 168],
            "horizon_value_coef": args.horizon_value_coef,
            "resource_growth_changes_main_reward": False,
            "league_checkpoints": [str(path) for path in args.league_checkpoint],
            "league_payoff_json": (
                str(args.league_payoff_json)
                if args.league_payoff_json is not None
                else None
            ),
            "league_mixture": league_mixture.tolist(),
            "league_selection_counts": league_selection_counts,
            "league_seat_alternation": True,
            "ppo_epochs": args.ppo_epochs,
            "ppo_clip": args.ppo_clip,
            "value_clip": args.value_clip,
            "minibatch_steps": args.minibatch_steps,
            "target_kl": args.target_kl,
            "updates": args.updates,
            "rollout_steps": args.rollout_steps,
            "episode_steps": args.episode_steps,
            "seed": args.seed,
            "init_checkpoint": (
                str(args.init_checkpoint) if args.init_checkpoint is not None else None
            ),
            "executor_diagnostics": executor_diagnostics,
        },
        args.output,
    )
    diagnostics_output = args.diagnostics_output or args.output.with_suffix(
        ".diagnostics.json"
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(
        json.dumps(executor_diagnostics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"saved={args.output}")
    print(f"executor_diagnostics={diagnostics_output}")


if __name__ == "__main__":
    main()
