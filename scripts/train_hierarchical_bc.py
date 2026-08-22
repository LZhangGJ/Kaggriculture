"""Collect audited, cluster-balanced task-chain traces and train the V7 hierarchy."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch.nn import functional as F

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.board_rl import (
    multi_horizon_resource_growth_targets,
    paired_liquidatable_net_asset_potentials,
    potential_shaped_value_targets,
)
from kaggriculture_lab.board_policy import encode_board_batch
from kaggriculture_lab.candidate_proposal import (
    dense_proposal_targets,
    expert_cards_from_soft_intents,
    phase_targets,
    proposal_cards_from_logits,
    soft_assignment_targets,
)
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.expert_sampling import (
    cluster_sampling_report,
    load_teacher_profiles,
    robust_cluster_indices,
)
from kaggriculture_lab.gpu_policy import flatten_environment_observations
from kaggriculture_lab.hierarchical_bc import (
    DailyBudgetSoftTarget,
    IntentLabel,
    behavior_targets,
    closure_diagnostics,
    daily_budget_soft_targets,
    market_teacher_masks,
)
from kaggriculture_lab.hierarchical_trace import (
    AgentStepTrace,
    SoftIntentDistribution,
    inverse_episode_traces,
    merge_trace_quality_audits,
    trace_quality_audit,
    write_traces_jsonl,
)
from kaggriculture_lab.hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    _as_tensors,
    encode_hierarchical_batch,
)
from kaggriculture_lab.hierarchical_schema import (
    HierarchicalMemory,
    strategy_mode_from_name,
)
from kaggriculture_lab.official_replay_v7 import (
    OfficialV7LoadResult,
    load_official_v7_training_records,
)
from kaggriculture_lab.opponent_model import build_opponent_history_sequences


@dataclass(frozen=True)
class Sample:
    observation: Any
    action: Mapping[str, Any]
    intents: list[IntentLabel]
    soft_intents: list[SoftIntentDistribution]
    strategy_mode: int
    value_target: float
    initial_mode_active: bool
    daily_switch_active: bool
    budget_target: DailyBudgetSoftTarget
    opponent_history: np.ndarray
    opponent_history_mask: np.ndarray
    opponent_cluster: int
    outcome: float
    horizon_targets: np.ndarray
    horizon_mask: np.ndarray
    teacher_name: str
    teacher_cluster: int
    source: str = "live"


def _collect(args: argparse.Namespace) -> tuple[list[Sample], list[AgentStepTrace]]:
    teacher_specs = list(args.teachers)
    teachers = [resolve_agent(value) for value in teacher_specs]
    teacher_profiles = load_teacher_profiles(
        args.teacher_clusters_json, teacher_specs
    )
    samples: list[Sample] = []
    traces: list[AgentStepTrace] = []
    for round_index in range(args.collection_rounds):
        environments = VectorFastEnv(
            args.envs,
            configuration={"episodeSteps": args.episode_steps},
            copy_observations=True,
        )
        seed_base = args.seed + round_index * args.envs
        observations = environments.reset(
            range(seed_base, seed_base + args.envs)
        )
        actor_observations: list[list[Any]] = [
            [] for _ in range(2 * args.envs)
        ]
        actor_actions: list[list[Mapping[str, Any]]] = [
            [] for _ in range(2 * args.envs)
        ]
        actor_teacher = [
            (index + round_index * 2 * args.envs) % len(teachers)
            for index in range(2 * args.envs)
        ]
        while True:
            flat = flatten_environment_observations(observations)
            actions = [
                teachers[actor_teacher[index]](
                    observation,
                    environments.envs[index // 2].configuration,
                )
                for index, observation in enumerate(flat)
            ]
            for index, (observation, action) in enumerate(
                zip(flat, actions, strict=True)
            ):
                actor_observations[index].append(observation)
                actor_actions[index].append(action)
            paired = [
                [actions[2 * index], actions[2 * index + 1]]
                for index in range(args.envs)
            ]
            results = environments.step(paired)
            observations = [result.observations for result in results]
            if all(result.done for result in results):
                break

        final_flat = flatten_environment_observations(observations)
        actor_value_targets: list[np.ndarray] = [
            np.empty(0, dtype=np.float32) for _ in range(2 * args.envs)
        ]
        actor_outcomes = np.zeros(2 * args.envs, dtype=np.float32)
        for environment_index, result in enumerate(results):
            left = 2 * environment_index
            potentials = [
                paired_liquidatable_net_asset_potentials(
                    [actor_observations[left][step], actor_observations[left + 1][step]]
                )
                for step in range(len(actor_observations[left]))
            ]
            potentials.append(
                paired_liquidatable_net_asset_potentials(
                    [final_flat[left], final_flat[left + 1]]
                )
            )
            potential_array = np.stack(potentials)
            left_reward, right_reward = result.rewards
            if left_reward is None or right_reward is None or left_reward == right_reward:
                outcomes = (0.0, 0.0)
            elif left_reward > right_reward:
                outcomes = (1.0, -1.0)
            else:
                outcomes = (-1.0, 1.0)
            for offset in range(2):
                actor_outcomes[left + offset] = outcomes[offset]
                actor_value_targets[left + offset] = potential_shaped_value_targets(
                    potential_array[:, offset],
                    gamma=args.value_gamma,
                    reward_scale=args.value_reward_scale,
                    terminal_outcome=outcomes[offset],
                    win_bonus=args.value_win_bonus,
                )
        for actor in range(2 * args.envs):
            teacher_name = teacher_specs[actor_teacher[actor]]
            teacher_profile = teacher_profiles[actor_teacher[actor]]
            mode = strategy_mode_from_name(teacher_name)
            actor_traces = inverse_episode_traces(
                actor_observations[actor],
                actor_actions[actor],
                strategy_mode=mode,
                teacher=teacher_name,
                horizon=args.intent_horizon,
                top_m=args.inverse_top_m,
            )
            traces.extend(actor_traces)
            actor_budget_targets = daily_budget_soft_targets(
                actor_observations[actor],
                actor_actions[actor],
                final_observation=final_flat[actor],
            )
            actor_opponent_histories, actor_opponent_masks = (
                build_opponent_history_sequences(actor_observations[actor])
            )
            actor_horizon_targets, actor_horizon_masks = (
                multi_horizon_resource_growth_targets(
                    actor_observations[actor],
                    final_observation=final_flat[actor],
                )
            )
            for sample_index, (
                observation,
                action,
                trace,
                budget_target,
                opponent_history,
                opponent_history_mask,
            ) in enumerate(zip(
                actor_observations[actor],
                actor_actions[actor],
                actor_traces,
                actor_budget_targets,
                actor_opponent_histories,
                actor_opponent_masks,
                strict=True,
            )):
                soft_intents = list(trace.workers)
                intents = [distribution.best for distribution in soft_intents]
                step = int(
                    (
                        observation.get("step", 0)
                        if isinstance(observation, Mapping)
                        else getattr(observation, "step", 0)
                    )
                    or 0
                )
                hour = int(
                    (
                        observation.get("hour", step % 24)
                        if isinstance(observation, Mapping)
                        else getattr(observation, "hour", step % 24)
                    )
                    or 0
                )
                samples.append(
                    Sample(
                        observation,
                        action,
                        intents,
                        soft_intents,
                        mode,
                        float(actor_value_targets[actor][sample_index]),
                        step == 0,
                        step > 0 and hour == 0,
                        budget_target,
                        opponent_history,
                        opponent_history_mask,
                        teacher_profiles[actor_teacher[actor ^ 1]].cluster,
                        float(actor_outcomes[actor]),
                        actor_horizon_targets[sample_index],
                        actor_horizon_masks[sample_index],
                        teacher_profile.agent,
                        teacher_profile.cluster,
                    )
                )
    return samples, traces


def _collect_official(args: argparse.Namespace) -> tuple[list[Sample], OfficialV7LoadResult]:
    for manifest_path in args.official_v7_manifest:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        label_config = dict(manifest.get("labels", {}))
        value_config = dict(manifest.get("value_targets", {}))
        expected = {
            "intent_horizon": (int(label_config.get("intent_horizon", -1)), args.intent_horizon),
            "inverse_top_m": (int(label_config.get("inverse_top_m", -1)), args.inverse_top_m),
            "value_gamma": (float(value_config.get("gamma", -1.0)), args.value_gamma),
            "value_reward_scale": (
                float(value_config.get("reward_scale", -1.0)),
                args.value_reward_scale,
            ),
            "value_win_bonus": (
                float(value_config.get("win_bonus", -1.0)),
                args.value_win_bonus,
            ),
        }
        mismatches = [
            f"{name}: dataset={left} training={right}"
            for name, (left, right) in expected.items()
            if not np.isclose(left, right)
        ]
        if mismatches:
            raise ValueError(
                f"official V7 label configuration mismatch in {manifest_path}: "
                + "; ".join(mismatches)
            )
    loaded = load_official_v7_training_records(
        args.official_v7_manifest,
        min_quality=args.official_min_quality,
        max_episodes_per_manifest=args.official_max_episodes,
        require_valid_audit=not args.allow_unverified_official_data,
    )
    samples = [
        Sample(
            observation=record.observation,
            action=record.action,
            intents=[distribution.best for distribution in record.soft_intents],
            soft_intents=list(record.soft_intents),
            strategy_mode=record.strategy_mode,
            value_target=record.value_target,
            initial_mode_active=record.initial_mode_active,
            daily_switch_active=record.daily_switch_active,
            budget_target=record.budget_target,
            opponent_history=record.opponent_history,
            opponent_history_mask=record.opponent_history_mask,
            opponent_cluster=record.opponent_cluster,
            outcome=record.outcome,
            horizon_targets=record.horizon_targets,
            horizon_mask=record.horizon_mask,
            teacher_name=record.teacher,
            teacher_cluster=record.teacher_cluster,
            source="official_v7",
        )
        for record in loaded.records
    ]
    return samples, loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teachers", nargs="+", default=["starter"])
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--collection-rounds", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--intent-horizon", type=int, default=24)
    parser.add_argument("--inverse-top-m", type=int, default=3)
    parser.add_argument("--proposal-top-k", type=int, default=24)
    parser.add_argument("--expert-dropout-start", type=float, default=0.10)
    parser.add_argument("--expert-dropout-end", type=float, default=0.80)
    parser.add_argument("--proposal-loss-coef", type=float, default=0.50)
    parser.add_argument("--budget-loss-coef", type=float, default=0.30)
    parser.add_argument("--opponent-belief-loss-coef", type=float, default=0.20)
    parser.add_argument("--horizon-value-loss-coef", type=float, default=0.20)
    parser.add_argument("--worst-cluster-fraction", type=float, default=0.35)
    parser.add_argument("--cvar-fraction", type=float, default=0.25)
    parser.add_argument("--value-gamma", type=float, default=0.997)
    parser.add_argument("--value-reward-scale", type=float, default=10.0)
    parser.add_argument("--value-win-bonus", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--board-width", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--official-v7-manifest",
        type=Path,
        action="append",
        default=[],
        help="audited official V7 manifest; repeat to combine daily datasets",
    )
    parser.add_argument(
        "--official-only",
        action="store_true",
        help="skip live teacher rollouts and train only from official V7 records",
    )
    parser.add_argument(
        "--official-max-episodes",
        type=int,
        default=0,
        help="per-manifest episode cap for smoke tests; zero loads every selected episode",
    )
    parser.add_argument(
        "--official-min-quality",
        choices=("gold", "silver", "bronze"),
        default="silver",
    )
    parser.add_argument(
        "--allow-unverified-official-data",
        action="store_true",
        help="allow a missing/failed sibling audit.json (unsafe; disabled by default)",
    )
    parser.add_argument(
        "--validate-data-only",
        action="store_true",
        help="load and validate BC records, print the audit, then exit before model creation",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="deterministic post-load cap for smoke tests; zero keeps every sample",
    )
    parser.add_argument(
        "--teacher-clusters-json",
        type=Path,
        default=Path("artifacts/agent_behavior_clustering_v1/behavior_clusters.json"),
        help="30-agent behavior audit used for cluster-balanced minibatches.",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hierarchical_bc_v7.pt")
    )
    parser.add_argument(
        "--diagnostics-output",
        type=Path,
        help="JSON closure audit path (defaults beside --output).",
    )
    parser.add_argument(
        "--trace-output",
        type=Path,
        help="JSONL internal/inverse trace path (defaults beside --output).",
    )
    args = parser.parse_args()
    if min(
        args.envs,
        args.collection_rounds,
        args.episode_steps,
        args.epochs,
        args.batch_size,
        args.inverse_top_m,
        args.proposal_top_k,
    ) <= 0:
        parser.error("collection and training sizes must be positive")
    if not (
        0.0 <= args.expert_dropout_start <= 1.0
        and 0.0 <= args.expert_dropout_end <= 1.0
    ):
        parser.error("expert dropout values must be in [0, 1]")
    if not 0.0 <= args.worst_cluster_fraction <= 1.0:
        parser.error("--worst-cluster-fraction must be in [0, 1]")
    if not 0.0 < args.cvar_fraction <= 1.0:
        parser.error("--cvar-fraction must be in (0, 1]")
    if not 0.0 <= args.value_gamma <= 1.0:
        parser.error("--value-gamma must be in [0, 1]")
    if min(
        args.budget_loss_coef,
        args.opponent_belief_loss_coef,
        args.horizon_value_loss_coef,
    ) < 0.0:
        parser.error("auxiliary loss coefficients must be non-negative")
    if args.official_max_episodes < 0:
        parser.error("--official-max-episodes must be non-negative")
    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    if args.official_only and not args.official_v7_manifest:
        parser.error("--official-only requires at least one --official-v7-manifest")

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    samples: list[Sample] = []
    traces: list[AgentStepTrace] = []
    trace_audits: list[Mapping[str, Any]] = []
    official_reports: list[dict[str, Any]] = []
    if not args.official_only:
        print("collecting live expert episodes...")
        live_samples, traces = _collect(args)
        samples.extend(live_samples)
        trace_audits.append(trace_quality_audit(traces))
    if args.official_v7_manifest:
        print("loading audited official V7 records...")
        official_samples, official = _collect_official(args)
        samples.extend(official_samples)
        trace_audits.append(official.trace_audit)
        official_reports.append(
            {
                "manifests": list(official.manifests),
                "episodes": official.episodes,
                "actors": official.actors,
                "samples": len(official.records),
                "quality_tiers": official.quality_tiers,
                "mode_supervision": False,
                "daily_switch_supervision": False,
            }
        )
    if not samples:
        parser.error("no BC samples were collected or loaded")
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"samples={len(samples)}")
    sampling_report = cluster_sampling_report(
        [sample.teacher_name for sample in samples],
        [sample.teacher_cluster for sample in samples],
    )
    source_counts: dict[str, int] = {}
    for sample in samples:
        source_counts[sample.source] = source_counts.get(sample.source, 0) + 1
    sampling_report["sources"] = dict(sorted(source_counts.items()))
    sampling_report["official_v7"] = official_reports
    trace_audit = merge_trace_quality_audits(trace_audits)
    value_targets_np = np.asarray(
        [sample.value_target for sample in samples], dtype=np.float32
    )
    budget_spend_np = np.asarray(
        [sample.budget_target.total_spend for sample in samples], dtype=np.float32
    )
    budget_reserve_np = np.asarray(
        [sample.budget_target.reserve_fraction for sample in samples],
        dtype=np.float32,
    )
    if args.validate_data_only:
        print(
            json.dumps(
                {
                    "valid": True,
                    "samples": len(samples),
                    "sampling": sampling_report,
                    "trace_quality": trace_audit,
                    "value_targets": {
                        "mean": float(value_targets_np.mean()),
                        "std": float(value_targets_np.std()),
                        "min": float(value_targets_np.min()),
                        "max": float(value_targets_np.max()),
                    },
                    "daily_budget_soft_targets": {
                        "spend_active_samples": int((budget_spend_np > 0).sum()),
                        "mean_realized_spend": float(budget_spend_np.mean()),
                        "mean_reserve_fraction": float(budget_reserve_np.mean()),
                    },
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return
    model = HierarchicalKaggriculturePolicy(
        hidden_size=args.hidden_size,
        board_width=args.board_width,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        proposal_top_k=args.proposal_top_k,
    ).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    final_closure: dict[str, Any] = {}
    for epoch in range(1, args.epochs + 1):
        order = robust_cluster_indices(
            [sample.teacher_cluster for sample in samples],
            [sample.outcome for sample in samples],
            rng,
            robust_cluster_ids=[sample.opponent_cluster for sample in samples],
            worst_fraction=args.worst_cluster_fraction,
            cvar_fraction=args.cvar_fraction,
        )
        total_loss = 0.0
        total_budget_loss = 0.0
        total_belief_loss = 0.0
        total_horizon_loss = 0.0
        total_closed = 0
        total_active = 0
        deploy_soft_mass = 0.0
        oracle_soft_mass = 0.0
        soft_active = 0
        batches = 0
        epoch_task_counts: dict[str, dict[str, Any]] = {}
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        expert_dropout = (
            args.expert_dropout_start
            + progress * (args.expert_dropout_end - args.expert_dropout_start)
        )
        for start in range(0, len(samples), args.batch_size):
            rows = [samples[index] for index in order[start : start + args.batch_size]]
            observations = [row.observation for row in rows]
            actions = [row.action for row in rows]

            boards, global_features, unit_features, unit_mask = encode_board_batch(
                observations
            )
            state_tensors = (
                torch.as_tensor(boards, device=device),
                torch.as_tensor(global_features, device=device),
                torch.as_tensor(unit_features, device=device),
                torch.as_tensor(unit_mask, device=device),
                torch.as_tensor(
                    np.stack([row.opponent_history for row in rows]),
                    device=device,
                ),
                torch.as_tensor(
                    np.stack([row.opponent_history_mask for row in rows]),
                    device=device,
                ),
            )
            context = model.encode_state(*state_tensors)
            proposal = model.propose(context)
            learned_cards = proposal_cards_from_logits(
                proposal.task_logits,
                proposal.item_logits,
                proposal.auxiliary_predictions,
                observations,
                top_k=args.proposal_top_k,
            )
            oracle_experts = [
                expert_cards_from_soft_intents(
                    row.soft_intents, max_cards=8, dropout=0.0, rng=rng
                )
                for row in rows
            ]
            train_experts = [
                expert_cards_from_soft_intents(
                    row.soft_intents,
                    max_cards=8,
                    dropout=expert_dropout,
                    rng=rng,
                )
                for row in rows
            ]
            deploy_extra = [tuple(value) for value in learned_cards]
            oracle_extra = [
                tuple(learned) + tuple(expert)
                for learned, expert in zip(
                    learned_cards, oracle_experts, strict=True
                )
            ]
            train_extra = [
                tuple(learned) + tuple(expert)
                for learned, expert in zip(
                    learned_cards, train_experts, strict=True
                )
            ]
            encoded = encode_hierarchical_batch(
                observations,
                [HierarchicalMemory() for _ in rows],
                extra_cards=train_extra,
                learned_quota=args.proposal_top_k,
                state_encoding=(
                    boards, global_features, unit_features, unit_mask
                ),
                opponent_histories=np.stack(
                    [row.opponent_history for row in rows]
                ),
                opponent_history_masks=np.stack(
                    [row.opponent_history_mask for row in rows]
                ),
            )
            deploy_encoded = encode_hierarchical_batch(
                observations,
                [HierarchicalMemory() for _ in rows],
                extra_cards=deploy_extra,
                expert_quota=0,
                learned_quota=args.proposal_top_k,
                state_encoding=(
                    boards, global_features, unit_features, unit_mask
                ),
                opponent_histories=np.stack(
                    [row.opponent_history for row in rows]
                ),
                opponent_history_masks=np.stack(
                    [row.opponent_history_mask for row in rows]
                ),
            )
            oracle_encoded = encode_hierarchical_batch(
                observations,
                [HierarchicalMemory() for _ in rows],
                extra_cards=oracle_extra,
                learned_quota=args.proposal_top_k,
                state_encoding=(
                    boards, global_features, unit_features, unit_mask
                ),
                opponent_histories=np.stack(
                    [row.opponent_history for row in rows]
                ),
                opponent_history_masks=np.stack(
                    [row.opponent_history_mask for row in rows]
                ),
            )
            targets = behavior_targets(
                encoded.encoded_sets,
                [row.intents for row in rows],
                actions,
            )
            deploy_targets = behavior_targets(
                deploy_encoded.encoded_sets,
                [row.intents for row in rows],
                actions,
            )
            soft_targets = [
                soft_assignment_targets(task_set, row.soft_intents)
                for task_set, row in zip(
                    encoded.encoded_sets, rows, strict=True
                )
            ]
            deploy_soft = [
                soft_assignment_targets(task_set, row.soft_intents)
                for task_set, row in zip(
                    deploy_encoded.encoded_sets, rows, strict=True
                )
            ]
            oracle_soft = [
                soft_assignment_targets(task_set, row.soft_intents)
                for task_set, row in zip(
                    oracle_encoded.encoded_sets, rows, strict=True
                )
            ]
            market_masks_np = market_teacher_masks(
                observations, targets.market_targets
            )
            tensors = _as_tensors(encoded, device)
            modes = torch.tensor(
                [row.strategy_mode for row in rows],
                dtype=torch.long,
                device=device,
            )
            budget_output = model.predict_budget(context, modes)
            budget_context = model.budget_plan_context(budget_output)
            budget_reserve_targets = torch.tensor(
                [row.budget_target.reserve_fraction for row in rows],
                dtype=torch.float32,
                device=device,
            )
            budget_category_targets = torch.as_tensor(
                np.stack(
                    [row.budget_target.category_fractions for row in rows]
                ),
                device=device,
            )
            budget_floor_targets = torch.as_tensor(
                np.stack(
                    [row.budget_target.category_floor_fractions for row in rows]
                ),
                device=device,
            )
            budget_spend_active = torch.tensor(
                [row.budget_target.spend_active for row in rows],
                dtype=torch.bool,
                device=device,
            )
            budget_confidence = torch.tensor(
                [row.budget_target.confidence for row in rows],
                dtype=torch.float32,
                device=device,
            ).clamp_min(0.10)
            initial_mode_active = torch.tensor(
                [row.initial_mode_active for row in rows],
                dtype=torch.bool,
                device=device,
            )
            daily_switch_active = torch.tensor(
                [row.daily_switch_active for row in rows],
                dtype=torch.bool,
                device=device,
            )
            assignment_probabilities = torch.as_tensor(
                np.stack([value.probabilities for value in soft_targets]),
                device=device,
            )
            assignment_train = torch.as_tensor(
                np.stack([value.active for value in soft_targets]),
                device=device,
            )
            assignment_confidence_np = np.zeros(
                (len(rows), assignment_train.shape[1]), dtype=np.float32
            )
            for batch_index, row in enumerate(rows):
                for distribution in row.soft_intents:
                    if 0 <= distribution.unit < assignment_confidence_np.shape[1]:
                        assignment_confidence_np[batch_index, distribution.unit] = (
                            distribution.confidence
                        )
            assignment_confidence = torch.as_tensor(
                assignment_confidence_np, device=device
            ).clamp_min(0.10)
            market_targets = torch.as_tensor(
                targets.market_targets, device=device
            )
            market_active = torch.as_tensor(
                targets.market_active, device=device
            )
            market_masks = torch.as_tensor(market_masks_np, device=device)
            values_target = torch.tensor(
                [row.value_target for row in rows],
                dtype=torch.float32,
                device=device,
            )
            assignment_logits = model.score_assignments(
                context,
                *tensors[4:12],
                modes,
            )
            plan_context = model.assignment_plan_context(
                tensors[5],
                tensors[6],
                tensors[11],
                tensors[15],
                assignment_probabilities,
            )
            market_context = plan_context + budget_context
            market_logits = model.market_teacher_logits(
                context,
                modes,
                market_targets,
                tensors[14],
                plan_context=market_context,
            )
            mode_logits = context.mode_logits
            switch_logits = context.switch_logits
            values = context.values
            floor = torch.finfo(assignment_logits.dtype).min
            assignment_logits = assignment_logits.masked_fill(
                ~tensors[12], floor
            )
            market_logits = market_logits.masked_fill(~market_masks, floor)
            assignment_log_probabilities = F.log_softmax(
                assignment_logits, dim=-1
            )
            assignment_cross_entropy = -(
                assignment_probabilities * assignment_log_probabilities
            ).sum(dim=-1)
            assignment_loss = (
                (
                    assignment_cross_entropy[assignment_train]
                    * assignment_confidence[assignment_train]
                ).sum()
                / assignment_confidence[assignment_train].sum().clamp_min(1.0)
            )
            market_loss = F.cross_entropy(
                market_logits[market_active], market_targets[market_active]
            )
            budget_reserve_error = F.smooth_l1_loss(
                budget_output.reserve_fraction,
                budget_reserve_targets,
                reduction="none",
            )
            budget_reserve_loss = (
                (budget_reserve_error * budget_confidence).sum()
                / budget_confidence.sum().clamp_min(1.0)
            )
            budget_category_loss = (
                -(
                    budget_category_targets[budget_spend_active]
                    * torch.log(
                        budget_output.category_fractions[
                            budget_spend_active
                        ].clamp_min(1e-8)
                    )
                ).sum(dim=-1).mean()
                if budget_spend_active.any()
                else budget_output.category_fractions.sum() * 0.0
            )
            predicted_category_envelopes = (
                (1.0 - budget_output.reserve_fraction[:, None])
                * budget_output.category_fractions
            )
            budget_floor_loss = torch.relu(
                budget_floor_targets
                - predicted_category_envelopes
                - budget_output.emergency_fraction[:, None]
            ).mean()
            budget_loss = (
                budget_reserve_loss
                + budget_category_loss
                + budget_floor_loss
                + 0.10 * budget_output.emergency_fraction.mean()
            )
            mode_loss = (
                F.cross_entropy(mode_logits[initial_mode_active], modes[initial_mode_active])
                if initial_mode_active.any()
                else mode_logits.sum() * 0.0
            )
            # Static expert-family labels imply KEEP at each observed day boundary.
            switch_loss = (
                F.cross_entropy(
                    switch_logits[daily_switch_active],
                    torch.zeros(
                        int(daily_switch_active.sum()),
                        dtype=torch.long,
                        device=device,
                    ),
                )
                if daily_switch_active.any()
                else switch_logits.sum() * 0.0
            )
            value_loss = F.smooth_l1_loss(values, values_target)
            opponent_clusters = torch.tensor(
                [row.opponent_cluster for row in rows],
                dtype=torch.long,
                device=device,
            )
            opponent_cluster_active = (
                (opponent_clusters >= 0)
                & (opponent_clusters < model.opponent_clusters)
            )
            belief_loss = (
                F.cross_entropy(
                    context.opponent_belief_logits[opponent_cluster_active],
                    opponent_clusters[opponent_cluster_active],
                )
                if opponent_cluster_active.any()
                else context.opponent_belief_logits.sum() * 0.0
            )
            horizon_targets = torch.as_tensor(
                np.stack([row.horizon_targets for row in rows]), device=device
            )
            horizon_mask = torch.as_tensor(
                np.stack([row.horizon_mask for row in rows]), device=device
            )
            horizon_loss = (
                F.smooth_l1_loss(
                    context.horizon_values[horizon_mask],
                    horizon_targets[horizon_mask],
                )
                if horizon_mask.any()
                else context.horizon_values.sum() * 0.0
            )
            phase_batch = [phase_targets(row.soft_intents) for row in rows]
            phase_probabilities = torch.as_tensor(
                np.stack([value.probabilities for value in phase_batch]),
                device=device,
            )
            phase_active = torch.as_tensor(
                np.stack([value.active for value in phase_batch]),
                device=device,
            )
            phase_confidence = torch.as_tensor(
                np.stack([value.confidence for value in phase_batch]),
                device=device,
            )
            phase_logits = model.predict_phases(context)
            phase_cross_entropy = -(
                phase_probabilities * F.log_softmax(phase_logits, dim=-1)
            ).sum(dim=-1)
            phase_weights = phase_confidence.clamp_min(0.10)
            phase_loss = (
                (phase_cross_entropy[phase_active] * phase_weights[phase_active]).sum()
                / phase_weights[phase_active].sum().clamp_min(1.0)
            )

            dense = dense_proposal_targets(
                [row.soft_intents for row in rows]
            )
            proposal_task_targets = torch.as_tensor(
                dense.task_probabilities, device=device
            )
            proposal_probability = torch.sigmoid(proposal.task_logits)
            proposal_bce = F.binary_cross_entropy_with_logits(
                proposal.task_logits, proposal_task_targets, reduction="none"
            )
            proposal_pt = (
                proposal_task_targets * proposal_probability
                + (1.0 - proposal_task_targets) * (1.0 - proposal_probability)
            )
            proposal_alpha = (
                0.75 * proposal_task_targets
                + 0.25 * (1.0 - proposal_task_targets)
            )
            proposal_task_loss = (
                proposal_alpha * (1.0 - proposal_pt).square() * proposal_bce
            ).mean()
            item_targets = torch.as_tensor(dense.item_targets, device=device)
            item_weights = torch.as_tensor(dense.item_weights, device=device)
            item_active = item_weights > 0
            proposal_item_loss = (
                (
                    F.cross_entropy(
                        proposal.item_logits.permute(0, 2, 3, 1)[item_active],
                        item_targets[item_active],
                        reduction="none",
                    )
                    * item_weights[item_active]
                ).sum()
                / item_weights[item_active].sum().clamp_min(1.0)
                if item_active.any()
                else proposal.item_logits.sum() * 0.0
            )
            auxiliary_targets = torch.as_tensor(
                dense.auxiliary_targets, device=device
            )
            spatial_weights = torch.as_tensor(
                dense.spatial_weights, device=device
            )
            auxiliary_error = F.smooth_l1_loss(
                proposal.auxiliary_predictions,
                auxiliary_targets,
                reduction="none",
            ).mean(dim=1)
            proposal_auxiliary_loss = (
                (auxiliary_error * spatial_weights).sum()
                / spatial_weights.sum().clamp_min(1.0)
            )
            proposal_loss = (
                proposal_task_loss
                + 0.30 * proposal_item_loss
                + 0.50 * proposal_auxiliary_loss
            )
            loss = (
                assignment_loss
                + 0.7 * market_loss
                + 0.3 * mode_loss
                + 0.15 * switch_loss
                + 0.2 * value_loss
                + 0.2 * phase_loss
                + args.proposal_loss_coef * proposal_loss
                + args.budget_loss_coef * budget_loss
                + args.opponent_belief_loss_coef * belief_loss
                + args.horizon_value_loss_coef * horizon_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach())
            total_budget_loss += float(budget_loss.detach())
            total_belief_loss += float(belief_loss.detach())
            total_horizon_loss += float(horizon_loss.detach())
            total_closed += int(deploy_targets.assignment_closed.sum())
            total_active += int(deploy_targets.assignment_active.sum())
            for deploy_value, oracle_value in zip(
                deploy_soft, oracle_soft, strict=True
            ):
                deploy_soft_mass += float(
                    deploy_value.closed_mass[deploy_value.active].sum()
                )
                oracle_soft_mass += float(
                    oracle_value.closed_mass[oracle_value.active].sum()
                )
                soft_active += int(deploy_value.active.sum())
            batch_closure = closure_diagnostics(deploy_targets)
            for task, row in batch_closure["tasks"].items():
                aggregate = epoch_task_counts.setdefault(
                    task,
                    {
                        "intents": 0,
                        "closed": 0,
                        "reasons": {key: 0 for key in row["reasons"]},
                    },
                )
                aggregate["intents"] += row["intents"]
                aggregate["closed"] += row["closed"]
                for reason, count in row["reasons"].items():
                    aggregate["reasons"][reason] += count
            batches += 1
        for row in epoch_task_counts.values():
            row["closure_rate"] = row["closed"] / max(row["intents"], 1)
        final_closure = {
            "schema": "kaggriculture.candidate-closure.v2",
            "epoch": epoch,
            "expert_dropout": expert_dropout,
            "deploy": {
                "intents": total_active,
                "closed": total_closed,
                "hard_closure_rate": total_closed / max(total_active, 1),
                "soft_closure_rate": deploy_soft_mass / max(soft_active, 1),
            },
            "oracle": {
                "soft_closure_rate": oracle_soft_mass / max(soft_active, 1),
            },
            "injection_gap": (
                oracle_soft_mass - deploy_soft_mass
            ) / max(soft_active, 1),
            "tasks": epoch_task_counts,
            "sampling": sampling_report,
            "robust_sampling": {
                "worst_cluster_fraction": args.worst_cluster_fraction,
                "cvar_fraction": args.cvar_fraction,
            },
            "trace_quality": trace_audit,
            "value_targets": {
                "definition": "discounted liquidatable-net-asset transition returns",
                "gamma": args.value_gamma,
                "reward_scale": args.value_reward_scale,
                "win_bonus": args.value_win_bonus,
                "mean": float(value_targets_np.mean()),
                "std": float(value_targets_np.std()),
                "min": float(value_targets_np.min()),
                "max": float(value_targets_np.max()),
            },
            "daily_budget_soft_targets": {
                "loss": total_budget_loss / max(batches, 1),
                "loss_coefficient": args.budget_loss_coef,
                "spend_active_samples": int((budget_spend_np > 0).sum()),
                "mean_realized_spend": float(budget_spend_np.mean()),
                "mean_reserve_fraction": float(budget_reserve_np.mean()),
            },
            "opponent_belief": {
                "clusters": model.opponent_clusters,
                "loss": total_belief_loss / max(batches, 1),
                "loss_coefficient": args.opponent_belief_loss_coef,
            },
            "resource_growth_critics": {
                "horizons_steps": [24, 72, 168],
                "loss": total_horizon_loss / max(batches, 1),
                "loss_coefficient": args.horizon_value_loss_coef,
                "changes_main_reward": False,
            },
        }
        compact = ",".join(
            f"{task}={row['closure_rate']:.3f}"
            for task, row in sorted(epoch_task_counts.items())
        )
        print(
            f"epoch={epoch} loss={total_loss / max(batches, 1):.4f} "
            f"budget_loss={total_budget_loss / max(batches, 1):.4f} "
            f"belief_loss={total_belief_loss / max(batches, 1):.4f} "
            f"horizon_loss={total_horizon_loss / max(batches, 1):.4f} "
            f"deploy_closure={deploy_soft_mass / max(soft_active, 1):.4f} "
            f"oracle_closure={oracle_soft_mass / max(soft_active, 1):.4f} "
            f"injection_gap={final_closure['injection_gap']:.4f} "
            f"closure_by_task={compact}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "architecture": "hierarchical_candidateformer_v8_belief_v1",
            "hidden_size": args.hidden_size,
            "board_width": args.board_width,
            "d_model": args.d_model,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "proposal_top_k": args.proposal_top_k,
            "inverse_top_m": args.inverse_top_m,
            "expert_dropout_start": args.expert_dropout_start,
            "expert_dropout_end": args.expert_dropout_end,
            "trace_schema": "kaggriculture.agent-trace.v2",
            "task_chain_phase_supervision": True,
            "persistent_runtime_task_chains": True,
            "joint_task_resource_market": True,
            "explicit_daily_budget_plan": True,
            "hard_budget_constrained_market_decoder": True,
            "budget_loss_coef": args.budget_loss_coef,
            "opponent_belief_loss_coef": args.opponent_belief_loss_coef,
            "opponent_clusters": model.opponent_clusters,
            "opponent_history_lags": [120, 72, 24, 0],
            "resource_growth_horizons": [24, 72, 168],
            "horizon_value_loss_coef": args.horizon_value_loss_coef,
            "opponent_conditioned_heads": [
                "strategy_mode", "budget", "proposal", "task", "critic"
            ],
            "cluster_balanced_sampling": True,
            "worst_cluster_fraction": args.worst_cluster_fraction,
            "cvar_fraction": args.cvar_fraction,
            "per_step_value_returns": True,
            "value_gamma": args.value_gamma,
            "value_reward_scale": args.value_reward_scale,
            "value_win_bonus": args.value_win_bonus,
            "teacher_clusters_json": str(args.teacher_clusters_json),
            "teachers": list(args.teachers),
            "official_v7_manifests": [
                str(path.resolve()) for path in args.official_v7_manifest
            ],
            "official_only": args.official_only,
            "max_samples": args.max_samples,
            "samples": len(samples),
            "seed": args.seed,
            "closure_diagnostics": final_closure,
        },
        args.output,
    )
    diagnostics_output = args.diagnostics_output or args.output.with_suffix(
        ".diagnostics.json"
    )
    diagnostics_output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_output.write_text(
        json.dumps(final_closure, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    trace_output = args.trace_output or args.output.with_suffix(".traces.jsonl")
    write_traces_jsonl(trace_output, traces)
    print(f"saved={args.output}")
    print(f"closure_diagnostics={diagnostics_output}")
    print(f"traces={trace_output}")


if __name__ == "__main__":
    main()
