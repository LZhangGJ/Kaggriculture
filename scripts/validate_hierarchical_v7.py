"""Full-game and official-runner differential validation for the hierarchical policy."""

from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    hierarchical_policy_batch,
    load_compatible_hierarchical_state_dict,
)
from kaggriculture_lab.hierarchical_schema import HierarchicalMemory


SUPPORTED_ARCHITECTURES = {
    f"hierarchical_candidateformer{suffix}_v{version}"
    for version in range(3, 8)
    for suffix in ("", "_a2c", "_ppo")
}
SUPPORTED_ARCHITECTURES.update(
    {
        "hierarchical_candidateformer_v7_budget_v1",
        "hierarchical_candidateformer_ppo_v7_budget_v1",
        "hierarchical_candidateformer_ppo_v7_budget_v2",
        "hierarchical_candidateformer_v8_belief_v1",
        "hierarchical_candidateformer_ppo_v8_belief_v1",
    }
)


def _model_from_checkpoint(
    checkpoint_path: Path | None,
    device: torch.device,
    args: argparse.Namespace,
) -> HierarchicalKaggriculturePolicy:
    checkpoint: Mapping[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        architecture = str(checkpoint.get("architecture", ""))
        if architecture not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"unsupported checkpoint architecture: {architecture!r}")
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
        opponent_clusters=int(checkpoint.get("opponent_clusters", 8))
        if checkpoint is not None
        else 8,
    ).to(device)
    if checkpoint is not None:
        load_compatible_hierarchical_state_dict(model, checkpoint["model"])
    return model.eval()


def _official_signature(state: Sequence[Any]) -> tuple[Any, ...]:
    observation = state[0].observation
    return (
        tuple(value.reward for value in state),
        tuple(value.status for value in state),
        int(observation.step),
        copy.deepcopy(observation.farms),
        copy.deepcopy(observation.market),
        copy.deepcopy(observation.town),
        tuple(copy.deepcopy(value.observation.private) for value in state),
    )


def _fast_signature(env: FastKaggricultureEnv) -> tuple[Any, ...]:
    observations = env.observations()
    state = env._require_state()  # validation intentionally audits engine state
    return (
        tuple(value.reward for value in state),
        tuple(value.status for value in state),
        int(observations[0].step),
        copy.deepcopy(observations[0].farms),
        copy.deepcopy(observations[0].market),
        copy.deepcopy(observations[0].town),
        tuple(copy.deepcopy(value.private) for value in observations),
    )


def _unit_is_active(action: Mapping[str, Any]) -> bool:
    unit_actions = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    return any(value and str(value[0]) != "PASS" for value in unit_actions)


def _market_key(action: Mapping[str, Any]) -> tuple[tuple[Any, ...], ...]:
    return tuple(tuple(value) for value in action.get("market", []) if value)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    device = torch.device(args.device)
    model = _model_from_checkpoint(args.checkpoint, device, args)
    episode_rows: list[dict[str, Any]] = []
    latency_ms: list[float] = []
    failure_counts: Counter[str] = Counter()
    termination_counts: Counter[str] = Counter()
    unique_chains: set[str] = set()
    stalled_chains: set[str] = set()
    max_stage_index = 0
    joint_market_unit_steps = 0
    resource_order_steps = 0
    max_repeated_market_steps = 0
    official_steps_compared = 0

    for episode in range(args.episodes):
        seed = args.seed + episode
        fast = FastKaggricultureEnv(
            configuration={"episodeSteps": args.episode_steps},
            copy_observations=False,
        )
        observations = list(fast.reset(seed))
        memories = [HierarchicalMemory(), HierarchicalMemory()]
        official = None
        if args.official_differential:
            from kaggle_environments import make

            official = make(
                "kaggriculture",
                configuration={"episodeSteps": args.episode_steps, "seed": seed},
                debug=False,
            )
            if _fast_signature(fast) != _official_signature(official.state):
                raise AssertionError(f"initial state mismatch at seed {seed}")

        started = time.perf_counter()
        previous_market: list[tuple[tuple[Any, ...], ...]] = [(), ()]
        repeated_market = [0, 0]
        while not fast.done:
            before = time.perf_counter()
            batch = hierarchical_policy_batch(
                model,
                observations,
                device,
                memories=memories,
                deterministic=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            latency_ms.append((time.perf_counter() - before) * 1000.0)
            actions = batch.actions
            for actor, action in enumerate(actions):
                market_key = _market_key(action)
                unit_active = _unit_is_active(action)
                if market_key and unit_active:
                    joint_market_unit_steps += 1
                if any(str(order[0]).startswith("BUY") for order in market_key):
                    resource_order_steps += 1
                if market_key and market_key == previous_market[actor]:
                    repeated_market[actor] += 1
                else:
                    repeated_market[actor] = 1 if market_key else 0
                previous_market[actor] = market_key
                max_repeated_market_steps = max(
                    max_repeated_market_steps, repeated_market[actor]
                )

            result = fast.step(actions)
            observations = list(result.observations)
            if official is not None:
                official.step(actions)
                official_steps_compared += 1
                if _fast_signature(fast) != _official_signature(official.state):
                    raise AssertionError(
                        f"official differential mismatch at seed={seed}, step={result.step}, "
                        f"actions={actions!r}"
                    )
            if any(status == "ERROR" for status in result.statuses):
                raise AssertionError(
                    f"environment error at seed={seed}, step={result.step}: {result.statuses}"
                )
            for memory in memories:
                for state in memory.task_states:
                    if state.chain_id:
                        unique_chains.add(state.chain_id)
                        max_stage_index = max(max_stage_index, state.stage_index)
                        if (
                            state.stage_started_step >= 0
                            and result.step - state.stage_started_step
                            >= args.stall_threshold
                        ):
                            stalled_chains.add(state.chain_id)

        expected_steps = args.episode_steps - 1
        if fast.step_index != expected_steps:
            raise AssertionError(
                f"seed {seed} ended at {fast.step_index}, expected {expected_steps}"
            )
        for memory in memories:
            diagnostics = memory.executor_diagnostics()
            failure_counts.update(diagnostics["by_reason"])
            termination_counts.update(
                row["termination_reason"]
                for row in diagnostics["task_state_machine"]["recent_transitions"]
                if row["phase"] in ("COMPLETED", "CANCELLED")
            )
        state = fast._require_state()
        episode_rows.append(
            {
                "seed": seed,
                "steps": fast.step_index,
                "statuses": [value.status for value in state],
                "rewards": [value.reward for value in state],
                "duration_seconds": time.perf_counter() - started,
            }
        )

    latency = np.asarray(latency_ms, dtype=np.float64)
    return {
        "schema": "kaggriculture.hierarchical-validation.v7",
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "device": str(device),
        "engine": "kaggle-environments 1.32.6 official interpreter",
        "episodes": episode_rows,
        "summary": {
            "episodes": args.episodes,
            "full_steps": int(sum(row["steps"] for row in episode_rows)),
            "all_done": all(
                row["statuses"] == ["DONE", "DONE"] for row in episode_rows
            ),
            "official_differential_steps": official_steps_compared,
            "inference_latency_ms_mean": float(latency.mean()),
            "inference_latency_ms_p95": float(np.percentile(latency, 95)),
            "inference_latency_ms_max": float(latency.max()),
            "executor_failures": int(sum(failure_counts.values())),
            "executor_failures_by_reason": dict(sorted(failure_counts.items())),
            "task_terminations": dict(sorted(termination_counts.items())),
            "unique_task_chains": len(unique_chains),
            "max_stage_index": max_stage_index,
            "stalled_task_chains": len(stalled_chains),
            "stall_threshold_steps": args.stall_threshold,
            "joint_market_and_unit_steps": joint_market_unit_steps,
            "resource_order_steps": resource_order_steps,
            "max_consecutive_identical_market_steps": max_repeated_market_steps,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stall-threshold", type=int, default=48)
    parser.add_argument("--official-differential", action="store_true")
    parser.add_argument("--hidden-size", type=int, default=384)
    parser.add_argument("--board-width", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--proposal-top-k", type=int, default=24)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hierarchical_v7_validation.json")
    )
    args = parser.parse_args()
    if min(args.episodes, args.episode_steps, args.stall_threshold) <= 0:
        parser.error("episode counts and thresholds must be positive")

    report = run_validation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
