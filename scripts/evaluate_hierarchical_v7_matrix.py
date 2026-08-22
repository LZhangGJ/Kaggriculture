"""Seat-swapped checkpoint evaluation and focused V7 ablations."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_policy import MAX_UNITS
from kaggriculture_lab.hierarchical_policy import (
    HierarchicalKaggriculturePolicy,
    hierarchical_policy_batch,
    load_compatible_hierarchical_state_dict,
)
from kaggriculture_lab.hierarchical_schema import HierarchicalMemory, UnitTaskState


VARIANTS = ("full", "no_learned_proposal", "stateless_tasks")


def _load_model(path: Path, device: torch.device) -> HierarchicalKaggriculturePolicy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = HierarchicalKaggriculturePolicy(
        hidden_size=int(checkpoint.get("hidden_size", 384)),
        board_width=int(checkpoint.get("board_width", 64)),
        d_model=int(checkpoint.get("d_model", 128)),
        transformer_layers=int(checkpoint.get("transformer_layers", 2)),
        transformer_heads=int(checkpoint.get("transformer_heads", 4)),
        proposal_top_k=int(checkpoint.get("proposal_top_k", 24)),
        opponent_clusters=int(checkpoint.get("opponent_clusters", 8)),
    ).to(device)
    load_compatible_hierarchical_state_dict(model, checkpoint["model"])
    return model.eval()


def _run_game(
    model: HierarchicalKaggriculturePolicy,
    opponent_spec: str,
    *,
    seed: int,
    model_seat: int,
    variant: str,
    episode_steps: int,
    device: torch.device,
) -> dict[str, Any]:
    opponent = resolve_agent(opponent_spec)
    env = FastKaggricultureEnv(configuration={"episodeSteps": episode_steps})
    observations = list(env.reset(seed))
    memory = HierarchicalMemory()
    inference_ms: list[float] = []
    original_top_k = model.proposal_top_k
    if variant == "no_learned_proposal":
        model.proposal_top_k = 0
    try:
        while not env.done:
            if variant == "stateless_tasks":
                memory.decision.reset()
                memory.task_states[:] = [UnitTaskState() for _ in range(MAX_UNITS)]
            started = time.perf_counter()
            policy = hierarchical_policy_batch(
                model,
                [observations[model_seat]],
                device,
                memories=[memory],
                deterministic=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_ms.append((time.perf_counter() - started) * 1000.0)
            opponent_seat = 1 - model_seat
            opponent_action = opponent(
                observations[opponent_seat], env.configuration
            )
            actions = [opponent_action, opponent_action]
            actions[model_seat] = policy.actions[0]
            result = env.step(actions)
            observations = list(result.observations)
            if any(status == "ERROR" for status in result.statuses):
                raise RuntimeError(
                    f"ERROR status at seed={seed}, seat={model_seat}, step={result.step}"
                )
    finally:
        model.proposal_top_k = original_top_k
    state = env._require_state()
    rewards = [value.reward for value in state]
    own_reward = float(rewards[model_seat] or 0.0)
    opponent_reward = float(rewards[1 - model_seat] or 0.0)
    outcome = 1 if own_reward > opponent_reward else -1 if own_reward < opponent_reward else 0
    diagnostics = memory.executor_diagnostics()
    latency = np.asarray(inference_ms, dtype=np.float64)
    return {
        "seed": seed,
        "model_seat": model_seat,
        "outcome": outcome,
        "reward": own_reward,
        "opponent_reward": opponent_reward,
        "margin": own_reward - opponent_reward,
        "statuses": [value.status for value in state],
        "steps": env.step_index,
        "executor_failures": diagnostics["failure_count"],
        "mode_switches": diagnostics["strategy"]["switches"],
        "mean_inference_ms": float(latency.mean()),
        "p95_inference_ms": float(np.percentile(latency, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--opponents", nargs="+", default=["starter", "pass"])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--episode-steps", type=int, default=720)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/hierarchical_v7_eval_matrix.json")
    )
    args = parser.parse_args()
    if args.seeds <= 0 or args.episode_steps <= 0:
        parser.error("--seeds and --episode-steps must be positive")
    device = torch.device(args.device)
    model = _load_model(args.checkpoint, device)
    matrix: list[dict[str, Any]] = []
    for variant in args.variants:
        for opponent in args.opponents:
            games = [
                _run_game(
                    model,
                    opponent,
                    seed=args.seed + offset,
                    model_seat=seat,
                    variant=variant,
                    episode_steps=args.episode_steps,
                    device=device,
                )
                for offset in range(args.seeds)
                for seat in (0, 1)
            ]
            outcomes = Counter(game["outcome"] for game in games)
            matrix.append(
                {
                    "variant": variant,
                    "opponent": opponent,
                    "games": len(games),
                    "wins": outcomes[1],
                    "ties": outcomes[0],
                    "losses": outcomes[-1],
                    "score_rate": (outcomes[1] + 0.5 * outcomes[0]) / len(games),
                    "mean_margin": float(np.mean([game["margin"] for game in games])),
                    "mean_inference_ms": float(
                        np.mean([game["mean_inference_ms"] for game in games])
                    ),
                    "executor_failures": int(
                        sum(game["executor_failures"] for game in games)
                    ),
                    "game_rows": games,
                }
            )
            row = matrix[-1]
            print(
                f"variant={variant} opponent={opponent} "
                f"score={row['score_rate']:.3f} margin={row['mean_margin']:.1f} "
                f"failures={row['executor_failures']}"
            )
    report = {
        "schema": "kaggriculture.hierarchical-eval-matrix.v7",
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "episode_steps": args.episode_steps,
        "seed_start": args.seed,
        "seed_count": args.seeds,
        "seat_swapped": True,
        "matrix": matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
