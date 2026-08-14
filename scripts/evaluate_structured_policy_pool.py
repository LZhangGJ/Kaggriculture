"""Evaluate one PolicyV2 checkpoint against every agent in a frozen pool."""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import torch

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.policy_v2 import policy_from_checkpoint, structured_policy_batch


def _isolated_agent(spec: str, tag: str) -> Callable[[Any, Any], dict[str, Any]]:
    path = Path(spec)
    if not path.is_file():
        return resolve_agent(spec)
    module_spec = importlib.util.spec_from_file_location(f"pool_agent_{tag}", path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot load agent from {path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    raw_agent = module.agent
    positional = [
        parameter
        for parameter in inspect.signature(raw_agent).parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return raw_agent
    return lambda observation, configuration: raw_agent(observation)


def _summary(margins: list[float], policy: list[float], opponents: list[float]) -> dict[str, float]:
    wins = sum(margin > 0 for margin in margins)
    losses = sum(margin < 0 for margin in margins)
    ties = len(margins) - wins - losses
    return {
        "games": len(margins),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "score_rate": (wins + 0.5 * ties) / max(1, len(margins)),
        "mean_policy_reward": sum(policy) / max(1, len(policy)),
        "mean_opponent_reward": sum(opponents) / max(1, len(opponents)),
        "mean_margin": sum(margins) / max(1, len(margins)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=50_000)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--no-unit-mask", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    pool = [(entry["name"], entry["path"]) for entry in manifest["agents"]]
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device).eval()

    games = [
        (name, path, seed, seat)
        for name, path in pool
        for seed in range(args.seed_start, args.seed_start + args.seeds)
        for seat in (0, 1)
    ]
    environments = VectorFastEnv(len(games))
    observations = environments.reset(seed for _, _, seed, _ in games)
    opponents = [
        _isolated_agent(path, f"{index}_{name}")
        for index, (name, path, _, _) in enumerate(games)
    ]
    results = None
    while not environments.envs[0].done:
        policy_observations = [
            pair[seat] for pair, (_, _, _, seat) in zip(observations, games, strict=True)
        ]
        policy_actions = structured_policy_batch(
            model,
            policy_observations,
            device,
            deterministic=True,
            mask_unit_actions=not args.no_unit_mask,
        ).actions
        action_pairs = []
        for index, (pair, (_, _, _, seat)) in enumerate(zip(observations, games, strict=True)):
            opponent_action = opponents[index](
                pair[1 - seat], environments.envs[index].configuration
            )
            action_pairs.append(
                [policy_actions[index], opponent_action]
                if seat == 0
                else [opponent_action, policy_actions[index]]
            )
        results = environments.step(action_pairs)
        observations = [result.observations for result in results]

    assert results is not None
    grouped: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"margins": [], "policy": [], "opponents": []}
    )
    grouped_by_seat: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: {"margins": [], "policy": [], "opponents": []}
    )
    all_margins: list[float] = []
    all_policy: list[float] = []
    all_opponents: list[float] = []
    for result, (name, _, _, seat) in zip(results, games, strict=True):
        policy_reward = float(result.rewards[seat])
        opponent_reward = float(result.rewards[1 - seat])
        margin = policy_reward - opponent_reward
        grouped[name]["margins"].append(margin)
        grouped[name]["policy"].append(policy_reward)
        grouped[name]["opponents"].append(opponent_reward)
        grouped_by_seat[seat]["margins"].append(margin)
        grouped_by_seat[seat]["policy"].append(policy_reward)
        grouped_by_seat[seat]["opponents"].append(opponent_reward)
        all_margins.append(margin)
        all_policy.append(policy_reward)
        all_opponents.append(opponent_reward)

    report = {
        "checkpoint": str(args.checkpoint),
        "pool_manifest": str(args.pool_manifest),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "overall": _summary(all_margins, all_policy, all_opponents),
        "by_seat": {
            str(seat): _summary(values["margins"], values["policy"], values["opponents"])
            for seat, values in sorted(grouped_by_seat.items())
        },
        "opponents": {
            name: _summary(values["margins"], values["policy"], values["opponents"])
            for name, values in sorted(grouped.items())
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
