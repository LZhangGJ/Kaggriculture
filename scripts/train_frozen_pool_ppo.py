"""Fine-tune PolicyV2 with PPO against an immutable opponent pool.

The official CPU interpreter remains the source of truth for rollouts.  Policy
inference and PPO minibatches are batched on CUDA.  Each rollout covers both
seats for every selected frozen opponent so seat bias cannot silently dominate
an update.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_policy import MAX_UNITS, UNIT_ACTIONS, encode_batch
from kaggriculture_lab.policy_v2 import (
    MARKET_TOKENS,
    MAX_MARKET_ORDERS,
    StructuredKaggriculturePolicy,
    decode_structured_actions,
    policy_from_checkpoint,
)


UNIT_QUANTITY_TOKENS = torch.tensor(
    [index for index, (op, _) in enumerate(UNIT_ACTIONS) if op in {"PICKUP", "PLACE"}],
    dtype=torch.long,
)
MARKET_QUANTITY_TOKENS = torch.tensor(
    [
        index
        for index, (op, _) in enumerate(MARKET_TOKENS)
        if op not in {"NONE", "HIRE", "BUY_LAND"}
    ],
    dtype=torch.long,
)


@dataclass
class Rollout:
    features: np.ndarray
    unit_context: np.ndarray
    unit_indices: np.ndarray
    unit_quantities: np.ndarray
    market_indices: np.ndarray
    market_quantities: np.ndarray
    old_log_prob: np.ndarray
    old_values: np.ndarray
    returns: np.ndarray
    games: int
    steps: int
    wins: int
    ties: int
    losses: int
    mean_margin: float
    mean_return: float


def _isolated_agent(spec: str, tag: str) -> Callable[[Any, Any], dict[str, Any]]:
    path = Path(spec)
    if not path.is_file():
        return resolve_agent(spec)
    module_spec = importlib.util.spec_from_file_location(f"ppo_pool_agent_{tag}", path)
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


def _selected(mask_source: torch.Tensor, choices: torch.Tensor) -> torch.Tensor:
    """Return whether each choice appears in a small one-dimensional token set."""

    tokens = mask_source.to(choices.device)
    return (choices.unsqueeze(-1) == tokens).any(dim=-1)


def _log_prob_and_entropy(
    logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    unit_context: torch.Tensor,
    unit_indices: torch.Tensor,
    unit_quantities: torch.Tensor,
    market_indices: torch.Tensor,
    market_quantities: torch.Tensor,
    components: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    unit_logits, unit_quantity_logits, market_logits, market_quantity_logits = logits
    unit_dist = torch.distributions.Categorical(logits=unit_logits)
    unit_quantity_dist = torch.distributions.Categorical(logits=unit_quantity_logits)
    market_dist = torch.distributions.Categorical(logits=market_logits)
    market_quantity_dist = torch.distributions.Categorical(logits=market_quantity_logits)

    unit_active = unit_context[..., 2] > 0.5
    unit_quantity_active = unit_active & _selected(UNIT_QUANTITY_TOKENS, unit_indices)
    market_quantity_active = _selected(MARKET_QUANTITY_TOKENS, market_indices)

    log_prob = market_dist.log_prob(market_indices).sum(dim=1)
    log_prob = log_prob + (
        market_quantity_dist.log_prob(market_quantities) * market_quantity_active
    ).sum(dim=1)
    entropy = market_dist.entropy().sum(dim=1)
    entropy = entropy + (
        market_quantity_dist.entropy() * market_quantity_active
    ).sum(dim=1)
    if components == "all":
        log_prob = log_prob + (unit_dist.log_prob(unit_indices) * unit_active).sum(dim=1)
        log_prob = log_prob + (
            unit_quantity_dist.log_prob(unit_quantities) * unit_quantity_active
        ).sum(dim=1)
        entropy = entropy + (unit_dist.entropy() * unit_active).sum(dim=1)
        entropy = entropy + (
            unit_quantity_dist.entropy() * unit_quantity_active
        ).sum(dim=1)
    return log_prob, entropy


@torch.no_grad()
def _sample_policy(
    model: StructuredKaggriculturePolicy,
    observations: list[Any],
    device: torch.device,
    components: str,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
]:
    features_np, unit_context_np, _ = encode_batch(observations)
    features = torch.as_tensor(features_np, device=device)
    unit_context = torch.as_tensor(unit_context_np, device=device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        unit_logits, unit_quantity_logits, market_logits, market_quantity_logits, values = model(
            features, unit_context
        )
    if components == "all":
        unit_indices = torch.distributions.Categorical(logits=unit_logits).sample()
        unit_quantities = torch.distributions.Categorical(logits=unit_quantity_logits).sample()
    else:
        unit_indices = unit_logits.argmax(dim=-1)
        unit_quantities = unit_quantity_logits.argmax(dim=-1)
    market_indices = torch.distributions.Categorical(logits=market_logits).sample()
    market_quantities = torch.distributions.Categorical(logits=market_quantity_logits).sample()
    log_prob, _ = _log_prob_and_entropy(
        (unit_logits, unit_quantity_logits, market_logits, market_quantity_logits),
        unit_context,
        unit_indices,
        unit_quantities,
        market_indices,
        market_quantities,
        components,
    )
    actions = decode_structured_actions(
        unit_indices.cpu().numpy(),
        unit_quantities.cpu().numpy(),
        market_indices.cpu().numpy(),
        market_quantities.cpu().numpy(),
        observations,
    )
    return (
        features_np.astype(np.float16),
        unit_context_np.astype(np.float16),
        unit_indices.cpu().numpy().astype(np.int16),
        unit_quantities.cpu().numpy().astype(np.int16),
        market_indices.cpu().numpy().astype(np.int16),
        market_quantities.cpu().numpy().astype(np.int16),
        log_prob.float().cpu().numpy(),
        values.float().cpu().numpy(),
        actions,
    )


def collect_rollout(
    model: StructuredKaggriculturePolicy,
    pool: list[tuple[str, str]],
    seed_start: int,
    seeds_per_opponent: int,
    device: torch.device,
    score_scale: float,
    components: str,
) -> Rollout:
    games = [
        (name, path, seed, seat)
        for name, path in pool
        for seed in range(seed_start, seed_start + seeds_per_opponent)
        for seat in (0, 1)
    ]
    environments = VectorFastEnv(len(games))
    observations = environments.reset(seed for _, _, seed, _ in games)
    opponents = [
        _isolated_agent(path, f"{index}_{name}")
        for index, (name, path, _, _) in enumerate(games)
    ]
    buffers: list[list[np.ndarray]] = [[] for _ in range(8)]
    steps = 0
    results = None
    model.eval()
    while not environments.envs[0].done:
        policy_observations = [
            pair[seat] for pair, (_, _, _, seat) in zip(observations, games, strict=True)
        ]
        sample = _sample_policy(model, policy_observations, device, components)
        for buffer, values in zip(buffers, sample[:8], strict=True):
            buffer.append(values)
        policy_actions = sample[8]
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
        steps += 1

    assert results is not None
    margins = np.asarray(
        [
            float(result.rewards[seat]) - float(result.rewards[1 - seat])
            for result, (_, _, _, seat) in zip(results, games, strict=True)
        ],
        dtype=np.float32,
    )
    signs = np.sign(margins)
    episode_returns = 0.75 * signs + 0.25 * np.tanh(margins / score_scale)
    flat = [np.concatenate(values, axis=0) for values in buffers]
    returns = np.tile(episode_returns, steps).astype(np.float32)
    return Rollout(
        features=flat[0],
        unit_context=flat[1],
        unit_indices=flat[2],
        unit_quantities=flat[3],
        market_indices=flat[4],
        market_quantities=flat[5],
        old_log_prob=flat[6].astype(np.float32),
        old_values=flat[7].astype(np.float32),
        returns=returns,
        games=len(games),
        steps=steps,
        wins=int((margins > 0).sum()),
        ties=int((margins == 0).sum()),
        losses=int((margins < 0).sum()),
        mean_margin=float(margins.mean()),
        mean_return=float(episode_returns.mean()),
    )


def ppo_update(
    model: StructuredKaggriculturePolicy,
    rollout: Rollout,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    epochs: int,
    clip_ratio: float,
    value_coefficient: float,
    entropy_coefficient: float,
    max_grad_norm: float,
    components: str,
) -> dict[str, float]:
    advantages = rollout.returns - rollout.old_values
    advantages = (advantages - advantages.mean()) / max(1e-6, float(advantages.std()))
    sample_count = len(advantages)
    totals = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    batches = 0
    model.train()
    for _ in range(epochs):
        order = np.random.permutation(sample_count)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            features = torch.as_tensor(
                rollout.features[indices], device=device, dtype=torch.float32
            )
            unit_context = torch.as_tensor(
                rollout.unit_context[indices], device=device, dtype=torch.float32
            )
            unit_indices = torch.as_tensor(
                rollout.unit_indices[indices], device=device, dtype=torch.long
            )
            unit_quantities = torch.as_tensor(
                rollout.unit_quantities[indices], device=device, dtype=torch.long
            )
            market_indices = torch.as_tensor(
                rollout.market_indices[indices], device=device, dtype=torch.long
            )
            market_quantities = torch.as_tensor(
                rollout.market_quantities[indices], device=device, dtype=torch.long
            )
            old_log_prob = torch.as_tensor(
                rollout.old_log_prob[indices], device=device, dtype=torch.float32
            )
            returns = torch.as_tensor(
                rollout.returns[indices], device=device, dtype=torch.float32
            )
            advantage = torch.as_tensor(advantages[indices], device=device, dtype=torch.float32)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                outputs = model(features, unit_context)
                new_log_prob, entropy = _log_prob_and_entropy(
                    outputs[:4],
                    unit_context,
                    unit_indices,
                    unit_quantities,
                    market_indices,
                    market_quantities,
                    components,
                )
                values = outputs[4].float()
                log_ratio = (new_log_prob.float() - old_log_prob).clamp(-20.0, 20.0)
                ratio = log_ratio.exp()
                unclipped = ratio * advantage
                clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantage
                policy_loss = -torch.minimum(unclipped, clipped).mean()
                value_loss = nn.functional.mse_loss(values, returns)
                entropy_mean = entropy.float().mean()
                loss = (
                    policy_loss
                    + value_coefficient * value_loss
                    - entropy_coefficient * entropy_mean
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss)
                totals["value_loss"] += float(value_loss)
                totals["entropy"] += float(entropy_mean)
                totals["approx_kl"] += float((old_log_prob - new_log_prob.float()).mean())
                totals["clip_fraction"] += float(
                    ((ratio - 1.0).abs() > clip_ratio).float().mean()
                )
                batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def _rescale_route_logits(
    model: StructuredKaggriculturePolicy, scale: float, components: str
) -> None:
    if math.isclose(scale, 1.0):
        return
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            selected = components == "all" or name.startswith("market_")
            if "route_logits" in name and selected:
                parameter.mul_(scale)


def _configure_trainable_parameters(
    model: StructuredKaggriculturePolicy, components: str
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Freeze unit behavior during conservative market-only PPO."""

    if components == "all":
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(
                name.startswith("market_token_head")
                or name.startswith("market_quantity_head")
                or name.startswith("market_route_logits")
                or name.startswith("market_quantity_route_logits")
                or name.startswith("value_head")
            )
    route_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "route_logits" in name
    ]
    base_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "route_logits" not in name
    ]
    return base_parameters, route_parameters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--seeds-per-opponent", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=60_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--route-lr", type=float, default=1e-5)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--value-coefficient", type=float, default=0.25)
    parser.add_argument("--entropy-coefficient", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--score-scale", type=float, default=10_000.0)
    parser.add_argument(
        "--ppo-components",
        choices=("market", "all"),
        default="market",
        help="market keeps the BC unit route deterministic and frozen",
    )
    parser.add_argument(
        "--route-scale",
        type=float,
        default=1.0,
        help="one-time exploration scaling applied to route logits before iteration 1",
    )
    parser.add_argument("--max-opponents", type=int)
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = policy_from_checkpoint(checkpoint, device)
    _rescale_route_logits(model, args.route_scale, args.ppo_components)
    manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    pool = [(entry["name"], entry["path"]) for entry in manifest["agents"]]
    if args.max_opponents is not None:
        pool = pool[: args.max_opponents]
    if not pool:
        parser.error("pool manifest contains no opponents")

    base_parameters, route_parameters = _configure_trainable_parameters(
        model, args.ppo_components
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.lr},
            {"params": route_parameters, "lr": args.route_lr, "weight_decay": 0.0},
        ],
        weight_decay=1e-4,
    )
    history: list[dict[str, Any]] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for iteration in range(1, args.iterations + 1):
        rollout_seed = args.seed_start + (iteration - 1) * args.seeds_per_opponent
        rollout = collect_rollout(
            model,
            pool,
            rollout_seed,
            args.seeds_per_opponent,
            device,
            args.score_scale,
            args.ppo_components,
        )
        update = ppo_update(
            model,
            rollout,
            optimizer,
            device,
            args.batch_size,
            args.ppo_epochs,
            args.clip_ratio,
            args.value_coefficient,
            args.entropy_coefficient,
            args.max_grad_norm,
            args.ppo_components,
        )
        record = {
            "iteration": iteration,
            "seed_start": rollout_seed,
            "samples": len(rollout.returns),
            "games": rollout.games,
            "steps": rollout.steps,
            "wins": rollout.wins,
            "ties": rollout.ties,
            "losses": rollout.losses,
            "score_rate": (rollout.wins + 0.5 * rollout.ties) / rollout.games,
            "mean_margin": rollout.mean_margin,
            "mean_return": rollout.mean_return,
            **update,
        }
        history.append(record)
        output_checkpoint = dict(checkpoint)
        output_checkpoint.update(
            {
                "model": model.state_dict(),
                "ppo": {
                    "source_checkpoint": str(args.checkpoint),
                    "pool_manifest": str(args.pool_manifest),
                    "route_scale": args.route_scale,
                    "components": args.ppo_components,
                    "history": history,
                },
            }
        )
        torch.save(output_checkpoint, args.output)
        print(json.dumps(record, sort_keys=True), flush=True)
        print(f"saved={args.output}", flush=True)


if __name__ == "__main__":
    main()
