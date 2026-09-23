#!/usr/bin/env python3
"""Read-only seed-block gradient agreement on a frozen native PPO rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

import numpy as np
import torch

from experiments.train_midgame_autofill_v3 import build_model
from experiments.train_student_action_event_rl_v3 import (
    _batch_terms, _day_bundle_objective, _day_state_crossfit_advantages,
    _load_native_rollout, _restore_native_games,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit(args: argparse.Namespace) -> dict:
    torch.set_num_threads(args.threads)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata, arrays, game_metadata = _load_native_rollout(args.rollout)
    if metadata.get("policy_version") != sha256(args.checkpoint):
        raise RuntimeError("rollout is not from the frozen checkpoint")
    reward = metadata["reward"]
    games = _restore_native_games(
        arrays, game_metadata, float(reward["margin_weight"]),
        float(reward["margin_scale"]))
    critic, _ = _day_state_crossfit_advantages(
        games, arrays, checkpoint, SimpleNamespace(
            margin_weight=float(reward["margin_weight"]),
            margin_scale=float(reward["margin_scale"]),
            day_state_critic_workers=args.critic_workers))

    by_seed: dict[int, list[dict]] = {}
    for game in games:
        by_seed.setdefault(int(game["seed"]), []).append(game)
    chosen = sorted(by_seed)[:args.seeds]
    if len(chosen) != args.seeds or args.seeds % args.blocks:
        raise ValueError("need enough whole seeds, divisible by blocks")
    model = build_model(
        checkpoint["model_dimensions"]["causal_context"],
        checkpoint["model_dimensions"]["packed_observation"],
        checkpoint["model_dimensions"]["event_resources"],
        checkpoint["model_scale"])
    model.load_state_dict(checkpoint["model"])
    model.train()
    selected = [game for seed in chosen for game in by_seed[seed]]
    if args.grouping == "train_first":
        selected = [games[int(index)] for index in
                    np.random.default_rng(args.policy_seed).permutation(
                        len(games))[:len(selected)]]
        blocks = np.array_split(np.asarray(selected, dtype=object), args.blocks)
    elif args.grouping == "game":
        np.random.default_rng(0).shuffle(selected)
        blocks = np.array_split(np.asarray(selected, dtype=object), args.blocks)
    else:
        blocks = [[game for seed in seed_block for game in by_seed[int(seed)]]
                  for seed_block in np.array_split(np.asarray(chosen), args.blocks)]
    gradients = []
    if args.device.startswith("npu"):
        import torch_npu  # noqa: F401
        torch.npu.set_device(args.device)
    device = torch.device(args.device)
    model.to(device)
    captured = {}
    original_initial_hidden = model.initial_hidden
    def capture_initial_hidden(*inputs):
        hidden = original_initial_hidden(*inputs)
        hidden.retain_grad()
        captured["hidden"] = hidden
        captured["hidden_before"] = hidden.detach().cpu().clone()
        captured["context"] = inputs[0]
        captured["observation"] = inputs[1]
        captured["length"] = inputs[2]
        captured["token_continuous"] = inputs[3]
        captured["token_categories"] = inputs[4]
        captured["token_count"] = inputs[5]
        return hidden
    model.initial_hidden = capture_initial_hidden
    for block_index, block in enumerate(blocks):
        new, old, entropy, active, _, event_days, day_games = _batch_terms(
            model, block, device, 1.0,
            safe_hidden_index_copy=args.safe_hidden_index_copy)
        day_advantages = torch.tensor([
            float(day["day_state_advantage"])
            if args.step is None or int(day["step"]) == args.step else 0.0
            for game in block for day in game["days"]], dtype=torch.float32)
        policy_loss, day_entropy, _ = _day_bundle_objective(
            new, old, entropy, active, event_days, day_games,
            torch.zeros(len(block), device=device), 0.2,
            day_advantages=day_advantages.to(device))
        loss = (policy_loss if args.component == "policy" else
                -0.01 * day_entropy if args.component == "entropy" else
                policy_loss - 0.01 * day_entropy)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if block_index == 0 and args.gradient_npz is not None:
            if args.gradient_npz.exists():
                raise FileExistsError(args.gradient_npz)
            np.savez(args.gradient_npz, **{
                name: parameter.grad.detach().cpu().numpy()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }, **{
                f"__token_category_{index}": value.detach().cpu().numpy()
                for index, value in enumerate(captured["token_categories"])
            }, __hidden_before=captured["hidden_before"].numpy(),
                __hidden_output=captured["hidden"].detach().cpu().numpy(),
                __hidden_grad=captured["hidden"].grad.detach().cpu().numpy(),
                __context=captured["context"].detach().cpu().numpy(),
                __observation=captured["observation"].detach().cpu().numpy(),
                __length=captured["length"].detach().cpu().numpy(),
                __token_continuous=captured["token_continuous"].detach().cpu().numpy(),
                __token_count=captured["token_count"].detach().cpu().numpy())
        gradient = torch.cat([
            parameter.grad.detach().flatten().cpu()
            for parameter in model.parameters() if parameter.grad is not None
        ]).numpy().copy()
        if not np.all(np.isfinite(gradient)):
            raise RuntimeError("non-finite policy gradient")
        gradients.append(gradient)
    values = np.stack(gradients).astype(np.float64)
    norms = np.linalg.norm(values, axis=1)
    mean = values.mean(axis=0)
    noise = float(np.sqrt(np.mean(np.sum((values - mean) ** 2, axis=1))))
    cosine = values @ values.T / np.outer(norms, norms)
    pairs = cosine[np.triu_indices(args.blocks, 1)]
    return {
        "scope": "read_only_frozen_policy_seed_block_gradient",
        "checkpoint_sha256": sha256(args.checkpoint),
        "rollout_sha256": sha256(args.rollout),
        "step": args.step,
        "grouping": args.grouping,
        "component": args.component,
        "device": args.device,
        "safe_hidden_index_copy": (args.safe_hidden_index_copy
                                   if args.safe_hidden_index_copy is not None
                                   else device.type == "npu"),
        "seeds": args.seeds,
        "games": sum(len(by_seed[seed]) for seed in chosen),
        "blocks": args.blocks,
        "parameters_with_grad": int(values.shape[1]),
        "mean_pairwise_cosine": float(pairs.mean()),
        "median_pairwise_cosine": float(np.median(pairs)),
        "pairwise_cosine_min": float(pairs.min()),
        "pairwise_cosine_max": float(pairs.max()),
        "mean_gradient_norm": float(np.linalg.norm(mean)),
        "median_block_gradient_norm": float(np.median(norms)),
        "first_block_gradient_norm": float(norms[0]),
        "block_noise_rms": noise,
        "full_batch_snr_proxy": (
            float(np.sqrt(args.blocks) * np.linalg.norm(mean) / noise)
            if noise else None),
        "critic_variance_ratio": critic["variance_ratio"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=8)
    parser.add_argument("--step", type=int)
    parser.add_argument("--grouping", choices=("seed", "game", "train_first"),
                        default="seed")
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--component", choices=("policy", "entropy", "full"),
                        default="policy")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--safe-hidden-index-copy", action="store_true",
                        default=None)
    parser.add_argument("--gradient-npz", type=Path)
    parser.add_argument("--critic-workers", type=int, default=4)
    args = parser.parse_args()
    if (args.seeds < 2 or args.blocks < 2 or args.threads < 1 or
            args.critic_workers < 1 or args.output.exists()):
        parser.error("invalid count or output already exists")
    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
