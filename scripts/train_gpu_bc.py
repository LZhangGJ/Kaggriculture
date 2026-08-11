"""Warm-start the CUDA policy by cloning trusted scripted agents."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.fast_env import resolve_agent
from kaggriculture_lab.gpu_policy import (
    KaggriculturePolicy,
    action_masks,
    action_targets,
    encode_batch,
    flatten_environment_observations,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=128)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--teacher-a", default="starter")
    parser.add_argument("--teacher-b", default="starter")
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/bc_starter.pt"))
    args = parser.parse_args()

    device = torch.device(args.device)
    model = KaggriculturePolicy(args.hidden_size).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    environments = VectorFastEnv(args.envs)
    observations = environments.reset(range(args.envs))
    teachers = (resolve_agent(args.teacher_a), resolve_agent(args.teacher_b))

    for update in range(1, args.updates + 1):
        flat = flatten_environment_observations(observations)
        teacher_actions = [teachers[index % 2](observation, environments.envs[index // 2].configuration) for index, observation in enumerate(flat)]
        features_np, unit_context_np, _ = encode_batch(flat)
        unit_masks_np, market_masks_np = action_masks(flat)
        unit_targets_np, market_targets_np, active_np = action_targets(flat, teacher_actions)

        features = torch.as_tensor(features_np, device=device)
        unit_context = torch.as_tensor(unit_context_np, device=device)
        unit_masks = torch.as_tensor(unit_masks_np, device=device)
        market_masks = torch.as_tensor(market_masks_np, device=device)
        unit_targets = torch.as_tensor(unit_targets_np, device=device)
        market_targets = torch.as_tensor(market_targets_np, device=device)
        active = torch.as_tensor(active_np, device=device)

        unit_logits, market_logits, _ = model(features, unit_context)
        unit_logits = unit_logits.masked_fill(~unit_masks, torch.finfo(unit_logits.dtype).min)
        market_logits = market_logits.masked_fill(~market_masks, torch.finfo(market_logits.dtype).min)
        unit_loss = F.cross_entropy(unit_logits[active], unit_targets[active])
        market_loss = F.cross_entropy(market_logits, market_targets)
        loss = unit_loss + market_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        paired_actions = [[teacher_actions[2 * index], teacher_actions[2 * index + 1]] for index in range(args.envs)]
        results = environments.step(paired_actions)
        if results[0].done:
            base = update * args.envs
            observations = environments.reset(range(base, base + args.envs))
        else:
            observations = [result.observations for result in results]

        if update == 1 or update % 50 == 0:
            unit_accuracy = (unit_logits[active].argmax(-1) == unit_targets[active]).float().mean().item()
            market_accuracy = (market_logits.argmax(-1) == market_targets).float().mean().item()
            print(f"update={update} loss={loss.item():.4f} unit_acc={unit_accuracy:.3f} market_acc={market_accuracy:.3f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "hidden_size": args.hidden_size}, args.output)
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()

