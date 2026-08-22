"""Self-play A2C for the V2 strategy-derived candidate policy."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn import functional as F

from kaggriculture_lab import VectorFastEnv
from kaggriculture_lab.board_rl import (
    generalized_advantage_estimates,
    paired_liquidatable_net_asset_potentials,
    relative_money_potential,
)
from kaggriculture_lab.decision_policy import DecisionKaggriculturePolicy
from kaggriculture_lab.decision_rl import (
    decision_actor_critic_batch,
    decision_values,
)
from kaggriculture_lab.decision_schema import DecisionMemory
from kaggriculture_lab.gpu_policy import flatten_environment_observations


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
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--win-bonus", type=float, default=1.0)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/decision_a2c_v2.pt")
    )
    args = parser.parse_args()
    if min(args.envs, args.updates, args.rollout_steps, args.episode_steps) <= 0:
        parser.error("environment and rollout counts must be positive")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = None
    if args.init_checkpoint is not None:
        checkpoint = torch.load(
            args.init_checkpoint, map_location="cpu", weights_only=True
        )
        for key in (
            "hidden_size",
            "board_width",
            "d_model",
            "transformer_layers",
            "transformer_heads",
        ):
            if key in checkpoint:
                setattr(args, key, int(checkpoint[key]))

    model = DecisionKaggriculturePolicy(
        hidden_size=args.hidden_size,
        board_width=args.board_width,
        d_model=args.d_model,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
    ).to(device).train()
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    environments = VectorFastEnv(
        args.envs, configuration={"episodeSteps": args.episode_steps}
    )
    observations = environments.reset(
        range(args.seed, args.seed + args.envs)
    )
    memories = [DecisionMemory() for _ in range(2 * args.envs)]
    next_seed = args.seed + args.envs

    for update in range(1, args.updates + 1):
        rollout_log_probs: list[torch.Tensor] = []
        rollout_entropies: list[torch.Tensor] = []
        rollout_values: list[torch.Tensor] = []
        rollout_rewards: list[torch.Tensor] = []
        rollout_dones: list[torch.Tensor] = []

        for _ in range(args.rollout_steps):
            flat = flatten_environment_observations(observations)
            before = torch.tensor(
                paired_liquidatable_net_asset_potentials(flat),
                dtype=torch.float32,
                device=device,
            )
            policy = decision_actor_critic_batch(
                model, flat, device, memories
            )
            paired_actions = [
                [policy.actions[2 * index], policy.actions[2 * index + 1]]
                for index in range(args.envs)
            ]
            results = environments.step(paired_actions)
            terminal_observations = [result.observations for result in results]
            terminal_flat = flatten_environment_observations(
                terminal_observations
            )
            after = torch.tensor(
                paired_liquidatable_net_asset_potentials(terminal_flat),
                dtype=torch.float32,
                device=device,
            )
            env_dones = [result.done for result in results]
            dones = torch.tensor(
                [done for done in env_dones for _ in range(2)],
                dtype=torch.bool,
                device=device,
            )
            rewards = args.reward_scale * (after - before)
            if args.win_bonus:
                money_outcome = torch.tensor(
                    [relative_money_potential(item) for item in terminal_flat],
                    dtype=torch.float32,
                    device=device,
                )
                rewards = rewards + dones * args.win_bonus * torch.sign(money_outcome)

            rollout_log_probs.append(policy.log_probs)
            rollout_entropies.append(policy.entropies)
            rollout_values.append(policy.values)
            rollout_rewards.append(rewards)
            rollout_dones.append(dones)
            if any(env_dones):
                if not all(env_dones):
                    raise RuntimeError("fixed-length vector games ended asynchronously")
                observations = environments.reset(
                    range(next_seed, next_seed + args.envs)
                )
                memories = [DecisionMemory() for _ in range(2 * args.envs)]
                next_seed += args.envs
            else:
                observations = terminal_observations

        values = torch.stack(rollout_values)
        rewards = torch.stack(rollout_rewards)
        dones = torch.stack(rollout_dones)
        with torch.no_grad():
            bootstrap = (
                torch.zeros_like(values[-1])
                if dones[-1].all()
                else decision_values(
                    model,
                    flatten_environment_observations(observations),
                    device,
                    memories,
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

        log_probs = torch.stack(rollout_log_probs)
        entropies = torch.stack(rollout_entropies)
        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        entropy = entropies.mean()
        loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm
        )
        optimizer.step()

        if update == 1 or update % args.log_every == 0:
            print(
                f"update={update} loss={loss.item():.4f} "
                f"policy={policy_loss.item():.4f} value={value_loss.item():.4f} "
                f"entropy={entropy.item():.4f} grad={float(grad_norm):.3f} "
                f"rollout_return={rewards.sum(dim=0).mean().item():.4f}"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "architecture": "decision_candidateformer_v2",
            "hidden_size": args.hidden_size,
            "board_width": args.board_width,
            "d_model": args.d_model,
            "transformer_layers": args.transformer_layers,
            "transformer_heads": args.transformer_heads,
            "updates": args.updates,
            "rollout_steps": args.rollout_steps,
            "episode_steps": args.episode_steps,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
