"""Evaluate two Kaggriculture agents on fixed seeds and both seats."""

from __future__ import annotations

import argparse
import json
from statistics import fmean

from kaggriculture_lab.fast_env import run_duel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent_a", help="built-in name, Python file, or module:callable")
    parser.add_argument("agent_b", help="built-in name, Python file, or module:callable")
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    seeds = range(args.seed_start, args.seed_start + args.seeds)
    results = run_duel(
        args.agent_a,
        args.agent_b,
        seeds,
        both_seats=True,
        workers=args.workers,
    )
    rewards_a = [float(result.rewards[0]) for result in results]
    rewards_b = [float(result.rewards[1]) for result in results]
    margins = [left - right for left, right in zip(rewards_a, rewards_b, strict=True)]
    wins = sum(margin > 0 for margin in margins)
    losses = sum(margin < 0 for margin in margins)
    ties = len(margins) - wins - losses

    paired_margins = [fmean(margins[index : index + 2]) for index in range(0, len(margins), 2)]
    paired_wins = sum(margin > 0 for margin in paired_margins)
    paired_losses = sum(margin < 0 for margin in paired_margins)
    paired_ties = len(paired_margins) - paired_wins - paired_losses

    summary = {
        "agent_a": args.agent_a,
        "agent_b": args.agent_b,
        "games": len(results),
        "seed_blocks": args.seeds,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "score_rate": (wins + 0.5 * ties) / len(results),
        "mean_reward_a": fmean(rewards_a),
        "mean_reward_b": fmean(rewards_b),
        "mean_margin_a": fmean(margins),
        "paired_wins": paired_wins,
        "paired_ties": paired_ties,
        "paired_losses": paired_losses,
        "paired_mean_margin_a": fmean(paired_margins),
        "all_done": all(result.statuses == ("DONE", "DONE") for result in results),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
