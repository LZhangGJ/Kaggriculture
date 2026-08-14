"""Evaluate one script agent against every member of a frozen pool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from kaggriculture_lab.fast_env import run_duel


def _summary(margins: list[float], rewards: list[float], opponents: list[float]) -> dict[str, float]:
    wins = sum(margin > 0 for margin in margins)
    ties = sum(margin == 0 for margin in margins)
    return {
        "games": len(margins),
        "wins": wins,
        "ties": ties,
        "losses": len(margins) - wins - ties,
        "score_rate": (wins + 0.5 * ties) / len(margins),
        "mean_reward": fmean(rewards),
        "mean_opponent_reward": fmean(opponents),
        "mean_margin": fmean(margins),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", required=True)
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=64_000)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    results_by_opponent = {}
    all_margins: list[float] = []
    all_rewards: list[float] = []
    all_opponents: list[float] = []
    all_done = True
    for entry in manifest["agents"]:
        results = run_duel(
            args.agent,
            entry["path"],
            range(args.seed_start, args.seed_start + args.seeds),
            both_seats=True,
            workers=args.workers,
        )
        rewards = [float(result.rewards[0]) for result in results]
        opponents = [float(result.rewards[1]) for result in results]
        margins = [left - right for left, right in zip(rewards, opponents, strict=True)]
        paired = [fmean(margins[index : index + 2]) for index in range(0, len(margins), 2)]
        summary = {
            **_summary(margins, rewards, opponents),
            "paired_wins": sum(margin > 0 for margin in paired),
            "paired_ties": sum(margin == 0 for margin in paired),
            "paired_losses": sum(margin < 0 for margin in paired),
            "paired_mean_margin": fmean(paired),
        }
        results_by_opponent[entry["name"]] = summary
        all_margins.extend(margins)
        all_rewards.extend(rewards)
        all_opponents.extend(opponents)
        all_done &= all(result.statuses == ("DONE", "DONE") for result in results)
        print(json.dumps({"opponent": entry["name"], **summary}, sort_keys=True), flush=True)

    report = {
        "agent": args.agent,
        "pool_manifest": str(args.pool_manifest),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "all_done": all_done,
        "overall": _summary(all_margins, all_rewards, all_opponents),
        "opponents": results_by_opponent,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"overall": report["overall"], "saved": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
