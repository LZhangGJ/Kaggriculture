"""Evaluate generated counter policies against a target with both-seat blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean

from kaggriculture_lab.fast_env import run_duel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--seed-start", type=int, default=63_000)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    summaries = []
    for entry in manifest["variants"]:
        results = run_duel(
            entry["path"],
            args.target,
            range(args.seed_start, args.seed_start + args.seeds),
            both_seats=True,
            workers=args.workers,
        )
        margins = [
            float(result.rewards[0]) - float(result.rewards[1]) for result in results
        ]
        paired = [fmean(margins[index : index + 2]) for index in range(0, len(margins), 2)]
        wins = sum(margin > 0 for margin in margins)
        ties = sum(margin == 0 for margin in margins)
        summary = {
            **entry,
            "games": len(margins),
            "wins": wins,
            "ties": ties,
            "losses": len(margins) - wins - ties,
            "score_rate": (wins + 0.5 * ties) / len(margins),
            "mean_margin": fmean(margins),
            "paired_wins": sum(margin > 0 for margin in paired),
            "paired_ties": sum(margin == 0 for margin in paired),
            "paired_losses": sum(margin < 0 for margin in paired),
            "paired_mean_margin": fmean(paired),
            "all_done": all(result.statuses == ("DONE", "DONE") for result in results),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    report = {
        "manifest": str(args.manifest),
        "target": args.target,
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "results": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
