"""Run a reproducible both-seat round robin for a frozen opponent pool."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from kaggriculture_lab.fast_env import _duel_worker


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-manifest", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=51_000)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.pool_manifest.read_text(encoding="utf-8"))
    agents = [(entry["name"], entry["path"]) for entry in manifest["agents"]]
    metadata = []
    tasks = []
    for left_index, (left_name, left_path) in enumerate(agents):
        for right_name, right_path in agents[left_index + 1 :]:
            for seed in range(args.seed_start, args.seed_start + args.seeds):
                for seat in (0, 1):
                    metadata.append((left_name, right_name, seed, seat))
                    tasks.append((left_path, right_path, seed, seat, {}))

    if args.workers <= 1:
        results = [_duel_worker(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(
                executor.map(
                    _duel_worker,
                    tasks,
                    chunksize=max(1, len(tasks) // (args.workers * 4)),
                )
            )

    totals = defaultdict(lambda: {"games": 0, "wins": 0, "ties": 0, "losses": 0, "margin": 0.0})
    for (left_name, right_name, _, _), result in zip(metadata, results, strict=True):
        left_reward, right_reward = map(float, result.rewards)
        margin = left_reward - right_reward
        totals[left_name]["games"] += 1
        totals[right_name]["games"] += 1
        totals[left_name]["margin"] += margin
        totals[right_name]["margin"] -= margin
        if margin > 0:
            totals[left_name]["wins"] += 1
            totals[right_name]["losses"] += 1
        elif margin < 0:
            totals[left_name]["losses"] += 1
            totals[right_name]["wins"] += 1
        else:
            totals[left_name]["ties"] += 1
            totals[right_name]["ties"] += 1

    ranking = []
    for name, values in totals.items():
        values["score_rate"] = (values["wins"] + 0.5 * values["ties"]) / values["games"]
        values["mean_margin"] = values.pop("margin") / values["games"]
        ranking.append({"name": name, **values})
    ranking.sort(key=lambda row: (row["score_rate"], row["mean_margin"]), reverse=True)
    report = {
        "pool_manifest": str(args.pool_manifest),
        "seed_start": args.seed_start,
        "seeds": args.seeds,
        "games": len(results),
        "all_done": all(result.statuses == ("DONE", "DONE") for result in results),
        "ranking": ranking,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
