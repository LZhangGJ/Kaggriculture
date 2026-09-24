#!/usr/bin/env python3
"""Warm A/B with handoff-state-fixed future shop/weed resampling."""

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from collections import defaultdict
from pathlib import Path

from run_strong_ab import BOTS, entry, play


def summarize(rows, bots):
    cells = defaultdict(dict)
    for row in rows:
        if not row["error"]:
            key = row["bot"], row["seed"], row["seat"], row["future_seed"]
            cells[key][row["label"]] = row
    paired = []
    for bot in bots:
        prefixes = defaultdict(list)
        for (name, seed, seat, future_seed), arms in cells.items():
            if name == bot and set(arms) == {"baseline", "candidate"}:
                base, candidate = arms["baseline"], arms["candidate"]
                prefixes[seed, seat].append((
                    int(candidate["margin"] > 0) - int(base["margin"] > 0),
                    candidate["margin"] - base["margin"],
                ))
        expected = [(sum(win for win, _ in values) / len(values),
                     sum(margin for _, margin in values) / len(values))
                    for values in prefixes.values()]
        paired.append({
            "bot": bot, "prefixes": len(expected),
            "seed_blocks": len({seed for seed, _ in prefixes}),
            "equivalent_win_delta": sum(win for win, _ in expected),
            "mean_margin_delta": (sum(margin for _, margin in expected) / len(expected)
                                  if expected else None),
        })
    return paired


def validate(rows, future_seeds):
    groups = defaultdict(list)
    for row in rows:
        groups[row["label"], row["bot"], row["seed"], row["seat"]].append(row)
    for key, values in groups.items():
        seeds = [row["future_seed"] for row in values]
        if len(seeds) != len(future_seeds) or set(seeds) != set(future_seeds) or any(row["error"] for row in values):
            raise ValueError(f"incomplete future replicas: {key}")
        hashes = {row["handoff_hash"] for row in values}
        if None in hashes or len(hashes) != 1:
            raise ValueError(f"future replicas changed the warm prefix: {key}")
    for label, bot, seed, seat in list(groups):
        if label == "baseline":
            left = groups[label, bot, seed, seat][0]["handoff_hash"]
            right = groups["candidate", bot, seed, seat][0]["handoff_hash"]
            if left != right:
                raise ValueError(f"A/B handoff mismatch: {(bot, seed, seat)}")
    arms = defaultdict(list)
    for row in rows:
        arms[row["bot"], row["seed"], row["seat"], row["future_seed"]].append(row["label"])
    if any(sorted(labels) != ["baseline", "candidate"] for labels in arms.values()):
        raise ValueError("each prefix/future seed requires exactly one A/B pair")


def self_check():
    rows = []
    for label in ("baseline", "candidate"):
        for future_seed, margin in ((10, -1), (11, 1)):
            rows.append({"label": label, "bot": "bot", "seed": 1, "seat": 0,
                         "future_seed": future_seed, "handoff_hash": "same", "error": None,
                         "margin": margin + 2 * (label == "candidate")})
    validate(rows, {10, 11})
    result = summarize(rows, ["bot"])[0]
    assert result["prefixes"] == 1 and result["equivalent_win_delta"] == .5
    print(json.dumps({"status": "PASS"}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=entry)
    parser.add_argument("--candidate", type=entry)
    parser.add_argument("--opponents", default="thomas_2945,melon_2749,demand_preserving,ahmed_v47,pipe8")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--start", type=int)
    parser.add_argument("--future-replicas", type=int, default=4)
    parser.add_argument("--future-start", type=int)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if None in (args.baseline, args.candidate, args.output, args.start, args.future_start):
        parser.error("--baseline, --candidate, --output, --start and --future-start are required")
    if min(args.seeds, args.future_replicas, args.workers) < 1 or args.output.exists():
        parser.error("positive sizes are required and --output must not exist")
    trajectory_dir = args.output.with_name(f"{args.output.stem}-trajectories").resolve()
    if trajectory_dir.exists():
        parser.error("trajectory output directory must not exist")
    names = [name for name in args.opponents.split(",") if name]
    if not names or len(names) != len(set(names)) or any(name not in BOTS for name in names):
        parser.error("invalid --opponents")
    tasks = [
        (label, str(path), None, "0", bot, BOTS[bot], seed, seat, "fast",
         str(trajectory_dir), future_seed)
        for label, path in (("baseline", args.baseline), ("candidate", args.candidate))
        for bot in names for seed in range(args.start, args.start + args.seeds)
        for seat in (0, 1)
        for future_seed in range(args.future_start, args.future_start + args.future_replicas)
    ]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn"),
                                max_tasks_per_child=1) as pool:
        rows = list(pool.map(play, tasks))
    future_seeds = set(range(args.future_start, args.future_start + args.future_replicas))
    validate(rows, future_seeds)
    result = {
        "engine": "fast_kaggriculture:FastEnv.reseed_future",
        "seed_range": [args.start, args.start + args.seeds],
        "future_seed_range": [args.future_start, args.future_start + args.future_replicas],
        "both_seats": True, "policies": {"baseline": str(args.baseline), "candidate": str(args.candidate)},
        "trajectory_format": "kaggriculture-bc-v1", "trajectory_dir": str(trajectory_dir),
        "diagnostic": True, "opponents": {name: BOTS[name] for name in names},
        "paired": summarize(rows, names), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "games": len(rows), "paired": result["paired"]}))


if __name__ == "__main__":
    main()
