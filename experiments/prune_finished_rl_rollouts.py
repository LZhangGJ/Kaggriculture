#!/usr/bin/env python3
"""Delete old, completed PPO rollouts; keep checkpoints and metrics."""

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "work/student-v1"
NAME = re.compile(r"v3-ppo-native-job-(triad|economic-v1)-v([0-9]+)-1536g\.rollout\.npz")


def prune(directory: Path, *, family: str = "triad", min_round: int,
          keep: int, delete: bool) -> dict:
    completed = []
    for path in directory.glob(f"v3-ppo-native-job-{family}-v*-1536g.rollout.npz"):
        match = NAME.fullmatch(path.name)
        if not match or match[1] != family or path.is_symlink() or not path.is_file():
            continue
        round_number = int(match[2])
        stem = path.name.removesuffix(".rollout.npz")
        metrics = directory / f"{stem}.metrics.json"
        checkpoint = directory / f"{stem}.pt"
        weights = directory / f"native-{family}-v{round_number}-1536g.bin"
        if not all(item.is_file() and item.stat().st_size > 0
                   for item in (metrics, checkpoint, weights)):
            continue
        try:
            report = json.loads(metrics.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (report.get("status") == "PASS" and report.get("games") == 1536
                and report.get("illegal") == report.get("fallbacks") == 0):
            completed.append((round_number, path))
    if not completed:
        return {"family": family, "latest": None, "files": 0, "bytes": 0,
                "deleted": delete}
    latest = max(round_number for round_number, _ in completed)
    targets = [path for round_number, path in completed
               if min_round <= round_number <= latest - keep]
    size = sum(path.stat().st_size for path in targets)
    if delete:
        for path in targets:
            path.unlink()
    return {"family": family, "latest": latest, "files": len(targets), "bytes": size,
            "deleted": delete}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT)
    parser.add_argument("--family", choices=("triad", "economic-v1"),
                        default="triad")
    parser.add_argument("--min-round", type=int, default=211)
    parser.add_argument("--keep", type=int, default=8)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()
    if args.keep < 1 or args.min_round < 1:
        parser.error("keep and min-round must be positive")
    print(json.dumps(prune(args.directory, family=args.family, min_round=args.min_round,
                           keep=args.keep, delete=args.delete)))


if __name__ == "__main__":
    main()
