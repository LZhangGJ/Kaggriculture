from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--opponent", default="soil_v219g")
    parser.add_argument("--seed", type=int, default=2610100000)
    parser.add_argument("--games", type=int, default=2)
    args = parser.parse_args()
    sys.path.insert(0, str(args.package))
    from arena import game

    rows = []
    for offset in range(args.games):
        seed = args.seed + offset // 2
        seat = offset % 2
        job = (args.opponent, seed, seat, str(args.baseline), "none", {}, 719, None)
        baseline = game(job)
        job = (args.opponent, seed, seat, str(args.candidate), "none", {}, 719, None)
        candidate = game(job)
        rows.append(
            {
                "seed": seed,
                "seat": seat,
                "baseline_error": baseline.get("runtime_error"),
                "candidate_error": candidate.get("runtime_error"),
                "baseline_cash": baseline.get("own_cash"),
                "candidate_cash": candidate.get("own_cash"),
                "baseline_margin": baseline.get("margin"),
                "candidate_margin": candidate.get("margin"),
                "same_actions": baseline.get("action_hash") == candidate.get("action_hash"),
                "candidate_latency": candidate.get("latency_max"),
            }
        )
    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
