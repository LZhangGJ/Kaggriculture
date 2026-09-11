#!/usr/bin/env python3
"""Official-1.32.7 paired smoke test for FC12G versus frozen FC2B."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import statistics
import time


ROOT = Path(__file__).resolve().parents[3]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _run_one(task: tuple[str, str, str, int, int]) -> dict:
    variant, candidate_path, opponent_path, seed, candidate_seat = task
    from kaggle_environments import make

    agents = [candidate_path, opponent_path]
    if candidate_seat == 1:
        agents.reverse()
    started = time.perf_counter()
    environment = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    environment.run(agents)
    elapsed = time.perf_counter() - started
    final = environment.steps[-1]
    candidate_reward = int(final[candidate_seat].reward)
    opponent_reward = int(final[1 - candidate_seat].reward)
    return {
        "variant": variant,
        "seed": seed,
        "candidate_seat": candidate_seat,
        "frames": len(environment.steps),
        "statuses": [str(row.status) for row in final],
        "candidate_cash": candidate_reward,
        "opponent_cash": opponent_reward,
        "margin": candidate_reward - opponent_reward,
        "result": "win" if candidate_reward > opponent_reward else "tie" if candidate_reward == opponent_reward else "loss",
        "elapsed_seconds": elapsed,
    }


def _summarize(rows: list[dict]) -> dict:
    return {
        "games": len(rows),
        "wins": sum(row["result"] == "win" for row in rows),
        "ties": sum(row["result"] == "tie" for row in rows),
        "losses": sum(row["result"] == "loss" for row in rows),
        "mean_margin": statistics.fmean(row["margin"] for row in rows),
        "mean_elapsed_seconds": statistics.fmean(row["elapsed_seconds"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "experiments/fusion_champion_v1/artifacts/fc12g_cpu/main.py",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=ROOT / "submission/55663355_fc2a_shadow_prt_fix/main.py",
    )
    parser.add_argument(
        "--opponent",
        type=Path,
        default=ROOT / "submission/55655402_rayk_k320_adaptive_rank1/main.py",
    )
    parser.add_argument("--seed-list", default="1811165014")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    seeds = [int(value.strip()) for value in args.seed_list.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seed list must be non-empty and unique")
    for path in (args.candidate, args.source, args.opponent):
        if not path.is_file():
            raise FileNotFoundError(path)

    tasks = [
        (variant, str(path), str(args.opponent), seed, seat)
        for variant, path in (("fc2b_source", args.source), ("fc12g_guard", args.candidate))
        for seed in seeds
        for seat in (0, 1)
    ]
    with ProcessPoolExecutor(max_workers=min(max(args.workers, 1), len(tasks))) as pool:
        rows = list(pool.map(_run_one, tasks))

    grouped = {
        variant: [row for row in rows if row["variant"] == variant]
        for variant in ("fc2b_source", "fc12g_guard")
    }
    source_map = {(row["seed"], row["candidate_seat"]): row for row in grouped["fc2b_source"]}
    candidate_map = {(row["seed"], row["candidate_seat"]): row for row in grouped["fc12g_guard"]}
    paired = []
    for key in sorted(source_map):
        source = source_map[key]
        candidate = candidate_map[key]
        paired.append(
            {
                "seed": key[0],
                "candidate_seat": key[1],
                "source_result": source["result"],
                "candidate_result": candidate["result"],
                "source_margin": source["margin"],
                "candidate_margin": candidate["margin"],
                "margin_delta": candidate["margin"] - source["margin"],
            }
        )

    all_done = all(row["frames"] == 720 and row["statuses"] == ["DONE", "DONE"] for row in rows)
    payload = {
        "schema": "kaggriculture.fusion_champion.fc12g-official-smoke.v1",
        "status": "PASS" if all_done else "FAIL",
        "official_package_version": version("kaggle-environments"),
        "seed_values": seeds,
        "seat_protocol": "same seeds with seats swapped",
        "all_done": all_done,
        "inputs": {
            "candidate": {"path": str(args.candidate), "sha256": _sha256(args.candidate)},
            "source": {"path": str(args.source), "sha256": _sha256(args.source)},
            "opponent": {"path": str(args.opponent), "sha256": _sha256(args.opponent)},
        },
        "summary": {variant: _summarize(group) for variant, group in grouped.items()},
        "source_losses_rescued": sum(row["source_result"] == "loss" and row["candidate_result"] == "win" for row in paired),
        "source_wins_harmed": sum(row["source_result"] == "win" and row["candidate_result"] != "win" for row in paired),
        "paired": paired,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in ("rows", "paired")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
