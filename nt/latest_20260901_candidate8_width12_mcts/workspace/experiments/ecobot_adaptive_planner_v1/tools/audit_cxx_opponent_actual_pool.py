#!/usr/bin/env python3
"""Measure deployable R6 strength against every clean native C++ route."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from audit_candidate8_multifuture_oracle import load_genome
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--actions", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--genomes", type=Path, required=True)
    parser.add_argument("--genome-index", type=int, default=0)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    clean = [
        row for row in metadata["opponent_routes"]
        if row.get("selected", False)
        and not row.get("oracle_only", False)
        and row.get("source_execution_hard_failures") == 0
    ]
    bundle = NativeTeammateBundle(args.source, args.actions, args.metadata)
    genome = load_genome(args.genomes, args.genome_index)
    tasks = [
        (str(row["family"]), args.seed_start + offset, seat)
        for row in clean
        for offset in range(args.seed_count)
        for seat in (0, 1)
    ]

    def play(task: tuple[str, int, int]) -> dict:
        family, seed, seat = task
        result = bundle.adaptive_executor.play(
            genome, bundle.index(family), seed, seat, False
        )
        own = float(result["rewards"][seat])
        rival = float(result["rewards"][1 - seat])
        return {
            "family": family,
            "own": own,
            "rival": rival,
            "margin": own - rival,
            "crop": int(result["avoidable_crop_losses"]),
            "animal": int(result["avoidable_animal_losses"]),
            "overflow": int(result["end_overflow"]),
        }

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(executor.map(play, tasks, chunksize=1))
    elapsed = time.perf_counter() - started

    by_family: dict[str, list[dict]] = {str(row["family"]): [] for row in clean}
    for result in results:
        by_family[result["family"]].append(result)

    rows = []
    for family, games in by_family.items():
        wins = sum(row["margin"] > 0 for row in games)
        ties = sum(row["margin"] == 0 for row in games)
        losses = len(games) - wins - ties
        rows.append({
            "opponent": family,
            "games": len(games),
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "win_rate": wins / len(games),
            "score_rate": (wins + 0.5 * ties) / len(games),
            "mean_margin": statistics.fmean(row["margin"] for row in games),
            "mean_cash": statistics.fmean(row["own"] for row in games),
            "avoidable_crop_losses": sum(row["crop"] for row in games),
            "avoidable_animal_losses": sum(row["animal"] for row in games),
            "end_overflow": sum(row["overflow"] for row in games),
        })

    total_games = len(results)
    total_wins = sum(row["wins"] for row in rows)
    total_ties = sum(row["ties"] for row in rows)
    payload = {
        "schema": "kaggriculture.cxx-clean-opponent-actual-pool.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "seed_start": args.seed_start,
            "seed_count": args.seed_count,
            "seats": [0, 1],
            "workers": args.workers,
            "opponents": len(rows),
        },
        "summary": {
            "games": total_games,
            "wins": total_wins,
            "ties": total_ties,
            "losses": total_games - total_wins - total_ties,
            "win_rate": total_wins / total_games,
            "score_rate": (total_wins + 0.5 * total_ties) / total_games,
            "opponents_at_least_90pct": sum(row["win_rate"] >= 0.90 for row in rows),
            "opponents_at_least_50pct": sum(row["win_rate"] >= 0.50 for row in rows),
            "opponents_zero_wins": sum(row["wins"] == 0 for row in rows),
            "seconds": elapsed,
            "games_per_second": total_games / elapsed,
            "avoidable_crop_losses": sum(row["avoidable_crop_losses"] for row in rows),
            "avoidable_animal_losses": sum(row["avoidable_animal_losses"] for row in rows),
            "end_overflow": sum(row["end_overflow"] for row in rows),
        },
        "opponents": sorted(rows, key=lambda row: (row["win_rate"], row["mean_margin"], row["opponent"])),
        "inputs": {
            "source": {"path": str(args.source), "sha256": sha256(args.source)},
            "actions": {"path": str(args.actions), "sha256": sha256(args.actions)},
            "metadata": {"path": str(args.metadata), "sha256": sha256(args.metadata)},
            "genomes": {"path": str(args.genomes), "sha256": sha256(args.genomes)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
