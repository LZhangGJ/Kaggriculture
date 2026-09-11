from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

from arena_workflow import game


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def key(row: dict) -> tuple:
    return row["opponent"], row["seed"], row["opponent_seat"]


def summary(rows: list[dict]) -> dict:
    good = [row for row in rows if not row.get("runtime_error")]
    wins = sum(bool(row["win"]) for row in good)
    ties = sum(bool(row["tie"]) for row in good)
    metric_names = [
        "captures", "fast_paths", "certificate_failures", "mismatches",
        "witness_used", "chain_repairs", "restored", "unresolved",
        "verify_calls", "repair_trials", "accounted", "deadline_misses", "deadline_days",
        "output_overrides_detected", "output_restores", "output_restore_rejected",
    ]
    workflow = {name: sum(row.get("workflow", {}).get(name, 0) for row in good) for name in metric_names}
    return {
        "games": len(rows),
        "wins": wins,
        "losses": len(good) - wins - ties,
        "ties": ties,
        "errors": len(rows) - len(good),
        "win_rate": wins / len(good) if good else None,
        "mean_cash": statistics.mean(row["own_cash"] for row in good) if good else None,
        "mean_margin": statistics.mean(row["margin"] for row in good) if good else None,
        "median_margin": statistics.median(row["margin"] for row in good) if good else None,
        "max_latency_seconds": max((row["latency_max"] for row in good), default=None),
        "mean_game_seconds": statistics.mean(row["seconds"] for row in good) if good else None,
        "workflow": workflow,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=2610100000)
    parser.add_argument("--seed-count", type=int, default=50)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--opponents", nargs="*")
    args = parser.parse_args()
    pool = [item["id"] for item in json.loads((args.package / "POOL.json").read_text())]
    opponents = args.opponents or pool
    unknown = set(opponents) - set(pool)
    if unknown:
        raise SystemExit(f"unknown opponents: {sorted(unknown)}")
    seeds = range(args.seed_start, args.seed_start + args.seed_count)
    jobs = [(str(args.package), opponent, seed, seat, str(args.binary)) for seed in seeds for opponent in opponents for seat in (0, 1)]
    args.out.mkdir(parents=True, exist_ok=True)
    rows_file = args.out / "rows.json"
    partial_file = args.out / "rows.partial.json"
    rows = json.loads(partial_file.read_text()) if partial_file.exists() and not rows_file.exists() else []
    done = {key(row) for row in rows}
    jobs = [job for job in jobs if (job[1], job[2], job[3]) not in done]
    started = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as executor:
        for row in executor.map(game, jobs, chunksize=1):
            if row.get("runtime_error"):
                save(args.out / "FAILED.json", row)
                raise RuntimeError(row["runtime_error"])
            rows.append(row)
            if len(rows) % 50 == 0:
                save(partial_file, rows)
                print(json.dumps({"done": len(rows), "summary": summary(rows)}, ensure_ascii=False), flush=True)
    rows.sort(key=key)
    save(rows_file, rows)
    result = {
        "summary": summary(rows),
        "by_opponent": {opponent: summary([row for row in rows if row["opponent"] == opponent]) for opponent in opponents},
        "seconds_this_run": time.perf_counter() - started,
    }
    save(args.out / "RESULTS.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
