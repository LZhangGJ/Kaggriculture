#!/usr/bin/env python3
"""Reuse oracle suffixes for horizon scoring; run only new diagnostic candidates."""
import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
from pathlib import Path

from run_r1_candidate_oracle import cases_from_trace, run

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("bot", "seed", "seat", "day")


def ranks(values):
    ordered = sorted(values)
    return [sum(i + 1 for i, item in enumerate(ordered) if item == value) /
            sum(item == value for item in ordered) for value in values]


def correlation(left, right):
    left, right = ranks(left), ranks(right)
    a, b = sum(left) / len(left), sum(right) / len(right)
    x = sum((u - a) * (v - b) for u, v in zip(left, right))
    y = sum((u - a) ** 2 for u in left) * sum((v - b) ** 2 for v in right)
    return x / y ** .5 if y else None


def metrics(rows, diagnostics):
    result = {}
    for day in (11, 15):
        groups = []
        keys = {tuple(row[name] for name in KEYS) for row in rows if row["day"] == day}
        for key in keys:
            values = [row for row in rows if tuple(row[name] for name in KEYS) == key and
                      (diagnostics or not row["candidate"]["diagnostic"])]
            groups.append(values)
        for horizon in range(5):
            matches = 0
            regrets = []
            correlations = []
            for values in groups:
                predicted = max(values, key=lambda row: row["candidate"]["scores_horizon"][horizon])
                actual = max(values, key=lambda row: row["margin"])
                matches += predicted is actual
                regrets.append(actual["margin"] - predicted["margin"])
                value = correlation([row["candidate"]["scores_horizon"][horizon] for row in values],
                                    [row["margin"] for row in values])
                if value is not None:
                    correlations.append(value)
            result[f"day{day}_h{horizon + 1}"] = {"groups": len(groups), "top_matches": matches,
                                                   "mean_regret": sum(regrets) / len(regrets),
                                                   "mean_spearman": sum(correlations) / len(correlations)}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", type=Path, default=ROOT / "work/r1-candidate-oracle-thomas-melon-day11-15.json")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    old = json.loads(args.oracle.read_text())
    cases = cases_from_trace(Path(old["trace"]))
    discovery_tasks = [(bot, seed, seat, day, None) for bot, seed, seat, _ in cases for day in (11, 15)]
    context = mp.get_context("spawn")
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, max_tasks_per_child=1) as pool:
        discovered = list(pool.map(run, discovery_tasks))
    metadata = {(row["bot"], row["seed"], row["seat"], row["day"], proposal["index"]): proposal
                for row in discovered for proposal in row["proposals"]}
    rows = old["rows"]
    for row in rows:
        row["candidate"] = metadata[tuple(row[name] for name in KEYS) + (row["candidate"]["index"],)]
    tasks = [(row["bot"], row["seed"], row["seat"], row["day"], proposal["index"])
             for row in discovered for proposal in row["proposals"] if proposal["diagnostic"]]
    with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=context, max_tasks_per_child=1) as pool:
        diagnostic_rows = list(pool.map(run, tasks))
    labels = {(bot, seed, seat): label for bot, seed, seat, label in cases}
    for row in diagnostic_rows:
        row["source_outcome"] = labels[(row["bot"], row["seed"], row["seat"])]
    rows = rows + diagnostic_rows
    result = {"base_oracle": str(args.oracle), "discovered": discovered,
              "diagnostic_games": len(diagnostic_rows),
              "errors": sum(bool(row["error"]) for row in diagnostic_rows),
              "base_metrics": metrics(rows, False), "expanded_metrics": metrics(rows, True),
              "rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "discovered")}, indent=2))


if __name__ == "__main__":
    main()
