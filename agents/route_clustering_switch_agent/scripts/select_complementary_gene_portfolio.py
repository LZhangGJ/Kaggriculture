#!/usr/bin/env python3
"""Select a compact mutation portfolio by incremental seed coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument(
        "--minimum-start", type=int, default=0,
        help="Reject candidates containing phase genes that begin before this step.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [
        row for row in payload["ranking"]
        if all(int(gene.get("start", args.minimum_start)) >= args.minimum_start
               for gene in (row.get("genes") or []))
    ]
    if not rows:
        raise ValueError("no candidates satisfy --minimum-start")
    scores = np.asarray([row["paired_seed_scores"] for row in rows], dtype=np.float64)
    margins = np.asarray([row["paired_seed_margins"] for row in rows], dtype=np.float64)
    if scores.ndim != 2 or margins.shape != scores.shape:
        raise ValueError("ranking rows must contain aligned paired seed outcomes")

    selected: list[int] = []
    remaining = set(range(len(rows)))
    covered_scores = np.zeros(scores.shape[1], dtype=np.float64)
    covered_margins = np.full(margins.shape[1], -np.inf, dtype=np.float64)
    trace = []
    for position in range(min(args.size, len(rows))):
        def contribution(index: int) -> tuple[float, float, float, float]:
            next_scores = np.maximum(covered_scores, scores[index])
            next_margins = np.maximum(covered_margins, margins[index])
            return (
                float(np.mean(next_scores)),
                float(np.mean(next_margins)),
                float(rows[index]["mean_score"]),
                float(rows[index]["mean_margin"]),
            )

        chosen = max(remaining, key=contribution)
        previous_score = float(np.mean(covered_scores))
        selected.append(chosen)
        remaining.remove(chosen)
        covered_scores = np.maximum(covered_scores, scores[chosen])
        covered_margins = np.maximum(covered_margins, margins[chosen])
        trace.append({
            "position": position + 1,
            "source_rank": rows[chosen].get("rank"),
            "source_index": chosen,
            "fixed_score": rows[chosen]["mean_score"],
            "portfolio_oracle_score": float(np.mean(covered_scores)),
            "incremental_score": float(np.mean(covered_scores)) - previous_score,
            "portfolio_oracle_margin": float(np.mean(covered_margins)),
        })

    ranking = []
    for position, index in enumerate(selected, 1):
        row = dict(rows[index])
        row["source_rank"] = row.get("rank")
        row["rank"] = position
        row["portfolio_trace"] = trace[position - 1]
        ranking.append(row)
    result = {
        "schema": "complementary-gene-portfolio-v1",
        "source": str(args.input.resolve()),
        "seeds": payload.get("seeds"),
        "requested_size": args.size,
        "minimum_start": args.minimum_start,
        "size": len(ranking),
        "oracle_score": float(np.mean(covered_scores)),
        "oracle_margin": float(np.mean(covered_margins)),
        "selection_trace": trace,
        "ranking": ranking,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "size": len(ranking),
        "oracle_score": result["oracle_score"],
        "oracle_margin": result["oracle_margin"],
        "trace": trace,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
