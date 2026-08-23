#!/usr/bin/env python3
"""Split one route-counterfactual receipt into seed-disjoint child receipts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--first-count", type=int, required=True)
    parser.add_argument("--output-a", type=Path, required=True)
    parser.add_argument("--output-b", type=Path, required=True)
    args = parser.parse_args()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    seeds = sorted({int(row["seed"]) for row in source["decision_features"]})
    if not 0 < args.first_count < len(seeds):
        raise ValueError("first count must split the seed set")

    def child(selected: list[int], label: str) -> dict:
        selected_set = set(selected)
        payload = deepcopy(source)
        payload["schema"] = source["schema"] + ".split.v1"
        payload["split_source"] = str(args.input.resolve())
        payload["split_label"] = label
        payload["seed_start"] = selected[0]
        payload["seed_count"] = len(selected)
        payload["seed_batch_size"] = len(selected)
        payload["seed_batches"] = 1
        payload["decision_features"] = [
            row
            for row in source["decision_features"]
            if int(row["seed"]) in selected_set
        ]
        matrices = []
        rows = []
        for original in source["rows"]:
            row = deepcopy(original)
            row["per_game"] = [
                game
                for game in original["per_game"]
                if int(game["seed"]) in selected_set
            ]
            margins = np.asarray(
                [int(game["margin"]) for game in row["per_game"]], dtype=np.int64
            )
            matrices.append(margins)
            row.update(
                games=int(len(margins)),
                wins=int(np.sum(margins > 0)),
                ties=int(np.sum(margins == 0)),
                losses=int(np.sum(margins < 0)),
                score_rate=float(
                    np.mean(margins > 0) + 0.5 * np.mean(margins == 0)
                ),
                mean_margin=float(np.mean(margins)),
            )
            row.pop("mean_candidate_cash", None)
            row.pop("mean_opponent_cash", None)
            rows.append(row)
        payload["rows"] = rows
        oracle = np.max(np.stack(matrices), axis=0)
        payload["oracle"] = {
            "wins": int(np.sum(oracle > 0)),
            "ties": int(np.sum(oracle == 0)),
            "losses": int(np.sum(oracle < 0)),
            "score_rate": float(
                np.mean(oracle > 0) + 0.5 * np.mean(oracle == 0)
            ),
            "mean_margin": float(np.mean(oracle)),
        }
        return payload

    outputs = (
        (args.output_a, child(seeds[: args.first_count], "A")),
        (args.output_b, child(seeds[args.first_count :], "B")),
    )
    for path, payload in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": "PASS",
                "outputs": [
                    {
                        "path": str(path.resolve()),
                        "seed_count": payload["seed_count"],
                        "games": payload["rows"][0]["games"],
                        "oracle": payload["oracle"],
                    }
                    for path, payload in outputs
                ],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
