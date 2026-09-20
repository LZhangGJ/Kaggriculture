#!/usr/bin/env python3
"""Merge rectangular opponent-column shards from paired route searches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    routes = sources[0]["routes"]
    families = sources[0]["families"]
    for source in sources[1:]:
        if source["routes"] != routes or source["families"] != families:
            raise ValueError("payoff shards have different own-route rows")
    opponent_routes = []
    opponent_families = []
    scores = []
    margins = []
    seen = set()
    for source in sources:
        local_routes = source.get("opponent_routes", source["routes"])
        local_families = source.get("opponent_families", source["families"])
        matrix = np.asarray(source["score_matrix"], dtype=np.float64)
        margin = np.asarray(source["margin_matrix"], dtype=np.float64)
        for column, (route, family) in enumerate(zip(local_routes, local_families)):
            if family in seen:
                continue
            seen.add(family)
            opponent_routes.append(route)
            opponent_families.append(family)
            scores.append(matrix[:, column])
            margins.append(margin[:, column])
    payload = {
        "schema_version": 1,
        "engine": "merged paired FastEnv shards",
        "routes": routes,
        "families": families,
        "opponent_routes": opponent_routes,
        "opponent_families": opponent_families,
        "score_matrix": np.stack(scores, axis=1).tolist(),
        "margin_matrix": np.stack(margins, axis=1).tolist(),
        "sources": [str(path) for path in args.inputs],
    }
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output), "rows": len(routes), "columns": len(opponent_routes)
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
