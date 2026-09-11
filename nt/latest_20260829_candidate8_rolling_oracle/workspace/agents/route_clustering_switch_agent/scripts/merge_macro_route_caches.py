#!/usr/bin/env python3
"""Merge macro-route feature caches with stable replay-side de-duplication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    merged: list[dict] = []
    seen: set[tuple[int, int]] = set()
    sources = []
    for path in args.input:
        with np.load(path, allow_pickle=True) as cached:
            rows = list(cached["rows"])
        added = duplicates = 0
        for row in rows:
            key = (int(row["episode_id"]), int(row["player_index"]))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            merged.append(row)
            added += 1
        sources.append({
            "path": str(path), "rows": len(rows), "added": added,
            "duplicates": duplicates,
        })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, rows=np.asarray(merged, dtype=object))
    payload = {
        "schema_version": 1,
        "sources": sources,
        "unique_replay_sides": len(merged),
        "episode_min": min(int(row["episode_id"]) for row in merged),
        "episode_max": max(int(row["episode_id"]) for row in merged),
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
