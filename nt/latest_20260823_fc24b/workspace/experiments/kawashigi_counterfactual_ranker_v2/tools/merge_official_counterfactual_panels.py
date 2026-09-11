#!/usr/bin/env python3
"""Merge disjoint official same-state counterfactual panels for the next round."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panels", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = [resolve(path) for path in args.panels]
    panels = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not all(panel.get("all_done") and panel.get("same_state_pass") for panel in panels):
        raise ValueError("all panels must pass completion and same-state gates")
    feature_names = panels[0]["feature_names"]
    if any(panel["feature_names"] != feature_names for panel in panels[1:]):
        raise ValueError("feature definitions differ")
    rows = [row for panel in panels for row in panel["rows"]]
    context_keys = {
        (str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        for row in rows
    }
    row_keys = {
        (str(row["candidate"]), str(row["opponent"]), int(row["seed"]), int(row["candidate_seat"]))
        for row in rows
    }
    if len(row_keys) != len(rows):
        raise ValueError("panels overlap")
    result = {
        "schema": "kawashigi-official-same-state-counterfactual-panel-merged-v1",
        "official_package_version_required": "1.32.7",
        "all_done": True,
        "same_state_pass": True,
        "decision_step": 145,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "game_count": len(rows),
        "context_count": len(context_keys),
        "seat_swapped": True,
        "sources": [{"path": str(path), "sha256": sha256(path), "contexts": panel["context_count"], "games": panel["game_count"]} for path, panel in zip(paths, panels)],
        "candidate_files": panels[0]["candidate_files"],
        "opponent_files": panels[0]["opponent_files"],
        "rows": rows,
    }
    output = resolve(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("all_done", "same_state_pass", "game_count", "context_count", "feature_count", "sources")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
