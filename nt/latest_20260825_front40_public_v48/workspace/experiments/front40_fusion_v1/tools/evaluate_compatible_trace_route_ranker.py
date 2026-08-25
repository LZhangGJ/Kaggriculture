"""Evaluate a frozen compatible-route ranker on independent panels."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from compatible_trace_route_ranker import (
    apply_conservative_switch,
    build_candidate_dataset,
    oracle_rows,
    selection_metrics,
)
from train_compatible_trace_route_ranker import compose_score, predict_components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--matrices", type=Path, nargs="+", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    route_ids = np.asarray(artifact["route_ids"], dtype=np.int32)
    dataset = build_candidate_dataset(
        args.contexts,
        args.matrices,
        Path(artifact["route_bank"]),
        Path(artifact["route_tree"]),
        route_ids,
        int(artifact["decision_step"]),
        int(artifact["prefix_start"]),
    )
    if dataset.feature_names != artifact["feature_names"]:
        raise RuntimeError("runtime feature schema differs from trained model")
    margin, win, rank = predict_components(artifact, dataset)
    score = compose_score(
        margin,
        win,
        rank,
        float(artifact["win_probability_weight"]),
        float(artifact["rank_weight"]),
        float(artifact["score_scale"]),
        float(artifact["rank_scale"]),
    )
    selected = apply_conservative_switch(
        score, dataset, float(artifact["switch_threshold"])
    )
    changed = selected != dataset.legacy_row
    payload = {
        "schema": "kaggriculture.front40_fusion.compatible-trace-route-ranker-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "runtime_boundary": artifact["runtime_boundary"],
        "contexts": [str(path) for path in args.contexts],
        "matrices": [str(path) for path in args.matrices],
        "model": str(args.model),
        "games": int(dataset.group_sizes.size),
        "mean_candidates": float(np.mean(dataset.group_sizes)),
        "switches": int(np.sum(changed)),
        "switch_rate": float(np.mean(changed)),
        "legacy": selection_metrics(dataset, dataset.legacy_row),
        "selected": selection_metrics(dataset, selected),
        "oracle": selection_metrics(dataset, oracle_rows(dataset)),
        "selected_route_counts": {
            str(int(route)): int(count)
            for route, count in zip(
                *np.unique(dataset.route_id[selected], return_counts=True), strict=True
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "switches": payload["switches"], "legacy": payload["legacy"], "selected": payload["selected"], "oracle": payload["oracle"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

