#!/usr/bin/env python3
"""Test whether shallow public-state rules predict replay macro-plan clusters."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_text

from analyze_macro_route_library import CATEGORIES, _clusters, _snapshot


CHECKPOINTS = tuple(range(24, 289, 24))
MODES = {
    "opponent_farm": ("opp_",),
    "opponent_and_shops": ("opp_", "shop_"),
    "shops_only": ("shop_",),
    "own_only": ("own_",),
    "simple_public": ("opp_", "own_", "shop_"),
    "all_public": ("opp_", "own_", "shop_", "market_"),
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--macro-cache", type=Path, required=True)
    parser.add_argument("--selector-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--min-team-samples", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--min-leaf", type=int, default=3)
    return parser.parse_args()


def _farm_features(prefix: str, farm: dict[str, Any]) -> dict[str, float]:
    counts, _ = _snapshot(farm)
    result = {f"{prefix}_{name.lower()}": float(counts[index]) for index, name in enumerate(CATEGORIES)}
    result[f"{prefix}_land"] = float(counts[-2])
    result[f"{prefix}_hands"] = float(counts[-1])
    result[f"{prefix}_money"] = float(farm.get("money", 0.0) or 0.0)
    return result


def _extract(task: tuple[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    import orjson

    replay_path, targets = task
    replay = orjson.loads(Path(replay_path).read_bytes())
    steps = replay.get("steps", [])
    names = list((replay.get("info") or {}).get("TeamNames") or [])
    output = []
    for target in targets:
        player = int(target["player_index"])
        values: dict[int, dict[str, float]] = {}
        for checkpoint in CHECKPOINTS:
            observation = steps[min(checkpoint, len(steps) - 1)][player].get("observation") or {}
            farms = list(observation.get("farms", []) or [])
            own = farms[player] if player < len(farms) else {}
            opponent_index = 1 - player
            opponent = farms[opponent_index] if opponent_index < len(farms) else {}
            features = _farm_features("own", own)
            features.update(_farm_features("opp", opponent))
            town = observation.get("town") or {}
            for shop in town.get("unlocked_shops", []) or []:
                features[f"shop_{str(shop).lower()}"] = 1.0
            market = observation.get("market") or {}
            for product, value in (market.get("inventory") or {}).items():
                features[f"market_inventory_{str(product).lower()}"] = float(value)
            for product, value in (market.get("prices") or {}).items():
                features[f"market_price_{str(product).lower()}"] = float(value)
            values[checkpoint] = features
        output.append({
            "episode_id": int(target["episode_id"]),
            "player_index": player,
            "team": str(target["team"]),
            "opponent": names[1 - player] if len(names) == 2 else "",
            "features": values,
        })
    return output


def _loo_majority(labels: np.ndarray) -> float:
    correct = 0
    for index, label in enumerate(labels):
        train = np.delete(labels, index)
        values, counts = np.unique(train, return_counts=True)
        prediction = values[np.argmax(counts)]
        correct += int(prediction == label)
    return correct / len(labels)


def _evaluate(
    rows: list[dict[str, Any]], labels: np.ndarray, checkpoint: int,
    prefixes: tuple[str, ...], max_depth: int, min_leaf: int,
) -> dict[str, Any]:
    names = sorted({name for row in rows for name in row["features"][checkpoint] if name.startswith(prefixes)})
    string_labels = np.asarray([f"plan_{int(label)}" for label in labels])
    if not names:
        predictions = []
        for index in range(len(string_labels)):
            train = np.delete(string_labels, index)
            values, counts = np.unique(train, return_counts=True)
            predictions.append(values[np.argmax(counts)])
        predictions = np.asarray(predictions)
        return {
            "accuracy": float(np.mean(predictions == string_labels)),
            "balanced_accuracy": float(balanced_accuracy_score(string_labels, predictions)),
            "top_features": [],
            "rules": "no features at this checkpoint",
        }
    matrix = np.asarray([[row["features"][checkpoint].get(name, 0.0) for name in names] for row in rows])
    # Do not use class_weight="balanced" with leave-one-out validation. Its
    # fold-specific weights can reveal the held-out class through class counts.
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=0)
    predictions = cross_val_predict(model, matrix, string_labels, cv=LeaveOneOut(), n_jobs=1)
    model.fit(matrix, string_labels)
    importances = sorted(
        ((names[index], float(value)) for index, value in enumerate(model.feature_importances_) if value > 0),
        key=lambda item: -item[1],
    )
    return {
        "accuracy": float(np.mean(predictions == string_labels)),
        "balanced_accuracy": float(balanced_accuracy_score(string_labels, predictions)),
        "top_features": importances[:8],
        "rules": export_text(model, feature_names=names, decimals=1),
    }


def main() -> None:
    args = _args()
    warnings.filterwarnings("ignore", message="The number of unique classes is greater than 50%")
    with np.load(args.macro_cache, allow_pickle=True) as cached:
        macro_rows = list(cached["rows"])
    teams: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in macro_rows:
        teams[str(row["team"])].append(row)

    route_label: dict[tuple[int, int, str], int] = {}
    team_labels: dict[str, np.ndarray] = {}
    eligible = {}
    for team, rows in teams.items():
        if len(rows) < args.min_team_samples:
            continue
        labels, _ = _clusters(rows, args.threshold)
        if len(set(labels.tolist())) < 2:
            continue
        team_labels[team] = labels
        eligible[team] = rows
        for row, label in zip(rows, labels):
            route_label[(int(row["episode_id"]), int(row["player_index"]), team)] = int(label)

    if args.selector_cache.exists():
        with np.load(args.selector_cache, allow_pickle=True) as cached:
            selector_rows = list(cached["rows"])
        print(f"loaded {len(selector_rows)} cached selector rows", flush=True)
    else:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        tasks = []
        for replay in manifest.get("replays", []):
            if replay.get("error") is not None:
                continue
            targets = []
            for target in replay.get("targets", []) or []:
                key = (int(replay["episode_id"]), int(target["player_index"]), str(target["team_name"]))
                if key in route_label:
                    targets.append({
                        "episode_id": key[0], "player_index": key[1], "team": key[2],
                    })
            if targets:
                tasks.append((str((args.manifest.parent / replay["replay"]).resolve()), targets))
        selector_rows = []
        workers = min(max(1, args.workers), len(tasks))
        with mp.get_context("spawn").Pool(workers) as pool:
            for result in pool.imap_unordered(_extract, tasks, chunksize=1):
                selector_rows.extend(result)
                if len(selector_rows) % 100 < len(result):
                    print(f"extracted {len(selector_rows)} selector rows", flush=True)
        args.selector_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.selector_cache, rows=np.asarray(selector_rows, dtype=object))
        print(f"cached {len(selector_rows)} selector rows", flush=True)

    selector_by_key = {
        (int(row["episode_id"]), int(row["player_index"]), str(row["team"])): row
        for row in selector_rows
    }
    reports = []
    for team, macro_team_rows in sorted(eligible.items()):
        labels = team_labels[team]
        rows = [selector_by_key[(int(row["episode_id"]), int(row["player_index"]), team)] for row in macro_team_rows]
        baseline = _loo_majority(labels)
        trials = []
        for checkpoint in CHECKPOINTS:
            for mode, prefixes in MODES.items():
                result = _evaluate(rows, labels, checkpoint, prefixes, args.max_depth, args.min_leaf)
                trials.append({"checkpoint": checkpoint, "mode": mode, **result})
        best = max(trials, key=lambda row: (row["accuracy"] - baseline, row["balanced_accuracy"], -row["checkpoint"]))
        early_candidates = [
            row for row in trials
            if row["checkpoint"] <= 168 and row["mode"] in {"opponent_farm", "opponent_and_shops", "shops_only"}
        ]
        early_best = max(
            early_candidates,
            key=lambda row: (row["accuracy"] - baseline, row["balanced_accuracy"], -row["checkpoint"]),
        )
        reports.append({
            "team": team,
            "samples": len(rows),
            "plans": int(len(set(labels.tolist()))),
            "plan_sizes": sorted(np.unique(labels, return_counts=True)[1].tolist(), reverse=True),
            "leave_one_out_majority_accuracy": baseline,
            "best": {**best, "gain": float(best["accuracy"] - baseline)},
            "early_public_proxy": {
                **early_best, "gain": float(early_best["accuracy"] - baseline),
                "caveat": "checkpoint may still be after the actual route fork",
            },
            "trials": trials,
        })
        print(
            f"{team}: {len(rows)} samples/{len(set(labels.tolist()))} plans, "
            f"baseline={baseline:.3f}, best={best['accuracy']:.3f} "
            f"at {best['checkpoint']} {best['mode']}", flush=True,
        )
    report = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "macro_cache": str(args.macro_cache),
        "macro_threshold": args.threshold,
        "model": {
            "type": "DecisionTreeClassifier", "max_depth": args.max_depth,
            "min_leaf": args.min_leaf, "class_weight": None,
        },
        "validation": "leave-one-replay-out; best checkpoint is exploratory and selection-biased",
        "eligible_teams": len(reports),
        "teams_report": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
