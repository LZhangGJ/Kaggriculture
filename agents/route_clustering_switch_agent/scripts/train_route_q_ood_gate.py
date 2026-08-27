#!/usr/bin/env python3
"""Train an observable-state gate and combine route-Q with the robust fallback."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier


def _special_mode(value: str) -> tuple[str, str, int, tuple[str, ...]]:
    schedule, separator, opponents = value.partition("=")
    parts = schedule.split(":")
    if not separator or len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError(
            "special mode must be LABEL:TARGET:CHECKPOINT=OPPONENT[,OPPONENT...]"
        )
    names = tuple(part for part in opponents.split(",") if part)
    if not names:
        raise argparse.ArgumentTypeError("special mode needs at least one opponent")
    return parts[0], parts[1], int(parts[2]), names


def _current(path: Path, checkpoint: int) -> tuple[np.ndarray, list[str]]:
    with np.load(path, allow_pickle=False) as saved:
        names = saved["feature_names"].astype(str).tolist()
        matrix = np.ascontiguousarray(
            saved[f"features_{checkpoint}"].reshape(-1, len(names)), dtype=np.float32
        )
    return matrix, names


def _public(path: Path) -> tuple[np.ndarray, list[dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["rows"] if row.get("error") is None]
    return np.asarray([row["features"] for row in rows], dtype=np.float32), rows


def _tree_payload(model: DecisionTreeClassifier) -> dict:
    tree = model.tree_
    raw_value = np.asarray(tree.value)
    if raw_value.shape[1:] == (1, len(model.classes_)):
        value = raw_value[:, 0, :]
    elif raw_value.shape[1:] == (len(model.classes_), 1):
        value = raw_value[:, :, 0]
    else:
        raise ValueError(f"unexpected classifier tree value shape: {raw_value.shape}")
    return {
        "class_kind": "route_q_ood_gate",
        "classes": [str(value) for value in model.classes_],
        "left": tree.children_left.astype(int).tolist(),
        "right": tree.children_right.astype(int).tolist(),
        "feature": tree.feature.astype(int).tolist(),
        "threshold": tree.threshold.astype(float).tolist(),
        "value": value.astype(float).tolist(),
    }


def _metrics(
    model: DecisionTreeClassifier, current: np.ndarray, public: np.ndarray,
    rows: list[dict], label_by_opponent: dict[str, str],
) -> dict:
    current_predictions = model.predict(current)
    public_predictions = model.predict(public)
    by_opponent: dict[str, list[str]] = defaultdict(list)
    for row, prediction in zip(rows, public_predictions):
        by_opponent[str(row["opponent"])].append(str(prediction))
    expected = np.asarray([label_by_opponent.get(str(row["opponent"]), "other") for row in rows])
    other_mask = expected == "other"
    recall_by_label = {}
    for label in sorted(set(expected.tolist())):
        mask = expected == label
        recall_by_label[label] = float(np.mean(public_predictions[mask] == label))
    return {
        "current_activation_rate": float(np.mean(current_predictions == "current")),
        "hard_recall": (
            recall_by_label.get("hard")
        ),
        "other_recall": (
            float(np.mean(public_predictions[other_mask] == "other"))
            if np.any(other_mask) else None
        ),
        "public_label_accuracy": float(np.mean(public_predictions == expected)),
        "public_activation_rate": float(np.mean(public_predictions == "current")),
        "public_activations": int(np.count_nonzero(public_predictions == "current")),
        "public_samples": len(public_predictions),
        "public_prediction_counts_by_opponent": {
            key: dict(Counter(values)) for key, values in sorted(by_opponent.items())
        },
        "public_prediction_counts": dict(Counter(public_predictions.tolist())),
        "recall_by_label": recall_by_label,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--q-policy", type=Path, required=True)
    parser.add_argument("--fallback-policy", type=Path, required=True)
    parser.add_argument("--current-train", type=Path, required=True)
    parser.add_argument("--current-valid", type=Path, required=True)
    parser.add_argument("--public-train", type=Path, required=True)
    parser.add_argument("--public-valid", type=Path, required=True)
    parser.add_argument("--fit-seeds", required=True, help="Comma-separated public seeds used to fit.")
    parser.add_argument("--checkpoint", type=int, default=144)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--min-leaf", type=int, default=1)
    parser.add_argument("--public-weight", type=float, default=1.0)
    parser.add_argument("--hard-weight", type=float, default=1.0)
    parser.add_argument(
        "--hard-opponents", default="",
        help="Comma-separated public opponent names assigned to the hard mode.",
    )
    parser.add_argument("--hard-target")
    parser.add_argument("--hard-checkpoint", type=int, default=168)
    parser.add_argument(
        "--special-mode", type=_special_mode, action="append", default=[],
        help=(
            "Repeated LABEL:TARGET:CHECKPOINT=OPPONENT[,OPPONENT...] modes; "
            "visible-state labels are learned jointly with current/other."
        ),
    )
    parser.add_argument("--output-policy", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()

    current_train, feature_names = _current(args.current_train, args.checkpoint)
    current_valid, valid_names = _current(args.current_valid, args.checkpoint)
    if feature_names != valid_names:
        raise ValueError("current feature schemas differ")
    public_all, public_rows = _public(args.public_train)
    fit_seeds = {int(value) for value in args.fit_seeds.split(",") if value.strip()}
    fit_mask = np.asarray([int(row["seed"]) in fit_seeds for row in public_rows])
    public_fit = public_all[fit_mask]
    fit_rows = [row for row, keep in zip(public_rows, fit_mask) if keep]
    public_valid, public_valid_rows = _public(args.public_valid)
    hard_opponents = {
        value for value in args.hard_opponents.split(",") if value.strip()
    }
    modes: dict[str, dict] = {}
    if hard_opponents:
        if not args.hard_target:
            parser.error("--hard-opponents requires --hard-target")
        modes["hard"] = {
            "target": args.hard_target,
            "checkpoint": args.hard_checkpoint,
            "opponents": hard_opponents,
        }
    for label, target, checkpoint, opponents in args.special_mode:
        if label in {"current", "other", "q", "fallback"}:
            parser.error(f"reserved special-mode label: {label}")
        if label in modes:
            parser.error(f"duplicate special-mode label: {label}")
        modes[label] = {
            "target": target,
            "checkpoint": checkpoint,
            "opponents": set(opponents),
        }
    label_by_opponent: dict[str, str] = {}
    for label, spec in modes.items():
        for opponent in spec["opponents"]:
            previous = label_by_opponent.setdefault(opponent, label)
            if previous != label:
                parser.error(
                    f"opponent {opponent} appears in both {previous} and {label}"
                )

    matrix = np.vstack((current_train, public_fit))
    public_fit_labels = [
        label_by_opponent.get(str(row["opponent"]), "other")
        for row in fit_rows
    ]
    labels = np.asarray(
        ["current"] * len(current_train) + public_fit_labels
    )
    class_weight = {"current": 1.0, "other": args.public_weight}
    for label in modes:
        class_weight[label] = args.hard_weight
    model = DecisionTreeClassifier(
        max_depth=args.max_depth,
        min_samples_leaf=args.min_leaf,
        class_weight=class_weight,
        random_state=20260827,
    )
    model.fit(matrix, labels)
    train_metrics = _metrics(
        model, current_train, public_fit, fit_rows, label_by_opponent
    )
    valid_metrics = _metrics(
        model, current_valid, public_valid, public_valid_rows, label_by_opponent
    )

    q_policy = json.loads(args.q_policy.read_text(encoding="utf-8"))
    fallback = json.loads(args.fallback_policy.read_text(encoding="utf-8"))
    q_policy["kind"] = (
        "multimode_gated_backward_counterfactual_route_q"
        if modes else "gated_backward_counterfactual_route_q"
    )
    q_policy["targets"] = list(dict.fromkeys(
        [
            *q_policy["targets"], *fallback.get("targets", ()),
            *(str(spec["target"]) for spec in modes.values()),
        ]
    ))
    q_policy["fallback_nodes"] = fallback["nodes"]
    if modes:
        q_policy["mode_nodes"] = {
            label: [{"selected": {
                "opening": str(q_policy["openings"][0]),
                "checkpoint": int(spec["checkpoint"]),
                "enabled": True,
                "tree": {
                    "class_kind": "family",
                    "classes": [str(spec["target"])],
                    "left": [-1], "right": [-1], "feature": [-2],
                    "threshold": [-2.0], "value": [[1.0]],
                },
            }}]
            for label, spec in modes.items()
        }
    q_policy["gate"] = {
        "schema": "observable-route-q-multimode-gate-v2",
        "checkpoint": args.checkpoint,
        "positive_label": "current",
        "mode_by_label": {
            "current": "q", "other": "fallback",
            **{label: label for label in modes},
        },
        "feature_names": feature_names,
        "tree": _tree_payload(model),
        "train_metrics": train_metrics,
        "validation_metrics": valid_metrics,
    }
    report = {
        "schema": "route-q-multimode-gate-training-report-v2",
        "checkpoint": args.checkpoint,
        "fit_seeds": sorted(fit_seeds),
        "model": {
            "max_depth": args.max_depth,
            "min_leaf": args.min_leaf,
            "public_weight": args.public_weight,
            "hard_weight": args.hard_weight,
            "nodes": int(model.tree_.node_count),
        },
        "hard_opponents": sorted(hard_opponents),
        "hard_schedule": (
            {"checkpoint": args.hard_checkpoint, "target": args.hard_target}
            if args.hard_target else None
        ),
        "special_modes": {
            label: {
                "target": str(spec["target"]),
                "checkpoint": int(spec["checkpoint"]),
                "opponents": sorted(spec["opponents"]),
            }
            for label, spec in modes.items()
        },
        "train": train_metrics,
        "validation": valid_metrics,
        "used_features": [
            feature_names[index]
            for index in sorted(set(int(value) for value in model.tree_.feature if value >= 0))
        ],
        "output_policy": str(args.output_policy.resolve()),
    }
    args.output_policy.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_policy.write_text(
        json.dumps(q_policy, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    args.output_report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
