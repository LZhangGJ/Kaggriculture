"""Train a public-state ranker over causally compatible day-six routes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np

from compatible_trace_route_ranker import (
    CandidateDataset,
    apply_conservative_switch,
    build_candidate_dataset,
    oracle_rows,
    selection_metrics,
)


def relevance_labels(dataset: CandidateDataset) -> np.ndarray:
    labels = np.zeros(dataset.margin.size, dtype=np.int32)
    for start, end in zip(dataset.group_offsets[:-1], dataset.group_offsets[1:], strict=True):
        values = dataset.margin[start:end]
        order = np.argsort(values)
        labels[start:end][values > 0] = 2
        if values.size >= 2:
            labels[start + order[-2]] = max(labels[start + order[-2]], 3)
        labels[start + order[-1]] = 5
    return labels


def multi_group_rows(dataset: CandidateDataset) -> tuple[np.ndarray, np.ndarray]:
    blocks = []
    groups = []
    for start, end in zip(dataset.group_offsets[:-1], dataset.group_offsets[1:], strict=True):
        if end - start > 1:
            blocks.append(np.arange(start, end, dtype=np.int64))
            groups.append(int(end - start))
    if not blocks:
        raise RuntimeError("no multi-candidate ranking groups")
    return np.concatenate(blocks), np.asarray(groups, dtype=np.int32)


def predict_components(artifact: dict, dataset: CandidateDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    margin = artifact["margin_model"].predict(dataset.features)
    win = artifact["win_model"].predict_proba(dataset.features)[:, 1]
    rank = artifact["rank_model"].predict(dataset.features)
    return margin, win, rank


def compose_score(
    margin: np.ndarray,
    win: np.ndarray,
    rank: np.ndarray,
    win_weight: float,
    rank_weight: float,
    scale: float,
    rank_scale: float,
) -> np.ndarray:
    return margin + win_weight * scale * win + rank_weight * scale * rank / rank_scale


def robust_key(metrics: dict[str, object]) -> tuple[float, float, float]:
    panel_rates = [float(row["win_rate"]) for row in metrics["by_panel"]]
    return min(panel_rates), float(metrics["win_rate"]), float(metrics["mean_margin"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--train-matrices", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-contexts", type=Path, nargs="+", required=True)
    parser.add_argument("--validation-matrices", type=Path, nargs="+", required=True)
    parser.add_argument("--route-bank", type=Path, required=True)
    parser.add_argument("--route-tree", type=Path, required=True)
    parser.add_argument("--route-ids", required=True)
    parser.add_argument("--decision-step", type=int, default=144)
    parser.add_argument("--prefix-start", type=int, default=72)
    parser.add_argument("--threads", type=int, default=18)
    parser.add_argument("--output-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    route_ids = np.asarray([int(value) for value in args.route_ids.split(",") if value], dtype=np.int32)
    train = build_candidate_dataset(
        args.train_contexts,
        args.train_matrices,
        args.route_bank,
        args.route_tree,
        route_ids,
        args.decision_step,
        args.prefix_start,
    )
    validation = build_candidate_dataset(
        args.validation_contexts,
        args.validation_matrices,
        args.route_bank,
        args.route_tree,
        route_ids,
        args.decision_step,
        args.prefix_start,
    )
    if train.feature_names != validation.feature_names:
        raise RuntimeError("train and validation feature schemas differ")

    margin_model = lgb.LGBMRegressor(
        objective="huber",
        n_estimators=700,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        random_state=20260824,
        n_jobs=args.threads,
        verbosity=-1,
    )
    margin_model.fit(
        train.features,
        train.margin,
        eval_set=[(validation.features, validation.margin)],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    win_model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=80,
        subsample=0.9,
        colsample_bytree=0.75,
        reg_lambda=8.0,
        random_state=20260825,
        n_jobs=args.threads,
        verbosity=-1,
    )
    win_model.fit(
        train.features,
        (train.margin > 0).astype(np.int8),
        eval_set=[(validation.features, (validation.margin > 0).astype(np.int8))],
        callbacks=[lgb.early_stopping(60, verbose=False)],
    )
    train_rank_rows, train_rank_groups = multi_group_rows(train)
    valid_rank_rows, valid_rank_groups = multi_group_rows(validation)
    train_relevance = relevance_labels(train)
    valid_relevance = relevance_labels(validation)
    rank_model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        eval_at=(1, 2, 3),
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=31,
        min_child_samples=30,
        colsample_bytree=0.8,
        reg_lambda=6.0,
        random_state=20260826,
        n_jobs=args.threads,
        verbosity=-1,
    )
    rank_model.fit(
        train.features[train_rank_rows],
        train_relevance[train_rank_rows],
        group=train_rank_groups,
        eval_set=[(validation.features[valid_rank_rows], valid_relevance[valid_rank_rows])],
        eval_group=[valid_rank_groups],
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    artifact = {
        "margin_model": margin_model,
        "win_model": win_model,
        "rank_model": rank_model,
    }
    train_margin, train_win, train_rank = predict_components(artifact, train)
    valid_margin, valid_win, valid_rank = predict_components(artifact, validation)
    scale = max(float(np.std(train_margin)), 1.0)
    rank_scale = max(float(np.std(train_rank)), 1e-6)

    search_rows = []
    best = None
    for win_weight in (0.0, 0.5, 1.0, 2.0, 4.0):
        for rank_weight in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0):
            valid_score = compose_score(
                valid_margin, valid_win, valid_rank,
                win_weight, rank_weight, scale, rank_scale,
            )
            raw_best = []
            raw_gain = []
            for group, (start, end) in enumerate(
                zip(validation.group_offsets[:-1], validation.group_offsets[1:], strict=True)
            ):
                row = int(start + np.argmax(valid_score[start:end]))
                raw_best.append(row)
                raw_gain.append(float(valid_score[row] - valid_score[validation.legacy_row[group]]))
            positive = np.asarray([value for value in raw_gain if value > 0], dtype=np.float32)
            thresholds = [0.0]
            if positive.size:
                thresholds.extend(float(np.quantile(positive, q)) for q in (0.25, 0.5, 0.75, 0.9, 0.95))
            thresholds.append(float("inf"))
            for threshold in sorted(set(thresholds)):
                selected = apply_conservative_switch(valid_score, validation, threshold)
                metrics = selection_metrics(validation, selected)
                row = {
                    "win_weight": win_weight,
                    "rank_weight": rank_weight,
                    "switch_threshold": threshold,
                    "validation": metrics,
                }
                search_rows.append(row)
                key = robust_key(metrics)
                if best is None or key > best[0]:
                    best = (key, win_weight, rank_weight, threshold, valid_score, selected)
    assert best is not None
    _, win_weight, rank_weight, threshold, valid_score, valid_selected = best
    train_score = compose_score(
        train_margin, train_win, train_rank,
        win_weight, rank_weight, scale, rank_scale,
    )
    train_selected = apply_conservative_switch(train_score, train, threshold)
    artifact.update(
        win_probability_weight=win_weight,
        rank_weight=rank_weight,
        score_scale=scale,
        rank_scale=rank_scale,
        switch_threshold=threshold,
        decision_step=args.decision_step,
        prefix_start=args.prefix_start,
        feature_names=train.feature_names,
        route_ids=route_ids,
        route_bank=str(args.route_bank),
        route_tree=str(args.route_tree),
        runtime_boundary=(
            "current public state plus own private inventory and frozen route descriptors; "
            "terminal outcomes are labels only"
        ),
    )
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_model)
    payload = {
        "schema": "kaggriculture.front40_fusion.compatible-trace-route-ranker.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TRAINING_ONLY",
        "runtime_boundary": artifact["runtime_boundary"],
        "route_bank": str(args.route_bank),
        "route_tree": str(args.route_tree),
        "route_ids": route_ids.tolist(),
        "decision_step": args.decision_step,
        "prefix_start": args.prefix_start,
        "features": train.features.shape[1],
        "training": {
            "contexts": int(train.group_sizes.size),
            "rows": int(train.features.shape[0]),
            "mean_candidates": float(np.mean(train.group_sizes)),
            "legacy": selection_metrics(train, train.legacy_row),
            "selected": selection_metrics(train, train_selected),
            "oracle": selection_metrics(train, oracle_rows(train)),
        },
        "validation": {
            "contexts": int(validation.group_sizes.size),
            "rows": int(validation.features.shape[0]),
            "mean_candidates": float(np.mean(validation.group_sizes)),
            "legacy": selection_metrics(validation, validation.legacy_row),
            "selected": selection_metrics(validation, valid_selected),
            "oracle": selection_metrics(validation, oracle_rows(validation)),
        },
        "selected_hyperparameters": {
            "win_probability_weight": win_weight,
            "rank_weight": rank_weight,
            "switch_threshold": threshold,
        },
        "search": search_rows,
        "model": str(args.output_model),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "training": payload["training"], "validation": payload["validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

