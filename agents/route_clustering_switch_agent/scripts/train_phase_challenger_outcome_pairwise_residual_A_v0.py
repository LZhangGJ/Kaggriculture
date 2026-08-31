#!/usr/bin/env python3
"""A-only candidate-vs-KEEP outcome representation sanity screen."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from typing import Any, Mapping

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor

import run_phase_challenger_compact_et_ab_v1 as compact_v1
import train_phase_challenger_outcome_multiclass_A_nested_v1 as outcome_v1
import train_phase_challenger_residual_paired_seat_two_head_v0 as paired_v0


SCHEMA = "phase-challenger-outcome-pairwise-residual-A-v0"
REGRESSION, NEUTRAL, WIN_GAIN = (
    outcome_v1.REGRESSION, outcome_v1.NEUTRAL, outcome_v1.WIN_GAIN,
)
FORMAL_SEEDS, SEED_FOLDS = outcome_v1.FORMAL_SEEDS, outcome_v1.SEED_FOLDS
GROUP_KEY = paired_v0.GROUP_KEY
SHARED_PREFIX_STOP = 907
DERIVABLE_CONTEXT_NAMES = (
    "runtime_actor_count", "source_actor_count", "segment", "segment_offset",
)


def _donor_provenance(name: str) -> bool:
    return (
        name == "donor_support_log"
        or name.startswith("donor_rank_")
        or name.startswith("donor_cluster_")
    )


def feature_schema() -> dict[str, Any]:
    names = tuple(compact_v1.COMPACT_FEATURE_NAMES)
    donor = tuple(i for i, name in enumerate(names) if _donor_provenance(name))
    derivable = tuple(names.index(name) for name in DERIVABLE_CONTEXT_NAMES)
    seat = names.index("seat")
    shared = (*range(SHARED_PREFIX_STOP), seat)
    excluded = (*donor, *derivable)
    relative = tuple(
        i for i in range(SHARED_PREFIX_STOP, len(names))
        if i != seat and i not in excluded
    )
    if (
        len(names) != 1078 or len(donor) != 12
        or len(shared) != 908 or len(relative) != 154
    ):
        raise AssertionError("pre-registered compact pair schema changed")
    parts = (set(shared), set(relative), set(excluded))
    if any(parts[a] & parts[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise AssertionError("pair feature partitions overlap")
    if set.union(*parts) != set(range(len(names))):
        raise AssertionError("pair feature partitions are incomplete")
    pair_names = tuple(
        [*(names[i] for i in shared)]
        + [f"candidate_minus_KEEP_{names[i]}" for i in relative]
    )
    forbidden = ("outcome", "margin", "reward", "target", "label")
    if any(token in name.lower() for name in pair_names for token in forbidden):
        raise AssertionError("label-like field entered pair X")
    return {
        "source_width": len(names), "shared_indices": shared,
        "relative_indices": relative, "excluded_indices": excluded,
        "excluded_donor_provenance_names": [names[i] for i in donor],
        "excluded_derivable_context_names": list(DERIVABLE_CONTEXT_NAMES),
        "pair_feature_names": pair_names, "pair_width": len(pair_names),
        "shared_width": len(shared), "relative_action_width": len(relative),
        "absolute_context": "KEEP[0:907] plus absolute seat",
        "relative_context": "candidate minus KEEP causal action fields",
        "donor_provenance_is_feature": False,
        "labels_outcome_margin_are_features": False,
    }


FEATURE_SCHEMA = feature_schema()
SHARED_INDICES = np.asarray(FEATURE_SCHEMA["shared_indices"], np.int64)
RELATIVE_INDICES = np.asarray(FEATURE_SCHEMA["relative_indices"], np.int64)
PAIR_WIDTH = int(FEATURE_SCHEMA["pair_width"])
PAIR_FEATURE_NAMES = tuple(FEATURE_SCHEMA["pair_feature_names"])
SEAT_FEATURE_INDEX = tuple(compact_v1.COMPACT_FEATURE_NAMES).index("seat")


def seed_fold_map() -> dict[int, int]:
    return outcome_v1.seed_fold_map()


def _subset(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[str, np.ndarray]:
    return {key: np.asarray(value)[rows] for key, value in arrays.items()}


def outcome_arrays(
    arrays: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    a, source = outcome_v1.outcome_arrays(arrays)
    for key in (GROUP_KEY, "seat", "step"):
        if key not in arrays or len(np.asarray(arrays[key])) != len(arrays["decision"]):
            raise ValueError(f"pairwise A requires aligned {key}")
        a[key] = np.asarray(arrays[key])[source]
    return a, source


def _fingerprint(*values: bytes) -> bytes:
    digest = hashlib.blake2b(digest_size=16)
    for value in values:
        digest.update(len(value).to_bytes(4, "little"))
        digest.update(value)
    return digest.digest()


def build_pair_features(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Build one candidate-vs-KEEP X row without reading target fields."""

    features = np.asarray(arrays["features"], np.float32)
    decision = np.asarray(arrays["decision"], np.int64)
    groups = np.asarray(arrays[GROUP_KEY])
    seat = np.asarray(arrays["seat"], np.int8)
    step = np.asarray(arrays["step"], np.int16)
    if features.shape != (len(decision), compact_v1.COMPACT_WIDTH):
        raise ValueError("pairwise input requires aligned compact1078 rows")
    if not np.array_equal(features[:, SEAT_FEATURE_INDEX], seat.astype(np.float32)):
        raise ValueError("compact absolute seat disagrees with metadata")
    slices = outcome_v1.base.decision_slices(decision)
    pair_count = sum(len(rows) - 1 for rows in slices)
    result = {
        "features": np.empty((pair_count, PAIR_WIDTH), np.float32),
        "decision": np.empty(pair_count, np.int64),
        "candidate_row": np.empty(pair_count, np.int64),
        "keep_row": np.empty(pair_count, np.int64),
        "opponent": np.empty(pair_count, np.int64),
        "seed": np.empty(pair_count, np.int64),
        "seat": np.empty(pair_count, np.int8),
        "step": np.empty(pair_count, np.int16),
        "pair_fingerprint": np.empty(pair_count, "S16"),
        "action_delta_fingerprint": np.empty(pair_count, "S16"),
        "action_delta_l1": np.empty(pair_count, np.float32),
    }
    edit = np.asarray(arrays["edit"])
    at = 0
    for rows in slices:
        keep = rows[edit[rows] == outcome_v1.base.KEEP_EDIT]
        if len(keep) != 1:
            raise ValueError("pairwise decision lost unique KEEP")
        keep_row = int(keep[0])
        shared = features[rows, :SHARED_PREFIX_STOP]
        if not np.array_equal(shared, np.broadcast_to(shared[:1], shared.shape)):
            raise ValueError("candidate rows changed shared online context")
        for candidate in rows[edit[rows] != outcome_v1.base.KEEP_EDIT]:
            relative = features[candidate, RELATIVE_INDICES] - features[keep_row, RELATIVE_INDICES]
            result["features"][at, :len(SHARED_INDICES)] = features[keep_row, SHARED_INDICES]
            result["features"][at, len(SHARED_INDICES):] = relative
            for key, value in (
                ("decision", decision[candidate]), ("candidate_row", candidate),
                ("keep_row", keep_row), ("opponent", arrays["opponent"][candidate]),
                ("seed", arrays["seed"][candidate]), ("seat", seat[candidate]),
                ("step", step[candidate]),
            ):
                result[key][at] = value
            raw_delta = np.asarray(relative, np.float32).tobytes()
            result["pair_fingerprint"][at] = _fingerprint(
                bytes(groups[keep_row]), bytes(groups[candidate]),
            )
            result["action_delta_fingerprint"][at] = _fingerprint(raw_delta)
            result["action_delta_l1"][at] = float(np.abs(relative).sum())
            at += 1
    if at != pair_count or pair_count == 0 or not np.isfinite(result["features"]).all():
        raise AssertionError("candidate-vs-KEEP materialization is incomplete")
    result["sample_weight"], _ = pair_weights(result)
    return result


def pair_weights(
    pairs: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Give both seats of each (seed, opponent, step) cell total mass one."""

    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row, values in enumerate(zip(
        pairs["seed"], pairs["opponent"], pairs["step"], strict=True,
    )):
        cells[tuple(map(int, values))].append(row)
    weights = np.zeros(len(pairs["decision"]), np.float64)
    per_cell, per_seat = [], []
    for cell, rows in cells.items():
        seats = np.asarray(pairs["seat"])[rows]
        counts = [int(np.count_nonzero(seats == value)) for value in (0, 1)]
        if set(map(int, seats)) != {0, 1} or counts[0] != counts[1]:
            raise ValueError(f"paired cell {cell} is incomplete or asymmetric")
        weights[rows] = 1.0 / len(rows)
        per_cell.append(len(rows))
        per_seat.append(counts[0])
    if any(not np.isclose(weights[rows].sum(), 1.0) for rows in cells.values()):
        raise AssertionError("paired cell weight does not sum to one")
    return weights, {
        "paired_seed_opponent_step_cells": len(cells), "pair_rows": len(weights),
        "total_weight": float(weights.sum()),
        "each_two_seat_cell_total_weight": 1.0,
        "all_cells_contain_both_seats": True,
        "equal_candidate_count_by_seat": True,
        "min_pairs_per_cell": min(per_cell), "max_pairs_per_cell": max(per_cell),
        "min_candidates_per_seat": min(per_seat),
        "max_candidates_per_seat": max(per_seat),
        "pair_rows_are_not_reported_as_independent_samples": True,
    }


def pair_rows_by_step(pairs: Mapping[str, np.ndarray]) -> dict[str, int]:
    counts = Counter(map(int, pairs["step"]))
    return {str(step): counts[step] for step in sorted(counts)}


def attach_pair_targets(
    arrays: Mapping[str, np.ndarray], pairs: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    candidate = np.asarray(pairs["candidate_row"], np.int64)
    return {
        **{key: np.asarray(value) for key, value in pairs.items()},
        "target_class": np.asarray(arrays["target_class"], np.int8)[candidate],
        "target_win_gain": np.asarray(arrays["target_win_gain"], np.int8)[candidate],
        "target_neutral_margin": np.asarray(arrays["delta_margin"], np.float64)[candidate],
    }


def model_params(trees: int, random_seed: int) -> dict[str, Any]:
    if trees < 2:
        raise ValueError("pairwise screen requires at least two trees")
    return {
        "n_estimators": int(trees), "max_depth": 6,
        "min_samples_leaf": 2, "max_features": 0.5,
        "bootstrap": False, "n_jobs": -1, "random_state": int(random_seed),
    }


def _class_counts(target: np.ndarray) -> dict[str, int]:
    counts = Counter(map(int, target))
    return {
        "regression": counts[REGRESSION], "neutral": counts[NEUTRAL],
        "win_gain": counts[WIN_GAIN],
    }


def fit_models(
    pairs: Mapping[str, np.ndarray], trees: int, random_seed: int,
) -> tuple[tuple[Any, Any], dict[str, Any]]:
    features = np.asarray(pairs["features"], np.float32)
    target = np.asarray(pairs["target_class"], np.int8)
    weights, weight_audit = pair_weights(pairs)
    classes = np.unique(target)
    if len(classes) == 1:
        outcome_model: Any = {"constant_class": int(classes[0])}
        outcome_kind = "constant_missing-class-safe"
    else:
        outcome_model = ExtraTreesClassifier(**model_params(trees, random_seed))
        outcome_model.fit(features, target, sample_weight=weights)
        outcome_kind = "shallow_nonlinear_ExtraTreesClassifier"
    neutral = np.flatnonzero(target == NEUTRAL)
    neutral_target = np.asarray(pairs["target_neutral_margin"], np.float64)
    if not len(neutral):
        margin_model: Any = {"constant": 0.0}
        margin_kind = "constant_no-neutral-safe"
    elif np.all(neutral_target[neutral] == neutral_target[neutral[0]]):
        margin_model = {"constant": float(neutral_target[neutral[0]])}
        margin_kind = "constant-neutral-margin"
    else:
        margin_model = ExtraTreesRegressor(**model_params(trees, random_seed + 1009))
        margin_model.fit(
            features[neutral], neutral_target[neutral], sample_weight=weights[neutral],
        )
        margin_kind = "neutral_only_shallow_ExtraTreesRegressor"
    return (outcome_model, margin_model), {
        "outcome_model_kind": outcome_kind, "margin_model_kind": margin_kind,
        "outcome_class_counts": _class_counts(target),
        "margin_training_rows": int(len(neutral)),
        "margin_training_target_class": "neutral_only",
        "weight_audit": weight_audit,
        "state_action_interaction": "shallow nonlinear tree splits",
    }


def predict_models(
    models: tuple[Any, Any], features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, np.float32)
    probability = np.zeros((len(values), 3), np.float32)
    outcome_model, margin_model = models
    if isinstance(outcome_model, dict):
        probability[:, int(outcome_model["constant_class"])] = 1.0
    else:
        raw = np.asarray(outcome_model.predict_proba(values), np.float32)
        probability[:, np.asarray(outcome_model.classes_, np.int64)] = raw
    if not np.allclose(probability.sum(axis=1), 1.0):
        raise RuntimeError("pair outcome probability alignment failed")
    margin = (
        np.full(len(values), float(margin_model["constant"]), np.float32)
        if isinstance(margin_model, dict)
        else np.asarray(margin_model.predict(values), np.float32)
    )
    return probability, margin


def group_oof_view(
    pairs: Mapping[str, np.ndarray], groups: np.ndarray, view: str,
    trees: int, random_seed: int,
) -> tuple[np.ndarray, np.ndarray, list[tuple[Any, Any]], list[dict[str, Any]]]:
    groups = np.asarray(groups)
    unique = tuple(sorted(set(map(int, groups))))
    if len(unique) < 2:
        raise ValueError(f"{view} OOF requires at least two groups")
    probability = np.full((len(groups), 3), np.nan, np.float32)
    margin = np.full(len(groups), np.nan, np.float32)
    models: list[tuple[Any, Any]] = []
    audit = []
    for offset, group in enumerate(unique):
        train = np.flatnonzero(groups != group)
        valid = np.flatnonzero(groups == group)
        model, model_audit = fit_models(
            _subset(pairs, train), trees, random_seed + offset * 1009,
        )
        probability[valid], margin[valid] = predict_models(
            model, np.asarray(pairs["features"])[valid],
        )
        models.append(model)
        audit.append({
            "view": view, "left_out_group": group,
            "train_seeds": sorted(set(map(int, np.asarray(pairs["seed"])[train]))),
            "valid_seeds": sorted(set(map(int, np.asarray(pairs["seed"])[valid]))),
            "train_valid_seed_overlap": sorted(
                set(map(int, np.asarray(pairs["seed"])[train]))
                & set(map(int, np.asarray(pairs["seed"])[valid]))
            ),
            "train_valid_decision_overlap": sorted(
                set(map(int, np.asarray(pairs["decision"])[train]))
                & set(map(int, np.asarray(pairs["decision"])[valid]))
            ),
            **model_audit,
        })
    if not np.isfinite(probability).all() or not np.isfinite(margin).all():
        raise RuntimeError(f"{view} pair OOF predictions are incomplete")
    return probability, margin, models, audit


def ensemble_mean(
    models: list[tuple[Any, Any]], features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = [predict_models(model, features) for model in models]
    if not predictions:
        raise ValueError("pairwise ensemble is empty")
    return (
        np.mean([value[0] for value in predictions], axis=0).astype(np.float32),
        np.mean([value[1] for value in predictions], axis=0).astype(np.float32),
    )


def candidate_scores(
    arrays: Mapping[str, np.ndarray], pairs: Mapping[str, np.ndarray],
    probability: np.ndarray, margin: np.ndarray,
) -> dict[str, np.ndarray]:
    candidate = np.asarray(pairs["candidate_row"], np.int64)
    if len(np.unique(candidate)) != len(candidate):
        raise ValueError("candidate-vs-KEEP mapping is not one-to-one")
    result = {
        "regression_probability": np.zeros(len(arrays["decision"]), np.float64),
        "win_gain_probability": np.zeros(len(arrays["decision"]), np.float64),
        "neutral_margin_prediction": np.zeros(len(arrays["decision"]), np.float64),
    }
    result["regression_probability"][candidate] = probability[:, REGRESSION]
    result["win_gain_probability"][candidate] = probability[:, WIN_GAIN]
    result["neutral_margin_prediction"][candidate] = margin
    return result


def select_choices(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
    params: Mapping[str, float],
) -> np.ndarray:
    """Margin is the final tie-break and can never create eligibility."""

    edit = np.asarray(arrays["edit"])
    risk = np.asarray(scores["regression_probability"], np.float64)
    gain = np.asarray(scores["win_gain_probability"], np.float64)
    margin = np.asarray(scores["neutral_margin_prediction"], np.float64)
    choices = []
    for rows in outcome_v1.base.decision_slices(np.asarray(arrays["decision"], np.int64)):
        keep = rows[edit[rows] == outcome_v1.base.KEEP_EDIT]
        chosen = int(keep[0])
        candidates = rows[edit[rows] != outcome_v1.base.KEEP_EDIT]
        eligible = candidates[
            (gain[candidates] > float(params["min_win_gain_probability"]))
            & (risk[candidates] <= float(params["max_regression_probability"]))
        ]
        if len(eligible):
            chosen = max(map(int, eligible), key=lambda row: (
                gain[row] - risk[row], gain[row], -risk[row], margin[row], -row,
            ))
        choices.append(chosen)
    return np.asarray(choices, np.int64)


def fire_attribution(
    arrays: Mapping[str, np.ndarray], choices: np.ndarray,
) -> dict[str, int]:
    result = {
        "win_gain_fires": 0, "neutral_positive_margin_fires": 0,
        "neutral_nonpositive_margin_fires": 0, "regression_fires": 0,
    }
    edit, outcome = np.asarray(arrays["edit"]), np.asarray(arrays["outcome"])
    for rows, choice in zip(
        outcome_v1.base.decision_slices(np.asarray(arrays["decision"])),
        np.asarray(choices), strict=True,
    ):
        if edit[choice] == outcome_v1.base.KEEP_EDIT:
            continue
        keep = int(rows[edit[rows] == outcome_v1.base.KEEP_EDIT][0])
        if outcome[choice] < outcome[keep]:
            result["regression_fires"] += 1
        elif outcome[choice] == 2 and outcome[keep] < 2:
            result["win_gain_fires"] += 1
        elif float(arrays["delta_margin"][choice]) > 0:
            result["neutral_positive_margin_fires"] += 1
        else:
            result["neutral_nonpositive_margin_fires"] += 1
    return result


def choice_metrics(arrays: Mapping[str, np.ndarray], choices: np.ndarray) -> dict[str, Any]:
    return {
        **outcome_v1.choice_metrics(arrays, choices),
        "fire_attribution": fire_attribution(arrays, choices),
    }


def _thresholds(values: np.ndarray, *, risk: bool) -> tuple[float, ...]:
    points = [0.0, 1.0]
    if len(values):
        points.extend(map(float, np.quantile(values, [0.25, 0.5, 0.75, 0.9])))
        if not risk:
            points.append(float(np.max(values)))
    return tuple(sorted(set(points)))


def calibrate(
    arrays: Mapping[str, np.ndarray], scores: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], np.ndarray]:
    candidate = np.asarray(arrays["edit"]) != outcome_v1.base.KEEP_EDIT
    trials: list[tuple[dict[str, Any], np.ndarray]] = []
    for risk in _thresholds(scores["regression_probability"][candidate], risk=True):
        for gain in _thresholds(scores["win_gain_probability"][candidate], risk=False):
            params = {
                "max_regression_probability": risk,
                "min_win_gain_probability": gain,
            }
            choices = select_choices(arrays, scores, params)
            trials.append(({**params, **choice_metrics(arrays, choices)}, choices))
    allowed = [item for item in trials if (
        item[0]["wins_lost"] == 0 and item[0]["outcome_regressions"] == 0
        and not any(item[0]["opponent_outcome_regressions"].values())
    )]
    chosen, choices = max(allowed, key=lambda item: (
        item[0]["raw_win_delta"], item[0]["wins_gained"],
        item[0]["outcome_delta_sum"], item[0]["same_outcome_margin_delta_sum"],
        -item[0]["fires"], -item[0]["max_regression_probability"],
        item[0]["min_win_gain_probability"],
    ))
    return {
        "max_regression_probability": chosen["max_regression_probability"],
        "min_win_gain_probability": chosen["min_win_gain_probability"],
        "outer_train_inner_seed_fold_OOF_metrics": chosen,
    }, choices


def _rank_metrics(
    arrays: Mapping[str, np.ndarray], score_by_row: np.ndarray,
) -> dict[str, Any]:
    headroom = top1 = 0
    reciprocal = 0.0
    edit, outcome = np.asarray(arrays["edit"]), np.asarray(arrays["outcome"])
    for rows in outcome_v1.base.decision_slices(np.asarray(arrays["decision"])):
        keep = int(rows[edit[rows] == outcome_v1.base.KEEP_EDIT][0])
        candidates = rows[edit[rows] != outcome_v1.base.KEEP_EDIT]
        winners = set(map(int, candidates[outcome[candidates] == 2]))
        if outcome[keep] == 2 or not winners:
            continue
        ranking = sorted(map(int, candidates), key=lambda row: (-float(score_by_row[row]), row))
        headroom += 1
        top1 += int(ranking[0] in winners)
        reciprocal += 1.0 / (1 + next(i for i, row in enumerate(ranking) if row in winners))
    return {
        "upgrade_decisions": headroom,
        "top1_win_recall": float(top1 / headroom) if headroom else 0.0,
        "mean_reciprocal_rank_first_win": float(reciprocal / headroom) if headroom else 0.0,
    }


def rank_diagnostics(
    train_pairs: Mapping[str, np.ndarray], valid: Mapping[str, np.ndarray],
    valid_pairs: Mapping[str, np.ndarray], valid_scores: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    primary = (
        np.asarray(valid_scores["win_gain_probability"])
        - np.asarray(valid_scores["regression_probability"])
    )
    frequency: dict[bytes, list[int]] = defaultdict(lambda: [0, 0])
    for fingerprint, target in zip(
        train_pairs["action_delta_fingerprint"], train_pairs["target_win_gain"],
        strict=True,
    ):
        counts = frequency[bytes(fingerprint)]
        counts[0] += int(target)
        counts[1] += 1
    frequency_score = np.zeros(len(valid["decision"]), np.float64)
    delta_score = np.zeros(len(valid["decision"]), np.float64)
    for pair_row, candidate in enumerate(valid_pairs["candidate_row"]):
        wins, total = frequency.get(
            bytes(valid_pairs["action_delta_fingerprint"][pair_row]), [0, 0],
        )
        frequency_score[int(candidate)] = wins / total if total else 0.0
        delta_score[int(candidate)] = -float(valid_pairs["action_delta_l1"][pair_row])
    return {
        "no_threshold_model_primary_rank": _rank_metrics(valid, primary),
        "outer_train_exact_action_delta_win_frequency_baseline": _rank_metrics(
            valid, frequency_score,
        ),
        "minimal_action_delta_L1_baseline": _rank_metrics(valid, delta_score),
        "margin_prediction_used_in_primary_rank_diagnostic": False,
        "outer_valid_labels_used_to_build_baselines": False,
    }


def _candidate_win_gain_count(arrays: Mapping[str, np.ndarray]) -> int:
    return int(_rank_metrics(
        arrays, np.asarray(arrays["target_win_gain"], np.float64),
    )["upgrade_decisions"])


def lopo_diagnostic(
    arrays: Mapping[str, np.ndarray], audit: list[dict[str, Any]],
) -> dict[str, Any]:
    positive = np.flatnonzero(np.asarray(arrays["target_win_gain"]) == 1)
    opponents = sorted(set(map(int, np.asarray(arrays["opponent"])[positive])))
    return {
        "mode": "unknown_opponent_lopo_coverage_diagnostic_only",
        "status": (
            "coverage_supported_diagnostic_only"
            if len(opponents) >= 2 else "unsupported_abstain"
        ),
        "win_gain_opponents": opponents,
        "used_for_choice_or_threshold": False,
        "audit": audit,
    }


def evaluate(
    arrays: Mapping[str, np.ndarray], *, trees: int = 48,
    random_seed: int = 20260829,
) -> dict[str, Any]:
    input_audit = outcome_v1.base.validate_arrays(arrays)
    a, a_source = outcome_arrays(arrays)
    if tuple(sorted(set(map(int, a["seed"])))) != FORMAL_SEEDS:
        raise ValueError("pairwise v0 requires all eight pre-registered seeds")
    fold_map = seed_fold_map()
    row_folds = np.asarray([fold_map[int(seed)] for seed in a["seed"]], np.int8)
    chosen_by_decision: dict[int, int] = {}
    folds = []
    for outer_fold in range(4):
        train_rows = np.flatnonzero(row_folds != outer_fold)
        valid_rows = np.flatnonzero(row_folds == outer_fold)
        train, valid = _subset(a, train_rows), _subset(a, valid_rows)
        train_pairs = attach_pair_targets(train, build_pair_features(train))
        inner_groups = np.asarray([
            fold_map[int(seed)] for seed in train_pairs["seed"]
        ], np.int8)
        probability, margin, seed_models, seed_audit = group_oof_view(
            train_pairs, inner_groups, "leave-one-seed-fold-out", trees,
            random_seed + outer_fold * 100_003 + 10_101,
        )
        train_scores = candidate_scores(train, train_pairs, probability, margin)
        calibration, _ = calibrate(train, train_scores)
        _, _, _, lopo_audit = group_oof_view(
            train_pairs, np.asarray(train_pairs["opponent"]), "LOPO", trees,
            random_seed + outer_fold * 100_003 + 101,
        )
        valid_pairs = attach_pair_targets(valid, build_pair_features(valid))
        valid_probability, valid_margin = ensemble_mean(
            seed_models, np.asarray(valid_pairs["features"], np.float32),
        )
        valid_scores = candidate_scores(
            valid, valid_pairs, valid_probability, valid_margin,
        )
        valid_choices = select_choices(valid, valid_scores, {
            "max_regression_probability": calibration["max_regression_probability"],
            "min_win_gain_probability": calibration["min_win_gain_probability"],
        })
        for rows, choice in zip(
            outcome_v1.base.decision_slices(valid["decision"]), valid_choices,
            strict=True,
        ):
            chosen_by_decision[int(valid["decision"][rows[0]])] = int(valid_rows[int(choice)])
        positive = _candidate_win_gain_count(valid)
        if outer_fold == 0 and positive != 0:
            raise ValueError("pre-registered fold0 is no longer pure negative")
        folds.append({
            "left_out_seed_fold": outer_fold,
            "train_seeds": sorted(set(map(int, train["seed"]))),
            "valid_seeds": sorted(set(map(int, valid["seed"]))),
            "train_valid_seed_overlap": [],
            "outer_labels_used_for_model_or_threshold": False,
            "threshold_source": "outer-train leave-one-seed-fold-out OOF only",
            "chosen_thresholds": calibration,
            "inner_OOF_audit": {
                "only_seed_fold_OOF_used_for_calibration": True,
                "leave_one_seed_fold_out": seed_audit,
            },
            "unknown_opponent_lopo": lopo_diagnostic(train, lopo_audit),
            "outer_metrics_all_decisions": choice_metrics(valid, valid_choices),
            "outer_oracle_headroom_all_decisions": outcome_v1.oracle_headroom(valid),
            "no_threshold_rank_diagnostics": rank_diagnostics(
                train_pairs, valid, valid_pairs, valid_scores,
            ),
            "fold0_pure_negative_control": {
                "pre_registered": outer_fold == 0,
                "candidate_win_gain_decisions": positive,
                "passed": outer_fold != 0 or positive == 0,
            },
            "outer_pair_weight_audit": pair_weights(valid_pairs)[1],
            "outer_pair_rows_by_step": pair_rows_by_step(valid_pairs),
        })
    slices = outcome_v1.base.decision_slices(a["decision"])
    decision_order = [int(a["decision"][rows[0]]) for rows in slices]
    if set(decision_order) != set(chosen_by_decision):
        raise RuntimeError("outer folds did not choose every A decision")
    choices = np.asarray([chosen_by_decision[value] for value in decision_order])
    metrics = choice_metrics(a, choices)
    improving_folds = sum(
        fold["outer_oracle_headroom_all_decisions"]["oracle_raw_win_gain"] > 0
        and fold["outer_metrics_all_decisions"]["raw_win_delta"] > 0
        for fold in folds
    )
    acceptance = {
        "at_least_two_headroom_outer_folds_add_wins": improving_folds >= 2,
        "headroom_outer_folds_adding_wins": int(improving_folds),
        "wins_lost_zero": metrics["wins_lost"] == 0,
        "outcome_regressions_zero": metrics["outcome_regressions"] == 0,
        "margin_head_cannot_create_eligibility": True,
    }
    acceptance["passed"] = all(
        acceptance[key] for key in (
            "at_least_two_headroom_outer_folds_add_wins", "wins_lost_zero",
            "outcome_regressions_zero", "margin_head_cannot_create_eligibility",
        )
    )
    pairs_by_step: Counter[int] = Counter()
    for fold in folds:
        pairs_by_step.update({
            int(step): int(count)
            for step, count in fold["outer_pair_rows_by_step"].items()
        })
    return {
        "schema": SCHEMA,
        "status": (
            "pairwise_representation_sanity_screen_passed"
            if acceptance["passed"]
            else "pairwise_representation_sanity_screen_failed"
        ),
        "evidence_class": "train_only_repair_on_dev_screen_not_final_model",
        "seed_folds": [list(values) for values in SEED_FOLDS],
        "outer_labels_used_for_model_or_threshold": False,
        "phase_enabled": False,
        "primary_mode": "known_pool_leave_seed_fold_out",
        "unknown_opponent_mode": "lopo_coverage_diagnostic_only",
        "model_params": model_params(trees, random_seed),
        "feature_schema": {
            key: value for key, value in FEATURE_SCHEMA.items()
            if key not in {"shared_indices", "relative_indices", "pair_feature_names"}
        },
        "folds": folds, "acceptance_gate": acceptance,
        "all_decision_metrics": metrics,
        "all_decision_oracle_headroom": outcome_v1.oracle_headroom(a),
        "A_choices": choices, "A_source_indices": a_source,
        "input_audit": input_audit, "A_rows": int(len(a["decision"])),
        "A_decisions": len(slices),
        "A_candidate_KEEP_pairs": int(sum(
            fold["outer_pair_weight_audit"]["pair_rows"] for fold in folds
        )),
        "A_candidate_KEEP_pairs_by_step": {
            str(step): pairs_by_step[step] for step in sorted(pairs_by_step)
        },
        "objective_contract": {
            "primary": "raw win gain probability",
            "safety": "regression probability is an independent hard gate",
            "tie_break_only": "neutral-only predicted margin; never eligibility",
            "fallback": "KEEP", "pair_rows_are_not_independent_samples": True,
        },
    }
