#!/usr/bin/env python3
"""Nested train-only screen for a tiny state x action listwise selector.

The candidate SHA names one frozen executable four-step action block.  It is
used only as a cross-fitted gain/risk prior feature: it never makes a row
eligible and never removes KEEP.  Counterfactual outcome and margin labels are
used for fitting/calibration inside an outer training split, and only for
evaluation/acceptance in that outer split.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from functools import lru_cache
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


SCHEMA = "phase-challenger-bilinear-listwise-A-v2"
FORMAL_SEEDS = tuple(range(2026086300, 2026086308))
SEED_FOLDS = (
    (2026086300, 2026086304),
    (2026086301, 2026086305),
    (2026086302, 2026086306),
    (2026086303, 2026086307),
)
STATE_WIDTH, ACTION_WIDTH, PROJECTION_WIDTH = 337, 154, 8
BILINEAR_WIDTH = PROJECTION_WIDTH * PROJECTION_WIDTH
BASE_WIDTH = BILINEAR_WIDTH + PROJECTION_WIDTH
PRIMARY_WIDTH = BASE_WIDTH + 2
KEEP_EDIT = 0
SHUFFLE_SEEDS = (2026082911, 2026082923, 2026082937)

DEFAULT_CONFIG: dict[str, Any] = {
    "state_width": STATE_WIDTH,
    "relative_action_width": ACTION_WIDTH,
    "projection_width": PROJECTION_WIDTH,
    "bilinear_width": BILINEAR_WIDTH,
    "base_width": BASE_WIDTH,
    "primary_width": PRIMARY_WIDTH,
    "projection": "sha256-signed-rademacher/sqrt(input_width)",
    "projection_seed": 20260829,
    "state_scaling": "outer-train decision-representative zscore",
    "action_scaling": "outer-train uncentered RMS; KEEP remains zero",
    "optimizer": "scipy.optimize.L-BFGS-B",
    "optimizer_initialization": "all-zero",
    "optimizer_maxiter": 120,
    "optimizer_ftol": 1e-11,
    "listwise_l2": 0.08,
    "risk_l2": 0.08,
    "sha_prior_alpha": 4.0,
    "risk_probability_cap": 0.5,
    "threshold_grid_quantiles": 129,
    "shuffle_seeds": SHUFFLE_SEEDS,
    "ordinal_labels": {"loss": 0, "tie": 1, "win": 2},
    "ordinal_objective": "negative log group-softmax probability mass on maximum outcome; no margin",
    "selection_score": "candidate group-softmax probability minus KEEP probability",
    "selection_score_symmetric_clip": 0.999,
    "margin_semantics": "inner-OOF final threshold tie-break and outer audit only",
    "candidate_sha_semantics": "frozen executable four-step block soft prior only",
}


def seed_fold_map() -> dict[int, int]:
    result = {seed: fold for fold, seeds in enumerate(SEED_FOLDS) for seed in seeds}
    if tuple(sorted(result)) != FORMAL_SEEDS:
        raise AssertionError("the pre-registered four seed folds changed")
    return result


def decision_slices(decision: np.ndarray, rows: np.ndarray | None = None) -> list[np.ndarray]:
    values = np.asarray(decision, np.int64)
    selected = np.arange(len(values), dtype=np.int64) if rows is None else np.asarray(rows, np.int64)
    if len(selected) == 0 or np.any(selected[1:] <= selected[:-1]):
        raise ValueError("decision rows must be nonempty, unique, and increasing")
    chosen = values[selected]
    starts = np.r_[0, np.flatnonzero(chosen[1:] != chosen[:-1]) + 1]
    groups = [selected[start:stop] for start, stop in zip(starts, np.r_[starts[1:], len(selected)], strict=True)]
    seen = [int(values[group[0]]) for group in groups]
    if len(seen) != len(set(seen)):
        raise ValueError("a decision reappears after its contiguous block")
    return groups


def _sha_text(value: Any) -> str:
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("ascii")
    text = str(value)
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise ValueError("candidate_sha256 must be lowercase hexadecimal SHA-256")
    return text


def _row_folds(seed: np.ndarray) -> np.ndarray:
    mapping = seed_fold_map()
    try:
        return np.asarray([mapping[int(value)] for value in seed], np.int8)
    except KeyError as error:
        raise ValueError(f"unknown formal seed: {error.args[0]}") from error


def validate_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    required = (
        "state", "action", "candidate_sha256", "decision", "union_index",
        "seed", "seat", "opponent", "step", "edit", "outcome", "margin",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise ValueError(f"missing v2 arrays: {missing}")
    n = len(np.asarray(arrays["decision"]))
    state = np.asarray(arrays["state"], np.float64)
    action = np.asarray(arrays["action"], np.float64)
    if state.shape != (n, STATE_WIDTH) or action.shape != (n, ACTION_WIDTH):
        raise ValueError("v2 requires aligned state337 and relative-action154")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise ValueError("state/action contains non-finite values")
    for name in required[2:]:
        if len(np.asarray(arrays[name])) != n:
            raise ValueError(f"{name} is not row-aligned")
    outcome = np.asarray(arrays["outcome"], np.int8)
    margin = np.asarray(arrays["margin"], np.float64)
    if not set(map(int, outcome)) <= {0, 1, 2} or not np.isfinite(margin).all():
        raise ValueError("outcome/margin labels are invalid")
    if tuple(sorted(set(map(int, np.asarray(arrays["seed"]))))) != FORMAL_SEEDS:
        raise ValueError("v2 requires all eight pre-registered formal seeds")

    edit = np.asarray(arrays["edit"])
    union = np.asarray(arrays["union_index"], np.int64)
    sha = np.asarray(arrays["candidate_sha256"])
    all_sha: set[str] = set()
    decisions = decision_slices(np.asarray(arrays["decision"], np.int64))
    cells: dict[tuple[int, int, int], list[tuple[int, int]]] = defaultdict(list)
    for group in decisions:
        if not np.array_equal(union[group], np.arange(len(group))):
            raise ValueError("union_index must be exact row order 0..K-1 per state")
        keep = group[edit[group] == KEEP_EDIT]
        if len(keep) != 1 or int(keep[0]) != int(group[0]) or union[int(keep[0])] != 0:
            raise ValueError("each decision requires union_index=0 as its sole KEEP")
        if not np.array_equal(action[int(keep[0])], np.zeros(ACTION_WIDTH)):
            raise ValueError("KEEP relative action must be exactly zero")
        if not np.array_equal(state[group], np.broadcast_to(state[group[:1]], state[group].shape)):
            raise ValueError("observable state337 changed within one decision")
        for name in ("seed", "seat", "opponent", "step"):
            values = np.asarray(arrays[name])[group]
            if not np.all(values == values[0]):
                raise ValueError(f"{name} changed within one decision")
        texts = [_sha_text(value) for value in sha[group]]
        if len(texts) != len(set(texts)):
            raise ValueError("candidate SHA collision within one canonical decision")
        all_sha.update(texts)
        cell = tuple(int(np.asarray(arrays[name])[group[0]]) for name in ("seed", "opponent", "step"))
        cells[cell].append((int(np.asarray(arrays["seat"])[group[0]]), len(group)))
    for cell, values in cells.items():
        if sorted(seat for seat, _ in values) != [0, 1] or len({count for _, count in values}) != 1:
            raise ValueError(f"paired-seat cell is incomplete/asymmetric: {cell}")
    return {
        "rows": n,
        "decisions": len(decisions),
        "paired_seed_opponent_step_cells": len(cells),
        "candidate_sha_unique": len(all_sha),
        "full_coverage": True,
        "candidate_sha_is_soft_prior_not_eligibility": True,
    }


@lru_cache(maxsize=None)
def signed_projection(tag: str, input_width: int, output_width: int, seed: int) -> np.ndarray:
    matrix = np.empty((input_width, output_width), np.float64)
    prefix = f"{tag}|{input_width}|{output_width}|{seed}|".encode("ascii")
    for row in range(input_width):
        for column in range(output_width):
            bit = hashlib.sha256(prefix + f"{row}|{column}".encode("ascii")).digest()[0] & 1
            matrix[row, column] = 1.0 if bit else -1.0
    matrix /= math.sqrt(input_width)
    matrix.setflags(write=False)
    return matrix


def projection_hash(seed: int = int(DEFAULT_CONFIG["projection_seed"])) -> str:
    digest = hashlib.sha256()
    for tag, width in (("observable-state337", STATE_WIDTH), ("relative-action154", ACTION_WIDTH)):
        matrix = signed_projection(tag, width, PROJECTION_WIDTH, seed)
        digest.update(tag.encode("ascii"))
        digest.update(matrix.tobytes())
    return digest.hexdigest()


def _decision_weights(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[int, float]:
    groups = decision_slices(np.asarray(arrays["decision"]), rows)
    cells: dict[tuple[int, int, int], list[np.ndarray]] = defaultdict(list)
    for group in groups:
        cell = tuple(int(np.asarray(arrays[name])[group[0]]) for name in ("seed", "opponent", "step"))
        cells[cell].append(group)
    result: dict[int, float] = {}
    for cell, members in cells.items():
        seats = [int(np.asarray(arrays["seat"])[group[0]]) for group in members]
        counts = [len(group) for group in members]
        if sorted(seats) != [0, 1] or len(set(counts)) != 1:
            raise ValueError(f"paired cell lost equal two-seat panels: {cell}")
        for group in members:
            result[int(np.asarray(arrays["decision"])[group[0]])] = 1.0 / len(members)
    return result


def _relative_labels(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gain = np.zeros(len(np.asarray(arrays["decision"])), np.float64)
    risk = np.zeros_like(gain)
    outcome = np.asarray(arrays["outcome"], np.int8)
    edit = np.asarray(arrays["edit"])
    for group in decision_slices(np.asarray(arrays["decision"]), rows):
        keep = int(group[edit[group] == KEEP_EDIT][0])
        gain[group] = outcome[group] > outcome[keep]
        risk[group] = outcome[group] < outcome[keep]
    return gain, risk


def _candidate_row_weights(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> np.ndarray:
    weights = np.zeros(len(np.asarray(arrays["decision"])), np.float64)
    cells: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    edit = np.asarray(arrays["edit"])
    for group in decision_slices(np.asarray(arrays["decision"]), rows):
        cell = tuple(int(np.asarray(arrays[name])[group[0]]) for name in ("seed", "opponent", "step"))
        cells[cell].extend(map(int, group[edit[group] != KEEP_EDIT]))
    for cell, members in cells.items():
        if not members:
            raise ValueError(f"cell has no non-KEEP candidates: {cell}")
        weights[members] = 1.0 / len(members)
    return weights


def _fit_prior(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray, target: np.ndarray, alpha: float,
) -> dict[str, Any]:
    edit = np.asarray(arrays["edit"])
    sha = np.asarray(arrays["candidate_sha256"])
    weights = _candidate_row_weights(arrays, rows)
    candidates = rows[edit[rows] != KEEP_EDIT]
    total_weight = float(weights[candidates].sum())
    positive_weight = float(np.dot(weights[candidates], target[candidates]))
    base = positive_weight / total_weight if total_weight else 0.0
    count: dict[str, float] = defaultdict(float)
    positive: dict[str, float] = defaultdict(float)
    for row in candidates:
        key = _sha_text(sha[row])
        count[key] += float(weights[row])
        positive[key] += float(weights[row] * target[row])
    return {"count": dict(count), "positive": dict(positive), "base": base, "alpha": float(alpha)}


def _apply_prior(model: Mapping[str, Any], sha_values: np.ndarray, edit_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = float(model["base"])
    alpha = float(model["alpha"])
    base_clip = float(np.clip(base, 1e-6, 1.0 - 1e-6))
    base_logit = math.log(base_clip / (1.0 - base_clip))
    feature = np.zeros(len(sha_values), np.float64)
    probability = np.full(len(sha_values), base, np.float64)
    for index, (value, edit) in enumerate(zip(sha_values, edit_values, strict=True)):
        if int(edit) == KEEP_EDIT:
            probability[index] = 0.0
            continue
        key = _sha_text(value)
        if key not in model["count"]:
            feature[index] = 0.0
            probability[index] = base
            continue
        rate = (float(model["positive"][key]) + alpha * base) / (float(model["count"][key]) + alpha)
        clipped = float(np.clip(rate, 1e-6, 1.0 - 1e-6))
        feature[index] = math.log(clipped / (1.0 - clipped)) - base_logit
        probability[index] = rate
    return feature, probability


def _crossfit_prior_features(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    gain, risk = _relative_labels(arrays, rows)
    folds = _row_folds(np.asarray(arrays["seed"]))
    result = np.zeros((len(np.asarray(arrays["decision"])), 2), np.float64)
    unique = sorted(set(map(int, folds[rows])))
    if len(unique) < 2:
        return result, np.zeros_like(result)
    probabilities = np.zeros_like(result)
    for heldout in unique:
        valid = rows[folds[rows] == heldout]
        train = rows[folds[rows] != heldout]
        for column, target in enumerate((gain, risk)):
            model = _fit_prior(arrays, train, target, alpha)
            result[valid, column], probabilities[valid, column] = _apply_prior(
                model, np.asarray(arrays["candidate_sha256"])[valid], np.asarray(arrays["edit"])[valid],
            )
    return result, probabilities


def _predict_prior_features(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, predict: np.ndarray, alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    gain, risk = _relative_labels(arrays, train)
    features = np.zeros((len(predict), 2), np.float64)
    probabilities = np.zeros_like(features)
    for column, target in enumerate((gain, risk)):
        model = _fit_prior(arrays, train, target, alpha)
        features[:, column], probabilities[:, column] = _apply_prior(
            model, np.asarray(arrays["candidate_sha256"])[predict], np.asarray(arrays["edit"])[predict],
        )
    return features, probabilities


def _shuffle_state_rows(arrays: Mapping[str, np.ndarray], rows: np.ndarray, seed: int) -> np.ndarray:
    state = np.asarray(arrays["state"], np.float64)[rows].copy()
    groups = decision_slices(np.asarray(arrays["decision"]), rows)
    positions = {int(row): at for at, row in enumerate(rows)}
    matched: dict[tuple[int, int], list[np.ndarray]] = defaultdict(list)
    for group in groups:
        key = (int(np.asarray(arrays["step"])[group[0]]), int(np.asarray(arrays["seat"])[group[0]]))
        matched[key].append(group)
    for key, members in sorted(matched.items()):
        digest = hashlib.sha256(f"{seed}|{key[0]}|{key[1]}".encode("ascii")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        order = rng.permutation(len(members))
        source_states = [np.asarray(arrays["state"], np.float64)[members[index][0]].copy() for index in order]
        for group, source_state in zip(members, source_states, strict=True):
            for row in group:
                state[positions[int(row)]] = source_state
    return state


def _projected_features(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, predict: np.ndarray,
    *, train_prior: np.ndarray | None, predict_prior: np.ndarray | None,
    shuffle_seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    train_state = (
        np.asarray(arrays["state"], np.float64)[train]
        if shuffle_seed is None else _shuffle_state_rows(arrays, train, shuffle_seed + 1)
    )
    predict_state = (
        np.asarray(arrays["state"], np.float64)[predict]
        if shuffle_seed is None else _shuffle_state_rows(arrays, predict, shuffle_seed + 2)
    )
    train_groups = decision_slices(np.asarray(arrays["decision"]), train)
    position = {int(row): at for at, row in enumerate(train)}
    representatives = np.asarray([position[int(group[0])] for group in train_groups], np.int64)
    mean = train_state[representatives].mean(axis=0)
    scale = train_state[representatives].std(axis=0)
    scale[scale < 1e-8] = 1.0
    action = np.asarray(arrays["action"], np.float64)
    action_scale = np.sqrt(np.mean(np.square(action[train]), axis=0))
    action_scale[action_scale < 1e-8] = 1.0
    seed = int(DEFAULT_CONFIG["projection_seed"])
    state_projection = signed_projection("observable-state337", STATE_WIDTH, PROJECTION_WIDTH, seed)
    action_projection = signed_projection("relative-action154", ACTION_WIDTH, PROJECTION_WIDTH, seed)

    def build(rows: np.ndarray, states: np.ndarray, prior: np.ndarray | None) -> np.ndarray:
        state8 = ((states - mean) / scale) @ state_projection
        action8 = (action[rows] / action_scale) @ action_projection
        bilinear = np.einsum("ni,nj->nij", state8, action8).reshape(len(rows), BILINEAR_WIDTH)
        base = np.concatenate((bilinear, action8), axis=1)
        return base if prior is None else np.concatenate((base, prior[rows] if len(prior) == len(action) else prior), axis=1)

    return build(train, train_state, train_prior), build(predict, predict_state, predict_prior)


def _fit_listwise(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    l2: float, maxiter: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    groups = decision_slices(np.asarray(arrays["decision"]), rows)
    position = {int(row): at for at, row in enumerate(rows)}
    local = [np.asarray([position[int(row)] for row in group], np.int64) for group in groups]
    outcome = np.asarray(arrays["outcome"], np.int8)
    best_mask = np.zeros(len(rows), np.float64)
    group_weight = _decision_weights(arrays, rows)
    weights = np.zeros(len(rows), np.float64)
    lengths = np.asarray([len(group) for group in local], np.int64)
    starts = np.r_[0, np.cumsum(lengths)[:-1]]
    for original, loc in zip(groups, local, strict=True):
        best = np.max(outcome[original])
        winners = loc[outcome[original] == best]
        best_mask[winners] = 1.0
        weights[loc] = group_weight[int(np.asarray(arrays["decision"])[original[0]])]
    signal_groups = sum(len(set(map(int, outcome[group]))) > 1 for group in groups)
    if signal_groups == 0:
        return np.zeros(x.shape[1], np.float64), {
            "kind": "constant_no_within_decision_ordinal_signal", "success": True,
            "iterations": 0, "signal_groups": 0,
        }

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        return _listwise_loss_gradient(
            coef, x, starts, lengths, best_mask, weights, l2,
        )

    fit = minimize(
        objective, np.zeros(x.shape[1], np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": int(maxiter), "ftol": float(DEFAULT_CONFIG["optimizer_ftol"])},
    )
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise RuntimeError("listwise L-BFGS-B produced non-finite parameters")
    if not bool(fit.success):
        raise RuntimeError(
            "listwise L-BFGS-B did not converge: "
            f"status={fit.status}; message={fit.message}; nit={fit.nit}; fun={fit.fun}"
        )
    return np.asarray(fit.x, np.float64), {
        "kind": "group-listwise-softmax-L-BFGS-B", "success": bool(fit.success),
        "iterations": int(fit.nit), "signal_groups": int(signal_groups),
        "objective": float(fit.fun),
    }


def _listwise_loss_gradient(
    coef: np.ndarray, x: np.ndarray, starts: np.ndarray, lengths: np.ndarray,
    best_mask: np.ndarray, row_decision_weight: np.ndarray, l2: float,
) -> tuple[float, np.ndarray]:
    """Return -log P(the tied-best set) and its exact gradient."""

    score = x @ coef
    maxima = np.maximum.reduceat(score, starts)
    shifted = score - np.repeat(maxima, lengths)
    exponents = np.exp(shifted)
    denominators = np.add.reduceat(exponents, starts)
    best_denominators = np.add.reduceat(exponents * best_mask, starts)
    if np.any(best_denominators <= 0.0):
        raise RuntimeError("a listwise group lost its ordinal-best candidate set")
    probability = exponents / np.repeat(denominators, lengths)
    best_conditional = (
        exponents * best_mask / np.repeat(best_denominators, lengths)
    )
    per_group = np.log(denominators) - np.log(best_denominators)
    decision_mass = row_decision_weight[starts]
    loss = float(np.dot(decision_mass, per_group) + 0.5 * l2 * np.dot(coef, coef))
    gradient = x.T @ (
        row_decision_weight * (probability - best_conditional)
    ) + l2 * coef
    return loss, gradient


def _fit_risk(
    x: np.ndarray, arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    l2: float, maxiter: int,
) -> tuple[np.ndarray | None, float, dict[str, Any]]:
    _, risk = _relative_labels(arrays, rows)
    edit = np.asarray(arrays["edit"])
    candidate = np.flatnonzero(edit[rows] != KEEP_EDIT)
    y = risk[rows][candidate]
    if len(np.unique(y)) == 1:
        value = float(y[0]) if len(y) else 0.0
        return None, value, {
            "kind": "constant_missing-class-safe", "constant_probability": value,
            "positive": int(y.sum()), "rows": int(len(y)),
        }
    values = x[candidate]
    sample_weight_all = _candidate_row_weights(arrays, rows)[rows]
    sample_weight = sample_weight_all[candidate]
    design = np.concatenate((values, np.ones((len(values), 1))), axis=1)

    def objective(coef: np.ndarray) -> tuple[float, np.ndarray]:
        logit = design @ coef
        probability = expit(logit)
        loss = float(np.dot(sample_weight, np.logaddexp(0.0, logit) - y * logit))
        loss += 0.5 * l2 * float(np.dot(coef[:-1], coef[:-1]))
        gradient = design.T @ (sample_weight * (probability - y))
        gradient[:-1] += l2 * coef[:-1]
        return loss, gradient

    fit = minimize(
        objective, np.zeros(design.shape[1], np.float64), method="L-BFGS-B", jac=True,
        options={"maxiter": int(maxiter), "ftol": float(DEFAULT_CONFIG["optimizer_ftol"])},
    )
    if not np.isfinite(fit.fun) or not np.isfinite(fit.x).all():
        raise RuntimeError("risk L-BFGS-B produced non-finite parameters")
    if not bool(fit.success):
        raise RuntimeError(
            "risk L-BFGS-B did not converge: "
            f"status={fit.status}; message={fit.message}; nit={fit.nit}; fun={fit.fun}"
        )
    return np.asarray(fit.x, np.float64), math.nan, {
        "kind": "independent-risk-logistic-L-BFGS-B", "success": bool(fit.success),
        "iterations": int(fit.nit), "positive": int(y.sum()), "rows": int(len(y)),
        "objective": float(fit.fun),
    }


def fit_predict(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, predict: np.ndarray,
    *, include_sha: bool, shuffle_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    alpha = float(DEFAULT_CONFIG["sha_prior_alpha"])
    if include_sha:
        train_prior, _ = _crossfit_prior_features(arrays, train, alpha)
        predict_prior, _ = _predict_prior_features(arrays, train, predict, alpha)
    else:
        train_prior = predict_prior = None
    train_x, predict_x = _projected_features(
        arrays, train, predict, train_prior=train_prior, predict_prior=predict_prior,
        shuffle_seed=shuffle_seed,
    )
    coef, listwise_audit = _fit_listwise(
        train_x, arrays, train, float(DEFAULT_CONFIG["listwise_l2"]),
        int(DEFAULT_CONFIG["optimizer_maxiter"]),
    )
    risk_coef, risk_constant, risk_audit = _fit_risk(
        train_x, arrays, train, float(DEFAULT_CONFIG["risk_l2"]),
        int(DEFAULT_CONFIG["optimizer_maxiter"]),
    )
    raw_score = predict_x @ coef
    score = np.zeros(len(predict), np.float64)
    predict_position = {int(row): at for at, row in enumerate(predict)}
    edit_all = np.asarray(arrays["edit"])
    for group in decision_slices(np.asarray(arrays["decision"]), predict):
        local = np.asarray([predict_position[int(row)] for row in group], np.int64)
        logits = raw_score[local]
        probability = np.exp(logits - np.max(logits))
        probability /= probability.sum()
        keep_local = local[edit_all[group] == KEEP_EDIT]
        if len(keep_local) != 1:
            raise ValueError("prediction group lost KEEP")
        score[local] = probability - probability[int(np.flatnonzero(local == keep_local[0])[0])]
        score[int(keep_local[0])] = 0.0
    score = np.clip(
        score, -float(DEFAULT_CONFIG["selection_score_symmetric_clip"]),
        float(DEFAULT_CONFIG["selection_score_symmetric_clip"]),
    )
    if risk_coef is None:
        risk_probability = np.full(len(predict), risk_constant, np.float64)
    else:
        risk_probability = expit(np.concatenate((predict_x, np.ones((len(predict), 1))), axis=1) @ risk_coef)
    edit = edit_all[predict]
    risk_probability[edit == KEEP_EDIT] = 0.0
    return score, risk_probability, {
        "feature_width": int(train_x.shape[1]), "include_sha_soft_prior": include_sha,
        "shuffle_seed": shuffle_seed, "listwise": listwise_audit, "risk": risk_audit,
    }


def prior_only_predict(
    arrays: Mapping[str, np.ndarray], train: np.ndarray, predict: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features, probabilities = _predict_prior_features(
        arrays, train, predict, float(DEFAULT_CONFIG["sha_prior_alpha"]),
    )
    score = features[:, 0] - features[:, 1]
    score[np.asarray(arrays["edit"])[predict] == KEEP_EDIT] = 0.0
    return score, probabilities[:, 1]


def _global_scores(n: int, rows: np.ndarray, values: np.ndarray) -> np.ndarray:
    result = np.full(n, np.nan, np.float64)
    result[rows] = values
    return result


def select_choices(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray, score: np.ndarray,
    risk_probability: np.ndarray, score_floor: float,
) -> np.ndarray:
    edit = np.asarray(arrays["edit"])
    union = np.asarray(arrays["union_index"])
    choices = []
    cap = float(DEFAULT_CONFIG["risk_probability_cap"])
    for group in decision_slices(np.asarray(arrays["decision"]), rows):
        keep = int(group[edit[group] == KEEP_EDIT][0])
        candidates = group[edit[group] != KEEP_EDIT]
        eligible = candidates[(risk_probability[candidates] <= cap) & (score[candidates] > score_floor)]
        chosen = keep if not len(eligible) else max(map(int, eligible), key=lambda row: (
            float(score[row]), -float(risk_probability[row]), -int(union[row]),
        ))
        choices.append(chosen)
    return np.asarray(choices, np.int64)


def choice_metrics(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray, choices: np.ndarray,
) -> dict[str, Any]:
    groups = decision_slices(np.asarray(arrays["decision"]), rows)
    if len(groups) != len(choices):
        raise ValueError("choices do not cover every decision")
    edit = np.asarray(arrays["edit"])
    outcome = np.asarray(arrays["outcome"], np.int8)
    margin = np.asarray(arrays["margin"], np.float64)
    opponent = np.asarray(arrays["opponent"])
    seed = np.asarray(arrays["seed"])
    outcome_delta: list[int] = []
    win_delta: list[int] = []
    same_margin: list[float] = []
    fired_same: list[float] = []
    fired = 0
    gain_seeds: set[int] = set()
    gain_opponents: set[int] = set()
    by_opponent: dict[int, dict[str, int]] = defaultdict(lambda: {"outcome_regressions": 0, "wins_lost": 0, "wins_gained": 0})
    for group, choice_value in zip(groups, choices, strict=True):
        choice = int(choice_value)
        if not np.any(group == choice):
            raise ValueError("choice escaped its decision")
        keep = int(group[edit[group] == KEEP_EDIT][0])
        delta = int(outcome[choice]) - int(outcome[keep])
        win = int(outcome[choice] == 2) - int(outcome[keep] == 2)
        is_fire = int(edit[choice]) != KEEP_EDIT
        margin_delta = float(margin[choice] - margin[keep])
        outcome_delta.append(delta)
        win_delta.append(win)
        same_margin.append(margin_delta if is_fire and delta == 0 else 0.0)
        if is_fire and delta == 0:
            fired_same.append(margin_delta)
        fired += is_fire
        opp = int(opponent[choice])
        by_opponent[opp]["outcome_regressions"] += int(delta < 0)
        by_opponent[opp]["wins_lost"] += int(win < 0)
        by_opponent[opp]["wins_gained"] += int(win > 0)
        if win > 0:
            gain_seeds.add(int(seed[choice]))
            gain_opponents.add(opp)
    delta_values = np.asarray(outcome_delta, np.int16)
    win_values = np.asarray(win_delta, np.int16)
    same_values = np.asarray(fired_same, np.float64)
    worst = max(1, int(math.ceil(0.2 * len(same_values)))) if len(same_values) else 0
    return {
        "decisions": len(groups), "fires": int(fired),
        "raw_win_delta": int(win_values.sum()),
        "wins_gained": int(np.count_nonzero(win_values > 0)),
        "wins_lost": int(np.count_nonzero(win_values < 0)),
        "outcome_delta_sum": int(delta_values.sum()),
        "outcome_upgrades": int(np.count_nonzero(delta_values > 0)),
        "outcome_regressions": int(np.count_nonzero(delta_values < 0)),
        "same_outcome_margin_delta_sum": float(np.sum(same_margin)),
        "negative_same_outcome_fires": int(np.count_nonzero(same_values < 0)),
        "same_outcome_margin_CVaR20": float(np.mean(np.sort(same_values)[:worst])) if worst else 0.0,
        "gain_seeds": sorted(gain_seeds), "gain_opponents": sorted(gain_opponents),
        "by_opponent": {str(key): value for key, value in sorted(by_opponent.items())},
        "all_decisions_evaluated": True, "KEEP_explicit": True,
    }


def calibrate_threshold(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    score: np.ndarray, risk_probability: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    edit = np.asarray(arrays["edit"])
    values = np.asarray(score[rows[edit[rows] != KEEP_EDIT]], np.float64)
    finite = values[np.isfinite(values)]
    thresholds = [math.inf]
    if len(finite):
        unique = np.unique(finite)
        if len(unique) > int(DEFAULT_CONFIG["threshold_grid_quantiles"]):
            unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, int(DEFAULT_CONFIG["threshold_grid_quantiles"]))))
        thresholds.extend(float(np.nextafter(value, -math.inf)) for value in unique)
    trials = []
    for threshold in thresholds:
        choices = select_choices(arrays, rows, score, risk_probability, threshold)
        metrics = choice_metrics(arrays, rows, choices)
        safe = metrics["wins_lost"] == 0 and metrics["outcome_regressions"] == 0 and all(
            value["wins_lost"] == 0 and value["outcome_regressions"] == 0
            for value in metrics["by_opponent"].values()
        )
        credible = math.isinf(threshold) or metrics["wins_gained"] > 0 or metrics["outcome_upgrades"] > 0
        if safe and credible:
            trials.append((threshold, metrics))
    if not trials:
        raise RuntimeError("KEEP threshold disappeared from calibration")
    threshold, metrics = max(trials, key=lambda item: (
        item[1]["raw_win_delta"], item[1]["outcome_upgrades"],
        item[1]["same_outcome_margin_delta_sum"], -item[1]["fires"], -item[0],
    ))
    return float(threshold), {
        "score_floor": float(threshold),
        "risk_probability_cap": float(DEFAULT_CONFIG["risk_probability_cap"]),
        "inner_OOF_metrics": metrics,
        "threshold_candidates": len(thresholds),
        "outer_valid_labels_used": False,
        "no_credible_ordinal_gain_forces_KEEP": math.isinf(threshold),
    }


def rank_metrics(
    arrays: Mapping[str, np.ndarray], rows: np.ndarray,
    score: np.ndarray, risk_probability: np.ndarray,
) -> dict[str, Any]:
    edit = np.asarray(arrays["edit"])
    outcome = np.asarray(arrays["outcome"], np.int8)
    headroom = top1 = 0
    reciprocal = 0.0
    cap = float(DEFAULT_CONFIG["risk_probability_cap"])
    for group in decision_slices(np.asarray(arrays["decision"]), rows):
        keep = int(group[edit[group] == KEEP_EDIT][0])
        candidates = group[edit[group] != KEEP_EDIT]
        best = int(np.max(outcome[candidates]))
        if best <= int(outcome[keep]):
            continue
        winners = set(map(int, candidates[outcome[candidates] == best]))
        ranking = sorted(map(int, candidates), key=lambda row: (
            -(float(score[row]) if risk_probability[row] <= cap else -math.inf),
            float(risk_probability[row]), int(np.asarray(arrays["union_index"])[row]),
        ))
        headroom += 1
        top1 += int(ranking[0] in winners)
        reciprocal += 1.0 / (1 + next(index for index, row in enumerate(ranking) if row in winners))
    return {
        "upgrade_decisions": headroom,
        "top1": float(top1 / headroom) if headroom else 0.0,
        "MRR": float(reciprocal / headroom) if headroom else 0.0,
    }


def pareto_better(primary: Mapping[str, Any], control: Mapping[str, Any]) -> bool:
    if int(primary["upgrade_decisions"]) != int(control["upgrade_decisions"]):
        raise ValueError("rank metrics compare different headroom decisions")
    return (
        float(primary["top1"]) >= float(control["top1"])
        and float(primary["MRR"]) >= float(control["MRR"])
        and (
            float(primary["top1"]) > float(control["top1"])
            or float(primary["MRR"]) > float(control["MRR"])
        )
    )


def _oracle_headroom(arrays: Mapping[str, np.ndarray], rows: np.ndarray) -> dict[str, int]:
    edit = np.asarray(arrays["edit"])
    outcome = np.asarray(arrays["outcome"], np.int8)
    upgrades = wins = nonwin = 0
    for group in decision_slices(np.asarray(arrays["decision"]), rows):
        keep = int(group[edit[group] == KEEP_EDIT][0])
        best = int(np.max(outcome[group]))
        upgrades += int(best > int(outcome[keep]))
        wins += int(best == 2 and int(outcome[keep]) < 2)
        nonwin += int(int(outcome[keep]) == 0 and np.any(outcome[group] == 1))
    return {"ordinal_upgrade_decisions": upgrades, "win_upgrade_decisions": wins, "loss_to_tie_decisions": nonwin}


def evaluate(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    input_audit = validate_arrays(arrays)
    n = len(np.asarray(arrays["decision"]))
    folds = _row_folds(np.asarray(arrays["seed"]))
    all_rows = np.arange(n, dtype=np.int64)
    model_names = ("primary", "bilinear_no_SHA", "SHA_prior_only", *(f"state_shuffle_{seed}" for seed in SHUFFLE_SEEDS))
    outer_score = {name: np.full(n, np.nan, np.float64) for name in model_names}
    outer_risk = {name: np.full(n, np.nan, np.float64) for name in model_names}
    chosen_by_decision: dict[int, int] = {}
    fold_reports = []
    for outer_fold in range(4):
        train = all_rows[folds != outer_fold]
        valid = all_rows[folds == outer_fold]
        inner_score = np.full(n, np.nan, np.float64)
        inner_risk = np.full(n, np.nan, np.float64)
        inner_audit = []
        for inner_fold in sorted(set(map(int, folds[train]))):
            inner_train = train[folds[train] != inner_fold]
            inner_valid = train[folds[train] == inner_fold]
            predicted_score, predicted_risk, audit = fit_predict(
                arrays, inner_train, inner_valid, include_sha=True,
            )
            inner_score[inner_valid] = predicted_score
            inner_risk[inner_valid] = predicted_risk
            inner_audit.append({
                "left_out_seed_fold": inner_fold,
                "train_seeds": sorted(set(map(int, np.asarray(arrays["seed"])[inner_train]))),
                "valid_seeds": sorted(set(map(int, np.asarray(arrays["seed"])[inner_valid]))),
                "train_valid_seed_overlap": [], "model": audit,
            })
        if not np.isfinite(inner_score[train]).all() or not np.isfinite(inner_risk[train]).all():
            raise RuntimeError("outer-train inner OOF predictions are incomplete")
        floor, calibration = calibrate_threshold(arrays, train, inner_score, inner_risk)

        score, risk, primary_audit = fit_predict(arrays, train, valid, include_sha=True)
        outer_score["primary"][valid], outer_risk["primary"][valid] = score, risk
        score, risk, no_sha_audit = fit_predict(arrays, train, valid, include_sha=False)
        outer_score["bilinear_no_SHA"][valid], outer_risk["bilinear_no_SHA"][valid] = score, risk
        score, risk = prior_only_predict(arrays, train, valid)
        outer_score["SHA_prior_only"][valid], outer_risk["SHA_prior_only"][valid] = score, risk
        shuffle_audits = []
        for shuffle_seed in SHUFFLE_SEEDS:
            name = f"state_shuffle_{shuffle_seed}"
            score, risk, audit = fit_predict(
                arrays, train, valid, include_sha=True, shuffle_seed=shuffle_seed,
            )
            outer_score[name][valid], outer_risk[name][valid] = score, risk
            shuffle_audits.append(audit)
        primary_choices = select_choices(
            arrays, valid, outer_score["primary"], outer_risk["primary"], floor,
        )
        for group, choice in zip(decision_slices(np.asarray(arrays["decision"]), valid), primary_choices, strict=True):
            chosen_by_decision[int(np.asarray(arrays["decision"])[group[0]])] = int(choice)
        fold_reports.append({
            "outer_fold": outer_fold,
            "train_seeds": sorted(set(map(int, np.asarray(arrays["seed"])[train]))),
            "valid_seeds": sorted(set(map(int, np.asarray(arrays["seed"])[valid]))),
            "train_valid_seed_overlap": [],
            "threshold_source": "outer-train inner seed-fold OOF only",
            "outer_labels_used_for_model_or_threshold": False,
            "calibration": calibration, "inner_OOF": inner_audit,
            "primary_model": primary_audit, "bilinear_no_SHA_model": no_sha_audit,
            "state_shuffle_models": shuffle_audits,
            "outer_metrics": choice_metrics(arrays, valid, primary_choices),
            "outer_oracle_headroom": _oracle_headroom(arrays, valid),
        })
    if any(not np.isfinite(values).all() for values in (*outer_score.values(), *outer_risk.values())):
        raise RuntimeError("outer OOF controls lost row coverage")
    groups = decision_slices(np.asarray(arrays["decision"]), all_rows)
    choices = np.asarray([chosen_by_decision[int(np.asarray(arrays["decision"])[group[0]])] for group in groups], np.int64)
    metrics = choice_metrics(arrays, all_rows, choices)
    ranks = {
        name: rank_metrics(arrays, all_rows, outer_score[name], outer_risk[name])
        for name in model_names
    }
    ranks["KEEP"] = {
        "upgrade_decisions": ranks["primary"]["upgrade_decisions"], "top1": 0.0, "MRR": 0.0,
    }
    comparators = ("SHA_prior_only", *(f"state_shuffle_{seed}" for seed in SHUFFLE_SEEDS))
    pareto = {name: pareto_better(ranks["primary"], ranks[name]) for name in comparators}
    no_regressions_by_fold_opponent = all(
        fold["outer_metrics"]["outcome_regressions"] == 0
        and fold["outer_metrics"]["wins_lost"] == 0
        and all(
            values["outcome_regressions"] == 0 and values["wins_lost"] == 0
            for values in fold["outer_metrics"]["by_opponent"].values()
        ) for fold in fold_reports
    )
    headroom_gain_folds = sum(
        fold["outer_oracle_headroom"]["win_upgrade_decisions"] > 0
        and fold["outer_metrics"]["raw_win_delta"] >= 1
        for fold in fold_reports
    )
    acceptance = {
        "each_fold_each_opponent_zero_outcome_regressions_and_wins_lost": no_regressions_by_fold_opponent,
        "at_least_two_headroom_folds_gain_at_least_one_win": headroom_gain_folds >= 2,
        "headroom_folds_with_gain": int(headroom_gain_folds),
        "total_raw_win_gain_at_least_two": metrics["raw_win_delta"] >= 2,
        "gain_covers_at_least_two_seeds": len(metrics["gain_seeds"]) >= 2,
        "gain_covers_at_least_two_opponents": len(metrics["gain_opponents"]) >= 2,
        "same_outcome_margin_nonnegative": metrics["same_outcome_margin_delta_sum"] >= 0.0,
        "primary_rank_Pareto_over_SHA_prior_and_all_state_shuffles": all(pareto.values()),
        "full_outer_decision_coverage": metrics["decisions"] == input_audit["decisions"],
        "outer_labels_evaluation_acceptance_only": True,
    }
    acceptance["passed"] = all(value for key, value in acceptance.items() if key not in {"headroom_folds_with_gain", "passed"})
    nonwin = _oracle_headroom(arrays, all_rows)["loss_to_tie_decisions"]
    return {
        "schema": SCHEMA,
        "status": "repair_on_dev_screen_passed" if acceptance["passed"] else "repair_on_dev_screen_failed",
        "evidence_class": "train-only repair-on-dev; not deployment evidence",
        "config": dict(DEFAULT_CONFIG),
        "projection_sha256": projection_hash(),
        "seed_folds": [list(values) for values in SEED_FOLDS],
        "candidate_sha_contract": {
            "semantic": "authoritative frozen executable four-step candidate block",
            "use": "cross-fitted soft gain/risk prior only",
            "eligibility": False, "KEEP_always_present": True,
        },
        "ordinal_contract": {
            "labels": {"loss": 0, "tie": 1, "win": 2},
            "loss_to_tie_decisions": int(nonwin),
            "nonwin_upgrade_status": "constant-safe-zero" if nonwin == 0 else "modeled-by-full-ordinal-listwise",
            "same_outcome_margin_model_feature": False,
        },
        "controls": {"rank_metrics": ranks, "primary_Pareto": pareto},
        "folds": fold_reports, "all_decision_metrics": metrics,
        "acceptance_gate": acceptance, "input_audit": input_audit,
        "A_choices": choices,
        "outer_labels_used_for_model_or_threshold": False,
    }


__all__ = [
    "ACTION_WIDTH", "BASE_WIDTH", "DEFAULT_CONFIG", "FORMAL_SEEDS",
    "KEEP_EDIT", "PRIMARY_WIDTH", "PROJECTION_WIDTH", "SCHEMA",
    "SEED_FOLDS", "SHUFFLE_SEEDS", "STATE_WIDTH", "choice_metrics",
    "decision_slices", "evaluate", "pareto_better", "projection_hash",
    "seed_fold_map", "signed_projection", "validate_arrays",
]












