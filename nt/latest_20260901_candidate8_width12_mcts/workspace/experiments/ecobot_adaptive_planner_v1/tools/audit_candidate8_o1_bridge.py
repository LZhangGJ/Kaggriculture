#!/usr/bin/env python3
"""Evaluate a no-clairvoyance Candidate8 bridge oracle with disjoint banks.

Bank A is the only bank used to choose one Candidate8 arm per public state.
Bank B is opened afterwards and is used only for evaluation.  The primary O1
selector is deliberately conservative: relative to KEEP it permits at most a
small Bank-A score-regression rate and requires a positive lower-tail margin
gain.  If no arm passes, it selects KEEP.

The audit also reports a scenario-mean selector and a per-future clairvoyant
O0 upper bound.  O0 is diagnostic only and cannot be deployed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


FAMILY_NAMES = [
    "KEEP",
    "SCHEDULE_LAYOUT",
    "CONTINUOUS_SCALE",
    "UNILATERAL",
    "MULTI_PROJECT",
    "TIMING",
    "MARKET_TRANSACTION",
    "PHASE_SUFFIX",
    "LOCAL_RECOVERY",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def score(own: np.ndarray, rival: np.ndarray) -> np.ndarray:
    return np.where(own > rival, 1.0, np.where(own == rival, 0.5, 0.0))


def lexicographic_best(
    score_value: np.ndarray,
    margin_value: np.ndarray,
    cash_value: np.ndarray,
) -> int:
    return int(max(
        range(len(score_value)),
        key=lambda index: (
            float(score_value[index]),
            float(margin_value[index]),
            float(cash_value[index]),
            -index,
        ),
    ))


def select_scenario_mean(own: np.ndarray, rival: np.ndarray) -> int:
    score_mean = score(own, rival).mean(axis=1)
    margin_mean = (own - rival).mean(axis=1)
    cash_mean = own.mean(axis=1)
    return lexicographic_best(score_mean, margin_mean, cash_mean)


def lower_quantile(values: np.ndarray, q: float) -> np.ndarray:
    return np.quantile(values, q, axis=1, method="lower")


def select_scenario_robust(
    own: np.ndarray,
    rival: np.ndarray,
    max_score_regression_rate: float,
    margin_quantile: float,
) -> tuple[int, dict[str, float]]:
    keep_own = own[0:1]
    keep_rival = rival[0:1]
    score_delta = score(own, rival) - score(keep_own, keep_rival)
    margin_delta = (own - rival) - (keep_own - keep_rival)
    cash_delta = own - keep_own

    regression_rate = (score_delta < 0).mean(axis=1)
    score_gain_mean = score_delta.mean(axis=1)
    margin_gain_mean = margin_delta.mean(axis=1)
    cash_gain_mean = cash_delta.mean(axis=1)
    margin_q = lower_quantile(margin_delta, margin_quantile)
    cash_q = lower_quantile(cash_delta, margin_quantile)

    eligible = np.flatnonzero(
        (regression_rate <= max_score_regression_rate)
        & (score_gain_mean >= 0.0)
        & (margin_q > 0.0)
    )
    if not len(eligible):
        return 0, {
            "score_regression_rate": 0.0,
            "score_gain_mean": 0.0,
            "margin_gain_mean": 0.0,
            "margin_gain_q": 0.0,
            "cash_gain_mean": 0.0,
            "cash_gain_q": 0.0,
        }

    chosen = int(max(
        eligible,
        key=lambda index: (
            float(score_gain_mean[index]),
            float(margin_q[index]),
            float(margin_gain_mean[index]),
            float(cash_q[index]),
            -int(index),
        ),
    ))
    return chosen, {
        "score_regression_rate": float(regression_rate[chosen]),
        "score_gain_mean": float(score_gain_mean[chosen]),
        "margin_gain_mean": float(margin_gain_mean[chosen]),
        "margin_gain_q": float(margin_q[chosen]),
        "cash_gain_mean": float(cash_gain_mean[chosen]),
        "cash_gain_q": float(cash_q[chosen]),
    }


def evaluate_fixed(own: np.ndarray, rival: np.ndarray, arm: int) -> dict[str, float]:
    margin = own[arm] - rival[arm]
    return {
        "score_rate": float(score(own[arm], rival[arm]).mean()),
        "mean_margin": float(margin.mean()),
        "mean_own_cash": float(own[arm].mean()),
        "margin_q10": float(np.quantile(margin, 0.10, method="lower")),
    }


def evaluate_clairvoyant(own: np.ndarray, rival: np.ndarray) -> dict[str, float]:
    margin = own - rival
    chosen = np.empty(own.shape[1], dtype=np.int32)
    for future in range(own.shape[1]):
        chosen[future] = lexicographic_best(
            score(own[:, future], rival[:, future]),
            margin[:, future],
            own[:, future],
        )
    future_index = np.arange(own.shape[1])
    selected_own = own[chosen, future_index]
    selected_rival = rival[chosen, future_index]
    selected_margin = selected_own - selected_rival
    return {
        "score_rate": float(score(selected_own, selected_rival).mean()),
        "mean_margin": float(selected_margin.mean()),
        "mean_own_cash": float(selected_own.mean()),
        "margin_q10": float(np.quantile(selected_margin, 0.10, method="lower")),
    }


def paired_delta(candidate: dict[str, float], keep: dict[str, float]) -> dict[str, float]:
    return {
        key: float(candidate[key] - keep[key])
        for key in ("score_rate", "mean_margin", "mean_own_cash", "margin_q10")
    }


def bootstrap_ci(values: np.ndarray, seed: int, samples: int = 2000) -> list[float]:
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def aggregate(rows: list[dict], method: str) -> dict:
    metrics = ("score_rate", "mean_margin", "mean_own_cash", "margin_q10")
    absolute = {
        key: float(np.mean([row[method][key] for row in rows]))
        for key in metrics
    }
    deltas = {
        key: np.asarray([
            row[method][key] - row["keep"][key] for row in rows
        ], dtype=np.float64)
        for key in metrics
    }
    result = {
        "states": len(rows),
        **absolute,
        "delta_vs_keep": {key: float(value.mean()) for key, value in deltas.items()},
        "state_block_bootstrap_95pct": {
            key: bootstrap_ci(value, 20260831 + offset)
            for offset, (key, value) in enumerate(deltas.items())
        },
    }
    if method in ("scenario_mean", "scenario_robust"):
        active = [row for row in rows if row[f"{method}_arm"] != 0]
        result["activation_rate"] = len(active) / len(rows) if rows else 0.0
        if active:
            result["active_score_regression_rate"] = float(np.mean([
                row[method]["score_rate"] < row["keep"]["score_rate"]
                for row in active
            ]))
            result["active_margin_negative_rate"] = float(np.mean([
                row[method]["mean_margin"] < row["keep"]["mean_margin"]
                for row in active
            ]))
        else:
            result["active_score_regression_rate"] = 0.0
            result["active_margin_negative_rate"] = 0.0
    return result


def text_values(array: np.ndarray) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in array
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-score-regression-rate", type=float, default=0.05)
    parser.add_argument("--margin-quantile", type=float, default=0.10)
    parser.add_argument(
        "--bank-split",
        choices=("interleaved", "halves"),
        default="interleaved",
    )
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"], dtype=np.int32)
        opponent = text_values(np.asarray(data["opponent"]))
        prefix_seed = np.asarray(data["prefix_seed"], dtype=np.int64)
        seat = np.asarray(data["seat"], dtype=np.int8)
        decision_day = np.asarray(data["decision_day"], dtype=np.int8)
        signature = np.asarray(data["signature"], dtype=np.uint64)
        family = np.asarray(data["family"], dtype=np.int8)
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)

    if own.shape != rival.shape or own.shape[0] != len(state_id):
        raise ValueError("future cash arrays do not match dataset rows")
    if own.shape[1] < 8 or own.shape[1] % 2:
        raise ValueError("future bank must contain an even count >= 8")

    if args.bank_split == "interleaved":
        bank_a = np.arange(0, own.shape[1], 2)
        bank_b = np.arange(1, own.shape[1], 2)
    else:
        half = own.shape[1] // 2
        bank_a = np.arange(0, half)
        bank_b = np.arange(half, own.shape[1])

    rows: list[dict] = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        if not len(indices):
            continue
        if int(family[indices[0]]) != 0:
            raise ValueError(f"state {state}: row zero is not KEEP")
        for name, array in {
            "opponent": opponent,
            "prefix_seed": prefix_seed,
            "seat": seat,
            "decision_day": decision_day,
        }.items():
            if len(np.unique(array[indices])) != 1:
                raise ValueError(f"state {state}: inconsistent {name}")

        state_own = own[indices]
        state_rival = rival[indices]
        own_a = state_own[:, bank_a]
        rival_a = state_rival[:, bank_a]
        own_b = state_own[:, bank_b]
        rival_b = state_rival[:, bank_b]

        mean_arm = select_scenario_mean(own_a, rival_a)
        robust_arm, robust_training = select_scenario_robust(
            own_a,
            rival_a,
            args.max_score_regression_rate,
            args.margin_quantile,
        )
        keep_result = evaluate_fixed(own_b, rival_b, 0)
        mean_result = evaluate_fixed(own_b, rival_b, mean_arm)
        robust_result = evaluate_fixed(own_b, rival_b, robust_arm)
        o0_result = evaluate_clairvoyant(own_b, rival_b)

        rows.append({
            "state_id": int(state),
            "opponent": str(opponent[indices[0]]),
            "prefix_seed": int(prefix_seed[indices[0]]),
            "seat": int(seat[indices[0]]),
            "decision_day": int(decision_day[indices[0]]),
            "arms": len(indices),
            "scenario_mean_arm": mean_arm,
            "scenario_mean_signature": int(signature[indices[mean_arm]]),
            "scenario_mean_family": FAMILY_NAMES[int(family[indices[mean_arm]])],
            "scenario_robust_arm": robust_arm,
            "scenario_robust_signature": int(signature[indices[robust_arm]]),
            "scenario_robust_family": FAMILY_NAMES[int(family[indices[robust_arm]])],
            "scenario_robust_bank_a": robust_training,
            "keep": keep_result,
            "scenario_mean": mean_result,
            "scenario_robust": robust_result,
            "o0_clairvoyant": o0_result,
            "scenario_mean_delta_vs_keep": paired_delta(mean_result, keep_result),
            "scenario_robust_delta_vs_keep": paired_delta(robust_result, keep_result),
            "o0_delta_vs_keep": paired_delta(o0_result, keep_result),
        })

    by_day: dict[str, dict] = {}
    by_opponent: dict[str, dict] = {}
    for key, key_rows in (
        (str(day), [row for row in rows if row["decision_day"] == day])
        for day in sorted({row["decision_day"] for row in rows})
    ):
        by_day[key] = {
            method: aggregate(key_rows, method)
            for method in ("keep", "scenario_mean", "scenario_robust", "o0_clairvoyant")
        }
    for key, key_rows in (
        (name, [row for row in rows if row["opponent"] == name])
        for name in sorted({row["opponent"] for row in rows})
    ):
        by_opponent[key] = {
            method: aggregate(key_rows, method)
            for method in ("keep", "scenario_mean", "scenario_robust", "o0_clairvoyant")
        }

    overall = {
        method: aggregate(rows, method)
        for method in ("keep", "scenario_mean", "scenario_robust", "o0_clairvoyant")
    }
    primary = overall["scenario_robust"]
    delta = primary["delta_vs_keep"]
    gate = {
        "score_rate_gain_ge_0_05": delta["score_rate"] >= 0.05,
        "mean_margin_gain_ge_1500": delta["mean_margin"] >= 1500.0,
        "score_gain_bootstrap_lower_nonnegative": (
            primary["state_block_bootstrap_95pct"]["score_rate"][0] >= 0.0
        ),
        "margin_gain_bootstrap_lower_positive": (
            primary["state_block_bootstrap_95pct"]["mean_margin"][0] > 0.0
        ),
        "active_score_regression_rate_le_0_05": (
            primary["active_score_regression_rate"] <= 0.05
        ),
        "active_margin_negative_rate_le_0_20": (
            primary["active_margin_negative_rate"] <= 0.20
        ),
    }

    payload = {
        "schema": "kaggriculture.candidate8-o1-bridge.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": {
            "bank_a": "selection only",
            "bank_b": "evaluation only; never read by either O1 selector",
            "o0": "per-future clairvoyant diagnostic upper bound on Bank B",
            "state_policy": (
                "Each decision day is audited as one Candidate8 intervention "
                "from its frozen R6 prefix, followed by the frozen executor."
            ),
        },
        "parameters": {
            "bank_split": args.bank_split,
            "future_count": int(own.shape[1]),
            "bank_a_indices": [int(value) for value in bank_a],
            "bank_b_indices": [int(value) for value in bank_b],
            "max_score_regression_rate": args.max_score_regression_rate,
            "margin_quantile": args.margin_quantile,
            "primary_selector": "scenario_robust",
            "bootstrap_seed": 20260831,
            "bootstrap_samples": 2000,
        },
        "dataset": {
            "path": str(args.dataset),
            "sha256": sha256(args.dataset),
            "rows": len(state_id),
            "states": len(rows),
        },
        "overall": overall,
        "by_day": by_day,
        "by_opponent": by_opponent,
        "primary_gate": gate,
        "primary_gate_passed": all(gate.values()),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "dataset": str(args.dataset),
        "states": len(rows),
        "primary": primary,
        "primary_gate": gate,
        "primary_gate_passed": all(gate.values()),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
