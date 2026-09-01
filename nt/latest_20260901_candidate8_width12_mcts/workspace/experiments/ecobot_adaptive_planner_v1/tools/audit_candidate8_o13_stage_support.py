#!/usr/bin/env python3
"""Audit stage-wise Candidate8 label support and bank stability for O1.3."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from train_candidate8_day6_safe_uplift import (
    paired_mean_margin_target,
    paired_targets,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def summarize(
    mask: np.ndarray,
    state_id: np.ndarray,
    state_effect: np.ndarray,
    score_a: np.ndarray,
    score_b: np.ndarray,
    lcb_a: np.ndarray,
    lcb_b: np.ndarray,
    mean_a: np.ndarray,
    mean_b: np.ndarray,
) -> dict:
    indices = np.flatnonzero(mask)
    states = np.unique(state_id[indices])
    effect = state_effect[indices]
    score_up_a = score_a[indices] > 1e-12
    score_up_b = score_b[indices] > 1e-12
    score_down_a = score_a[indices] < -1e-12
    score_down_b = score_b[indices] < -1e-12
    positive_a = lcb_a[indices] > 0.0
    positive_b = lcb_b[indices] > 0.0
    negative_a = lcb_a[indices] < 0.0
    negative_b = lcb_b[indices] < 0.0

    def rate(values: np.ndarray) -> float:
        return float(np.mean(values)) if len(values) else 0.0

    opportunity = {}
    for name, target in (
        ("score_up_a", score_a > 1e-12),
        ("score_up_b", score_b > 1e-12),
        ("positive_lcb_a", lcb_a > 0.0),
        ("positive_lcb_b", lcb_b > 0.0),
    ):
        opportunity[name] = float(np.mean([
            np.any(mask & (state_id == state) & state_effect & target)
            for state in states
        ])) if len(states) else 0.0

    stable_score_up = score_up_a & score_up_b
    stable_score_down = score_down_a & score_down_b
    stable_positive = positive_a & positive_b
    stable_negative = negative_a & negative_b
    return {
        "rows": int(len(indices)),
        "states": int(len(states)),
        "state_effect_rate": rate(effect),
        "bank_a": {
            "score_up_rate": rate(score_up_a),
            "score_down_rate": rate(score_down_a),
            "positive_lcb_rate": rate(positive_a),
            "negative_lcb_rate": rate(negative_a),
            "mean_margin_delta": float(np.mean(mean_a[indices])),
            "mean_lcb_delta": float(np.mean(lcb_a[indices])),
        },
        "bank_b": {
            "score_up_rate": rate(score_up_b),
            "score_down_rate": rate(score_down_b),
            "positive_lcb_rate": rate(positive_b),
            "negative_lcb_rate": rate(negative_b),
            "mean_margin_delta": float(np.mean(mean_b[indices])),
            "mean_lcb_delta": float(np.mean(lcb_b[indices])),
        },
        "cross_bank": {
            "stable_score_up_rate": rate(stable_score_up),
            "stable_score_down_rate": rate(stable_score_down),
            "stable_positive_lcb_rate": rate(stable_positive),
            "stable_negative_lcb_rate": rate(stable_negative),
            "score_sign_agreement_rate": rate(
                np.sign(score_a[indices]) == np.sign(score_b[indices])
            ),
            "lcb_sign_agreement_rate": rate(
                np.sign(lcb_a[indices]) == np.sign(lcb_b[indices])
            ),
        },
        "state_opportunity_rate": opportunity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--z", type=float, default=1.0)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        decision_day = np.asarray(data["decision_day"])
        family = np.asarray(data["family"])
        own = np.asarray(data["future_own_cash"], dtype=np.float64)
        rival = np.asarray(data["future_opponent_cash"], dtype=np.float64)
        state_effect = np.asarray(data["state_change_count_full"]) > 0

    if own.shape[1] < 8 or own.shape[1] % 2:
        raise ValueError("requires an even number of at least eight futures")
    bank_a = np.arange(0, own.shape[1], 2)
    bank_b = np.arange(1, own.shape[1], 2)
    score_a, lcb_a, _ = paired_targets(
        state_id, own[:, bank_a], rival[:, bank_a], args.z
    )
    score_b, lcb_b, _ = paired_targets(
        state_id, own[:, bank_b], rival[:, bank_b], args.z
    )
    mean_a = paired_mean_margin_target(
        state_id, own[:, bank_a], rival[:, bank_a]
    )
    mean_b = paired_mean_margin_target(
        state_id, own[:, bank_b], rival[:, bank_b]
    )

    non_keep = np.ones(len(state_id), dtype=bool)
    for state in np.unique(state_id):
        non_keep[np.flatnonzero(state_id == state)[0]] = False

    by_day = {
        str(int(day)): summarize(
            non_keep & (decision_day == day), state_id, state_effect,
            score_a, score_b, lcb_a, lcb_b, mean_a, mean_b,
        )
        for day in np.unique(decision_day)
    }
    by_family = {
        str(int(value)): summarize(
            non_keep & (family == value), state_id, state_effect,
            score_a, score_b, lcb_a, lcb_b, mean_a, mean_b,
        )
        for value in np.unique(family)
        if int(value) != 0
    }
    payload = {
        "schema": "kaggriculture.candidate8-o13-stage-support.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "bank_a_columns": bank_a.tolist(),
        "bank_b_columns": bank_b.tolist(),
        "overall": summarize(
            non_keep, state_id, state_effect,
            score_a, score_b, lcb_a, lcb_b, mean_a, mean_b,
        ),
        "by_day": by_day,
        "by_family": by_family,
        "stage_recommendation": {
            "EARLY": [6],
            "MID": [12],
            "LATE": [18],
            "reason": (
                "The three days have materially different label support. "
                "Day6 has the highest score-change opportunity and downside; "
                "Day12 is a lower-frequency production adjustment stage; "
                "Day18 is dominated by terminal margin and market timing."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "overall": payload["overall"],
        "by_day": by_day,
        "stage_recommendation": payload["stage_recommendation"],
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
