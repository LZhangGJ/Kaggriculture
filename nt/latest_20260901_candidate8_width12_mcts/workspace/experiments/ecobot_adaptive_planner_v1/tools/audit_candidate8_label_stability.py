#!/usr/bin/env python3
"""Compare two independent future banks on identical Candidate8 states."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


SCORE_WEIGHT = 10_000_000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def utility(data: dict[str, np.ndarray], indices: np.ndarray) -> np.ndarray:
    return (
        SCORE_WEIGHT * data["expected_score_rate"][indices]
        + data["expected_margin"][indices]
        + 0.001 * data["expected_own_cash"][indices]
    )


def summarize(rows: list[dict]) -> dict:
    return {
        "states": len(rows),
        "exact_oracle_signature_rate": float(np.mean([
            row["exact_oracle_signature"] for row in rows
        ])),
        "oracle_family_rate": float(np.mean([
            row["same_oracle_family"] for row in rows
        ])),
        "mutual_near_oracle_rate": float(np.mean([
            row["mutual_near_oracle"] for row in rows
        ])),
        "mean_a_choice_regret_on_b": float(np.mean([
            row["a_choice_regret_on_b"] for row in rows
        ])),
        "mean_b_choice_regret_on_a": float(np.mean([
            row["b_choice_regret_on_a"] for row in rows
        ])),
        "mean_score_regret": float(np.mean([
            row["mean_score_regret"] for row in rows
        ])),
        "mean_within_state_utility_correlation": float(np.mean([
            row["utility_correlation"] for row in rows
        ])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-a", required=True, type=Path)
    parser.add_argument("--bank-b", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--near-margin", type=float, default=1000.0)
    args = parser.parse_args()

    a = load(args.bank_a)
    b = load(args.bank_b)
    if not np.array_equal(a["state_id"], b["state_id"]):
        raise RuntimeError("state rows differ")
    for name in ("opponent", "prefix_seed", "seat", "decision_day", "signature"):
        if not np.array_equal(a[name], b[name]):
            raise RuntimeError(f"{name} differs between banks")

    rows = []
    by_day: dict[int, list[dict]] = defaultdict(list)
    for state in np.unique(a["state_id"]):
        indices = np.flatnonzero(a["state_id"] == state)
        ua = utility(a, indices)
        ub = utility(b, indices)
        oracle_a = int(np.argmax(ua))
        oracle_b = int(np.argmax(ub))
        score_a = a["expected_score_rate"][indices]
        score_b = b["expected_score_rate"][indices]
        margin_a = a["expected_margin"][indices]
        margin_b = b["expected_margin"][indices]
        a_on_b_score_regret = float(score_b[oracle_b] - score_b[oracle_a])
        b_on_a_score_regret = float(score_a[oracle_a] - score_a[oracle_b])
        a_on_b_margin_regret = float(margin_b[oracle_b] - margin_b[oracle_a])
        b_on_a_margin_regret = float(margin_a[oracle_a] - margin_a[oracle_b])
        near_ab = (
            a_on_b_score_regret <= 0.0
            and a_on_b_margin_regret <= args.near_margin
        )
        near_ba = (
            b_on_a_score_regret <= 0.0
            and b_on_a_margin_regret <= args.near_margin
        )
        correlation = 1.0
        if len(indices) > 1 and np.std(ua) > 0 and np.std(ub) > 0:
            correlation = float(np.corrcoef(ua, ub)[0, 1])
        row = {
            "state_id": int(state),
            "opponent": str(a["opponent"][indices[0]]),
            "seat": int(a["seat"][indices[0]]),
            "day": int(a["decision_day"][indices[0]]),
            "arms": int(len(indices)),
            "exact_oracle_signature": bool(
                a["signature"][indices[oracle_a]]
                == b["signature"][indices[oracle_b]]
            ),
            "same_oracle_family": bool(
                a["family"][indices[oracle_a]] == b["family"][indices[oracle_b]]
            ),
            "mutual_near_oracle": bool(near_ab and near_ba),
            "a_choice_regret_on_b": float(ub[oracle_b] - ub[oracle_a]),
            "b_choice_regret_on_a": float(ua[oracle_a] - ua[oracle_b]),
            "mean_score_regret": 0.5 * (
                a_on_b_score_regret + b_on_a_score_regret
            ),
            "utility_correlation": correlation,
        }
        rows.append(row)
        by_day[row["day"]].append(row)

    payload = {
        "schema": "kaggriculture.candidate8_label_stability.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "near_margin": args.near_margin,
        "summary": summarize(rows),
        "by_day": {str(day): summarize(day_rows) for day, day_rows in sorted(by_day.items())},
        "rows": rows,
        "inputs": {
            "bank_a": {"path": str(args.bank_a), "sha256": sha256(args.bank_a)},
            "bank_b": {"path": str(args.bank_b), "sha256": sha256(args.bank_b)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "summary": payload["summary"],
        "by_day": payload["by_day"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
