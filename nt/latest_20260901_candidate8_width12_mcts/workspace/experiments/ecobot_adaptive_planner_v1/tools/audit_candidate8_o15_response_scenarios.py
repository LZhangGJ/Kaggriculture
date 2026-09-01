#!/usr/bin/env python3
"""Audit whether identity-free rival response scenarios select useful arms.

The response preview is treated as a fixed public-information teacher.  True
multi-future outcomes are used only after selection to measure the chosen arm;
they never participate in the scenario score or tie-break.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


IDENTITY = (
    "state_id", "opponent", "prefix_seed", "seat", "decision_day",
    "signature", "family", "features", "feature_names",
)
HOLDOUT_OPPONENTS = {"G001", "G003", "G049", "G245"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def score_rate(margin: np.ndarray) -> np.ndarray:
    return np.mean(np.where(margin > 0, 1.0, np.where(margin == 0, 0.5, 0.0)), axis=1)


def paired_metrics(own: np.ndarray, rival: np.ndarray, keep: int, arm: int) -> dict:
    own_margin = own[arm] - rival[arm]
    keep_margin = own[keep] - rival[keep]
    score_delta = float(score_rate(own_margin[None, :])[0] - score_rate(keep_margin[None, :])[0])
    paired = own_margin - keep_margin
    mean = float(np.mean(paired))
    std = float(np.std(paired, ddof=1)) if len(paired) > 1 else 0.0
    lcb = mean - std / np.sqrt(max(1, len(paired)))
    return {"score_delta": score_delta, "margin_delta": mean, "margin_lcb": lcb}


def summarize(rows: list[dict]) -> dict:
    active = [row for row in rows if row["active"]]
    return {
        "states": len(rows),
        "activation_rate": len(active) / len(rows) if rows else 0.0,
        "mean_score_delta_all": float(np.mean([r["score_delta"] for r in rows])) if rows else 0.0,
        "mean_margin_delta_all": float(np.mean([r["margin_delta"] for r in rows])) if rows else 0.0,
        "mean_margin_lcb_all": float(np.mean([r["margin_lcb"] for r in rows])) if rows else 0.0,
        "mean_score_delta_active": float(np.mean([r["score_delta"] for r in active])) if active else 0.0,
        "mean_margin_delta_active": float(np.mean([r["margin_delta"] for r in active])) if active else 0.0,
        "mean_margin_lcb_active": float(np.mean([r["margin_lcb"] for r in active])) if active else 0.0,
        "active_negative_lcb_rate": float(np.mean([r["margin_lcb"] < 0 for r in active])) if active else 0.0,
        "active_score_regression_rate": float(np.mean([r["score_delta"] < 0 for r in active])) if active else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outcomes", required=True, type=Path)
    parser.add_argument("--response", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with np.load(args.outcomes, allow_pickle=False) as frozen, np.load(
        args.response, allow_pickle=False
    ) as response:
        exact = {key: bool(np.array_equal(frozen[key], response[key])) for key in IDENTITY}
        if not all(exact.values()):
            raise RuntimeError(f"response rows do not match frozen labels: {exact}")
        state_id = np.asarray(frozen["state_id"])
        opponents = np.asarray(frozen["opponent"])
        seeds = np.asarray(frozen["prefix_seed"])
        days = np.asarray(frozen["decision_day"])
        own = np.asarray(frozen["future_own_cash"], dtype=np.float64)
        rival = np.asarray(frozen["future_opponent_cash"], dtype=np.float64)
        scenario_outcomes = np.asarray(
            response["response_scenario_outcomes"], dtype=np.float64
        ) / 100.0
        scenario_names = [str(value) for value in response["response_scenario_names"]]

    unique_seeds = sorted(int(value) for value in np.unique(seeds))
    test_seeds = set(unique_seeds[-2:])
    calibration_seeds = set(unique_seeds[-4:-2])
    train_seeds = set(unique_seeds[:-4])

    rows: list[dict] = []
    pass_rows: list[dict] = []
    robust_vs_pass_rows: list[dict] = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        keep = int(indices[0])
        terminal_margin = scenario_outcomes[indices, :, 8]
        terminal_margin_delta = terminal_margin - terminal_margin[0:1, :]
        scenario_score = score_rate(terminal_margin)
        scenario_score_delta = scenario_score - scenario_score[0]
        q25 = np.quantile(terminal_margin_delta, 0.25, axis=1)
        worst = np.min(terminal_margin_delta, axis=1)
        mean = np.mean(terminal_margin_delta, axis=1)
        std = np.std(terminal_margin_delta, axis=1)
        local = max(
            range(len(indices)),
            key=lambda i: (
                float(scenario_score[i]), float(q25[i]), float(worst[i]),
                float(mean[i]), -float(std[i]), -i,
            ),
        )
        chosen = int(indices[local])
        metrics = paired_metrics(own, rival, keep, chosen)
        pass_local = max(
            range(len(indices)),
            key=lambda i: (float(scenario_outcomes[indices[i], 0, 6]), -i),
        )
        pass_chosen = int(indices[pass_local])
        pass_metrics = paired_metrics(own, rival, keep, pass_chosen)
        robust_vs_pass = paired_metrics(own, rival, pass_chosen, chosen)
        row0 = int(indices[0])
        opponent = str(opponents[row0])
        seed = int(seeds[row0])
        if opponent in HOLDOUT_OPPONENTS and seed in test_seeds:
            partition = "joint_unseen"
        elif opponent in HOLDOUT_OPPONENTS and seed in train_seeds:
            partition = "unseen_opponents"
        elif opponent in HOLDOUT_OPPONENTS:
            # Exactly mirror the frozen O1.4 split. Holdout-opponent rows on
            # calibration seeds are neither threshold data nor an internal
            # test partition.
            partition = "unused_holdout_calibration"
        elif seed in test_seeds:
            partition = "unseen_seeds"
        elif seed in calibration_seeds:
            partition = "calibration"
        else:
            partition = "fit"
        rows.append({
            "state": int(state), "day": int(days[row0]),
            "opponent": opponent, "seed": seed, "partition": partition,
            "selected_local_rank": int(local), "active": chosen != keep,
            "scenario_score": float(scenario_score[local]),
            "scenario_score_delta_vs_keep": float(scenario_score_delta[local]),
            "scenario_q25_margin_delta_vs_keep": float(q25[local]),
            "scenario_worst_margin_delta_vs_keep": float(worst[local]),
            "scenario_mean_margin_delta_vs_keep": float(mean[local]),
            "scenario_margin_delta_std": float(std[local]),
            **metrics,
        })
        pass_rows.append({
            "state": int(state), "day": int(days[row0]),
            "opponent": opponent, "seed": seed, "partition": partition,
            "selected_local_rank": int(pass_local),
            "active": pass_chosen != keep,
            **pass_metrics,
        })
        robust_vs_pass_rows.append({
            "state": int(state), "day": int(days[row0]),
            "opponent": opponent, "seed": seed, "partition": partition,
            "selected_local_rank": int(local),
            "active": chosen != pass_chosen,
            **robust_vs_pass,
        })

    summaries = {}
    pass_summaries = {}
    robust_vs_pass_summaries = {}
    for day in sorted({row["day"] for row in rows}):
        summaries[f"day{day}"] = {
            partition: summarize([
                row for row in rows
                if row["day"] == day and row["partition"] == partition
            ])
            for partition in (
                "fit", "calibration", "unseen_seeds",
                "unseen_opponents", "joint_unseen",
            )
        }
        pass_summaries[f"day{day}"] = {
            partition: summarize([
                row for row in pass_rows
                if row["day"] == day and row["partition"] == partition
            ])
            for partition in (
                "fit", "calibration", "unseen_seeds",
                "unseen_opponents", "joint_unseen",
            )
        }
        robust_vs_pass_summaries[f"day{day}"] = {
            partition: summarize([
                row for row in robust_vs_pass_rows
                if row["day"] == day and row["partition"] == partition
            ])
            for partition in (
                "fit", "calibration", "unseen_seeds",
                "unseen_opponents", "joint_unseen",
            )
        }
    payload = {
        "schema": "kaggriculture.candidate8-o15-response-scenario-audit.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "AUDIT_COMPLETE",
        "identity_arrays_exact": exact,
        "scenario_names": scenario_names,
        "selection_contract": (
            "Maximise absolute scenario win rate first, then q25, worst and "
            "mean margin improvement relative to the same-state KEEP arm, "
            "then lower variance. True labels are evaluation only."
        ),
        "split": {
            "train_seeds": sorted(train_seeds),
            "calibration_seeds": sorted(calibration_seeds),
            "test_seeds": sorted(test_seeds),
            "holdout_opponents": sorted(HOLDOUT_OPPONENTS),
        },
        "summaries": summaries,
        "pass_only_summaries": pass_summaries,
        "robust_vs_pass_summaries": robust_vs_pass_summaries,
        "rows": rows,
        "pass_only_rows": pass_rows,
        "robust_vs_pass_rows": robust_vs_pass_rows,
        "inputs": {
            "outcomes": {"path": str(args.outcomes), "sha256": sha256(args.outcomes)},
            "response": {"path": str(args.response), "sha256": sha256(args.response)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "robust": summaries,
        "pass_only": pass_summaries,
        "robust_vs_pass": robust_vs_pass_summaries,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
