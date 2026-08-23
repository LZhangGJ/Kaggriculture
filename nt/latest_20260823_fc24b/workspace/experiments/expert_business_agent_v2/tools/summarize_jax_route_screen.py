#!/usr/bin/env python3
"""Create an auditable summary of a JAX fixed-route outcome panel.

This is a screening report only.  It deliberately keeps strict wins, ties,
seat balance, mean terminal margin, and impossible hindsight-oracle coverage
separate so that a large route library cannot be mistaken for a deployable
router.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def metric(margins: np.ndarray) -> dict[str, Any]:
    # Expected layout is seat x seed.
    return {
        "games": int(margins.size),
        "wins": int(np.sum(margins > 0)),
        "ties": int(np.sum(margins == 0)),
        "losses": int(np.sum(margins < 0)),
        "win_rate": float(np.mean(margins > 0)),
        "score_rate": float(np.mean((margins > 0) + 0.5 * (margins == 0))),
        "mean_margin": float(np.mean(margins)),
        "by_seat": [
            {
                "seat": seat,
                "games": int(margins[seat].size),
                "win_rate": float(np.mean(margins[seat] > 0)),
                "mean_margin": float(np.mean(margins[seat])),
            }
            for seat in range(margins.shape[0])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument(
        "--opponents",
        required=True,
        help="Comma-separated opponent names to admit into the summary.",
    )
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    bank_path = resolve(args.bank_receipt)
    panel_path = resolve(args.panel_receipt)
    outcomes_path = resolve(args.outcomes)
    output_path = resolve(args.output)
    bank = json.loads(bank_path.read_text(encoding="utf-8"))
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    candidate_names = [str(value) for value in bank["candidate_names"]]
    opponent_names = [str(row["name"]) for row in bank["opponents"]]
    selected_names = [value.strip() for value in args.opponents.split(",") if value.strip()]
    missing = sorted(set(selected_names) - set(opponent_names))
    if missing:
        raise ValueError(f"unknown opponents: {missing}")
    selected_ids = [opponent_names.index(name) for name in selected_names]

    with np.load(outcomes_path, allow_pickle=False) as data:
        margins = np.asarray(data["margins"], dtype=np.int64)
        seeds = np.asarray(data["seeds"], dtype=np.int64)
    if margins.ndim != 5 or margins.shape[2] != 1:
        raise ValueError(f"unexpected margins shape: {margins.shape}")
    margins = margins[:, :, 0]
    if margins.shape[:3] != (len(candidate_names), 2, len(opponent_names)):
        raise ValueError(
            "bank/outcome shape mismatch: "
            f"{margins.shape} vs candidates={len(candidate_names)} opponents={len(opponent_names)}"
        )

    candidate_rows = []
    for candidate_id, candidate in enumerate(candidate_names):
        per_opponent = {
            name: metric(margins[candidate_id, :, opponent_id, :])
            for name, opponent_id in zip(selected_names, selected_ids)
        }
        values = list(per_opponent.values())
        candidate_rows.append({
            "candidate_id": candidate_id,
            "candidate": candidate,
            "minimum_opponent_win_rate": min(row["win_rate"] for row in values),
            "overall_win_rate": float(
                np.mean(margins[candidate_id, :, selected_ids, :] > 0)
            ),
            "overall_mean_margin": float(
                np.mean(margins[candidate_id, :, selected_ids, :])
            ),
            "per_opponent": per_opponent,
        })
    candidate_rows.sort(
        key=lambda row: (
            row["minimum_opponent_win_rate"],
            row["overall_win_rate"],
            row["overall_mean_margin"],
        ),
        reverse=True,
    )

    per_opponent: dict[str, Any] = {}
    for name, opponent_id in zip(selected_names, selected_ids):
        fixed = []
        for candidate_id, candidate in enumerate(candidate_names):
            row = metric(margins[candidate_id, :, opponent_id, :])
            row.update({"candidate_id": candidate_id, "candidate": candidate})
            fixed.append(row)
        fixed.sort(key=lambda row: (row["win_rate"], row["mean_margin"]), reverse=True)
        best_margin = np.max(margins[:, :, opponent_id, :], axis=0)
        per_opponent[name] = {
            "best_fixed": fixed[: args.top],
            "hindsight_oracle": metric(best_margin),
        }

    selected_margin = margins[:, :, selected_ids, :]
    best_margin = np.max(selected_margin, axis=0)
    # metric expects seat x seed, while this tensor is seat x opponent x seed.
    oracle_flat = best_margin.reshape(2, -1)
    routes_at_90 = [
        row for row in candidate_rows
        if row["minimum_opponent_win_rate"] >= 0.90
    ]
    route37 = next(
        (row for row in candidate_rows if row["candidate"].startswith("route37_")),
        None,
    )
    result = {
        "schema": "kaggriculture-jax-route-screen-summary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "JAX_SCREEN_ONLY",
        "all_done": bool(panel.get("all_done")),
        "backend": panel.get("backend"),
        "compile_inclusive_transitions_per_second": panel.get(
            "compile_inclusive_transitions_per_second"
        ),
        "seed_start": int(seeds[0]),
        "seed_count": int(seeds.size),
        "candidate_count": len(candidate_names),
        "selected_opponents": selected_names,
        "games_in_selected_summary": int(
            len(candidate_names) * 2 * len(selected_ids) * seeds.size
        ),
        "routes_meeting_every_opponent_90pct": routes_at_90,
        "top_candidates_minimax_then_overall": candidate_rows[: args.top],
        "route37_baseline": route37,
        "per_opponent": per_opponent,
        "hindsight_oracle_all_selected": metric(oracle_flat),
        "interpretation": (
            "The hindsight oracle may select a route after seeing terminal outcomes and is "
            "therefore an upper-bound diagnostic, not a legal policy. Promotion requires a "
            "public-state router and official Python 1.32.7 independent-seed validation."
        ),
        "inputs": {
            "bank_receipt": str(bank_path),
            "bank_receipt_sha256": sha256(bank_path),
            "panel_receipt": str(panel_path),
            "panel_receipt_sha256": sha256(panel_path),
            "outcomes": str(outcomes_path),
            "outcomes_sha256": sha256(outcomes_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "all_done": result["all_done"],
        "games": result["games_in_selected_summary"],
        "routes_at_90": len(routes_at_90),
        "top": candidate_rows[:3],
        "oracle": result["hindsight_oracle_all_selected"],
        "output": str(output_path),
    }, ensure_ascii=False, indent=2))
    return 0 if result["all_done"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
