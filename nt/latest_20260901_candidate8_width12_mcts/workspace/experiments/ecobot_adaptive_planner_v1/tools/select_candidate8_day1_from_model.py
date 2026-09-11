#!/usr/bin/env python3
"""Select Day1 Candidate8 ranks using only public features and a frozen model."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from train_candidate8_day1_lambdarank import make_features


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def parse_strings(raw: str) -> set[str]:
    return {value.strip() for value in raw.split(",") if value.strip()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--opponents", type=parse_strings)
    parser.add_argument("--seeds", type=parse_strings)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    with np.load(args.dataset, allow_pickle=False) as data:
        state_id = np.asarray(data["state_id"])
        opponents = np.asarray(data["opponent"])
        seeds = np.asarray(data["prefix_seed"])
        seat = np.asarray(data["seat"])
        day = np.asarray(data["decision_day"])
        signature = np.asarray(data["signature"])
        family = np.asarray(data["family"])
        raw = np.asarray(data["features"])
        names = [str(value) for value in data["feature_names"]]
    if np.any(day != 1):
        raise ValueError("selection dataset must contain Day1 only")
    x, feature_names = make_features(raw, family, seat, names)
    if feature_names != artifact["feature_names"]:
        raise RuntimeError("model and dataset feature schemas differ")
    started = time.perf_counter()
    prediction = artifact["model"].predict(x)
    inference_seconds = time.perf_counter() - started
    threshold = float(artifact["safe_threshold"])

    wanted_seeds = {int(value) for value in args.seeds} if args.seeds else None
    rows = []
    for state in np.unique(state_id):
        indices = np.flatnonzero(state_id == state)
        opponent = str(opponents[indices[0]])
        seed = int(seeds[indices[0]])
        if args.opponents and opponent not in args.opponents:
            continue
        if wanted_seeds is not None and seed not in wanted_seeds:
            continue
        keep = int(indices[0])
        order = sorted(
            indices,
            key=lambda index: (-float(prediction[index]), int(signature[index])),
        )
        raw_selected = int(order[0])
        uplift = float(prediction[raw_selected] - prediction[keep])
        selected = keep if uplift <= threshold else raw_selected
        # Rank is local to the deterministic shortlist, not the dataset row.
        rank = int(np.flatnonzero(indices == selected)[0])
        rows.append({
            "state_id": int(state),
            "opponent": opponent,
            "prefix_seed": seed,
            "seat": int(seat[indices[0]]),
            "selected_rank": rank,
            "selected_signature": int(signature[selected]),
            "selected_family": int(family[selected]),
            "activated": rank != 0,
            "predicted_uplift": uplift,
        })

    payload = {
        "schema": "kaggriculture.candidate8-day1-public-selection.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary": (
            "Selections use candidate plus public-state features and seat only. "
            "Stored future outcomes, opponent identity, and seed are not inputs."
        ),
        "dataset": {"path": str(args.dataset), "sha256": sha256(args.dataset)},
        "model": {"path": str(args.model), "sha256": sha256(args.model)},
        "safe_threshold": threshold,
        "inference": {
            "rows": int(len(x)),
            "seconds": inference_seconds,
            "rows_per_second": len(x) / inference_seconds if inference_seconds else 0.0,
        },
        "states": len(rows),
        "activation_rate": float(np.mean([row["activated"] for row in rows])),
        "selected_family_counts": dict(Counter(
            str(row["selected_family"]) for row in rows
        )),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "states": len(rows),
        "activation_rate": payload["activation_rate"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
