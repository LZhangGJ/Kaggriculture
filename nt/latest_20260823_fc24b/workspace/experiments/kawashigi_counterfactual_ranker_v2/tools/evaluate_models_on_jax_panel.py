#!/usr/bin/env python3
"""Evaluate portable route-value models on an untouched JAX outcome panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_hybrid_official_jax_lgbm import (  # noqa: E402
    ROUTES,
    jax_metrics,
    load_jax,
    portable_predict,
)


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def exact_one_sided_sign_p(gained: int, lost: int) -> float:
    total = gained + lost
    if total == 0 or gained <= lost:
        return 1.0
    return float(sum(math.comb(total, k) for k in range(gained, total + 1)) / (2**total))


def choose_payload(payload: dict[str, Any], x: np.ndarray) -> np.ndarray:
    if payload.get("schema") == "kawashigi-v3-v4-public-state-safety-gate-v1":
        baseline = choose_payload(payload["baseline"], x)
        challenger = choose_payload(payload["challenger"], x)
        gate = portable_predict(payload["gate_model"], x)
        return np.where(gate >= float(payload["threshold"]), challenger, baseline).astype(np.int16)
    scores = np.stack(
        [portable_predict(payload["models"][route], x) for route in ROUTES],
        axis=1,
    )
    return np.argmax(scores, axis=1).astype(np.int16)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument("--jax-features", type=Path, required=True)
    parser.add_argument("--jax-outcomes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = {name: resolve(value) for name, value in vars(args).items()}
    config = json.loads(paths["models"].read_text(encoding="utf-8"))
    bank_receipt = json.loads(paths["bank_receipt"].read_text(encoding="utf-8"))
    opponent_names = [str(row["name"]) for row in bank_receipt["opponents"]]
    data = load_jax(paths["jax_features"], paths["jax_outcomes"])

    predictions: dict[str, np.ndarray] = {}
    metrics: dict[str, Any] = {}
    model_sources: dict[str, Any] = {}
    for spec in config["models"]:
        name = str(spec["name"])
        path = resolve(Path(spec["path"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if list(payload["routes"]) != list(ROUTES):
            raise ValueError(f"route order mismatch for {name}")
        if list(payload["feature_names"]) != list(data["feature_names"]):
            raise ValueError(f"feature names mismatch for {name}")
        chosen = choose_payload(payload, data["x"])
        predictions[name] = chosen
        row = jax_metrics(data, chosen)
        row["by_opponent"] = {
            opponent_names[int(index)]: value
            for index, value in row.pop("by_opponent_id").items()
        }
        metrics[name] = row
        model_sources[name] = {
            "path": str(path),
            "sha256": sha256(path),
            "training_source": payload.get("training_source", payload.get("schema")),
            "jax_weight": payload.get("jax_weight"),
        }

    fallback = np.asarray([ROUTES.index(route) for route in data["fallback"]], dtype=np.int16)
    oracle = np.argmax(data["utility"], axis=1).astype(np.int16)
    metrics["centroid_fallback"] = jax_metrics(data, fallback)
    metrics["oracle"] = jax_metrics(data, oracle)
    for control in ("centroid_fallback", "oracle"):
        metrics[control]["by_opponent"] = {
            opponent_names[int(index)]: value
            for index, value in metrics[control].pop("by_opponent_id").items()
        }

    baseline_name = str(config["models"][0]["name"])
    baseline = predictions[baseline_name]
    row_index = np.arange(len(baseline))
    baseline_wins = data["wins"][row_index, baseline]
    baseline_margins = data["margins"][row_index, baseline]
    paired = {}
    for name, chosen in predictions.items():
        chosen_wins = data["wins"][row_index, chosen]
        chosen_margins = data["margins"][row_index, chosen]
        gained = int(np.sum(chosen_wins & ~baseline_wins))
        lost = int(np.sum(~chosen_wins & baseline_wins))
        paired[name] = {
            "baseline": baseline_name,
            "different_route_contexts": int(np.sum(chosen != baseline)),
            "gained_wins": gained,
            "lost_wins": lost,
            "net_wins": gained - lost,
            "one_sided_p_value": exact_one_sided_sign_p(gained, lost),
            "mean_margin_delta": float(np.mean(chosen_margins - baseline_margins)),
        }

    result = {
        "schema": "kawashigi-jax-fresh-model-evaluation-v1",
        "status": "FRESH_JAX_EVALUATED",
        "truth_boundary": "Independent JAX screening only; promotion still requires fresh official Python 1.32.7 evaluation.",
        "contexts": int(len(data["x"])),
        "opponents": opponent_names,
        "models": model_sources,
        "metrics": metrics,
        "paired_vs_first_model": paired,
        "sources": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
            if name != "output"
        },
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": result["status"],
        "contexts": result["contexts"],
        "metrics": {
            name: {
                "wins": row["wins"],
                "win_rate": row["win_rate"],
                "mean_margin": row["mean_margin"],
                "minimum_opponent_win_rate": row["minimum_opponent_win_rate"],
            }
            for name, row in metrics.items()
        },
        "paired_vs_first_model": paired,
        "output": str(paths["output"]),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
