"""Summarize the achievable upper bound of a route-screen matrix.

Oracle numbers are diagnostics only: they use terminal outcomes and therefore
must never be used as runtime inputs.  The report makes this boundary explicit.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.matrix)
    margin = np.asarray(data["margin"], dtype=np.int64)
    invalid = np.asarray(data["invalid"], dtype=np.int64)
    source_episode = np.asarray(data["source_episode_id"], dtype=np.int64)
    source_reward = np.asarray(data["source_reward"], dtype=np.int64)
    seats, seeds, routes = margin.shape

    best_route = np.argmax(margin, axis=2)
    oracle_margin = np.take_along_axis(margin, best_route[..., None], axis=2)[..., 0]
    oracle_invalid = np.take_along_axis(invalid, best_route[..., None], axis=2)[..., 0]
    fixed_wins = np.sum(margin > 0, axis=(0, 1))
    fixed_margin = np.mean(margin, axis=(0, 1))
    fixed_order = np.lexsort((source_reward, fixed_margin, fixed_wins))[::-1]

    route_frequency = np.bincount(best_route.reshape(-1), minlength=routes)
    frequency_order = np.argsort(route_frequency)[::-1]
    top_fixed = []
    for route_id in fixed_order[:15]:
        top_fixed.append(
            {
                "route_id": int(route_id),
                "source_episode_id": int(source_episode[route_id]),
                "source_reward": int(source_reward[route_id]),
                "games": int(seats * seeds),
                "wins": int(fixed_wins[route_id]),
                "win_rate": float(fixed_wins[route_id] / (seats * seeds)),
                "mean_margin": float(fixed_margin[route_id]),
                "invalid_mean": float(np.mean(invalid[:, :, route_id])),
            }
        )
    top_oracle_choices = []
    for route_id in frequency_order[:20]:
        if route_frequency[route_id] <= 0:
            break
        top_oracle_choices.append(
            {
                "route_id": int(route_id),
                "source_episode_id": int(source_episode[route_id]),
                "source_reward": int(source_reward[route_id]),
                "oracle_choice_count": int(route_frequency[route_id]),
                "oracle_choice_share": float(route_frequency[route_id] / (seats * seeds)),
            }
        )

    payload = {
        "schema": "kaggriculture.front40_fusion.route-screen-oracle.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "source_matrix": str(args.matrix),
        "boundary": (
            "Terminal oracle is an analysis-only upper bound. Runtime policy may use only "
            "public current/past state and own private inventory; never oracle route IDs."
        ),
        "shape": {"seats": seats, "seeds": seeds, "routes": routes},
        "oracle": {
            "games": int(seats * seeds),
            "wins": int(np.sum(oracle_margin > 0)),
            "win_rate": float(np.mean(oracle_margin > 0)),
            "mean_margin": float(np.mean(oracle_margin)),
            "median_margin": float(np.median(oracle_margin)),
            "invalid_mean": float(np.mean(oracle_invalid)),
            "unique_routes_selected": int(np.sum(route_frequency > 0)),
        },
        "top_fixed_routes": top_fixed,
        "top_oracle_choices": top_oracle_choices,
        "oracle_route_id": best_route.astype(int).tolist(),
        "oracle_margin": oracle_margin.astype(int).tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "oracle": payload["oracle"], "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
