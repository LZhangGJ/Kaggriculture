"""Evaluate causal first-shop route maps on one or more saved route screens."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def evaluate(matrix: np.lib.npyio.NpzFile, routes_by_seat: np.ndarray) -> dict:
    margin = np.asarray(matrix["margin"])
    invalid = np.asarray(matrix["invalid"])
    first_shop = np.asarray(matrix["first_shop"])
    global_routes = np.asarray(
        matrix["route_ids"] if "route_ids" in matrix else np.arange(margin.shape[2]),
        dtype=np.int32,
    )
    lookup = {int(route): index for index, route in enumerate(global_routes)}
    selected_margin: list[np.ndarray] = []
    selected_invalid: list[np.ndarray] = []
    by_seat = []
    for seat in (0, 1):
        global_selected = routes_by_seat[seat, first_shop[seat]]
        try:
            column = np.asarray([lookup[int(route)] for route in global_selected], np.int32)
        except KeyError as exc:
            raise ValueError(f"route {exc.args[0]} absent from screen") from exc
        rows = np.arange(margin.shape[1])
        value = margin[seat, rows, column]
        invalid_value = invalid[seat, rows, column]
        selected_margin.append(value)
        selected_invalid.append(invalid_value)
        by_seat.append(
            {
                "seat": seat,
                "games": int(value.size),
                "wins": int(np.sum(value > 0)),
                "ties": int(np.sum(value == 0)),
                "win_rate": float(np.mean(value > 0)),
                "mean_margin": float(np.mean(value)),
                "invalid_mean": float(np.mean(invalid_value)),
            }
        )
    values = np.concatenate(selected_margin)
    invalid_values = np.concatenate(selected_invalid)
    oracle = np.max(margin, axis=2).reshape(-1)
    return {
        "games": int(values.size),
        "wins": int(np.sum(values > 0)),
        "ties": int(np.sum(values == 0)),
        "win_rate": float(np.mean(values > 0)),
        "mean_margin": float(np.mean(values)),
        "median_margin": float(np.median(values)),
        "invalid_mean": float(np.mean(invalid_values)),
        "oracle_win_rate": float(np.mean(oracle > 0)),
        "by_seat": by_seat,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screens", type=Path, nargs="+", required=True)
    parser.add_argument("--maps", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for map_path in args.maps:
        payload = json.loads(map_path.read_text(encoding="utf-8"))
        routes = np.asarray(payload["route_ids"], dtype=np.int32)
        if routes.shape == (8,):
            routes = np.broadcast_to(routes[None, :], (2, 8))
        if routes.shape != (2, 8):
            raise ValueError(f"{map_path}: expected route_ids [8] or [2,8], got {routes.shape}")
        panels = []
        for screen_path in args.screens:
            npz_path = screen_path if screen_path.suffix == ".npz" else screen_path.with_suffix(".npz")
            with np.load(npz_path, allow_pickle=False) as matrix:
                metrics = evaluate(matrix, routes)
            panels.append({"screen": str(npz_path), **metrics})
        total_games = sum(panel["games"] for panel in panels)
        rows.append(
            {
                "map": str(map_path),
                "route_ids": routes.tolist(),
                "aggregate": {
                    "games": total_games,
                    "wins": sum(panel["wins"] for panel in panels),
                    "ties": sum(panel["ties"] for panel in panels),
                    "win_rate": sum(panel["wins"] for panel in panels) / total_games,
                    "mean_margin": sum(panel["mean_margin"] * panel["games"] for panel in panels)
                    / total_games,
                    "worst_panel_win_rate": min(panel["win_rate"] for panel in panels),
                },
                "panels": panels,
            }
        )

    output = {
        "schema": "kaggriculture.front40_fusion.shop-route-map-evaluation.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "screens": [str(path) for path in args.screens],
        "maps": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    best = max(rows, key=lambda row: (row["aggregate"]["win_rate"], row["aggregate"]["mean_margin"]))
    print(json.dumps({"status": "PASS", "best": best["map"], **best["aggregate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
