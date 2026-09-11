"""Verify online two-shop routing against compatible full-route outcomes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online-receipt", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--screen", type=Path, required=True)
    parser.add_argument("--base-map", type=Path, required=True)
    parser.add_argument("--second-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    online = json.loads(args.online_receipt.read_text(encoding="utf-8"))
    rows = [row for row in online["results"] if row["candidate"] == args.candidate]
    if sorted(int(row["candidate_seat"]) for row in rows) != [0, 1]:
        raise RuntimeError("online receipt does not contain both seats")
    base_routes = np.asarray(
        json.loads(args.base_map.read_text(encoding="utf-8"))["route_ids"],
        dtype=np.int32,
    )
    second_payload = json.loads(args.second_map.read_text(encoding="utf-8"))
    second = np.asarray(second_payload["second_route_ids_by_first_shop"], dtype=np.int32)
    matrix = np.load(args.screen.with_suffix(".npz"), allow_pickle=False)
    seeds = np.asarray(matrix["seeds"], dtype=np.int64)
    wanted = np.arange(
        int(online["seed_start"]),
        int(online["seed_start"]) + int(online["batch_per_seat"]),
        dtype=np.int64,
    )
    seed_lookup = {int(seed): index for index, seed in enumerate(seeds.tolist())}
    try:
        seed_columns = np.asarray([seed_lookup[int(seed)] for seed in wanted], np.int32)
    except KeyError as exc:
        raise ValueError(f"online seed absent from screen: {exc.args[0]}") from exc
    route_ids = np.asarray(matrix["route_ids"], dtype=np.int32)
    route_lookup = {int(route): index for index, route in enumerate(route_ids.tolist())}
    observed = np.asarray(matrix["observed_shops"])
    cash = np.asarray(matrix["cash"])
    opponent_cash = np.asarray(matrix["opponent_cash"])

    seat_rows: list[dict[str, object]] = []
    total_cash_mismatch = 0
    total_opponent_mismatch = 0
    total_route_mismatch = 0
    for row in sorted(rows, key=lambda item: int(item["candidate_seat"])):
        seat = int(row["candidate_seat"])
        expected_cash: list[int] = []
        expected_opponent: list[int] = []
        expected_routes: list[int] = []
        for seed_column in seed_columns:
            first = int(observed[seat, seed_column, 0, 0])
            base = int(base_routes[first])
            base_column = route_lookup[base]
            second_shop = int(observed[seat, seed_column, base_column, 1])
            selected = int(second[first, second_shop])
            selected_column = route_lookup[selected]
            expected_cash.append(int(cash[seat, seed_column, selected_column]))
            expected_opponent.append(
                int(opponent_cash[seat, seed_column, selected_column])
            )
            expected_routes.append(selected)
        actual_cash = np.asarray(row["candidate_cash"], dtype=np.int64)
        actual_opponent = np.asarray(row["fc24b_cash"], dtype=np.int64)
        actual_routes = np.asarray(row["final_route"], dtype=np.int64)
        expected_cash_array = np.asarray(expected_cash, dtype=np.int64)
        expected_opponent_array = np.asarray(expected_opponent, dtype=np.int64)
        expected_routes_array = np.asarray(expected_routes, dtype=np.int64)
        cash_mismatch = int(np.sum(actual_cash != expected_cash_array))
        opponent_mismatch = int(np.sum(actual_opponent != expected_opponent_array))
        route_mismatch = int(np.sum(actual_routes != expected_routes_array))
        total_cash_mismatch += cash_mismatch
        total_opponent_mismatch += opponent_mismatch
        total_route_mismatch += route_mismatch
        seat_rows.append(
            {
                "seat": seat,
                "games": int(actual_cash.size),
                "candidate_cash_mismatch": cash_mismatch,
                "opponent_cash_mismatch": opponent_mismatch,
                "final_route_mismatch": route_mismatch,
                "max_candidate_cash_abs_error": int(
                    np.max(np.abs(actual_cash - expected_cash_array))
                ),
                "max_opponent_cash_abs_error": int(
                    np.max(np.abs(actual_opponent - expected_opponent_array))
                ),
            }
        )

    status = (
        "PASS"
        if total_cash_mismatch == total_opponent_mismatch == total_route_mismatch == 0
        else "FAIL"
    )
    payload = {
        "schema": "kaggriculture.front40_fusion.online-second-shop-screen-parity.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "online_receipt": str(args.online_receipt),
        "candidate": args.candidate,
        "screen": str(args.screen),
        "base_map": str(args.base_map),
        "second_map": str(args.second_map),
        "games": int(2 * wanted.size),
        "candidate_cash_mismatch_total": total_cash_mismatch,
        "opponent_cash_mismatch_total": total_opponent_mismatch,
        "final_route_mismatch_total": total_route_mismatch,
        "seats": seat_rows,
        "status": status,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
