"""Replay representative M2.6 traces in official Python 1.32.7."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
GPU_SIM = REPO_ROOT / "gpu_sim"
for source_dir in (GPU_SIM / "src", GPU_SIM / "tests", GPU_SIM / "tools"):
    sys.path.insert(0, str(source_dir))

from kaggle_environments import make  # noqa: E402
from generate_reference_traces import canonical_frame  # noqa: E402
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS, CROPS, NUM_PRODUCTS, PRODUCTS, SHED_ITEMS, MarketOp, UnitOp,
)
from kaggriculture_jax.state import reset  # noqa: E402
from kaggriculture_jax.types import Action, State  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _unit(op: int, item: int, amount: int) -> list[object]:
    try:
        name = UnitOp(op).name
    except ValueError:
        name = "PASS"
    if op == UnitOp.PLANT and 0 <= item < len(CROPS):
        return [name, CROPS[item]]
    if op in (UnitOp.PICKUP, UnitOp.PLACE) and 0 <= item < len(SHED_ITEMS):
        return [name, SHED_ITEMS[item], int(amount)]
    return [name]


def _market(op: int, item: int, amount: int) -> list[object] | None:
    if op == MarketOp.NONE:
        return None
    name = MarketOp(op).name
    if op in (MarketOp.HIRE, MarketOp.BUY_LAND):
        return [name]
    if op == MarketOp.BUY_SEED:
        value = CROPS[item] if 0 <= item < len(CROPS) else None
    elif op == MarketOp.BUY_ANIMAL:
        animal = item - NUM_PRODUCTS
        value = ANIMALS[animal] if 0 <= animal < len(ANIMALS) else None
    else:
        value = PRODUCTS[item] if 0 <= item < len(PRODUCTS) else None
    return [name, value, int(amount)] if value is not None else None


def _decode_action(trace: Action, step: int, lane: int, seat: int) -> dict:
    units = [
        _unit(
            int(trace.unit_op[step, lane, seat, unit]),
            int(trace.unit_item[step, lane, seat, unit]),
            int(trace.unit_amount[step, lane, seat, unit]),
        )
        for unit in range(int(trace.unit_count[step, lane, seat]))
    ]
    market = []
    for slot in range(int(trace.market_count[step, lane, seat])):
        order = _market(
            int(trace.market_op[step, lane, seat, slot]),
            int(trace.market_item[step, lane, seat, slot]),
            int(trace.market_amount[step, lane, seat, slot]),
        )
        if order is not None:
            market.append(order)
    return {"farmer": units[0] if units else ["PASS"], "hands": units[1:], "market": market}


def _load_action(data, seat: int) -> Action:
    return Action(*(np.asarray(data[f"seat{seat}_action_{name}"]) for name in Action._fields))


def _load_state(data, seat: int) -> State:
    return State(*(np.asarray(data[f"seat{seat}_state_{name}"]) for name in State._fields))


def _state_at(trajectory: State, step: int, lane: int) -> State:
    return State(*(field[step, lane] for field in trajectory))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace", type=Path,
        default=PROJECT_DIR / "artifacts" / "traces" / "m26_parity_trace_v1.npz",
    )
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT_DIR / "receipts" / "m26_official_parity_v1.json",
    )
    args = parser.parse_args()
    package_version = importlib.metadata.version("kaggle-environments")
    if package_version != "1.32.7":
        raise RuntimeError(f"expected kaggle-environments==1.32.7, got {package_version}")
    rows = []
    with np.load(args.trace, allow_pickle=False) as data:
        seeds = np.asarray(data["seeds"], dtype=np.int64)
        names = np.asarray(data["candidate_names"]).astype(str)
        for seat in (0, 1):
            actions = _load_action(data, seat)
            trajectory = _load_state(data, seat)
            expected_cash = np.asarray(data[f"seat{seat}_summary_final_bank"], dtype=np.int64)
            for lane, seed in enumerate(seeds.tolist()):
                env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": int(seed)}, debug=False)
                env.reset(2)
                assert_state_matches_frame(reset(int(seed)), canonical_frame(0, env.steps[0]))
                null = {"farmer": ["PASS"], "hands": [], "market": []}
                started = time.perf_counter()
                for step in range(719):
                    candidate = _decode_action(actions, step, lane, seat)
                    env.step([candidate, null] if seat == 0 else [null, candidate])
                    assert_state_matches_frame(_state_at(trajectory, step, lane), canonical_frame(step + 1, env.steps[-1]))
                final = env.steps[-1]
                official_cash = int(round(float(final[seat].reward)))
                rows.append({
                    "candidate": names[lane], "seed": int(seed), "seat": seat,
                    "frames_compared": 720,
                    "statuses": [str(value.status) for value in final],
                    "official_cash": official_cash, "jax_cash": int(expected_cash[lane]),
                    "cash_delta": official_cash - int(expected_cash[lane]),
                    "seconds": time.perf_counter() - started,
                })
    exact = all(row["cash_delta"] == 0 for row in rows)
    done = all(row["statuses"] == ["DONE", "DONE"] for row in rows)
    receipt = {
        "receipt_id": "M26_OFFICIAL_1327_REPRESENTATIVE_STEPWISE_PARITY_V1",
        "status": "PASS" if exact and done else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "official_package_version": package_version,
        "seeds": seeds.tolist(), "candidate_names": names.tolist(),
        "seat_swapped": True, "games": len(rows),
        "frames_compared_including_initial": len(rows) * 720,
        "exact_full_state_stepwise_parity": exact, "all_done": done,
        "max_abs_terminal_cash_delta": max(abs(row["cash_delta"]) for row in rows),
        "trace_sha256": _sha(args.trace),
        "state_scope": "step/status/reward; farms, tiles, units, inventories; market; shops; diagnostics",
        "boundary": "REPRESENTATIVE_BRANCHES_NOT_EXHAUSTIVE_GENOME_PARITY",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in receipt.items() if key != "rows"}, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
