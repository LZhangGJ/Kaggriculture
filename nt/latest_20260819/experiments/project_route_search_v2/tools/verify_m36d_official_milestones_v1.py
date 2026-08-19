"""Replay an M3.6D JAX controller trace in official Python 1.32.7."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
GPU_SIM = ROOT / "gpu_sim"
for source in (GPU_SIM / "src", GPU_SIM / "tests", GPU_SIM / "tools"):
    sys.path.insert(0, str(source))

from kaggle_environments import make  # noqa: E402
from generate_reference_traces import canonical_frame  # noqa: E402
from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    FLAG_FERTILIZER_AVAILABLE,
    NUM_PRODUCTS,
    PRODUCTS,
    SHED_ITEMS,
    TileKind,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.state import reset  # noqa: E402
from kaggriculture_jax.types import Action, State  # noqa: E402
from reference_assertions import assert_state_matches_frame  # noqa: E402


REPLAY = (
    ROOT
    / "replay"
    / "gold_top20_latest_2026-08-18_082723"
    / "latest_per_gold"
    / "episode-94051618-replay.json"
)
TRACE = PROJECT / "artifacts" / "traces" / "m36d_controller_trace_v1.npz"
GENERATION = PROJECT / "receipts" / "m36d_controller_trace_generation_v1.json"
MILESTONE_STEPS = (1, 6 * 24, 11 * 24, 12 * 24, 20 * 24, 25 * 24, 29 * 24, 719)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=REPLAY)
    parser.add_argument("--trace", type=Path, default=TRACE)
    parser.add_argument("--generation", type=Path, default=GENERATION)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "receipts" / "m36d_official_milestone_parity_v1.json",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def unit_action(op: int, item: int, amount: int) -> list[object]:
    try:
        name = UnitOp(op).name
    except ValueError:
        name = "PASS"
    if op == UnitOp.PLANT and 0 <= item < len(CROPS):
        return [name, CROPS[item]]
    if op in (UnitOp.PICKUP, UnitOp.PLACE) and 0 <= item < len(SHED_ITEMS):
        return [name, SHED_ITEMS[item], int(amount)]
    return [name]


def market_action(op: int, item: int, amount: int) -> list[object] | None:
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


def decode_action(trace: Action, step: int, lane: int = 0) -> dict[str, object]:
    units = [
        unit_action(
            int(trace.unit_op[step, lane, unit]),
            int(trace.unit_item[step, lane, unit]),
            int(trace.unit_amount[step, lane, unit]),
        )
        for unit in range(int(trace.unit_count[step, lane]))
    ]
    market = []
    for slot in range(int(trace.market_count[step, lane])):
        order = market_action(
            int(trace.market_op[step, lane, slot]),
            int(trace.market_item[step, lane, slot]),
            int(trace.market_amount[step, lane, slot]),
        )
        if order is not None:
            market.append(order)
    return {
        "farmer": units[0] if units else ["PASS"],
        "hands": units[1:],
        "market": market,
    }


def load_action(data) -> Action:
    return Action(*(np.asarray(data[f"action_{name}"]) for name in Action._fields))


def load_state(data) -> State:
    return State(*(np.asarray(data[f"state_{name}"]) for name in State._fields))


def state_at(trajectory: State, step: int, lane: int = 0) -> State:
    return State(*(field[step, lane] for field in trajectory))


def official_metrics(frame: dict, player: int) -> dict[str, object]:
    farm = frame["farms"][player]
    private = frame["private"][player]
    crops = Counter()
    crop_yield = Counter()
    animals = Counter()
    animal_held = Counter()
    fertilizer_available = 0
    fertilized_crop_tiles = 0
    for row in farm["tiles"]:
        for tile in row:
            if not isinstance(tile, dict):
                continue
            if tile.get("kind") == "PLANT":
                name = str(tile.get("crop"))
                crops[name] += 1
                crop_yield[name] += int(tile.get("yield_units", 0))
                if int(tile.get("fertilized_until_day", -1)) >= int(frame["day"]):
                    fertilized_crop_tiles += 1
            if tile.get("animal"):
                name = str(tile["animal"])
                animals[name] += 1
                animal_held[name] += int(tile.get("yield_units", 0))
                fertilizer_available += int(bool(tile.get("fertilizer_available", False)))
    inventories = private.get("inventories", [])
    return {
        "state_step": int(frame["step"]),
        "cash": int(farm["money"]),
        "units": 1 + len(farm.get("hands", [])),
        "hands": len(farm.get("hands", [])),
        "land": len(farm.get("unlocked_quadrants", [])),
        "crop_tiles": [int(crops[name]) for name in CROPS],
        "crop_harvestable_units": [int(crop_yield[name]) for name in CROPS],
        "animals": [int(animals[name]) for name in ANIMALS],
        "animal_held_products": [int(animal_held[name]) for name in ANIMALS],
        "shed": [int(private["shed"].get(name, 0)) for name in SHED_ITEMS],
        "fertilizer": {
            "shed": int(private["shed"].get("FERTILIZER", 0)),
            "carried": int(
                sum(inventory.get("FERTILIZER", 0) for inventory in inventories)
            ),
            "available_animal_tiles": fertilizer_available,
            "fertilized_crop_tiles": fertilized_crop_tiles,
        },
    }


def jax_metrics(state: State, player: int) -> dict[str, object]:
    crop = np.asarray(state.tile_crop[player])
    animal = np.asarray(state.tile_animal[player])
    kind = np.asarray(state.tile_kind[player])
    tile_yield = np.asarray(state.tile_yield[player])
    flags = np.asarray(state.tile_flags[player])
    day = int(state.step) // 24
    active = np.asarray(state.unit_active[player])
    inventories = np.asarray(state.unit_inventory[player])
    return {
        "state_step": int(state.step),
        "cash": int(state.money[player]),
        "units": int(active.sum()),
        "hands": int(active.sum()) - 1,
        "land": int(state.unlocked_count[player]),
        "crop_tiles": [
            int(((kind == TileKind.PLANT) & (crop == item)).sum())
            for item in range(len(CROPS))
        ],
        "crop_harvestable_units": [
            int(tile_yield[(kind == TileKind.PLANT) & (crop == item)].sum())
            for item in range(len(CROPS))
        ],
        "animals": [int((animal == item).sum()) for item in range(len(ANIMALS))],
        "animal_held_products": [
            int(tile_yield[animal == item].sum()) for item in range(len(ANIMALS))
        ],
        "shed": np.asarray(state.shed[player]).astype(int).tolist(),
        "fertilizer": {
            "shed": int(state.shed[player, 8]),
            "carried": int(inventories[:, 8].sum()),
            "available_animal_tiles": int(
                ((animal >= 0) & ((flags & FLAG_FERTILIZER_AVAILABLE) != 0)).sum()
            ),
            "fertilized_crop_tiles": int(
                (
                    (kind == TileKind.PLANT)
                    & (np.asarray(state.tile_fertilized_until[player]) >= day)
                ).sum()
            ),
        },
    }


def replay_metrics(replay: dict, player: int, step: int) -> dict[str, object]:
    row = replay["steps"][step]
    observation = row[player]["observation"]
    frame = {
        "step": step,
        "day": int(observation["day"]),
        "farms": observation["farms"],
        "private": [None, None],
    }
    frame["private"][player] = observation["private"]
    return official_metrics(frame, player)


def vector_delta(left: list[int], right: list[int]) -> list[int]:
    return (np.asarray(left, dtype=np.int64) - np.asarray(right, dtype=np.int64)).tolist()


def difference_reasons(candidate: dict, gold: dict) -> list[str]:
    reasons = []
    if candidate["cash"] != gold["cash"]:
        reasons.append(f"cash_delta={candidate['cash'] - gold['cash']:+d}")
    if candidate["land"] != gold["land"]:
        reasons.append(f"land_delta={candidate['land'] - gold['land']:+d}")
    if candidate["hands"] != gold["hands"]:
        reasons.append(f"hands_delta={candidate['hands'] - gold['hands']:+d}")
    crop_delta = vector_delta(candidate["crop_tiles"], gold["crop_tiles"])
    if any(crop_delta):
        reasons.append(f"crop_tile_delta={crop_delta}")
    animal_delta = vector_delta(candidate["animals"], gold["animals"])
    if any(animal_delta):
        reasons.append(f"animal_delta={animal_delta}")
    shed_delta = vector_delta(candidate["shed"], gold["shed"])
    if any(shed_delta):
        reasons.append(f"shed_delta={shed_delta}")
    return reasons or ["recorded_fields_equal"]


def main() -> None:
    args = parse_args()
    package_version = importlib.metadata.version("kaggle-environments")
    if package_version != "1.32.7":
        raise RuntimeError(
            f"expected kaggle-environments==1.32.7, got {package_version}"
        )
    generation = json.loads(args.generation.read_text(encoding="utf-8"))
    replay = json.loads(args.replay.read_text(encoding="utf-8"))
    with np.load(args.trace, allow_pickle=False) as data:
        seed = int(np.asarray(data["seed"])[0])
        player = int(np.asarray(data["player"])[0])
        milestones = tuple(np.asarray(data["milestone_steps"]).astype(int).tolist())
        backlog = json.loads(str(np.asarray(data["milestone_backlog_json"])[0]))
        actions = load_action(data)
        trajectory = load_state(data)

    if milestones != MILESTONE_STEPS:
        raise ValueError(f"unexpected milestones: {milestones}")
    if seed != int(replay["info"]["seed"]):
        raise ValueError("trace and gold Replay seeds differ")
    if generation.get("trace_sha256") != sha256(args.trace):
        raise ValueError("trace hash does not match generation receipt")
    if generation.get("hard_error_count") != 0:
        raise ValueError("generation receipt contains controller hard errors")

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    env.reset(2)
    initial_frame = canonical_frame(0, env.steps[0])
    assert_state_matches_frame(reset(seed), initial_frame)
    null = {"farmer": ["PASS"], "hands": [], "market": []}
    exact_frames = 1
    milestone_rows = []
    started = time.perf_counter()
    for action_step in range(719):
        candidate = decode_action(actions, action_step)
        env.step([candidate, null] if player == 0 else [null, candidate])
        frame = canonical_frame(action_step + 1, env.steps[-1])
        expected = state_at(trajectory, action_step)
        assert_state_matches_frame(expected, frame)
        exact_frames += 1
        state_step = action_step + 1
        if state_step in milestones:
            official = official_metrics(frame, player)
            jax_value = jax_metrics(expected, player)
            if official != jax_value:
                raise AssertionError(
                    f"milestone metric mismatch at state step {state_step}"
                )
            gold = replay_metrics(replay, player, state_step)
            milestone_rows.append(
                {
                    "state_step": state_step,
                    "official_1327": official,
                    "jax": jax_value,
                    "exact": True,
                    "controller_backlog": backlog[str(state_step)],
                    "gold_replay_official": gold,
                    "candidate_minus_gold": {
                        "cash": int(official["cash"] - gold["cash"]),
                        "hands": int(official["hands"] - gold["hands"]),
                        "land": int(official["land"] - gold["land"]),
                        "crop_tiles": vector_delta(
                            official["crop_tiles"], gold["crop_tiles"]
                        ),
                        "animals": vector_delta(
                            official["animals"], gold["animals"]
                        ),
                        "shed": vector_delta(official["shed"], gold["shed"]),
                    },
                    "difference_reasons": difference_reasons(official, gold),
                }
            )

    elapsed = time.perf_counter() - started
    final = env.steps[-1]
    statuses = [str(value.status) for value in final]
    official_final_bank = int(round(float(final[player].reward)))
    jax_final_bank = int(state_at(trajectory, 718).money[player])
    gate_72 = official_final_bank >= 72_261
    gate_81 = official_final_bank >= 81_294
    authorized = 10_000 if gate_81 else (1_000 if gate_72 else 0)
    exact = exact_frames == 720 and official_final_bank == jax_final_bank
    done = statuses == ["DONE", "DONE"]
    receipt = {
        "receipt_id": "M36D_OFFICIAL_1327_MILESTONE_PARITY_V1",
        "status": "PASS" if exact and done else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "official_package_version": package_version,
        "episode_id": int(replay["info"]["EpisodeId"]),
        "seed": seed,
        "player": player,
        "opponent": "NullOpponent",
        "frames_compared_including_initial": exact_frames,
        "exact_full_state_stepwise_parity": exact,
        "all_done": done,
        "statuses": statuses,
        "official_final_bank": official_final_bank,
        "jax_final_bank": jax_final_bank,
        "terminal_cash_delta": official_final_bank - jax_final_bank,
        "official_replay_seconds": elapsed,
        "milestone_schema": {
            "crop_tiles": list(CROPS),
            "crop_harvestable_units": list(CROPS),
            "animals": list(ANIMALS),
            "animal_held_products": list(ANIMALS),
            "shed": list(SHED_ITEMS),
            "backlog": "controller-internal; official simulator has no backlog field",
        },
        "milestones": milestone_rows,
        "economic_search_gates": {
            "gate_72261": gate_72,
            "gate_81294": gate_81,
            "authorized_neighborhood_candidates": authorized,
            "freeze_recommendation": (
                "ELIGIBLE_FOR_10000_REVIEW"
                if gate_81
                else (
                    "ELIGIBLE_FOR_1000_REVIEW"
                    if gate_72
                    else "DO_NOT_FREEZE_OR_START_SEARCH"
                )
            ),
            "candidate_160000_authorized": False,
        },
        "trace": args.trace.resolve().relative_to(ROOT.resolve()).as_posix(),
        "trace_sha256": sha256(args.trace),
        "generation_receipt_sha256": sha256(args.generation),
        "gold_replay_sha256": sha256(args.replay),
        "truth_boundary": (
            "Official/JAX equality is exact for the controller action trace, same seed, "
            "seat and NullOpponent. Gold Replay rows are official observations from the "
            "same seed but a different opponent/action trajectory, so their deltas are "
            "semantic capacity diagnostics rather than a causal policy comparison."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {key: value for key, value in receipt.items() if key != "milestones"},
            indent=2,
            ensure_ascii=False,
        ),
        flush=True,
    )
    if receipt["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
