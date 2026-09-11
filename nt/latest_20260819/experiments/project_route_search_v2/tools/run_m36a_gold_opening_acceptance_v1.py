"""Official-Replay acceptance for the M3.6A same-turn commitment bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
for source_dir in (
    PROJECT_DIR / "src",
    REPO_ROOT / "gpu_sim" / "src",
    REPO_ROOT / "experiments" / "strategic_v5" / "src",
):
    sys.path.insert(0, str(source_dir))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from kaggriculture_jax.constants import (  # noqa: E402
    ANIMALS,
    CROPS,
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    PRODUCTS,
    SHED_ITEMS,
    MarketOp,
    TileKind,
    UnitOp,
)
from kaggriculture_jax.policy import combine_player_actions  # noqa: E402
from kaggriculture_jax.simulator import batched_step_sync  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Events  # noqa: E402
from project_route_search_v2.m36_calendar import (  # noqa: E402
    kawashigi_opening_calendar_v3,
    validate_route_calendar_v3,
)
from project_route_search_v2.m36_schema import (  # noqa: E402
    M36IntentFailureV3,
    M36IntentStatusV3,
)
from project_route_search_v2.m36_transaction import (  # noqa: E402
    m36_player_action_dict_v3,
    m36a_policy_step_v3,
    reconcile_commitment_bundle_v3,
)


UNIT_IDS = {name: int(value) for name, value in UnitOp.__members__.items()}
MARKET_IDS = {name: int(value) for name, value in MarketOp.__members__.items()}
PRODUCT_IDS = {name: index for index, name in enumerate(PRODUCTS)}
CROP_IDS = {name: index for index, name in enumerate(CROPS)}
ANIMAL_IDS = {name: len(PRODUCTS) + index for index, name in enumerate(ANIMALS)}
SHED_IDS = {name: index for index, name in enumerate(SHED_ITEMS)}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _unit(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return UNIT_IDS["PASS"], -1, 1
    op = UNIT_IDS.get(str(raw[0]), UNIT_IDS["PASS"])
    item = -1
    amount = 1
    if op == UNIT_IDS["PLANT"] and len(raw) >= 2:
        item = CROP_IDS.get(str(raw[1]), -1)
    elif op in (UNIT_IDS["PICKUP"], UNIT_IDS["PLACE"]) and len(raw) >= 2:
        item = SHED_IDS.get(str(raw[1]), -1)
        amount = int(raw[2]) if len(raw) >= 3 else 1
    return op, item, amount


def _market(raw) -> tuple[int, int, int]:
    if not isinstance(raw, list) or not raw:
        return MARKET_IDS["NONE"], -1, 0
    op = MARKET_IDS.get(str(raw[0]), MARKET_IDS["NONE"])
    if op in (MARKET_IDS["HIRE"], MARKET_IDS["BUY_LAND"]):
        return op, -1, 0
    if len(raw) < 3:
        return MARKET_IDS["NONE"], -1, 0
    if op == MARKET_IDS["BUY_SEED"]:
        item = CROP_IDS[str(raw[1])]
    elif op == MARKET_IDS["BUY_ANIMAL"]:
        item = ANIMAL_IDS[str(raw[1])]
    else:
        item = PRODUCT_IDS[str(raw[1])]
    return op, item, int(raw[2])


def _encode_player_action(raw: dict) -> dict:
    units = [raw.get("farmer", ["PASS"]), *(raw.get("hands", []) or [])]
    unit_op = np.full((1, MAX_UNITS), UNIT_IDS["PASS"], dtype=np.int8)
    unit_item = np.full((1, MAX_UNITS), -1, dtype=np.int8)
    unit_amount = np.ones((1, MAX_UNITS), dtype=np.int32)
    for index, value in enumerate(units[:MAX_UNITS]):
        unit_op[0, index], unit_item[0, index], unit_amount[0, index] = _unit(value)
    orders = raw.get("market", []) or []
    market_op = np.full((1, MAX_MARKET_ORDERS), MARKET_IDS["NONE"], dtype=np.int8)
    market_item = np.full((1, MAX_MARKET_ORDERS), -1, dtype=np.int8)
    market_amount = np.zeros((1, MAX_MARKET_ORDERS), dtype=np.int32)
    for index, value in enumerate(orders[:MAX_MARKET_ORDERS]):
        market_op[0, index], market_item[0, index], market_amount[0, index] = _market(value)
    return {
        "unit_op": jnp.asarray(unit_op),
        "unit_item": jnp.asarray(unit_item),
        "unit_amount": jnp.asarray(unit_amount),
        "unit_count": jnp.asarray((min(len(units), MAX_UNITS),), dtype=jnp.int8),
        "market_op": jnp.asarray(market_op),
        "market_item": jnp.asarray(market_item),
        "market_amount": jnp.asarray(market_amount),
        "market_count": jnp.asarray(
            (min(len(orders), MAX_MARKET_ORDERS),), dtype=jnp.int8
        ),
    }


def _official_post_step(data: dict, seat: int) -> dict:
    observation = data["steps"][1][seat]["observation"]
    farm = observation["farms"][seat]
    private = observation["private"]
    pasture = sum(
        isinstance(tile, dict) and tile.get("kind") == "PASTURE"
        for row in farm["tiles"]
        for tile in row
    )
    return {
        "cash": int(farm["money"]),
        "hand": len(farm["hands"]),
        "pasture": int(pasture),
        "cow": int(private["shed"]["COW"]),
        "sheep": int(private["shed"]["SHEEP"]),
        "wheat_seed": int(private["seeds"]["WHEAT"]),
        "melon_seed": int(private["seeds"]["MELON"]),
        "shed_wheat": int(private["shed"]["WHEAT"]),
    }


def _jax_post_step(state, seat: int) -> dict:
    pasture = np.sum(np.asarray(state.tile_kind[0, seat]) == TileKind.PASTURE)
    return {
        "cash": int(state.money[0, seat]),
        "hand": int(state.hires_today[0, seat]),
        "pasture": int(pasture),
        "cow": int(state.shed[0, seat, 10]),
        "sheep": int(state.shed[0, seat, 11]),
        "wheat_seed": int(state.seeds[0, seat, 0]),
        "melon_seed": int(state.seeds[0, seat, 4]),
        "shed_wheat": int(state.shed[0, seat, 0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        type=Path,
        default=REPO_ROOT
        / "replay"
        / "gold_top3_recent12_2026-08-18"
        / "kawashigi_current_55540317"
        / "episodes"
        / "94051618.json",
    )
    parser.add_argument("--seat", type=int, default=1)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "receipts" / "m36a_gold_opening_acceptance_v1.json",
    )
    args = parser.parse_args()
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")
    data = json.loads(args.replay.read_text(encoding="utf-8"))
    if data.get("module_version") != "1.32.7":
        raise ValueError("M3.6A acceptance requires official 1.32.7")
    seat = int(args.seat)
    opponent = 1 - seat
    seed = int(data["info"]["seed"])
    signed_seed = np.asarray(seed, dtype=np.uint32).view(np.int32).item()
    states = jax.vmap(reset)(jnp.asarray((signed_seed,), dtype=jnp.int32))
    calendar = kawashigi_opening_calendar_v3(1)
    validation_errors = validate_route_calendar_v3(calendar)
    decide = jax.jit(lambda s, c: m36a_policy_step_v3(s, c, seat))
    started = time.perf_counter()
    decision, projected = decide(states, calendar)
    jax.block_until_ready(decision)
    compile_and_decide_seconds = time.perf_counter() - started
    opponent_action = _encode_player_action(data["steps"][1][opponent]["action"])
    player_action = m36_player_action_dict_v3(decision)
    joint = (
        combine_player_actions(player_action, opponent_action)
        if seat == 0
        else combine_player_actions(opponent_action, player_action)
    )
    events = Events(
        jnp.zeros((1, 30, 100), dtype=jnp.bool_),
        jnp.zeros((1, 30, 101), dtype=jnp.int8),
    )
    next_states = jax.jit(batched_step_sync)(states, joint, events, load_tables())
    reconciled, effect_diagnostics = reconcile_commitment_bundle_v3(
        projected, next_states, decision.bundle, seat
    )
    decision, next_states, reconciled, effect_diagnostics = jax.device_get(
        (decision, next_states, reconciled, effect_diagnostics)
    )
    replay_player = _encode_player_action(data["steps"][1][seat]["action"])
    intent_fields = (
        "unit_op",
        "unit_item",
        "unit_amount",
        "unit_count",
        "market_op",
        "market_item",
        "market_amount",
        "market_count",
    )
    intent_match = all(
        np.array_equal(
            np.asarray(getattr(decision, name)), np.asarray(replay_player[name])
        )
        for name in intent_fields
    )
    official_post = _official_post_step(data, seat)
    jax_post = _jax_post_step(next_states, seat)
    feed_slots = np.where(
        (np.asarray(reconciled.market_op[0]) == MarketOp.BUY_PRODUCT)
        & (np.asarray(reconciled.market_item[0]) == 0)
    )[0]
    feed_slot = int(feed_slots[0]) if len(feed_slots) == 1 else -1
    requested = int(reconciled.requested_quantity[0, feed_slot])
    filled = int(reconciled.filled_quantity[0, feed_slot])
    post_quantity = int(reconciled.post_step_quantity[0, feed_slot])
    feed_status = int(reconciled.intent_status[0, feed_slot])
    feed_failure = int(reconciled.failure_code[0, feed_slot])
    checks = {
        "calendar_validation_pass": not validation_errors,
        "controller_generated_intent_matches_replay": intent_match,
        "market_order_count_is_10": int(decision.market_count[0]) == 10,
        "requested_filled_post_wheat_is_6_5_5": (
            requested == 6 and filled == 5 and post_quantity == 5
        ),
        "partial_fill_classified_as_insufficient_cash": (
            feed_status == M36IntentStatusV3.PARTIAL
            and feed_failure == M36IntentFailureV3.INSUFFICIENT_CASH
        ),
        "post_step_exact_official_match": jax_post == official_post,
        "planning_hard_error_zero": int(decision.diagnostics.hard_error_count[0])
        == 0,
        "effect_hard_error_zero": int(effect_diagnostics.hard_error_count[0]) == 0,
        "simulator_caps_zero": (
            int(next_states.hand_cap_hits[0, seat]) == 0
            and int(next_states.market_loop_cap_hits[0]) == 0
            and int(next_states.price_lut_oob[0]) == 0
        ),
    }
    receipt = {
        "receipt_id": "M36A_GOLD_OPENING_ACCEPTANCE_V1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "compile_and_decide_seconds": compile_and_decide_seconds,
        "official_version": data["module_version"],
        "replay": str(args.replay.resolve().relative_to(REPO_ROOT.resolve())).replace(
            "\\", "/"
        ),
        "replay_sha256": _sha256(args.replay),
        "episode_id": int(data["info"]["EpisodeId"]),
        "seed": seed,
        "gold_seat": seat,
        "gold_player": data["info"]["Agents"][seat]["Name"],
        "opponent": data["info"]["Agents"][opponent]["Name"],
        "calendar_validation_errors": validation_errors,
        "checks": checks,
        "wheat_market_effect": {
            "requested_quantity": requested,
            "filled_quantity": filled,
            "post_step_quantity": post_quantity,
            "intent_status": feed_status,
            "failure_code": feed_failure,
        },
        "official_post_step": official_post,
        "jax_post_step": jax_post,
        "bundle": {
            "market_count": int(reconciled.market_count[0]),
            "market_op": np.asarray(reconciled.market_op[0]).astype(int).tolist(),
            "market_item": np.asarray(reconciled.market_item[0]).astype(int).tolist(),
            "requested_quantity": np.asarray(
                reconciled.requested_quantity[0]
            ).astype(int).tolist(),
            "filled_quantity": np.asarray(reconciled.filled_quantity[0])
            .astype(int)
            .tolist(),
            "post_step_quantity": np.asarray(reconciled.post_step_quantity[0])
            .astype(int)
            .tolist(),
            "failure_code": np.asarray(reconciled.failure_code[0])
            .astype(int)
            .tolist(),
            "pending_structures_after_unit_projection": np.asarray(
                reconciled.pending_structures[0]
            )
            .astype(int)
            .tolist(),
            "future_place_plan": np.asarray(reconciled.future_place_plan[0])
            .astype(int)
            .tolist(),
        },
        "truth_boundary": (
            "The M3.6 controller receives only high-level day-0 business targets. "
            "The opponent action and post-step reference come from official Replay "
            "94051618; the controller does not replay the gold raw action."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
