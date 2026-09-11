"""Validate V2 against the exact source Replay seed and opponent action stream."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "experiments/hasegawa_jax_v2/tools"
for path in (ROOT / "experiments/hasegawa_jax_v2/src", ROOT / "gpu_sim/src", TOOLS):
    sys.path.insert(0, str(path))

from build_trace_bank_v2 import _encode, _hasegawa_seat  # noqa: E402
from hasegawa_jax_v2 import (  # noqa: E402
    hasegawa_step_with_external_v2,
    initialize_hasegawa_carry_v2,
    load_hasegawa_trace_bank_v2,
)
from kaggriculture_jax.constants import NUM_DAYS, SHOP_NAMES  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402


EVENT_DRAWS = 100
WEED_CHANCE = 0.005


def build_events(seed: int):
    weed = np.zeros((NUM_DAYS, EVENT_DRAWS), dtype=np.bool_)
    choice = np.zeros((NUM_DAYS, EVENT_DRAWS + 1), dtype=np.int8)
    for day in range(NUM_DAYS):
        rng = random.Random((seed * 1_000_003) ^ day)
        for consumed in range(EVENT_DRAWS + 1):
            clone = random.Random()
            clone.setstate(rng.getstate())
            choice[day, consumed] = SHOP_NAMES.index(clone.choice(SHOP_NAMES))
            if consumed < EVENT_DRAWS:
                weed[day, consumed] = rng.random() < WEED_CHANCE
    return Events(jnp.asarray(weed[None]), jnp.asarray(choice[None]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.replay.read_text(encoding="utf-8"))
    player = _hasegawa_seat(document)
    opponent = 1 - player
    opponent_trace = _encode(document, opponent)
    external = Action(
        *(jnp.asarray(opponent_trace[field])[:, None] for field in (
            "unit_op", "unit_item", "unit_amount", "unit_count",
            "market_op", "market_item", "market_amount", "market_count",
        ))
    )
    bank = load_hasegawa_trace_bank_v2(args.bank)
    seed = int(document["info"]["seed"])
    events = build_events(seed)
    tables = load_tables()
    state = jax.vmap(reset)(jnp.asarray([seed], dtype=jnp.int32))
    carry = initialize_hasegawa_carry_v2(1)

    def body(value, external_step):
        state, carry, invalid, resync, trimmed = value
        next_state, next_carry, diagnostics, _ = hasegawa_step_with_external_v2(
            state, carry, bank, Action(*external_step), player, events, tables
        )
        return (
            next_state,
            next_carry,
            invalid + diagnostics.invalid_unit_intent_count,
            resync + diagnostics.resync_unit_count,
            trimmed + diagnostics.market_trim_count,
        ), (
            next_state.money[:, player],
            diagnostics.invalid_unit_intent_count,
            diagnostics.resync_unit_count,
            diagnostics.market_trim_count,
            diagnostics.branch_id,
        )

    initial = (state, carry, jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.int32), jnp.zeros((1,), jnp.int32))
    final, scan_output = jax.jit(lambda value: jax.lax.scan(body, value, external))(initial)
    money, invalid_by_step, resync_by_step, trim_by_step, branch_by_step = scan_output
    jax.block_until_ready(money)
    final_state, final_carry, invalid, resync, trimmed = final
    source_reward = int(document["rewards"][player])
    source_money = np.asarray([
        int(document["steps"][step][player]["observation"]["farms"][player]["money"])
        for step in range(1, 720)
    ])
    jax_money = np.asarray(money[:, 0])
    mismatch_steps = np.flatnonzero(source_money != jax_money)
    invalid_steps = np.flatnonzero(np.asarray(invalid_by_step[:, 0]) > 0)
    trim_steps = np.flatnonzero(np.asarray(trim_by_step[:, 0]) > 0)
    receipt = {
        "schema": "kaggriculture.hasegawa_jax_v2.source_alignment",
        "replay": str(args.replay.resolve()), "seed": seed, "player": player,
        "source_reward": source_reward, "jax_final_cash": int(final_state.money[0, player]),
        "cash_exact_match": int(final_state.money[0, player]) == source_reward,
        "invalid_intents": int(invalid[0]), "resyncs": int(resync[0]),
        "market_trims": int(trimmed[0]), "hard_counters": int(final_carry.hard_counter_total[0]),
        "first_cash_mismatch_action_step": int(mismatch_steps[0]) if len(mismatch_steps) else None,
        "first_invalid_action_step": int(invalid_steps[0]) if len(invalid_steps) else None,
        "first_market_trim_action_step": int(trim_steps[0]) if len(trim_steps) else None,
        "first_20_nonzero_invalid_steps": invalid_steps[:20].astype(int).tolist(),
        "branch_at_step72": int(branch_by_step[72, 0]),
        "milestones": {
            str(step): {
                "source": int(document["steps"][step][player]["observation"]["farms"][player]["money"]),
                "jax": int(money[step - 1, 0]),
            }
            for step in (1, 24, 72, 144, 264, 384, 600, 719)
        },
    }
    receipt["status"] = "PASS" if receipt["cash_exact_match"] and not receipt["invalid_intents"] and not receipt["resyncs"] and not receipt["hard_counters"] else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
