"""Run a full passive-opponent GPU/CPU acceptance rollout for Hasegawa V2."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT / "experiments/hasegawa_jax_v2/src", ROOT / "gpu_sim/src"):
    sys.path.insert(0, str(path))

from hasegawa_jax_v2 import (  # noqa: E402
    hasegawa_step_with_external_v2,
    initialize_hasegawa_carry_v2,
    load_hasegawa_trace_bank_v2,
)
from kaggriculture_jax.constants import (  # noqa: E402
    MAX_MARKET_ORDERS,
    MAX_UNITS,
    MarketOp,
    UnitOp,
)
from kaggriculture_jax.state import load_event_bank, load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402


def null_action(batch: int):
    return Action(
        jnp.full((batch, MAX_UNITS), UnitOp.PASS, dtype=jnp.int8),
        jnp.full((batch, MAX_UNITS), -1, dtype=jnp.int8),
        jnp.ones((batch, MAX_UNITS), dtype=jnp.int32),
        jnp.ones((batch,), dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), MarketOp.NONE, dtype=jnp.int8),
        jnp.full((batch, MAX_MARKET_ORDERS), -1, dtype=jnp.int8),
        jnp.zeros((batch, MAX_MARKET_ORDERS), dtype=jnp.int32),
        jnp.zeros((batch,), dtype=jnp.int8),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument(
        "--bank",
        type=Path,
        default=ROOT / "experiments/hasegawa_jax_v2/artifacts/hasegawa_trace_bank_v2.npz",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "experiments/hasegawa_jax_v2/receipts/passive_full_v2.json",
    )
    args = parser.parse_args()
    batch = args.batch
    bank = load_hasegawa_trace_bank_v2(args.bank)
    tables = load_tables()
    _, source_events = load_event_bank()
    events = Events(source_events.weed_spawn[:batch], source_events.shop_choice[:batch])
    states = jax.vmap(reset)(jnp.arange(batch, dtype=jnp.int32) + 310_000)
    carry = initialize_hasegawa_carry_v2(batch)
    external = null_action(batch)

    def rollout(initial_state, initial_carry):
        def body(value, _):
            state, controller = value
            next_state, next_controller, _, _ = hasegawa_step_with_external_v2(
                state, controller, bank, external, 0, events, tables
            )
            return (next_state, next_controller), None

        return jax.lax.scan(body, (initial_state, initial_carry), None, length=719)[0]

    compiled = jax.jit(rollout)
    start = time.perf_counter()
    final_state, final_carry = compiled(states, carry)
    jax.block_until_ready(final_state.money)
    compile_and_first = time.perf_counter() - start
    start = time.perf_counter()
    second_state, second_carry = compiled(states, carry)
    jax.block_until_ready(second_state.money)
    steady = time.perf_counter() - start

    cash = np.asarray(second_state.money[:, 0], dtype=np.int64)
    branches = np.asarray(second_carry.branch_id, dtype=np.int64)
    receipt = {
        "schema": "kaggriculture.hasegawa_jax_v2.passive_full",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(jax.devices()[0]),
        "batch": batch,
        "steps_per_game": 719,
        "compile_and_first_seconds": compile_and_first,
        "steady_seconds": steady,
        "transitions_per_second": batch * 719 / steady,
        "done_count": int(np.sum(np.asarray(second_state.done))),
        "final_step_min": int(np.min(np.asarray(second_state.step))),
        "final_step_max": int(np.max(np.asarray(second_state.step))),
        "cash": {
            "min": int(np.min(cash)),
            "p10": float(np.percentile(cash, 10)),
            "median": float(np.median(cash)),
            "mean": float(np.mean(cash)),
            "p90": float(np.percentile(cash, 90)),
            "max": int(np.max(cash)),
        },
        "branch_histogram": {str(index): int(np.sum(branches == index)) for index in range(9)},
        "invalid_intent_total": int(np.sum(np.asarray(second_carry.invalid_intent_total))),
        "resync_total": int(np.sum(np.asarray(second_carry.resync_total))),
        "market_trim_total": int(np.sum(np.asarray(second_carry.market_trim_total))),
        "hard_counter_total": int(np.sum(np.asarray(second_carry.hard_counter_total))),
        "acceptance": {
            "all_finished": bool(np.all(np.asarray(second_state.done))),
            "hard_counter_zero": int(np.sum(np.asarray(second_carry.hard_counter_total))) == 0,
            "economic_milestone_pass": float(np.median(cash)) >= 90_000,
        },
    }
    receipt["status"] = "PASS" if all(receipt["acceptance"].values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
