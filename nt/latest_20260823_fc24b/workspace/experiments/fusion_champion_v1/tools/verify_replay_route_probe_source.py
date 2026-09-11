#!/usr/bin/env python3
"""Verify two encoded Replay routes on their original official event seed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
for path in (
    ROOT / "experiments/expert_business_agent_v2/tools",
    ROOT / "experiments/route_playbook_v1/src",
):
    sys.path.insert(0, str(path))

import run_all_exact_jax_round_robin as rr  # noqa: E402
from kaggriculture_jax.state import load_tables, reset  # noqa: E402
from kaggriculture_jax.types import Action, Events  # noqa: E402
from route_playbook_v1.event_bank import build_events_v1  # noqa: E402


FIELDS = Action._fields


def load(path: Path) -> dict[str, jax.Array]:
    with np.load(path, allow_pickle=False) as data:
        return {field: jnp.asarray(data[field]) for field in FIELDS}


def action_at(bank: dict[str, jax.Array], step: int) -> Action:
    return Action(*(bank[field][:, step] for field in FIELDS))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-bank", type=Path, required=True)
    parser.add_argument("--right-bank", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-left", type=int, required=True)
    parser.add_argument("--expected-right", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"GPU required, got {jax.default_backend()}")

    left = load(args.left_bank.resolve())
    right = load(args.right_bank.resolve())
    tables = load_tables()
    simulator = rr.make_simulator_step(tables)
    seed = np.asarray([args.seed], dtype=np.int32)
    weed, shops = build_events_v1(seed.tolist())
    events = Events(jnp.asarray(weed), jnp.asarray(shops))
    states = jax.vmap(reset)(jnp.asarray(seed))
    for step in range(719):
        states = simulator(states, action_at(left, step), action_at(right, step), events)
    terminal = jax.device_get(states)
    observed = [int(value) for value in np.asarray(terminal.money[0]).tolist()]
    expected = [args.expected_left, args.expected_right]
    payload = {
        "schema": "kaggriculture.fusion_champion.replay-route-source-verification.v1",
        "status": "PASS" if observed == expected else "FAIL",
        "seed": args.seed,
        "expected_money": expected,
        "observed_money": observed,
        "exact_terminal_money": observed == expected,
        "all_done": bool(np.all(np.asarray(terminal.done))),
        "hand_cap_hits": int(np.sum(np.asarray(terminal.hand_cap_hits))),
        "market_loop_cap_hits": int(np.sum(np.asarray(terminal.market_loop_cap_hits))),
        "price_lut_oob": int(np.sum(np.asarray(terminal.price_lut_oob))),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
