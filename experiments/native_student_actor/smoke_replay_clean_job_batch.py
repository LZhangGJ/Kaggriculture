#!/usr/bin/env python3
"""Step/terminal parity gate for the JobBatch replay allowlist."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np

from experiments.native_student_actor.smoke_arbitrary_job_batch import (
    META, ROOT, ROLLOUT, THOMAS, WEIGHTS, settings,
)
from experiments.student_action_event_agent import production
from experiments.train_student_action_event_rl_v3 import (
    NATIVE_JOB_REPLAY_FAMILIES,
)
from fast_kaggriculture import Config, FastEnv, NativeReplayOpponent
from meta_agent.src.native_teammate_executor import NativeTeammateBundle


def encode(action: dict) -> list[int]:
    def atom(value):
        if not value or value[0] not in production.policy._OPS:
            return 0, -1, 1
        op = production.policy._OPS.index(value[0])
        item = (production.policy._IDS.get(value[1], -1)
                if len(value) > 1 and isinstance(value[1], str) else -1)
        quantity = (value[2] if item >= 0 and len(value) > 2 else
                    value[1] if item < 0 and len(value) > 1 else 1)
        return op, item, int(quantity)

    units = [action.get("farmer", ["PASS"]), *action.get("hands", [])]
    market = list(action.get("market", []))
    result = [len(units), len(market)]
    for value in (*units, *market):
        result.extend(atom(value))
    return result


def decode(values: list[int]) -> dict:
    units, market = values[:2]
    at = 2

    def atom():
        nonlocal at
        op, item, quantity = values[at:at + 3]
        at += 3
        result = [production.policy._OPS[op]]
        if item >= 0:
            result.append(production.policy._ITEMS[item])
        if quantity != 1 or item >= 0:
            result.append(quantity)
        return result

    actors = [atom() for _ in range(units)]
    orders = [atom() for _ in range(market)]
    if at != len(values) or not actors:
        raise RuntimeError("invalid JobBatch action trace")
    return {"farmer": actors[0], "hands": actors[1:], "market": orders}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=(
        ROOT / "work/agent-student-actor-owned-v3.so"))
    parser.add_argument("--weights", type=Path, default=WEIGHTS)
    parser.add_argument("--seed-start", type=int, default=2632300000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, default=(
        ROOT / "work/native-student-rollout/replay-clean-job-parity-b4.json"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    sys.path.insert(0, str(ROLLOUT / "build"))
    native = importlib.import_module("_paused_plan")
    bundle = NativeTeammateBundle(
        ROOT / "agent/teammate_base.py",
        ROOT / "agent/route_actions.json.zlib",
        ROOT / "agent/route_library.json")
    metadata = json.loads((ROOT / "agent/route_library.json").read_text())
    by_family = {row["family"]: row for row in metadata["opponent_routes"]}
    cases = []
    for offset, family in enumerate(NATIVE_JOB_REPLAY_FAMILIES):
        row = by_family[family]
        route = bundle.index(family)
        if (int(row.get("source_execution_hard_failures") or 0) != 0 or
                bundle.route_ids[route] != row["route_id"]):
            raise RuntimeError(f"unclean replay allowlist entry: {family}")
        for seat in (0, 1):
            cases.append({
                "family": family, "route_id": row["route_id"],
                "route": route, "seed": args.seed_start + offset,
                "seat": seat,
            })
    deployment_routes = [bundle.index(name) for name in
                         ("G275", "G195", "G024", "G316", "G267")]
    batch = native.JobBatch(
        str(args.binary), bundle.executor,
        [row["seed"] for row in cases], [row["seat"] for row in cases],
        [3] * len(cases), [row["route"] for row in cases],
        [2026092500 + index for index in range(len(cases))],
        settings(args.binary), deployment_routes, str(THOMAS), str(META))
    batch.run(args.threads, 2 << 20, True)
    batch.run_native_actor_suffix(
        str(args.weights), 0, args.threads, 2 << 20, True)
    traces = batch.action_traces()
    summary = batch.summary()["cases"]
    arrays = {name: np.asarray(value) for name, value in batch.ppo_arrays().items()}
    first_mismatch = None
    for index, (case, terminal) in enumerate(zip(cases, summary)):
        own_trace, rival_trace = traces["own"][index], traces["rival"][index]
        if len(own_trace) != 719 or len(rival_trace) != 719:
            raise RuntimeError("JobBatch did not capture all 719 actions")
        env = FastEnv(Config(), case["seed"])
        opponent = NativeReplayOpponent(bundle.executor, case["route"], True, 0)
        for step, (own, want) in enumerate(zip(own_trace, rival_trace)):
            got = opponent.action(env, 1 - case["seat"], step)
            if encode(got) != want:
                first_mismatch = {
                    "case": index, "family": case["family"], "seat": case["seat"],
                    "step": step, "job": want, "python": encode(got),
                }
                break
            actions = [None, None]
            actions[case["seat"]] = decode(own)
            actions[1 - case["seat"]] = got
            env.step_raw(actions)
        if first_mismatch:
            break
        if (not env.done or env.rewards[case["seat"]] != terminal["own_cash"] or
                env.rewards[1 - case["seat"]] != terminal["rival_cash"]):
            raise RuntimeError(f"terminal mismatch for {case}")
    expected_routes = np.asarray([row["route"] for row in cases])
    identity_equal = (
        np.array_equal(arrays["opponent"], np.full(len(cases), 3)) and
        np.array_equal(arrays["route"], expected_routes))
    result = {
        "status": "PASS" if first_mismatch is None and identity_equal else "FAIL",
        "scope": "replay-clean-python-vs-jobbatch-step-terminal-parity",
        "families": list(NATIVE_JOB_REPLAY_FAMILIES),
        "routes": expected_routes.tolist(),
        "route_ids": [row["route_id"] for row in cases[::2]],
        "seeds": sorted({row["seed"] for row in cases}),
        "both_seats": True,
        "games": len(cases),
        "steps_compared": 719 * len(cases) if first_mismatch is None else None,
        "terminal_equal": first_mismatch is None,
        "macro_failure_free": first_mismatch is None,
        "route_identity_equal": bool(identity_equal),
        "first_mismatch": first_mismatch,
        "terminals": [{
            "family": case["family"], "seed": case["seed"], "seat": case["seat"],
            "own_cash": terminal["own_cash"], "rival_cash": terminal["rival_cash"],
        } for case, terminal in zip(cases, summary)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
