"""Frozen ThreeDay C++ adapter vs published main.py + agent.so, official referee.

Python is only the differential referee here. Bulk games use batch_three_day.
The downloaded binary's process-global reference Session is used sequentially.
"""
from pathlib import Path
import argparse
import copy
import ctypes
import gzip
import hashlib
import json
import sys
import time

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "native/build"))
sys.path.insert(0, str(ROOT / "gpt_review/codex/G001_CPU_FOR_GPT_20260903"))
import _dp7_native as native
from cpu_runtime import LocalGame, load_agent
from check_native import canon, normalized


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flatten(value):
    if isinstance(value, ctypes.Structure):
        for name, *_ in value._fields_:
            yield from flatten(getattr(value, name))
    elif isinstance(value, ctypes.Array):
        for child in value:
            yield from flatten(child)
    else:
        yield float(value)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--seed", type=int, default=20261401)
    p.add_argument("--count", type=int, default=2)
    p.add_argument("--seeds", help="Optional explicit development seeds; supersedes seed/count")
    p.add_argument("--seats", default="0,1")
    p.add_argument("--out", required=True)
    p.add_argument("--save-replays", action="store_true")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    params = json.loads(Path(a.configs).read_text())[a.label]
    source = EXP / "opponents/yhay81_three_day/output/main.py"
    library = source.with_name("agent.so")
    assert digest(source) == "23b0d0c93bcf3dfecaabfecd950ec4ab3b4c46000c0c481bddcf0a63ce019f02"
    assert digest(library) == "d3b4e19dfd20f1d9b229d1d15cab9ee20068d100fb99fcf4c44051d59fb174ea"
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    assert digest(Path(native.__file__)) == build["binary_sha256"]
    for rel, expected in build["source_hashes"].items():
        assert digest(EXP / rel) == expected, rel
    receipt = dict(status="RUNNING", args=vars(a), params=params, build=build,
                   original_entry_sha256=digest(source), original_binary_sha256=digest(library),
                   original_policy_unmodified=True, rows=[])
    (out / "plan.json").write_text(json.dumps(receipt, indent=2))

    def fail(kind, **detail):
        (out / "failure.json").write_text(json.dumps(dict(kind=kind, **detail), indent=2))
        raise AssertionError((kind, detail.get("seed"), detail.get("seat"), detail.get("step")))

    started = time.perf_counter()
    seeds = list(map(int, a.seeds.split(","))) if a.seeds else range(a.seed, a.seed + a.count)
    for seed in seeds:
        for seat in map(int, a.seats.split(",")):
            env = native.Env(seed)
            ctl = native.Controller(params)
            rival = native.ThreeDay()
            official = LocalGame(seed)
            reference = load_agent(source)
            pack_observation = reference.__globals__["_pack_observation"]
            trace = []
            max_candidate_latency = max_opponent_latency = 0
            fields_checked = 0
            for step in range(719):
                obs = official.observation(1-seat)
                expected_input = list(flatten(pack_observation(obs, 1-seat)))
                actual_input = native.three_day_input_fields(env, 1-seat)
                if expected_input != actual_input:
                    mismatch = [(i, x, y) for i, (x, y) in enumerate(zip(expected_input, actual_input)) if x != y]
                    fail("observation_projection", seed=seed, seat=seat, step=step,
                         expected_length=len(expected_input), actual_length=len(actual_input),
                         mismatches=mismatch[:30], observation=canon(obs))
                fields_checked += len(actual_input)
                tic = time.perf_counter()
                own = ctl.act(env, seat)
                max_candidate_latency = max(max_candidate_latency, time.perf_counter()-tic)
                tic = time.perf_counter()
                other = rival.act(env, 1-seat)
                max_opponent_latency = max(max_opponent_latency, time.perf_counter()-tic)
                ref = reference(obs, copy.deepcopy(official.configuration))
                if normalized(other) != normalized(ref):
                    fail("opponent_action", seed=seed, seat=seat, step=step,
                         actual=other, expected=ref, selected_route=rival.route(),
                         observation=canon(obs))
                actions = [None, None]
                actions[seat], actions[1-seat] = own, other
                reference_actions = copy.deepcopy(actions)
                reference_actions[1-seat] = ref
                if a.save_replays:
                    trace.append(dict(step=step, observation=canon(env.observation(seat)),
                                      actions=actions, original_opponent_action=ref,
                                      selected_route=rival.route()))
                # Independent transition inputs: the official side uses original ref.
                env.step(actions)
                official.advance(reference_actions)
                for side in (0, 1):
                    actual, expected = canon(env.observation(side)), canon(official.observation(side))
                    for field in ("farms", "private", "market", "town", "day", "hour", "step"):
                        if actual[field] != expected[field]:
                            fail("official_state", seed=seed, seat=seat, step=step+1,
                                 side=side, field=field, actual=actual[field], expected=expected[field])
            assert env.done and env.step_count == 719
            farms = env.observation(seat)["farms"]
            row = dict(seed=seed, seat=seat, steps=719, cash=farms[seat]["money"],
                       opponent_cash=farms[1-seat]["money"], selected_route=rival.route(),
                       action_mismatches=0, official_state_mismatches=0,
                       projection_fields_checked=fields_checked,
                       max_candidate_action_seconds=max_candidate_latency,
                       max_opponent_action_seconds=max_opponent_latency)
            receipt["rows"].append(row)
            print(json.dumps(row), flush=True)
            if a.save_replays:
                (out / f"{seed}_seat{seat}.json.gz").write_bytes(gzip.compress(json.dumps(
                    dict(trace=trace, final=canon(env.observation(seat)))).encode()))
            (out / "progress.json").write_text(json.dumps(receipt, indent=2))
    receipt.update(status="PASS", seconds=time.perf_counter()-started,
                   check="published_binary_actions_input_projection_and_official_719_step_state",
                   coverage_caveat="Finite differential sample, not all possible policy branches.")
    (out / "acceptance.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
