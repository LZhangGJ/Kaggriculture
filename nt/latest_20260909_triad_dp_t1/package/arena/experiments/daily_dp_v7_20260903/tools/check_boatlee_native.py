"""Original Boatlee V29 action/state parity with official 1.32.7 referee."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import json
import sys
import time
import zlib

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "native/build"))
sys.path.insert(0, str(ROOT / "gpt_review/codex/G001_CPU_FOR_GPT_20260903"))
import _dp7_native as native
from cpu_runtime import LocalGame, load_agent
from check_native import canon, normalized

SHOPS = ["BAKERY", "BRUNCH_SPOT", "FARMERS_MARKET", "ICE_CREAM_SHOP",
         "PET_CAFE", "PIZZA_SHOP", "SMOOTHIE_SHOP", "YARN_STORE"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ref_state(reference, seat):
    ns = reference.__globals__
    state, weed = ns["_AM_STATE"][seat], ns["_WEED_STATE"][seat]
    result = {name: {item: state.get(name, {}).get(item, 0) for item in ns["_AM_ITEMS"]}
              for name in ("pressure", "added", "last_inventory", "last_sold")}
    result.update(last_step=state["last_step"], weed_last=weed["last_step"],
                  near_mirror=-1 if state["near_mirror"] is None else int(state["near_mirror"]),
                  last_shops=[SHOPS.index(x) for x in state["last_shops"]],
                  active={str(0 if u == "farmer" else int(u)+1): {
                      "start": r["start"], "intended": normalized({"farmer":r["intended"]})["farmer"]}
                      for u, r in weed["active"].items()})
    return result


def cpp_state(state):
    d = canon(state.debug())
    for v in d["active"].values():
        v["intended"] = normalized({"farmer":v["intended"]})["farmer"]
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--configs", default=str(EXP / "profiles/candidates.json"))
    p.add_argument("--label", default="S3C03")
    p.add_argument("--seeds", default="20261401,20261402")
    p.add_argument("--seats", default="0,1")
    p.add_argument("--mirror", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--save-replays", action="store_true")
    a = p.parse_args()
    out = Path(a.out);out.mkdir(parents=True, exist_ok=False)
    config = json.loads(Path(a.configs).read_text())[a.label]
    asset = EXP / "native/boatlee_v29_frozen.json.zlib"
    data = json.loads(zlib.decompress(asset.read_bytes()))
    source = EXP / "opponents/boatlee_v29/output/main.py"
    assert digest(source) == data["source_sha256"]
    opponent = native.BoatleeV29(data)
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    assert digest(Path(native.__file__)) == build["binary_sha256"]
    for rel, h in build["source_hashes"].items():assert digest(EXP / rel) == h, rel
    receipt = dict(status="RUNNING", args=vars(a), config=config,
                   opponent_source_sha256=digest(source), opponent_asset_sha256=digest(asset),
                   build=build, rows=[])
    (out / "plan.json").write_text(json.dumps(receipt, indent=2))

    def fail(kind, **details):
        (out / "failure.json").write_text(json.dumps(dict(kind=kind, **details), indent=2))
        raise AssertionError((kind, details.get("seed"), details.get("seat"), details.get("step")))

    started = time.perf_counter()
    for seed in map(int, a.seeds.split(",")):
        for seat in map(int, a.seats.split(",")):
            env, official = native.Env(seed), LocalGame(seed)
            own, st, own_st = native.Controller(config), native.BoatleeState(), native.BoatleeState()
            reference = load_agent(source)
            trace = [];weed_frames=market_frames=0;max_own=max_rival=0
            for step in range(719):
                tic = time.perf_counter()
                ours = opponent.act(env, seat, own_st) if a.mirror else own.act(env, seat)
                max_own = max(max_own, time.perf_counter()-tic)
                tic = time.perf_counter()
                actual = opponent.act(env, 1-seat, st)
                max_rival = max(max_rival, time.perf_counter()-tic)
                expected = reference(official.observation(1-seat))
                if normalized(actual) != normalized(expected):
                    fail("opponent_action", seed=seed, seat=seat, step=step, actual=actual,
                         expected=expected, native_state=cpp_state(st), reference_state=ref_state(reference,1-seat),
                         observation=canon(env.observation(1-seat)))
                actual_state, expected_state = cpp_state(st), ref_state(reference, 1-seat)
                if actual_state != expected_state:
                    fail("opponent_memory", seed=seed, seat=seat, step=step,
                         actual=actual_state, expected=expected_state)
                weed_frames += bool(actual_state["active"])
                base = reference.__globals__["_ACTIONS"][step]
                market_frames += normalized(actual)["market"] != normalized(base)["market"]
                actions = [None, None];actions[seat], actions[1-seat] = ours, actual
                ref_actions = copy.deepcopy(actions);ref_actions[1-seat] = expected
                if a.save_replays:trace.append(dict(step=step, observation=canon(env.observation(seat)), actions=actions, opponent_state=actual_state))
                env.step(actions);official.advance(ref_actions)
                for side in (0, 1):
                    actual_obs, expected_obs = canon(env.observation(side)), canon(official.observation(side))
                    for field in ("farms", "private", "market", "town", "day", "hour", "step"):
                        if actual_obs[field] != expected_obs[field]:
                            fail("official_state", seed=seed, seat=seat, step=step+1, field=field,
                                 actual=actual_obs[field], expected=expected_obs[field])
            assert env.done and env.step_count == 719
            farms = env.observation(seat)["farms"]
            row = dict(seed=seed, seat=seat, steps=719, cash=farms[seat]["money"], opponent_cash=farms[1-seat]["money"],
                       opponent_state=cpp_state(st), weed_repair_frames=weed_frames, adaptive_market_frames=market_frames,
                       action_mismatches=0, memory_mismatches=0, official_state_mismatches=0,
                       max_candidate_action_seconds=max_own, max_opponent_action_seconds=max_rival)
            receipt["rows"].append(row)
            print(json.dumps(row), flush=True)
            if a.save_replays:
                (out / f"{seed}_seat{seat}.json.gz").write_bytes(gzip.compress(json.dumps(dict(trace=trace, final=canon(env.observation(seat)))).encode()))
            (out / "progress.json").write_text(json.dumps(receipt, indent=2))
    receipt.update(status="PASS", seconds=time.perf_counter()-started,
                   check="original_Boatlee_actions_and_persistent_state_plus_official_719_step_state",
                   coverage_caveat="Finite differential sample; not an exhaustive proof for all legal observations.")
    (out / "acceptance.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
