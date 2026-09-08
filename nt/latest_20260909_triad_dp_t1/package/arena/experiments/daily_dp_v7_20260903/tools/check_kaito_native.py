"""Full V58 reference actions AND ten child memories versus native/official."""
from pathlib import Path
import argparse
import copy
import gzip
import hashlib
import inspect
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
from check_native import canon, normalized as common_normalized
from check_boatlee_native import SHOPS, digest


def normalized(action):
    # This source contains numeric-string quantities. The official interpreter
    # calls int(quantity). Compare that defined semantic action, not JSON types.
    result = common_normalized(action)
    for atom in [result["farmer"], *result["hands"], *result["market"]]:
        if atom[0] in ("PICKUP", "PLACE", "BUY_PRODUCT", "BUY_ANIMAL", "BUY_SEED", "SELL") and len(atom)>2:
            atom[2] = int(atom[2])
    return result


def load_reference(source):
    reference = load_agent(source)
    ns = reference.__globals__
    original_builder = ns["_v51_build_policy"]
    def builder(*args, **kwargs):
        child = original_builder(*args, **kwargs)
        def recording(*a, **kw):
            result = child(*a, **kw)
            recording.emitted = copy.deepcopy(result)
            return result
        recording.controller = child.controller
        return recording
    ns["_v51_build_policy"] = builder
    return reference


def ref_state(reference, seat):
    ns = reference.__globals__
    result = copy.deepcopy(ns["_V58_STATES"][seat])
    rows = []
    for name, child in ns["_V58_POLICIES"].items():
        state = child.controller.states[seat]
        row = {key: copy.deepcopy(state[key]) for key in (
            "last_step", "near_streak", "near_latched", "last_inventory", "last_market_net",
            "opponent_supply", "mirror_confidence", "mirror_evidence_turns", "due")}
        row["last_shops"] = [SHOPS.index(s) for s in state["last_shops"]]
        base = inspect.getclosurevars(child.controller.base_policy).nonlocals["base"]
        weed = inspect.getclosurevars(base).nonlocals["states"][seat]
        row["active"] = {str(0 if u == "farmer" else int(u)+1): dict(start=r["start"],
                         intended=normalized({"farmer":r["intended"]})["farmer"])
                         for u, r in weed["active"].items()}
        row["emitted"] = normalized(child.emitted)
        row["collisions"] = base.telemetry["collisions"]
        for k in ("preempt_turns", "preempt_units", "repaid_units"):
            row[k] = child.controller.telemetry[k]
        rows.append(row)
    result["policies"] = rows
    return canon(result)


def cpp_state(st, built):
    result = canon(st.debug())
    result.pop("selected")
    result["policies"] = result["policies"][:built]
    for row in result["policies"]:
        row["emitted"] = normalized(row["emitted"])
        for rec in row["active"].values():
            rec["intended"] = normalized({"farmer":rec["intended"]})["farmer"]
    return result


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
    source = EXP / "opponents/kaito_v58/output/main.py"
    asset = EXP / "native/kaito_v58_frozen.json.zlib"
    data = json.loads(zlib.decompress(asset.read_bytes()))
    assert digest(source) == data["source_sha256"]
    opponent = native.KaitoV58(data)
    build = json.loads((EXP / "native/build/build_receipt.json").read_text())
    assert digest(Path(native.__file__)) == build["binary_sha256"]
    for rel, h in build["source_hashes"].items():assert digest(EXP / rel) == h, rel
    receipt = dict(status="RUNNING", args=vars(a), config=config, build=build,
                   opponent_source_sha256=digest(source), opponent_asset_sha256=digest(asset), rows=[])
    (out / "plan.json").write_text(json.dumps(receipt, indent=2))
    def fail(kind, **details):
        (out / "failure.json").write_text(json.dumps(dict(kind=kind, **details), indent=2))
        raise AssertionError((kind, details.get("seed"), details.get("seat"), details.get("step")))
    start = time.perf_counter()
    for seed in map(int, a.seeds.split(",")):
        for seat in map(int, a.seats.split(",")):
            env, official = native.Env(seed), LocalGame(seed)
            own, st, own_st = native.Controller(config), native.KaitoState(), native.KaitoState()
            reference = load_reference(source)
            trace=[];selected=[0]*10;max_own=max_rival=0
            for step in range(719):
                tic=time.perf_counter()
                ours=opponent.act(env,seat,own_st) if a.mirror else own.act(env,seat)
                max_own=max(max_own,time.perf_counter()-tic)
                tic=time.perf_counter();actual=opponent.act(env,1-seat,st)
                max_rival=max(max_rival,time.perf_counter()-tic)
                expected=reference(official.observation(1-seat),copy.deepcopy(official.configuration))
                cpp,ref=cpp_state(st,min(step+1,10)),ref_state(reference,1-seat)
                if normalized(actual)!=normalized(expected):
                    fail("opponent_action",seed=seed,seat=seat,step=step,actual=actual,expected=expected,
                         native_state=cpp,reference_state=ref,observation=canon(env.observation(1-seat)))
                if cpp!=ref:
                    fail("opponent_memory",seed=seed,seat=seat,step=step,actual=cpp,expected=ref)
                selected[st.debug()["selected"]]+=1
                actions=[None,None];actions[seat],actions[1-seat]=ours,actual
                ref_actions=copy.deepcopy(actions);ref_actions[1-seat]=expected
                if a.save_replays:trace.append(dict(step=step,observation=canon(env.observation(seat)),actions=actions))
                env.step(actions);official.advance(ref_actions)
                for side in (0,1):
                    co,ro=canon(env.observation(side)),canon(official.observation(side))
                    for field in ("farms","private","market","town","day","hour","step"):
                        if co[field]!=ro[field]:fail("official_state",seed=seed,seat=seat,step=step+1,field=field,actual=co[field],expected=ro[field])
            assert env.done and env.step_count==719
            farms=env.observation(seat)["farms"]
            row=dict(seed=seed,seat=seat,steps=719,cash=farms[seat]["money"],opponent_cash=farms[1-seat]["money"],
                     selected_route_frames=selected,action_mismatches=0,memory_mismatches=0,official_state_mismatches=0,
                     preempt_units_by_route=[r["preempt_units"] for r in ref["policies"]],
                     collisions_by_route=[r["collisions"] for r in ref["policies"]],
                     max_candidate_action_seconds=max_own,max_opponent_action_seconds=max_rival)
            receipt["rows"].append(row);print(json.dumps(row),flush=True)
            if a.save_replays:(out/f"{seed}_seat{seat}.json.gz").write_bytes(gzip.compress(json.dumps(dict(trace=trace,final=canon(env.observation(seat)))).encode()))
            (out/"progress.json").write_text(json.dumps(receipt,indent=2))
    receipt.update(status="PASS",seconds=time.perf_counter()-start,
                   check="full_published_Kaito_actions_all_built_child_memory_and_official_719_steps",
                   coverage_caveat="Finite sample, not exhaustive router signature coverage. Disabled original defer/MM remain disabled.")
    (out/"acceptance.json").write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":main()
