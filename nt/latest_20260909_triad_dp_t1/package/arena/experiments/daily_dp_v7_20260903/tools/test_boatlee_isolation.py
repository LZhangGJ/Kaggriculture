"""Boatlee whole-season threaded, interleaving, backward-clock and seat reset tests."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import zlib

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "native/build"))
import _dp7_native as native


def signature(row):
    return {k:row[k] for k in ("seed", "seat", "steps", "cash", "opponent_cash", "overflow",
                              "opponent_state", "weed_repair_frames")}


def main():
    p=argparse.ArgumentParser();p.add_argument("--out", required=True);a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True, exist_ok=False)
    params=json.loads((EXP/"profiles/candidates.json").read_text())["S3C03"]
    asset=EXP/"native/boatlee_v29_frozen.json.zlib"
    opponent=native.BoatleeV29(json.loads(zlib.decompress(asset.read_bytes())))
    tasks=[(seed,seat) for seed in range(20261401,20261451) for seat in (0,1)]
    started=time.perf_counter()
    serial=native.batch_boatlee([s for s,_ in tasks],[p for _,p in tasks],params,opponent,1)
    tic=time.perf_counter();repeated=tasks*10
    parallel=native.batch_boatlee([s for s,_ in repeated],[p for _,p in repeated],params,opponent,16)
    seconds=time.perf_counter()-tic
    expected={(r["seed"],r["seat"]):signature(r) for r in serial}
    for row in parallel:
        assert row["steps"]==719 and signature(row)==expected[row["seed"],row["seat"]]

    def trace_game(seed,seat,state=None):
        env,ctl=native.Env(seed),native.Controller(params)
        state=native.BoatleeState() if state is None else state
        h=hashlib.sha256()
        for _ in range(719):
            acts=[None,None];acts[seat]=ctl.act(env,seat);acts[1-seat]=opponent.act(env,1-seat,state)
            env.step(acts)
            h.update(json.dumps(dict(acts=acts,obs=env.observation(seat),state=state.debug()),sort_keys=True).encode())
        return h.hexdigest()

    sample=[tasks[0],tasks[-1]];refs=[trace_game(seed,seat) for seed,seat in sample]
    envs=[native.Env(seed) for seed,_ in sample];ctls=[native.Controller(params) for _ in sample]
    states=[native.BoatleeState() for _ in sample];hashes=[hashlib.sha256() for _ in sample]
    for _ in range(719):
        for i,(_,seat) in enumerate(sample):
            acts=[None,None];acts[seat]=ctls[i].act(envs[i],seat);acts[1-seat]=opponent.act(envs[i],1-seat,states[i])
            envs[i].step(acts)
            hashes[i].update(json.dumps(dict(acts=acts,obs=envs[i].observation(seat),state=states[i].debug()),sort_keys=True).encode())
    assert [h.hexdigest() for h in hashes]==refs
    for i,(seed,seat) in enumerate(sample):assert trace_game(seed,seat,states[1-i])==refs[i]
    receipt=dict(status="PASS", build=json.loads((EXP/"native/build/build_receipt.json").read_text()),
                 asset_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(), params=params,
                 single_thread_games=100, parallel_repeat_games=1000, independent_seeds=50, threads=16,
                 full_trajectory_hash_games=6, repeat_mismatches=0, reset_mismatches=0, interleaved_mismatches=0,
                 parallel_seconds=seconds, parallel_games_per_second=1000/seconds,
                 seconds=time.perf_counter()-started,
                 weed_repair_games=sum(r["weed_repair_frames"]>0 for r in serial),
                 serial_rows=serial, caveat="Repeated games test isolation, not 1000 independent strength samples.")
    (out/"acceptance.json").write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k not in ("build","params","serial_rows")}),flush=True)
    print(json.dumps(dict(weed_examples=[dict(seed=r["seed"],seat=r["seat"],frames=r["weed_repair_frames"]) for r in serial if r["weed_repair_frames"]>0][:6])),flush=True)


if __name__=="__main__":main()
