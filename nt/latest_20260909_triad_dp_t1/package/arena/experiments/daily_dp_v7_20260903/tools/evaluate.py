"""Frozen-config CPU screens for v7. Uses exact official interpreter via verified host."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "gpt_review/codex/G001_CPU_FOR_GPT_20260903"))
from cpu_runtime import LocalGame, load_agent, pass_agent

FLAGS = ("fix_resources", "fix_expiry", "fix_values", "fix_calendar")


def variant_config(name, overrides):
    conf = {f: name == "core" or f == "fix_" + name for f in FLAGS}
    if name not in ("core", "none", "resources", "expiry", "values", "calendar"):
        raise ValueError(name)
    conf.update(overrides)
    return conf


def match(task):
    candidate_path, opponent_path, seed, seat, config, variant, replay_folder = task
    spec = importlib.util.spec_from_file_location("v7_eval_" + uuid.uuid4().hex, candidate_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    candidate = mod.agent
    if config:
        mod._CONTROLLERS = {s: mod.DailyDPController(mod.Params(**config)) for s in (0, 1)}
    opponent = load_agent(opponent_path) if opponent_path else pass_agent
    agents = [candidate, opponent] if seat == 0 else [opponent, candidate]
    game = LocalGame(seed)
    frames = [game.snapshot()] if replay_folder else None
    latencies=[]; began=time.perf_counter()
    while not game.done:
        actions=[]
        for s, fn in enumerate(agents):
            t=time.perf_counter()
            action=fn(game.observation(s),copy.deepcopy(game.configuration))
            if s==seat: latencies.append(time.perf_counter()-t)
            actions.append(action)
        game.advance(actions)
        if frames is not None: frames.append(game.snapshot())
    assert game.t==719 and all(s.status=="DONE" for s in game.state)
    own,rival=game.state[seat].reward,game.state[1-seat].reward
    result={"variant":variant,"seed":seed,"seat":seat,"cash":own,"opponent_cash":rival,"margin":own-rival,
            "win":own>rival,"tie":own==rival,"steps":game.t,"max_call_ms":max(latencies)*1000,
            "mean_call_ms":statistics.fmean(latencies)*1000,"seconds":time.perf_counter()-began}
    if replay_folder:
        import gzip
        path=Path(replay_folder)/f"{variant}_seed{seed}_seat{seat}.json.gz"
        with gzip.open(path,"xt",encoding="utf-8") as handle:
            json.dump({"configuration":dict(game.configuration),"info":game.info,"steps":frames},handle,separators=(",",":"))
        result["replay"]=str(path)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--candidate",type=Path,default=HERE/"agents/agent_v7.py")
    p.add_argument("--opponent",choices=("pass","g001"),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--count",type=int,required=True)
    p.add_argument("--seats",default="0,1")
    p.add_argument("--variants",default="core")
    p.add_argument("--params",default="{}")
    p.add_argument("--workers",type=int,default=8)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--save-replays",action="store_true")
    args=p.parse_args()
    assert 1<=args.workers<=16 and args.count>=1
    args.out.mkdir(parents=True,exist_ok=False)
    candidate=args.out/"candidate_snapshot.py"
    shutil.copy2(args.candidate,candidate)
    opponent=ROOT/"gpt_review/codex/G001_CPU_FOR_GPT_20260903/agents/g001.py" if args.opponent=="g001" else None
    configs={name:variant_config(name,json.loads(args.params)) for name in args.variants.split(",")}
    if args.save_replays:(args.out/"replays").mkdir()
    tasks=[(str(candidate.resolve()),str(opponent) if opponent else None,seed,int(seat),conf,name,
            str((args.out/"replays").resolve()) if args.save_replays else None)
           for name,conf in configs.items() for seed in range(args.seed,args.seed+args.count) for seat in args.seats.split(",")]
    metadata={"candidate_source":str(args.candidate.resolve()),"candidate_sha256":hashlib.sha256(candidate.read_bytes()).hexdigest(),
              "opponent":args.opponent,"opponent_sha256":hashlib.sha256(opponent.read_bytes()).hexdigest() if opponent else None,
              "configuration":configs,"seed_start":args.seed,"seed_count":args.count,"seats":args.seats,
              "host":"Verified lightweight host around official1.32.7; not Kaggle sandbox/timeout certification"}
    (args.out/"metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    rows=[];errors=[];start=time.perf_counter()
    with ProcessPoolExecutor(max_workers=min(args.workers,len(tasks))) as pool:
        futures={pool.submit(match,t):t for t in tasks}
        for f in as_completed(futures):
            try: rows.append(f.result())
            except Exception as exc: errors.append({"seed":futures[f][2],"seat":futures[f][3],"variant":futures[f][5],"error":repr(exc)})
    rows.sort(key=lambda r:(r["variant"],r["seed"],r["seat"]))
    summary={}
    for name in configs:
        r=[x for x in rows if x["variant"]==name]
        if not r: continue
        summary[name]={"games":len(r),"wins":sum(x["win"] for x in r),"ties":sum(x["tie"] for x in r),
                       "mean_cash":statistics.fmean(x["cash"] for x in r),"mean_opponent_cash":statistics.fmean(x["opponent_cash"] for x in r),
                       "mean_margin":statistics.fmean(x["margin"] for x in r),"min_cash":min(x["cash"] for x in r),
                       "max_call_ms":max(x["max_call_ms"] for x in r)}
    result={"metadata":metadata,"summary":summary,"rows":rows,"errors":errors,"wall_seconds":time.perf_counter()-start}
    (args.out/"results.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k not in ("metadata","rows")},ensure_ascii=False,indent=2),flush=True)
    if errors or len(rows)!=len(tasks):raise RuntimeError(f"Incomplete screen: {len(rows)}/{len(tasks)}, errors={errors}")


if __name__=="__main__":main()
