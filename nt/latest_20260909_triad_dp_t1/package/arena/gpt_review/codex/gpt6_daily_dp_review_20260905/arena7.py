"""Frozen GPT policies vs seven complete native opponents; no strategy tuning."""
from __future__ import annotations

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import copy
import gzip
import hashlib
import json
import math
import multiprocessing
import statistics
import sys
import time
import traceback
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = ROOT / "experiments/daily_dp_v7_20260903"
HOST = ROOT / "gpt_review/codex/G001_CPU_FOR_GPT_20260903"
SOURCE = ROOT / "gpt_review/gpt_code/gpt-6-dp"
OUT = HERE / "arena7_v1"
NAMES = ["g001", "g003", "boatlee_v29", "kaito_v58", "lynn_v5", "yhay81_six_day", "yhay81_three_day"]
FILES = {"v1": SOURCE / "gpt-6-kaggriculture_daily_dp_agent.py", "v2": SOURCE / "gpt-6-kaggriculture_daily_dp_agent-v2.py"}
EXPECTED = {"v1": "7dc2b0fc4439f38343d3912e98182f6b504ab6badce210de2923ab34c66be2a6", "v2": "00f9c50a5505780bf7ba9edb5d19bf58b147fcae96c1cee990ed905bab5f42f2"}
SEEDS = list(range(61020, 61030))
sys.path.insert(0, str(HOST))
sys.path.insert(0, str(EXP / "native/build"))
from cpu_runtime import LocalGame, load_agent
import _dp7_native as native


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def canon(obj):
    return json.loads(json.dumps(obj))


@lru_cache(None)
def asset(name):
    info = read(EXP / "opponents/registry.json")["opponents"][name]
    assert sha(EXP / info["source"]) == info["source_sha256"]
    payload = read_asset(EXP / info["asset"])
    assert payload["source_sha256"] == info["source_sha256"]
    cls = {"g001": native.G001, "g003": native.G001, "boatlee_v29": native.BoatleeV29,
           "kaito_v58": native.KaitoV58, "lynn_v5": native.LynnV5}[name]
    return cls(payload)


def read_asset(path):
    return json.loads(zlib.decompress(path.read_bytes()))


def opponent(name):
    if name == "yhay81_six_day":
        return native.Fieldbook(), None
    if name == "yhay81_three_day":
        return native.ThreeDay(), None
    state = {"g001": native.G001State, "g003": native.G001State,
             "boatlee_v29": native.BoatleeState, "kaito_v58": native.KaitoState,
             "lynn_v5": native.LynnState}[name]()
    return asset(name), state


def configuration():
    specification = read(HOST / "official/kaggriculture.json")
    values = {k: copy.deepcopy(v.get("default") if isinstance(v, dict) else v)
              for k,v in specification["configuration"].items()}
    values.update(seed=None, runTimeout=1200)
    return values


def run_one(job):
    version, name, seed, seat, referee = job
    started = time.perf_counter()
    trace, days, latency, rival_latency = [], [], [], []
    row = dict(version=version, opponent=name, seed=seed, seat=seat, referee=referee, status="RUNNING")
    step = -1
    try:
        assert sha(FILES[version]) == EXPECTED[version]
        entry = load_agent(FILES[version])
        env = native.Env(seed)
        rival, state = opponent(name)
        cfg = configuration()
        official = LocalGame(seed) if referee else None
        if official:
            for side in (0,1):
                for field in ("farms","private","market","town","day","hour","player"):
                    assert canon(env.observation(side)[field]) == canon(official.observation(side)[field]), ("initial",side,field)
        for step in range(719):
            obs = env.observation(seat)
            tic = time.perf_counter()
            own = entry(obs, copy.deepcopy(cfg))
            latency.append(time.perf_counter()-tic)
            tic = time.perf_counter()
            other = rival.act(env,1-seat) if state is None else rival.act(env,1-seat,state)
            rival_latency.append(time.perf_counter()-tic)
            pair = [None,None]
            pair[seat], pair[1-seat] = own, other
            if not isinstance(own,dict):
                raise TypeError("Candidate did not return a dict action")
            trace.append(canon(pair))
            if obs["hour"] == 0:
                scales=[]
                for farm in obs["farms"]:
                    counts={}
                    for tiles in farm["tiles"]:
                        for t in tiles:
                            if isinstance(t,dict):
                                k=t.get("animal") or t.get("crop") or t.get("kind")
                                counts[k]=counts.get(k,0)+1
                    scales.append(dict(cash=farm["money"],land=len(farm["unlocked_quadrants"]),counts=counts))
                days.append(dict(day=obs["day"],farms=scales))
            env.step(pair)
            if official:
                official.advance(pair)
                for side in (0,1):
                    actual, reference = env.observation(side), official.observation(side)
                    for field in ("farms","private","market","town","day","hour","player"):
                        assert canon(actual[field]) == canon(reference[field]), ("official_state",step,side,field)
            if env.done:
                break
        assert env.done and env.step_count == 719, (env.done,env.step_count)
        farms=env.observation(seat)["farms"]
        cash, other_cash = farms[seat]["money"], farms[1-seat]["money"]
        if official:
            assert official.done and official.t == 719
            assert official.state[seat].reward == cash and official.state[1-seat].reward == other_cash
        row.update(status="PASS",steps=env.step_count,cash=cash,opponent_cash=other_cash,
                   margin=cash-other_cash,win=cash>other_cash,tie=cash==other_cash,
                   action_mean_ms=1000*statistics.mean(latency),action_max_ms=1000*max(latency),
                   action_over_1s=sum(x>1 for x in latency),rival_action_seconds=sum(rival_latency),
                   own_action_seconds=sum(latency),official_checked_steps=719 if official else 0,
                   wall_seconds=time.perf_counter()-started,
                   terminal_private=canon(env.observation(seat)["private"]),days=days)
    except Exception:
        row.update(status="ERROR",failed_step=step,error=traceback.format_exc(),steps=len(trace),
                   wall_seconds=time.perf_counter()-started)
    key=f"{version}_{name}_{seed}_seat{seat}"
    trace_bytes=json.dumps(dict(metadata={k:v for k,v in row.items() if k not in ("days","terminal_private")},actions=trace),separators=(",",":")).encode()
    trace_path=OUT/"traces"/(key+".json.gz")
    trace_path.write_bytes(gzip.compress(trace_bytes))
    row["trace_sha256"]=sha(trace_path)
    write(OUT/"games"/(key+".json"),row)
    return row


def summarize(rows, nominal):
    complete=[r for r in rows if r["status"]=="PASS"]
    wins=sum(r["win"] for r in complete)
    ties=sum(r["tie"] for r in complete)
    result=dict(requested_games=nominal,completed_games=len(complete),errors=len(rows)-len(complete),
                wins=wins,ties=ties,losses=sum(not r["win"] and not r["tie"] for r in complete),
                win_rate=wins/nominal,score_rate=(wins+.5*ties)/nominal)
    if complete:
        result.update(mean_cash=statistics.mean(r["cash"] for r in complete),
                      mean_opponent_cash=statistics.mean(r["opponent_cash"] for r in complete),
                      mean_margin=statistics.mean(r["margin"] for r in complete),
                      min_cash=min(r["cash"] for r in complete),max_cash=max(r["cash"] for r in complete),
                      max_action_ms=max(r["action_max_ms"] for r in complete),
                      action_over_1s=sum(r["action_over_1s"] for r in complete),
                      own_action_seconds=sum(r["own_action_seconds"] for r in complete),
                      official_checked_steps=sum(r["official_checked_steps"] for r in complete))
    return result


def freeze(workers):
    build=read(EXP/"native/build/build_receipt.json")
    for path,expected in build["source_hashes"].items():
        assert sha(EXP/path)==expected, ("build drift",path)
    binary=next((EXP/"native/build").glob("_dp7_native*.so"))
    assert sha(binary)==build["binary_sha256"]
    registry=read(EXP/"opponents/registry.json")
    identities={}
    for name in NAMES:
        info=registry["opponents"][name]
        assert sha(EXP/info["source"])==info["source_sha256"]
        identities[name]=dict(source=info["source"],source_sha256=info["source_sha256"],runtime=info["runtime"])
        if "asset" in info:
            path=EXP/info["asset"]
            if "asset_sha256" in info:assert sha(path)==info["asset_sha256"]
            identities[name]["asset_sha256"]=sha(path)
            payload=read_asset(path)
            assert payload["source_sha256"]==info["source_sha256"]
            if name=="lynn_v5":
                for rel,h in payload["source_hashes"].items():
                    assert sha((EXP/info["source"]).parent/rel)==h
        for field in ("initial_parity_receipt","isolation_receipt"):
            if field in info:
                receipt=read(EXP/info[field])
                assert receipt["status"]=="PASS"
                if name not in ("g001","g003"):
                    assert receipt["build"]["binary_sha256"]==build["binary_sha256"]
                identities[name][field]=info[field]
    for version,path in FILES.items():assert sha(path)==EXPECTED[version]
    assert sha(HOST/"official/kaggriculture.py")=="bc8a54879ef02c7ea64b8b333d6a976f0ea65c4949149d01f463f23bccee653e"
    return dict(status="FROZEN",seeds=SEEDS,seats=[0,1],opponents=NAMES,versions=EXPECTED,
                requested_games=280,workers=workers,blas_threads=1,identities=identities,
                native_binary_sha256=build["binary_sha256"],script_sha256=sha(Path(__file__)),
                official_sha256=sha(HOST/"official/kaggriculture.py"),
                exact_source_agents=True,live_opponents=True,policy_settings_modified=False,
                official_referee="first seed, both seats, both versions, seven opponents = 28 complete games",
                timeout_scope="Measured Agent calls; not Kaggle sandbox timeout enforcement",
                evidence_role="10-seed paired screening; not stable leaderboard strength")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--workers",type=int,default=8)
    args=parser.parse_args()
    assert 1<=args.workers<=16
    protocol=freeze(args.workers)
    OUT.mkdir(exist_ok=False)
    (OUT/"games").mkdir()
    (OUT/"traces").mkdir()
    write(OUT/"protocol.json",protocol)
    jobs=[(version,name,seed,seat,seed==SEEDS[0]) for seed in SEEDS for name in NAMES for version in FILES for seat in (0,1)]
    print(json.dumps(dict(status="START",games=len(jobs),workers=args.workers)),flush=True)
    started=time.perf_counter()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context("spawn")) as pool:
        for future in as_completed([pool.submit(run_one,job) for job in jobs]):
            row=future.result();rows.append(row)
            if len(rows)%14==0 or row["status"]!="PASS":
                progress=dict(completed=len(rows),total=len(jobs),elapsed_seconds=time.perf_counter()-started,
                              v1=summarize([r for r in rows if r["version"]=="v1"],max(1,sum(r["version"]=="v1" for r in rows))),
                              v2=summarize([r for r in rows if r["version"]=="v2"],max(1,sum(r["version"]=="v2" for r in rows))))
                write(OUT/"progress.json",progress)
                print(json.dumps(progress),flush=True)
                if row["status"]!="PASS":print(json.dumps(row),flush=True)
    result=dict(status="PASS_COMPLETE_SCREEN" if all(r["status"]=="PASS" for r in rows) else "COMPLETED_WITH_ERRORS",
                protocol=protocol,wall_seconds=time.perf_counter()-started,overall={},per_opponent={})
    for version in FILES:
        result["overall"][version]=summarize([r for r in rows if r["version"]==version],140)
        result["per_opponent"][version]={name:summarize([r for r in rows if r["version"]==version and r["opponent"]==name],20) for name in NAMES}
    assert len(rows)==280
    for version,path in FILES.items():assert sha(path)==EXPECTED[version]
    result["source_hashes_unchanged"]=True
    write(OUT/"result.json",result)
    lines=["# GPT两版对七对手：实时对战筛查", "", "同一组10个seed，双座位，每版每对手20局，总计280局。GPT策略保持原代码/参数；对手与规则使用冻结C++实现。", "",
           "| 对手 | V1胜/局 | V1平均现金 | V1平均分差 | V2胜/局 | V2平均现金 | V2平均分差 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in NAMES:
        a,b=(result["per_opponent"][v][name] for v in FILES)
        lines.append(f"| {name} | {a['wins']}/20 | {a.get('mean_cash',0):,.0f} | {a.get('mean_margin',0):+,.0f} | {b['wins']}/20 | {b.get('mean_cash',0):,.0f} | {b.get('mean_margin',0):+,.0f} |")
    lines += ["", "## 合计", ""]
    for version in FILES:
        a=result["overall"][version]
        lines.append(f"- {version}: {a['wins']}/140，胜率{a['win_rate']:.2%}；平均现金{a.get('mean_cash',0):,.2f}，平均分差{a.get('mean_margin',0):+,.2f}；异常{a['errors']}局；最慢Agent调用{a.get('max_action_ms',0):.2f}ms，超过1秒的调用{a.get('action_over_1s',0)}次。")
    steps=sum(a.get("official_checked_steps",0) for a in result["overall"].values())
    lines += ["",f"官方Python1.32.7同步逐步状态核对：{steps:,}步；预定28局，共20,132步。其余对局使用同一冻结C++引擎。对手原策略的移植验收沿用protocol里的既有收据，本次没有重新穷尽所有原版策略分支。",
              "",f"批量耗时{result['wall_seconds']:.2f}秒，8进程是默认设置，实际进程数见protocol。不是全C++性能：GPT原Python每步仍执行。",
              "", "这是10个独立seed的方向筛查；双座位不是20个独立随机样本。未进行选型或调参，不可将小样本胜率当稳定天梯胜率。Kaggle端到端超时/内存限制未验证。",
              "", "逐局现金、每日产业规模、动作耗时见games；完整双方动作序列见traces，可按seed重放。原始结果为result.json，冻结协议为protocol.json。"]
    (OUT/"RESULT_ZH.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="protocol"}),flush=True)


if __name__=="__main__":
    main()
