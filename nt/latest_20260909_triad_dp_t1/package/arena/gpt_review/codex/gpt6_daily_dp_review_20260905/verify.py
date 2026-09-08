"""Read-only policy review using the frozen official interpreter; no tuning."""
from __future__ import annotations

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import copy
import argparse
import hashlib
import json
import statistics
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "gpt_review/gpt_code/gpt-6-dp"
HOST = ROOT / "gpt_review/codex/G001_CPU_FOR_GPT_20260903"
sys.path.insert(0, str(HOST))
from cpu_runtime import LocalGame, load_agent, pass_agent

FILES = {
    "v1": SOURCE / "gpt-6-kaggriculture_daily_dp_agent.py",
    "v2": SOURCE / "gpt-6-kaggriculture_daily_dp_agent-v2.py",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_one(job):
    version, seed, seat, group, expected = job
    started = time.perf_counter()
    try:
        entry = load_agent(FILES[version])
        game = LocalGame(seed)
        timings, days, forecast_probe = [], [], None
        for step in range(719):
            obs = game.observation(seat)
            t0 = time.perf_counter()
            action = entry(obs, copy.deepcopy(game.configuration))
            timings.append(time.perf_counter() - t0)
            if obs["hour"] == 0:
                farm = obs["farms"][seat]
                assets = {}
                for row in farm["tiles"]:
                    for tile in row:
                        if isinstance(tile, dict):
                            kind = tile.get("animal") or tile.get("crop") or tile.get("kind")
                            assets[kind] = assets.get(kind, 0) + 1
                if version == "v1":
                    plan = copy.deepcopy(entry.__globals__["MEM"]["daily_records"][-1])
                else:
                    policy = entry.__globals__["_POLICY"]
                    plan = copy.deepcopy(policy.events[-1])
                    if forecast_probe is None and obs["day"] >= 8:
                        for y, row in enumerate(farm["tiles"]):
                            for x, tile in enumerate(row):
                                if isinstance(tile, dict) and tile.get("crop") in ("TOMATO", "STRAWBERRY"):
                                    other = copy.deepcopy(obs)
                                    old_yield = tile.get("yield_units", 0)
                                    new_yield = 0 if old_yield else 4
                                    other["farms"][seat]["tiles"][y][x]["yield_units"] = new_yield
                                    a = policy.forecast_base(obs)
                                    b = policy.forecast_base(other)
                                    forecast_probe = dict(day=obs["day"], tile=[x,y], crop=tile["crop"],
                                        old_yield=old_yield, changed_yield=new_yield,
                                        all_forecast_arrays_identical=all((aa == bb).all().item() for aa,bb in zip(a,b)))
                                    break
                            if forecast_probe is not None:
                                break
                days.append(dict(day=obs["day"], cash=farm["money"], land=len(farm["unlocked_quadrants"]), assets=assets, plan=plan))
            actions = [None, None]
            actions[seat] = action
            actions[1-seat] = pass_agent(game.observation(1-seat), game.configuration)
            game.advance(actions)
            if game.done:
                break
        obs = game.observation(seat)
        cash = obs["farms"][seat]["money"]
        assert game.done and game.t == 719, (game.done, game.t)
        assert obs["farms"][1-seat]["money"] == 3000
        assert game.state[seat].reward == cash
        result = dict(version=version, seed=seed, seat=seat, group=group, status="PASS",
                      cash=cash, expected_cash=expected,
                      exact_report_match=None if expected is None else cash == expected,
                      steps=game.t, opponent_cash=3000,
                      action_mean_ms=1000*statistics.mean(timings),
                      action_max_ms=1000*max(timings), action_seconds=sum(timings),
                      wall_seconds=time.perf_counter()-started, days=days,
                      forecast_probe=forecast_probe)
    except Exception:
        result = dict(version=version,seed=seed,seat=seat,group=group,status="ERROR",error=traceback.format_exc())
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-reports", action="store_true")
    args = parser.parse_args()
    jobs = [
        ("v1",60000,0,"report_anchor",195030),
        ("v1",60047,1,"report_anchor",144272),
        ("v2",43091,1,"report_anchor",181732),
    ]
    for version in FILES:
        jobs += [(version,seed,seat,"paired_fresh",None) for seed in range(61010,61014) for seat in (0,1)]
    out = HERE
    if args.full_reports:
        jobs = [("v1",seed,(seed-60000)%2,"v1_report50",None) for seed in range(60000,60050)]
        jobs += [("v2",seed,(seed-43000)%2,"v2_report_first50" if seed<43050 else "v2_report_second50",None) for seed in range(43000,43100)]
        out = HERE / "report_reproduction"
        out.mkdir(exist_ok=True)
    protocol = dict(purpose="Review only; 3 report anchors and 4 newly chosen paired seeds per version",
                    jobs=jobs, workers=4, sources={k:sha(v) for k,v in FILES.items()},
                    official_sha256=sha(HOST/"official/kaggriculture.py"),
                    interpreter=sys.version, no_source_modifications=True)
    if args.full_reports:
        protocol["purpose"] = "Reproduce reported 50+100 PASS games; V2 seat assignment assumed alternating as V2 prose does not give the exact map"
    (out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    started = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(run_one, job) for job in jobs]):
            result = future.result()
            results.append(result)
            name = f"{result['version']}_{result['seed']}_seat{result['seat']}.json"
            (out/name).write_text(json.dumps(result,indent=2),encoding="utf-8")
            if not args.full_reports or len(results)%10==0 or result["status"]!="PASS":
                print(json.dumps(dict(completed=len(results),total=len(jobs),last={k:v for k,v in result.items() if k not in ("days","forecast_probe")})),flush=True)
    summary = dict(protocol=protocol, wall_seconds=time.perf_counter()-started,
                   reports_reproduced=[{k:r[k] for k in ("version","seed","seat","cash","expected_cash","exact_report_match")} for r in results if r["group"]=="report_anchor" and r["status"]=="PASS"],
                   errors=[r for r in results if r["status"]!="PASS"],versions={})
    for version in FILES:
        rows = [r for r in results if r["version"]==version and r["group"]=="paired_fresh" and r["status"]=="PASS"]
        if rows:
            summary["versions"][version]=dict(games=len(rows),mean_cash=statistics.mean(r["cash"] for r in rows),
                min_cash=min(r["cash"] for r in rows),max_cash=max(r["cash"] for r in rows),
                max_action_ms=max(r["action_max_ms"] for r in rows),
                mean_game_action_seconds=statistics.mean(r["action_seconds"] for r in rows),
                rows=[{k:r[k] for k in ("seed","seat","cash")} for r in sorted(rows,key=lambda r:(r["seed"],r["seat"]))])
    if args.full_reports:
        summary["report_groups"] = {}
        expected_means={"v1_report50":196771.68,"v2_report_first50":193288.56,"v2_report_second50":196583.06}
        for group,expected in expected_means.items():
            rows=[r for r in results if r["group"]==group and r["status"]=="PASS"]
            if rows:
                amounts=[r["cash"] for r in rows]
                summary["report_groups"][group]=dict(games=len(rows),mean_cash=statistics.mean(amounts),
                    expected_mean_cash=expected,exact_mean_match=statistics.mean(amounts)==expected,
                    median_cash=statistics.median(amounts),min_cash=min(amounts),max_cash=max(amounts),
                    std_cash=statistics.stdev(amounts),count_at_least_190k=sum(x>=190000 for x in amounts),
                    max_action_ms=max(r["action_max_ms"] for r in rows))
    (out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
