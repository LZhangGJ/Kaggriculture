#!/usr/bin/env python3
"""Prefix-only audit of PublicFlowScenario rival-flow realization."""
import argparse
import ctypes
import importlib.util
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlation(xs, ys):
    if len(xs) < 2:
        return 0.0
    ax, ay = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x - ax) ** 2 for x in xs)
    yy = sum((y - ay) ** 2 for y in ys)
    return 0.0 if xx == 0 or yy == 0 else sum((x - ax) * (y - ay) for x, y in zip(xs, ys)) / math.sqrt(xx * yy)


def checkpoint(bot, seed, seat, binary):
    os.environ["R1_BINARY_PATH"] = str(binary)
    hybrid = load(ROOT / "agent/main.py", f"hybrid_flow_{bot}_{seed}_{seat}").create_agent(seat)
    rival = load(ROOT / "opponents" / bot / "main.py", f"rival_flow_{bot}_{seed}_{seat}").agent
    from fast_kaggriculture import Config, FastEnv
    env = FastEnv(Config(), seed)
    state = list(env.reset(seed))
    while state[0]["day"] * 24 + state[0]["hour"] < 288:
        actions = []
        for player in (0, 1):
            obs = json.loads(json.dumps(state[player]));obs["player"] = player
            obs["step"] = int(obs["day"]) * 24 + int(obs["hour"])
            actions.append(hybrid(obs, {}) if player == seat else rival(obs, {}))
        state = env.step(actions)
    obs = json.loads(json.dumps(state[seat]));obs["player"] = seat;obs["step"] = 288
    start_rival_cash = float(obs["farms"][1 - seat]["money"])
    rows = hybrid.dynamic.prepare_candidates(obs)
    lib = hybrid.dynamic.lib
    lib.td_candidate_flow_json.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.td_candidate_flow_json.restype = ctypes.c_char_p
    for row in rows:
        flow = json.loads(lib.td_candidate_flow_json(hybrid.dynamic.handle, row["index"]))
        row["flow"] = flow
        row["rounding_l1"] = sum(abs(a - b) for a, b in zip(flow["requested"], flow["rounded"]))
        row["fill_shortfall_l1"] = sum(abs(a - b) for a, b in zip(flow["rounded"], flow["filled"]))
        row["rival_flow_cash"] = sum(flow["cash_delta"])
        row["cash_conservation_error"] = row["scenario_rollout_rival_cash"] - start_rival_cash - row["rival_flow_cash"]
        row["market_floor_nonadvance_l1"] = sum(abs(a - b) for a, b in zip(flow["filled"], flow["market_delta"]))
    hybrid.close()
    return {"bot": bot, "seed": seed, "seat": seat,
            "start_rival_cash": start_rival_cash, "candidates": rows}


def summarize(states):
    per_state = []
    centered_score, centered_rounding, centered_shortfall, centered_cash = [], [], [], []
    for state in states:
        rows = [r for r in state["candidates"] if not r.get("diagnostic")]
        top = sorted((r["score"] for r in rows), reverse=True)
        def span(key):
            values = [r[key] for r in rows]
            return max(values) - min(values)
        per_state.append({"bot": state["bot"], "seed": state["seed"], "seat": state["seat"],
                          "candidates": len(rows), "top2_gap": top[0] - top[1],
                          "rounding_l1_range": span("rounding_l1"),
                          "fill_shortfall_l1_range": span("fill_shortfall_l1"),
                          "rival_flow_cash_range": span("rival_flow_cash"),
                          "max_abs_cash_conservation_error": max(abs(r["cash_conservation_error"]) for r in rows),
                          "max_market_floor_nonadvance_l1": max(r["market_floor_nonadvance_l1"] for r in rows)})
        means = {key: sum(r[key] for r in rows) / len(rows)
                 for key in ("score", "rounding_l1", "fill_shortfall_l1", "rival_flow_cash")}
        for row in rows:
            centered_score.append(row["score"] - means["score"])
            centered_rounding.append(row["rounding_l1"] - means["rounding_l1"])
            centered_shortfall.append(row["fill_shortfall_l1"] - means["fill_shortfall_l1"])
            centered_cash.append(row["rival_flow_cash"] - means["rival_flow_cash"])
    return {"states": per_state,
            "within_state_correlations": {
                "score_vs_rounding_l1": correlation(centered_score, centered_rounding),
                "score_vs_fill_shortfall_l1": correlation(centered_score, centered_shortfall),
                "score_vs_rival_flow_cash": correlation(centered_score, centered_cash)}}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--binary", type=Path, default=ROOT / "work/agent-flow-audit.so")
    p.add_argument("--output", type=Path, default=ROOT / "work/public-flow-audit-4states.json")
    p.add_argument("--seeds", default="2609800000,2609800001")
    args = p.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value]
    states = [checkpoint(bot, seed, 0, args.binary.resolve())
              for bot in ("thomas_2945", "melon_2749") for seed in seeds]
    payload = {"summary": summarize(states), "records": states}
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
