#!/usr/bin/env python3
"""Seed-paired evaluator with honest error bars.

run_strong_ab.py reports "N/896" where 896 = 64 seeds x 7 bots x 2 seats. That is
misleading in this game: the match is symmetric and both agents are deterministic, so
seat0 and seat1 produce IDENTICAL margins in ~87% of pairs. The effective sample size
is therefore ~1.14x the number of seeds, not 2x, and every historical "+N/224" style
decision was made on roughly half the sample it appeared to have.

This script reports, per opponent:
  * wins / games                (what run_strong_ab reports)
  * independent seed units      (seeds, counting a both-seat pair as one unit)
  * win rate with a seed-clustered 95% mean CI
  * the number of additional seeds that must flip to reach the target win rate

Usage:  eval_seed_paired.py <result.json> [--target 0.80]
"""
from __future__ import annotations
import argparse, json, math, sys
from collections import defaultdict


def clustered_mean_ci(values, z=1.96):
    """CI for seed-level outcomes in {0, .5, 1}; they are not binomial counts."""
    n = len(values)
    if not n:
        return (0.0, 0.0)
    mean = sum(values) / n
    if n == 1:
        return (mean, mean)
    variance = sum((value - mean) ** 2 for value in values) / (n - 1)
    # Second-order approximation to the two-sided Student-t 97.5% quantile.
    df = n - 1
    critical = z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    radius = critical * math.sqrt(variance / n)
    return (max(0.0, mean - radius), min(1.0, mean + radius))


def seed_units(rows, label):
    """Collapse both seats into ONE unit per (bot, seed) -- the honest sample.

    A unit is a win only if BOTH seats won (0.5 if the seats disagree, which happens in
    ~13% of pairs); that keeps the unit count equal to the number of independent seeds
    instead of silently double-counting them.
    """
    per = defaultdict(dict)
    for r in rows:
        if r.get("error") or r.get("label") != label:
            continue
        per[(r["bot"], r["seed"])][r["seat"]] = r["cash"] > r["opponent_cash"]
    out = defaultdict(list)
    for (bot, seed), seats in per.items():
        if len(seats) < 2:
            continue
        vals = list(seats.values())
        out[bot].append(1.0 if all(vals) else 0.0 if not any(vals) else 0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("result")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--target", type=float, default=0.80)
    a = ap.parse_args()
    d = json.load(open(a.result))
    rows = d["rows"]

    raw = defaultdict(lambda: [0, 0])
    for r in rows:
        if r.get("error") or r.get("label") != a.label:
            continue
        raw[r["bot"]][1] += 1
        if r["cash"] > r["opponent_cash"]:
            raw[r["bot"]][0] += 1

    units = seed_units(rows, a.label)
    seeds = d.get("seed_range", [0, 0])
    print(f"file   : {a.result}   label={a.label}")
    print(f"seeds  : {seeds[0]}..{seeds[1]-1}  ({seeds[1]-seeds[0]} seeds)")
    print()
    hdr = (f"{'opponent':20s} {'raw':>10} {'raw%':>7} | {'units':>6} {'paired%':>8} "
           f"{'95% CI':>16} | {'target':>7} {'need':>6}")
    print(hdr)
    print("-" * len(hdr))
    total_need = 0
    for bot in sorted(raw):
        w, n = raw[bot]
        u = units[bot]
        k = sum(u)
        m = len(u)
        lo, hi = clustered_mean_ci(u)
        need = max(0, math.ceil(a.target * m - k))
        total_need += need
        ok = "" if k / m >= a.target else "  <-- below"
        print(f"{bot:20s} {w:4d}/{n:<5d} {100*w/n:6.1f}% | {m:6d} {100*k/m:7.1f}% "
              f"[{100*lo:5.1f},{100*hi:5.1f}] | {100*a.target:6.1f}% {need:6d}{ok}")
    print()
    print(f"seeds that must flip to reach {100*a.target:.0f}% on every opponent: {total_need}")
    print("NOTE: the CI treats each both-seat seed pair as one cluster. Fractional 0.5"
          " outcomes are bounded means, not Bernoulli successes, so Wilson is not used.")


if __name__ == "__main__":
    main()
