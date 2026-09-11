#!/usr/bin/env python3
"""Explain where an official two-player trace gains or loses score margin.

The input is the gzip JSONL emitted by ``generate_official_single_trace.py``.
This is a diagnostic tool: it does not modify or replay either agent.
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]


def resolve(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_trace(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows or rows[0].get("record_type") != "header":
        raise ValueError(f"Malformed trace: {path}")
    frames = rows[1:]
    if len(frames) != 720 or frames[-1].get("status") != ["DONE", "DONE"]:
        raise ValueError(f"Trace is not a complete 720-frame game: {path}")
    return rows[0], frames


def compact_market(action: dict[str, Any]) -> list[list[Any]]:
    return [list(item) for item in action.get("market", [])]


def market_totals(action: dict[str, Any], verb: str) -> dict[str, int]:
    totals: defaultdict[str, int] = defaultdict(int)
    for item in action.get("market", []):
        if item and item[0] == verb and len(item) >= 3:
            totals[str(item[1])] += int(item[2])
    return dict(sorted(totals.items()))


def money(frame: dict[str, Any], seat: int) -> float:
    return float(frame["farms"][seat]["money"])


def summarize_one(path: Path, top_n: int) -> dict[str, Any]:
    header, frames = load_trace(path)
    candidate = int(header["candidate_seat"])
    opponent = 1 - candidate
    first_margin = money(frames[0], candidate) - money(frames[0], opponent)
    previous_margin = first_margin
    previous_money = [money(frames[0], 0), money(frames[0], 1)]
    events: list[dict[str, Any]] = []
    cumulative_sell = [Counter(), Counter()]
    cumulative_buy = [Counter(), Counter()]

    for frame in frames[1:]:
        current_money = [money(frame, 0), money(frame, 1)]
        margin = current_money[candidate] - current_money[opponent]
        delta = margin - previous_margin
        actions = frame["actions"]
        for seat in (0, 1):
            cumulative_sell[seat].update(market_totals(actions[seat], "SELL"))
            for item in actions[seat].get("market", []):
                if item and str(item[0]).startswith("BUY"):
                    key = ":".join(str(value) for value in item[:2])
                    quantity = int(item[2]) if len(item) >= 3 else 1
                    cumulative_buy[seat][key] += quantity
        if delta or any(compact_market(actions[seat]) for seat in (0, 1)):
            events.append({
                "frame": int(frame["frame"]),
                "day": int(frame["day"]),
                "hour": int(frame["hour"]),
                "candidate_money": current_money[candidate],
                "opponent_money": current_money[opponent],
                "candidate_money_delta": current_money[candidate] - previous_money[candidate],
                "opponent_money_delta": current_money[opponent] - previous_money[opponent],
                "margin": margin,
                "margin_delta": delta,
                "candidate_market": compact_market(actions[candidate]),
                "opponent_market": compact_market(actions[opponent]),
                "candidate_sell": market_totals(actions[candidate], "SELL"),
                "opponent_sell": market_totals(actions[opponent], "SELL"),
                "prices": dict(frame["market"]["prices"]),
                "inventory": dict(frame["market"]["inventory"]),
            })
        previous_margin = margin
        previous_money = current_money

    negative = sorted(events, key=lambda row: (row["margin_delta"], row["frame"]))[:top_n]
    positive = sorted(events, key=lambda row: (-row["margin_delta"], row["frame"]))[:top_n]
    terminal_rewards = [float(value) for value in frames[-1]["reward"]]
    reward_margin = terminal_rewards[candidate] - terminal_rewards[opponent]

    return {
        "trace": str(path),
        "official_package_version": header["official_package_version"],
        "seed": int(header["seed"]),
        "candidate_seat": candidate,
        "candidate": header["candidate"],
        "opponent": header["opponent"],
        "frames": len(frames),
        "terminal_status": frames[-1]["status"],
        "terminal_rewards": terminal_rewards,
        "terminal_reward_margin": reward_margin,
        "terminal_farm_money": [money(frames[-1], 0), money(frames[-1], 1)],
        "terminal_farm_money_margin": money(frames[-1], candidate) - money(frames[-1], opponent),
        "candidate_sell_totals": dict(sorted(cumulative_sell[candidate].items())),
        "opponent_sell_totals": dict(sorted(cumulative_sell[opponent].items())),
        "candidate_buy_totals": dict(sorted(cumulative_buy[candidate].items())),
        "opponent_buy_totals": dict(sorted(cumulative_buy[opponent].items())),
        "top_negative_margin_events": negative,
        "top_positive_margin_events": positive,
        "market_events": events,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    traces = [resolve(path) for path in args.traces]
    result = {
        "schema": "kaggriculture-official-loss-trace-analysis-v1",
        "analyses": [summarize_one(path, args.top) for path in traces],
    }
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = resolve(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
