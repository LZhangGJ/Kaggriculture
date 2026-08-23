from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "references/public_latest6_20260822/prvsiyan_moon_v92_latest/main.py"


def load_source():
    spec = importlib.util.spec_from_file_location("moon_v92_pipeline_audit", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def observation(frame: dict, player: int) -> dict:
    return {
        "step": frame["step"],
        "day": frame["day"],
        "hour": frame["hour"],
        "player": player,
        "farms": frame["farms"],
        "private": frame["private"][player],
        "market": frame["market"],
        "town": frame["town"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--player", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    args = parser.parse_args()
    with gzip.open(args.trace, "rt", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    frames = [row for row in records if row.get("record_type") == "frame"]
    module = load_source()
    for step in range(args.step):
        module.agent(observation(frames[step], args.player))

    obs = observation(frames[args.step], args.player)
    step = args.step
    module._observe_opponent_market(obs, step)
    actions = module._kawa_actions(obs)
    action = module._weed_repair_action(obs, module._copy_action(actions[step]), step)
    stages = [
        ("raw_weed", action),
    ]
    for name, function, extra in (
        ("feed", module._v17_feed_guard, (step,)),
        ("room_evac", module._v17_room_evac, (step,)),
        ("repay", module._repay_shift, (step,)),
        ("rank", module._rank_sell_slots, (None,)),
        ("preempt", module._preempt_shift, (step,)),
        ("r5", module._v17_r5_counter, (step,)),
        ("md", module._v17_md_counter, (step,)),
        ("v38_tomato", module._v38_farm_pair_tomatoes, (step,)),
        ("v73_tomato", module._v73_route_tomatoes, (step,)),
        ("v71_carrot", module._v71_route_carrots, (step,)),
        ("v79_strawberry", module._v79_rival_strawberry_flush, (step,)),
        ("v88_tomato", module._v88_late_tomato_overlay, (step,)),
        ("v35_egg", module._v35_egg_late_pair, (step,)),
        ("room_guard", module._v17_room_guard, (step,)),
        ("v88_capacity", module._v88_capacity_overlay, (step,)),
        ("liquidation", module._terminal_liquidation, (step,)),
    ):
        action = function(obs, action, *extra)
        stages.append((name, action))
    previous = None
    for name, value in stages:
        market = value.get("market", [])
        units = [value.get("farmer"), *(value.get("hands") or [])]
        snapshot = (market, units)
        if snapshot != previous:
            print(json.dumps({"stage": name, "market": market, "units": units}, ensure_ascii=False))
        previous = snapshot
    raw_market = stages[0][1].get("market", [])
    print(
        json.dumps(
            {
                "source_scores": [
                    [order, module._order_score(obs, None, order)]
                    for order in raw_market
                    if module._is_sell(order)
                ]
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
