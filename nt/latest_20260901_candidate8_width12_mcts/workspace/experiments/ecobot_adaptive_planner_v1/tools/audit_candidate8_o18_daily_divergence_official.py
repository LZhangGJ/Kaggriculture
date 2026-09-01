#!/usr/bin/env python3
"""O1.8: locate the first material divergence from strong reference routes.

The audit replays both the complete reference action stream and forced semantic
plans with the frozen official Kaggriculture 1.32.7 environment.  It reports
three different notions of divergence:

* first action difference (diagnostic only; equivalent actions may differ),
* first player-economic-state difference,
* first material daily difference that survives to the end of a game day.

No policy is changed by this script.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from kaggle_environments import make


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (
    "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
    "EGG", "MILK", "WOOL", "FERTILIZER",
)
ALL_ITEMS = PRODUCTS + ANIMALS


def _read_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def _to_int(value: Any) -> int:
    return int(value or 0)


def _dict_counts(source: Mapping[str, Any] | None, names: Iterable[str]) -> dict[str, int]:
    source = source or {}
    return {name: _to_int(source.get(name, 0)) for name in names}


def _add_counts(target: dict[str, int], source: Mapping[str, Any] | None) -> None:
    if not source:
        return
    for name, value in source.items():
        target[str(name)] = target.get(str(name), 0) + _to_int(value)


def _state(step: Mapping[str, Any], seat: int) -> dict[str, Any]:
    obs = step["observation"]
    farm = obs["farms"][seat]
    private = obs["private"]
    crop_count = {name: 0 for name in CROPS}
    crop_yield = {name: 0 for name in CROPS}
    animal_field = {name: 0 for name in ANIMALS}
    animal_yield = {name: 0 for name in ANIMALS}
    structures = {"COOP": 0, "PASTURE": 0}
    weeds = 0
    for row in farm.get("tiles", []):
        for tile in row:
            if not isinstance(tile, Mapping):
                continue
            kind = str(tile.get("kind", ""))
            if kind == "PLANT":
                crop = str(tile.get("crop", ""))
                if crop in crop_count:
                    crop_count[crop] += 1
                    crop_yield[crop] += _to_int(tile.get("yield_units", 0))
            elif kind in structures:
                structures[kind] += 1
                animal = str(tile.get("animal", ""))
                if animal in animal_field:
                    animal_field[animal] += 1
                    animal_yield[animal] += _to_int(tile.get("yield_units", 0))
            elif kind == "WEED":
                weeds += 1

    shed = _dict_counts(private.get("shed"), ALL_ITEMS)
    seeds = _dict_counts(private.get("seeds"), CROPS)
    carried = {name: 0 for name in ALL_ITEMS}
    for inventory in private.get("inventories", []):
        _add_counts(carried, inventory)
    animal_owned = {
        name: animal_field[name] + shed.get(name, 0) + carried.get(name, 0)
        for name in ANIMALS
    }
    crop_commitment = {
        name: crop_count[name] + seeds[name]
        for name in CROPS
    }
    market = obs.get("market", {})
    positions = [list(farm.get("farmer", []))]
    positions.extend(list(value) for value in farm.get("hands", []))
    return {
        "step": _to_int(obs.get("step", 0)),
        "day": _to_int(obs.get("day", 0)),
        "hour": _to_int(obs.get("hour", 0)),
        "cash": _to_int(farm.get("money", 0)),
        "hands": len(farm.get("hands", [])),
        "hires_today": _to_int(farm.get("hires_today", 0)),
        "land": len(farm.get("unlocked_quadrants", [])),
        "structures": structures,
        "crops": crop_count,
        "crop_yield": crop_yield,
        "crop_commitment": crop_commitment,
        "animals_field": animal_field,
        "animal_yield": animal_yield,
        "animal_owned": animal_owned,
        "weeds": weeds,
        "shed": shed,
        "seeds": seeds,
        "carried": carried,
        "positions": positions,
        "market_prices": _dict_counts(market.get("prices"), PRODUCTS),
        "market_inventory": _dict_counts(market.get("inventory"), PRODUCTS),
    }


def _player_economic_projection(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: state[name]
        for name in (
            "cash", "hands", "hires_today", "land", "structures", "crops",
            "crop_yield", "crop_commitment", "animals_field", "animal_yield",
            "animal_owned", "weeds", "shed", "seeds", "carried",
        )
    }


def _l1(a: Mapping[str, Any], b: Mapping[str, Any]) -> int:
    return sum(abs(_to_int(a.get(name, 0)) - _to_int(b.get(name, 0))) for name in set(a) | set(b))


def _nested_delta(reference: Mapping[str, Any], forced: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in sorted(set(reference) | set(forced)):
        left = reference.get(name, 0)
        right = forced.get(name, 0)
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            nested = _nested_delta(
                left if isinstance(left, Mapping) else {},
                right if isinstance(right, Mapping) else {},
            )
            if nested:
                out[name] = nested
        elif left != right:
            out[name] = {"reference": left, "forced": right, "delta": _to_int(right) - _to_int(left)}
    return out


def _canonical_action(action: Mapping[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _action_ops(action: Mapping[str, Any]) -> Counter[str]:
    result: Counter[str] = Counter()
    farmer = action.get("farmer", ["PASS"])
    if farmer:
        result[f"UNIT:{farmer[0]}"] += 1
    for hand in action.get("hands", []):
        if hand:
            result[f"UNIT:{hand[0]}"] += 1
    for order in action.get("market", []):
        if order:
            key = f"MARKET:{order[0]}"
            if len(order) > 1:
                key += f":{order[1]}"
            quantity = _to_int(order[2]) if len(order) > 2 else 1
            result[key] += quantity
    return result


def _market_signature(action: Mapping[str, Any]) -> list[list[Any]]:
    return [list(order) for order in action.get("market", [])]


def _unit_signature(action: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "farmer": list(action.get("farmer", ["PASS"])),
        "hands": [list(value) for value in action.get("hands", [])],
    }


def _replay(game: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": int(game["seed"])},
        debug=False,
    )
    env.reset(2)
    for joint_action in game["trace"]:
        env.step(joint_action)
    replay = env.toJSON()
    seat = int(game["candidate_seat"])
    states = [_state(frame[seat], seat) for frame in replay["steps"]]
    return replay, states


def _first(predicate, length: int) -> int | None:
    for index in range(length):
        if predicate(index):
            return index
    return None


def _transition_context(
    step: int | None,
    reference_trace: list[Any],
    forced_trace: list[Any],
    seat: int,
    reference_states: list[dict[str, Any]],
    forced_states: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if step is None:
        return None
    reference_action = reference_trace[step][seat]
    forced_action = forced_trace[step][seat]
    return {
        "transition_step": step,
        "day": step // 24,
        "hour": step % 24,
        "reference_action": reference_action,
        "forced_action": forced_action,
        "market_differs": _market_signature(reference_action) != _market_signature(forced_action),
        "unit_differs": _unit_signature(reference_action) != _unit_signature(forced_action),
        "post_state_delta": _nested_delta(
            _player_economic_projection(reference_states[step + 1]),
            _player_economic_projection(forced_states[step + 1]),
        ),
    }


def _daily_summary(states: list[dict[str, Any]], trace: list[Any], seat: int, day: int) -> dict[str, Any]:
    transition_start = day * 24
    transition_stop = min((day + 1) * 24, len(trace))
    frame_start = transition_start
    frame_stop = min(transition_stop + 1, len(states))
    frames = states[frame_start:frame_stop]
    end = states[min(transition_stop, len(states) - 1)]
    action_ops: Counter[str] = Counter()
    market_sequences: list[list[list[Any]]] = []
    for step in range(transition_start, transition_stop):
        action = trace[step][seat]
        action_ops.update(_action_ops(action))
        if action.get("market"):
            market_sequences.append(_market_signature(action))
    return {
        "day": day,
        "cash_start": states[frame_start]["cash"],
        "cash_end": end["cash"],
        "cash_min": min(frame["cash"] for frame in frames),
        "cash_max": max(frame["cash"] for frame in frames),
        "peak_hands": max(frame["hands"] for frame in frames),
        "land_end": end["land"],
        "structures_end": end["structures"],
        "crops_end": end["crops"],
        "crop_yield_end": end["crop_yield"],
        "animals_end": end["animal_owned"],
        "animal_yield_end": end["animal_yield"],
        "shed_end": end["shed"],
        "seeds_end": end["seeds"],
        "weeds_end": end["weeds"],
        "action_ops": dict(sorted(action_ops.items())),
        "market_sequences": market_sequences,
    }


def _daily_delta(reference: Mapping[str, Any], forced: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "day": int(reference["day"]),
        "cash_end_gap": int(forced["cash_end"]) - int(reference["cash_end"]),
        "cash_min_gap": int(forced["cash_min"]) - int(reference["cash_min"]),
        "peak_hands_gap": int(forced["peak_hands"]) - int(reference["peak_hands"]),
        "land_gap": int(forced["land_end"]) - int(reference["land_end"]),
        "structure_l1": _l1(reference["structures_end"], forced["structures_end"]),
        "crop_l1": _l1(reference["crops_end"], forced["crops_end"]),
        "crop_yield_l1": _l1(reference["crop_yield_end"], forced["crop_yield_end"]),
        "animal_l1": _l1(reference["animals_end"], forced["animals_end"]),
        "animal_yield_l1": _l1(reference["animal_yield_end"], forced["animal_yield_end"]),
        "shed_l1": _l1(reference["shed_end"], forced["shed_end"]),
        "seed_l1": _l1(reference["seeds_end"], forced["seeds_end"]),
        "market_order_counter_diff": _nested_delta(reference["action_ops"], forced["action_ops"]),
        "market_sequence_exact": reference["market_sequences"] == forced["market_sequences"],
    }


def _material_day(delta: Mapping[str, Any]) -> bool:
    return (
        abs(int(delta["cash_end_gap"])) >= 1000
        or int(delta["peak_hands_gap"]) != 0
        or int(delta["land_gap"]) != 0
        or int(delta["structure_l1"]) > 0
        or int(delta["crop_l1"]) >= 2
        or int(delta["animal_l1"]) > 0
        or int(delta["shed_l1"]) >= 2
        or int(delta["seed_l1"]) >= 2
    )


def _root_category(context: Mapping[str, Any] | None) -> str:
    if not context:
        return "NO_MATERIAL_DIVERGENCE"
    market = bool(context["market_differs"])
    unit = bool(context["unit_differs"])
    if market and unit:
        return "COUPLED_TRANSACTION_AND_UNIT_EXECUTION"
    if market:
        return "MARKET_TRANSACTION_COMPILATION"
    if unit:
        return "UNIT_SCHEDULING_OR_LAYOUT"
    return "STATE_TRANSITION_OR_SEMANTIC_TARGET"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-traces", required=True, type=Path)
    parser.add_argument("--forced-traces", required=True, type=Path)
    parser.add_argument("--o17-audit", required=True, type=Path)
    parser.add_argument("--case-ids", required=True, type=int, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    version = importlib.metadata.version("kaggle-environments")
    if version != "1.32.7":
        raise RuntimeError(f"official package 1.32.7 required, got {version}")
    reference_bundle = _read_gzip_json(args.reference_traces)
    forced_bundle = _read_gzip_json(args.forced_traces)
    o17 = json.loads(args.o17_audit.read_text(encoding="utf-8"))
    references = {int(game["case_id"]): game for game in reference_bundle["games"]}
    forced_games: dict[tuple[int, str], dict[str, Any]] = {}
    for game in forced_bundle["games"]:
        arm = str(game["arm"])
        prefix, profile = arm.split("_", 1)
        forced_games[(int(prefix.removeprefix("case")), profile)] = game
    o17_rows = {int(row["case_id"]): row for row in o17["rows"]}

    rows: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()
    for progress, case_id in enumerate(args.case_ids, start=1):
        reference_game = references[case_id]
        reference_replay, reference_states = _replay(reference_game)
        native_reference_rewards = [float(value) for value in reference_game["native_rewards"]]
        official_reference_rewards = [float(value) for value in reference_replay["rewards"]]
        if official_reference_rewards != native_reference_rewards:
            raise RuntimeError(f"case {case_id} reference official/native mismatch")
        seat = int(reference_game["candidate_seat"])
        profiles: dict[str, Any] = {}
        for profile in ("full", "target", "milestone"):
            forced_game = forced_games[(case_id, profile)]
            forced_replay, forced_states = _replay(forced_game)
            native_forced_rewards = [float(value) for value in forced_game["native_rewards"]]
            official_forced_rewards = [float(value) for value in forced_replay["rewards"]]
            if official_forced_rewards != native_forced_rewards:
                raise RuntimeError(f"case {case_id} {profile} official/native mismatch")
            reference_trace = reference_game["trace"]
            forced_trace = forced_game["trace"]
            first_action = _first(
                lambda index: _canonical_action(reference_trace[index][seat])
                != _canonical_action(forced_trace[index][seat]),
                len(reference_trace),
            )
            first_market_action = _first(
                lambda index: _market_signature(reference_trace[index][seat])
                != _market_signature(forced_trace[index][seat]),
                len(reference_trace),
            )
            first_unit_action = _first(
                lambda index: _unit_signature(reference_trace[index][seat])
                != _unit_signature(forced_trace[index][seat]),
                len(reference_trace),
            )
            first_economic = _first(
                lambda index: _player_economic_projection(reference_states[index + 1])
                != _player_economic_projection(forced_states[index + 1]),
                len(reference_trace),
            )
            first_structural = _first(
                lambda index: any(
                    reference_states[index + 1][name] != forced_states[index + 1][name]
                    for name in ("hands", "land", "structures", "crops", "animal_owned")
                ),
                len(reference_trace),
            )
            first_cash_1k = _first(
                lambda index: abs(reference_states[index + 1]["cash"] - forced_states[index + 1]["cash"]) >= 1000,
                len(reference_trace),
            )
            first_cash_5k = _first(
                lambda index: abs(reference_states[index + 1]["cash"] - forced_states[index + 1]["cash"]) >= 5000,
                len(reference_trace),
            )
            daily_reference = [_daily_summary(reference_states, reference_trace, seat, day) for day in range(30)]
            daily_forced = [_daily_summary(forced_states, forced_trace, seat, day) for day in range(30)]
            daily_deltas = [
                _daily_delta(daily_reference[day], daily_forced[day])
                for day in range(30)
            ]
            first_material_day = _first(lambda day: _material_day(daily_deltas[day]), 30)
            first_material_step_candidates = [
                value for value in (first_structural, first_cash_1k, first_economic)
                if value is not None
            ]
            first_material_step = min(first_material_step_candidates) if first_material_step_candidates else None
            context = _transition_context(
                first_material_step,
                reference_trace,
                forced_trace,
                seat,
                reference_states,
                forced_states,
            )
            category = _root_category(context)
            category_counts[category] += 1
            profiles[profile] = {
                "official_rewards": official_forced_rewards,
                "candidate_margin": official_forced_rewards[seat] - official_forced_rewards[1 - seat],
                "official_native_reward_exact": True,
                "complete": len(forced_replay["steps"]) == 720 and forced_replay["statuses"] == ["DONE", "DONE"],
                "first_action_divergence_step": first_action,
                "first_market_action_divergence_step": first_market_action,
                "first_unit_action_divergence_step": first_unit_action,
                "first_economic_state_divergence_step": first_economic,
                "first_structural_divergence_step": first_structural,
                "first_cash_gap_1000_step": first_cash_1k,
                "first_cash_gap_5000_step": first_cash_5k,
                "first_material_day": first_material_day,
                "root_category": category,
                "first_material_context": context,
                "daily": [
                    {
                        "reference": daily_reference[day],
                        "forced": daily_forced[day],
                        "delta": daily_deltas[day],
                    }
                    for day in range(30)
                ],
            }
        best_profile = max(
            profiles,
            key=lambda name: float(profiles[name]["candidate_margin"]),
        )
        rows.append(
            {
                "case_id": case_id,
                "opponent": str(reference_game["opponent"]),
                "seed": int(reference_game["seed"]),
                "seat": seat,
                "reference_route": str(reference_game["reference_route"]),
                "official_reference_rewards": official_reference_rewards,
                "reference_margin": official_reference_rewards[seat] - official_reference_rewards[1 - seat],
                "best_forced_profile": best_profile,
                "profiles": profiles,
            }
        )
        print(json.dumps({"progress": progress, "case": case_id, "best_profile": best_profile}), flush=True)

    payload = {
        "schema": "kaggriculture.candidate8-o18-daily-divergence.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "boundary": "Read-only diagnosis. No formal policy, planner, route or executor was changed.",
        "official_package_version": version,
        "case_ids": args.case_ids,
        "cases": len(rows),
        "profile_runs": len(rows) * 3,
        "official_native_reward_exact": True,
        "root_category_counts": dict(sorted(category_counts.items())),
        "material_day_definition": (
            "At day end: abs cash gap >=1000, different hand/land/structure/animal scale, "
            "crop count L1 >=2, or shed/seed L1 >=2."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
