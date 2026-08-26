from __future__ import annotations

import copy
import json
from pathlib import Path

from meta_agent.src.route_compiler import compile_route_genome
from meta_agent.src.route_plan import CarrierRoute


def _replay(tmp_path: Path) -> Path:
    farm = {
        "money": 3000,
        "farmer": [1, 1],
        "hands": [],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }
    observation = {
        "player": 0,
        "step": 0,
        "farms": [farm, copy.deepcopy(farm)],
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }
    steps = []
    for step in range(5):
        left = {"observation": copy.deepcopy(observation), "action": {}}
        right = {"observation": copy.deepcopy(observation), "action": {}}
        left["observation"]["step"] = step
        right["observation"]["step"] = step
        steps.append([left, right])
    steps[1][0]["action"] = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [],
        "market": [["BUY_SEED", "WHEAT", 1]],
    }
    path = tmp_path / "episode-1-replay.json"
    path.write_text(json.dumps({
        "info": {"EpisodeId": 1, "seed": 2},
        "configuration": {},
        "rewards": [1, 0],
        "steps": steps,
    }), encoding="utf-8")
    return path


def _genome(genome_id: str, crop: str, market: list[dict]) -> dict:
    return {
        "genome_id": genome_id,
        "genetic_payload": {
            "anchor_targets": [{
                "step": 168,
                "counts": {crop: 1},
                "placements": [{"x": 1, "y": 1, "kind": crop}],
            }],
            "phase_macro_counts": [],
            "structural_events": [],
            "market_actions": market,
        },
    }


def test_crop_substitution_patches_plant_and_seed_prerequisite(tmp_path: Path) -> None:
    carrier = CarrierRoute.from_replay(_replay(tmp_path), 0, horizon=4)
    parent = _genome("parent", "WHEAT", [
        {"operation": "BUY_SEED", "item": "WHEAT", "quantity": 1}
    ])
    target = _genome("target", "CARROT", [
        {"operation": "BUY_SEED", "item": "WHEAT", "quantity": 1}
    ])
    target["experiment"] = {
        "operator": "crop_substitute",
        "detail": "substitute (1,1) WHEAT->CARROT across 1 anchors",
    }
    plan = compile_route_genome(carrier, parent, target)
    assert plan.actions[0]["farmer"] == ["PASS"]
    assert not any(
        order[:2] == ["BUY_SEED", "WHEAT"] for order in plan.actions[0]["market"]
    )
    assert len(plan.tasks) == 1
    assert plan.tasks[0].operation == "PLANT"
    assert plan.tasks[0].item == "CARROT"
    assert plan.report.status == "compiled"


def test_market_quantity_delta_changes_only_aggregate_quantity(tmp_path: Path) -> None:
    carrier = CarrierRoute.from_replay(_replay(tmp_path), 0, horizon=4)
    parent = _genome("parent", "WHEAT", [
        {"operation": "BUY_SEED", "item": "WHEAT", "quantity": 1}
    ])
    target = _genome("target", "WHEAT", [
        {"operation": "BUY_SEED", "item": "WHEAT", "quantity": 4}
    ])
    target["experiment"] = {
        "operator": "market_quantity",
        "detail": "BUY_SEED:WHEAT quantity 1->4",
    }
    plan = compile_route_genome(carrier, parent, target)
    assert plan.actions[0]["market"] == [["BUY_SEED", "WHEAT", 4]]
    assert plan.report.static_market_quantity_delta == {"BUY_SEED:WHEAT": 3}


def test_adjacent_relocation_emits_state_aware_detour(tmp_path: Path) -> None:
    carrier = CarrierRoute.from_replay(_replay(tmp_path), 0, horizon=4)
    parent = _genome("parent", "WHEAT", [])
    target = _genome("target", "WHEAT", [])
    target["experiment"] = {
        "operator": "layout_relocate",
        "detail": "relocate WHEAT (1,1)->(2,1)",
    }
    plan = compile_route_genome(carrier, parent, target)
    assert plan.actions[0]["farmer"] == ["PASS"]
    assert len(plan.tasks) == 1
    task = plan.tasks[0]
    assert task.target_xy == (2, 1)
    assert task.return_xy == (1, 1)
    assert task.operation == "PLANT"
