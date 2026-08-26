from __future__ import annotations

from dataclasses import replace

from meta_agent.src.generative_route import (
    CROPS,
    GenerativeRouteAgent,
    default_seed_genomes,
    genome_from_route_summary,
    materialize_animal_targets,
    materialize_targets,
    repair_genome,
)
from meta_agent.src.route_evolution import (
    AdaptiveOperatorPool,
    applicable_mutation_operators,
    mutate_genome,
)


def _observation(genome, *, seeds=None, tile=None):
    tiles = [[None for _ in range(10)] for _ in range(10)]
    target = materialize_targets(genome)[0]
    tiles[target.y][target.x] = tile
    farm = {
        "money": 3000,
        "farmer": [target.x, target.y],
        "hands": [],
        "tiles": tiles,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    return {
        "player": 0,
        "farms": [farm, dict(farm)],
        "private": {
            "shed": {crop: 0 for crop in CROPS},
            "seeds": {crop: int((seeds or {}).get(crop, 0)) for crop in CROPS},
            "inventories": [{}],
        },
        "market": {"prices": {crop: 100 for crop in CROPS}},
        "day": 0,
        "hour": 0,
    }


def test_behavioral_id_ignores_provenance() -> None:
    genome = default_seed_genomes()[0]
    assert genome.genome_id == replace(genome, source="another-replay").genome_id
    assert genome.genome_id == "GRG1-99ff0138df1f21c3"


def test_repair_makes_counts_monotonic() -> None:
    genome = default_seed_genomes()[0]
    broken = replace(
        genome,
        crop_counts=((9, 0, 0, 0, 0), (1, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    )
    fixed = repair_genome(broken)
    assert fixed.crop_counts[0][0] == 9
    assert fixed.crop_counts[1][0] >= fixed.crop_counts[0][0]
    assert fixed.crop_counts[2][0] >= fixed.crop_counts[1][0]


def test_agent_generates_market_and_state_actions_without_tape() -> None:
    genome = default_seed_genomes()[0]
    target = materialize_targets(genome)[0]
    observation = _observation(genome)
    action = GenerativeRouteAgent(genome)(observation)
    assert any(order[0] == "BUY_SEED" for order in action["market"])
    assert any(order[0] == "HIRE" for order in action["market"])

    observation = _observation(genome, seeds={target.crop: 1})
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["PLANT", target.crop]

    plant = {
        "kind": "PLANT",
        "crop": target.crop,
        "planted_day": 0,
        "watered_today": False,
        "consecutive_unwatered": 0,
        # One-shot crops carry one latent yield unit immediately; it is not
        # harvestable until max_yield_day.
        "yield_units": 1,
    }
    observation = _observation(genome, tile=plant)
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["WATER"]

    observation["day"] = 4
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["HARVEST"]

    # The route fits in NW; a land date is permission, not an unconditional buy.
    observation = _observation(genome)
    observation["day"] = 29
    action = GenerativeRouteAgent(genome)(observation)
    assert not any(order[0] == "BUY_LAND" for order in action["market"])


def test_replay_bootstrap_uses_only_macro_anchors() -> None:
    summary = {
        "genome_id": "RG2-example",
        "genetic_payload": {
            "anchor_targets": [
                {"step": 168, "counts": {"CARROT": 2}},
                {"step": 719, "counts": {"CARROT": 7, "MELON": 3}},
            ],
            "actions": [["THIS", "MUST", "NOT", "BE", "USED"]],
        },
    }
    genome = genome_from_route_summary(summary)
    assert genome.crop_counts[-1][CROPS.index("CARROT")] == 7
    assert genome.crop_counts[-1][CROPS.index("MELON")] == 3
    assert genome.source == "replay-macro:RG2-example"


def test_adaptive_pool_updates_operator_feedback() -> None:
    import random

    rng = random.Random(3)
    genome = default_seed_genomes()[0]
    child = mutate_genome(genome, "labour", rng)
    assert child.genome_id != genome.genome_id
    pool = AdaptiveOperatorPool()
    pool.update("labour", True, 12.0)
    assert pool.to_dict()["labour"]["successes"] == 1
    assert "land_timing" not in applicable_mutation_operators(genome)


def test_animal_policy_builds_buys_picks_up_and_places() -> None:
    genome = default_seed_genomes()[4]
    target = materialize_animal_targets(genome)[0]
    tiles = [[None for _ in range(10)] for _ in range(10)]
    farm = {
        "money": 3000,
        "farmer": [target.x, target.y],
        "hands": [],
        "tiles": tiles,
        "unlocked_quadrants": ["NW"],
        "hires_today": 0,
    }
    private = {
        "shed": {crop: 0 for crop in (*CROPS, "GOOSE", "COW", "SHEEP")},
        "seeds": {crop: 0 for crop in CROPS},
        "inventories": [{}],
    }
    observation = {
        "player": 0,
        "farms": [farm, dict(farm)],
        "private": private,
        "market": {"prices": {crop: 100 for crop in (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")}},
        "day": 4,
        "hour": 0,
    }
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["BUILD_COOP"]
    assert any(order[:2] == ["BUY_ANIMAL", "GOOSE"] for order in action["market"])

    tiles[target.y][target.x] = {"kind": "COOP"}
    farm["farmer"] = [4, 4]
    private["shed"]["GOOSE"] = 1
    observation["hour"] = 1
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["PICKUP", "GOOSE", 1]

    farm["farmer"] = [target.x, target.y]
    private["shed"]["GOOSE"] = 0
    private["inventories"] = [{"GOOSE": 1}]
    action = GenerativeRouteAgent(genome)(observation)
    assert action["farmer"] == ["PLACE", "GOOSE"]
