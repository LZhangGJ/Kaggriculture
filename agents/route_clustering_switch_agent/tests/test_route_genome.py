from __future__ import annotations

import copy
import unittest

from meta_agent.src.route_genome import RouteGenome, extract_route_genome


def _farm(money: float, farmer: tuple[int, int], hands: list[tuple[int, int]] | None = None) -> dict:
    return {
        "money": money,
        "farmer": list(farmer),
        "hands": [list(value) for value in (hands or [])],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }


def _step(step: int, action0: dict | None = None, money: float = 3000.0) -> list[dict]:
    observation0 = {
        "step": step,
        "player": 0,
        "farms": [_farm(money, (2, 3), [(4, 5)]), _farm(3000.0, (0, 0))],
        "market": {
            "prices": {"MILK": 190, "WHEAT": 23},
            "inventory": {},
        },
    }
    observation1 = copy.deepcopy(observation0)
    observation1["player"] = 1
    return [
        {"observation": observation0, "action": action0 or {}, "reward": 0.0, "status": "ACTIVE"},
        {"observation": observation1, "action": {}, "reward": 0.0, "status": "ACTIVE"},
    ]


def _replay() -> dict:
    steps = [
        _step(0),
        _step(1, {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}, 2900.0),
        _step(
            2,
            {
                "farmer": ["BUILD_PASTURE"],
                "hands": [["PLACE", "COW"]],
                "market": [["BUY_LAND"], ["HIRE"], ["SELL", "MILK", 2]],
            },
            1800.0,
        ),
        _step(3, {"farmer": ["PASS"], "hands": [["PASS"]], "market": []}, 2200.0),
    ]
    steps[-1][0]["reward"] = 100.0
    steps[-1][1]["reward"] = 50.0
    return {
        "id": 123,
        "info": {"EpisodeId": 123, "TeamNames": ["alpha", "beta"]},
        "rewards": [100.0, 50.0],
        "steps": steps,
    }


class RouteGenomeTest(unittest.TestCase):
    def test_failed_intent_still_enters_anchor_layout(self) -> None:
        genome = extract_route_genome(
            _replay(),
            0,
            provenance={"datasets": ["fixture"]},
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )

        self.assertEqual(genome.source["source_id"], "123:0")
        self.assertEqual(genome.source["result"], "win")
        first = genome.anchor_targets[0]
        self.assertEqual(first["placements"], [{"x": 2, "y": 3, "kind": "WHEAT"}])
        self.assertEqual(first["counts"]["WHEAT"], 1)

        second = genome.anchor_targets[1]
        self.assertIn({"x": 2, "y": 3, "kind": "PASTURE"}, second["placements"])
        self.assertIn({"x": 4, "y": 5, "kind": "COW"}, second["placements"])
        self.assertEqual(genome.phase_macro_counts[0]["counts"]["WHEAT"], 1)
        self.assertEqual(genome.phase_macro_counts[0]["counts"]["BUILD_PASTURE"], 1)
        self.assertEqual(genome.phase_macro_counts[0]["counts"]["COW"], 1)
        self.assertEqual(genome.phase_macro_counts[0]["counts"]["BUY_LAND"], 1)
        self.assertEqual(genome.phase_macro_counts[0]["counts"]["HIRE"], 1)

        milk = next(
            value
            for value in genome.market_profile
            if value["operation"] == "SELL" and value["item"] == "MILK"
        )
        self.assertEqual(milk["quantity"], 2)
        self.assertEqual(milk["observed_price"]["median"], 190.0)

    def test_genome_id_excludes_outcome_and_provenance(self) -> None:
        replay = _replay()
        first = extract_route_genome(
            replay,
            0,
            provenance={"datasets": ["first"]},
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        replay["rewards"] = [-1.0, 999.0]
        replay["info"]["TeamNames"] = ["renamed", "other"]
        second = extract_route_genome(
            replay,
            0,
            provenance={"datasets": ["second"]},
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        self.assertEqual(first.genome_id, second.genome_id)
        self.assertNotEqual(first.source["result"], second.source["result"])

    def test_genome_id_excludes_observed_prices_but_keeps_market_actions(self) -> None:
        replay = _replay()
        first = extract_route_genome(
            replay,
            0,
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        changed_prices = copy.deepcopy(replay)
        for step in changed_prices["steps"]:
            step[0]["observation"]["market"]["prices"]["MILK"] = 999
        second = extract_route_genome(
            changed_prices,
            0,
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        self.assertEqual(first.genome_id, second.genome_id)
        self.assertNotEqual(first.market_profile, second.market_profile)

        changed_quantity = copy.deepcopy(replay)
        changed_quantity["steps"][2][0]["action"]["market"][2][2] = 3
        third = extract_route_genome(
            changed_quantity,
            0,
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        self.assertNotEqual(first.genome_id, third.genome_id)

    def test_json_round_trip_validates_content_hash(self) -> None:
        genome = extract_route_genome(
            _replay(),
            0,
            anchor_steps=(1, 2, 3),
            phase_width=2,
            horizon=3,
        )
        restored = RouteGenome.from_dict(genome.to_dict())
        self.assertEqual(restored, genome)
        damaged = genome.to_dict()
        damaged["phase_macro_counts"][0]["counts"]["WHEAT"] += 1
        with self.assertRaisesRegex(ValueError, "genome_id"):
            RouteGenome.from_dict(damaged)


if __name__ == "__main__":
    unittest.main()
