from __future__ import annotations

import unittest

from materialize_shared_prefix_routes import shared_prefix_route


def _tape(label: str) -> list[dict[str, object]]:
    return [
        {
            "farmer": [label, step],
            "hands": [[label, step]],
            "market": [["HIRE"], ["SELL", "WHEAT", step + 1]],
        }
        for step in range(4)
    ]


class SharedPrefixRouteTest(unittest.TestCase):
    def test_full_branch_keeps_exact_prefix_and_replaces_suffix(self) -> None:
        base, donor = _tape("base"), _tape("donor")
        result = shared_prefix_route(base, donor, 2, "full")
        self.assertEqual(result[:2], base[:2])
        self.assertEqual(result[2:], donor[2:])
        self.assertIsNot(result, base)
        self.assertIsNot(result[2], donor[2])

    def test_market_branch_keeps_workers_and_structural_orders(self) -> None:
        base, donor = _tape("base"), _tape("donor")
        result = shared_prefix_route(base, donor, 2, "market")
        self.assertEqual(result[:2], base[:2])
        self.assertEqual(result[2]["farmer"], base[2]["farmer"])
        self.assertEqual(result[2]["hands"], base[2]["hands"])
        self.assertEqual(result[2]["market"], [["HIRE"], ["SELL", "WHEAT", 3]])

    def test_checkpoint_must_be_inside_tape(self) -> None:
        with self.assertRaises(ValueError):
            shared_prefix_route(_tape("base"), _tape("donor"), 0, "full")


if __name__ == "__main__":
    unittest.main()
