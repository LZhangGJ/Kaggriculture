from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from scripts.materialize_replay_route_panel import _next_family_ordinal, _records


class ReplayRoutePanelTest(unittest.TestCase):
    def test_family_numbering_continues_without_collisions(self) -> None:
        entries = [
            {"family": "G001"},
            {"family": "NR001"},
            {"family": "NR120"},
            {"family": "NRbad"},
        ]
        self.assertEqual(_next_family_ordinal(entries, "NR"), 121)
        self.assertEqual(_next_family_ordinal(entries, "LIVE"), 1)

    def test_records_ignore_uncommitted_trailing_line(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "pool.jsonl"
            path.write_bytes(b'{"id":1}\n{"id":2')
            self.assertEqual(list(_records(path)), [{"id": 1}])


if __name__ == "__main__":
    unittest.main()
