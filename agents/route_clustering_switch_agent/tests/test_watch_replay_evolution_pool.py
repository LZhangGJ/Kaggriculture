from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from scripts.watch_replay_evolution_pool import (
    _discover_replays_fast,
    _load_state,
    _pending_tasks,
    _write_json_atomic,
)


class ReplayWatcherTest(unittest.TestCase):
    def test_fast_discovery_uses_supported_replay_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            nested = root / "123_team" / "456_agent" / "replays"
            nested.mkdir(parents=True)
            (nested / "episode-7-replay.json").write_text("{}", encoding="utf-8")
            ignored = root / "agent_source" / "deep" / "unrelated"
            ignored.mkdir(parents=True)
            (ignored / "episode-8-replay.json").write_text("{}", encoding="utf-8")

            tasks, summary = _discover_replays_fast([("top40", root)])

            self.assertEqual([task["episode_id"] for task in tasks], [7])
            self.assertEqual(summary["source_files"], {"top40": 1})

    def test_pending_retries_only_after_failed_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            replay = root / "episode-7-replay.json"
            replay.write_text("{}", encoding="utf-8")
            old = time.time() - 60
            os.utime(replay, (old, old))
            task = {"episode_id": 7, "path": str(replay), "references": []}
            state = {"schema_version": 1, "files": {}}

            self.assertEqual(
                _pending_tasks([task], set(), (0, 1), state, 30, time.time()),
                [task],
            )
            key = str(replay.resolve())
            state["files"][key]["status"] = "error"
            self.assertEqual(
                _pending_tasks([task], set(), (0, 1), state, 30, time.time()),
                [],
            )

            replay.write_text('{"steps": []}', encoding="utf-8")
            os.utime(replay, (old, old))
            self.assertEqual(
                _pending_tasks([task], set(), (0, 1), state, 30, time.time()),
                [task],
            )
            self.assertEqual(
                _pending_tasks([task], {"7:0", "7:1"}, (0, 1), state, 30, time.time()),
                [],
            )

    def test_state_write_is_atomic_and_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "state.json"
            value = {"schema_version": 1, "files": {"x": {"status": "ingested"}}}
            _write_json_atomic(path, value)
            self.assertEqual(_load_state(path), value)
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
