from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_replay_evolution_pool_v2 as watcher


def _task(episode: int, *datasets: str):
    return {
        "episode_id": episode,
        "path": f"episode-{episode}-replay.json",
        "references": [{"dataset": value} for value in datasets],
    }


def test_split_is_deterministic_and_episode_level() -> None:
    first = [watcher.split_episode(value, "fixed") for value in range(100)]
    second = [watcher.split_episode(value, "fixed") for value in range(100)]
    assert first == second
    assert {name for name, _ in first} <= {"train", "dev", "sealed"}
    assert all(0 <= bucket < 1000 for _, bucket in first)


def test_top40_requires_public_manifest_and_excludes_validation() -> None:
    manifest = {
        1: {watcher.PUBLIC},
        2: {watcher.VALIDATION},
    }
    public = watcher.classify_task(_task(1, "top40"), manifest, "fixed")
    validation = watcher.classify_task(_task(2, "top40"), manifest, "fixed")
    missing = watcher.classify_task(_task(3, "top40"), manifest, "fixed")
    own = watcher.classify_task(_task(4, "our_latest"), manifest, "fixed")
    assert public["split"] in {"train", "dev", "sealed"}
    assert validation["split"] == "excluded_source_validation"
    assert missing["split"] == "excluded_not_public_in_snapshot"
    assert own["split"] in {"train", "dev", "sealed"}
