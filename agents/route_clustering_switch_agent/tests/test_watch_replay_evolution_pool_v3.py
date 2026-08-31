from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import watch_replay_evolution_pool_v2 as v2
import watch_replay_evolution_pool_v3 as v3


def _task(episode: int):
    return {
        "episode_id": episode,
        "path": f"episode-{episode}-replay.json",
        "references": [{"dataset": "top40"}],
    }


def _manifest(path: Path, split: str, salt: str) -> None:
    path.write_text(json.dumps({
        "schema_version": 2,
        "top40_manifest_sha256": "old-manifest",
        "split_salt_sha256": hashlib.sha256(salt.encode()).hexdigest(),
        "episodes": [{
            "episode_id": 1,
            "split": split,
            "bucket": 950,
            "episode_types": [v2.PUBLIC],
            "datasets": ["top40"],
            "path": "old.json",
        }],
    }), encoding="utf-8")


def test_sticky_sealed_survives_current_manifest_removal(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    _manifest(path, "sealed", "fixed")
    resolver = v3.StickyResolver(path, "new-manifest", "fixed")
    result = resolver(_task(1), {}, "fixed")
    assert result["split"] == "sealed"
    assert result["assigned_manifest_sha256"] == "old-manifest"
    assert result["assignment_sticky"] is True


def test_sticky_train_to_source_validation_conflict_stops(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    _manifest(path, "train", "fixed")
    resolver = v3.StickyResolver(path, "new-manifest", "fixed")
    with pytest.raises(ValueError, match="changed from train"):
        resolver(_task(1), {1: {v2.VALIDATION}}, "fixed")


def test_split_salt_cannot_change_after_freeze(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    _manifest(path, "dev", "fixed")
    with pytest.raises(ValueError, match="split salt changed"):
        v3.StickyResolver(path, "new-manifest", "different")
