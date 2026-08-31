from __future__ import annotations

import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_replay_pool_episode_types as audit


def test_audit_counts_manifest_episode_types(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "episode_id,episode_type\n1,EPISODE_TYPE_PUBLIC\n2,EPISODE_TYPE_VALIDATION\n",
        encoding="utf-8",
    )
    pool = tmp_path / "pool.jsonl"
    pool.write_bytes(
        b'{"source":{"episode_id":1}}\n'
        b'{"source":{"episode_id":1}}\n'
        b'{"source":{"episode_id":2}}\n'
        b'{"source":{"episode_id":3}}\n'
    )
    result = audit.audit(manifest, pool)
    assert result["pool_records_scanned"] == 4
    assert result["pool_unique_episodes"] == 3
    assert result["validation_episode_ids_in_pool"] == 1
    assert result["public_episode_ids_in_pool"] == 1
    assert result["pool_episode_types"] == {
        "EPISODE_TYPE_PUBLIC": 1,
        "EPISODE_TYPE_VALIDATION": 1,
        "NOT_IN_CURRENT_MANIFEST": 1,
    }
