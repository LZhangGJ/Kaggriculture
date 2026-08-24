from __future__ import annotations

import io
import json
from pathlib import Path
import tarfile

from unseen_generalization_v1.inventory import build_asset_manifest


def _minimal_replay():
    obs = {
        "player": 0,
        "step": 0,
        "day": 0,
        "hour": 0,
        "farms": [
            {"money": 3000, "tiles": [[None]], "farmer": [0, 0], "hands": [], "unlocked_quadrants": ["NW"]},
            {"money": 3000, "tiles": [[None]], "farmer": [0, 0], "hands": [], "unlocked_quadrants": ["NW"]},
        ],
        "private": {"shed": {}, "seeds": {}, "inventories": [[]]},
        "market": {"inventory": {}, "prices": {}},
        "town": {"unlocked_shops": []},
    }
    return {
        "steps": [[
            {"observation": obs, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "reward": 1, "status": "DONE"},
            {"observation": {"player": 1}, "action": {"farmer": ["PASS"], "hands": [], "market": []}, "reward": 0, "status": "DONE"},
        ]]
    }


def test_inventory_detects_replay_archive_duplicates_and_excludes_secrets(tmp_path: Path):
    replay_text = json.dumps(_minimal_replay())
    (tmp_path / "episode-1-replay.json").write_text(replay_text, encoding="utf-8")
    (tmp_path / "copy.json").write_text(replay_text, encoding="utf-8")
    (tmp_path / "access_token").write_text("dummy-secret-not-a-real-token", encoding="utf-8")

    archive_path = tmp_path / "submission.tar.gz"
    main_py = b"def agent(obs):\n    return {'farmer':['PASS'],'hands':[],'market':[]}\n"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("main.py")
        info.size = len(main_py)
        archive.addfile(info, io.BytesIO(main_py))

    manifest = build_asset_manifest({"fixture": tmp_path})
    assert manifest["kind_counts"]["replay"] == 2
    assert manifest["kind_counts"]["submission_archive"] == 1
    archive = next(row for row in manifest["records"] if row["kind"] == "submission_archive")
    assert archive["metadata"]["has_root_main_py"] is True
    assert all(row["relative_path"] != "access_token" for row in manifest["records"])
    assert len(manifest["duplicate_content_groups"]) == 1


def test_inventory_can_exclude_generated_output_subtree(tmp_path: Path):
    (tmp_path / "agent.py").write_text("def agent(obs): return {}\n", encoding="utf-8")
    output = tmp_path / "live_assets" / "ug0_corpus_v1"
    output.mkdir(parents=True)
    (output / "old_report.md").write_text("old", encoding="utf-8")
    manifest = build_asset_manifest({"workspace": tmp_path}, exclude_paths=(output,))
    paths = {row["relative_path"] for row in manifest["records"]}
    assert "agent.py" in paths
    assert "live_assets/ug0_corpus_v1/old_report.md" not in paths
