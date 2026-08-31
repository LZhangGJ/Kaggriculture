from __future__ import annotations

import hashlib
import json
import sys
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import select_replay_execution_panel_v1 as panel


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _replay(
    path: Path, farmer: list[str], market: list[list[object]] | None = None,
) -> None:
    steps = []
    for recorded_step in range(720):
        action = {"farmer": ["PASS"], "hands": [], "market": []}
        if recorded_step == 217:
            action = {
                "farmer": list(farmer), "hands": [],
                "market": [list(value) for value in market or ()],
            }
        steps.append([
            {"action": action},
            {"action": {"farmer": ["PASS"], "hands": [], "market": []}},
        ])
    path.write_text(json.dumps({"steps": steps}), encoding="utf-8")


def _record(
    episode: int, player: int, replay: Path, genome: str, team: str,
    opponent_team: str, reward: float, opponent_reward: float,
    *, datasets: tuple[str, ...] = ("top40",), market_price: int = 1,
) -> dict[str, object]:
    result = "win" if reward > opponent_reward else (
        "tie" if reward == opponent_reward else "loss"
    )
    return {
        "schema_version": 2,
        "genome_id": genome,
        "anchor_targets": [],
        "phase_macro_counts": [],
        "structural_events": [],
        "market_profile": [{
            "operation": "BUY_SEED", "item": "WHEAT",
            "orders": 1, "quantity": 1,
            "observed_price": {"median": market_price},
        }],
        "cash_profile": {"minimum": 10},
        "source": {
            "source_id": f"{episode}:{player}", "episode_id": episode,
            "player_index": player, "team_name": team,
            "opponent_team_name": opponent_team, "datasets": list(datasets),
            "ingestion_split": "train", "replay_steps": 720,
            "result": result, "final_reward": reward,
            "opponent_reward": opponent_reward,
            "replay_sources": [{
                "dataset": datasets[0], "absolute_path": str(replay.resolve()),
            }],
        },
    }


def _write_pool(path: Path, rows: list[dict[str, object]]) -> str:
    payload = b"".join(_canonical(row) + b"\n" for row in rows)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, str, str]:
    specs = [
        (1, "gA", "teamA", 100, ["PASS"], None, ("top40",), 1),
        (2, "gA", "teamB", 90, ["EAST"], None, ("top40",), 2),
        (3, "gA", "teamC", 80, ["WEST"], None, ("top40",), 3),
        (4, "gB", "teamA", 95, ["NORTH"], None, ("top40",), 4),
        (5, "gC", "teamD", 70, ["SOUTH"], None, ("top40", "our_latest"), 5),
        (6, "gD", "teamE", 60, ["PLANT", "WHEAT"], None,
         ("top40", "our_latest"), 6),
        (7, "gE", "teamF", -1, ["HARVEST"], None, ("top40",), 7),
        # Same macro genome and unit path as episode 1, distinct market overlay.
        (8, "gA", "teamG", 50, ["PASS"], [["SELL", "WHEAT", 1]],
         ("top40",), 999999),
    ]
    records: list[dict[str, object]] = []
    split_rows = []
    for episode, genome, team, reward, farmer, market, datasets, price in specs:
        replay = tmp_path / f"replay-{episode}.json"
        _replay(replay, farmer, market)
        records.extend([
            _record(
                episode, 0, replay, genome, team, "opponent",
                reward, 0, datasets=datasets, market_price=price,
            ),
            _record(
                episode, 1, replay, "opponent_shared", "opponent", team,
                0, reward, datasets=datasets, market_price=price,
            ),
        ])
        split_rows.append({
            "episode_id": episode, "split": "train",
            "datasets": list(datasets), "episode_types": [panel.PUBLIC],
        })
    pool = tmp_path / "pool.jsonl"
    pool_sha = _write_pool(pool, records)
    split = tmp_path / "episode_splits.json"
    split.write_text(json.dumps({
        "schema_version": 2,
        "top40_manifest_sha256": "f" * 64,
        "episodes": split_rows,
    }), encoding="utf-8")
    split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    return pool, split, pool_sha, split_sha


def _build(
    pool: Path, split: Path, pool_sha: str, split_sha: str, output: Path,
    **kwargs: object,
) -> dict[str, object]:
    return panel.build_panel(
        pool=pool, split_manifest=split, output_root=output,
        pool_start_byte=0, pool_end_byte=pool.stat().st_size,
        pool_segment_sha256=pool_sha, split_manifest_sha256=split_sha,
        **kwargs,
    )

def test_action_identity_caps_sidecar_progress_and_dual_outputs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    pool, split, pool_sha, split_sha = _fixture(tmp_path)
    output = tmp_path / "panel"
    manifest = _build(
        pool, split, pool_sha, split_sha, output,
        count=4, macro_cap=2, team_cap=1,
        focus_count=2, team_seed_count=2, progress_every=2,
    )

    progress = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
        if line.strip()
    ]
    assert [row["files"] for row in progress] == [2, 4, 6, 8]
    assert all(row["status"] == "replay_execution_scan" for row in progress)
    assert all(row["total_files"] == 8 for row in progress)
    assert all(row["gib"] > 0 and row["elapsed_seconds"] > 0 for row in progress)
    assert all(row["mib_per_second"] > 0 for row in progress)

    assert manifest["catalog"]["records"] == 16
    assert manifest["catalog"]["eligible_execution_identities"] == 8
    assert manifest["catalog"]["rejection_counts"] == {}
    assert manifest["selection"]["focus_selected"] == 2
    assert manifest["selection"]["focus_shortfall"] == 0
    assert manifest["selection"]["raw_feature_dimensions"] == 567 + 512
    assert manifest["selection"]["actor_time_hash"]["dimensions"] == 512
    routes = manifest["routes"]
    assert len(routes) == 4
    assert max(Counter(
        row["execution_identity"]["genome_id"] for row in routes
    ).values()) <= 2
    assert max(Counter(
        row["representative_team_name"] for row in routes
    ).values()) <= 1

    grouped = next(
        row for row in routes
        if row["execution_identity"]["genome_id"] == "gA"
        and row["support"] == 2
    )
    assert len(grouped["full_action_variants"]) == 2
    path_base = json.loads(zlib.decompress(
        (output / "path_base_actions.json.zlib").read_bytes()
    ))
    overlays = json.loads(zlib.decompress(
        (output / "market_overlays.json.zlib").read_bytes()
    ))
    base = path_base[grouped["execution_id"]]
    for variant in grouped["full_action_variants"]:
        market = overlays[variant["market_overlay_ref"]["key"]]
        reconstructed = [{
            "farmer": action["farmer"], "hands": action["hands"],
            "market": market[step],
        } for step, action in enumerate(base)]
        assert hashlib.sha256(_canonical(reconstructed)).hexdigest() == (
            variant["full_tape_sha256"]
        )

    feature_path = output / "execution_action_features.npz"
    with np.load(feature_path, allow_pickle=False) as sidecar:
        assert set(sidecar.files) == {"execution_ids", "action_features", "anchors"}
        ids = sidecar["execution_ids"].tolist()
        features = sidecar["action_features"]
        assert ids == sorted(ids)
        assert len(ids) == 8
        assert features.shape == (8, 1079)
        assert features.dtype == np.float32
        assert sidecar["anchors"].tolist() == list(panel.ANCHORS)
        assert not any("replay" in key or "tape" in key for key in sidecar.files)
    feature_artifact = manifest["artifacts"]["execution_action_features"]
    assert hashlib.sha256(feature_path.read_bytes()).hexdigest() == (
        feature_artifact["sha256"]
    )
    assert manifest["representation"]["eligible_feature_sidecar"] == {
        "file": "execution_action_features.npz",
        "rows": 8,
        "execution_id_order": "ascending",
        "contains_replay_or_tape": False,
        "arrays": {
            "execution_ids": "unicode",
            "action_features": "float32",
            "anchors": "int16",
        },
    }

    catalog = [
        json.loads(line)
        for line in (output / "record_catalog.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert all(len(row["block_hashes"]) == len(panel.ANCHORS) for row in catalog)
    assert manifest["selection"]["observed_market_prices_used"] is False
    manifest_bytes = (output / "panel_manifest.json").read_bytes()
    claimed = (output / "panel_manifest.json.sha256").read_text().split()[0]
    assert hashlib.sha256(manifest_bytes).hexdigest() == claimed

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(pool, split, pool_sha, split_sha, output, count=4)


def test_frozen_input_hashes_and_public_train_gate_fail_closed(
    tmp_path: Path,
) -> None:
    pool, split, pool_sha, split_sha = _fixture(tmp_path)
    with pytest.raises(ValueError, match="pool segment SHA-256 mismatch"):
        _build(
            pool, split, "0" * 64, split_sha, tmp_path / "bad-pool", count=1,
        )
    assert not (tmp_path / "bad-pool").exists()

    with pytest.raises(ValueError, match="split manifest SHA-256 mismatch"):
        _build(
            pool, split, pool_sha, "0" * 64, tmp_path / "bad-split", count=1,
        )
    assert not (tmp_path / "bad-split").exists()

    payload = json.loads(split.read_text(encoding="utf-8"))
    payload["episodes"][0]["episode_types"] = [panel.VALIDATION]
    split.write_text(json.dumps(payload), encoding="utf-8")
    changed_split_sha = hashlib.sha256(split.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="validation episode"):
        _build(
            pool, split, pool_sha, changed_split_sha, tmp_path / "leak", count=1,
        )
    assert not (tmp_path / "leak").exists()


def test_episode_pairing_missing_load_and_incomplete_are_fail_closed(
    tmp_path: Path,
) -> None:
    pool, split, _, split_sha = _fixture(tmp_path)
    rows = [
        json.loads(line)
        for line in pool.read_text(encoding="utf-8").splitlines()
    ]

    missing_pair = tmp_path / "missing-pair.jsonl"
    missing_pair_sha = _write_pool(missing_pair, rows[:-1])
    with pytest.raises(ValueError, match="exactly player0/player1"):
        _build(
            missing_pair, split, missing_pair_sha, split_sha,
            tmp_path / "missing-pair-output", count=1,
        )

    alternate = tmp_path / "alternate.json"
    alternate.write_bytes((tmp_path / "replay-1.json").read_bytes())
    different_rows = json.loads(json.dumps(rows))
    different_rows[1]["source"]["replay_sources"][0]["absolute_path"] = str(
        alternate.resolve()
    )
    different_pool = tmp_path / "different-replay.jsonl"
    different_sha = _write_pool(different_pool, different_rows)
    with pytest.raises(ValueError, match="same readable replay"):
        _build(
            different_pool, split, different_sha, split_sha,
            tmp_path / "different-output", count=1,
        )

    replay_one = tmp_path / "replay-1.json"
    replay_one.write_text("{malformed", encoding="utf-8")
    malformed_pool = tmp_path / "malformed-replay.jsonl"
    malformed_sha = _write_pool(malformed_pool, rows)
    with pytest.raises(ValueError, match="failed to load replay"):
        _build(
            malformed_pool, split, malformed_sha, split_sha,
            tmp_path / "malformed-output", count=1,
        )

    _replay(replay_one, ["PASS"])
    replay_one.unlink()
    missing_pool = tmp_path / "missing-replay.jsonl"
    missing_sha = _write_pool(missing_pool, rows)
    with pytest.raises(FileNotFoundError, match="no readable replay"):
        _build(
            missing_pool, split, missing_sha, split_sha,
            tmp_path / "missing-output", count=1,
        )

    _replay(replay_one, ["PASS"])
    replay_payload = json.loads(replay_one.read_text(encoding="utf-8"))
    replay_payload["steps"] = replay_payload["steps"][:10]
    replay_one.write_text(json.dumps(replay_payload), encoding="utf-8")
    incomplete_pool = tmp_path / "incomplete.jsonl"
    incomplete_sha = _write_pool(incomplete_pool, rows)
    with pytest.raises(ValueError, match="invalid replay tape"):
        _build(
            incomplete_pool, split, incomplete_sha, split_sha,
            tmp_path / "incomplete-output", count=1,
        )


def _blank_tape(hand_count: int = 2) -> list[dict[str, object]]:
    return [{
        "farmer": ["PASS"],
        "hands": [["PASS"] for _ in range(hand_count)],
        "market": [],
    } for _ in range(panel.HORIZON)]


def test_actor_time_hash_distinguishes_slot_timing_and_parameters() -> None:
    slot_a = _blank_tape()
    slot_b = _blank_tape()
    slot_a[216]["hands"] = [["EAST"], ["WEST"]]
    slot_b[216]["hands"] = [["WEST"], ["EAST"]]
    assert np.array_equal(
        panel.pooled_action_feature(slot_a), panel.pooled_action_feature(slot_b),
    )
    assert not np.array_equal(panel.action_feature(slot_a), panel.action_feature(slot_b))

    time_a = _blank_tape()
    time_b = _blank_tape()
    time_a[216]["hands"][0] = ["EAST"]
    time_a[217]["hands"][0] = ["WEST"]
    time_b[216]["hands"][0] = ["WEST"]
    time_b[217]["hands"][0] = ["EAST"]
    assert np.array_equal(
        panel.pooled_action_feature(time_a), panel.pooled_action_feature(time_b),
    )
    assert not np.array_equal(panel.action_feature(time_a), panel.action_feature(time_b))

    parameter_a = _blank_tape()
    parameter_b = _blank_tape()
    parameter_a[216]["hands"][0] = ["PLANT", "WHEAT"]
    parameter_b[216]["hands"][0] = ["PLANT", "CORN"]
    assert np.array_equal(
        panel.pooled_action_feature(parameter_a),
        panel.pooled_action_feature(parameter_b),
    )
    assert not np.array_equal(
        panel.action_feature(parameter_a), panel.action_feature(parameter_b),
    )







