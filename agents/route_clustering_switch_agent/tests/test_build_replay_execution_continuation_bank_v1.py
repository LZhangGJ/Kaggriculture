from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_adaptive_tail_block_library_v0 as adaptive
import build_replay_execution_continuation_bank_v1 as bank
import select_replay_execution_panel_v1 as panel


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _farm(money: int) -> dict[str, object]:
    return {
        "money": money,
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW"],
        "tiles": [[None for _ in range(10)] for _ in range(10)],
    }


def _replay(path: Path, seed_quantity: int) -> None:
    steps = []
    for recorded_step in range(720):
        actions = [
            {"farmer": ["PASS"], "hands": [], "market": []},
            {"farmer": ["PASS"], "hands": [], "market": []},
        ]
        if recorded_step == 217:
            actions[0]["market"] = [["BUY_SEED", "WHEAT", seed_quantity]]
            actions[1]["farmer"] = ["EAST"]
        farms = [_farm(1000 + recorded_step), _farm(500 + recorded_step)]
        steps.append([{
            "action": actions[player],
            "observation": {
                "step": recorded_step,
                "player": player,
                "farms": farms,
                "private": {"shed": {}, "seeds": {}, "inventories": []},
                "market": {"inventory": {}, "prices": {"WHEAT": 999999}},
                "town": {"unlocked_shops": []},
            },
        } for player in (0, 1)])
    path.write_text(json.dumps({
        "steps": steps, "rewards": [10, 0], "info": {"seed": seed_quantity},
    }), encoding="utf-8")


def _record(
    episode: int, player: int, replay: Path, seed_quantity: int,
) -> dict[str, object]:
    own = player == 0
    datasets = ["our_latest"] if own else ["top40"]
    return {
        "schema_version": 2,
        "genome_id": "same_path" if own else "opponent_path",
        "anchor_targets": [],
        "phase_macro_counts": [],
        "structural_events": [],
        "market_profile": [{
            "operation": "BUY_SEED", "item": "WHEAT", "orders": 1,
            "quantity": seed_quantity, "observed_price": {"median": 999999},
        }],
        "cash_profile": {"minimum": 500},
        "source": {
            "source_id": f"{episode}:{player}",
            "episode_id": episode,
            "player_index": player,
            "team_name": f"team-{player}",
            "opponent_team_name": f"team-{1-player}",
            "datasets": datasets,
            "ingestion_split": "train",
            "replay_steps": 720,
            "result": "win" if own else "loss",
            "final_reward": 10 if own else 0,
            "opponent_reward": 0 if own else 10,
            "replay_sources": [{
                "dataset": datasets[0], "absolute_path": str(replay.resolve()),
            }],
        },
    }


def _panel(tmp_path: Path, count: int = 1) -> tuple[Path, str, list[Path]]:
    rows = []
    split_rows = []
    replays = []
    for episode, quantity in ((1, 1), (2, 2)):
        replay = tmp_path / f"replay-{episode}.json"
        _replay(replay, quantity)
        replays.append(replay)
        rows.extend(_record(episode, player, replay, quantity) for player in (0, 1))
        split_rows.append({
            "episode_id": episode,
            "split": "train",
            "datasets": ["our_latest", "top40"],
            "episode_types": [panel.PUBLIC],
        })
    pool = tmp_path / "pool.jsonl"
    pool_payload = b"".join(_canonical(row) + b"\n" for row in rows)
    pool.write_bytes(pool_payload)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({
        "top40_manifest_sha256": "f" * 64, "episodes": split_rows,
    }), encoding="utf-8")
    panel_root = tmp_path / "panel"
    panel.build_panel(
        pool=pool,
        split_manifest=split,
        output_root=panel_root,
        pool_start_byte=0,
        pool_end_byte=len(pool_payload),
        pool_segment_sha256=hashlib.sha256(pool_payload).hexdigest(),
        split_manifest_sha256=hashlib.sha256(split.read_bytes()).hexdigest(),
        count=count,
        macro_cap=2,
        team_cap=4,
        focus_dataset="our_latest",
        focus_count=1,
        team_seed_count=0,
        progress_every=1,
    )
    manifest_sha = hashlib.sha256(
        (panel_root / "panel_manifest.json").read_bytes(),
    ).hexdigest()
    return panel_root, manifest_sha, replays


def test_dual_overlay_contract_bank_and_replay_tamper_gate(tmp_path: Path) -> None:
    panel_root, panel_sha, replays = _panel(tmp_path)
    output = tmp_path / "continuation-bank"
    manifest = bank.build_bank(
        panel_root=panel_root,
        panel_manifest_sha256=panel_sha,
        output_root=output,
    )

    assert manifest["selected_execution_count"] == 1
    assert manifest["full_action_variant_count"] == 2
    assert manifest["provenance_count"] == 2
    assert manifest["identity"]["observed_market_price_in_path_identity"] is False
    prepared = adaptive.load_prepared(output)
    assert prepared.contracts.shape == (2, 21, 147)
    assert sorted(prepared.contracts[:, 0, 129].tolist()) == [10.0, 20.0]
    assert len({row["execution_id"] for row in manifest["routes"]}) == 1
    assert len({row["full_tape_sha256"] for row in manifest["routes"]}) == 2

    library = adaptive.build_library(
        [output], tmp_path / "library", clusters_per_anchor=1,
        deduplication_mode="unit_only",
    )
    assert library["per_anchor"]["216"]["unique_blocks"] == 1
    assert library["per_anchor"]["216"]["full_action_unique_blocks"] == 2

    replays[0].write_text("{}", encoding="utf-8")
    failed_output = tmp_path / "tampered-output"
    with pytest.raises(ValueError, match="replay SHA-256 mismatch"):
        bank.build_bank(
            panel_root=panel_root,
            panel_manifest_sha256=panel_sha,
            output_root=failed_output,
        )
    assert not failed_output.exists()


def test_grouped_loader_is_live_one_once_and_path_order_invariant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Two executions put both players from each physical replay into different
    # logical variant tasks, so one load must serve multiple task ordinals.
    panel_root, panel_sha, replays = _panel(tmp_path, count=2)
    original_load = bank._load_replay
    loads: dict[Path, int] = {}

    class LiveReplay(dict):
        live = 0
        peak = 0

        def __init__(self, value: object) -> None:
            super().__init__(value)
            type(self).live += 1
            type(self).peak = max(type(self).peak, type(self).live)

        def __del__(self) -> None:
            type(self).live -= 1

    def tracked_load(path: Path) -> tuple[str, dict, int]:
        path = path.resolve()
        loads[path] = loads.get(path, 0) + 1
        digest, replay, size = original_load(path)
        return digest, LiveReplay(replay), size

    monkeypatch.setattr(bank, "_load_replay", tracked_load)
    first_root = tmp_path / "grouped-first"
    first = bank.build_bank(
        panel_root=panel_root,
        panel_manifest_sha256=panel_sha,
        output_root=first_root,
    )
    assert loads == {path.resolve(): 1 for path in replays}
    assert LiveReplay.peak == 1
    assert LiveReplay.live == 0
    expected_provenance = [
        provenance_id
        for route in first["routes"]
        for provenance_id in route["provenance_ids"]
    ]
    expected_route_ids = [
        route["route_id"]
        for route in first["routes"]
        for _ in route["provenance_ids"]
    ]
    assert [row["provenance_id"] for row in first["provenance"]] == (
        expected_provenance
    )
    with np.load(first_root / "contracts.npz", allow_pickle=False) as archive:
        assert archive["provenance_ids"].tolist() == expected_provenance
        assert archive["route_ids"].tolist() == expected_route_ids
    assert first["replay_loading"] == {
        "grouping_key": "resolved selected_replay_path",
        "one_load_per_unique_path": True,
        "peak_live_parsed_replays": 1,
        "output_order": (
            "preallocated ordinal slots in existing variant(rank,full_tape_sha256) "
            "then sorted-source_id order; replay path I/O order cannot change artifacts"
        ),
        "progress_every_unique_replays": 32,
    }

    default_order = bank._ordered_replay_paths
    monkeypatch.setattr(
        bank, "_ordered_replay_paths",
        lambda groups: list(reversed(default_order(groups))),
    )
    loads.clear()
    second_root = tmp_path / "grouped-reversed"
    second = bank.build_bank(
        panel_root=panel_root,
        panel_manifest_sha256=panel_sha,
        output_root=second_root,
    )
    assert loads == {path.resolve(): 1 for path in replays}
    assert LiveReplay.peak == 1
    assert LiveReplay.live == 0
    assert second["actions_sha256"] == first["actions_sha256"]
    assert second["contracts_sha256"] == first["contracts_sha256"]
    assert second["routes"] == first["routes"]
    assert second["provenance"] == first["provenance"]
    assert second["manifest_sha256"] == first["manifest_sha256"]
    assert (second_root / "contracts.npz").read_bytes() == (
        first_root / "contracts.npz"
    ).read_bytes()

    failure_calls = 0

    def fail_on_second_path(path: Path) -> tuple[str, dict, int]:
        nonlocal failure_calls
        failure_calls += 1
        if failure_calls == 2:
            raise RuntimeError("synthetic grouped loader failure")
        return original_load(path)

    monkeypatch.setattr(bank, "_load_replay", fail_on_second_path)
    failed_root = tmp_path / "grouped-failed"
    with pytest.raises(RuntimeError, match="synthetic grouped loader failure"):
        bank.build_bank(
            panel_root=panel_root,
            panel_manifest_sha256=panel_sha,
            output_root=failed_root,
        )
    assert not failed_root.exists()


def test_replay_progress_reports_required_throughput_fields(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(bank.time, "perf_counter", lambda: 10.0)
    bank._report_replay_progress(32, 2 * 1024 * 1024, 8.0)
    row = json.loads(capsys.readouterr().err)
    assert row == {
        "progress": bank.SCHEMA,
        "files": 32,
        "bytes": 2 * 1024 * 1024,
        "elapsed": 2.0,
        "MiB/s": 1.0,
    }
