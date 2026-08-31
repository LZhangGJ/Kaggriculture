from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "run_block_mvp_continuation_multitail_v1.py"
)
SPEC = importlib.util.spec_from_file_location("block_mvp_continuation_multitail_v1", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _observation() -> dict:
    tiles = [[None for _ in range(10)] for _ in range(10)]
    tiles[0][0] = "LOCKED"
    tiles[0][1] = {"kind": "WEED"}
    tiles[0][2] = {"kind": "PLANT", "crop": "WHEAT"}
    tiles[0][3] = {"kind": "PASTURE", "animal": "COW"}
    tiles[0][4] = {"kind": "SOIL"}
    own = {
        "money": 1000,
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW", "SE"],
        "tiles": tiles,
    }
    return {"step": 216, "player": 0, "farms": [own, dict(own)]}


def test_fixed_commitments_ignore_variable_market_orders() -> None:
    actions = [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(MODULE.HORIZON)
    ]
    actions[216]["market"] = [
        ["BUY_SEED", "WHEAT", 2],
        ["BUY_PRODUCT", "MILK", 999],
        ["SELL", "WOOL", 999],
    ]
    actions[217]["market"] = [
        ["BUY_ANIMAL", "COW", 2], ["BUY_LAND", None, 1], ["HIRE", None, 1],
    ]
    vector = MODULE.fixed_commitments(_observation(), actions)
    assert vector.shape == (18,)
    for offset in (0, 6, 12):
        assert vector[offset:offset + 6].tolist() == [
            2821.0, 2821.0, -1821.0, 1.0, 1.0, 2.0,
        ]


def test_layout_and_all_21_anchor_contract() -> None:
    layout, unlocked = MODULE.own_layout(_observation())
    assert layout.shape == (100,)
    assert layout[:5].tolist() == [
        MODULE.LAYOUT_CODE["LOCKED"], MODULE.LAYOUT_CODE["WEED"],
        MODULE.LAYOUT_CODE["WHEAT"], MODULE.LAYOUT_CODE["COW"],
        MODULE.LAYOUT_CODE["SOIL"],
    ]
    assert unlocked.tolist() == [1, 0, 0, 1]
    assert MODULE.ANCHORS == tuple(range(216, 697, 24))
    assert len(MODULE.ANCHORS) == 21


def test_contract_distance_excludes_market_and_uses_best_provenance() -> None:
    query = np.zeros(MODULE.FEATURE_DIM, np.float32)
    source = np.zeros((3, MODULE.FEATURE_DIM), np.float32)
    source[0, 0] = 50.0              # poor provenance for route A
    source[1, 86:104] = 1_000_000.0  # excluded market-only mismatch
    source[2, 129:] = 2.0            # route B plan mismatch is not diluted
    layouts = np.zeros((3, 100), np.uint8)
    masks = np.zeros((3, 4), np.uint8)
    selected, distances = MODULE.contract_topk(
        query, layouts[0], masks[0], source, layouts, masks,
        ["A", "A", "B"], np.ones(MODULE.FEATURE_DIM), k=2,
    )
    assert selected == ["A", "B"]
    assert distances["A"] == 0.0
    assert distances["B"] == 2.0 / 3.0


def test_pool_byte_segment_hash_top8_and_baseline_label(tmp_path: Path) -> None:
    prefix = {"source": {"source_id": "0:0"}}
    frozen = [
        {"source": {"source_id": "1:0"}},
        {"source": {"source_id": "1:1"}},
    ]
    path = tmp_path / "pool.jsonl"
    first = (json.dumps(prefix) + "\n").encode()
    payload = b"".join((json.dumps(row) + "\n").encode() for row in frozen)
    path.write_bytes(first + payload)
    assert MODULE._pool_segment(path, len(first), len(first) + len(payload)) == frozen
    routes = ["C", "A", "B"]
    one = MODULE.deterministic_hash_topk(routes, "query", 2)
    assert one == MODULE.deterministic_hash_topk(list(reversed(routes)), "query", 2)
    assert len(one) == 2
    assert MODULE._best([], {}, {}, "BASE", 0, -3.0) == ("BASE", 0, -3.0)


def test_frozen_v1_openings_ignore_mutable_input_signatures(tmp_path: Path) -> None:
    tape = [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(MODULE.HORIZON)
    ]
    packed = zlib.compress(json.dumps({"BASE": tape, "RB96_C02": tape}).encode())
    (tmp_path / "block_actions.json.zlib").write_bytes(packed)
    (tmp_path / "blocks.jsonl").write_text(
        json.dumps({"block_id": "BASE", "kind": "baseline"}) + "\n"
        + json.dumps({"block_id": "RB96_C02", "kind": "replay"}) + "\n",
        encoding="utf-8",
    )
    names = (
        "records", "source", "opening_actions", "opening_metadata",
        "base_actions", "base_metadata",
    )
    manifest = {
        "schema": "block-mvp-96-216-candidates-v1",
        "start_step": 96,
        "end_step": 216,
        "actions_sha256": MODULE.hashlib.sha256(packed).hexdigest(),
        "inputs": {name: {"path": str(tmp_path / name)} for name in names},
        "generation_contract": {
            "opening": "X", "prefilter": 1, "carriers": 1, "random_seed": 1,
        },
    }
    (tmp_path / "candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    baseline_id, baseline, c02_id, c02, _, loaded = MODULE.load_frozen_v1_openings(
        tmp_path
    )
    assert (baseline_id, c02_id) == ("BASE", "RB96_C02")
    assert len(baseline) == len(c02) == MODULE.HORIZON
    assert loaded["inputs"]["records"]["path"].endswith("records")


def test_prepared_bank_hash_shape_and_route_set_contract(tmp_path: Path) -> None:
    route_id = "MTV1_x"
    tape = [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(MODULE.HORIZON)
    ]
    actions = zlib.compress(json.dumps({route_id: tape}).encode())
    action_path = tmp_path / "actions.zlib"
    action_path.write_bytes(actions)
    contract_path = tmp_path / "contracts.npz"
    np.savez_compressed(
        contract_path,
        contracts=np.zeros((1, 21, 147), np.float32),
        layouts=np.zeros((1, 21, 100), np.uint8),
        unlocked_masks=np.zeros((1, 21, 4), np.uint8),
        provenance_ids=np.asarray(["1:0"]),
        route_ids=np.asarray([route_id]),
        anchors=np.asarray(MODULE.ANCHORS, np.int16),
    )
    manifest = {
        "schema": "block-mvp-continuation-multitail-v1",
        "provenance_count": 1,
        "actions_file": action_path.name,
        "actions_sha256": MODULE.hashlib.sha256(actions).hexdigest(),
        "contracts_file": contract_path.name,
        "contracts_sha256": MODULE.hashlib.sha256(contract_path.read_bytes()).hexdigest(),
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _, bank, arrays = MODULE.load_prepared_bank(tmp_path)
    assert set(bank) == {route_id}
    assert arrays["contracts"].shape == (1, 21, 147)
    manifest["contracts_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contract archive digest"):
        MODULE.load_prepared_bank(tmp_path)
