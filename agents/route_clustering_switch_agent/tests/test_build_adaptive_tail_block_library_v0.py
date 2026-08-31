from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zlib

import numpy as np
import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "build_adaptive_tail_block_library_v0.py"
)
SPEC = importlib.util.spec_from_file_location("adaptive_tail_block_library_v0", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pass_tape() -> list[dict]:
    return [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(MODULE.HORIZON)
    ]


def _prepared(root: Path, rows: list[tuple]) -> Path:
    root.mkdir()
    actions: dict[str, list[dict]] = {}
    provenance = []
    route_ids = []
    provenance_ids = []
    route_sources: dict[str, list[str]] = {}
    contracts = np.zeros((len(rows), len(MODULE.ANCHORS), MODULE.FEATURE_DIM), np.float32)
    layouts = np.zeros((len(rows), len(MODULE.ANCHORS), MODULE.LAYOUT_DIM), np.uint8)
    masks = np.zeros((len(rows), len(MODULE.ANCHORS), MODULE.MASK_DIM), np.uint8)
    for index, row in enumerate(rows):
        provenance_id, tape, reward, opponent, *optional = row
        team_name = optional[0] if optional else f"team-{provenance_id}"
        tail_sha = MODULE._sha256_bytes(MODULE._canonical_bytes(tape[216:]))
        route_id = f"MTV1_{tail_sha}"
        actions.setdefault(route_id, tape)
        route_sources.setdefault(route_id, []).append(provenance_id)
        provenance_ids.append(provenance_id)
        route_ids.append(route_id)
        provenance.append({
            "provenance_id": provenance_id,
            "route_id": route_id,
            "episode_id": index + 1,
            "player_index": index % 2,
            "team_name": team_name,
            "opponent_team_name": "opponent",
            "final_reward": reward,
            "opponent_reward": opponent,
        })
        contracts[index, :, 0] = index + 1
        contracts[index, :, MODULE.MARKET_START:MODULE.MARKET_STOP] = 1000 * index
        layouts[index, :, 0] = index
        masks[index, :, index % MODULE.MASK_DIM] = 1

    action_bytes = zlib.compress(MODULE._canonical_bytes(actions), level=9)
    action_path = root / "tail_actions.json.zlib"
    action_path.write_bytes(action_bytes)
    contract_path = root / "contracts.npz"
    np.savez_compressed(
        contract_path,
        contracts=contracts,
        layouts=layouts,
        unlocked_masks=masks,
        provenance_ids=np.asarray(provenance_ids),
        route_ids=np.asarray(route_ids),
        anchors=np.asarray(MODULE.ANCHORS, np.int16),
    )
    routes = [{
        "route_id": route_id,
        "tail_sha256_step216": route_id.removeprefix("MTV1_"),
        "representative_source_id": source_ids[0],
        "provenance_ids": source_ids,
    } for route_id, source_ids in sorted(route_sources.items())]
    manifest = {
        "schema": MODULE.INPUT_SCHEMA,
        "status": "prepared",
        "anchors": list(MODULE.ANCHORS),
        "provenance_count": len(rows),
        "unique_tail_count": len(actions),
        "actions_file": action_path.name,
        "actions_sha256": hashlib.sha256(action_bytes).hexdigest(),
        "contracts_file": contract_path.name,
        "contracts_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "routes": routes,
        "provenance": provenance,
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    return root


def _fixture_banks(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    tape_a = _pass_tape()
    tape_a[240] = {"farmer": ["PLANT", "WHEAT"], "hands": [], "market": []}
    tape_b = _pass_tape()
    tape_b[240] = {"farmer": ["PLANT", "CARROT"], "hands": [], "market": []}
    tape_c = _pass_tape()
    tape_c[216] = {"farmer": ["PLANT", "MELON"], "hands": [], "market": []}
    tape_c[696] = {
        "farmer": ["PLANT", "WHEAT"], "hands": [],
        "market": [["SELL", "WHEAT", 3]],
    }
    old = _prepared(tmp_path / "old", [("1:0", tape_a, 10, 5)])
    new = _prepared(tmp_path / "new", [
        ("2:0", tape_b, 20, 5), ("2:1", tape_c, 4, 9),
    ])
    return old, new, tape_c


def test_54d_intent_and_terminal_23_step_contract(tmp_path: Path) -> None:
    _, _, tape = _fixture_banks(tmp_path)
    vector, events = MODULE.action_intent(tape, 696)
    assert vector.shape == (54,) and events.shape == (14,)
    assert vector[0] == 1 and vector[12] == 1
    assert vector[53] == 23
    assert np.isclose(vector[30], 1 / 23)
    assert np.isclose(vector[34], 22 / 23)
    assert np.all(events[:7] == 23)


def test_cluster_distance_excludes_market_environment_only() -> None:
    def block(
        block_id: str, contract: np.ndarray,
        action_vector: np.ndarray | None = None,
    ) -> MODULE.Block:
        return MODULE.Block(
            block_id=block_id, anchor=216, duration=24, action_sha256=block_id,
            actions=[], action_vector=(
                np.zeros(54, np.float32)
                if action_vector is None else action_vector
            ),
            event_profile=np.zeros(14, np.float32), contract=contract,
            layout=np.zeros(100, np.uint8), mask=np.zeros(4, np.uint8),
            source_provenance_id=block_id, source_route_id=block_id,
            sources=[], quality=0,
        )

    left = np.zeros(MODULE.FEATURE_DIM, np.float32)
    market_only = left.copy()
    market_only[MODULE.MARKET_START:MODULE.MARKET_STOP] = 1_000_000
    space = MODULE._feature_space([block("A", left), block("B", market_only)])
    assert MODULE._distance_to(space, 0)[1] == 0
    non_market = left.copy()
    non_market[0] = 10
    space = MODULE._feature_space([block("A", left), block("C", non_market)])
    assert MODULE._distance_to(space, 0)[1] > 0

    market_action = np.zeros(MODULE.ACTION_DIM, np.float32)
    market_action[MODULE.MARKET_ACTION_START:MODULE.MARKET_ACTION_STOP] = 100
    blocks = [block("A", left), block("D", left, market_action)]
    assert MODULE._distance_to(MODULE._feature_space(blocks), 0)[1] > 0
    path_space = MODULE._feature_space(
        blocks, MODULE.PATH_CLUSTER_ACTION_FEATURES,
    )
    assert MODULE._distance_to(path_space, 0)[1] == 0


def test_zero_distance_exact_blocks_keep_all_k_clusters() -> None:
    blocks = []
    for index in range(8):
        tape = _pass_tape()
        tape[216] = {
            "farmer": ["DIG", index, 0], "hands": [], "market": [],
        }
        vector, events = MODULE.action_intent(tape, 216)
        digest = MODULE._sha256_bytes(MODULE._canonical_bytes(tape[216:240]))
        blocks.append(MODULE.Block(
            block_id=f"ATB216_{digest}", anchor=216, duration=24,
            action_sha256=digest, actions=tape[216:240], action_vector=vector,
            event_profile=events, contract=np.zeros(147, np.float32),
            layout=np.zeros(100, np.uint8), mask=np.zeros(4, np.uint8),
            source_provenance_id=str(index), source_route_id=str(index),
            sources=[], quality=float(index),
        ))
    assert len({block.block_id for block in blocks}) == 8
    assert len({block.action_vector.tobytes() for block in blocks}) == 1
    representatives = MODULE._cluster(blocks, 8)
    assert len(representatives) == 8
    assert [row["support"] for row in representatives] == [1] * 8
    assert sorted(block.cluster for block in blocks) == list(range(8))


def test_merge_dedup_cluster_and_artifact_hashes(tmp_path: Path) -> None:
    old, new, _ = _fixture_banks(tmp_path)
    output = tmp_path / "library"
    manifest = MODULE.build_library([new, old], output, clusters_per_anchor=8)
    on_disk = json.loads(
        (output / "block_library_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest == on_disk
    assert manifest["schema"] == MODULE.SCHEMA
    assert manifest["implementation"] == {
        "builder_path": "agents/route_clustering_switch_agent/scripts/"
        "build_adaptive_tail_block_library_v0.py",
        "builder_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }
    assert manifest["anchors"] == list(MODULE.ANCHORS)
    assert manifest["block_count"] == 25
    assert len(manifest["prepared_roots"]) == 2
    assert set(manifest["active_representatives"]) == {
        str(anchor) for anchor in MODULE.ANCHORS
    }
    assert all(
        {"block_id", "source_route_id"} <= set(row)
        for rows in manifest["active_representatives"].values() for row in rows
    )
    assert manifest["per_anchor"]["216"]["occurrences"] == 3
    assert manifest["per_anchor"]["216"]["unique_blocks"] == 2
    assert manifest["per_anchor"]["696"]["duration"] == 23
    assert manifest["feature_contract"]["cluster_group_weights"] == MODULE.GROUP_WEIGHTS
    assert manifest["feature_contract"]["market_environment_excluded_feature_indices"] == list(range(86, 104))

    action_path = output / manifest["actions_file"]
    vector_path = output / manifest["vectors_file"]
    prototype_path = output / manifest["contract_prototypes_file"]
    assert hashlib.sha256(action_path.read_bytes()).hexdigest() == manifest["actions_sha256"]
    assert hashlib.sha256(vector_path.read_bytes()).hexdigest() == manifest["vectors_sha256"]
    assert hashlib.sha256(prototype_path.read_bytes()).hexdigest() == manifest["contract_prototypes_sha256"]
    actions = json.loads(zlib.decompress(action_path.read_bytes()))
    assert len(actions) == 25
    assert {len(value) for key, value in actions.items() if key.startswith("ATB696_")} == {23}
    with np.load(vector_path, allow_pickle=False) as arrays:
        assert arrays["action_vectors"].shape == (25, 54)
        assert arrays["event_profiles"].shape == (25, 14)
        assert arrays["entry_contracts"].shape == (25, 147)
        assert arrays["layouts"].shape == (25, 100)
        assert arrays["unlocked_masks"].shape == (25, 4)
        assert set(map(str, arrays["block_ids"])) == set(actions)
        assert np.isfinite(arrays["action_vectors"]).all()
    assert manifest["contract_prototype_count"] == 3 * len(MODULE.ANCHORS)
    with np.load(prototype_path, allow_pickle=False) as arrays:
        assert arrays["block_ids"].shape == (3 * len(MODULE.ANCHORS),)
        assert arrays["entry_contracts"].shape == (
            3 * len(MODULE.ANCHORS), MODULE.FEATURE_DIM,
        )
    assert not list(output.glob(".*.tmp"))


def test_strict_lineage_filters_before_dedup_and_path_clustering(
    tmp_path: Path,
) -> None:
    allowed_a = _pass_tape()
    allowed_a[216] = {
        "farmer": ["PLANT", "WHEAT"], "hands": [], "market": [],
    }
    allowed_b = _pass_tape()
    allowed_b[216] = {
        "farmer": ["PLANT", "CARROT"], "hands": [], "market": [],
    }
    heldout_unique = _pass_tape()
    heldout_unique[240] = {
        "farmer": ["PLANT", "MELON"], "hands": [], "market": [],
    }
    prepared = _prepared(tmp_path / "prepared", [
        ("allowed-a", allowed_a, 1, 0, "train-team"),
        ("allowed-b", allowed_b, 2, 0, "other-train-team"),
        ("heldout-shared", allowed_a, 1_000_000, 0, "Crop Dusta"),
        ("heldout-unique", heldout_unique, 1_000_000, 0, "tetsuya"),
    ])
    output = tmp_path / "strict-library"
    heldout = ["tetsuya", "Ryo Hasegawa", "Crop Dusta"]
    manifest = MODULE.build_library(
        [prepared], output, clusters_per_anchor=8,
        heldout_team_names=heldout,
    )

    audit = manifest["lineage_filter"]
    assert manifest["deduplication_mode"] == "unit_only"
    assert audit["heldout_team_names"] == sorted(heldout)
    assert audit["input_provenance_count"] == 4
    assert audit["retained_provenance_count"] == 2
    assert audit["excluded_provenance_count"] == 2
    assert audit["member_provenance_audit"] == {
        "checked_count": 2 * len(MODULE.ANCHORS),
        "leaked_count": 0,
        "passed": True,
    }
    assert manifest["block_count"] == 22
    assert manifest["per_anchor"]["216"]["input_occurrences"] == 4
    assert manifest["per_anchor"]["216"]["occurrences"] == 2
    assert manifest["per_anchor"]["216"]["excluded_occurrences"] == 2
    assert manifest["per_anchor"]["216"]["unique_blocks"] == 2
    assert manifest["per_anchor"]["216"]["active_count"] == 2
    assert sum(
        row["support"] for row in manifest["active_representatives"]["216"]
    ) == 2
    assert sum(
        row["unique_member_count"]
        for row in manifest["active_representatives"]["216"]
    ) == 2
    member_ids = {
        row["provenance_id"]
        for block in manifest["blocks"] for row in block["source_lineage"]
    }
    member_teams = {
        row["team_name"]
        for block in manifest["blocks"] for row in block["source_lineage"]
    }
    assert member_ids == {"allowed-a", "allowed-b"}
    assert member_teams.isdisjoint(heldout)
    assert all(
        block["source_provenance_id"] in {"allowed-a", "allowed-b"}
        for block in manifest["blocks"]
    )

    features = manifest["feature_contract"]
    assert features["market_action_quantity_feature_indices"] == list(range(35, 53))
    assert features["market_action_quantities_retained_for_audit"] is True
    assert features["market_action_quantities_included_in_cluster_distance"] is False
    assert features["cluster_action_and_event_excluded_feature_indices"] == list(range(35, 53))
    assert set(features["cluster_action_and_event_feature_indices"]) == (
        set(range(68)) - set(range(35, 53))
    )
    assert features["market_environment_excluded_feature_indices"] == list(range(86, 104))

    prototype_path = output / manifest["contract_prototypes_file"]
    assert hashlib.sha256(prototype_path.read_bytes()).hexdigest() == (
        manifest["contract_prototypes_sha256"]
    )
    assert manifest["contract_prototype_count"] == 2 * len(MODULE.ANCHORS)
    assert manifest["contract_prototypes"][
        "retrieval_entry_contract_feature_indices"
    ] == [*range(86), *range(104, MODULE.FEATURE_DIM)]
    with np.load(prototype_path, allow_pickle=False) as prototypes:
        assert prototypes["entry_contracts"].shape == (
            2 * len(MODULE.ANCHORS), MODULE.FEATURE_DIM,
        )
        assert prototypes["layouts"].shape == (
            2 * len(MODULE.ANCHORS), MODULE.LAYOUT_DIM,
        )
        assert prototypes["unlocked_masks"].shape == (
            2 * len(MODULE.ANCHORS), MODULE.MASK_DIM,
        )
        assert set(map(str, prototypes["provenance_ids"])) == {
            "allowed-a", "allowed-b",
        }
        assert set(map(str, prototypes["team_names"])).isdisjoint(heldout)
        assert np.any(
            prototypes["entry_contracts"][
                :, MODULE.MARKET_START:MODULE.MARKET_STOP
            ] != 0
        )
        prototype_block_ids, counts = np.unique(
            prototypes["block_ids"], return_counts=True,
        )
    expected_support = {
        row["block_id"]: row["occurrence_support"]
        for row in manifest["blocks"]
    }
    assert dict(zip(map(str, prototype_block_ids), map(int, counts))) == (
        expected_support
    )

    actions = json.loads(zlib.decompress(
        (output / manifest["actions_file"]).read_bytes()
    ))
    heldout_only_digest = MODULE._sha256_bytes(
        MODULE._canonical_bytes(heldout_unique[240:264])
    )
    assert f"ATB240_{heldout_only_digest}" not in actions


def test_unit_only_fingerprint_merges_market_variants_and_support(
    tmp_path: Path,
) -> None:
    plain = _pass_tape()
    traded = _pass_tape()
    traded[216] = {
        "farmer": ["PASS"], "hands": [],
        "market": [["BUY_SEED", "WHEAT", 3], ["HIRE"]],
    }
    prepared = _prepared(tmp_path / "prepared-market-variants", [
        ("plain", plain, 1, 0, "train-a"),
        ("traded", traded, 10, 0, "train-b"),
    ])
    full_root = tmp_path / "full-action"
    unit_root = tmp_path / "unit-only"
    full = MODULE.build_library(
        [prepared], full_root, deduplication_mode="full_action",
    )
    unit = MODULE.build_library(
        [prepared], unit_root, deduplication_mode="unit_only",
    )

    assert full["block_count"] == 22
    assert unit["block_count"] == 21
    assert unit["per_anchor"]["216"]["full_action_unique_blocks"] == 2
    assert unit["per_anchor"]["216"]["unique_blocks"] == 1
    assert unit["per_anchor"]["216"]["unit_only_merged_full_action_blocks"] == 1
    block = next(row for row in unit["blocks"] if row["anchor"] == 216)
    assert block["occurrence_support"] == 2
    assert block["full_action_variant_count"] == 2
    assert len(block["source_full_action_sha256s"]) == 2
    assert sum(
        row["occurrence_support"]
        for row in unit["active_representatives"]["216"]
    ) == 2
    assert sum(
        row["support"] for row in unit["active_representatives"]["216"]
    ) == 2

    actions = json.loads(zlib.decompress(
        (unit_root / unit["actions_file"]).read_bytes()
    ))
    assert all(not action["market"] for action in actions[block["block_id"]])
    with np.load(unit_root / unit["vectors_file"], allow_pickle=False) as arrays:
        index = list(map(str, arrays["block_ids"])).index(block["block_id"])
        path_vector = arrays["action_vectors"][index]
        audit_vector = arrays["audit_full_action_vectors"][index]
        path_events = arrays["event_profiles"][index]
        audit_events = arrays["audit_full_event_profiles"][index]
    assert np.all(path_vector[35:53] == 0)
    assert not np.array_equal(path_vector, audit_vector)
    assert not np.array_equal(path_events, audit_events)


def test_prepared_validation_rejects_hash_and_shape_corruption(tmp_path: Path) -> None:
    old, _, _ = _fixture_banks(tmp_path)
    manifest_path = old / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["actions_sha256"] = "bad"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="action archive digest"):
        MODULE.load_prepared(old)

    tape = _pass_tape()
    malformed = _prepared(tmp_path / "malformed", [("3:0", tape, 1, 0)])
    contract_path = malformed / "contracts.npz"
    np.savez_compressed(
        contract_path,
        contracts=np.zeros((1, 20, 147), np.float32),
        layouts=np.zeros((1, 21, 100), np.uint8),
        unlocked_masks=np.zeros((1, 21, 4), np.uint8),
        provenance_ids=np.asarray(["3:0"]),
        route_ids=np.asarray([next(iter(json.loads(zlib.decompress(
            (malformed / "tail_actions.json.zlib").read_bytes()
        ))))]),
        anchors=np.asarray(MODULE.ANCHORS, np.int16),
    )
    manifest_path = malformed / "candidate_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["contracts_sha256"] = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid fixed shape"):
        MODULE.load_prepared(malformed)
