from __future__ import annotations

import hashlib
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_adaptive_tail_block_library_v0 as adaptive
import build_offline_block_portfolio_v1 as portfolio
import run_block_mvp_continuation_multitail_v1 as continuation


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _prepared(tmp_path: Path, paths: int, quality: bool) -> Path:
    root = tmp_path / ("prepared-quality" if quality else "prepared-fallback")
    root.mkdir()
    route_actions = {}
    routes = []
    provenance = []
    contracts = []
    layouts = []
    masks = []
    route_ids = []
    provenance_ids = []
    for index in range(paths + 1):
        path_index = min(index, paths - 1)
        route_id = f"R{index:03d}"
        tape = [{
            "farmer": ["PASS"], "hands": [], "market": [],
        } for _ in range(adaptive.HORIZON)]
        for anchor in adaptive.ANCHORS:
            tape[anchor] = {
                "farmer": ["PLANT", "WHEAT", path_index, anchor],
                "hands": [],
                "market": [],
            }
        # Last route shares path 0 but supplies a second market overlay.
        if index == paths:
            tape[216]["market"] = [["BUY_SEED", "WHEAT", 2]]
        route_actions[route_id] = tape
        full_tape_sha = hashlib.sha256(_canonical(tape)).hexdigest()
        provenance_id = f"P{index:03d}"
        route_ids.append(route_id)
        provenance_ids.append(provenance_id)
        row = {
            "provenance_id": provenance_id,
            "route_id": route_id,
            "team_name": f"team-{path_index % 8}",
            "ingestion_split": "train",
            "datasets": ["top40"],
            "execution_id": f"E{path_index:03d}",
            "full_tape_sha256": full_tape_sha,
        }
        if quality:
            reward = 1000.0 if path_index == 0 else float(path_index)
            opponent = 0.0 if path_index == 0 else float(paths - path_index)
            row.update({
                "result": "win" if reward > opponent else (
                    "tie" if reward == opponent else "loss"
                ),
                "final_reward": reward,
                "opponent_reward": opponent,
            })
        provenance.append(row)
        contract = np.zeros((len(adaptive.ANCHORS), adaptive.FEATURE_DIM), np.float32)
        contract[:, 0] = 1000 + path_index
        requirement = path_index * 10 + int(index == paths) * 5
        for offset in (0, 6, 12):
            contract[:, continuation.PLAN_START + offset] = requirement
            contract[:, continuation.PLAN_START + offset + 1] = requirement
            contract[:, continuation.PLAN_START + offset + 2] = 1000 + path_index - requirement
        contracts.append(contract)
        layout = np.zeros((len(adaptive.ANCHORS), adaptive.LAYOUT_DIM), np.uint8)
        layout[:, 0] = path_index % 14
        layouts.append(layout)
        mask = np.zeros((len(adaptive.ANCHORS), adaptive.MASK_DIM), np.uint8)
        mask[:, path_index % 4] = 1
        masks.append(mask)
        routes.append({
            "route_id": route_id,
            "execution_id": f"E{path_index:03d}",
            "full_tape_sha256": full_tape_sha,
            "tail_sha256_step216": hashlib.sha256(
                _canonical(tape[216:]),
            ).hexdigest(),
            "market_overlay_ref": {
                "file": "market_overlays.json.zlib", "key": full_tape_sha,
            },
            "representative_source_id": provenance_id,
            "provenance_ids": [provenance_id],
        })

    action_payload = zlib.compress(_canonical(route_actions), level=9)
    (root / "tail_actions.json.zlib").write_bytes(action_payload)
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        contracts=np.stack(contracts),
        layouts=np.stack(layouts),
        unlocked_masks=np.stack(masks),
        provenance_ids=np.asarray(provenance_ids),
        route_ids=np.asarray(route_ids),
        anchors=np.asarray(adaptive.ANCHORS, np.int16),
    )
    contract_payload = buffer.getvalue()
    (root / "contracts.npz").write_bytes(contract_payload)
    manifest = {
        "schema": adaptive.INPUT_SCHEMA,
        "status": "prepared",
        "anchors": list(adaptive.ANCHORS),
        "provenance_count": len(provenance),
        "unique_tail_count": len(route_actions),
        "actions_file": "tail_actions.json.zlib",
        "actions_sha256": hashlib.sha256(action_payload).hexdigest(),
        "contracts_file": "contracts.npz",
        "contracts_sha256": hashlib.sha256(contract_payload).hexdigest(),
        "routes": routes,
        "provenance": provenance,
        "data_gate": {
            "train_only": True,
            "top40_public_only": True,
            "validation_or_sealed_replay_read": False,
        },
    }
    (root / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return root


def _library(tmp_path: Path, paths: int = 40, quality: bool = True) -> tuple[Path, str]:
    prepared = _prepared(tmp_path, paths, quality)
    root = tmp_path / ("library-quality" if quality else "library-fallback")
    adaptive.build_library(
        [prepared], root, clusters_per_anchor=8, deduplication_mode="unit_only",
    )
    digest = hashlib.sha256((root / "block_library_manifest.json").read_bytes()).hexdigest()
    return root, digest


def test_cold_hot_active_refs_variants_quality_and_hash_gate(tmp_path: Path) -> None:
    library, digest = _library(tmp_path)
    output = tmp_path / "portfolio"
    result = portfolio.build_portfolio(
        library_root=library,
        library_manifest_sha256=digest,
        output_root=output,
    )
    assert result["selection"]["historical_quality"]["mode"] == (
        "historical_provenance_tiebreaker"
    )
    assert result["selection"]["runtime_state_conditioned"] is False
    for anchor in adaptive.ANCHORS:
        tiers = result["per_anchor"][str(anchor)]
        assert tiers["cold"]["count"] == 40
        assert tiers["hot"]["count"] == 32
        assert tiers["active"]["count"] == 8
        hot_ids = {row["block_id"] for row in tiers["hot"]["blocks"]}
        active_ids = {row["block_id"] for row in tiers["active"]["blocks"]}
        assert active_ids <= hot_ids

    manifest = json.loads((library / "block_library_manifest.json").read_text(
        encoding="utf-8",
    ))
    by_id = {row["block_id"]: row for row in manifest["blocks"]}
    active216 = result["per_anchor"]["216"]["active"]["blocks"]
    assert any(row["market_variant_ref"]["variant_count"] == 2 for row in active216)
    teams = {
        source["team_name"]
        for selected in active216
        for source in by_id[selected["block_id"]]["source_lineage"]
    }
    assert len(teams) >= 4
    assert result["storage"] == {
        "actions_copied": False,
        "contracts_copied": False,
        "output_contains": "block/artifact row refs, selection reasons, and hashes only",
    }
    assert {path.name for path in output.iterdir()} == {
        "portfolio_manifest.json", "portfolio_manifest.json.sha256",
    }

    repeated = portfolio.build_portfolio(
        library_root=library,
        library_manifest_sha256=digest,
        output_root=tmp_path / "portfolio-repeat",
    )
    assert repeated["content_sha256"] == result["content_sha256"]
    vectors = library / manifest["vectors_file"]
    vectors.write_bytes(vectors.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="block vectors artifact SHA-256 mismatch"):
        portfolio.build_portfolio(
            library_root=library,
            library_manifest_sha256=digest,
            output_root=tmp_path / "tampered-output",
        )


def test_missing_quality_is_explicit_diversity_only_fallback(tmp_path: Path) -> None:
    library, digest = _library(tmp_path, paths=9, quality=False)
    result = portfolio.build_portfolio(
        library_root=library,
        library_manifest_sha256=digest,
        output_root=tmp_path / "fallback-portfolio",
    )
    quality = result["selection"]["historical_quality"]
    assert quality["mode"] == "diversity_only_missing_historical_outcomes"
    assert quality["missing_or_invalid_count"] == 10
    assert result["selection"]["hot_historical_quality_bonus"] == 0
    assert result["selection"]["active_historical_quality_bonus"] == 0
    active = result["per_anchor"]["216"]["active"]["blocks"]
    assert all(row["reason"].get("historical_quality") is None for row in active)
