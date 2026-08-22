from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from kaggriculture_lab import FastKaggricultureEnv
from kaggriculture_lab.official_replay_v7 import (
    OFFICIAL_V7_MANIFEST_SCHEMA,
    behavior_signature,
    build_v7_label_bundle,
    extract_replay_actor_sequences,
    load_official_v7_training_records,
    resimulation_audit,
    write_v7_label_bundle,
)


def _small_replay() -> dict:
    environment = FastKaggricultureEnv(
        configuration={"episodeSteps": 4}, copy_observations=True
    )
    observations = environment.reset(seed=17)
    placeholder = {"farmer": ["PASS"], "hands": [], "market": []}
    steps = [
        [
            {
                "status": "ACTIVE",
                "observation": copy.deepcopy(observations[player]),
                "action": copy.deepcopy(placeholder),
                "reward": 0.0,
            }
            for player in range(2)
        ]
    ]
    actions = [
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        },
        {"farmer": ["PASS"], "hands": [], "market": []},
    ]
    while not environment.done:
        result = environment.step(actions)
        steps.append(
            [
                {
                    "status": result.statuses[player],
                    "observation": copy.deepcopy(result.observations[player]),
                    "action": copy.deepcopy(actions[player]),
                    "reward": result.rewards[player],
                }
                for player in range(2)
            ]
        )
    # Match Kaggle's compact shared-step representation for player 1.
    for step in steps:
        step[1]["observation"]["step"] = None
    return {
        "name": "kaggriculture",
        "configuration": {"episodeSteps": 4},
        "info": {"seed": 17, "TeamNames": ["left", "right"]},
        "rewards": list(result.rewards),
        "steps": steps,
    }


def test_extract_replay_uses_next_state_action_and_restores_shared_step() -> None:
    replay = _small_replay()
    left, right = extract_replay_actor_sequences(replay)
    assert len(left.actions) == len(replay["steps"]) - 1
    assert left.actions[0]["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert left.observations[0]["step"] == 0
    assert right.observations[1]["step"] == 1
    assert left.teacher == "left"
    assert right.teacher == "right"


def test_resimulation_audit_matches_trusted_interpreter() -> None:
    audit = resimulation_audit(_small_replay())
    assert audit["available"] is True
    assert audit["matched"] is True
    assert audit["checked_transitions"] == 3


def test_behavior_signature_is_finite_and_action_hash_is_stable() -> None:
    left, _ = extract_replay_actor_sequences(_small_replay())
    signature = behavior_signature(left)
    assert signature.ndim == 1
    assert signature.size > 20
    assert signature.tolist() == pytest.approx(behavior_signature(left).tolist())
    assert len(left.action_hash) == 64


def test_audited_official_labels_load_as_bc_records_without_mode_supervision(
    tmp_path: Path,
) -> None:
    replay = _small_replay()
    raw_dir = tmp_path / "raw"
    dataset_dir = tmp_path / "processed"
    raw_dir.mkdir()
    raw_path = raw_dir / "17.json"
    raw_path.write_text(json.dumps(replay), encoding="utf-8")
    bundle = build_v7_label_bundle(
        replay,
        episode_id=17,
        raw_path=raw_path,
        source_date="2026-08-22",
        daily_score_rank=1,
        average_score=10.0,
        teacher_clusters=(2, 3),
        validate_resimulation=False,
    )
    assert len(bundle["actors"][0]["budget_targets"]) == 3
    assert bundle["actors"][0]["budget_targets"][0]["total_spend"] == pytest.approx(
        30.0
    )
    label_path = dataset_dir / "labels" / "episode-17-v7.json.gz"
    write_v7_label_bundle(label_path, bundle)
    manifest_path = dataset_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": OFFICIAL_V7_MANIFEST_SCHEMA,
                "source_replay_dir": str(raw_dir),
                "labels": {"intent_horizon": 24, "inverse_top_m": 3},
                "value_targets": {
                    "gamma": 0.997,
                    "reward_scale": 10.0,
                    "win_bonus": 1.0,
                },
                "episodes": [
                    {
                        "episode_id": 17,
                        "raw_file": raw_path.name,
                        "label_file": str(label_path.relative_to(dataset_dir)),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (dataset_dir / "audit.json").write_text(
        json.dumps({"valid": True}), encoding="utf-8"
    )

    loaded = load_official_v7_training_records([manifest_path])

    assert loaded.episodes == 1
    assert loaded.actors == 2
    assert len(loaded.records) == 6
    assert loaded.quality_tiers == {"silver": 1}
    assert loaded.trace_audit["discontinuous_routes"] == 0
    assert all(not record.initial_mode_active for record in loaded.records)
    assert all(not record.daily_switch_active for record in loaded.records)
    assert loaded.records[0].action["market"] == [["BUY_SEED", "WHEAT", 1]]
    assert loaded.records[0].budget_target.total_spend == pytest.approx(30.0)
    assert loaded.records[0].budget_target.spend_active
    assert {record.teacher_cluster for record in loaded.records} == {
        1_000_002,
        1_000_003,
    }
