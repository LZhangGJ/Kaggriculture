from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from kaggriculture_replay_diagnostic_simulator_v1 import (  # noqa: E402
    ReplayDiagnostics,
    ReplayValidationError,
)


DEFAULT_G304_REPLAY = (
    REPO_ROOT
    / "replay"
    / "public_submission_G304_55799276_20260827"
    / "G304_55799276"
    / "episodes"
    / "100198640.json"
)
G304_REPLAY = Path(os.environ.get("KAGGRICULTURE_G304_REPLAY", DEFAULT_G304_REPLAY))


@pytest.fixture(scope="module")
def g304() -> ReplayDiagnostics:
    if not G304_REPLAY.is_file():
        pytest.skip("G304 integration Replay is not available")
    return ReplayDiagnostics.from_path(G304_REPLAY)


def test_rejects_non_kaggriculture_replay() -> None:
    with pytest.raises(ReplayValidationError):
        ReplayDiagnostics({"name": "other", "steps": []})


def test_g304_metadata_and_timeline_are_complete(g304: ReplayDiagnostics) -> None:
    meta = g304.meta()
    assert meta["module_version"] == "1.32.7"
    assert meta["steps"] == 720
    assert meta["players"] == ["Noct-Tech zombie", "QQ Farming"]
    assert len(g304.timeline()) == 720


def test_worker_action_success_and_failure_are_explained(g304: ReplayDiagnostics) -> None:
    frame10 = g304.frame(10)
    seat0 = frame10["seats"][0]
    assert any(
        worker["op"] == "WATER" and worker["status"] == "success"
        for worker in seat0["workers"]
    )
    assert any(
        worker["op"] == "PLANT" and worker["status"] == "success"
        for worker in seat0["workers"]
    )

    frame84 = g304.frame(84)
    seat1 = frame84["seats"][1]
    failed = [worker for worker in seat1["workers"] if worker["status"] == "noop"]
    assert any(worker["op"] == "WATER" for worker in failed)


def test_imminent_plant_and_animal_loss_alarms_exist(g304: ReplayDiagnostics) -> None:
    plant_keys = {
        alarm["key"]
        for alarm in g304.frame(3)["seats"][0]["alarms"]
    }
    assert "plant_at_risk" in plant_keys

    animal_keys = {
        alarm["key"]
        for alarm in g304.frame(216)["seats"][1]["alarms"]
    }
    assert "animal_at_risk" in animal_keys


def test_cross_day_daily_flags_are_not_misreported_as_failure(g304: ReplayDiagnostics) -> None:
    frame24 = g304.frame(24)
    cross_day = [
        worker
        for worker in frame24["seats"][1]["workers"]
        if worker["op"] in {"WATER", "FEED", "CARE"}
    ]
    assert cross_day
    assert all(worker["status"] != "noop" for worker in cross_day)
