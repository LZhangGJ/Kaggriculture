from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parents[1]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_seed_panels_are_cumulative_and_holdout_is_independent() -> None:
    manifest = _load_json(PROJECT_DIR / "configs" / "seed_panels_v1.json")
    panels = {name: spec["seeds"] for name, spec in manifest["panels"].items()}
    assert len(panels["E0_SEARCH_16"]) == 16
    assert len(panels["E0_PROMOTE_32"]) == 32
    assert len(panels["E0_AUDIT_128"]) == 128
    assert len(panels["E0_OFFICIAL_HOLDOUT_64"]) == 64
    assert panels["E0_PROMOTE_32"][:16] == panels["E0_SEARCH_16"]
    assert panels["E0_AUDIT_128"][:32] == panels["E0_PROMOTE_32"]
    assert set(panels["E0_AUDIT_128"]).isdisjoint(
        panels["E0_OFFICIAL_HOLDOUT_64"]
    )


def test_derived_event_bank_matches_manifest_and_selected_seeds() -> None:
    manifest = _load_json(PROJECT_DIR / "configs" / "seed_panels_v1.json")
    derived = REPO_ROOT / manifest["derived_event_bank"]
    source = REPO_ROOT / manifest["source_event_bank"]
    assert _sha256(source) == manifest["source_event_bank_sha256"]
    assert _sha256(derived) == manifest["derived_event_bank_sha256"]
    with np.load(derived, allow_pickle=False) as bank:
        assert bank.files == ["event_seeds", "weed_spawn", "shop_choice"]
        assert bank["event_seeds"].shape == (194,)
        assert bank["weed_spawn"].shape == (194, 30, 200)
        assert bank["shop_choice"].shape == (194, 30, 201)
        assert len(np.unique(bank["event_seeds"])) == 194


def test_freeze_receipt_hashes_every_declared_file() -> None:
    receipt = _load_json(PROJECT_DIR / "receipts" / "m0_contract_freeze_v1.json")
    assert receipt["status"] == "PASS"
    assert receipt["scope"] == "M0_CONTRACT_AND_EMPTY_PACKAGE_ONLY"
    for relative, facts in receipt["files"].items():
        path = REPO_ROOT / relative
        assert path.stat().st_size == facts["bytes"]
        assert _sha256(path) == facts["sha256"]

