from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_full_econ_implementation_catalog_covers_official_catalog_exactly() -> None:
    official = json.loads(
        (ROOT / "rules" / "official_action_event_catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    implemented = json.loads(
        (ROOT / "rules" / "full_econ_implementation_catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    for group in (
        "unit_actions",
        "market_actions",
        "automatic_and_random_events",
    ):
        assert set(implemented[group]) == set(official[group])
        assert all(
            isinstance(value.get("economic_item"), str)
            and value["economic_item"].strip()
            for value in implemented[group].values()
        )
    for source in implemented["evidence_tests"]:
        assert (ROOT / source).is_file()


def test_full_econ_implementation_catalog_has_no_untyped_future_event() -> None:
    implemented = json.loads(
        (ROOT / "rules" / "full_econ_implementation_catalog_v1.json").read_text(
            encoding="utf-8"
        )
    )
    allowed = {"EXACT", "EXPECTED", "SCENARIO", "CORE_ONLY"}
    for value in implemented["automatic_and_random_events"].values():
        encoded = value["mode"].replace("CORE_ONLY", "CORE-ONLY")
        modes = {
            "CORE_ONLY" if mode == "CORE-ONLY" else mode
            for mode in encoded.split("_")
        }
        assert modes
        assert modes <= allowed
