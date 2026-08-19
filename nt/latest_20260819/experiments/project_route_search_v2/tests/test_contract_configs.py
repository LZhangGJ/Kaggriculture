from __future__ import annotations

import json
from pathlib import Path


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def _load(name: str) -> dict:
    return json.loads((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_all_m0_config_files_are_valid_json() -> None:
    expected = {
        "e0_contract_v1.json",
        "seed_panels_v1.json",
        "family_domains_v1.json",
        "species_capacity_and_coverage_v1.json",
        "performance_gates_v1.json",
        "search_budgets_v1.json",
    }
    assert expected.issubset({path.name for path in CONFIG_DIR.glob("*.json")})
    for name in expected:
        assert isinstance(_load(name), dict)


def test_family_and_domain_budgets_are_exactly_frozen() -> None:
    budgets = _load("search_budgets_v1.json")
    rows = budgets["family_domain_budgets"]
    assert sum(row["native"] for row in rows) == 66000
    assert sum(row["causal_hybrid"] for row in rows) == 94000
    assert sum(row["total"] for row in rows) == 160000
    assert all(row["native"] + row["causal_hybrid"] == row["total"] for row in rows)
    assert sum(budgets["candidate_generation_mix"].values()) == 1.0


def test_contract_forbids_future_events_and_competitive_null_claims() -> None:
    contract = _load("e0_contract_v1.json")
    constraints = contract["constraints"]
    assert constraints["future_events_visible_to_agent"] is False
    assert constraints["machine_learning_in_e0_search"] is False
    assert constraints["null_opponent_is_competitive_evidence"] is False
    assert constraints["official_holdout_requires_seat_swap"] is True

