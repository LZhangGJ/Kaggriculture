from __future__ import annotations

import copy

import fast_kaggriculture as fk


FAMILIES = {
    "KEEP",
    "SCHEDULE_LAYOUT",
    "CONTINUOUS_SCALE",
    "UNILATERAL",
    "MULTI_PROJECT",
    "TIMING",
    "MARKET_TRANSACTION",
    "PHASE_SUFFIX",
    "LOCAL_RECOVERY",
}


def candidate_context() -> dict:
    # A deliberately permissive mid-game state.  The values describe game
    # constraints, not an expert route or an opponent identity.
    return {
        "day": 8,
        "targets": [8, 4, 2, 10, 5, 2, 4, 4],
        "irreversible_floor": [4, 0, 0, 4, 2, 1, 2, 2],
        "caps": [40, 24, 24, 40, 24, 12, 20, 20],
        "marginal_value": [600, 900, 1500, 2600, 2200, 6000, 4600, 3200],
        "purchase_cost": [48, 42, 96, 228, 169, 1800, 1200, 600],
        "daily_action_load": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.7, 0.5],
        "first_cash_lag_days": [3, 4, 5, 6, 7, 3, 4, 2],
        "liquid_cash": 100_000,
        "protected_cash": 1_000,
        "financeable_inventory_value": 5_000,
        "unlocked_quadrants": 2,
        "maximum_quadrants": 4,
        "productive_tiles": 39,
        "hands": 6,
        "maximum_hands": 12,
        "next_hand_cost": 100,
        "next_quadrant_cost": 4_000,
        "tiles_per_quadrant": 25,
        "current_daily_action_load": 80,
        "hard_deadline_load": 10,
        "estimated_travel_load": 20,
        "delayed_loss": 300,
        "market_slots_available": 10,
        "sellable_inventory": [10, 8, 4, 2, 6, 3, 5, 2, 1],
        "market_prices": [48, 42, 96, 228, 169, 60, 80, 120, 90],
        "demand_within_day": [2, 1, 0, 4, 2, 0, 3, 1, 0],
        "recovery_issues": 15,
    }


def test_candidate_kernel_exposes_all_rank1_derived_families() -> None:
    assert set(fk.adaptive_candidate_family_names()) == FAMILIES
    result = fk.generate_adaptive_candidates(candidate_context())

    assert 300 <= result["raw_count"] <= 1_000
    assert 100 <= result["feasible_count"] <= result["raw_count"]
    assert len(result["shortlist"]) == 64
    assert set(result["feasible_by_family"]) == FAMILIES
    assert all(result["feasible_by_family"][family] > 0 for family in FAMILIES)
    assert all(result["shortlisted_by_family"][family] > 0 for family in FAMILIES)


def test_shortlist_is_deterministic_unique_and_behavioral() -> None:
    first = fk.generate_adaptive_candidates(candidate_context())
    second = fk.generate_adaptive_candidates(candidate_context())
    first_signatures = [row["signature"] for row in first["shortlist"]]
    second_signatures = [row["signature"] for row in second["shortlist"]]

    assert first_signatures == second_signatures
    assert len(first_signatures) == len(set(first_signatures))
    for row in first["shortlist"]:
        if row["family"] == "KEEP":
            continue
        assert (
            any(row["target_delta"])
            or row["hand_delta"]
            or row["quadrant_delta"]
            or row["effective_delay_days"]
            or row["schedule_profile"]
            or row["market_profile"]
            or row["recovery_profile"]
            or row["suffix_project"] >= 0
            or row["market_item"] >= 0
            or row["recovery_issue"]
        )


def test_joint_capacity_and_irreversible_floors_are_respected() -> None:
    context = candidate_context()
    result = fk.generate_adaptive_candidates(context, max_shortlist=256)
    for row in result["feasible"]:
        final_targets = [
            target + change
            for target, change in zip(context["targets"], row["target_delta"])
        ]
        assert all(
            floor <= target <= cap
            for target, floor, cap in zip(
                final_targets, context["irreversible_floor"], context["caps"]
            )
        )
        assert context["hands"] + row["hand_delta"] <= context["maximum_hands"]
        assert (
            context["unlocked_quadrants"] + row["quadrant_delta"]
            <= context["maximum_quadrants"]
        )


def test_live_commitment_above_new_cap_can_reduce_or_change_scheduler() -> None:
    context = candidate_context()
    context["targets"][5] = 8
    context["irreversible_floor"][5] = 4
    context["caps"][5] = 6
    result = fk.generate_adaptive_candidates(context, max_shortlist=256)

    assert result["feasible_count"] > 1
    assert any(
        row["family"] == "SCHEDULE_LAYOUT" for row in result["feasible"]
    )
    assert any(row["target_delta"][5] < 0 for row in result["feasible"])
    assert all(
        context["targets"][5] + row["target_delta"][5] <= 8
        for row in result["feasible"]
    )


def test_live_market_and_recovery_context_gates_candidates() -> None:
    context = candidate_context()
    context["sellable_inventory"] = [0] * 9
    context["recovery_issues"] = 0
    result = fk.generate_adaptive_candidates(context)
    assert result["raw_by_family"]["MARKET_TRANSACTION"] == 0
    assert result["raw_by_family"]["LOCAL_RECOVERY"] == 0
    assert result["feasible_by_family"]["MARKET_TRANSACTION"] == 0
    assert result["feasible_by_family"]["LOCAL_RECOVERY"] == 0


def test_endgame_and_cash_constraints_remove_infeasible_expansion() -> None:
    context = candidate_context()
    context["day"] = 29
    context["liquid_cash"] = context["protected_cash"]
    context["financeable_inventory_value"] = 0
    result = fk.generate_adaptive_candidates(context, max_shortlist=256)
    for row in result["feasible"]:
        # No positive production target may survive when its first cash event
        # lies beyond the official 30-day horizon.
        assert not any(change > 0 for change in row["target_delta"])


def test_soft_quotas_scale_without_becoming_hard_required_slots() -> None:
    full = fk.generate_adaptive_candidates(candidate_context(), max_shortlist=64)
    compact = fk.generate_adaptive_candidates(candidate_context(), max_shortlist=32)
    assert len(full["shortlist"]) == 64
    assert len(compact["shortlist"]) == 32
    assert compact["shortlisted_by_family"]["KEEP"] == 1
    assert compact["shortlisted_by_family"]["MULTI_PROJECT"] >= 7
    assert compact["shortlisted_by_family"]["CONTINUOUS_SCALE"] >= 6


def test_context_changes_candidate_set_without_author_or_route_switches() -> None:
    base = candidate_context()
    changed = copy.deepcopy(base)
    changed["targets"][6] += 3
    changed["irreversible_floor"][6] += 3
    changed["estimated_travel_load"] += 50
    changed["recovery_issues"] = 1

    base_result = fk.generate_adaptive_candidates(base)
    changed_result = fk.generate_adaptive_candidates(changed)
    assert [row["signature"] for row in base_result["shortlist"]] != [
        row["signature"] for row in changed_result["shortlist"]
    ]
    # The public interface intentionally has no expert/player/source selector.
    forbidden = {"author", "player", "rank", "replay", "route_id", "opponent_id"}
    assert not (forbidden & set(base))


def test_expansion_can_compose_sell_first_financing_transaction() -> None:
    context = candidate_context()
    context["liquid_cash"] = 1_500
    context["protected_cash"] = 1_000
    context["financeable_inventory_value"] = 5_000

    financed = fk.generate_adaptive_candidates(context, max_shortlist=256)
    cow_expansions = [
        row
        for row in financed["feasible"]
        if row["target_delta"][5] > 0
    ]
    assert cow_expansions
    assert any(
        row["market_profile"] == 4 and row["market_item"] >= 0
        for row in cow_expansions
    )

    no_inventory = copy.deepcopy(context)
    no_inventory["financeable_inventory_value"] = 0
    no_inventory["sellable_inventory"] = [0] * 9
    without_financing = fk.generate_adaptive_candidates(
        no_inventory, max_shortlist=256
    )
    assert not any(
        row["target_delta"][5] > 0
        for row in without_financing["feasible"]
    )


def test_local_recovery_is_a_sparse_edit_of_only_the_live_issue() -> None:
    result = fk.generate_adaptive_candidates(candidate_context(), max_shortlist=256)
    recovery = [
        row for row in result["feasible"] if row["family"] == "LOCAL_RECOVERY"
    ]
    assert recovery
    for row in recovery:
        assert row["target_delta"] == [0] * 8
        assert row["hand_delta"] == 0
        assert row["quadrant_delta"] == 0
        assert row["market_profile"] == 0
        assert row["market_item"] == -1
        assert row["recovery_profile"] in {1, 2, 3}
        assert row["recovery_issue"] in {1, 2, 4, 8}
