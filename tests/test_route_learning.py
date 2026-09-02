import math

from kaggriculture_lab.route_learning import (
    CounterfactualRecord,
    ROUTE_CONTEXT_FEATURE_NAMES,
    RouteNormalizer,
    RouteValueNetwork,
    encode_route_context,
    records_to_batch,
    route_value_loss,
    scenario_batches,
    split_records_by_scenario,
    validate_counterfactual_groups,
)
from kaggriculture_lab.route_planner import PlanSpec


def _observation():
    empty_row = [None] * 10
    locked_row = ["LOCKED"] * 10
    farm = {
        "money": 3000,
        "farmer": [4, 4],
        "hands": [],
        "hires_today": 0,
        "unlocked_quadrants": ["NW"],
        "tiles": [empty_row[:] for _ in range(5)] + [locked_row[:] for _ in range(5)],
    }
    return {
        "player": 0,
        "step": 168,
        "day": 7,
        "hour": 0,
        "farms": [farm, dict(farm, money=2500)],
        "market": {
            "inventory": {item: 10000 for item in (
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER",
            )},
            "prices": {item: 100 for item in (
                "WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON",
                "EGG", "MILK", "WOOL", "FERTILIZER",
            )},
        },
        "town": {"unlocked_shops": ["YARN_STORE"]},
        "private": {"shed": {}, "seeds": {}, "inventories": [{}]},
    }


def _record(scenario, route, win, margin, plan):
    context = encode_route_context(_observation())
    return CounterfactualRecord(
        scenario_id=scenario,
        context_hash=f"hash-{scenario}",
        route_name=route,
        seed=1,
        seat=0,
        decision_step=168,
        opponent="starter",
        prefix_agent="starter",
        context=tuple(float(value) for value in context),
        plan=plan.normalized_features(),
        completed=1.0,
        win=win,
        margin=margin,
        terminal_cash=5000 + margin,
        opponent_cash=5000,
        steps=719,
        candidate_status="DONE",
        opponent_status="DONE",
    )


def test_context_dimension_and_group_validation():
    context = encode_route_context(_observation())
    assert context.shape == (len(ROUTE_CONTEXT_FEATURE_NAMES),)
    records = [
        _record("s1", "a", 1.0, 100, PlanSpec("a", cow_target=8)),
        _record("s1", "b", 0.0, -100, PlanSpec("b", sheep_target=8)),
    ]
    assert validate_counterfactual_groups(records) == {"s1": ("a", "b")}


def test_route_value_loss_backpropagates_and_uses_pairs():
    records = [
        _record("s1", "a", 1.0, 100, PlanSpec("a", cow_target=8)),
        _record("s1", "b", 0.0, -100, PlanSpec("b", sheep_target=8)),
        _record("s2", "a", 0.0, -50, PlanSpec("a", cow_target=8)),
        _record("s2", "b", 1.0, 50, PlanSpec("b", sheep_target=8)),
    ]
    normalizer = RouteNormalizer.fit(records)
    batch = records_to_batch(records, list(range(4)), normalizer, device="cpu")
    model = RouteValueNetwork(hidden_size=64)
    output = model(batch.context, batch.plan)
    losses = route_value_loss(output, batch)
    assert losses.pair_count == 2
    assert math.isfinite(float(losses.total.detach()))
    losses.total.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_split_and_scenario_batch_keep_groups_together():
    records = [
        _record(scenario, route, float(route == "a"), 10 if route == "a" else -10, PlanSpec(route))
        for scenario in ("s1", "s2", "s3")
        for route in ("a", "b")
    ]
    train, validation = split_records_by_scenario(records, validation_fraction=1 / 3, seed=3)
    assert {row.scenario_id for row in train}.isdisjoint(
        {row.scenario_id for row in validation}
    )
    batches = list(
        scenario_batches(records, scenarios_per_batch=1, shuffle=False, seed=0)
    )
    assert all(len(batch) == 2 for batch in batches)
