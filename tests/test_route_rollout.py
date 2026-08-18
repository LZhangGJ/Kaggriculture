import pytest

pytest.importorskip("kaggle_environments")

from kaggriculture_lab.route_learning import validate_counterfactual_groups
from kaggriculture_lab.route_planner import PlanSpec
from kaggriculture_lab.route_rollout import (
    RolloutScenario,
    RouteCandidate,
    collect_counterfactual_group,
)


def test_counterfactual_routes_share_one_decision_state():
    scenario = RolloutScenario(
        seed=123,
        seat=0,
        decision_step=718,
        prefix_agent="starter",
        opponent="starter",
    )
    candidates = [
        RouteCandidate("starter", "starter", PlanSpec("starter")),
        RouteCandidate("pass", "pass", PlanSpec("pass", cash_reserve=3000)),
    ]
    records = collect_counterfactual_group(scenario, candidates)
    assert len(records) == 2
    assert records[0].context_hash == records[1].context_hash
    assert validate_counterfactual_groups(records)[records[0].scenario_id] == (
        "starter",
        "pass",
    )
