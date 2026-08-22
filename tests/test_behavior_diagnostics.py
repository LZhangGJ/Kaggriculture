from __future__ import annotations

import numpy as np

from kaggriculture_lab.behavior_diagnostics import (
    ResponseRecord,
    build_response_features,
    cluster_behavior_profiles,
    common_opponents,
)


def test_common_opponent_features_and_balanced_clusters_are_deterministic() -> None:
    agents = ("a0", "a1", "b0", "b1")
    opponents = ("x", "y", "z")
    records: list[ResponseRecord] = []
    for agent_index, agent in enumerate(agents):
        high = agent.startswith("a")
        for opponent_index, opponent in enumerate(opponents):
            score = (0.82 if high else 0.18) + 0.01 * opponent_index
            records.append(
                ResponseRecord(
                    agent,
                    opponent,
                    score,
                    100_000.0 if high else 70_000.0,
                    12_000.0 if high else -12_000.0,
                    100,
                    "synthetic",
                )
            )
    assert common_opponents(records, agents) == opponents
    features, names, missing = build_response_features(records, agents, opponents)
    first = cluster_behavior_profiles(
        agents, opponents, features, names, missing, max_clusters=2
    )
    second = cluster_behavior_profiles(
        agents, opponents, features, names, missing, max_clusters=2
    )

    np.testing.assert_array_equal(first.clusters, second.clusters)
    assert first.cluster_count == 2
    assert first.clusters[0] == first.clusters[1]
    assert first.clusters[2] == first.clusters[3]
    assert first.clusters[0] != first.clusters[2]
    assert np.isclose(first.balanced_weights.mean(), 1.0)


def test_missing_self_response_is_explicitly_imputed() -> None:
    records = [
        ResponseRecord("a", "b", 0.7, 90_000, 1_000, 10, "x"),
        ResponseRecord("b", "a", 0.3, 80_000, -1_000, 10, "x"),
    ]
    features, _, missing = build_response_features(
        records, ("a", "b"), ("a", "b")
    )
    assert np.isfinite(features).all()
    assert int(missing.sum()) == 6
