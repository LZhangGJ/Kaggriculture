#!/usr/bin/env python3
"""Small regression check for direct-payoff leaf selection and serialization."""

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from meta_agent.src.search_route_policy import NumpySearchTree
from scripts.train_robust_search_route_trees import (
    _direct_node_classes,
    _direct_tree_payload,
    _predict_direct,
)


def main() -> None:
    matrix = np.asarray([[0.0], [0.1], [1.0], [1.1]], dtype=np.float32)
    wins = np.asarray([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float64)
    margins = np.asarray([[1, 9], [1, 9], [9, 1], [9, 1]], dtype=np.float64)
    model = DecisionTreeRegressor(max_depth=1, random_state=1).fit(matrix, wins)
    node_classes = _direct_node_classes(model, matrix, wins, margins)
    expected = _predict_direct(model, matrix, node_classes, ["left", "right"])
    runtime = NumpySearchTree(_direct_tree_payload(
        model, node_classes, ["left", "right"]
    ))
    assert expected.tolist() == ["left", "left", "right", "right"]
    assert [runtime.predict(row) for row in matrix] == expected.tolist()

    tied_wins = np.ones_like(wins)
    tied_classes = _direct_node_classes(model, matrix, tied_wins, margins)
    assert _predict_direct(model, matrix, tied_classes, ["left", "right"]).tolist() == [
        "right", "right", "left", "left",
    ]
    opponent_wins = np.asarray([[1, 0], [0, 1], [1, 1], [0, 1]], dtype=np.float64)
    opponent_classes = _direct_node_classes(
        DecisionTreeRegressor(max_depth=1, random_state=2).fit(
            np.zeros_like(matrix), opponent_wins
        ),
        np.zeros_like(matrix), opponent_wins, margins,
        np.asarray(["a", "b", "a", "b"]),
    )
    assert opponent_classes[0] == 1  # route 0 has worst-opponent win rate 0
    print("direct-payoff route tree: PASS")


if __name__ == "__main__":
    main()
