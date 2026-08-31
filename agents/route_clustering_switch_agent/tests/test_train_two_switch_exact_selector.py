import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "train_two_switch_exact_selector.py"
SPEC = importlib.util.spec_from_file_location("train_two_switch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_backward_policy_uses_both_stages() -> None:
    early = np.asarray([[0], [0], [1], [1]], dtype=np.float32)
    late = np.stack([early, early])
    scores = np.zeros((2, 2, 4), dtype=np.float32)
    margins = np.zeros_like(scores)
    scores[0, 1, :2] = 1
    scores[1, 1, 2:] = 1
    policy = MODULE.fit_policy(
        early, late, scores, margins, np.arange(4), 0, 0.0
    )
    selected, _ = MODULE.selected_values(policy)
    assert selected.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert policy["early_actions"].tolist() == [0, 1]


if __name__ == "__main__":
    test_backward_policy_uses_both_stages()
