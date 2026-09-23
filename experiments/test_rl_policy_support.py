"""Small contract check for native rollout policy-support metrics."""

import numpy as np

from experiments.train_student_action_event_rl_v3 import _native_policy_support_by_step


def test_native_policy_support_by_step():
    report = _native_policy_support_by_step({
        "day_step": np.array([288, 480]),
        "event_day_index": np.array([0, 0, 1]),
        "event_stage": np.array([0, 1, 1]),
        "event_legal_mask": np.array([1, 3, 3]),
        "old_logprob": np.log(np.array([1.0, 0.995, 0.5])),
    })
    assert report["288"]["actionable_events"] == 1
    assert report["288"]["chosen_probability_ge_0_99"] == 1.0
    assert report["480"]["chosen_probability_ge_0_99"] == 0.0
    assert report["288"]["by_stage"]["1"]["actionable_events"] == 1
    assert "0" not in report["288"]["by_stage"]


if __name__ == "__main__":
    test_native_policy_support_by_step()
