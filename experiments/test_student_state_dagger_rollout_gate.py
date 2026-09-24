#!/usr/bin/env python3
"""Small regression check for the multi-day student rollout publication gate."""

from __future__ import annotations

import copy
import unittest

from experiments.build_student_state_dagger_v3 import _validate_student_rollout


class StudentRolloutGateTest(unittest.TestCase):
    def setUp(self):
        self.summary = {
            "student_steps": [288, 312, 336],
            "successful_steps": [288, 312, 336],
            "checkpoint_attested_action_steps": [288, 312],
            "unattested_exploration_steps": [336],
            "days": 3,
            "fallbacks": 0,
            "failures": [],
            "illegal": 0,
        }

    def validate(self, summary):
        _validate_student_rollout(summary, (288, 312, 336), (288, 312), (336,))

    def test_exact_rollout_passes(self):
        self.validate(self.summary)

    def test_missing_duplicate_fallback_and_illegal_fail(self):
        mutations = (
            ("successful_steps", [288, 312]),
            ("successful_steps", [288, 312, 312]),
            ("fallbacks", 1),
            ("failures", [{"step": 336}]),
            ("illegal", 1),
            ("unattested_exploration_steps", []),
        )
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                summary = copy.deepcopy(self.summary)
                summary[key] = value
                with self.assertRaises(RuntimeError):
                    self.validate(summary)


if __name__ == "__main__":
    unittest.main()
