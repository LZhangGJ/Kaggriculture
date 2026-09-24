#!/usr/bin/env python3
from audit_outer_handoff_suffix import closed_loop_candidates


def test_nested_shortlist():
    rows = [
        {"index": 1, "score": 9, "diagnostic": ""},
        {"index": 2, "score": 8, "diagnostic": ""},
        {"index": 3, "score": 10, "diagnostic": "outer_a"},
        {"index": 4, "score": 7, "diagnostic": "outer_b"},
    ]
    baseline = closed_loop_candidates(rows, "baseline", 2)
    outer = closed_loop_candidates(rows, "outer", 2)
    assert {row["index"] for row in baseline} <= {row["index"] for row in outer}


if __name__ == "__main__":
    test_nested_shortlist()
