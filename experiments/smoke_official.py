#!/usr/bin/env python3
"""One full official-engine game crossing the replay-to-R1 handoff."""
from tempfile import TemporaryDirectory

from run_strong_ab import BOTS, play

with TemporaryDirectory() as trajectory_dir:
    row = play(("smoke", "agent/main.py", None, "0", "thomas_2945",
                BOTS["thomas_2945"], 2609400000, 0, "official", trajectory_dir))
print(row)
assert row["error"] is None, row
