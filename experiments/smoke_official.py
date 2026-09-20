#!/usr/bin/env python3
"""One full official-engine game crossing the replay-to-R1 handoff."""
from run_strong_ab import BOTS, play

row = play(("smoke", "agent/main.py", None, "0", "thomas_2945",
            BOTS["thomas_2945"], 2609400000, 0))
print(row)
assert row["error"] is None, row
