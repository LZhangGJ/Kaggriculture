#!/usr/bin/env python3
"""Verify offline diagnostics change only their named setting."""
from run_r1_candidate_oracle import run


row = run(("thomas_2945", 2610100049, 0, "handoff", None))
assert not row["error"], row["error"]
bases = [p for p in row["proposals"] if not p["diagnostic"]]
default = max(bases, key=lambda p: p["score"])
diagnostics = {p["diagnostic"]: p for p in row["proposals"] if p["diagnostic"]}
sale, crop = diagnostics["competitive_sale"], diagnostics["crop_succession"]
for candidate in (sale, crop):
    assert candidate["discount"] == default["discount"]
    assert candidate["capital_power"] == default["capital_power"]
assert sale["delay_sale"] == 1 and crop["rotation"] == crop["repeat"] == 1
print({"status": "PASS", "default_id": default["id"],
       "discount": default["discount"], "capital_power": default["capital_power"]})
