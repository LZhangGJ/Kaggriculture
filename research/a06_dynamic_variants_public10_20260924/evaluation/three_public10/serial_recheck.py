"""Reproduce selected strong and weak matchups in a single process."""

from __future__ import annotations

import json

from run import HERE, play, read


def main() -> None:
    protocol = read(HERE / "PROTOCOL.json")
    prior = [json.loads(s) for s in (HERE / "games.jsonl").read_text(encoding="utf-8").splitlines() if s]
    candidates = {r["id"]: r for r in protocol["candidates"]}
    opponents = {r["ref"]: r for r in protocol["opponents"]}
    first_public = protocol["opponents"][0]["ref"]
    liquidity_win_seed = next(seed for seed in protocol["seeds"]
                              if all(any(r["local"] == "r14_liquidity" and r["public"] == first_public
                                             and r["seed"] == seed and r["local_seat"] == seat and r["local_win"] == 1
                                             for r in prior) for seat in (0, 1)))
    selected = [
        ("r14_liquidity", first_public, liquidity_win_seed),
        ("r14_liquidity", protocol["opponents"][6]["ref"], protocol["seeds"][1]),
        ("r14_tl5", protocol["opponents"][0]["ref"], protocol["seeds"][2]),
    ]
    receipt = []
    for candidate, opponent, seed in selected:
        for seat in (0, 1):
            old = next(r for r in prior if r["local"] == candidate and r["public"] == opponent
                       and r["seed"] == seed and r["local_seat"] == seat)
            job = (candidate, candidates[candidate]["entry"], opponent, opponents[opponent]["entry"], seed, seat)
            fresh = play(job)
            fields = ("steps", "local_cash", "public_cash", "margin", "local_win", "tie")
            if fresh["error"] or any(fresh.get(field) != old.get(field) for field in fields):
                raise ValueError(f"serial result mismatch: {candidate} / {opponent} / {seed} / {seat}")
            receipt.append({"candidate": candidate, "public": opponent, "seed": seed, "seat": seat,
                            "margin": fresh["margin"], "max_local_action_s": fresh["max_local_action_s"]})
    result = {"games": len(receipt), "exact_cash_and_result_matches": len(receipt), "rows": receipt}
    (HERE / "SERIAL_RECHECK.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"games": len(receipt), "exact_matches": len(receipt)}))


if __name__ == "__main__":
    main()
