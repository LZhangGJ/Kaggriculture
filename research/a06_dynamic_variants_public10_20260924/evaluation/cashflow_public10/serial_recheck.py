"""Reproduce representative match results with one game at a time."""

from __future__ import annotations

import json
from pathlib import Path

from run import HERE, play, read


def main() -> None:
    protocol = read(HERE / "PROTOCOL.json")
    rows = [json.loads(line) for line in (HERE / "games.jsonl").read_text(encoding="utf-8").splitlines() if line]
    selected = [(protocol["opponents"][0], protocol["seeds"][0]),
                (protocol["opponents"][6], protocol["seeds"][1])]
    result = []
    for opponent, seed in selected:
        for seat in (0, 1):
            original = next(row for row in rows if row["public"] == opponent["ref"]
                            and row["seed"] == seed and row["local_seat"] == seat)
            job = (protocol["candidate"]["id"], protocol["candidate"]["entry"],
                   opponent["ref"], opponent["entry"], seed, seat)
            fresh = play(job)
            fields = ("steps", "local_cash", "public_cash", "margin", "local_win", "tie")
            if fresh["error"] or any(fresh.get(field) != original.get(field) for field in fields):
                raise ValueError(f"serial mismatch: {opponent['ref']} seed={seed} seat={seat}")
            result.append({"public": opponent["ref"], "seed": seed, "seat": seat,
                           "margin": fresh["margin"], "max_local_action_s": fresh["max_local_action_s"],
                           "max_public_action_s": fresh["max_public_action_s"]})
    output = {"games": len(result), "exact_cash_and_result_matches": len(result), "rows": result}
    (HERE / "SERIAL_RECHECK.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"games": len(result), "exact_matches": len(result)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
