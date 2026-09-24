"""Run one bundled live-policy matchup with the frozen official referee.

Run this under compatible Linux x86-64/WSL Python; bundled a06.so is not a
Windows DLL. This verifies a single game, not Kaggle sandbox timing.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "referee"))
from policy_host import LocalGame, Policy, load_engine  # noqa: E402


def entry(group: str, name: str) -> Path:
    base = ROOT / group / name
    path = base / "main.py"
    if not path.is_file() or path.resolve().parent != base.resolve():
        raise ValueError(f"unknown {group} id: {name}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="one of the 13 bundled Agent IDs")
    parser.add_argument("--b", required=True, help="opponent ID")
    parser.add_argument("--b-group", choices=("agents", "opponents"), default="opponents")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--seat-a", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        own = Policy(str(entry("agents", args.a)), "bundled_a")
        rival = Policy(str(entry(args.b_group, args.b)), "bundled_b")
        env = LocalGame(args.seed, load_engine())
        while not env.done:
            actions = [None, None]
            for seat, policy in ((args.seat_a, own), (1 - args.seat_a, rival)):
                actions[seat] = policy(env.observation(seat), copy.deepcopy(env.configuration))
            env.advance(actions)
        cash = [farm["money"] for farm in env.state[0].observation.farms]
    assert env.t == 719
    print(json.dumps({"a": args.a, "b": args.b, "seed": args.seed, "seat_a": args.seat_a,
                      "steps": env.t, "cash_a": cash[args.seat_a],
                      "cash_b": cash[1 - args.seat_a]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
