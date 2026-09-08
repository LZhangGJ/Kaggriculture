"""Per-game context/reset/seat isolation and threaded repeat determinism."""
from pathlib import Path
import argparse
import collections
import hashlib
import json
import sys
import time

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "native/build"))
import _dp7_native as native


def signature(row):
    return {k: row[k] for k in ("seed", "seat", "steps", "cash", "opponent_cash",
                               "overflow", "selected_route")}


def trace_game(seed, seat, params, rival=None):
    env, own = native.Env(seed), native.Controller(params)
    rival = native.ThreeDay() if rival is None else rival
    digest = hashlib.sha256()
    for _ in range(719):
        actions = [None, None]
        actions[seat], actions[1-seat] = own.act(env, seat), rival.act(env, 1-seat)
        env.step(actions)
        digest.update(json.dumps(dict(actions=actions, state=env.observation(seat)), sort_keys=True).encode())
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    params = json.loads((EXP / "profiles/candidates.json").read_text())["S3C03"]
    tasks = [(seed, seat) for seed in range(20261401, 20261451) for seat in (0, 1)]
    started = time.perf_counter()
    serial = native.batch_three_day([x[0] for x in tasks], [x[1] for x in tasks], params, 1)
    repeated = tasks * 10
    tic = time.perf_counter()
    parallel = native.batch_three_day([x[0] for x in repeated], [x[1] for x in repeated], params, 16)
    parallel_seconds = time.perf_counter()-tic
    expected = {(r["seed"], r["seat"]): signature(r) for r in serial}
    for row in parallel:
        assert row["steps"] == 719
        assert signature(row) == expected[row["seed"], row["seat"]], row

    # Two environments on the same OS thread are deliberately interleaved.
    # Compare every action and post-step state via a full-trajectory digest.
    small = [tasks[0], tasks[-1]]
    independent = [trace_game(seed, seat, params) for seed, seat in small]
    envs = [native.Env(seed) for seed, _ in small]
    owns = [native.Controller(params) for _ in small]
    rivals = [native.ThreeDay() for _ in small]
    digests = [hashlib.sha256() for _ in small]
    for _ in range(719):
        for i, (_, seat) in enumerate(small):
            actions = [None, None]
            actions[seat] = owns[i].act(envs[i], seat)
            actions[1-seat] = rivals[i].act(envs[i], 1-seat)
            envs[i].step(actions)
            digests[i].update(json.dumps(dict(actions=actions, state=envs[i].observation(seat)), sort_keys=True).encode())
    assert independent == [d.hexdigest() for d in digests]
    # Reuse a completed controller across step zero, change seed and acting seat.
    for i, (seed, seat) in enumerate(small):
        assert trace_game(seed, seat, params, rivals[1-i]) == independent[i]

    branches = collections.Counter(r["selected_route"] for r in serial)
    receipt = dict(status="PASS", params=params,
                   build=json.loads((EXP / "native/build/build_receipt.json").read_text()),
                   independent_seed_count=50, serial_games=100, parallel_repeat_games=1000,
                   total_trajectory_hash_games=6, threads=16,
                   same_seed_both_seat_thread_repeat_mismatches=0,
                   interleaved_full_trajectory_mismatches=0, reused_reset_mismatches=0,
                   branch_counts={str(k): v for k, v in branches.items()},
                   parallel_seconds=parallel_seconds, parallel_games_per_second=1000/parallel_seconds,
                   seconds=time.perf_counter()-started,
                   caveat="Repeated games test isolation, not 1000 independent strength samples.",
                   serial_rows=serial)
    (out / "acceptance.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps({k: v for k, v in receipt.items() if k not in ("build", "params", "serial_rows")}), flush=True)


if __name__ == "__main__":
    main()
