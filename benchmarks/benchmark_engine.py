"""Benchmark official and fast Kaggriculture episode runners."""

from __future__ import annotations

import argparse
import io
import time
from contextlib import redirect_stderr, redirect_stdout

from kaggriculture_lab import ENGINE_VERSION, run_duel, run_fast_episode


def official_episode(seed: int) -> tuple[float | None, float | None]:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    env = make(
        "kaggriculture",
        configuration={"episodeSteps": 720, "seed": seed},
        debug=False,
    )
    env.run(["starter", "starter"])
    return tuple(state.reward for state in env.steps[-1])


def timed(label: str, count: int, function) -> tuple[str, int, float, float]:
    started = time.perf_counter()
    function()
    elapsed = time.perf_counter() - started
    return label, count, elapsed, count / elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-episodes", type=int, default=3)
    parser.add_argument("--fast-episodes", type=int, default=40)
    parser.add_argument("--parallel-episodes", type=int, default=128)
    parser.add_argument("--workers", type=int, nargs="*", default=[2, 4, 8])
    args = parser.parse_args()

    official_episode(0)
    run_fast_episode(seed=0)

    rows = [
        timed(
            "official",
            args.official_episodes,
            lambda: [official_episode(seed) for seed in range(args.official_episodes)],
        ),
        timed(
            "fast-serial",
            args.fast_episodes,
            lambda: [run_fast_episode(seed=seed) for seed in range(args.fast_episodes)],
        ),
    ]
    for workers in args.workers:
        rows.append(
            timed(
                f"fast-{workers}proc",
                args.parallel_episodes,
                lambda workers=workers: run_duel(
                    "starter",
                    "starter",
                    range(args.parallel_episodes),
                    both_seats=False,
                    workers=workers,
                ),
            )
        )

    official_eps = rows[0][3]
    print(f"kaggle-environments={ENGINE_VERSION}")
    print(f"{'runner':16s} {'episodes':>8s} {'seconds':>10s} {'episodes/s':>12s} {'speedup':>9s}")
    for label, count, seconds, throughput in rows:
        print(
            f"{label:16s} {count:8d} {seconds:10.3f} "
            f"{throughput:12.2f} {throughput / official_eps:8.1f}x"
        )


if __name__ == "__main__":
    main()

