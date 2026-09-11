from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

EPISODE_STEPS: int = 720
REPLAYS_DIR: Path = Path("replays")
BUILTIN_AGENTS: frozenset[str] = frozenset({"random", "starter", "pass"})


@dataclass(frozen=True)
class MatchResult:
    seed: int
    bot_a: str
    bot_b: str
    reward_a: float
    reward_b: float
    error: str | None
    replay_path: str | None = None


def _resolve_bot(name: str) -> str:
    if name in BUILTIN_AGENTS:
        return name
    path = Path(name)
    if not path.is_file():
        raise FileNotFoundError(f"Bot script not found: {name}")
    return str(path.resolve())


def _bot_display_name(name: str) -> str:
    if name in BUILTIN_AGENTS:
        return name
    p = Path(name)
    return p.parent.name if p.stem == "main" and p.parent.name else p.stem


def _save_replay(env: object, bot_a: str, bot_b: str, seed: int, fmt: str) -> str:
    REPLAYS_DIR.mkdir(exist_ok=True)
    base_name = f"replay_{seed}_{_bot_display_name(bot_a)}_vs_{_bot_display_name(bot_b)}"
    saved: list[str] = []

    if fmt in ("html", "both"):
        html_path = REPLAYS_DIR / f"{base_name}.html"
        html_path.write_text(getattr(env, "render")(mode="html"), encoding="utf-8")
        saved.append(str(html_path))

    if fmt in ("json", "both"):
        json_path = REPLAYS_DIR / f"{base_name}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(getattr(env, "toJSON")(), f)
        saved.append(str(json_path))

    return ", ".join(saved)


def run_single_match(bot_a: str, bot_b: str, seed: int, save_format: str = "none") -> MatchResult:
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        from kaggle_environments import make

    spec_a, spec_b = _resolve_bot(bot_a), _resolve_bot(bot_b)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            env = make(
                "kaggriculture",
                configuration={"episodeSteps": EPISODE_STEPS, "seed": seed},
                debug=True,
            )
        env.run([spec_a, spec_b])
        final = env.steps[-1]

        error = next(
            (
                f"Player {i} ({[bot_a, bot_b][i]}): status={state.status}"
                for i, state in enumerate(final)
                if state.status != "DONE"
            ),
            None,
        )

        reward_a = float(final[0].reward) if final[0].reward is not None else 0.0
        reward_b = float(final[1].reward) if final[1].reward is not None else 0.0

        replay_path = None
        if error is not None or save_format != "none":
            fmt = save_format if save_format != "none" else "json"
            replay_path = _save_replay(env, bot_a, bot_b, seed, fmt)

        return MatchResult(seed, bot_a, bot_b, reward_a, reward_b, error, replay_path)

    except Exception:
        return MatchResult(seed, bot_a, bot_b, 0.0, 0.0, traceback.format_exc(), None)


def _run_match_pair(args: tuple[str, str, int, str]) -> tuple[MatchResult, MatchResult]:
    bot_a, bot_b, seed, save_fmt = args
    return (
        run_single_match(bot_a, bot_b, seed, save_fmt),
        run_single_match(bot_b, bot_a, seed, save_fmt),
    )


@dataclass
class BotStats:
    games: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_coins: float = 0.0
    total_opp_coins: float = 0.0
    errors: int = 0

    @property
    def winrate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games * 100 if self.games > 0 else 0.0

    @property
    def avg_coins(self) -> float:
        return self.total_coins / self.games if self.games > 0 else 0.0

    @property
    def avg_opp_coins(self) -> float:
        return self.total_opp_coins / self.games if self.games > 0 else 0.0

    @property
    def avg_delta(self) -> float:
        return self.avg_coins - self.avg_opp_coins


def compute_all_stats(results: list[MatchResult], bots: list[str]) -> dict[str, BotStats]:
    stats = {b: BotStats() for b in bots}
    for r in results:
        for bot, self_r, opp_r in ((r.bot_a, r.reward_a, r.reward_b), (r.bot_b, r.reward_b, r.reward_a)):
            if bot not in stats:
                continue
            s = stats[bot]
            s.games += 1
            s.total_coins += self_r
            s.total_opp_coins += opp_r
            if r.error is not None:
                s.errors += 1
            elif self_r > opp_r:
                s.wins += 1
            elif self_r == opp_r:
                s.draws += 1
            else:
                s.losses += 1
    return stats


def print_results_table(stats: dict[str, BotStats]) -> None:
    rows = sorted(stats.items(), key=lambda kv: kv[1].winrate, reverse=True)
    header = f"{'Bot':<16} {'Games':>5} {'W':>4} {'D':>4} {'L':>4} {'Win%':>7} {'AvgCoins':>9} {'AvgOpp':>9} {'Δ':>8} {'Err':>4}"
    sep = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for name, s in rows:
        print(
            f"{_bot_display_name(name):<16} {s.games:>5} {s.wins:>4} {s.draws:>4} {s.losses:>4} "
            f"{s.winrate:>6.1f}% {s.avg_coins:>9.0f} {s.avg_opp_coins:>9.0f} {s.avg_delta:>+8.0f} {s.errors:>4}"
        )
    print(sep)


def run_arena(
    bots: list[str],
    num_games: int = 10,
    workers: int = 4,
    save_replays: str = "none",
    base_seed: int | None = 42,
) -> list[MatchResult]:
    seeds = [random.Random(base_seed).randint(0, 2**31 - 1) for _ in range(num_games)]
    tasks = [(a, b, s, save_replays) for a, b in combinations(bots, 2) for s in seeds]
    results: list[MatchResult] = []

    print(f"Running tournament: {len(bots)} bots, {len(tasks)*2} matches...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for future in as_completed([executor.submit(_run_match_pair, t) for t in tasks]):
            r1, r2 = future.result()
            results.extend([r1, r2])

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bots", nargs="+", required=True)
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--save-replays", choices=["none", "json", "html", "both"], default="none")
    args = parser.parse_args()

    results = run_arena(args.bots, args.games, args.workers, args.save_replays, args.seed)
    stats = compute_all_stats(results, args.bots)
    print_results_table(stats)


if __name__ == "__main__":
    main()
