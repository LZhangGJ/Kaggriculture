#!/usr/bin/env python3
"""Continuously optimise replay-independent route policies with MAP-Elites."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import random
import runpy
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


AGENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_DEPS = Path(__file__).resolve().parents[4] / ".ke"
sys.path.insert(0, str(AGENT_ROOT / "src"))
if WORKSPACE_DEPS.exists():
    sys.path.insert(0, str(WORKSPACE_DEPS))

from meta_agent.src.generative_route import (  # noqa: E402
    GenerativeRouteAgent,
    GenerativeRouteGenome,
    POLICY_RUNTIME_VERSION,
    genome_from_route_summary,
)
from meta_agent.src.route_evolution import (  # noqa: E402
    AdaptiveOperatorPool,
    MAPElitesArchive,
    applicable_mutation_operators,
    crossover_genomes,
    mutate_genome,
    random_immigrant,
    seed_population,
)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _json_lines(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _field_summary(environment: Any, seat: int) -> dict[str, Any]:
    farm = _get(environment.state[seat].observation, "farms", [])[seat]
    counts: Counter[str] = Counter()
    watered = yield_ready = 0
    animal_counts: Counter[str] = Counter()
    fed = cared = 0
    for row in farm["tiles"]:
        for tile in row:
            if isinstance(tile, Mapping) and tile.get("kind") == "PLANT":
                counts[str(tile.get("crop"))] += 1
                watered += int(bool(tile.get("watered_today")))
                yield_ready += int(int(tile.get("yield_units", 0) or 0) > 0)
            if isinstance(tile, Mapping) and tile.get("animal"):
                animal_counts[str(tile["animal"])] += 1
                fed += int(bool(tile.get("fed_today")))
                cared += int(bool(tile.get("cared_today")))
    return {
        "crop_counts": dict(counts),
        "plants": sum(counts.values()),
        "watered": watered,
        "yield_ready": yield_ready,
        "unlocked_quadrants": list(farm.get("unlocked_quadrants", []) or []),
        "animal_counts": dict(animal_counts),
        "animals": sum(animal_counts.values()),
        "fed_animals": fed,
        "cared_animals": cared,
    }


def _cash_summary(environment: Any, seat: int) -> dict[str, float]:
    values = []
    for step in environment.steps:
        try:
            observation = _get(step[seat], "observation", {}) or {}
            farm = list(_get(observation, "farms", []) or [])[seat]
            values.append(float(_get(farm, "money", 0.0) or 0.0))
        except (IndexError, TypeError):
            continue
    return {
        "minimum": min(values) if values else 0.0,
        "maximum": max(values) if values else 0.0,
        "final": values[-1] if values else 0.0,
    }


class OfficialEvaluator:
    """Deterministic scenario panel around the official local simulator."""

    def __init__(
        self,
        seeds: list[int],
        both_seats: bool = True,
        opponent_agent: Path | None = None,
    ):
        from kaggle_environments.envs.kaggriculture.kaggriculture import starter_agent

        self.seeds = list(seeds)
        self.seats = (0, 1) if both_seats else (0,)
        self.opponent = starter_agent
        self.opponent_path = opponent_agent.resolve() if opponent_agent else None
        if self.opponent_path:
            digest = hashlib.sha256(self.opponent_path.read_bytes()).hexdigest()[:16]
            opponent_signature = f"file:{self.opponent_path.name}:{digest}"
        else:
            opponent_signature = "starter"
        self.signature = (
            f"{opponent_signature}|runtime={POLICY_RUNTIME_VERSION}|seeds="
            + ",".join(map(str, self.seeds))
            + "|seats=" + ",".join(map(str, self.seats))
        )

    def _fresh_opponent(self):
        if self.opponent_path is None:
            return self.opponent
        namespace = runpy.run_path(str(self.opponent_path))
        opponent = namespace.get("agent")
        if not callable(opponent):
            raise ValueError(f"submission has no callable agent: {self.opponent_path}")
        return opponent

    def evaluate(self, genome: GenerativeRouteGenome) -> dict[str, Any]:
        from kaggle_environments import make

        scenarios = []
        combined_actions: Counter[str] = Counter()
        for seed in self.seeds:
            for seat in self.seats:
                agent = GenerativeRouteAgent(genome)
                opponent = self._fresh_opponent()
                agents = [opponent, opponent]
                agents[seat] = agent
                environment = make(
                    "kaggriculture",
                    configuration={"episodeSteps": 720},
                    info={"seed": int(seed)},
                    debug=True,
                )
                environment.run(agents)
                rewards = [float(_get(state, "reward", 0.0) or 0.0) for state in environment.state]
                combined_actions.update(agent.action_counts)
                scenarios.append({
                    "seed": seed,
                    "seat": seat,
                    "reward": rewards[seat],
                    "opponent_reward": rewards[1 - seat],
                    "margin": rewards[seat] - rewards[1 - seat],
                    "completed": len(environment.steps) == 720,
                    "cash": _cash_summary(environment, seat),
                    "final_field": _field_summary(environment, seat),
                })
        rewards = [row["reward"] for row in scenarios]
        margins = [row["margin"] for row in scenarios]
        mean_reward = statistics.fmean(rewards)
        minimum_reward = min(rewards)
        mean_margin = statistics.fmean(margins)
        # Reward dominates; worst-case reward protects robustness; margin keeps
        # interaction with the shared market in the objective.
        fitness = 0.65 * mean_reward + 0.25 * minimum_reward + 0.10 * mean_margin
        if not all(row["completed"] for row in scenarios):
            fitness -= 1_000_000
        return {
            "fitness": fitness,
            "mean_reward": mean_reward,
            "minimum_reward": minimum_reward,
            "mean_margin": mean_margin,
            "completed": sum(bool(row["completed"]) for row in scenarios),
            "scenario_count": len(scenarios),
            "action_counts": dict(combined_actions),
            "scenarios": scenarios,
        }


def _bootstrap_replays(path: Path, limit: int) -> list[GenerativeRouteGenome]:
    rows = list(_json_lines(path.resolve()))

    def replay_reward(row: Mapping[str, Any]) -> float:
        own = (row.get("reward_summary", {}) or {}).get("own", {}) or {}
        return float(own.get("max", own.get("mean", 0)) or 0)

    rows.sort(
        key=lambda row: (
            -replay_reward(row),
            -int(row.get("sample_count", 1) or 1),
            str(row.get("genome_id", "")),
        )
    )
    return [genome_from_route_summary(row) for row in rows[:limit]]


def _checkpoint(
    archive: MAPElitesArchive,
    operators: AdaptiveOperatorPool,
    cache: Mapping[str, Mapping[str, Any]],
    *,
    generation: int,
    settings: Mapping[str, Any],
    history: list[Mapping[str, Any]],
    elapsed: float,
) -> dict[str, Any]:
    best = archive.best()
    return {
        "schema": "generative-route-search-v1",
        "generation": generation,
        "settings": dict(settings),
        "elapsed_seconds": elapsed,
        "archive_size": len(archive.elites),
        "best": best.to_dict(),
        "operator_stats": operators.to_dict(),
        "archive": archive.to_list(),
        "evaluation_cache": dict(cache),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--initial-population", type=int, default=8)
    parser.add_argument("--offspring", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--one-seat", action="store_true")
    parser.add_argument("--opponent-agent", type=Path)
    parser.add_argument("--random-seed", type=int, default=20260826)
    parser.add_argument("--replay-groups", type=Path)
    parser.add_argument("--replay-bootstrap-limit", type=int, default=12)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--merge-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--merge-top-k", type=int, default=0)
    parser.add_argument("--revalidate-resume", action="store_true")
    parser.add_argument("--elite-parent-rate", type=float, default=0.60)
    parser.add_argument("--best-parent-rate", type=float, default=0.35)
    parser.add_argument("--reset-operator-stats", action="store_true")
    parser.add_argument("--max-mutation-depth", type=int, default=2)
    args = parser.parse_args()

    started = time.perf_counter()
    rng = random.Random(args.random_seed)
    evaluator = OfficialEvaluator(
        args.seeds,
        both_seats=not args.one_seat,
        opponent_agent=args.opponent_agent,
    )
    archive = MAPElitesArchive()
    operators = AdaptiveOperatorPool()
    cache: dict[str, dict[str, Any]] = {}
    history: list[dict[str, Any]] = []
    start_generation = 0
    needs_revalidation = False

    if args.resume:
        state = json.loads(args.resume.resolve().read_text(encoding="utf-8"))
        archive = MAPElitesArchive(state.get("archive", []))
        operators = AdaptiveOperatorPool(state.get("operator_stats", {}))
        cache = {str(key): dict(value) for key, value in state.get("evaluation_cache", {}).items()}
        history = [dict(value) for value in state.get("history", [])]
        start_generation = int(state.get("generation", -1)) + 1
        old_settings = state.get("settings", {}) or {}
        old_signature = old_settings.get("evaluation_signature")
        if old_signature is None:
            old_seeds = old_settings.get("seeds", [17])
            old_seats = (0,) if not old_settings.get("both_seats", True) else (0, 1)
            old_signature = "starter|seeds=" + ",".join(map(str, old_seeds)) + "|seats=" + ",".join(map(str, old_seats))
        needs_revalidation = old_signature != evaluator.signature
        if needs_revalidation and not args.revalidate_resume:
            raise ValueError(
                "resume evaluation panel changed; pass --revalidate-resume to rerun all elites"
            )

    if args.reset_operator_stats:
        operators = AdaptiveOperatorPool()
        history.append({
            "generation": start_generation,
            "event": "operator_stats_reset",
            "reason": "explicit reset",
        })

    for merge_path in args.merge_checkpoint:
        merged = json.loads(merge_path.resolve().read_text(encoding="utf-8"))
        merged_settings = merged.get("settings", {}) or {}
        merged_signature = merged_settings.get("evaluation_signature")
        if merged_signature is None:
            merged_seeds = merged_settings.get("seeds", [17])
            merged_seats = (0,) if not merged_settings.get("both_seats", True) else (0, 1)
            merged_signature = "starter|seeds=" + ",".join(map(str, merged_seeds)) + "|seats=" + ",".join(map(str, merged_seats))
        merge_needs_revalidation = merged_signature != evaluator.signature
        if merge_needs_revalidation and not args.revalidate_resume:
            raise ValueError(
                f"merge checkpoint panel/runtime differs ({merged_signature}); "
                "pass --revalidate-resume to rerun merged elites"
            )
        if not merge_needs_revalidation:
            cache.update({
                str(key): dict(value)
                for key, value in merged.get("evaluation_cache", {}).items()
            })
        merged_elites = sorted(
            list(merged.get("archive", []) or []),
            key=lambda raw: float(raw["evaluation"]["fitness"]),
            reverse=True,
        )
        if args.merge_top_k > 0:
            merged_elites = merged_elites[:args.merge_top_k]
        for raw in merged_elites:
            genome = GenerativeRouteGenome.from_dict(raw["genome"])
            archive.consider(genome, raw["evaluation"], int(raw.get("generation", 0)))
        needs_revalidation = needs_revalidation or merge_needs_revalidation
        start_generation = max(start_generation, int(merged.get("generation", -1)) + 1)
        history.append({
            "generation": start_generation,
            "event": "archive_merge",
            "source": str(merge_path.resolve()),
            "archive_size": len(archive.elites),
        })

    settings = {
        "generations": args.generations,
        "initial_population": args.initial_population,
        "offspring": args.offspring,
        "seeds": args.seeds,
        "both_seats": not args.one_seat,
        "opponent_agent": str(args.opponent_agent.resolve()) if args.opponent_agent else None,
        "random_seed": args.random_seed,
        "evaluation_signature": evaluator.signature,
        "elite_parent_rate": args.elite_parent_rate,
        "best_parent_rate": args.best_parent_rate,
        "max_mutation_depth": args.max_mutation_depth,
        "replay_groups": str(args.replay_groups.resolve()) if args.replay_groups else None,
        "replay_bootstrap_limit": args.replay_bootstrap_limit,
        "merged_checkpoints": [str(path.resolve()) for path in args.merge_checkpoint],
        "merge_top_k": args.merge_top_k,
    }

    def evaluate(genome: GenerativeRouteGenome) -> dict[str, Any]:
        cache_key = f"{evaluator.signature}|{genome.genome_id}"
        # Backward compatibility for checkpoints created before signatures.
        legacy = cache.get(genome.genome_id) if not needs_revalidation else None
        if cache_key not in cache:
            cache[cache_key] = legacy or evaluator.evaluate(genome)
        return cache[cache_key]

    if needs_revalidation:
        previous_elites = list(archive.elites.values())
        archive = MAPElitesArchive()
        for elite in previous_elites:
            archive.consider(elite.genome, evaluate(elite.genome), start_generation)
        history.append({
            "generation": start_generation,
            "event": "evaluation_panel_revalidation",
            "evaluation_signature": evaluator.signature,
            "archive_size": len(archive.elites),
            "best_fitness": archive.best().fitness,
        })

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    if not archive.elites:
        population = seed_population(args.initial_population, rng)
        if args.replay_groups:
            population.extend(_bootstrap_replays(
                args.replay_groups, args.replay_bootstrap_limit
            ))
        # Behavioural IDs make de-duplication independent of replay provenance.
        population = list({genome.genome_id: genome for genome in population}.values())
        for index, genome in enumerate(population, 1):
            evaluation = evaluate(genome)
            inserted, _ = archive.consider(genome, evaluation, 0)
            print(
                f"initial {index}/{len(population)} {genome.genome_id} "
                f"fitness={evaluation['fitness']:.1f} reward={evaluation['mean_reward']:.1f} "
                f"niche={'/'.join(map(str, genome.descriptor()))} inserted={inserted}",
                flush=True,
            )
        history.append({
            "generation": 0,
            "evaluated": len(population),
            "archive_size": len(archive.elites),
            "best_fitness": archive.best().fitness,
            "best_genome_id": archive.best().genome.genome_id,
        })
        start_generation = 1
        state = _checkpoint(
            archive, operators, cache, generation=0, settings=settings,
            history=history, elapsed=time.perf_counter() - started,
        )
        args.output.resolve().write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    end_generation = start_generation + args.generations

    def choose_parent():
        draw = rng.random()
        best_rate = min(1.0, max(0.0, args.best_parent_rate))
        elite_rate = min(1.0 - best_rate, max(0.0, args.elite_parent_rate))
        if draw < best_rate:
            return archive.best()
        if draw >= best_rate + elite_rate:
            return archive.choose(rng)
        ranked = sorted(archive.elites.values(), key=lambda elite: elite.fitness, reverse=True)
        top = ranked[:max(1, math.ceil(len(ranked) / 4))]
        return rng.choice(top)

    for generation in range(start_generation, end_generation):
        candidates: list[tuple[GenerativeRouteGenome, tuple[str, ...], float]] = []
        seen = {elite.genome.genome_id for elite in archive.elites.values()}
        seen.update(
            key.rsplit("|", 1)[-1]
            for key in cache
            if key.startswith(f"{evaluator.signature}|")
        )
        attempts = 0
        while len(candidates) < args.offspring and attempts < args.offspring * 30:
            attempts += 1
            parent = choose_parent()
            parent_fitness = parent.fitness
            if rng.random() < 0.15:
                base = random_immigrant(rng)
            else:
                base = parent.genome
                if len(archive.elites) > 1 and rng.random() < 0.30:
                    mate = archive.choose(rng).genome
                    base = crossover_genomes(base, mate, rng)
            maximum_depth = max(1, int(args.max_mutation_depth))
            depths = list(range(1, maximum_depth + 1))
            depth = rng.choices(depths, weights=[1.0 / value for value in depths], k=1)[0]
            child = base
            applied = []
            for _ in range(depth):
                operator = operators.choose(rng, applicable_mutation_operators(child))
                child = mutate_genome(child, operator, rng)
                applied.append(operator)
            if child.genome_id in seen:
                continue
            seen.add(child.genome_id)
            candidates.append((child, tuple(applied), parent_fitness))

        replacements = 0
        for index, (genome, applied, parent_fitness) in enumerate(candidates, 1):
            evaluation = evaluate(genome)
            inserted, _niche_gain = archive.consider(genome, evaluation, generation)
            gain = float(evaluation["fitness"]) - parent_fitness
            # Archive insertion is a diversity event, not proof that the
            # generator improved quality.  Operator learning uses parent gain.
            for operator in applied:
                operators.update(operator, gain > 0, max(0.0, gain) / len(applied))
            replacements += int(inserted)
            print(
                f"generation {generation} {index}/{len(candidates)} op={'+'.join(applied)} "
                f"fitness={evaluation['fitness']:.1f} gain={gain:+.1f} inserted={inserted}",
                flush=True,
            )
        history.append({
            "generation": generation,
            "evaluated": len(candidates),
            "archive_replacements": replacements,
            "archive_size": len(archive.elites),
            "best_fitness": archive.best().fitness,
            "best_genome_id": archive.best().genome.genome_id,
            "operator_scores": operators.scores(),
        })
        state = _checkpoint(
            archive, operators, cache, generation=generation, settings=settings,
            history=history, elapsed=time.perf_counter() - started,
        )
        args.output.resolve().write_text(
            json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            f"checkpoint generation={generation} archive={len(archive.elites)} "
            f"best={archive.best().fitness:.1f} output={args.output.resolve()}",
            flush=True,
        )


if __name__ == "__main__":
    main()
