"""Adaptive quality-diversity search for replay-independent route genomes."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .generative_route import (
    ANIMALS,
    CROPS,
    GenerativeRouteGenome,
    default_seed_genomes,
    repair_genome,
)


MUTATION_OPERATORS = (
    "crop_capacity",
    "crop_mix",
    "animal_capacity",
    "animal_mix",
    "phase_timing",
    "labour",
    "land_timing",
    "capital",
    "layout",
    "seed_lookahead",
    "feed_buffer",
)


def applicable_mutation_operators(
    genome: GenerativeRouteGenome,
) -> tuple[str, ...]:
    """Exclude operators that cannot alter this route's executed phenotype."""

    operators = list(MUTATION_OPERATORS)
    if sum(genome.crop_counts[-1]) + sum(genome.animal_counts[-1]) <= 24:
        operators.remove("land_timing")
    if (
        genome.crop_counts[0] == genome.crop_counts[1] == genome.crop_counts[2]
        and genome.animal_counts[0] == genome.animal_counts[1] == genome.animal_counts[2]
    ):
        operators.remove("phase_timing")
    if sum(genome.animal_counts[-1]) == 0:
        operators.remove("animal_mix")
        operators.remove("feed_buffer")
    return tuple(operators)


def mutate_genome(
    genome: GenerativeRouteGenome,
    operator: str,
    rng: random.Random,
) -> GenerativeRouteGenome:
    """Apply one high-level variation operator and repair feasibility."""

    if operator not in MUTATION_OPERATORS:
        raise ValueError(f"unknown mutation operator: {operator}")
    child = genome
    if operator == "crop_capacity":
        phase = rng.randrange(3)
        crop = rng.randrange(len(CROPS))
        rows = [list(row) for row in genome.crop_counts]
        rows[phase][crop] += rng.choice((-4, -2, -1, 1, 2, 4))
        # A capacity decision normally persists into later phases.
        if rng.random() < 0.7:
            for later in range(phase + 1, 3):
                rows[later][crop] += rows[phase][crop] - genome.crop_counts[phase][crop]
        child = replace(child, crop_counts=tuple(tuple(row) for row in rows))  # type: ignore[arg-type]
    elif operator == "crop_mix":
        phase = rng.randrange(3)
        source, target = rng.sample(range(len(CROPS)), 2)
        rows = [list(row) for row in genome.crop_counts]
        amount = rng.choice((1, 1, 2, 3))
        for current in range(phase, 3):
            moved = min(amount, rows[current][source])
            rows[current][source] -= moved
            rows[current][target] += moved
        child = replace(child, crop_counts=tuple(tuple(row) for row in rows))  # type: ignore[arg-type]
    elif operator == "animal_capacity":
        phase = rng.randrange(3)
        animal = rng.randrange(len(ANIMALS))
        rows = [list(row) for row in genome.animal_counts]
        rows[phase][animal] += rng.choice((-2, -1, 1, 1, 2, 3))
        if rng.random() < 0.8:
            for later in range(phase + 1, 3):
                rows[later][animal] += (
                    rows[phase][animal] - genome.animal_counts[phase][animal]
                )
        child = replace(child, animal_counts=tuple(tuple(row) for row in rows))  # type: ignore[arg-type]
    elif operator == "animal_mix":
        phase = rng.randrange(3)
        source, target = rng.sample(range(len(ANIMALS)), 2)
        rows = [list(row) for row in genome.animal_counts]
        amount = rng.choice((1, 1, 2))
        for current in range(phase, 3):
            moved = min(amount, rows[current][source])
            rows[current][source] -= moved
            rows[current][target] += moved
        child = replace(child, animal_counts=tuple(tuple(row) for row in rows))  # type: ignore[arg-type]
    elif operator == "phase_timing":
        days = list(genome.phase_days)
        index = rng.choice((1, 2))
        days[index] += rng.choice((-4, -2, -1, 1, 2, 4))
        child = replace(child, phase_days=tuple(days))  # type: ignore[arg-type]
    elif operator == "labour":
        workers = list(genome.workers)
        phase = rng.randrange(3)
        workers[phase] += rng.choice((-3, -2, -1, 1, 2, 3))
        if rng.random() < 0.5:
            for later in range(phase + 1, 3):
                workers[later] += workers[phase] - genome.workers[phase]
        child = replace(child, workers=tuple(workers))  # type: ignore[arg-type]
    elif operator == "land_timing":
        days = list(genome.land_days)
        index = rng.randrange(3)
        days[index] += rng.choice((-5, -3, -1, 1, 3, 5))
        child = replace(child, land_days=tuple(days))  # type: ignore[arg-type]
    elif operator == "capital":
        child = replace(
            child,
            cash_reserve=genome.cash_reserve + rng.choice((-500, -250, -100, 100, 250, 500)),
        )
    elif operator == "layout":
        child = replace(child, layout_seed=rng.randrange(1_000_003))
    elif operator == "seed_lookahead":
        child = replace(
            child,
            seed_lookahead_days=genome.seed_lookahead_days + rng.choice((-2, -1, 1, 2)),
        )
    elif operator == "feed_buffer":
        child = replace(
            child,
            feed_buffer_days=genome.feed_buffer_days + rng.choice((-1, 1, 1, 2)),
        )
    return repair_genome(replace(child, source=f"mutation:{operator}:{genome.genome_id}"))


def crossover_genomes(
    left: GenerativeRouteGenome,
    right: GenerativeRouteGenome,
    rng: random.Random,
) -> GenerativeRouteGenome:
    """Uniform macro crossover followed by constraint repair."""

    rows = []
    animal_rows = []
    for phase in range(3):
        rows.append(tuple(
            left.crop_counts[phase][crop] if rng.random() < 0.5
            else right.crop_counts[phase][crop]
            for crop in range(len(CROPS))
        ))
        animal_rows.append(tuple(
            left.animal_counts[phase][animal] if rng.random() < 0.5
            else right.animal_counts[phase][animal]
            for animal in range(len(ANIMALS))
        ))
    child = GenerativeRouteGenome(
        phase_days=tuple(
            left.phase_days[index] if rng.random() < 0.5 else right.phase_days[index]
            for index in range(3)
        ),  # type: ignore[arg-type]
        crop_counts=tuple(rows),  # type: ignore[arg-type]
        workers=tuple(
            left.workers[index] if rng.random() < 0.5 else right.workers[index]
            for index in range(3)
        ),  # type: ignore[arg-type]
        land_days=tuple(
            left.land_days[index] if rng.random() < 0.5 else right.land_days[index]
            for index in range(3)
        ),  # type: ignore[arg-type]
        cash_reserve=rng.choice((left.cash_reserve, right.cash_reserve)),
        sell_reserves=tuple(
            left.sell_reserves[index] if rng.random() < 0.5 else right.sell_reserves[index]
            for index in range(len(CROPS))
        ),  # type: ignore[arg-type]
        layout_seed=rng.choice((left.layout_seed, right.layout_seed)),
        seed_lookahead_days=rng.choice((left.seed_lookahead_days, right.seed_lookahead_days)),
        source=f"crossover:{left.genome_id}:{right.genome_id}",
        animal_counts=tuple(animal_rows),  # type: ignore[arg-type]
        feed_buffer_days=rng.choice((left.feed_buffer_days, right.feed_buffer_days)),
    )
    return repair_genome(child)


def random_immigrant(rng: random.Random) -> GenerativeRouteGenome:
    """Create a route outside the local replay-derived neighbourhood."""

    phase_days = (0, rng.randint(3, 13), rng.randint(14, 26))
    final_total = rng.randint(4, 48)
    weights = [rng.random() ** 1.5 for _ in CROPS]
    weight_sum = sum(weights)
    final = [round(final_total * weight / weight_sum) for weight in weights]
    if sum(final) == 0:
        final[rng.randrange(len(CROPS))] = 1
    early = [rng.randint(0, max(0, value // 2)) for value in final]
    middle = [rng.randint(early[index], final[index]) for index in range(len(CROPS))]
    final_animals = [0] * len(ANIMALS)
    if rng.random() < 0.60:
        animal_total = rng.randint(1, 12)
        for _ in range(animal_total):
            final_animals[rng.randrange(len(ANIMALS))] += 1
    early_animals = [rng.randint(0, max(0, value // 3)) for value in final_animals]
    middle_animals = [
        rng.randint(early_animals[index], final_animals[index])
        for index in range(len(ANIMALS))
    ]
    genome = GenerativeRouteGenome(
        phase_days=phase_days,
        crop_counts=(tuple(early), tuple(middle), tuple(final)),  # type: ignore[arg-type]
        workers=(rng.randint(0, 4), rng.randint(2, 8), rng.randint(3, 12)),
        land_days=tuple(sorted(rng.sample(range(3, 29), 3))),  # type: ignore[arg-type]
        cash_reserve=rng.randrange(0, 1601, 100),
        sell_reserves=tuple(rng.choice((0, 0, 0, 1, 2)) for _ in CROPS),  # type: ignore[arg-type]
        layout_seed=rng.randrange(1_000_003),
        seed_lookahead_days=rng.randint(0, 6),
        source="random-immigrant",
        animal_counts=(
            tuple(early_animals), tuple(middle_animals), tuple(final_animals)
        ),  # type: ignore[arg-type]
        feed_buffer_days=rng.randint(1, 4),
    )
    return repair_genome(genome)


@dataclass
class OperatorStat:
    attempts: int = 0
    successes: int = 0
    cumulative_gain: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempts": self.attempts,
            "successes": self.successes,
            "cumulative_gain": self.cumulative_gain,
            "success_rate": self.successes / self.attempts if self.attempts else None,
        }


class AdaptiveOperatorPool:
    """UCB-style operator selection learned from archive replacements."""

    def __init__(self, stats: Mapping[str, Mapping[str, Any]] | None = None):
        self.stats = {operator: OperatorStat() for operator in MUTATION_OPERATORS}
        for operator, value in (stats or {}).items():
            if operator in self.stats:
                self.stats[operator] = OperatorStat(
                    attempts=int(value.get("attempts", 0)),
                    successes=int(value.get("successes", 0)),
                    cumulative_gain=float(value.get("cumulative_gain", 0.0)),
                )

    def scores(self) -> dict[str, float]:
        total = sum(stat.attempts for stat in self.stats.values()) + 1
        output = {}
        for operator, stat in self.stats.items():
            posterior = (stat.successes + 1.0) / (stat.attempts + 2.0)
            exploration = math.sqrt(2.0 * math.log(total + 1.0) / (stat.attempts + 1.0))
            output[operator] = posterior + 0.35 * exploration
        return output

    def choose(
        self, rng: random.Random, allowed: Sequence[str] | None = None
    ) -> str:
        scores = self.scores()
        choices = tuple(allowed or MUTATION_OPERATORS)
        if not choices:
            raise ValueError("no applicable mutation operators")
        # Softmax avoids permanently discarding an operator after a few misses.
        maximum = max(scores[operator] for operator in choices)
        weights = [math.exp(2.0 * (scores[operator] - maximum)) for operator in choices]
        return rng.choices(choices, weights=weights, k=1)[0]

    def update(self, operator: str, success: bool, gain: float = 0.0) -> None:
        stat = self.stats[operator]
        stat.attempts += 1
        stat.successes += int(success)
        stat.cumulative_gain += float(gain)

    def to_dict(self) -> dict[str, Any]:
        return {operator: stat.to_dict() for operator, stat in self.stats.items()}


@dataclass
class Elite:
    genome: GenerativeRouteGenome
    evaluation: dict[str, Any]
    generation: int

    @property
    def fitness(self) -> float:
        return float(self.evaluation["fitness"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor": list(self.genome.descriptor()),
            "genome": self.genome.to_dict(),
            "evaluation": self.evaluation,
            "generation": self.generation,
        }


class MAPElitesArchive:
    """One best route per coarse behavioural niche."""

    def __init__(self, elites: Sequence[Mapping[str, Any]] | None = None):
        self.elites: dict[tuple[str, int, int, int], Elite] = {}
        for raw in elites or []:
            genome = GenerativeRouteGenome.from_dict(raw["genome"])
            elite = Elite(genome, dict(raw["evaluation"]), int(raw.get("generation", 0)))
            self.elites[genome.descriptor()] = elite

    def consider(
        self,
        genome: GenerativeRouteGenome,
        evaluation: Mapping[str, Any],
        generation: int,
    ) -> tuple[bool, float]:
        descriptor = genome.descriptor()
        previous = self.elites.get(descriptor)
        previous_fitness = previous.fitness if previous else float("-inf")
        fitness = float(evaluation["fitness"])
        if previous is None or fitness > previous_fitness:
            self.elites[descriptor] = Elite(genome, dict(evaluation), generation)
            gain = fitness - previous_fitness if previous else 0.0
            return True, gain
        return False, fitness - previous_fitness

    def choose(self, rng: random.Random) -> Elite:
        if not self.elites:
            raise ValueError("cannot choose from an empty archive")
        return rng.choice(list(self.elites.values()))

    def best(self) -> Elite:
        if not self.elites:
            raise ValueError("archive is empty")
        return max(self.elites.values(), key=lambda elite: elite.fitness)

    def to_list(self) -> list[dict[str, Any]]:
        return [
            elite.to_dict()
            for _, elite in sorted(self.elites.items(), key=lambda item: item[0])
        ]


def seed_population(size: int, rng: random.Random) -> list[GenerativeRouteGenome]:
    seeds = default_seed_genomes()
    while len(seeds) < size:
        seeds.append(random_immigrant(rng))
    return seeds[:size]
