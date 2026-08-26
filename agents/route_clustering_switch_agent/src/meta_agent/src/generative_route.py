"""Replay-independent route genomes and a state-feedback farming controller.

Replays may be used to *initialise* :class:`GenerativeRouteGenome`, but the
agent never reads or replays a recorded action.  Every action is produced from
the current observation and a compact macro policy: phased crop capacity,
land timing, labour, liquidity, and selling reserves.

This module intentionally starts with crops.  Animal husbandry can be added as
another target/task type without changing the genome/search contract.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence


CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
ANIMAL_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
ANIMAL_STRUCTURE = {"GOOSE": "COOP", "COW": "PASTURE", "SHEEP": "PASTURE"}
ANIMAL_PRODUCT = {"GOOSE": "EGG", "COW": "MILK", "SHEEP": "WOOL"}
POLICY_RUNTIME_VERSION = "generative-controller-v2-resource-commit"
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
SEED_COST = {
    "WHEAT": 10,
    "CARROT": 20,
    "TOMATO": 50,
    "STRAWBERRY": 100,
    "MELON": 80,
}
CROP_GROWTH = {
    "WHEAT": {"max_yield_day": 4, "ongoing": False},
    "CARROT": {"max_yield_day": 3, "ongoing": False},
    "TOMATO": {"max_yield_day": 8, "ongoing": True},
    "STRAWBERRY": {"max_yield_day": 10, "ongoing": True},
    "MELON": {"max_yield_day": 12, "ongoing": False},
}
QUADRANTS = ("NW", "NE", "SW", "SE")
LAND_COSTS = (1000, 2000, 4000)
MOVES = {
    (0, -1): "NORTH",
    (0, 1): "SOUTH",
    (1, 0): "EAST",
    (-1, 0): "WEST",
}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(value, key, default)


def _as_int_tuple(values: Iterable[Any], size: int) -> tuple[int, ...]:
    result = tuple(int(value) for value in values)
    if len(result) != size:
        raise ValueError(f"expected {size} values, got {len(result)}")
    return result


def _canonical_id(payload: Mapping[str, Any], prefix: str = "GRG1") -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return f"{prefix}-{hashlib.sha256(raw).hexdigest()[:16]}"


@dataclass(frozen=True)
class GenerativeRouteGenome:
    """Compact macro policy used to create actions without a replay tape.

    ``crop_counts`` contains three cumulative five-crop capacity vectors.  The
    corresponding activation days are in ``phase_days``.  Repair guarantees
    that counts are non-decreasing, so early fields remain stable while the
    route expands.
    """

    phase_days: tuple[int, int, int]
    crop_counts: tuple[
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
        tuple[int, int, int, int, int],
    ]
    workers: tuple[int, int, int]
    land_days: tuple[int, int, int]
    cash_reserve: int = 100
    sell_reserves: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    layout_seed: int = 0
    seed_lookahead_days: int = 2
    source: str = "synthetic"
    animal_counts: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ] = ((0, 0, 0), (0, 0, 0), (0, 0, 0))
    feed_buffer_days: int = 2

    def __post_init__(self) -> None:
        if len(self.phase_days) != 3 or self.phase_days[0] != 0:
            raise ValueError("phase_days must contain three days beginning at zero")
        if len(self.crop_counts) != 3:
            raise ValueError("crop_counts must contain three phases")
        if any(len(row) != len(CROPS) for row in self.crop_counts):
            raise ValueError("each crop_counts phase must contain five crop counts")
        if len(self.workers) != 3 or len(self.land_days) != 3:
            raise ValueError("workers and land_days must contain three values")
        if len(self.sell_reserves) != len(CROPS):
            raise ValueError("sell_reserves must contain five values")
        if len(self.animal_counts) != 3 or any(
            len(row) != len(ANIMALS) for row in self.animal_counts
        ):
            raise ValueError("animal_counts must contain three three-animal phases")

    @property
    def genome_id(self) -> str:
        # Provenance is metadata, not behaviour: identical policies imported
        # from different replays must share one evaluation/cache identity.
        payload = self.to_dict(include_id=False)
        payload.pop("source", None)
        return _canonical_id(payload)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        has_animals = any(sum(row) for row in self.animal_counts)
        value = {
            "schema": "generative-route-genome-v2" if has_animals else "generative-route-genome-v1",
            "phase_days": list(self.phase_days),
            "crop_counts": [list(row) for row in self.crop_counts],
            "workers": list(self.workers),
            "land_days": list(self.land_days),
            "cash_reserve": self.cash_reserve,
            "sell_reserves": list(self.sell_reserves),
            "layout_seed": self.layout_seed,
            "seed_lookahead_days": self.seed_lookahead_days,
            "source": self.source,
        }
        if has_animals:
            value["animal_counts"] = [list(row) for row in self.animal_counts]
            value["feed_buffer_days"] = self.feed_buffer_days
        if include_id:
            value["genome_id"] = self.genome_id
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GenerativeRouteGenome":
        rows = tuple(_as_int_tuple(row, len(CROPS)) for row in value["crop_counts"])
        if len(rows) != 3:
            raise ValueError("crop_counts must contain three phases")
        animal_rows = tuple(
            _as_int_tuple(row, len(ANIMALS))
            for row in value.get("animal_counts", ((0, 0, 0),) * 3)
        )
        if len(animal_rows) != 3:
            raise ValueError("animal_counts must contain three phases")
        return cls(
            phase_days=_as_int_tuple(value["phase_days"], 3),
            crop_counts=rows,  # type: ignore[arg-type]
            workers=_as_int_tuple(value["workers"], 3),
            land_days=_as_int_tuple(value["land_days"], 3),
            cash_reserve=int(value.get("cash_reserve", 100)),
            sell_reserves=_as_int_tuple(
                value.get("sell_reserves", (0, 0, 0, 0, 0)), len(CROPS)
            ),
            layout_seed=int(value.get("layout_seed", 0)),
            seed_lookahead_days=int(value.get("seed_lookahead_days", 2)),
            source=str(value.get("source", "loaded")),
            animal_counts=animal_rows,  # type: ignore[arg-type]
            feed_buffer_days=int(value.get("feed_buffer_days", 2)),
        )

    def phase_index(self, day: int) -> int:
        return max(index for index, start in enumerate(self.phase_days) if day >= start)

    def active_counts(self, day: int) -> tuple[int, int, int, int, int]:
        return self.crop_counts[self.phase_index(day)]

    def desired_workers(self, day: int) -> int:
        return self.workers[self.phase_index(day)]

    def active_animal_counts(self, day: int) -> tuple[int, int, int]:
        return self.animal_counts[self.phase_index(day)]

    def descriptor(self) -> tuple[str, int, int, int]:
        """A coarse quality-diversity niche, independent of simulator reward."""

        final = self.crop_counts[-1]
        final_animals = self.animal_counts[-1]
        choices = [(final[index], CROPS[index]) for index in range(len(CROPS))]
        choices.extend(
            (final_animals[index], ANIMALS[index]) for index in range(len(ANIMALS))
        )
        dominant = max(choices)[1]
        total = sum(final) + sum(final_animals)
        field_bin = min(5, total // 8)
        labour_bin = min(4, max(self.workers) // 2)
        land_bin = min(3, max(0, math.ceil(max(0, total - 24) / 24)))
        return dominant, field_bin, labour_bin, land_bin


def repair_genome(genome: GenerativeRouteGenome) -> GenerativeRouteGenome:
    """Project arbitrary offspring back into the feasible search space."""

    middle = min(18, max(2, int(genome.phase_days[1])))
    late = min(27, max(middle + 1, int(genome.phase_days[2])))
    rows: list[tuple[int, int, int, int, int]] = []
    previous = [0] * len(CROPS)
    for raw_row in genome.crop_counts:
        row = []
        for index, value in enumerate(raw_row):
            repaired = min(19, max(previous[index], int(value)))
            row.append(repaired)
        # Keep the total under the 96 non-shed layout slots.
        overflow = max(0, sum(row) - 80)
        for index in reversed(range(len(row))):
            reducible = max(0, row[index] - previous[index])
            take = min(reducible, overflow)
            row[index] -= take
            overflow -= take
        previous = row
        rows.append(tuple(row))  # type: ignore[arg-type]
    animal_rows: list[tuple[int, int, int]] = []
    previous_animals = [0] * len(ANIMALS)
    for phase, raw_row in enumerate(genome.animal_counts):
        row = [
            min(16, max(previous_animals[index], int(value)))
            for index, value in enumerate(raw_row)
        ]
        overflow = max(0, sum(rows[phase]) + sum(row) - 80)
        for index in reversed(range(len(row))):
            reducible = max(0, row[index] - previous_animals[index])
            take = min(reducible, overflow)
            row[index] -= take
            overflow -= take
        if overflow:
            crop_row = list(rows[phase])
            previous_crops = list(rows[phase - 1]) if phase else [0] * len(CROPS)
            for index in reversed(range(len(crop_row))):
                reducible = max(0, crop_row[index] - previous_crops[index])
                take = min(reducible, overflow)
                crop_row[index] -= take
                overflow -= take
            rows[phase] = tuple(crop_row)  # type: ignore[assignment]
        previous_animals = row
        animal_rows.append(tuple(row))  # type: ignore[arg-type]
    if sum(rows[-1]) + sum(animal_rows[-1]) == 0:
        rows[0] = (1, *rows[0][1:])
        rows[1] = tuple(max(a, b) for a, b in zip(rows[1], rows[0]))  # type: ignore[assignment]
        rows[2] = tuple(max(a, b) for a, b in zip(rows[2], rows[1]))  # type: ignore[assignment]
    workers = tuple(min(12, max(0, int(value))) for value in genome.workers)
    land_days = []
    previous_day = 0
    for value in genome.land_days:
        day = min(29, max(previous_day, int(value)))
        land_days.append(day)
        previous_day = day
    return replace(
        genome,
        phase_days=(0, middle, late),
        crop_counts=tuple(rows),  # type: ignore[arg-type]
        animal_counts=tuple(animal_rows),  # type: ignore[arg-type]
        workers=workers,  # type: ignore[arg-type]
        land_days=tuple(land_days),  # type: ignore[arg-type]
        cash_reserve=min(4000, max(0, int(genome.cash_reserve))),
        sell_reserves=tuple(
            min(50, max(0, int(value))) for value in genome.sell_reserves
        ),  # type: ignore[arg-type]
        layout_seed=int(genome.layout_seed) % 1_000_003,
        seed_lookahead_days=min(8, max(0, int(genome.seed_lookahead_days))),
        feed_buffer_days=min(5, max(1, int(genome.feed_buffer_days))),
    )


def default_seed_genomes() -> list[GenerativeRouteGenome]:
    """Diverse route hypotheses; these are priors, not scripted solutions."""

    raw = [
        # Fast, cheap one-shot turnover.
        GenerativeRouteGenome(
            (0, 7, 16), ((4, 4, 0, 0, 0), (8, 8, 0, 0, 0), (12, 12, 0, 0, 0)),
            (2, 4, 6), (8, 18, 25), 120, (0, 0, 0, 0, 0), 11, 2, "prior:fast",
        ),
        # Persistent crops reduce replant travel.
        GenerativeRouteGenome(
            (0, 8, 17), ((2, 2, 2, 0, 0), (2, 2, 7, 3, 0), (2, 2, 10, 6, 0)),
            (2, 4, 6), (10, 19, 26), 250, (0, 0, 0, 0, 0), 23, 3, "prior:persistent",
        ),
        # High-value late crop mix.
        GenerativeRouteGenome(
            (0, 6, 14), ((3, 2, 0, 0, 1), (4, 3, 2, 1, 5), (4, 3, 3, 3, 10)),
            (2, 5, 7), (6, 15, 22), 300, (0, 0, 0, 0, 0), 37, 4, "prior:value",
        ),
        # Small route gives the archive a low-complexity control.
        GenerativeRouteGenome(
            (0, 9, 19), ((2, 2, 0, 0, 0), (3, 3, 1, 0, 0), (4, 4, 2, 0, 0)),
            (1, 2, 3), (14, 24, 29), 500, (0, 0, 0, 0, 0), 51, 1, "prior:small",
        ),
        # Early goose loop with self-produced wheat feed.
        GenerativeRouteGenome(
            (0, 4, 12), ((4, 1, 0, 0, 0), (6, 1, 0, 0, 0), (8, 1, 0, 0, 0)),
            (2, 4, 6), (10, 20, 27), 350, (1, 0, 0, 0, 0), 71, 3, "prior:goose",
            animal_counts=((0, 0, 0), (2, 0, 0), (5, 0, 0)),
            feed_buffer_days=2,
        ),
        # Mixed ruminants test higher-value milk and wool production.
        GenerativeRouteGenome(
            (0, 5, 13), ((4, 1, 0, 0, 0), (8, 1, 0, 0, 0), (10, 1, 0, 0, 0)),
            (2, 5, 8), (9, 18, 26), 500, (2, 0, 0, 0, 0), 83, 3, "prior:ruminant",
            animal_counts=((0, 0, 0), (0, 1, 1), (0, 3, 3)),
            feed_buffer_days=2,
        ),
    ]
    return [repair_genome(genome) for genome in raw]


def genome_from_route_summary(value: Mapping[str, Any]) -> GenerativeRouteGenome:
    """Initialise a macro genome from replay-derived anchors, never its actions."""

    payload = value.get("genetic_payload") or value
    anchors = sorted(
        list(payload.get("anchor_targets", []) or []), key=lambda row: int(row.get("step", 0))
    )
    sampled: list[list[int]] = []
    sampled_animals: list[list[int]] = []
    if anchors:
        first_counts = anchors[0].get("counts", {}) or {}
        final_counts = anchors[-1].get("counts", {}) or {}
        first = [int(first_counts.get(crop, 0) or 0) for crop in CROPS]
        final = [max(first[index], int(final_counts.get(crop, 0) or 0)) for index, crop in enumerate(CROPS)]
        first_animals = [int(first_counts.get(animal, 0) or 0) for animal in ANIMALS]
        final_animals = [
            max(first_animals[index], int(final_counts.get(animal, 0) or 0))
            for index, animal in enumerate(ANIMALS)
        ]
        # The first recorded anchor is normally day seven, not the opening
        # state.  Infer a conservative setup tranche instead of pretending the
        # entire day-seven layout existed on turn zero.
        sampled = [
            [round(value / 3) for value in first],
            first,
            final,
        ]
        sampled_animals = [
            [round(value / 3) for value in first_animals],
            first_animals,
            final_animals,
        ]
        first_anchor_day = max(2, min(14, int(anchors[0].get("step", 168) or 168) // 24))
        middle_anchor = anchors[len(anchors) // 2]
        final_phase_day = max(
            first_anchor_day + 1,
            min(25, int(middle_anchor.get("step", 432) or 432) // 24),
        )
        phase_days = (0, first_anchor_day, final_phase_day)
    else:
        sampled = [[1, 0, 0, 0, 0] for _ in range(3)]
        sampled_animals = [[0, 0, 0] for _ in range(3)]
        phase_days = (0, 9, 18)
    total = sum(sampled[-1]) + sum(sampled_animals[-1])
    peak_workers = min(10, max(1, math.ceil(max(1, total) / 5)))
    source_id = str(value.get("genome_id") or value.get("id") or "replay-summary")
    seed = int(hashlib.sha256(source_id.encode("utf-8")).hexdigest()[:8], 16)
    land_days = [29, 29, 29]
    for anchor in anchors:
        anchor_day = int(anchor.get("step", 0) or 0) // 24
        seen_quadrants = {
            _quadrant(int(row["x"]), int(row["y"]), 10)
            for row in anchor.get("placements", []) or []
            if "x" in row and "y" in row
        }
        for index, quadrant in enumerate(QUADRANTS[1:]):
            if quadrant in seen_quadrants:
                land_days[index] = min(land_days[index], max(0, anchor_day - 2))
    genome = GenerativeRouteGenome(
        phase_days=phase_days,
        crop_counts=tuple(tuple(row) for row in sampled),  # type: ignore[arg-type]
        workers=(max(1, peak_workers // 2), max(1, peak_workers * 3 // 4), peak_workers),
        land_days=tuple(land_days),  # type: ignore[arg-type]
        cash_reserve=200,
        sell_reserves=(0, 0, 0, 0, 0),
        layout_seed=seed,
        seed_lookahead_days=3,
        source=f"replay-macro:{source_id}",
        animal_counts=tuple(tuple(row) for row in sampled_animals),  # type: ignore[arg-type]
        feed_buffer_days=2,
    )
    return repair_genome(genome)


@dataclass(frozen=True)
class FieldTarget:
    x: int
    y: int
    crop: str
    activation_day: int
    quadrant: str


@dataclass(frozen=True)
class AnimalTarget:
    x: int
    y: int
    animal: str
    activation_day: int
    quadrant: str


@dataclass(frozen=True)
class FieldTask:
    x: int
    y: int
    operation: str
    item: str
    priority: int
    required_item: str | None = None
    quantity: int = 1


def _quadrant(x: int, y: int, board_size: int) -> str:
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _shed_tiles(board_size: int) -> tuple[tuple[int, int], ...]:
    half = board_size // 2
    return ((half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half))


def _layout_cells(board_size: int, layout_seed: int) -> list[tuple[int, int]]:
    """Near-shed cells first in land purchase order, with reproducible variety."""

    shed = set(_shed_tiles(board_size))
    result = []
    rng = random.Random(layout_seed)
    for quadrant in QUADRANTS:
        cells = [
            (x, y)
            for y in range(board_size)
            for x in range(board_size)
            if _quadrant(x, y, board_size) == quadrant and (x, y) not in shed
        ]
        access = next(xy for xy in _shed_tiles(board_size) if _quadrant(*xy, board_size) == quadrant)
        # Random tie breaker keeps locality while making layout_seed meaningful.
        tie = {xy: rng.random() for xy in cells}
        cells.sort(key=lambda xy: (abs(xy[0] - access[0]) + abs(xy[1] - access[1]), tie[xy]))
        result.extend(cells)
    return result


def materialize_targets(
    genome: GenerativeRouteGenome, board_size: int = 10
) -> tuple[FieldTarget, ...]:
    """Turn phased capacities into stable crop/coordinate assignments."""

    cells = iter(_layout_cells(board_size, genome.layout_seed))
    targets: list[FieldTarget] = []
    final = genome.crop_counts[-1]
    for crop_index, crop in enumerate(CROPS):
        for ordinal in range(final[crop_index]):
            try:
                x, y = next(cells)
            except StopIteration:
                return tuple(targets)
            phase = next(
                index
                for index, counts in enumerate(genome.crop_counts)
                if counts[crop_index] > ordinal
            )
            targets.append(FieldTarget(
                x=x,
                y=y,
                crop=crop,
                activation_day=genome.phase_days[phase],
                quadrant=_quadrant(x, y, board_size),
            ))
    return tuple(targets)


def materialize_animal_targets(
    genome: GenerativeRouteGenome, board_size: int = 10
) -> tuple[AnimalTarget, ...]:
    """Allocate animal structures after crop cells in the same stable layout."""

    cells = iter(_layout_cells(board_size, genome.layout_seed))
    for _ in range(sum(genome.crop_counts[-1])):
        next(cells, None)
    targets: list[AnimalTarget] = []
    final = genome.animal_counts[-1]
    for animal_index, animal in enumerate(ANIMALS):
        for ordinal in range(final[animal_index]):
            xy = next(cells, None)
            if xy is None:
                return tuple(targets)
            x, y = xy
            phase = next(
                index
                for index, counts in enumerate(genome.animal_counts)
                if counts[animal_index] > ordinal
            )
            targets.append(AnimalTarget(
                x=x,
                y=y,
                animal=animal,
                activation_day=genome.phase_days[phase],
                quadrant=_quadrant(x, y, board_size),
            ))
    return tuple(targets)


class GenerativeRouteAgent:
    """State-feedback controller for :class:`GenerativeRouteGenome`."""

    def __init__(self, genome: GenerativeRouteGenome):
        self.genome = repair_genome(genome)
        self._targets_by_size: dict[int, tuple[FieldTarget, ...]] = {}
        self._animal_targets_by_size: dict[int, tuple[AnimalTarget, ...]] = {}
        self.action_counts: dict[str, int] = {}

    def _count(self, action: Sequence[Any]) -> None:
        operation = str(action[0]) if action else "PASS"
        self.action_counts[operation] = self.action_counts.get(operation, 0) + 1

    def _targets(self, board_size: int) -> tuple[FieldTarget, ...]:
        if board_size not in self._targets_by_size:
            self._targets_by_size[board_size] = materialize_targets(self.genome, board_size)
        return self._targets_by_size[board_size]

    def _animal_targets(self, board_size: int) -> tuple[AnimalTarget, ...]:
        if board_size not in self._animal_targets_by_size:
            self._animal_targets_by_size[board_size] = materialize_animal_targets(
                self.genome, board_size
            )
        return self._animal_targets_by_size[board_size]

    def _all_targets(self, board_size: int) -> tuple[FieldTarget | AnimalTarget, ...]:
        return (*self._targets(board_size), *self._animal_targets(board_size))

    @staticmethod
    def _tile_task(tile: Any, target: FieldTarget, day: int) -> FieldTask | None:
        if tile == "LOCKED":
            return None
        if tile is None:
            return FieldTask(target.x, target.y, "PLANT", target.crop, 40)
        if not isinstance(tile, Mapping):
            return None
        if tile.get("kind") == "WEED":
            return FieldTask(target.x, target.y, "DIG", target.crop, 30)
        if tile.get("kind") != "PLANT" or tile.get("crop") != target.crop:
            planted_day = int(tile.get("planted_day", day))
            crop = str(tile.get("crop") or target.crop)
            mature = (
                CROP_GROWTH.get(crop, {}).get("ongoing", False)
                or day - planted_day >= int(CROP_GROWTH.get(crop, {}).get("max_yield_day", 0))
            )
            if mature and int(tile.get("yield_units", 0) or 0) > 0:
                return FieldTask(target.x, target.y, "HARVEST", target.crop, 5)
            return FieldTask(target.x, target.y, "DIG", target.crop, 25)
        growth = CROP_GROWTH[target.crop]
        planted_day = int(tile.get("planted_day", day))
        mature = bool(growth["ongoing"]) or day - planted_day >= int(growth["max_yield_day"])
        if mature and int(tile.get("yield_units", 0) or 0) > 0:
            return FieldTask(target.x, target.y, "HARVEST", target.crop, 0)
        if not bool(tile.get("watered_today", False)):
            urgency = 2 if int(tile.get("consecutive_unwatered", 0) or 0) else 10
            return FieldTask(target.x, target.y, "WATER", target.crop, urgency)
        return None

    @staticmethod
    def _animal_tile_task(tile: Any, target: AnimalTarget) -> FieldTask | None:
        structure = ANIMAL_STRUCTURE[target.animal]
        if tile == "LOCKED":
            return None
        if tile is None:
            operation = "BUILD_COOP" if structure == "COOP" else "BUILD_PASTURE"
            return FieldTask(target.x, target.y, operation, target.animal, 25)
        if not isinstance(tile, Mapping):
            return None
        if tile.get("kind") == "WEED":
            return FieldTask(target.x, target.y, "DIG", target.animal, 24)
        if tile.get("animal") == target.animal:
            if int(tile.get("yield_units", 0) or 0) > 0:
                return FieldTask(target.x, target.y, "HARVEST", target.animal, 0)
            if not bool(tile.get("fed_today", False)):
                return FieldTask(
                    target.x, target.y, "FEED", target.animal, 1,
                    required_item="WHEAT",
                )
            if not bool(tile.get("cared_today", False)):
                return FieldTask(target.x, target.y, "CARE", target.animal, 4)
            if bool(tile.get("fertilizer_available", False)):
                return FieldTask(
                    target.x, target.y, "COLLECT_FERTILIZER", target.animal, 12
                )
            return None
        if tile.get("kind") == structure and "animal" not in tile:
            return FieldTask(
                target.x, target.y, "PLACE", target.animal, 8,
                required_item=target.animal,
            )
        # A mismatched crop/empty structure is removable. A placed wrong animal
        # cannot be dug out by the environment and is left for audit instead.
        if "animal" not in tile:
            return FieldTask(target.x, target.y, "DIG", target.animal, 23)
        return None

    @staticmethod
    def _move(position: Sequence[Any], target: tuple[int, int]) -> list[str]:
        x, y = int(position[0]), int(position[1])
        dx, dy = target[0] - x, target[1] - y
        # Move along the larger remaining axis to reduce long zig-zags.
        if abs(dx) >= abs(dy) and dx:
            return [MOVES[(1 if dx > 0 else -1, 0)]]
        if dy:
            return [MOVES[(0, 1 if dy > 0 else -1)]]
        return ["PASS"]

    def _unit_actions(
        self,
        farm: Mapping[str, Any],
        private: Mapping[str, Any],
        day: int,
        hour: int,
    ) -> tuple[list[str], list[list[str]]]:
        tiles = list(farm.get("tiles", []) or [])
        board_size = len(tiles)
        unlocked = set(farm.get("unlocked_quadrants", []) or [])
        active = [
            target for target in self._targets(board_size)
            if target.activation_day <= day and target.quadrant in unlocked
        ]
        tasks = []
        for target in active:
            try:
                task = self._tile_task(tiles[target.y][target.x], target, day)
            except (IndexError, TypeError):
                task = None
            if task is not None:
                tasks.append(task)
        active_animals = [
            target for target in self._animal_targets(board_size)
            if target.activation_day <= day and target.quadrant in unlocked
        ]
        for target in active_animals:
            try:
                task = self._animal_tile_task(tiles[target.y][target.x], target)
            except (IndexError, TypeError):
                task = None
            if task is not None:
                tasks.append(task)

        inventories = list(private.get("inventories", []) or [])
        carried: Counter[str]
        # Avoid importing a second collection type into the public contract.
        carried = Counter()
        for inventory in inventories:
            carried.update({str(item): int(quantity) for item, quantity in inventory.items()})
        shed = private.get("shed", {}) or {}
        required = Counter(
            task.required_item for task in tasks if task.required_item is not None
        )
        pickup_xy = _shed_tiles(board_size)[0]
        for item, quantity in required.items():
            shortfall = max(0, quantity - carried[item])
            available = int(shed.get(item, 0) or 0)
            if shortfall and available:
                tasks.append(FieldTask(
                    pickup_xy[0], pickup_xy[1], "PICKUP", item, -1,
                    quantity=min(shortfall, available, 4),
                ))
        tasks.sort(key=lambda task: (task.priority, task.y, task.x))

        actors = [farm.get("farmer"), *(farm.get("hands", []) or [])]
        remaining = list(tasks)
        orders: list[list[str]] = []
        for actor_index, position in enumerate(actors):
            if not isinstance(position, Sequence) or len(position) < 2:
                orders.append(["PASS"])
                continue
            inventory = inventories[actor_index] if actor_index < len(inventories) else {}
            # On the last day, deliver harvested goods instead of relying on a
            # day transition that never occurs after the terminal turn.
            if day >= 29 and inventory:
                shed_target = min(
                    _shed_tiles(board_size),
                    key=lambda xy: abs(xy[0] - int(position[0])) + abs(xy[1] - int(position[1])),
                )
                if tuple(position[:2]) == shed_target:
                    orders.append(["DROP"])
                else:
                    orders.append(self._move(position, shed_target))
                continue
            eligible = [
                task for task in remaining
                if task.required_item is None
                or int(inventory.get(task.required_item, 0) or 0) > 0
            ]
            if not eligible:
                orders.append(["PASS"])
                continue
            committed = [
                task for task in eligible
                if task.required_item is not None
                and int(inventory.get(task.required_item, 0) or 0) > 0
            ]
            assignment_pool = committed or eligible
            # Priority dominates, distance breaks ties inside a narrow urgency band.
            best = min(
                assignment_pool,
                key=lambda task: (
                    task.priority,
                    abs(task.x - int(position[0])) + abs(task.y - int(position[1])),
                ),
            )
            remaining.remove(best)
            if (int(position[0]), int(position[1])) != (best.x, best.y):
                orders.append(self._move(position, (best.x, best.y)))
            elif best.operation == "PLANT":
                seeds = private.get("seeds", {}) or {}
                orders.append(["PLANT", best.item] if int(seeds.get(best.item, 0) or 0) > 0 else ["PASS"])
            elif best.operation == "PICKUP":
                orders.append(["PICKUP", best.item, best.quantity])
            elif best.operation == "PLACE":
                orders.append(["PLACE", best.item])
            else:
                orders.append([best.operation])
        farmer = orders[0] if orders else ["PASS"]
        return farmer, orders[1:]

    @staticmethod
    def _hire_cost(already: int) -> int:
        a, b = 1, 1
        for _ in range(already):
            a, b = b, a + b
        return a

    def _market_actions(
        self,
        observation: Any,
        farm: Mapping[str, Any],
        private: Mapping[str, Any],
        day: int,
        hour: int,
    ) -> list[list[Any]]:
        market_state = _get(observation, "market", {}) or {}
        prices = market_state.get("prices", {}) or {}
        shed = private.get("shed", {}) or {}
        orders: list[list[Any]] = []
        expected_cash = float(farm.get("money", 0) or 0)

        tiles = list(farm.get("tiles", []) or [])
        board_size = len(tiles)
        placed_animals = Counter()
        for row in tiles:
            for tile in row:
                if isinstance(tile, Mapping) and tile.get("animal") in ANIMALS:
                    placed_animals[str(tile["animal"])] += 1

        # Realise output continuously; reserves are themselves evolvable.
        for index, crop in enumerate(CROPS):
            reserve = self.genome.sell_reserves[index]
            if crop == "WHEAT":
                reserve = max(
                    reserve,
                    sum(placed_animals.values()) * self.genome.feed_buffer_days,
                )
            quantity = max(0, int(shed.get(crop, 0) or 0) - reserve)
            if quantity:
                orders.append(["SELL", crop, quantity])
                expected_cash += quantity * int(prices.get(crop, 0) or 0)
        for product in ("EGG", "MILK", "WOOL", "FERTILIZER"):
            quantity = max(0, int(shed.get(product, 0) or 0))
            if quantity:
                orders.append(["SELL", product, quantity])
                expected_cash += quantity * int(prices.get(product, 0) or 0)

        budget = max(0.0, expected_cash - self.genome.cash_reserve)
        unlocked = list(farm.get("unlocked_quadrants", []) or [])
        next_land_index = len(unlocked) - 1
        if 0 <= next_land_index < len(LAND_COSTS):
            cost = LAND_COSTS[next_land_index]
            next_quadrant = QUADRANTS[len(unlocked)]
            needs_next_land = any(
                target.quadrant == next_quadrant
                and target.activation_day <= day + self.genome.seed_lookahead_days
                for target in self._all_targets(board_size)
            )
            if (
                needs_next_land
                and day >= self.genome.land_days[next_land_index]
                and budget >= cost
            ):
                orders.append(["BUY_LAND"])
                budget -= cost

        # Hire early enough for workers to be available for most of the day.
        desired = self.genome.desired_workers(day)
        already = int(farm.get("hires_today", 0) or 0)
        current = len(farm.get("hands", []) or [])
        if hour <= 4 and current < desired:
            for offset in range(desired - current):
                cost = self._hire_cost(already + offset)
                if budget < cost or len(orders) >= 10:
                    break
                orders.append(["HIRE"])
                budget -= cost

        # Buy seeds for empty active fields plus fields entering the lookahead.
        seed_stock = private.get("seeds", {}) or {}
        needed = {crop: 0 for crop in CROPS}
        horizon_day = day + self.genome.seed_lookahead_days
        unlocked_set = set(unlocked)
        for target in self._targets(board_size):
            if target.activation_day > horizon_day or target.quadrant not in unlocked_set:
                continue
            try:
                tile = tiles[target.y][target.x]
            except (IndexError, TypeError):
                continue
            if tile is None or (isinstance(tile, Mapping) and tile.get("kind") == "WEED"):
                needed[target.crop] += 1
        for crop in CROPS:
            shortfall = max(0, needed[crop] - int(seed_stock.get(crop, 0) or 0))
            affordable = int(budget // SEED_COST[crop])
            quantity = min(shortfall, affordable)
            if quantity and len(orders) < 10:
                orders.append(["BUY_SEED", crop, quantity])
                budget -= quantity * SEED_COST[crop]

        inventories = list(private.get("inventories", []) or [])
        carried = Counter()
        for inventory in inventories:
            carried.update({str(item): int(quantity) for item, quantity in inventory.items()})
        desired_animals = Counter()
        for target in self._animal_targets(board_size):
            if (
                target.activation_day <= horizon_day
                and target.quadrant in unlocked_set
            ):
                desired_animals[target.animal] += 1
        for animal in ANIMALS:
            owned = (
                placed_animals[animal]
                + int(shed.get(animal, 0) or 0)
                + carried[animal]
            )
            shortfall = max(0, desired_animals[animal] - owned)
            affordable = int(budget // ANIMAL_COST[animal])
            quantity = min(shortfall, affordable)
            if quantity and len(orders) < 10:
                orders.append(["BUY_ANIMAL", animal, quantity])
                budget -= quantity * ANIMAL_COST[animal]

        feed_target = sum(placed_animals.values()) * self.genome.feed_buffer_days
        feed_stock = int(shed.get("WHEAT", 0) or 0) + carried["WHEAT"]
        feed_shortfall = max(0, feed_target - feed_stock)
        wheat_price = max(1, int(prices.get("WHEAT", 25) or 25))
        feed_quantity = min(feed_shortfall, int(budget // wheat_price))
        if feed_quantity and len(orders) < 10:
            orders.append(["BUY_PRODUCT", "WHEAT", feed_quantity])
            budget -= feed_quantity * wheat_price
        return orders[:10]

    def __call__(
        self, observation: Any, configuration: Any | None = None
    ) -> dict[str, Any]:
        # Kaggle's wrapper supplies configuration to callable objects even
        # though this controller currently needs only the observation.
        del configuration
        farms = list(_get(observation, "farms", []) or [])
        player = int(_get(observation, "player", 0) or 0)
        private = _get(observation, "private", {}) or {}
        if player >= len(farms):
            return {"farmer": ["PASS"], "hands": [], "market": []}
        farm = farms[player]
        day = int(_get(observation, "day", 0) or 0)
        hour = int(_get(observation, "hour", 0) or 0)
        farmer, hands = self._unit_actions(farm, private, day, hour)
        market = self._market_actions(observation, farm, private, day, hour)
        for action in [farmer, *hands]:
            self._count(action)
        for action in market:
            self._count(action)
        return {"farmer": farmer, "hands": hands, "market": market}
