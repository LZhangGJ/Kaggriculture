from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_block_sequence_oracle_v1 as oracle


def test_single_mutants_change_exactly_one_gene() -> None:
    values = oracle.single_mutants("B0", ["A", "B"], length=3)
    assert len(values) == 6
    assert values[0] == ("A", "B0", "B0")
    assert values[-1] == ("B0", "B0", "B")
    assert all(sum(gene != "B0" for gene in value) == 1 for value in values)


def test_native_schedules_places_controlled_route_in_requested_seat() -> None:
    genomes = np.arange(2 * len(oracle.ANCHORS), dtype=np.int64).reshape(
        2, len(oracle.ANCHORS),
    )
    left = oracle.native_schedules(genomes, opponent=999, seat=0)
    right = oracle.native_schedules(genomes, opponent=999, seat=1)
    np.testing.assert_array_equal(left[:, :, 0], genomes)
    np.testing.assert_array_equal(right[:, :, 1], genomes)
    assert np.all(left[:, :, 1] == 999)
    assert np.all(right[:, :, 0] == 999)


class _FakeEvaluator:
    def __init__(self) -> None:
        self.cache: dict[tuple[str, ...], tuple[float, float]] = {}

    def evaluate(self, genomes):
        for raw in genomes:
            genome = tuple(raw)
            matches = int(genome[0] == "A") + int(genome[1] == "B")
            self.cache.setdefault(genome, (float(matches), 0.0))
        return {tuple(value): self.cache[tuple(value)] for value in genomes}

    def rank(self, genome):
        matches = int(self.cache[tuple(genome)][0])
        return (2 if matches == 2 else 0, float(matches))


def test_evolve_can_combine_two_individually_insufficient_blocks() -> None:
    pools = {
        anchor: ["B0", "A", "B"] for anchor in oracle.ANCHORS
    }
    single_a = ["B0"] * len(oracle.ANCHORS)
    single_b = single_a.copy()
    single_a[0] = "A"
    single_b[1] = "B"
    evaluator = _FakeEvaluator()

    best, logs = oracle.evolve(
        evaluator, "B0", ["A", "B"], pools,
        [tuple(single_a), tuple(single_b)], None,
        population=64, generations=10, restarts=1,
        elite_count=8, patience=3, random_seed=7,
    )

    assert best[0:2] == ("A", "B")
    assert evaluator.rank(best)[0] == 2
    assert logs


def test_evolve_ranks_all_fixed_seeds_when_they_exceed_population() -> None:
    pools = {anchor: ["B0", "A", "B"] for anchor in oracle.ANCHORS}
    baseline = ("B0",) * len(oracle.ANCHORS)
    left = list(baseline)
    left[0] = "A"
    right = list(baseline)
    right[1] = "B"
    combined = list(left)
    combined[1] = "B"
    fixed_seeds = [baseline, tuple(left), tuple(right), tuple(combined)]
    evaluator = _FakeEvaluator()

    best, _ = oracle.evolve(
        evaluator, "B0", ["A", "B"], pools, fixed_seeds, None,
        population=4, generations=0, restarts=1,
        elite_count=2, patience=1, random_seed=11,
    )

    assert best == tuple(combined)
    assert all(seed in evaluator.cache for seed in fixed_seeds)


class _CompositionEvaluator:
    seat = 0

    def __init__(self, target, lose_on_ablation: bool) -> None:
        self.target = tuple(target)
        self.lose_on_ablation = lose_on_ablation

    def evaluate(self, genomes):
        result = {}
        for genome in genomes:
            value = tuple(genome)
            loses = self.lose_on_ablation and value[0] == "B0"
            result[value] = (0.0, 1000.0) if loses else (400.0, 0.0)
        return result

    def rank(self, genome):
        genome = tuple(genome)
        if genome == self.target:
            return 2, 1000.0
        if self.lose_on_ablation and genome[0] == "B0":
            return 0, -1.0
        return 2, 0.0


def test_composition_requires_an_ablation_that_removes_the_win() -> None:
    genome = ["B0"] * len(oracle.ANCHORS)
    genome[0:2] = ["A", "B"]
    constants = {
        (route,) * len(oracle.ANCHORS): (0.0, 1.0)
        for route in ("A", "B")
    }

    margin_only = oracle.composition_analysis(
        _CompositionEvaluator(genome, False), tuple(genome), "B0", constants,
    )
    win_critical = oracle.composition_analysis(
        _CompositionEvaluator(genome, True), tuple(genome), "B0", constants,
    )

    assert margin_only["margin_contributions"]
    assert not margin_only["win_critical_ablations"]
    assert not margin_only["true_composition"]
    assert win_critical["win_critical_ablations"]
    assert win_critical["true_composition"]


def test_frozen_mode_cannot_enter_search(monkeypatch) -> None:
    called = []
    monkeypatch.setattr(oracle, "run_holdout", lambda args: called.append(args))
    monkeypatch.setattr(
        oracle, "run", lambda args: (_ for _ in ()).throw(
            AssertionError("search path was entered"),
        ),
    )

    assert oracle.main(["--frozen-run", "D:/does-not-need-to-exist"]) == 0
    assert len(called) == 1


def test_search_artifact_path_rejects_escape() -> None:
    root = Path("D:/frozen/search").resolve()

    assert oracle._search_artifact_path(root, "results.json") == (
        root / "results.json"
    )
    for value in ("../outside.json", "D:/outside.json", ""):
        try:
            oracle._search_artifact_path(root, value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"escape was accepted: {value}")


def test_implementation_identity_pins_python_helpers_and_native_runtime() -> None:
    identity = oracle._implementation_identity()

    assert {
        "runner", "adaptive_oracle", "continuation_oracle",
        "routed_intent", "block_mvp_v1", "native_bundle",
        "expanded_routes", "fast_package", "native_extension",
    } == set(identity)
    assert all(len(record["sha256"]) == 64 for record in identity.values())
