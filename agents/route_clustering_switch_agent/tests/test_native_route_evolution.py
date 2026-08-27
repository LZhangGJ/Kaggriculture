from __future__ import annotations

import sys
import random
import json
import tempfile
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from evolve_native_route_library import (  # noqa: E402
    _apply_evolution_genes,
    _behavior_descriptor,
    _canonical,
    _donor_atoms,
    _mutate,
    _robust_checkpoint_metrics,
    _seed_genomes,
    _select_elites,
)


def _action(farmer, hand, market):
    return {"farmer": farmer, "hands": [hand], "market": market}


def test_single_parent_mutation_never_attempts_empty_parent_switch() -> None:
    atoms = {"A": [{"operator": "reduce_hires", "start": 0, "stop": 144,
                    "remove_per_day": 1}]}
    rng = random.Random(3)
    genome = ("A", ())
    for _ in range(100):
        genome = _mutate(genome, atoms, ("A",), 3, rng)
        assert genome[0] == "A"


def test_cross_parent_market_phase_preserves_worker_skeleton_and_structural_orders() -> None:
    base = [
        _action(["EAST"], ["NORTH"], [["HIRE"], ["BUY_SEED", "WHEAT", 2]]),
        _action(["SOUTH"], ["WEST"], [["BUY_LAND"], ["SELL", "WHEAT", 1]]),
    ]
    donor = [
        _action(["WEST"], ["SOUTH"], [["BUY_SEED", "CARROT", 7]]),
        _action(["NORTH"], ["EAST"], [["SELL", "CARROT", 5], ["HIRE"]]),
    ]
    result = _apply_evolution_genes(
        base,
        [{"operator": "replace_market_phase", "donor": "B", "start": 0, "stop": 2}],
        {"A": base, "B": donor},
    )

    assert [value["farmer"] for value in result] == [["EAST"], ["SOUTH"]]
    assert [value["hands"] for value in result] == [[["NORTH"]], [["WEST"]]]
    assert result[0]["market"] == [["HIRE"], ["BUY_SEED", "CARROT", 7]]
    assert result[1]["market"] == [["BUY_LAND"], ["SELL", "CARROT", 5]]
    assert base[0]["market"][-1] == ["BUY_SEED", "WHEAT", 2]


def test_donor_atoms_cover_only_changed_phases() -> None:
    base = [_action(["PASS"], ["PASS"], []) for _ in range(719)]
    donor = [_action(["PASS"], ["PASS"], []) for _ in range(719)]
    donor[150]["market"] = [["SELL", "WHEAT", 1]]

    atoms = _donor_atoms("A", ("A", "B"), {"A": base, "B": donor})

    assert [value for value in atoms if value["operator"] == "replace_market_phase"] == [
        {
            "operator": "replace_market_phase", "donor": "B",
            "start": 144, "stop": 168,
        },
        {
            "operator": "replace_market_phase", "donor": "B",
            "start": 144, "stop": 288,
        },
    ]


def test_worker_phase_recombination_preserves_market_and_hand_assignment() -> None:
    base = [
        _action(["EAST"], ["NORTH"], [["BUY_SEED", "WHEAT", 2]]),
        _action(["SOUTH"], ["WEST"], [["SELL", "WHEAT", 1]]),
    ]
    donor = [
        _action(["WEST"], ["SOUTH"], [["BUY_SEED", "CARROT", 7]]),
        _action(["NORTH"], ["EAST"], [["SELL", "CARROT", 5]]),
    ]

    result = _apply_evolution_genes(
        base,
        [{"operator": "replace_worker_phase", "donor": "B", "start": 0, "stop": 2}],
        {"A": base, "B": donor},
    )

    assert [value["farmer"] for value in result] == [["WEST"], ["NORTH"]]
    assert [value["hands"] for value in result] == [[["SOUTH"]], [["EAST"]]]
    assert result[0]["market"] == [["BUY_SEED", "WHEAT", 2]]
    assert result[1]["market"] == [["SELL", "WHEAT", 1]]


def test_canonical_genome_keeps_only_one_donor_per_phase() -> None:
    atoms = [
        {"operator": "replace_market_phase", "donor": "B", "start": 0, "stop": 144},
        {"operator": "replace_market_phase", "donor": "C", "start": 0, "stop": 144},
        {"operator": "reduce_hires", "start": 0, "stop": 144, "remove_per_day": 1},
    ]

    result = _canonical({0, 1, 2}, 5, random.Random(3), atoms)

    assert 2 in result
    assert len(set(result) & {0, 1}) == 1


def test_canonical_drops_overlapping_broad_donor_and_shadowed_local_gene() -> None:
    atoms = [
        {
            "operator": "scale_quantity", "operation": "SELL", "item": "MILK",
            "start": 0, "stop": 24, "numerator": 3, "denominator": 4,
        },
        {"operator": "replace_market_phase", "donor": "B", "start": 0, "stop": 24},
        {"operator": "replace_market_phase", "donor": "C", "start": 0, "stop": 144},
    ]

    result = _canonical({0, 1, 2}, 5, random.Random(3), atoms)

    assert result == (1,)


def test_canonical_keeps_one_allele_per_local_mutation_locus() -> None:
    atoms = [
        {
            "operator": "shift_market", "operation": "SELL", "item": "MILK",
            "start": 0, "stop": 144, "delta": -6,
        },
        {
            "operator": "shift_market", "operation": "SELL", "item": "MILK",
            "start": 0, "stop": 144, "delta": 6,
        },
        {
            "operator": "scale_quantity", "operation": "SELL", "item": "MILK",
            "start": 0, "stop": 144, "numerator": 3, "denominator": 4,
        },
    ]

    result = _canonical({0, 1, 2}, 5, random.Random(3), atoms)

    assert 2 in result
    assert len(set(result) & {0, 1}) == 1


def test_robust_checkpoint_metrics_exposes_seat_and_seed_fold_failure() -> None:
    import numpy as np

    scores = np.ones((1, 1, 4, 2), dtype=np.float32)
    scores[:, :, :2, 1] = 0
    margins = np.where(scores == 1, 10, -10).astype(np.float32)

    result = _robust_checkpoint_metrics(scores, margins, ("B",), seed_folds=2)

    assert result["minimum_opponent_raw_win_rate"] == .75
    assert result["minimum_opponent_seat_raw_win_rate"] == .5
    assert result["minimum_opponent_seed_fold_raw_win_rate"] == .5
    assert result["robust_raw_win_rate"] == .5


def test_behavior_descriptor_uses_executable_market_and_worker_changes() -> None:
    parent = [_action(["PASS"], ["PASS"], []) for _ in range(48)]
    changed = [_action(["PASS"], ["PASS"], []) for _ in range(48)]
    changed[2]["market"] = [["BUY_SEED", "WHEAT", 40]]
    changed[25]["farmer"] = ["EAST"]

    descriptor = _behavior_descriptor(changed, parent)

    assert descriptor == (0, 6, 4, 4, 0)


def test_elites_preserve_opponent_specialists() -> None:
    robust = ("A", ())
    left_specialist = ("A", (1,))
    right_specialist = ("B", (2,))
    rows = {
        robust: {
            "minimum_opponent_raw_win_rate": .7,
            "combined_raw_win_rate": .7,
            "minimum_opponent_mean_margin": 0,
            "incremental_oracle_score": 0,
            "incremental_oracle_margin": 0,
            "candidate_score": .7,
            "candidate_margin": 0,
            "genome": [],
            "opponent_raw_win_rates": {"L": .7, "R": .7},
            "opponent_mean_margins": {"L": 0, "R": 0},
        },
        left_specialist: {
            "minimum_opponent_raw_win_rate": .2,
            "combined_raw_win_rate": .55,
            "minimum_opponent_mean_margin": -1,
            "incremental_oracle_score": 0,
            "incremental_oracle_margin": 0,
            "candidate_score": .55,
            "candidate_margin": 0,
            "genome": [1],
            "opponent_raw_win_rates": {"L": 1.0, "R": .2},
            "opponent_mean_margins": {"L": 10, "R": -1},
        },
        right_specialist: {
            "minimum_opponent_raw_win_rate": .2,
            "combined_raw_win_rate": .55,
            "minimum_opponent_mean_margin": -1,
            "incremental_oracle_score": 0,
            "incremental_oracle_margin": 0,
            "candidate_score": .55,
            "candidate_margin": 0,
            "genome": [2],
            "opponent_raw_win_rates": {"L": .2, "R": 1.0},
            "opponent_mean_margins": {"L": -1, "R": 10},
        },
    }

    selected = _select_elites(
        [robust, left_specialist, right_specialist], rows,
        ("A", "B"), ("L", "R"), global_count=1,
        per_parent=0, per_opponent=1, population_size=5,
    )

    assert left_specialist in selected
    assert right_specialist in selected


def test_seed_archive_restores_opponent_specialist_genome() -> None:
    atom = {
        "operator": "reduce_hires", "start": 0, "stop": 144,
        "remove_per_day": 1,
    }
    ranking = [
        {
            "parent": "A", "genes": [],
            "opponent_raw_win_rates": {"hard": .5},
            "opponent_mean_margins": {"hard": 0},
        }
        for _ in range(32)
    ]
    ranking.append({
        "parent": "A", "genes": [atom],
        "opponent_raw_win_rates": {"hard": 1.0},
        "opponent_mean_margins": {"hard": 10},
    })
    payload = {"opponents": ["hard"], "ranking": ranking}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "archive.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        restored = _seed_genomes([path], {"A": [atom]}, per_opponent=1)

    assert ("A", (0,)) in restored
