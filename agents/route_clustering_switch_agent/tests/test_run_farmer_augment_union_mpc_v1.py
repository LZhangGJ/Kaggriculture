from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_farmer_augment_union_mpc_v1 as union_mpc


def _candidate(
    code: str,
    unit: str,
    *,
    kind: str = "SINGLE",
    assignments: tuple[tuple[int, int], ...] = ((1, 1),),
    aliases: tuple[str, ...] | None = None,
) -> union_mpc.path_v1.PathCandidate:
    return union_mpc.path_v1.PathCandidate(
        code=code, kind=kind,
        units=(((unit,),),), assignments=assignments,
        donor_id="d", donor_cluster="C00", donor_support=1,
        donor_rank=0, trace_novelty=1,
        aliases=aliases or (code,), raw_sha256=f"sha-{unit}",
    )


def test_union_membership_preserves_ordered_a_sha_and_only_adds_farmer() -> None:
    keep = _candidate("KEEP", "PASS", kind="KEEP", assignments=tuple())
    a_edit = _candidate("A", "EAST")
    c_alias = _candidate("C_ALIAS", "EAST")
    farmer = _candidate(
        "GUARANTEE_FARMER_A0_d", "WEST", assignments=((0, 0),),
        aliases=("GUARANTEE_FARMER_A0_d",),
    )
    union, a_indices, _, farmer_indices = union_mpc._union_membership(
        [keep, a_edit], [keep, c_alias, farmer],
    )
    assert [union[index].raw_sha256 for index in a_indices] == ["sha-PASS", "sha-EAST"]
    assert farmer_indices == [2]
    assert union[2].assignments == ((0, 0),)


def _signal_row(seed: int, opponent: str, anchor: int, positive: bool = True) -> dict[str, object]:
    return {
        "seed": seed, "opponent": opponent, "anchor": anchor,
        "farmer_candidates": [{
            "path_pure": True,
            "novelty_uplift_vs_best_path_pure_A": 1.0 if positive else 0.0,
        }],
    }


def test_signal_gate_requires_eight_decisions_four_seed_folds_and_breadth() -> None:
    seeds = tuple(range(100, 108))
    rows = [
        _signal_row(seed, f"O{index % 2}", 216 if index % 2 else 240)
        for index, seed in enumerate(seeds)
    ]
    gate = union_mpc.signal_gate(rows, seeds)
    assert gate["passed"]
    assert gate["positive_seed_folds"] == [0, 1, 2, 3]
    assert len(gate["positive_opponents"]) == 2
    assert len(gate["positive_anchors"]) == 2

    failed = union_mpc.signal_gate(rows[:7], seeds)
    assert not failed["passed"]
    assert "positive_novelty_decisions_below_minimum" in failed["reasons"]


def _farmer_arrays(deployment: list[float]) -> dict[str, np.ndarray]:
    count = len(deployment)
    return {
        "decision": np.arange(count, dtype=np.int32),
        "deployment_uplift": np.asarray(deployment, np.float64),
        "opponent": np.asarray([index % 2 for index in range(count)], np.int16),
        "seed": np.arange(200, 200 + count, dtype=np.int64),
    }


def test_zero_harm_calibration_uses_lopo_hard_constraint() -> None:
    arrays = _farmer_arrays([-5.0, 3.0, -4.0, 2.0])
    mean = np.asarray([0.1, 2.0, 0.1, 2.0])
    std = np.zeros(4)
    positive = np.ones(4)
    chosen, _, rows = union_mpc.zero_harm_calibration(
        arrays, mean, std, positive, farmer_only=True,
    )
    assert chosen["harmful"] == 0
    assert not any(chosen["opponent_harm"].values())
    assert chosen["selected"] == 2
    assert np.count_nonzero(rows >= 0) == 2

    impossible = _farmer_arrays([-5.0, 3.0, -4.0, 2.0])
    chosen, _, rows = union_mpc.zero_harm_calibration(
        impossible, np.asarray([2.0, 1.0, 2.0, 1.0]), std, positive,
        farmer_only=True,
    )
    assert chosen["harmful"] == 0
    assert chosen["selected"] == 0
    assert np.all(rows == -1)


def test_farmer_pair_target_uses_a_oof_choice_and_keeps_novelty_separate() -> None:
    a = {
        "features": np.zeros((2, len(union_mpc.path_v1.FEATURE_NAMES)), np.float32),
        "delta_margin": np.asarray([0.0, 4.0]),
        "decision": np.asarray([7, 7], np.int32),
    }
    a["features"][1, 0] = 1.0
    farmer = {
        "features": np.full((1, len(union_mpc.path_v1.FEATURE_NAMES)), 2.0, np.float32),
        "novelty_uplift": np.asarray([3.0]),
        "delta_margin": np.asarray([10.0]),
        "seed": np.asarray([1]), "seat": np.asarray([0]),
        "opponent": np.asarray([0]), "step": np.asarray([217]),
        "anchor": np.asarray([216]), "decision": np.asarray([7]),
        "candidate_sha256": np.asarray(["f"]), "union_index": np.asarray([2]),
    }
    result = union_mpc.farmer_pair_arrays(a, farmer, np.asarray([1]))
    width = len(union_mpc.path_v1.FEATURE_NAMES)
    assert result["deployment_uplift"].tolist() == [6.0]
    assert result["novelty_uplift"].tolist() == [3.0]
    assert result["features"][0, width] == 1.0


class _Tree:
    def __init__(self, prediction: list[float]) -> None:
        self.prediction = np.asarray(prediction, np.float64)

    def predict(self, values: np.ndarray) -> np.ndarray:
        assert len(values) == len(self.prediction)
        return self.prediction


class _Model:
    def __init__(self, prediction: list[float]) -> None:
        self.estimators_ = [_Tree(prediction), _Tree(prediction)]


def test_farmer_gate_rejection_returns_exact_a_fallback_sentinel() -> None:
    features = np.zeros((2, 4), np.float32)
    rejected, *_ = union_mpc.choose_farmer_override(
        _Model([1.0, 2.0]),
        {"chosen": {"beta": 0.0, "threshold": 3.0, "min_positive_fraction": 0.5}},
        features,
    )
    assert rejected == -1
    empty, mean, std, positive = union_mpc.choose_farmer_override(
        _Model([]),
        {"chosen": {"beta": 0.0, "threshold": 0.0, "min_positive_fraction": 0.5}},
        np.empty((0, 4), np.float32),
    )
    assert empty == -1 and not len(mean) and not len(std) and not len(positive)


def test_cli_defaults_register_lopo_and_signal_contract() -> None:
    args = union_mpc.parser().parse_args([])
    assert args.train_seed_count == 8
    assert args.validation_seed_count == 4
    assert args.signal_min_decisions == 8
    assert args.block_library == union_mpc.farmer_ab.STRICT_LIBRARY

