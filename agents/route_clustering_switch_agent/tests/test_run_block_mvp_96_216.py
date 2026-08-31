from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_block_mvp_96_216.py"
SPEC = importlib.util.spec_from_file_location("block_mvp_96_216", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _pass_tape():
    return [
        {"farmer": ["PASS"], "hands": [], "market": []}
        for _ in range(MODULE.HORIZON)
    ]


def _block(block_id, tape, entry=None):
    macro, roles, market, events = MODULE._action_parts(tape)
    vector = np.concatenate((
        macro, np.zeros(12), np.zeros(6), roles, market, [120],
    )).astype(np.float32)
    return MODULE.Block(
        block_id=block_id, kind="baseline", parent_id=block_id,
        cluster_id="T", source={}, entry=np.zeros(12, np.float32)
        if entry is None else np.asarray(entry, np.float32),
        vector=vector, event_profile=events, tape=tape, genes=[],
    )


def test_block_vector_and_hybrid_contract():
    tape = _pass_tape()
    tape[96] = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [["BUILD_COOP"]],
        "market": [["BUY_SEED", "WHEAT", 4], ["HIRE"]],
    }
    macro, roles, market, events = MODULE._action_parts(tape)
    assert len(MODULE.block_feature_names()) == MODULE.BLOCK_DIM == 102
    assert macro.shape == (60,)
    assert roles.shape == (5,) and np.isclose(roles.sum(), 1)
    assert market.shape == (18,)
    assert events.shape == (14,)
    assert macro[0] == 1
    assert macro[5] == 1
    assert macro[11] == 1
    assert market[0] == 4

    base = _pass_tape()
    hybrid = MODULE.hybrid_tape(base, tape)
    assert hybrid[:96] == base[:96]
    assert hybrid[96:216] == tape[96:216]
    assert hybrid[216:] == base[216:]


def test_plan_vector_has_no_replay_outcomes_and_facilities_match_native():
    tape = _pass_tape()
    tape[100]["market"] = [["BUY_SEED", "WHEAT", 3], ["HIRE"]]
    tape[101]["farmer"] = ["BUILD_COOP"]
    vector, _ = MODULE._plan_vector(tape)
    assert vector.shape == (MODULE.BLOCK_DIM,)
    assert vector[72] == 3
    assert vector[73] == 0
    assert vector[74] == 4
    assert vector[75] == MODULE.STOP - MODULE.START
    assert vector[76] == 4
    assert vector[77] == MODULE.STOP - MODULE.START
    shifted = _pass_tape()
    shifted[105]["market"] = tape[100]["market"]
    shifted_vector, _ = MODULE._plan_vector(shifted)
    assert not np.array_equal(vector, shifted_vector)
    raw = np.zeros(12, np.float32)
    raw[5:10] = [1, 2, 3, 4, 5]
    total = MODULE._total_facilities(raw)
    assert total[8] == 5
    assert total[9] == 10


def test_pair_feature_contract_and_hard_compatibility():
    tape = _pass_tape()
    blocks = [_block("KEEP", tape), _block("B1", tape)]
    states = np.zeros((2, 147), np.float32)
    pairs, compatible = MODULE._pair_features(states, blocks)
    assert pairs.shape == (2, 2, MODULE.PAIR_DIM)
    assert compatible.all()
    blocks[1].entry[-1] = 1
    _, compatible = MODULE._pair_features(states, blocks)
    assert compatible[:, 0].all()
    assert not compatible[:, 1].any()


def test_entry_round_trip_preserves_feature_order():
    entry = np.arange(len(MODULE.ENTRY_NAMES), dtype=np.float32)
    block = _block("B1", _pass_tape(), entry)
    row = json.loads(json.dumps(block.row(), sort_keys=True))
    restored = np.asarray(
        [row["entry"][name] for name in MODULE.ENTRY_NAMES], np.float32,
    )
    np.testing.assert_array_equal(restored, entry)


def test_oracle_metrics_are_falsifiable():
    outcome = np.asarray([[0, 2, 0, 2], [2, 0, 2, 0]], np.uint8)
    margin = np.asarray([[-1, 1, -1, 1], [1, -1, 1, -1]], np.float32)
    metrics = MODULE.portfolio_metrics(outcome, margin, [0, 1])
    assert metrics["best_constant_raw_win_rate"] == .5
    assert metrics["oracle_raw_win_rate"] == 1.0
    assert metrics["oracle_gain_pp"] == 50.0


def test_keep_reference_prevents_replay_only_false_positive():
    outcome = np.asarray([
        [2, 2, 2, 2],
        [2, 0, 2, 0],
        [0, 2, 0, 2],
    ], np.uint8)
    margin = np.where(outcome == 2, 1, -1).astype(np.float32)
    replay_only = MODULE.portfolio_metrics(outcome, margin, [1, 2])
    keep_fixed = MODULE.portfolio_metrics(
        outcome, margin, [0, 1, 2], reference_index=0,
    )
    assert replay_only["oracle_gain_pp"] == 50.0
    assert keep_fixed["oracle_gain_pp"] == 0.0


def test_opponent_splits_do_not_leak_lineages(tmp_path):
    rows = [
        {"family": f"F{team:02d}_{route}", "team": f"T{team:02d}", "selected": True}
        for team in range(50) for route in range(2)
    ]
    path = tmp_path / "routes.json"
    path.write_text(json.dumps({"opponent_routes": rows}), encoding="utf-8")
    splits, lineages = MODULE._opponent_splits(path, 7)
    assert {name: len(value) for name, value in splits.items()} == {
        "evolution": 16, "train": 20, "validation": 10,
        "championship": 20, "ood": 10,
    }
    groups = [set(value) for value in lineages.values()]
    assert all(not left & right for index, left in enumerate(groups)
               for right in groups[index + 1:])
