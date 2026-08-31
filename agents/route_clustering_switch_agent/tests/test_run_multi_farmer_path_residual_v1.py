from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
FAST_PACKAGE = Path(__file__).resolve().parents[1] / "fast_kaggriculture" / "python"
for path in (SCRIPTS, FAST_PACKAGE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fast_kaggriculture import Config, FastEnv
import run_multi_farmer_path_residual_v1 as path_v1
from meta_agent.src.route_switch_features import RouteSwitchHistory


def _action(*units: list[object], market: list[list[object]] | None = None) -> dict[str, object]:
    values = list(units) or [["PASS"]]
    return {
        "farmer": values[0], "hands": values[1:], "market": list(market or []),
    }


def _tape() -> list[dict[str, object]]:
    return [_action() for _ in range(path_v1.HORIZON)]


def _donor(
    block_id: str,
    traces: list[tuple[list[object], ...]],
    *,
    support: int = 1,
    team: str = "train",
) -> path_v1.DonorBlock:
    actions = [_action()]
    actions.extend(_action(*values) for values in traces)
    while len(actions) < 24:
        actions.append(_action())
    return path_v1.DonorBlock(
        block_id=block_id, anchor=216, cluster_id="C00",
        source_provenance_id=f"p:{block_id}", source_route_id=f"r:{block_id}",
        support=support, actions=tuple(actions), team_name=team,
    )


def test_donor_candidates_require_two_differences_and_a_movement() -> None:
    base = _tape()
    for step in range(217, 221):
        base[step] = _action(["PASS"], ["PASS"])
    donor = _donor("d", [
        (["NORTH"], ["EAST"]),
        (["PASS"], ["WEST"]),
        (["PASS"], ["PASS"]),
        (["PASS"], ["PASS"]),
    ])

    candidates = path_v1.donor_candidates(base, [donor], 216, 2)
    codes = [candidate.code for candidate in candidates]
    assert codes[0] == "KEEP"
    assert not any("A0_" in code for code in codes[1:])
    assert any(code.startswith("SINGLE_A1_") for code in codes)
    assert not any(code.startswith("PAIR_") for code in codes)


def test_pair_uses_top_two_eligible_combinations_and_keeps_both_singles() -> None:
    base = _tape()
    for step in range(217, 221):
        base[step] = _action(["PASS"], ["PASS"], ["PASS"])
    donor = _donor("pair", [
        (["NORTH"], ["EAST"], ["WEST"]),
        (["SOUTH"], ["WEST"], ["EAST"]),
        (["NORTH"], ["EAST"], ["PASS"]),
        (["SOUTH"], ["PASS"], ["PASS"]),
    ])

    candidates = path_v1.donor_candidates(base, [donor], 216, 3)
    codes = {candidate.code for candidate in candidates}
    assert "PAIR_A0_A1_pair" in codes
    assert "PAIR_A0_A2_pair" in codes
    assert "PAIR_A1_A2_pair" not in codes
    for actor in (0, 1, 2):
        assert f"SINGLE_A{actor}_pair" in codes


def test_future_hire_slots_stay_on_the_base_trace() -> None:
    base = _tape()
    base[217] = _action(["PASS"], ["PASS"])
    base[218] = _action(["PASS"], ["PASS"], ["NORTH"])
    base[219] = _action(["PASS"], ["PASS"], ["SOUTH"])
    base[220] = _action(["PASS"], ["PASS"], ["EAST"])
    donor = _donor("hire", [
        (["NORTH"], ["EAST"]),
        (["SOUTH"], ["WEST"]),
        (["NORTH"], ["EAST"]),
        (["SOUTH"], ["WEST"]),
    ])

    candidates = path_v1.donor_candidates(base, [donor], 216, 2)
    pair = next(candidate for candidate in candidates if candidate.kind == "PAIR")
    assert [len(offset) for offset in pair.units] == [2, 3, 3, 3]
    assert pair.units[1][2] == ("NORTH",)
    assert pair.units[2][2] == ("SOUTH",)
    packed, counts = path_v1.pack_unit_plans(candidates)
    assert packed.dtype == counts.dtype == np.int32
    assert counts[0].tolist() == [-1] * 4
    assert counts[candidates.index(pair)].tolist() == [2, 3, 3, 3]


def test_mpc_later_step_uses_the_matching_donor_offset() -> None:
    base = _tape()
    for step in range(217, 222):
        base[step] = _action(["PASS"])
    donor = _donor("offset", [
        (["NORTH"],), (["EAST"],), (["SOUTH"],), (["WEST"],),
        (["NORTH"],),
    ])
    at_217 = path_v1.donor_candidates(
        base, [donor], 216, 1, decision_step=217,
    )
    at_218 = path_v1.donor_candidates(
        base, [donor], 216, 1, decision_step=218,
    )
    first = next(candidate for candidate in at_217 if not candidate.keep)
    second = next(candidate for candidate in at_218 if not candidate.keep)
    assert first.units[0][0] == ("NORTH",)
    assert second.units[0][0] == ("EAST",)


def test_mpc1_last_window_offset_scores_only_the_action_it_executes() -> None:
    base = _tape()
    for step in range(217, 224):
        base[step] = _action(["PASS"])
    donor = _donor("last-offset", [
        (["NORTH"],), (["SOUTH"],), (["EAST"],), (["WEST"],),
        (["NORTH"],), (["EAST"],), (["SOUTH"],), (["WEST"],),
    ])

    raw = path_v1.donor_candidates(
        base, [donor], 216, 1, decision_step=220,
    )
    raw_edit = next(candidate for candidate in raw if not candidate.keep)
    assert [units[0][0] for units in raw_edit.units] == [
        "WEST", "NORTH", "EAST", "SOUTH",
    ]

    mpc1 = path_v1.effective_candidates(raw, path_v1.MPC1_OVERRIDE_HORIZON)
    edit = next(candidate for candidate in mpc1 if not candidate.keep)
    assert [units[0][0] for units in edit.units] == ["WEST"]
    _, counts = path_v1.pack_unit_plans(mpc1)
    assert counts[mpc1.index(edit)].tolist() == [1, -1, -1, -1]


def test_commit4_still_scores_and_executes_all_four_offsets() -> None:
    base = _tape()
    for step in range(217, 221):
        base[step] = _action(["PASS"])
    donor = _donor("commit", [
        (["NORTH"],), (["SOUTH"],), (["EAST"],), (["WEST"],),
    ])

    raw = path_v1.donor_candidates(base, [donor], 216, 1)
    commit4 = path_v1.effective_candidates(raw, path_v1.COMMIT4_OVERRIDE_HORIZON)
    edit = next(candidate for candidate in commit4 if not candidate.keep)
    assert len(edit.units) == 4
    _, counts = path_v1.pack_unit_plans(commit4)
    assert counts[commit4.index(edit)].tolist() == [1, 1, 1, 1]


def test_mpc1_features_hide_the_three_unexecuted_donor_actions() -> None:
    base = _tape()
    for step in range(217, 221):
        base[step] = _action(["PASS"])
    donor = _donor("feature-h1", [
        (["NORTH"],), (["SOUTH"],), (["EAST"],), (["WEST"],),
    ])
    raw = path_v1.donor_candidates(base, [donor], 216, 1)
    candidates = path_v1.effective_candidates(
        raw, path_v1.MPC1_OVERRIDE_HORIZON,
    )
    edit_index = next(
        index for index, candidate in enumerate(candidates) if not candidate.keep
    )
    env = FastEnv(Config(), 20260829)
    observation = dict(env.observation(0))
    observation.update(step=217, day=9, hour=1, player=0)
    history = RouteSwitchHistory()
    history.update(observation)

    rows = path_v1.candidate_feature_rows(
        observation, history, base, base, tuple(), candidates, 0, 217,
    )
    names = {name: index for index, name in enumerate(path_v1.FEATURE_NAMES)}
    assert rows[edit_index, names["plan_0_actor_0_changed"]] == 1.0
    assert rows[edit_index, names["trace_novelty"]] == 0.5
    for offset in range(1, path_v1.PLAN_HORIZON):
        assert rows[edit_index, names[f"plan_{offset}_actor_0_changed"]] == 0.0
        assert rows[edit_index, names[f"plan_{offset}_actor_0_op_pass"]] == 1.0
        assert rows[edit_index, names[f"plan_{offset}_actor_0_op_south"]] == 0.0
        assert rows[edit_index, names[f"plan_{offset}_actor_0_op_east"]] == 0.0
        assert rows[edit_index, names[f"plan_{offset}_actor_0_op_west"]] == 0.0


def test_diversity_selection_beats_duplicate_support_top_three() -> None:
    base = _tape()
    for step in range(217, 221):
        base[step] = _action(["PASS"], ["PASS"])
    duplicate = [
        (["NORTH"], ["EAST"]), (["SOUTH"], ["WEST"]),
        (["NORTH"], ["EAST"]), (["SOUTH"], ["WEST"]),
    ]
    different = [
        (["EAST"], ["NORTH"]), (["WEST"], ["SOUTH"]),
        (["EAST"], ["NORTH"]), (["WEST"], ["SOUTH"]),
    ]
    donors = [
        _donor("a", duplicate, support=100),
        _donor("b", duplicate, support=90),
        _donor("c", duplicate, support=80),
        _donor("z", different, support=1),
    ]

    selected, audit = path_v1.select_diverse_donors(base, donors, 216, 2)
    assert "z" in {donor.block_id for donor in selected}
    assert audit["diversity_unique_residuals"] > audit["support_top_unique_residuals"]


def test_lineage_filter_is_exact_and_support_ranked() -> None:
    manifest = {"active_representatives": {
        str(anchor): [
            {"block_id": f"keep-{anchor}", "source_provenance_id": "p0", "support": 2},
            {"block_id": f"drop-{anchor}", "source_provenance_id": "p1", "support": 99},
            {"block_id": f"best-{anchor}", "source_provenance_id": "p2", "support": 4},
        ]
        for anchor in path_v1.ANCHORS
    }}
    selected = path_v1._select_active_donor_rows(
        manifest, {"p0": "train", "p1": "tetsuya", "p2": "train"},
        {"tetsuya"}, 2,
    )
    assert [row["block_id"] for row in selected[216]] == ["best-216", "keep-216"]


def test_fixed_slot_features_keep_actor_identity_and_have_frozen_shape() -> None:
    base = _tape()
    donor = _donor("feature", [
        (["NORTH"],), (["SOUTH"],), (["NORTH"],), (["SOUTH"],),
    ])
    candidates = path_v1.donor_candidates(base, [donor], 216, 1)
    env = FastEnv(Config(), 20260829)
    observation = dict(env.observation(0))
    observation.update(step=216, day=9, hour=0, player=0)
    history = RouteSwitchHistory()
    history.update(observation)
    rows = path_v1.candidate_feature_rows(
        observation, history, base, base, tuple(), candidates, 0, 216,
    )
    assert rows.shape == (len(candidates), len(path_v1.FEATURE_NAMES))
    assert np.isfinite(rows).all()
    assert not np.array_equal(rows[0], rows[-1])


def test_training_labels_use_h4_only_at_start_and_h1_at_every_mpc_step() -> None:
    anchor = path_v1.ANCHORS[0]
    assert path_v1.training_label_contract(anchor + 1, 1) == (
        ("commit4_h4", 4), ("mpc1_h1", 1),
    )
    for offset in (2, 3, 4):
        assert path_v1.training_label_contract(anchor + offset, 1) == (
            ("mpc1_h1", 1),
        )
    assert path_v1.training_label_contract(anchor, 1) == ()
    assert path_v1.training_label_contract(anchor + 5, 1) == ()


def test_learned_controller_rejects_the_other_horizon_ranker() -> None:
    h4 = {"controller": "commit4_h4", "training_override_horizon": 4}
    h1 = {"controller": "mpc1_h1", "training_override_horizon": 1}
    assert path_v1._validate_ranker_contract("commit4", h4) == 4
    assert path_v1._validate_ranker_contract("mpc1", h1) == 1
    with pytest.raises(ValueError, match="controller horizon"):
        path_v1._validate_ranker_contract("mpc1", h4)
    with pytest.raises(ValueError, match="controller horizon"):
        path_v1._validate_ranker_contract("commit4", h1)


def _tiny_panel(target_scale: float) -> path_v1.PathPanel:
    panel = path_v1.PathPanel.empty()
    for decision in range(4):
        for edit in (0, 1):
            row = np.zeros(len(path_v1.FEATURE_NAMES), np.float32)
            row[(decision * 2 + edit) % len(row)] = 1.0
            delta = 0.0 if edit == 0 else target_scale * (decision - 1.5)
            panel.features.append(row)
            panel.target.append(path_v1.residual._signed_log(delta))
            panel.delta_margin.append(delta)
            panel.rewards.append((delta, 0.0))
            panel.seed.append(10 + decision // 2)
            panel.seat.append(0)
            panel.opponent.append(0)
            panel.step.append(path_v1.ANCHORS[0] + 1)
            panel.edit.append(edit)
            panel.decision.append(decision)
            panel.selected.append(edit == int(delta > 0))
            panel.genome.append(0)
            panel.candidate_sha256.append(f"candidate-{decision}-{edit}")
    return panel


def test_dual_ranker_artifacts_have_separate_names_and_hashes(tmp_path: Path) -> None:
    split = tmp_path / "split_manifest.json"
    vocabulary = tmp_path / "path_residual_vocabulary.json"
    split.write_text("{}\n", encoding="utf-8")
    vocabulary.write_text("{}\n", encoding="utf-8")
    args = SimpleNamespace(trees=8, cv_trees=4, cv_folds=2, random_seed=7)

    _, h4, _, h4_paths = path_v1._fit_ranker_artifacts(
        tmp_path, _tiny_panel(1.0), "commit4_h4", 4, (1,),
        args, split, vocabulary,
    )
    _, h1, _, h1_paths = path_v1._fit_ranker_artifacts(
        tmp_path, _tiny_panel(2.0), "mpc1_h1", 1, (1, 2, 3, 4),
        args, split, vocabulary,
    )

    assert {path.name for path in h4_paths}.isdisjoint(
        path.name for path in h1_paths
    )
    assert h4["controller"] == "commit4_h4"
    assert h1["controller"] == "mpc1_h1"
    assert h4["training_override_horizon"] == 4
    assert h1["training_override_horizon"] == 1
    assert h4["training_panel_sha256"] != h1["training_panel_sha256"]
    assert h4["model_sha256"] != h1["model_sha256"]
