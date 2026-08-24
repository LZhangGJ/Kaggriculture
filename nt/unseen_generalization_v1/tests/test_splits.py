from __future__ import annotations

from unseen_generalization_v1.schema import PlayerBehavior, ReplaySummary
from unseen_generalization_v1.splits import build_group_splits


def _summary(index: int, signature: str, family: str) -> ReplaySummary:
    behavior = PlayerBehavior(
        player_index=0,
        final_reward=float(index),
        final_status="DONE",
        farmer_ops={"PASS": 1},
        hand_ops={},
        market_ops={},
        market_items_bought={},
        market_items_sold={},
        first_op_step={},
        daily_public_snapshots=(),
        behavior_signature=signature,
        coarse_family=family,
    )
    return ReplaySummary(
        source_path=f"replay-{index}.json",
        replay_sha256=f"{index:064x}",
        episode_id=str(index),
        submission_ids=(),
        step_count=1,
        player_count=1,
        episode_seed=index,
        players=(behavior,),
    )


def test_split_keeps_signatures_and_holdout_families_together():
    summaries = [
        _summary(1, "a" * 64, "MILK_PRIMARY"),
        _summary(2, "a" * 64, "MILK_PRIMARY"),
        _summary(3, "b" * 64, "WOOL_PRIMARY"),
        _summary(4, "c" * 64, "EGG_PRIMARY"),
        _summary(5, "d" * 64, "STRAWBERRY_PRIMARY"),
    ]
    result = build_group_splits(
        summaries,
        seed=7,
        validation_fraction=0.4,
        family_holdout_fraction=0.25,
    )
    assert result["leakage_checks"]["status"] == "PASS"
    rows = result["rows"]
    signature_to_splits = {}
    for row in rows.values():
        signature_to_splits.setdefault(row["behavior_signature"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in signature_to_splits.values())
    holdout = set(result["holdout_families"])
    for row in rows.values():
        if row["coarse_family"] in holdout:
            assert row["split"] == "family_holdout"


def test_same_submission_id_never_crosses_split_even_with_different_signatures():
    left = _summary(10, "e" * 64, "MILK_PRIMARY")
    right = _summary(11, "f" * 64, "MILK_PRIMARY")
    left = ReplaySummary(**{**left.__dict__, "submission_ids": ("9001",)})
    right = ReplaySummary(**{**right.__dict__, "submission_ids": ("9001",)})
    result = build_group_splits(
        [left, right, _summary(12, "g" * 64, "WOOL_PRIMARY"), _summary(13, "h" * 64, "EGG_PRIMARY")],
        seed=99,
        validation_fraction=0.5,
        family_holdout_fraction=0.0,
    )
    own_rows = [row for row in result["rows"].values() if row["agent_hint"] == "9001"]
    assert len({row["split"] for row in own_rows}) == 1
    assert result["leakage_checks"]["submission_id_cross_split"] == 0
