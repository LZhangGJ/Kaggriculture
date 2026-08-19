from __future__ import annotations

import json
from pathlib import Path

from replay_task_cards_compiler import (
    compile_profile,
    discover_replay_files,
    expand_bundle,
    extract_bundle,
    normalize_action,
    write_profile,
)
from strategic_v5 import load_replay_task_card_program_v1


OPENING = {
    "farmer": ["BUILD_PASTURE"],
    "hands": [],
    "market": [
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["HIRE"],
        ["BUY_ANIMAL", "COW", 2],
        ["BUY_ANIMAL", "SHEEP", 2],
        ["BUY_SEED", "WHEAT", 7],
        ["BUY_SEED", "MELON", 12],
        ["BUY_PRODUCT", "WHEAT", 6],
    ],
}


def _write_replay(path: Path, episode_id: int) -> None:
    document = {
        "id": f"internal-{episode_id}",
        "info": {"TeamNames": ["Expert", "Opponent"]},
        "steps": [
            [
                {
                    "action": {"farmer": ["PASS"], "hands": [], "market": []},
                    # Deliberately omit `step`: seat-1 official observations may
                    # expose only day/hour, and alignment must still be exact.
                    "observation": {"day": 0, "hour": 0},
                },
                {"action": {}, "observation": {"day": 0, "hour": 0}},
            ],
            [
                {"action": OPENING, "observation": {"day": 0, "hour": 1}},
                {"action": {}, "observation": {"day": 0, "hour": 1}},
            ],
        ],
    }
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")


def test_bundle_roundtrip_preserves_atomic_order_and_explicit_amount() -> None:
    action = {
        "farmer": ["PICKUP", "SHEEP", 1],
        "hands": [["WEST"]],
        "market": OPENING["market"],
    }
    bundle = extract_bundle(action)
    assert bundle["cards"][0]["explicit_quantity"] is True
    hire = next(card for card in bundle["cards"] if card.get("op") == "HIRE")
    assert hire["quantity"] == 5
    assert hire["repeat"] == 5
    assert expand_bundle(bundle) == normalize_action(action)


def test_compile_profile_creates_fixed_runtime_program(tmp_path: Path) -> None:
    _write_replay(tmp_path / "1002.json", 1002)
    _write_replay(tmp_path / "1001.json", 1001)
    files = discover_replay_files([tmp_path], ["Expert"], max_episodes=2)
    assert [path.stem for path in files] == ["1002", "1001"]

    document = compile_profile(
        files,
        ["Expert"],
        profile_name="expert_test_v1",
        min_support=2,
        min_consensus=1.0,
    )
    assert document["coverage"]["aligned_samples"] == 2
    assert document["coverage"]["roundtrip_failures"] == 0
    assert document["coverage"]["accepted_templates"] == 1
    assert document["activation_gate"] == {"passed": True, "reasons": []}
    template = document["templates"][0]
    assert template["source_step"] == 0
    assert template["support"] == 2
    assert template["consensus"] == 1.0

    output = write_profile(document, tmp_path / "profile.json")
    program = load_replay_task_card_program_v1(output)
    assert program.enabled.shape == (720,)
    assert program.market_quantity.shape == (720, 21)
    assert program.market_quantity[0, :12].tolist() == [
        0,
        0,
        6,
        0,
        2,
        2,
        7,
        0,
        0,
        0,
        12,
        5,
    ]
    assert int(program.build_animal_id[0]) == 1
    assert int(program.support[0]) == 2

    rejected = compile_profile(
        files,
        ["Expert"],
        profile_name="rejected_test_v1",
        min_support=3,
        min_consensus=1.0,
    )
    assert rejected["activation_gate"]["passed"] is False
    assert "no_accepted_templates" in rejected["activation_gate"]["reasons"]
