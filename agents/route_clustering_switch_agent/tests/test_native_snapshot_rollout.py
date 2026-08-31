from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "fast_kaggriculture" / "python"
)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from fast_kaggriculture import (
    Config,
    FastEnv,
    NativeAgentState,
    NativeTeammateExecutor,
)


HORIZON = 719
OPENING = 0
TAIL = 1
OPPONENT = 2
SEED = 20260828
STOPS = np.asarray([*range(240, 697, 24), HORIZON], dtype=np.int64)


def _pass_action() -> dict[str, object]:
    return {"farmer": ["PASS"], "hands": [], "market": []}


def _tape() -> list[dict[str, object]]:
    return [_pass_action() for _ in range(HORIZON)]


def _executor(
    tail_216_market: list[list[object]] | None = None,
) -> NativeTeammateExecutor:
    opening = _tape()
    opening[0]["market"] = [
        ["BUY_SEED", "WHEAT", 8], ["HIRE"],
    ]
    opening[1]["farmer"] = ["PLANT", "WHEAT"]
    opening[1]["hands"] = [["PLANT", "WHEAT"]]

    tail = _tape()
    tail[216]["market"] = (
        [["BUY_LAND"]]
        if tail_216_market is None
        else list(tail_216_market)
    )
    tail[240]["market"] = [["BUY_LAND"]]

    opponent = _tape()
    reference = _tape()
    return NativeTeammateExecutor(
        [opening, tail, opponent], reference, reference,
        [reference] * 5, [reference] * 5,
    )


def _stateful_config() -> Config:
    config = Config()
    config.weed_spawn_chance = 1.0
    return config


def _stateful_patch_action() -> dict[str, object]:
    return {
        "farmer": ["PASS"],
        "hands": [["NORTH"]],
        "market": [
            ["BUY_SEED", "CARROT", 1],
            ["BUY_PRODUCT", "WHEAT", 3],
        ],
    }


def _stateful_executor(
    step_25_action: dict[str, object] | None = None,
) -> NativeTeammateExecutor:
    route = _tape()
    # With guaranteed weeds, step 24 DIGs and records an intended wheat PLANT.
    # The plant is committed from the ledger at step 25, after the HIRE exists.
    route[24] = {
        "farmer": ["PLANT", "WHEAT"],
        "hands": [],
        "market": [["HIRE"], ["BUY_SEED", "WHEAT", 1]],
    }
    if step_25_action is not None:
        route[25] = json.loads(json.dumps(step_25_action))
    route[26] = {
        "farmer": ["PASS"],
        "hands": [["PLANT", "CARROT"]],
        "market": [],
    }
    # Weed catch-up replays these one turn later, watering both planted tiles.
    route[28] = {
        "farmer": ["WATER"],
        "hands": [["WATER"]],
        "market": [],
    }
    route[48] = {
        "farmer": ["WATER"],
        "hands": [],
        "market": [["HIRE"]],
    }
    route[49]["hands"] = [["NORTH"]]
    route[50]["hands"] = [["WATER"]]
    route[72] = {
        "farmer": ["HARVEST"],
        "hands": [],
        "market": [["HIRE"]],
    }
    route[73]["hands"] = [["NORTH"]]
    route[74]["hands"] = [["HARVEST"]]
    route[96]["market"] = [
        ["SELL", "WHEAT", 1], ["SELL", "CARROT", 1],
    ]

    opponent = _tape()
    reference = _tape()
    return NativeTeammateExecutor(
        [route, opponent], reference, reference,
        [reference] * 5, [reference] * 5,
    )


def _stateful_snapshot(
    executor: NativeTeammateExecutor,
) -> tuple[FastEnv, list[NativeAgentState]]:
    env = FastEnv(_stateful_config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    executor.advance_segment(env, states[0], states[1], 0, 1, 25)
    return env, states


def _snapshot_at_216(
    executor: NativeTeammateExecutor,
    player: int,
) -> tuple[FastEnv, list[NativeAgentState], tuple[int, int]]:
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    prefix = (OPENING, OPPONENT) if player == 0 else (OPPONENT, OPENING)
    executor.advance_segment(env, states[0], states[1], *prefix, 216)
    active = (TAIL, OPPONENT) if player == 0 else (OPPONENT, TAIL)
    return env, states, active


def _constant_schedule(active: tuple[int, int]) -> np.ndarray:
    return np.tile(np.asarray(active, dtype=np.int64), (len(STOPS), 1))


def _raw_candidate_batch() -> tuple[np.ndarray, ...]:
    # Candidate 0 is KEEP. Candidate 1 is the unmodified raw TAIL action at
    # step 216. Candidate 2 removes its BUY_LAND order.
    units = np.asarray([
        [[0, -1, 1]],
        [[0, -1, 1]],
        [[0, -1, 1]],
    ], dtype=np.int32)
    unit_counts = np.asarray([-1, 1, 1], dtype=np.int32)
    market = np.asarray([
        [[0, -1, 1]],
        [[19, -1, 1]],
        [[0, -1, 1]],
    ], dtype=np.int32)
    market_counts = np.asarray([-1, 1, 0], dtype=np.int32)
    return units, unit_counts, market, market_counts


def _canonical(value: object) -> object:
    return json.loads(json.dumps(value))


def _manual_advance(
    executor: NativeTeammateExecutor,
    env: FastEnv,
    states: list[NativeAgentState],
    route0: int,
    route1: int,
    stop_step: int,
) -> None:
    while env.step_count < stop_step and not env.done:
        env.step([
            executor.action_at(env, 0, route0, states[0]),
            executor.action_at(env, 1, route1, states[1]),
        ])


def test_step0_snapshot_rollout_matches_play_batch() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    route_pairs = np.asarray([
        [OPENING, OPPONENT],
        [TAIL, OPPONENT],
    ], dtype=np.int64)

    actual = np.asarray(executor.rollout_from_batch(
        env, states[0], states[1], route_pairs,
    ))
    tasks = np.asarray([
        [OPENING, OPPONENT, SEED, -1, -1, -1, -1],
        [TAIL, OPPONENT, SEED, -1, -1, -1, -1],
    ], dtype=np.int64)
    expected = np.asarray(executor.play_batch(tasks))

    np.testing.assert_array_equal(actual, expected)
    assert env.step_count == 0


def test_step216_snapshot_matches_switch_and_does_not_mutate() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    executor.advance_segment(
        env, states[0], states[1], OPENING, OPPONENT, 216,
    )
    before = [_canonical(env.observation(player)) for player in (0, 1)]
    route_pairs = np.asarray([
        [TAIL, OPPONENT],
        [OPENING, OPPONENT],
    ], dtype=np.int64)

    first = np.asarray(executor.rollout_from_batch(
        env, states[0], states[1], route_pairs,
    ))
    second = np.asarray(executor.rollout_from_batch(
        env, states[0], states[1], route_pairs,
    ))
    expected = np.asarray(executor.play_batch(np.asarray([
        [OPENING, OPPONENT, SEED, 216, TAIL, -1, -1],
        [OPENING, OPPONENT, SEED, 216, OPENING, -1, -1],
    ], dtype=np.int64)))

    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert env.step_count == 216
    assert before == [_canonical(env.observation(player)) for player in (0, 1)]

    control = FastEnv(Config(), SEED)
    control_states = [NativeAgentState(), NativeAgentState()]
    _manual_advance(
        executor, control, control_states, OPENING, OPPONENT, 216,
    )
    _manual_advance(
        executor, control, control_states, TAIL, OPPONENT, 240,
    )
    executor.advance_segment(
        env, states[0], states[1], TAIL, OPPONENT, 240,
    )
    assert [
        _canonical(env.observation(player)) for player in (0, 1)
    ] == [
        _canonical(control.observation(player)) for player in (0, 1)
    ]

    executor.advance_segment(
        env, states[0], states[1], TAIL, OPPONENT, HORIZON,
    )
    np.testing.assert_array_equal(np.asarray(env.rewards), expected[0])


def test_step216_snapshot_matches_seat1_switch() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    executor.advance_segment(
        env, states[0], states[1], OPPONENT, OPENING, 216,
    )

    actual = np.asarray(executor.rollout_from_batch(
        env,
        states[0],
        states[1],
        np.asarray([[OPPONENT, TAIL]], dtype=np.int64),
    ))
    expected = np.asarray(executor.play_batch(np.asarray([[
        OPPONENT, OPENING, SEED, -1, -1, 216, TAIL,
    ]], dtype=np.int64)))

    np.testing.assert_array_equal(actual, expected)


def test_schedule_batch_matches_constant_and_manual_mixed_routes() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    executor.advance_segment(
        env, states[0], states[1], OPENING, OPPONENT, 216,
    )
    before = [_canonical(env.observation(player)) for player in (0, 1)]

    constant = np.tile(
        np.asarray([TAIL, OPPONENT], dtype=np.int64), (len(STOPS), 1),
    )
    mixed = constant.copy()
    mixed[1:, 0] = OPENING
    actual = np.asarray(executor.rollout_schedule_batch(
        env, states[0], states[1], np.stack((constant, mixed)), STOPS,
    ))
    expected_constant = np.asarray(executor.rollout_from_batch(
        env, states[0], states[1],
        np.asarray([[TAIL, OPPONENT]], dtype=np.int64),
    ))[0]

    control = FastEnv(Config(), SEED)
    control_states = [NativeAgentState(), NativeAgentState()]
    executor.advance_segment(
        control, control_states[0], control_states[1],
        OPENING, OPPONENT, 216,
    )
    executor.advance_segment(
        control, control_states[0], control_states[1],
        TAIL, OPPONENT, 240,
    )
    executor.advance_segment(
        control, control_states[0], control_states[1],
        OPENING, OPPONENT, HORIZON,
    )

    np.testing.assert_array_equal(actual[0], expected_constant)
    np.testing.assert_array_equal(actual[1], np.asarray(control.rewards))
    assert env.step_count == 216
    assert before == [_canonical(env.observation(player)) for player in (0, 1)]


def test_schedule_batch_rejects_nonterminal_schedule() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    states = [NativeAgentState(), NativeAgentState()]
    schedule = np.asarray([[[OPENING, OPPONENT]]], dtype=np.int64)
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_batch(
            env, states[0], states[1], schedule,
            np.asarray([240], dtype=np.int64),
        )


def test_live_raw_override_keep_and_patch_match_explicit_tapes() -> None:
    raw_pass = {"farmer": ["PASS"], "hands": [], "market": []}
    for player in (0, 1):
        executor = _executor()
        normal_env, normal_states, active = _snapshot_at_216(executor, player)
        keep_env, keep_states, _ = _snapshot_at_216(executor, player)

        normal = executor.action_at(
            normal_env, player, active[player], normal_states[player],
        )
        keep = executor.action_at_with_raw_override(
            keep_env, player, active[player], keep_states[player], None,
        )
        assert _canonical(keep) == _canonical(normal)

        other = 1 - player
        normal_other = executor.action_at(
            normal_env, other, active[other], normal_states[other],
        )
        keep_other = executor.action_at(
            keep_env, other, active[other], keep_states[other],
        )
        normal_actions = [None, None]
        keep_actions = [None, None]
        normal_actions[player], normal_actions[other] = normal, normal_other
        keep_actions[player], keep_actions[other] = keep, keep_other
        normal_env.step(normal_actions)
        keep_env.step(keep_actions)
        executor.advance_segment(
            normal_env, normal_states[0], normal_states[1], *active, HORIZON,
        )
        executor.advance_segment(
            keep_env, keep_states[0], keep_states[1], *active, HORIZON,
        )
        np.testing.assert_array_equal(keep_env.rewards, normal_env.rewards)

        patch_env, patch_states, _ = _snapshot_at_216(executor, player)
        actions = [None, None]
        actions[player] = executor.action_at_with_raw_override(
            patch_env, player, active[player], patch_states[player], raw_pass,
        )
        actions[other] = executor.action_at(
            patch_env, other, active[other], patch_states[other],
        )
        patch_env.step(actions)
        executor.advance_segment(
            patch_env, patch_states[0], patch_states[1], *active, HORIZON,
        )

        patched_executor = _executor([])
        # TAIL is first activated at this boundary, so no prefix action could
        # have inspected the replaced raw action through route lookahead.
        task = (
            [OPENING, OPPONENT, SEED, 216, TAIL, -1, -1]
            if player == 0
            else [OPPONENT, OPENING, SEED, -1, -1, 216, TAIL]
        )
        expected = np.asarray(patched_executor.play_batch(
            np.asarray([task], dtype=np.int64),
        ))[0]
        np.testing.assert_array_equal(np.asarray(patch_env.rewards), expected)


def test_live_raw_override_runs_the_existing_overlay() -> None:
    executor = _executor()
    env = FastEnv(Config(), SEED)
    state = NativeAgentState()
    final = executor.action_at_with_raw_override(
        env, 0, OPENING, state,
        {
            "farmer": ["PASS"],
            "hands": [],
            "market": [["BUY_SEED", "WHEAT", 1]],
        },
    )

    # The existing step-zero cash guard rewrites raw wheat seed quantity to 8.
    assert final["market"] == [["BUY_SEED", "WHEAT", 8]]


def test_raw_override_preserves_stateful_ledger_and_packed_fields() -> None:
    executor = _stateful_executor()
    env, states = _stateful_snapshot(executor)
    before = [_canonical(env.observation(player)) for player in (0, 1)]
    schedule = np.asarray([[0, 1]], dtype=np.int64)
    stops = np.asarray([HORIZON], dtype=np.int64)
    units = np.asarray([
        [[0, -1, 1], [0, -1, 1]],
        [[0, -1, 1], [1, -1, 1]],
    ], dtype=np.int32)
    unit_counts = np.asarray([-1, 2], dtype=np.int32)
    market = np.asarray([
        [[0, -1, 1], [0, -1, 1]],
        [[20, 1, 1], [21, 0, 3]],
    ], dtype=np.int32)
    market_counts = np.asarray([-1, 2], dtype=np.int32)

    actual = np.asarray(executor.rollout_schedule_raw_override_batch(
        env, states[0], states[1], schedule, stops, 0,
        units, unit_counts, market, market_counts,
    ))
    assert env.step_count == 25
    assert before == [_canonical(env.observation(player)) for player in (0, 1)]

    # A separate live snapshot proves the branch is exactly current-step
    # commit plus the same frozen continuation. The active weed ledger must
    # replace the raw farmer PASS with its pending PLANT, while preserving the
    # packed hand and item/quantity market fields.
    live_env, live_states = _stateful_snapshot(executor)
    raw_patch = _stateful_patch_action()
    live_action = executor.action_at_with_raw_override(
        live_env, 0, 0, live_states[0], raw_patch,
    )
    assert live_action["farmer"] == ["PLANT", "WHEAT", 1]
    assert live_action["hands"] == [["NORTH"]]
    assert live_action["market"] == [
        ["BUY_SEED", "CARROT", 1],
        ["BUY_PRODUCT", "WHEAT", 3],
    ]
    live_env.step([
        live_action,
        executor.action_at(live_env, 1, 1, live_states[1]),
    ])
    executor.advance_segment(
        live_env, live_states[0], live_states[1], 0, 1, HORIZON,
    )
    np.testing.assert_array_equal(actual[1], np.asarray(live_env.rewards))

    # The explicit tape changes only the current BUY_SEED and hand move; those
    # operations have no prior route-lookahead effect. Replaying it from step 0
    # therefore also checks that the snapshot ledger was copied, not reset.
    patched = _stateful_executor(raw_patch)
    patched_env = FastEnv(_stateful_config(), SEED)
    patched_states = [NativeAgentState(), NativeAgentState()]
    patched.advance_segment(
        patched_env, patched_states[0], patched_states[1], 0, 1, HORIZON,
    )
    np.testing.assert_array_equal(actual[1], np.asarray(patched_env.rewards))
    assert not np.array_equal(actual[0], actual[1])

    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_raw_override_batch(
            env, NativeAgentState(), NativeAgentState(), schedule, stops, 0,
            units, unit_counts, market, market_counts,
        )


def test_raw_override_batch_matches_keep_original_and_explicit_patch() -> None:
    units, unit_counts, market, market_counts = _raw_candidate_batch()
    for player in (0, 1):
        executor = _executor()
        env, states, active = _snapshot_at_216(executor, player)
        schedule = _constant_schedule(active)
        schedule[1:, player] = OPENING
        before = [_canonical(env.observation(p)) for p in (0, 1)]
        baseline = np.asarray(executor.rollout_schedule_batch(
            env, states[0], states[1], schedule[None, :, :], STOPS,
        ))[0]

        first = np.asarray(executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, player,
            units, unit_counts, market, market_counts,
        ))
        second = np.asarray(executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, player,
            units, unit_counts, market, market_counts,
        ))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first[0], baseline)
        np.testing.assert_array_equal(first[1], baseline)

        patched_executor = _executor([])
        patched_env, patched_states, _ = _snapshot_at_216(
            patched_executor, player,
        )
        for pair, stop in zip(schedule, STOPS, strict=True):
            patched_executor.advance_segment(
                patched_env, patched_states[0], patched_states[1],
                int(pair[0]), int(pair[1]), int(stop),
            )
        np.testing.assert_array_equal(first[2], np.asarray(patched_env.rewards))

        # At an arbitrary live decision point, the selected branch must equal
        # committing that raw action once and then following the same schedule.
        live_env, live_states, _ = _snapshot_at_216(executor, player)
        other = 1 - player
        live_actions = [None, None]
        live_actions[player] = executor.action_at_with_raw_override(
            live_env, player, active[player], live_states[player],
            {"farmer": ["PASS"], "hands": [], "market": []},
        )
        live_actions[other] = executor.action_at(
            live_env, other, active[other], live_states[other],
        )
        live_env.step(live_actions)
        for pair, stop in zip(schedule, STOPS, strict=True):
            executor.advance_segment(
                live_env, live_states[0], live_states[1],
                int(pair[0]), int(pair[1]), int(stop),
            )
        np.testing.assert_array_equal(first[2], np.asarray(live_env.rewards))

        assert env.step_count == 216
        assert before == [_canonical(env.observation(p)) for p in (0, 1)]
        for pair, stop in zip(schedule, STOPS, strict=True):
            executor.advance_segment(
                env, states[0], states[1],
                int(pair[0]), int(pair[1]), int(stop),
            )
        np.testing.assert_array_equal(np.asarray(env.rewards), baseline)


def test_raw_override_apis_fail_closed() -> None:
    executor = _executor()
    env, states, active = _snapshot_at_216(executor, 0)
    schedule = _constant_schedule(active)
    units, unit_counts, market, market_counts = _raw_candidate_batch()
    before = [_canonical(env.observation(player)) for player in (0, 1)]

    with np.testing.assert_raises(ValueError):
        executor.action_at(env, 2, TAIL, states[0])
    with np.testing.assert_raises(ValueError):
        executor.action_at(env, 0, 99, states[0])

    with np.testing.assert_raises(ValueError):
        executor.action_at_with_raw_override(
            env, 0, TAIL, states[0],
            {"farmer": ["NOT_AN_OP"], "hands": [], "market": []},
        )
    with np.testing.assert_raises(ValueError):
        executor.action_at_with_raw_override(
            env, 0, TAIL, states[0],
            {"farmer": ["PASS"], "hands": [], "market": [["HIRE"]] * 11},
        )

    bad_counts = unit_counts.copy()
    bad_counts[0] = 0
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units, bad_counts, market, market_counts,
        )
    bad_units = units.copy()
    bad_units[1, 0, 0] = 18
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            bad_units, unit_counts, market, market_counts,
        )
    bad_stops = STOPS.copy()
    bad_stops[-1] = np.iinfo(np.int64).max
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, bad_stops, 0,
            units, unit_counts, market, market_counts,
        )
    with np.testing.assert_raises((TypeError, ValueError)):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units.astype(np.int64), unit_counts, market, market_counts,
        )
    with np.testing.assert_raises((TypeError, ValueError)):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule.astype(np.int32), STOPS, 0,
            units, unit_counts, market, market_counts,
        )
    with np.testing.assert_raises((TypeError, ValueError)):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units, unit_counts, market, market_counts.astype(np.int64),
        )
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_raw_override_batch(
            env, states[0], states[0], schedule, STOPS, 0,
            units, unit_counts, market, market_counts,
        )

    bad_config = Config()
    bad_config.episode_steps = 721
    with np.testing.assert_raises(ValueError):
        executor.action_at(
            FastEnv(bad_config, SEED), 0, OPENING, NativeAgentState(),
        )

    reference = _tape()
    with np.testing.assert_raises(ValueError):
        NativeTeammateExecutor(
            [[]], reference, reference, [reference] * 5, [reference] * 5,
        )

    stale_env = env.clone()
    stale_env.reset_raw(SEED)
    with np.testing.assert_raises(ValueError):
        executor.action_at_with_raw_override(
            stale_env, 0, TAIL, states[0], None,
        )

    assert env.step_count == 216
    assert before == [_canonical(env.observation(player)) for player in (0, 1)]
    control_env, control_states, _ = _snapshot_at_216(executor, 0)
    executor.advance_segment(
        env, states[0], states[1], TAIL, OPPONENT, HORIZON,
    )
    executor.advance_segment(
        control_env, control_states[0], control_states[1],
        TAIL, OPPONENT, HORIZON,
    )
    np.testing.assert_array_equal(env.rewards, control_env.rewards)

    terminal_env, terminal_states, _ = _snapshot_at_216(executor, 0)
    executor.advance_segment(
        terminal_env, terminal_states[0], terminal_states[1],
        TAIL, OPPONENT, HORIZON,
    )
    with np.testing.assert_raises(ValueError):
        executor.action_at_with_raw_override(
            terminal_env, 0, TAIL, terminal_states[0], None,
        )


def test_unit_override_sequence_keep_matches_schedule_and_is_pure() -> None:
    executor = _executor()
    env, states, active = _snapshot_at_216(executor, 0)
    schedule = _constant_schedule(active)
    units = np.zeros((2, 4, 1, 3), dtype=np.int32)
    unit_counts = np.full((2, 4), -1, dtype=np.int32)
    before = [_canonical(env.observation(player)) for player in (0, 1)]
    baseline = np.asarray(executor.rollout_schedule_batch(
        env, states[0], states[1], schedule[None, :, :], STOPS,
    ))[0]

    first, first_market_diff = (
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units, unit_counts,
        )
    )
    second, second_market_diff = (
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units, unit_counts,
        )
    )

    np.testing.assert_array_equal(np.asarray(first), np.tile(baseline, (2, 1)))
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first_market_diff, np.zeros(2, dtype=np.int32))
    np.testing.assert_array_equal(first_market_diff, second_market_diff)
    assert env.step_count == 216
    assert before == [_canonical(env.observation(player)) for player in (0, 1)]


def test_live_unit_override_preserves_route_market_and_h1_matches_raw() -> None:
    executor = _executor()
    unit_env, unit_states, active = _snapshot_at_216(executor, 0)
    raw_env, raw_states, _ = _snapshot_at_216(executor, 0)
    unit_action = executor.action_at_with_unit_override(
        unit_env, 0, TAIL, unit_states[0],
        {"farmer": ["PASS"], "hands": []},
    )
    raw_action = executor.action_at_with_raw_override(
        raw_env, 0, TAIL, raw_states[0],
        {"farmer": ["PASS"], "hands": [], "market": [["BUY_LAND"]]},
    )
    assert _canonical(unit_action) == _canonical(raw_action)

    unit_env, unit_states, active = _snapshot_at_216(executor, 0)
    raw_env, raw_states, _ = _snapshot_at_216(executor, 0)
    schedule = _constant_schedule(active)
    units = np.asarray([[[[0, -1, 1]]]], dtype=np.int32)
    counts = np.asarray([[1]], dtype=np.int32)
    actual, market_diff = (
        executor.rollout_schedule_unit_override_sequence_batch(
            unit_env, unit_states[0], unit_states[1], schedule, STOPS, 0,
            units, counts,
        )
    )
    raw_units = units[:, 0, :, :]
    raw_counts = counts[:, 0]
    raw_market = np.asarray([[[19, -1, 1]]], dtype=np.int32)
    raw_market_counts = np.asarray([1], dtype=np.int32)
    expected = executor.rollout_schedule_raw_override_batch(
        raw_env, raw_states[0], raw_states[1], schedule, STOPS, 0,
        raw_units, raw_counts, raw_market, raw_market_counts,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(market_diff, np.zeros(1, dtype=np.int32))


def test_unit_override_sequence_uses_absolute_step_across_segments() -> None:
    executor = _executor()
    env, states, active = _snapshot_at_216(executor, 0)
    schedule = np.asarray([
        active,
        (OPENING, OPPONENT),
    ], dtype=np.int64)
    stops = np.asarray([217, HORIZON], dtype=np.int64)
    units = np.asarray([[
        [[3, -1, 1]],
        [[4, -1, 1]],
    ]], dtype=np.int32)
    counts = np.asarray([[1, 1]], dtype=np.int32)
    actual, market_diff = (
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, stops, 0, units, counts,
        )
    )

    control, control_states, _ = _snapshot_at_216(executor, 0)
    for route, raw_units in (
        (TAIL, {"farmer": ["EAST"], "hands": []}),
        (OPENING, {"farmer": ["WEST"], "hands": []}),
    ):
        control.step([
            executor.action_at_with_unit_override(
                control, 0, route, control_states[0], raw_units,
            ),
            executor.action_at(
                control, 1, OPPONENT, control_states[1],
            ),
        ])
    executor.advance_segment(
        control, control_states[0], control_states[1],
        OPENING, OPPONENT, HORIZON,
    )
    np.testing.assert_array_equal(actual[0], np.asarray(control.rewards))
    np.testing.assert_array_equal(market_diff, np.zeros(1, dtype=np.int32))


def test_unit_override_sequence_apis_fail_closed() -> None:
    executor = _executor()
    env, states, active = _snapshot_at_216(executor, 0)
    schedule = _constant_schedule(active)
    units = np.zeros((1, 1, 1, 3), dtype=np.int32)
    counts = np.ones((1, 1), dtype=np.int32)

    with np.testing.assert_raises(ValueError):
        executor.action_at_with_unit_override(
            env, 0, TAIL, states[0],
            {"farmer": ["PASS"], "hands": [], "market": []},
        )
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            np.zeros((1, 0, 1, 3), dtype=np.int32),
            np.zeros((1, 0), dtype=np.int32),
        )
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            np.zeros((1, 9, 1, 3), dtype=np.int32),
            np.ones((1, 9), dtype=np.int32),
        )
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units, np.zeros((1, 1), dtype=np.int32),
        )
    bad_op = units.copy()
    bad_op[0, 0, 0, 0] = 18
    with np.testing.assert_raises(ValueError):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            bad_op, counts,
        )
    with np.testing.assert_raises((TypeError, ValueError)):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule, STOPS, 0,
            units.astype(np.int64), counts,
        )
    with np.testing.assert_raises((TypeError, ValueError)):
        executor.rollout_schedule_unit_override_sequence_batch(
            env, states[0], states[1], schedule.astype(np.int32), STOPS, 0,
            units, counts,
        )
