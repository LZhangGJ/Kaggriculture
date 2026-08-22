"""Official Kaggle replay ingestion for the hierarchical V7 policy.

Kaggle replay files store the action chosen from observation ``t`` on replay
state ``t + 1``.  This module owns that alignment rule, reconstructs the shared
``step`` field for player 1, produces inverse-planning task-chain labels, and
computes the same liquidatable-net-asset returns used by V7 self play.

The serialized label bundle intentionally references the immutable raw replay
instead of copying every observation into every label row.  This keeps the
dataset auditable and avoids multiplying the size of the official daily dump.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .board_rl import (
    multi_horizon_resource_growth_targets,
    paired_liquidatable_net_asset_potentials,
    potential_shaped_value_targets,
)
from .decision_schema import TaskType
from .hierarchical_bc import (
    DailyBudgetSoftTarget,
    IntentLabel,
    daily_budget_soft_targets,
)
from .hierarchical_schema import MODE_INDEX
from .hierarchical_trace import (
    IntentPhase,
    SoftIntentDistribution,
    SoftIntentHypothesis,
    TraceSource,
    inverse_episode_traces,
    merge_trace_quality_audits,
    trace_quality_audit,
)
from .opponent_model import build_opponent_history_sequences


OFFICIAL_V7_SCHEMA = "kaggriculture.official-replay-v7.v1"
OFFICIAL_V7_MANIFEST_SCHEMA = "kaggriculture.official-replay-v7-manifest.v1"
_SHARED_FIELDS = ("day", "hour", "farms", "market", "town")
_UNIT_OPERATIONS = (
    "NORTH",
    "SOUTH",
    "EAST",
    "WEST",
    "PASS",
    "PICKUP",
    "DROP",
    "PLANT",
    "WATER",
    "HARVEST",
    "FERTILIZE",
    "BUILD_COOP",
    "BUILD_PASTURE",
    "DIG",
    "PLACE",
    "FEED",
    "COLLECT_FERTILIZER",
    "CARE",
)
_MARKET_OPERATIONS = (
    "BUY_SEED",
    "BUY_PRODUCT",
    "BUY_ANIMAL",
    "SELL",
    "HIRE",
    "BUY_LAND",
)


@dataclass(frozen=True)
class ReplayActorSequence:
    """One player's correctly aligned observations and actions."""

    player: int
    observations: tuple[dict[str, Any], ...]
    actions: tuple[dict[str, Any], ...]
    final_observation: dict[str, Any]
    reward: float
    teacher: str

    @property
    def action_hash(self) -> str:
        payload = json.dumps(
            self.actions,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class OfficialV7TrainingRecord:
    """One official sample ready to be adapted by the V7 BC entry point."""

    observation: dict[str, Any]
    action: dict[str, Any]
    soft_intents: tuple[SoftIntentDistribution, ...]
    strategy_mode: int
    value_target: float
    initial_mode_active: bool
    daily_switch_active: bool
    budget_target: DailyBudgetSoftTarget
    opponent_history: np.ndarray
    opponent_history_mask: np.ndarray
    opponent_cluster: int
    outcome: float
    horizon_targets: np.ndarray
    horizon_mask: np.ndarray
    teacher: str
    teacher_cluster: int
    episode_id: int
    player: int
    quality_tier: str


@dataclass(frozen=True)
class OfficialV7LoadResult:
    records: tuple[OfficialV7TrainingRecord, ...]
    trace_audit: dict[str, Any]
    manifests: tuple[str, ...]
    episodes: int
    actors: int
    quality_tiers: dict[str, int]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_plain(item) for item in value]
    return value


def replay_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _team_names(replay: Mapping[str, Any]) -> tuple[str, str]:
    info = replay.get("info", {}) or {}
    names = list(info.get("TeamNames", []) or [])
    if len(names) < 2:
        names = [
            str(row.get("Name", ""))
            for row in list(info.get("Agents", []) or [])[:2]
        ]
    names.extend(f"unknown-player-{index}" for index in range(len(names), 2))
    return str(names[0]), str(names[1])


def _episode_rewards(replay: Mapping[str, Any]) -> tuple[float, float]:
    rewards = list(replay.get("rewards", []) or [])
    steps = list(replay.get("steps", []) or [])
    if len(rewards) < 2 and steps:
        rewards = [row.get("reward") for row in steps[-1][:2]]
    rewards.extend(0.0 for _ in range(len(rewards), 2))
    return tuple(float(value or 0.0) for value in rewards[:2])  # type: ignore[return-value]


def _canonical_observation(
    replay_step: Sequence[Mapping[str, Any]], player: int, step: int
) -> dict[str, Any]:
    raw = _plain(replay_step[player].get("observation", {}) or {})
    shared = _plain(replay_step[0].get("observation", {}) or {})
    for field in _SHARED_FIELDS:
        if raw.get(field) is None and shared.get(field) is not None:
            raw[field] = copy.deepcopy(shared[field])
    raw["player"] = player
    raw["step"] = int(shared.get("step", step) or step)
    return raw


def extract_replay_actor_sequences(
    replay: Mapping[str, Any],
) -> tuple[ReplayActorSequence, ReplayActorSequence]:
    """Extract two V7 actor streams using Kaggle's one-state action shift."""

    steps = list(replay.get("steps", []) or [])
    if len(steps) < 2:
        raise ValueError("replay must contain at least two states")
    if any(len(step) < 2 for step in steps):
        raise ValueError("V7 official replay ingestion requires two-player states")
    rewards = _episode_rewards(replay)
    teams = _team_names(replay)
    observations: list[list[dict[str, Any]]] = [[], []]
    actions: list[list[dict[str, Any]]] = [[], []]
    for step_index in range(len(steps) - 1):
        current = steps[step_index]
        following = steps[step_index + 1]
        for player in range(2):
            status = str(current[player].get("status", "ACTIVE"))
            action = following[player].get("action")
            if status not in {"ACTIVE", "INACTIVE"}:
                raise ValueError(
                    f"player {player} is not active before terminal step {step_index}: {status}"
                )
            if not isinstance(action, Mapping):
                raise ValueError(
                    f"missing mapping action for player {player} at replay state {step_index + 1}"
                )
            observations[player].append(
                _canonical_observation(current, player, step_index)
            )
            actions[player].append(_plain(action))
    final = [
        _canonical_observation(steps[-1], player, len(steps) - 1)
        for player in range(2)
    ]
    return tuple(
        ReplayActorSequence(
            player=player,
            observations=tuple(observations[player]),
            actions=tuple(actions[player]),
            final_observation=final[player],
            reward=rewards[player],
            teacher=teams[player],
        )
        for player in range(2)
    )  # type: ignore[return-value]


def behavior_signature(sequence: ReplayActorSequence) -> np.ndarray:
    """Return a scale-stable strategy fingerprint for diversity clustering."""

    unit_index = {name: index for index, name in enumerate(_UNIT_OPERATIONS)}
    market_offset = len(unit_index)
    market_index = {
        name: market_offset + index for index, name in enumerate(_MARKET_OPERATIONS)
    }
    phase_offset = market_offset + len(market_index)
    values = np.zeros(phase_offset + 4, dtype=np.float32)
    for step, action in enumerate(sequence.actions):
        units = [action.get("farmer", ["PASS"]), *list(action.get("hands", []) or [])]
        for raw in units:
            operation = str(raw[0]) if raw else "PASS"
            if operation in unit_index:
                values[unit_index[operation]] += 1.0
        for raw in list(action.get("market", []) or []):
            operation = str(raw[0]) if raw else ""
            if operation in market_index:
                values[market_index[operation]] += 1.0
        phase = min(3, 4 * step // max(len(sequence.actions), 1))
        values[phase_offset + phase] += len(units) + len(action.get("market", []) or [])
    values[:market_offset] /= max(values[:market_offset].sum(), 1.0)
    values[market_offset:phase_offset] /= max(
        values[market_offset:phase_offset].sum(), 1.0
    )
    values[phase_offset:] /= max(values[phase_offset:].sum(), 1.0)
    return values


def _comparable_observation(observation: Any, *, player: int, step: int) -> Any:
    value = _plain(observation)
    value.pop("remainingOverageTime", None)
    value["player"] = player
    value["step"] = step
    return value


def _first_differences(
    actual: Any,
    expected: Any,
    *,
    path: str = "$",
    limit: int = 8,
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []

    def visit(left: Any, right: Any, current: str) -> None:
        if len(differences) >= limit:
            return
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            keys = sorted(set(left) | set(right))
            for key in keys:
                if key not in left or key not in right:
                    differences.append(
                        {
                            "path": f"{current}.{key}",
                            "actual": left.get(key, "<missing>"),
                            "expected": right.get(key, "<missing>"),
                        }
                    )
                else:
                    visit(left[key], right[key], f"{current}.{key}")
                if len(differences) >= limit:
                    break
            return
        if (
            isinstance(left, Sequence)
            and isinstance(right, Sequence)
            and not isinstance(left, (str, bytes, bytearray))
            and not isinstance(right, (str, bytes, bytearray))
        ):
            if len(left) != len(right):
                differences.append(
                    {
                        "path": f"{current}.length",
                        "actual": len(left),
                        "expected": len(right),
                    }
                )
                return
            for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
                visit(left_item, right_item, f"{current}[{index}]")
                if len(differences) >= limit:
                    break
            return
        if left != right:
            differences.append(
                {"path": current, "actual": left, "expected": right}
            )

    visit(actual, expected, path)
    return differences


def resimulation_audit(replay: Mapping[str, Any]) -> dict[str, Any]:
    """Replay every official action through the installed trusted interpreter."""

    from .fast_env import FastKaggricultureEnv

    info = replay.get("info", {}) or {}
    seed = info.get("seed")
    if seed is None:
        return {
            "available": False,
            "matched": False,
            "reason": "missing replay info.seed",
            "checked_transitions": 0,
        }
    sequences = extract_replay_actor_sequences(replay)
    configuration = dict(replay.get("configuration", {}) or {})
    environment = FastKaggricultureEnv(
        configuration=configuration,
        copy_observations=True,
    )
    actual = environment.reset(int(seed))
    for player in range(2):
        expected = sequences[player].observations[0]
        actual_value = _comparable_observation(actual[player], player=player, step=0)
        expected_value = _comparable_observation(expected, player=player, step=0)
        if actual_value != expected_value:
            return {
                "available": True,
                "matched": False,
                "reason": "initial observation mismatch",
                "first_mismatch_step": 0,
                "first_mismatch_player": player,
                "checked_transitions": 0,
                "differences": _first_differences(actual_value, expected_value),
            }
    for transition in range(len(sequences[0].actions)):
        result = environment.step(
            [sequences[0].actions[transition], sequences[1].actions[transition]]
        )
        expected_step = transition + 1
        expected = (
            sequences[0].final_observation,
            sequences[1].final_observation,
        ) if expected_step == len(sequences[0].observations) else (
            sequences[0].observations[expected_step],
            sequences[1].observations[expected_step],
        )
        for player in range(2):
            actual_value = _comparable_observation(
                result.observations[player], player=player, step=expected_step
            )
            expected_value = _comparable_observation(
                expected[player], player=player, step=expected_step
            )
            if actual_value != expected_value:
                return {
                    "available": True,
                    "matched": False,
                    "reason": "observation mismatch",
                    "first_mismatch_step": expected_step,
                    "first_mismatch_player": player,
                    "checked_transitions": transition + 1,
                    "differences": _first_differences(actual_value, expected_value),
                }
    return {
        "available": True,
        "matched": True,
        "reason": "exact trusted-interpreter match",
        "checked_transitions": len(sequences[0].actions),
    }


def build_v7_label_bundle(
    replay: Mapping[str, Any],
    *,
    episode_id: int,
    raw_path: str | Path,
    source_date: str,
    daily_score_rank: int,
    average_score: float,
    teacher_clusters: Sequence[int] = (0, 0),
    horizon: int = 24,
    top_m: int = 3,
    value_gamma: float = 0.997,
    value_reward_scale: float = 10.0,
    value_win_bonus: float = 1.0,
    validate_resimulation: bool = True,
) -> dict[str, Any]:
    """Create one compressed-ready official V7 label bundle."""

    sequences = extract_replay_actor_sequences(replay)
    step_count = len(sequences[0].observations)
    if len(sequences[1].observations) != step_count:
        raise ValueError("two actor streams have different lengths")
    potentials = np.stack(
        [
            paired_liquidatable_net_asset_potentials(
                [sequences[0].observations[step], sequences[1].observations[step]]
            )
            for step in range(step_count)
        ]
        + [
            paired_liquidatable_net_asset_potentials(
                [sequences[0].final_observation, sequences[1].final_observation]
            )
        ]
    )
    if sequences[0].reward == sequences[1].reward:
        outcomes = (0.0, 0.0)
    elif sequences[0].reward > sequences[1].reward:
        outcomes = (1.0, -1.0)
    else:
        outcomes = (-1.0, 1.0)
    replay_validation = (
        resimulation_audit(replay)
        if validate_resimulation
        else {
            "available": False,
            "matched": False,
            "reason": "resimulation disabled",
            "checked_transitions": 0,
        }
    )
    private_complete = all(
        bool(observation.get("private"))
        for sequence in sequences
        for observation in (*sequence.observations, sequence.final_observation)
    )
    terminal_statuses = [
        str(row.get("status", ""))
        for row in list(replay.get("steps", []) or [])[-1][:2]
    ]
    terminal_complete = len(terminal_statuses) == 2 and all(
        value == "DONE" for value in terminal_statuses
    )
    actors: list[dict[str, Any]] = []
    trace_audits: list[dict[str, Any]] = []
    for player, sequence in enumerate(sequences):
        traces = inverse_episode_traces(
            sequence.observations,
            sequence.actions,
            strategy_mode=-1,
            teacher=sequence.teacher,
            horizon=horizon,
            top_m=top_m,
        )
        audit = trace_quality_audit(traces)
        trace_audits.append(audit)
        returns = potential_shaped_value_targets(
            potentials[:, player],
            gamma=value_gamma,
            reward_scale=value_reward_scale,
            terminal_outcome=outcomes[player],
            win_bonus=value_win_bonus,
        )
        budget_targets = daily_budget_soft_targets(
            sequence.observations,
            sequence.actions,
            final_observation=sequence.final_observation,
        )
        actors.append(
            {
                "player": player,
                "teacher": sequence.teacher,
                "teacher_cluster": int(teacher_clusters[player]),
                "action_hash": sequence.action_hash,
                "strategy_mode": -1,
                "strategy_mode_supervised": False,
                "daily_switch_supervised": False,
                "reward": sequence.reward,
                "outcome": outcomes[player],
                "value_targets": returns.astype(np.float32).tolist(),
                "budget_targets": [target.as_dict() for target in budget_targets],
                "traces": [trace.as_dict() for trace in traces],
                "trace_audit": audit,
            }
        )
    trace_valid = all(
        audit["invalid_probability_rows"] == 0
        and audit["missing_chain_ids"] == 0
        and audit["invalid_stage_indices"] == 0
        and audit["discontinuous_routes"] == 0
        for audit in trace_audits
    )
    if replay_validation["matched"] and private_complete and terminal_complete and trace_valid:
        quality_tier = "gold"
    elif private_complete and terminal_complete and trace_valid:
        quality_tier = "silver"
    else:
        quality_tier = "bronze"
    path = Path(raw_path)
    return {
        "schema": OFFICIAL_V7_SCHEMA,
        "episode_id": int(episode_id),
        "source_date": str(source_date),
        "raw_file": path.name,
        "raw_sha256": replay_sha256(path),
        "daily_score_rank": int(daily_score_rank),
        "average_score": float(average_score),
        "step_count": step_count,
        "action_alignment": "observation[t] -> replay.steps[t+1][player].action",
        "high_level_label_policy": "latent_unknown_no_mode_or_switch_supervision",
        "value_target_policy": {
            "definition": "discounted liquidatable-net-asset transition returns",
            "gamma": value_gamma,
            "reward_scale": value_reward_scale,
            "win_bonus": value_win_bonus,
        },
        "quality": {
            "tier": quality_tier,
            "private_state_complete": private_complete,
            "terminal_complete": terminal_complete,
            "trace_valid": trace_valid,
            "resimulation": replay_validation,
        },
        "actors": actors,
    }


def write_v7_label_bundle(path: str | Path, bundle: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as stream:
        json.dump(bundle, stream, ensure_ascii=False, separators=(",", ":"))
        stream.write("\n")


def read_v7_label_bundle(path: str | Path) -> dict[str, Any]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    if bundle.get("schema") != OFFICIAL_V7_SCHEMA:
        raise ValueError(f"unsupported official V7 label schema: {bundle.get('schema')!r}")
    return bundle


def _intent_from_label(raw: Mapping[str, Any]) -> IntentLabel:
    target = list(raw.get("target", [-1, -1]) or [-1, -1])
    target.extend(-1 for _ in range(len(target), 2))
    return IntentLabel(
        task_type=TaskType[str(raw["task_type"])],
        target_x=int(target[0]),
        target_y=int(target[1]),
        item_id=int(raw.get("item_id", -1)),
    )


def soft_intent_distribution_from_dict(
    raw: Mapping[str, Any],
) -> SoftIntentDistribution:
    """Restore the dataclass representation consumed by V7 BC losses."""

    hypotheses: list[SoftIntentHypothesis] = []
    for hypothesis in list(raw.get("hypotheses", []) or []):
        chain_rows = list(hypothesis.get("chain", []) or [])
        chain = tuple(_intent_from_label(stage) for stage in chain_rows)
        hypotheses.append(
            SoftIntentHypothesis(
                intent=_intent_from_label(hypothesis),
                probability=float(hypothesis.get("probability", 0.0)),
                eta_steps=int(hypothesis.get("eta_steps", 0)),
                route=tuple(
                    (int(position[0]), int(position[1]))
                    for position in list(hypothesis.get("route", []) or [])
                ),
                evidence=str(hypothesis.get("evidence", "")),
                phase=IntentPhase(str(hypothesis.get("phase", "IDLE"))),
                chain_id=str(hypothesis.get("chain_id", "")),
                stage_index=int(hypothesis.get("stage_index", 0)),
                chain=chain,
                chain_operations=tuple(
                    str(stage.get("operation", "")) for stage in chain_rows
                ),
                chain_etas=tuple(
                    int(stage.get("eta_steps", -1)) for stage in chain_rows
                ),
            )
        )
    return SoftIntentDistribution(
        unit=int(raw["unit"]),
        hypotheses=tuple(hypotheses),
        confidence=float(raw.get("confidence", 0.0)),
        source=TraceSource(str(raw.get("source", "inverse_planning"))),
    )


def load_official_v7_training_records(
    manifest_paths: Sequence[str | Path],
    *,
    min_quality: str = "silver",
    max_episodes_per_manifest: int = 0,
    require_valid_audit: bool = True,
    cluster_namespace: int = 1_000_000,
) -> OfficialV7LoadResult:
    """Join audited compressed labels back to raw observations for V7 BC.

    The public high-level mode remains latent.  ``BALANCED`` is used only as a
    neutral conditioning input; both mode-loss activation flags are false, so it
    never becomes a fabricated behavior-cloning target.
    """

    quality_order = {"bronze": 0, "silver": 1, "gold": 2}
    if min_quality not in quality_order:
        raise ValueError(f"unknown minimum quality tier: {min_quality}")
    if max_episodes_per_manifest < 0:
        raise ValueError("max_episodes_per_manifest must be non-negative")
    records: list[OfficialV7TrainingRecord] = []
    audits: list[Mapping[str, Any]] = []
    quality_counts: dict[str, int] = {}
    episode_count = 0
    actor_count = 0
    resolved_manifests: list[str] = []
    for manifest_index, raw_manifest_path in enumerate(manifest_paths):
        manifest_path = Path(raw_manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != OFFICIAL_V7_MANIFEST_SCHEMA:
            raise ValueError(
                f"unsupported official V7 manifest schema in {manifest_path}: "
                f"{manifest.get('schema')!r}"
            )
        if require_valid_audit:
            audit_path = manifest_path.with_name("audit.json")
            if not audit_path.is_file():
                raise FileNotFoundError(
                    f"official V7 audit is required before BC: {audit_path}"
                )
            dataset_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if not bool(dataset_audit.get("valid", False)):
                raise ValueError(f"official V7 dataset audit failed: {audit_path}")
        resolved_manifests.append(str(manifest_path))
        dataset_dir = manifest_path.parent
        replay_dir = Path(manifest["source_replay_dir"])
        episode_rows = list(manifest.get("episodes", []) or [])
        if max_episodes_per_manifest:
            episode_rows = episode_rows[:max_episodes_per_manifest]
        for episode_row in episode_rows:
            label_path = dataset_dir / str(episode_row["label_file"])
            raw_path = replay_dir / str(episode_row["raw_file"])
            bundle = read_v7_label_bundle(label_path)
            tier = str(bundle["quality"]["tier"])
            if quality_order.get(tier, -1) < quality_order[min_quality]:
                continue
            with raw_path.open(encoding="utf-8") as stream:
                replay = json.load(stream)
            sequences = extract_replay_actor_sequences(replay)
            actor_rows = list(bundle.get("actors", []) or [])
            if len(actor_rows) != 2:
                raise ValueError(f"expected two actors in {label_path}")
            episode_count += 1
            quality_counts[tier] = quality_counts.get(tier, 0) + 1
            for actor_row in actor_rows:
                player = int(actor_row["player"])
                sequence = sequences[player]
                opponent_histories, opponent_history_masks = (
                    build_opponent_history_sequences(sequence.observations)
                )
                horizon_targets, horizon_masks = (
                    multi_horizon_resource_growth_targets(
                        sequence.observations,
                        final_observation=sequence.final_observation,
                    )
                )
                trace_rows = list(actor_row.get("traces", []) or [])
                value_targets = list(actor_row.get("value_targets", []) or [])
                raw_budget_targets = list(actor_row.get("budget_targets", []) or [])
                if raw_budget_targets:
                    budget_targets = [
                        DailyBudgetSoftTarget.from_dict(target)
                        for target in raw_budget_targets
                    ]
                else:
                    # Existing audited V7 bundles remain usable; labels are
                    # deterministically reconstructed from their raw replay.
                    budget_targets = daily_budget_soft_targets(
                        sequence.observations,
                        sequence.actions,
                        final_observation=sequence.final_observation,
                    )
                if not (
                    len(sequence.observations)
                    == len(sequence.actions)
                    == len(trace_rows)
                    == len(value_targets)
                    == len(budget_targets)
                ):
                    raise ValueError(
                        f"official V7 length mismatch: episode={bundle['episode_id']} "
                        f"player={player}"
                    )
                actor_count += 1
                audits.append(dict(actor_row.get("trace_audit", {})))
                raw_cluster = int(actor_row["teacher_cluster"])
                namespaced_cluster = (
                    cluster_namespace + manifest_index * 1_000 + raw_cluster
                )
                for sample_index, (
                    observation,
                    action,
                    trace_row,
                    value_target,
                    budget_target,
                ) in enumerate(
                    zip(
                        sequence.observations,
                        sequence.actions,
                        trace_rows,
                        value_targets,
                        budget_targets,
                        strict=True,
                    )
                ):
                    if dict(trace_row.get("action", {}) or {}) != action:
                        raise ValueError(
                            f"official V7 action mismatch: episode={bundle['episode_id']} "
                            f"player={player} sample={sample_index}"
                        )
                    records.append(
                        OfficialV7TrainingRecord(
                            observation=observation,
                            action=action,
                            soft_intents=tuple(
                                soft_intent_distribution_from_dict(distribution)
                                for distribution in list(trace_row.get("workers", []) or [])
                            ),
                            strategy_mode=MODE_INDEX["BALANCED"],
                            value_target=float(value_target),
                            initial_mode_active=False,
                            daily_switch_active=False,
                            budget_target=budget_target,
                            opponent_history=opponent_histories[sample_index],
                            opponent_history_mask=opponent_history_masks[sample_index],
                            opponent_cluster=int(
                                actor_rows[1 - player]["teacher_cluster"]
                            ),
                            outcome=float(actor_row.get("outcome", 0.0)),
                            horizon_targets=horizon_targets[sample_index],
                            horizon_mask=horizon_masks[sample_index],
                            teacher=str(actor_row["teacher"]),
                            teacher_cluster=namespaced_cluster,
                            episode_id=int(bundle["episode_id"]),
                            player=player,
                            quality_tier=tier,
                        )
                    )
    if not records:
        raise ValueError("no official V7 BC records passed the requested filters")
    return OfficialV7LoadResult(
        records=tuple(records),
        trace_audit=merge_trace_quality_audits(audits),
        manifests=tuple(resolved_manifests),
        episodes=episode_count,
        actors=actor_count,
        quality_tiers=dict(sorted(quality_counts.items())),
    )
